# -*- coding: utf-8 -*-
"""★ 正式重算(口径 A: 0:00 一次性决策"购电计划 + 电池响应规则"; 日内只执行规则)
   预测模型: 升级版(负荷 水平×形状+分类凸组合; 光伏/电价 因果岭回归堆叠)
   问题二: 0:00 计划 + 实时平衡规则(先调计划充放、再动用储能, 兜不住才 5 倍紧急购电)
   问题三/四之三: 0:00 计划 + 12:00 用最新预报对剩余时段增量调整(带该预报口径的报童裕度) + 同一平衡规则
   输出: 5 个结果文件 + data/A_numbers.json
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import merge_spans, span_label, solve_day, ETA_C, ETA_D, SOC_MIN, SOC_MAX, P_MAX
from causal_core import D, price, PR, ROOT, balance
from io_results import load_all, write_result2, write_result3, dates_of_results
import fc_upgraded as FU

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
FCADJ = D.fca_m
DATA = os.path.join(ROOT, '04_建模求解', 'data')
t0 = time.time()
out = {}

print('=== 1) 升级版预测模型 ===')
L, INFO = FU.build_load()
PV = FU.build_pv()
P = FU.build_price()


def buffer_of(Lh, PVh, q, i, W=30):
    lo = max(0, i - W)
    if i - lo < 7 or q <= 0:
        return np.zeros(H)
    e = (L_ACT[lo:i] - Lh[lo:i]) - (PV_ACT[lo:i] - PVh[lo:i])
    return np.quantile(e, q, axis=0)


def adj_buffer(Lh, PVh, hh, h0, q, i, W=30):
    lo = max(0, i - W)
    if i - lo < 7 or q <= 0:
        return np.zeros(H - h0)
    e = (L_ACT[lo:i, h0:] - Lh[lo:i, h0:]) - (PV_ACT[lo:i, h0:] - FCADJ[hh][lo:i, h0:])
    return np.quantile(e, q, axis=0)


def run_q2(Lh, PVh, q, pmat=None, upto=None):
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    soc = 6000.0
    P_m = np.zeros((n, H)); C_m = np.zeros((n, H)); D_m = np.zeros((n, H)); E_m = np.zeros((n, H))
    s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        r = solve_day(p, Lh[i] + buffer_of(Lh, PVh, q, i), PVh[i], soc, soc_end=None, v=v)
        c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        P_m[i], C_m[i], D_m[i], E_m[i] = r['P'], c, d, e
        pc[i] = float(p @ r['P']); ec[i] = float(5 * p @ e)
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P=P_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(E_m[m2:upto].sum()))


def run_q3(Lh, PVh, q, pmat=None, adjust=(12,), upto=None, use_adjbuf=True):
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    soc = 6000.0; HH = [36, 72, 108]; fh = [6, 12, 18]
    P0 = np.zeros((n, H)); Pf = np.zeros((n, H)); C = np.zeros((n, H)); Dm = np.zeros((n, H))
    Em = np.zeros((n, H)); s0 = np.zeros(n); s24 = np.zeros(n)
    pc = np.zeros(n); dv = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        Lbar = Lh[i] + buffer_of(Lh, PVh, q, i)
        r0 = solve_day(p, Lbar, PVh[i], soc, soc_end=None, v=v)
        Pv, Cc, Dd = r0['P'].copy(), r0['C'].copy(), r0['D'].copy()
        SOC = r0['E'].copy()
        Pinit = Pv.copy()
        for h0, hh in zip(HH, fh):
            if hh not in adjust:
                continue
            Ladj = Lh[i, h0:] + (adj_buffer(Lh, PVh, hh, h0, q, i) if use_adjbuf else 0.0)
            r = solve_day(p[h0:], Ladj, FCADJ[hh][i, h0:], SOC[h0], soc_end=None, v=v, base_P=Pv[h0:])
            Pv[h0:], Cc[h0:], Dd[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0 + 1], r['E'][1:]])      # 继承调整后的储能轨迹
        c, d, sEnd, e = balance(Pv, Cc, Dd, soc, L_ACT[i], PV_ACT[i])
        P0[i], Pf[i], C[i], Dm[i], Em[i] = Pinit, Pv, c, d, e
        pc[i] = float(p @ Pinit)
        dv[i] = float(np.sum(-0.5 * p * np.maximum(Pinit - Pv, 0) + 1.5 * p * np.maximum(Pv - Pinit, 0)))
        ec[i] = float(5 * p @ e)
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P0=P0, Pf=Pf, C=C, D=Dm, Em=Em, s0=s0, s24=s24, pc=pc, dv=dv, ec=ec,
                plan=float(pc[m2:upto].sum()), dev=float(dv[m2:upto].sum()),
                em=float(ec[m2:upto].sum()), total=float((pc + dv + ec)[m2:upto].sum()),
                em_kwh=float(Em[m2:upto].sum()))


print('=== 2) 1 月标定 q ===')
jan = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    R = run_q2(L, PV, q=q, upto=31)
    jan[str(q)] = float((R['pc'] + R['ec'])[7:31].sum())
Q = float(min(jan, key=lambda k: jan[k]))
out['q_tune'] = jan; out['q_star'] = Q
print(f'  { {k: round(v) for k, v in jan.items()} } -> q* = {Q}')

print('=== 3) 问题三/四之三 调整方案(含"调整时刻是否重设裕度"对照) ===')
for tag, ub in [('仅0:00', None), ('+12:00(带裕度)', True), ('+12:00(不带裕度)', False),
                ('全滚动(带裕度)', True)]:
    if tag == '仅0:00':
        R = run_q3(L, PV, q=Q, adjust=())
    elif tag == '全滚动(带裕度)':
        R = run_q3(L, PV, q=Q, adjust=(6, 12, 18), use_adjbuf=True)
    else:
        R = run_q3(L, PV, q=Q, adjust=(12,), use_adjbuf=ub)
    out['q3_' + tag] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']),
                            total=float(R['total']))
    print(f'  {tag:16s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')

print('=== 4) 四问主结果 ===')
R2 = run_q2(L, PV, q=Q)
print(f'问题二: 计划 {R2["plan"]:,.0f} + 紧急 {R2["em"]:,.0f} = {R2["total"]:,.0f} 元')
USE_ADJBUF = out['q3_+12:00(带裕度)']['total'] <= out['q3_+12:00(不带裕度)']['total']
print(f'  [调整时刻重设裕度] {"采用" if USE_ADJBUF else "不采用"}')
R3 = run_q3(L, PV, q=Q, adjust=(12,), use_adjbuf=USE_ADJBUF)
print(f'问题三(12:00): 计划 {R3["plan"]:,.0f} + 调整 {R3["dev"]:,.0f} + 紧急 {R3["em"]:,.0f} = {R3["total"]:,.0f} 元')
R42 = run_q2(L, PV, q=Q, pmat=PR)
print(f'问题四之二: 计划 {R42["plan"]:,.0f} + 紧急 {R42["em"]:,.0f} = {R42["total"]:,.0f} 元')
R42p = run_q2(L, PV, q=Q, pmat=P)
print(f'问题四之二(预测电价): {R42p["total"]:,.0f} 元')
o43 = {}
for at, nm in [((), '仅0:00'), ((12,), '+12:00'), ((6, 12, 18), '全滚动')]:
    o43[nm] = run_q3(L, PV, q=Q, pmat=PR, adjust=at, use_adjbuf=USE_ADJBUF)
    print(f'  Q4-3 {nm:8s}: {o43[nm]["total"]:,.0f} 元')
BEST43 = min(o43, key=lambda k: o43[k]['total'])
R43 = o43[BEST43]
print(f'问题四之三采用: {BEST43} = {R43["total"]:,.0f} 元')
out['q2'] = dict(plan=float(R2['plan']), em=float(R2['em']), total=float(R2['total']), em_kwh=float(R2['em_kwh']))
out['q3'] = dict(plan=float(R3['plan']), dev=float(R3['dev']), em=float(R3['em']), total=float(R3['total']),
                 em_kwh=float(R3['em_kwh']), adjust='12:00', adjbuf=bool(USE_ADJBUF))
out['q42'] = dict(plan=float(R42['plan']), em=float(R42['em']), total=float(R42['total']),
                  em_kwh=float(R42['em_kwh']), pred_price=float(R42p['total']))
out['q43'] = dict(plan=float(R43['plan']), dev=float(R43['dev']), em=float(R43['em']), total=float(R43['total']),
                  em_kwh=float(R43['em_kwh']), adjust=BEST43)

print('=== 5) 写出 5 个结果文件 ===')
day, typ, fc0, fca, pers, tmpl, const = load_all()
res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in D.dates])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
seg = lambda E, i: [(span_label(a, b), float(E[i][a - 1:b].sum()))
                    for a, b in merge_spans(np.where(E[i] > 1e-6)[0] + 1)]
write_result2('result2.xlsx', res_dates, R2['P'][i0:i1], R2['pc'][i0:i1], R2['s0'][i0:i1], R2['s24'][i0:i1],
              R2['C'][i0:i1], R2['D'][i0:i1], [seg(R2['Em'], i0 + k) for k in range(len(res_dates))])
write_result3('result3.xlsx', res_dates, R3['P0'][i0:i1], R3['pc'][i0:i1], R3['Pf'][i0:i1],
              (R3['pc'] + R3['dv'])[i0:i1], R3['s0'][i0:i1], R3['s24'][i0:i1], R3['C'][i0:i1], R3['D'][i0:i1],
              [seg(R3['Em'], i0 + k) for k in range(len(res_dates))])
write_result2('result4-2.xlsx', res_dates, R42['P'][i0:i1], R42['pc'][i0:i1], R42['s0'][i0:i1], R42['s24'][i0:i1],
              R42['C'][i0:i1], R42['D'][i0:i1], [seg(R42['Em'], i0 + k) for k in range(len(res_dates))],
              template='result4-2.xlsx')
write_result3('result4-3.xlsx', res_dates, R43['P0'][i0:i1], R43['pc'][i0:i1], R43['Pf'][i0:i1],
              (R43['pc'] + R43['dv'])[i0:i1], R43['s0'][i0:i1], R43['s24'][i0:i1], R43['C'][i0:i1], R43['D'][i0:i1],
              [seg(R43['Em'], i0 + k) for k in range(len(res_dates))], template='result4-3.xlsx')
np.savez_compressed(os.path.join(DATA, 'A_preds.npz'), L=L, PV=PV, P=P)
json.dump(out, open(os.path.join(DATA, 'A_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1, default=float)
print(f'[完成] {time.time()-t0:.0f}s')
