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
import upgraded_core as UC
import price_fc as PF

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
P_FU = FU.build_price()            # 乙(对照): 廓线+自适应水平+岭回归堆叠
P, _, PD_PRICE = PF.build_price()  # 甲(采用): 周周期加权结构模型 + LightGBM 残差修正
P_ROLL = PF.build_price_rolling(P, damp=0.7)   # 6/12/18 用当日已实现价格滚动修正后的价格预测
print(f'  电价预测: 甲(采用) MAE={PD_PRICE["mae_final"]:.4f} 元/kWh | 乙(对照) MAE={np.abs(P_FU[m2:] - PR[m2:]).mean():.4f}')


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


def run_q2(Lh, PVh, q, pmat=None, upto=None, bill=None):
    bl = (np.broadcast_to(price, (n, H)) if pmat is None else pmat) if bill is None else bill
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
        pc[i] = float(bl[i] @ r['P']); ec[i] = float(5 * bl[i] @ e)   # 结算按实际电价
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P=P_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(E_m[m2:upto].sum()))


def run_q3(Lh, PVh, q, pmat=None, adjust=(12,), upto=None, use_adjbuf=True,
           bill=None, pmat_adj=None):
    """问题三: 委托 upgraded_core.run_q3(逐段滚动 + 每段用实际执行后的 SOC + 偏差恒以 0:00 原始计划为基准
    + 因果去偏预报; use_adjbuf=True 时在调整时刻按"该时刻预报"重设报童裕度)"""
    return UC.run_q3(Lh, PVh, q=q, pmat=pmat, adjust=adjust, upto=upto,
                     debias=True, use_adjbuf=use_adjbuf, bill=bill, pmat_adj=pmat_adj)


print('=== 2) 1 月标定问题二 q* ===')
jan = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    R = run_q2(L, PV, q=q, upto=31)
    jan[str(q)] = float((R['pc'] + R['ec'])[7:31].sum())
Q = float(min(jan, key=lambda k: jan[k]))
out['q_tune'] = jan; out['q_star'] = Q
print(f'  { {k: round(v) for k, v in jan.items()} } -> q* = {Q}')

print('=== 2b) 1 月独立标定问题三 q* (全时刻滚动, 逐段实测结算) ===')
q3jan = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    R = run_q3(L, PV, q=q, adjust=(6, 12, 18), upto=31)
    q3jan[str(q)] = float((R['pc'] + R['dv'] + R['ec'])[7:31].sum())
Q3 = float(min(q3jan, key=lambda k: q3jan[k]))
out['q3_tune'] = q3jan; out['q3_star'] = Q3
print(f'  { {k: round(v) for k, v in q3jan.items()} } -> q3* = {Q3}')

print('=== 3) 问题三 调整方案对照(逐段滚动口径, q=q3*) ===')
for tag, adj in [('仅0:00', ()), ('+6:00', (6,)), ('+12:00', (12,)), ('+18:00', (18,)),
                 ('+6:00、+12:00', (6, 12)), ('+12:00、+18:00', (12, 18)),
                 ('全滚动(带裕度)', (6, 12, 18))]:
    R = run_q3(L, PV, q=Q3, adjust=adj, use_adjbuf=True)
    out['q3_' + tag] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']),
                            total=float(R['total']))
    print(f'  {tag:18s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
R3nb = run_q3(L, PV, q=Q3, adjust=(6, 12, 18), use_adjbuf=False)
out['q3_全滚动(不带裕度)'] = dict(plan=float(R3nb['plan']), dev=float(R3nb['dev']),
                                  em=float(R3nb['em']), total=float(R3nb['total']))
out['q3_+12:00(带裕度)'] = dict(out['q3_+12:00'])
_r12nb = run_q3(L, PV, q=Q3, adjust=(12,), use_adjbuf=False)
out['q3_+12:00(不带裕度)'] = dict(plan=float(_r12nb['plan']), dev=float(_r12nb['dev']),
                                  em=float(_r12nb['em']), total=float(_r12nb['total']))

print('=== 4) 四问主结果 ===')
R2 = run_q2(L, PV, q=Q)
print(f'问题二: 计划 {R2["plan"]:,.0f} + 紧急 {R2["em"]:,.0f} = {R2["total"]:,.0f} 元')
USE_ADJBUF = out['q3_全滚动(带裕度)']['total'] <= out['q3_全滚动(不带裕度)']['total']
print(f'  [调整时刻重设裕度] {"采用" if USE_ADJBUF else "不采用"}')
ADJ = (6, 12, 18)                       # ★ 正式采用: 全时刻滚动(每次调整均以 0:00 原始计划为偏差基准)
R3 = run_q3(L, PV, q=Q3, adjust=ADJ, use_adjbuf=USE_ADJBUF)
print(f'问题三(全滚动): 计划 {R3["plan"]:,.0f} + 调整 {R3["dev"]:,.0f} + 紧急 {R3["em"]:,.0f} = {R3["total"]:,.0f} 元')
R42 = run_q2(L, PV, q=Q, pmat=P, bill=PR)          # 采用: 预测电价定计划、实际电价结算
print(f'问题四之二: 计划 {R42["plan"]:,.0f} + 紧急 {R42["em"]:,.0f} = {R42["total"]:,.0f} 元')
R42y = run_q2(L, PV, q=Q, pmat=P_FU, bill=PR)      # 对照: 乙 电价模型
R42u = run_q2(L, PV, q=Q, pmat=PR, bill=PR)        # 上界: 0:00 已知实际电价(不可达)
print(f'问题四之二(乙对照): {R42y["total"]:,.0f} 元 | 上界(已知实际电价, 不可达): {R42u["total"]:,.0f} 元')
o43 = {}
for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((18,), '+18:00'),
               ((6, 12, 18), '全滚动')]:
    o43[nm] = run_q3(L, PV, q=Q3, pmat=P, bill=PR, pmat_adj=P_ROLL, adjust=at, use_adjbuf=USE_ADJBUF)
    print(f'  Q4-3 {nm:8s}: {o43[nm]["total"]:,.0f} 元')
BEST43 = '全滚动' if o43['全滚动']['total'] <= min(v['total'] for k, v in o43.items() if k != '全滚动') else min(o43, key=lambda k: o43[k]['total'])
R43 = o43[BEST43]
print(f'问题四之三采用: {BEST43} = {R43["total"]:,.0f} 元')
out['q2'] = dict(plan=float(R2['plan']), em=float(R2['em']), total=float(R2['total']), em_kwh=float(R2['em_kwh']))
out['q3'] = dict(plan=float(R3['plan']), dev=float(R3['dev']), em=float(R3['em']), total=float(R3['total']),
                 em_kwh=float(R3['em_kwh']), adjust='全滚动(6/12/18)', adjbuf=bool(USE_ADJBUF), q=Q3)
out['q42'] = dict(plan=float(R42['plan']), em=float(R42['em']), total=float(R42['total']),
                  em_kwh=float(R42['em_kwh']), price_yi=float(R42y['total']),
                  price_known_bound=float(R42u['total']),
                  mae_price=float(PD_PRICE['mae_final']), mae_price_yi=float(np.abs(P_FU[m2:] - PR[m2:]).mean()))
out['q43'] = dict(plan=float(R43['plan']), dev=float(R43['dev']), em=float(R43['em']), total=float(R43['total']),
                  em_kwh=float(R43['em_kwh']), adjust=BEST43, q=Q3)
out['q43_options'] = {k: dict(plan=float(v['plan']), dev=float(v['dev']), em=float(v['em']),
                              total=float(v['total'])) for k, v in o43.items()}
R43u = run_q3(L, PV, q=Q3, pmat=PR, bill=PR, adjust=(6, 12, 18), use_adjbuf=USE_ADJBUF)
out['q43_price_known_bound'] = float(R43u['total'])


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
