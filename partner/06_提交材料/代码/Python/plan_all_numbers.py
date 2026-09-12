# -*- coding: utf-8 -*-
"""★ 口径 C 正式重算 + 论文全部数字
   规则: 0:00 一次性决策(购电量+充放电计划), 计划照原样执行(缺口 5 倍紧急购电, 富余弃光);
        问题三只在 12:00 引入一次预报调整(带该时刻预报口径的报童裕度), 付 0.5/1.5 倍偏差费。
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import merge_spans, span_label, solve_day, ETA_C, ETA_D
from causal_core import D, price, PR, ROOT
from io_results import load_all, write_result2, write_result3, dates_of_results
import fc_upgraded as FU
import plan_core as PC
import engine

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
DATA = os.path.join(ROOT, '04_建模求解', 'data')
t0 = time.time()
out = {}

print('=== 1) 预测模型(升级版) ===')
L_UP, INFO = FU.build_load()
PV_UP = FU.build_pv()
P_UP = FU.build_price()
for nm, A, B, sc in [('负荷', L_UP, L_ACT, 6), ('光伏', PV_UP, PV_ACT, 6), ('电价', P_UP, D.PR, 1)]:
    m = PC.metrics(A, B, sc)
    base = PC.metrics(D.Lb if nm == '负荷' else (D.Pb if nm == '光伏' else P_UP), B, sc)
    out[f'mae_{nm}'] = dict(new=m, baseline=base)
    print(f'  {nm}: 新 {m["mae"]:.3f} / 基线 {base["mae"]:.3f} (RMSE {m["rmse"]:.2f}, MAPE {m["mape"]:.2f}%)')
e = np.abs(L_UP[m2:] - L_ACT[m2:]).mean(axis=1) * 6
eb = np.abs(D.Lb[m2:] - L_ACT[m2:]).mean(axis=1) * 6
lo = PC.LOW[m2:]
out['load_class'] = dict(new=dict(total=float(e.mean()), low=float(e[lo].mean()), high=float(e[~lo].mean())),
                         base=dict(total=float(eb.mean()), low=float(eb[lo].mean()), high=float(eb[~lo].mean())))
print(f'  负荷分类: 新 {out["load_class"]["new"]}')
print(f'            基线 {out["load_class"]["base"]}')
out['load_weights'] = INFO

print('=== 2) 1 月标定 q(计划照原样执行口径) ===')
jan = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    R = PC.run_q2(L_UP, PV_UP, q=q, upto=31)
    jan[str(q)] = float((R['pc'] + R['ec'])[7:31].sum())
Q = float(min(jan, key=lambda k: jan[k]))
out['q_tune'] = jan; out['q_star'] = Q
print(f'  { {k: round(v) for k, v in jan.items()} } -> q*={Q}')

print('=== 3) 四问主结果(问题三/四之三: 仅 12:00 调整) ===')
R2 = PC.run_q2(L_UP, PV_UP, q=Q)
R3 = PC.run_q3(L_UP, PV_UP, q=Q, adjust=(12,))
R42 = PC.run_q2(L_UP, PV_UP, q=Q, pmat=PR)
R43 = PC.run_q3(L_UP, PV_UP, q=Q, pmat=PR, adjust=(12,))
for k, R in [('q2', R2), ('q3', R3), ('q42', R42), ('q43', R43)]:
    out[k] = {kk: float(R[kk]) for kk in ('plan', 'dev', 'em', 'total', 'em_kwh') if kk in R}
    print(f'  {k}: {out[k]}')

print('=== 4) 对照方法(问题二) ===')
out['cmp_典型日'] = float(PC.run_q2(np.broadcast_to(D.L_typ, (n, H)),
                                 np.broadcast_to(D.PV_typ, (n, H)), q=Q)['total'])
out['cmp_无储能'] = float(PC.nostorage(L_UP, PV_UP, Q))
out['cmp_天眼下界'] = float(PC.clairvoyant())
print(f'  典型日 {out["cmp_典型日"]:,.0f} | 无储能 {out["cmp_无储能"]:,.0f} | 天眼下界 {out["cmp_天眼下界"]:,.0f}')
# 昨日外推: 自身基值 + 自身残差裕度(1 月标定 q)
L_PER = D.L_act[np.r_[0, np.arange(n - 1)]]     # 昨日实测
PV_PER = D.PV_act[np.r_[0, np.arange(n - 1)]]
for tag, qq in [('M7_q0.6', 0.6), ('M5_q0.8', 0.8), ('M1_q0', 0.0)]:
    out[f'cmp_昨日外推_{tag}'] = float(PC.run_q2(L_PER, PV_PER, q=qq)['total'])
    print(f'  昨日外推 {tag}: {out[f"cmp_昨日外推_{tag}"]:,.0f}')

print('=== 5) 灵敏度 ===')
# (a) q 全年
out['sens_q_year'] = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    out['sens_q_year'][str(q)] = float(PC.run_q2(L_UP, PV_UP, q=q)['total'])
print(f'  q 全年: { {k: round(v) for k, v in out["sens_q_year"].items()} }')
# (b) 效率口径(往返 0.9)  —— 计划与执行同步替换
old_eta = (engine.ETA_C, engine.ETA_D, PC.ETA_C, PC.ETA_D)
engine.ETA_C = engine.ETA_D = PC.ETA_C = PC.ETA_D = 0.9 ** 0.5
out['sens_eta_roundtrip'] = float(PC.run_q2(L_UP, PV_UP, q=Q)['total'])
engine.ETA_C, engine.ETA_D, PC.ETA_C, PC.ETA_D = old_eta
print(f'  往返0.9: {out["sens_eta_roundtrip"]:,.0f}')
# (c) 2 月: v=0 / SOC 等式
def feb(mode):
    B = PC.buffer_mat(L_UP, PV_UP, Q)
    tot = 0.0; soc = 6000.0
    for i in range(59):
        v = 0.0 if mode == 'zero' else (0.9 * price.mean() if i < n - 1 else 0.0)
        end = soc if mode == 'eq' else None
        r = solve_day(price, L_UP[i] + B[i], PV_UP[i], soc, soc_end=end, v=v)
        em, curt, sEnd = PC.execute_as_planned(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        if mode == 'eq':
            sEnd = soc
        if i >= m2:
            tot += float(price @ r['P']) + float(5 * price @ em)
        soc = sEnd
    return tot
out['sens_feb_base'] = feb('std'); out['sens_feb_v0'] = feb('zero'); out['sens_feb_socEq'] = feb('eq')
print(f'  2月: 基准 {out["sens_feb_base"]:,.0f} / v=0 {out["sens_feb_v0"]:,.0f} / SOC等式 {out["sens_feb_socEq"]:,.0f}')
# (d) 电价信息(计划用预测电价, 结算按实际)
out['sens_price_pred'] = float(PC.run_q2(L_UP, PV_UP, q=Q, pmat=P_UP)['total'])
print(f'  计划用预测电价: {out["sens_price_pred"]:,.0f}')
# (e) 偏差计价口径 A(计划全额计 + 上调1.5倍, 下调不退)
S = np.maximum(R3['P0'] - R3['Pf'], 0.0)
out['sens_rule_A'] = float(R3['plan'] + R3['em']
                           + sum(1.5 * price @ np.maximum(R3['Pf'][i] - R3['P0'][i], 0) for i in range(m2, n)))
out['sens_rule_A_dev'] = out['sens_rule_A'] - R3['plan'] - R3['em']
print(f'  计价口径 A: {out["sens_rule_A"]:,.0f} (B={R3["total"]:,.0f})')
# (f) 缓冲形式
def run_with_buf(mode, q=Q, W=30):
    e_ = (L_ACT - L_UP) - (PV_ACT - PV_UP)
    B = np.zeros((n, H))
    for i in range(n):
        lo_ = max(0, i - W)
        if i - lo_ < 7:
            continue
        if mode == 'net':
            B[i] = np.quantile(e_[lo_:i], q, axis=0)
        elif mode == 'split':
            B[i] = (np.quantile((L_ACT - L_UP)[lo_:i], q, axis=0)
                    + np.quantile(-(PV_ACT - PV_UP)[lo_:i], q, axis=0))
        elif mode == 'normal':
            from scipy import stats
            z = stats.norm.ppf(q); h = np.arange(H) // 6
            for k in range(24):
                B[i][h == k] = z * e_[lo_:i][:, h == k].ravel().std()
    return float(PC.run_q2(L_UP, PV_UP, q=q, buffer_mat_in=B)['total'])
out['buf_net'] = float(R2['total'])
for tag, kw in [('norm', dict(mode='normal')), ('split', dict(mode='split')),
                ('w60', dict(mode='net', W=60)), ('w90', dict(mode='net', W=90))]:
    out['buf_' + tag] = run_with_buf(**kw)
    print(f'  缓冲 {tag}: {out["buf_"+tag]:,.0f}')
# (g) 预言机分解
out['oracle_pv'] = float(PC.run_q2(L_UP, PV_ACT, q=Q)['total'])
out['oracle_load'] = float(PC.run_q2(L_ACT, PV_UP, q=Q)['total'])
print(f'  预言机: O_PV {out["oracle_pv"]:,.0f} | O_L {out["oracle_load"]:,.0f}')
# (h) 问题三各调整方案
out['q3_options'] = {}
for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((18,), '+18:00'),
               ((6, 12), '+6,+12'), ((12, 18), '+12,+18'), ((6, 12, 18), '全滚动')]:
    R = PC.run_q3(L_UP, PV_UP, q=Q, adjust=at)
    out['q3_options'][nm] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']),
                                 total=float(R['total']))
print('  问题三方案:', {k: round(v['total']) for k, v in out['q3_options'].items()})
out['q43_options'] = {}
for at, nm in [((), '仅0:00'), ((12,), '+12:00'), ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]:
    R = PC.run_q3(L_UP, PV_UP, q=Q, pmat=PR, adjust=at)
    out['q43_options'][nm] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']),
                                  total=float(R['total']))
print('  问题四之三方案:', {k: round(v['total']) for k, v in out['q43_options'].items()})
# (i) 问题四 无储能/下界 + 周内净充电
out['q4_nostorage'] = float(PC.nostorage(L_UP, PV_UP, Q, pmat=PR))
out['q4_bound'] = float(PC.clairvoyant(pmat=PR))
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
net = (R42['C'] - R42['D'])[m2:]
dw = D.dow[m2:]
out['weekday'] = [float(net[dw == k].sum()) / max(1, int((dw == k).sum())) for k in range(7)]
print(f'  Q4 无储能 {out["q4_nostorage"]:,.0f} | 下界 {out["q4_bound"]:,.0f}')
print('  Q4-2 周内净充电:', ' '.join(f'{names[k]}{out["weekday"][k]:+.0f}' for k in range(7)))
# (j) 设备/损耗统计
out['energy'] = dict(load=float(L_ACT[m2:].sum()), pv=float(PV_ACT[m2:].sum()),
                     buy=float(R2['P'][m2:].sum()), em=float(R2['Em'][m2:].sum()),
                     chg=float(R2['C'][m2:].sum()), dis=float(R2['D'][m2:].sum()),
                     curt=float(R2['Curt'][m2:].sum()))

print('=== 6) 写出 5 个结果文件(问题三/四之三 = 仅 12:00 调整) ===')
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
np.savez_compressed(os.path.join(DATA, 'plan_preds.npz'), L=L_UP, PV=PV_UP, P=P_UP)
json.dump(out, open(os.path.join(DATA, 'plan_all_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1, default=float)
print(f'\n[完成] {time.time()-t0:.0f}s -> data/plan_all_numbers.json')
