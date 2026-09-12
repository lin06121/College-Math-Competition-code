# -*- coding: utf-8 -*-
"""★ 正式重算(口径 C: 0:00 一次性决策 + 计划照原样执行)
   预测模型: 升级版(负荷 水平×形状+分类凸组合; 光伏/电价 因果岭回归堆叠)
   步骤: 1) 构造预测 2) 1 月标定报童分位 q 3) 四问全年重算 4) 写出 5 个结果文件
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import merge_spans, span_label
from causal_core import D, price, PR, ROOT
from io_results import load_all, write_result2, write_result3, dates_of_results
import fc_upgraded as FU
import plan_core as PC

n, H = D.n, 144
m2 = D.m2
DATA = os.path.join(ROOT, '04_建模求解', 'data')
t0 = time.time()

print('=== 1) 构造升级版预测(负荷/光伏/电价) ===')
L_UP, INFO = FU.build_load()
PV_UP = FU.build_pv()
P_UP = FU.build_price()

print('=== 2) 1 月标定报童分位 q(计划照原样执行口径) ===')
jan = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    R = PC.run_q2(L_UP, PV_UP, q=q, upto=31)
    jan[q] = float((R['pc'] + R['ec'])[7:31].sum())
    print(f'    q={q}: 1月费用 {jan[q]:,.0f} 元')
Q_STAR = min(jan, key=jan.get)
print(f'    [采用] q* = {Q_STAR}')

print('=== 3) 四问全年重算(2.1-12.31) ===')
R2 = PC.run_q2(L_UP, PV_UP, q=Q_STAR)
print(f'问题二: 计划 {R2["plan"]:,.0f} + 紧急 {R2["em"]:,.0f} = {R2["total"]:,.0f} 元')
opts = {}
for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((18,), '+18:00'),
               ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]:
    R = PC.run_q3(L_UP, PV_UP, q=Q_STAR, adjust=at)
    opts[nm] = R
    print(f'问题三 {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
BEST3 = min(opts, key=lambda k: opts[k]['total'])
print(f'    -> 问题三最优方案: {BEST3} ({opts[BEST3]["total"]:,.0f} 元)')
R42 = PC.run_q2(L_UP, PV_UP, q=Q_STAR, pmat=PR)
print(f'问题四之二: 计划 {R42["plan"]:,.0f} + 紧急 {R42["em"]:,.0f} = {R42["total"]:,.0f} 元')
o43 = {}
for at, nm in [((), '仅0:00'), ((12,), '+12:00'), ((6, 12, 18), '全滚动')]:
    R = PC.run_q3(L_UP, PV_UP, q=Q_STAR, pmat=PR, adjust=at)
    o43[nm] = R
    print(f'问题四之三 {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
BEST43 = min(o43, key=lambda k: o43[k]['total'])
print(f'    -> 问题四之三最优方案: {BEST43} ({o43[BEST43]["total"]:,.0f} 元)')
BEST3_AT = {'仅0:00': (), '+6:00': (6,), '+12:00': (12,), '+18:00': (18,),
            '+6,+12': (6, 12), '全滚动': (6, 12, 18)}[BEST3]
BEST43_AT = {'仅0:00': (), '+12:00': (12,), '全滚动': (6, 12, 18)}[BEST43]
R3 = opts[BEST3]; R43 = o43[BEST43]

print('=== 4) 写出 5 个结果文件 ===')
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
num = dict(adopt_adjust_q3=BEST3, adopt_adjust_q43=BEST43, q_star=Q_STAR, jan_grid=jan,
           q2=dict(plan=R2['plan'], em=R2['em'], total=R2['total'], em_kwh=R2['em_kwh'],
                   curt_kwh=R2['curt_kwh']),
           q3=dict(plan=R3['plan'], dev=R3['dev'], em=R3['em'], total=R3['total'],
                   em_kwh=R3['em_kwh'], adjust=BEST3, zero=opts['仅0:00']['total'],
                   full=opts['全滚动']['total']),
           q42=dict(plan=R42['plan'], em=R42['em'], total=R42['total'], em_kwh=R42['em_kwh']),
           q43=dict(plan=R43['plan'], dev=R43['dev'], em=R43['em'], total=R43['total'],
                    em_kwh=R43['em_kwh'], adjust=BEST43, zero=o43['仅0:00']['total'],
                    full=o43['全滚动']['total']),
           q3_options={k: dict(plan=v['plan'], dev=v['dev'], em=v['em'], total=v['total']) for k, v in opts.items()},
           q43_options={k: dict(plan=v['plan'], dev=v['dev'], em=v['em'], total=v['total']) for k, v in o43.items()})
json.dump(num, open(os.path.join(DATA, 'plan_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\n[完成] 耗时 {time.time()-t0:.0f}s -> data/plan_numbers.json')
