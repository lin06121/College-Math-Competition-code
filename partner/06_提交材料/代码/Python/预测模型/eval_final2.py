# -*- coding: utf-8 -*-
"""阶段四(修正版): 端到端费用评价
   ① 基准(论文口径): Q2 13,848,183 元 / Q4-2 14,531,823 元
   ② 光伏堆叠 / 负荷ef3 / 条件分位缓冲 / 电价预测 的逐项与组合效果
   ★ 实时电价场景必须显式传入 RTP 矩阵(price_plan=RTP), 否则管道会退回固定电价
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from fc_common import (D, L_ACT, PV_ACT, RTP, PRICE, run_q2_fc, run_q3_fc, metrics,
                       buffer_from_resid, DATA)
import models_extra as MX

PZ = dict(np.load(os.path.join(DATA, 'final_preds.npz')))
LB = 'L|基线(近4个同类日均值)'
LEF = 'L|误差反馈ef3'
PVB = 'PV|基线(近4日均值)'
PVS = 'PV|堆叠(mean4+ef2+lgbm+fc0)'
rows = []
t0 = time.time()


def tune_q(Lh, PVh, pmat, grid=(0.5, 0.6, 0.7, 0.8, 0.9)):
    tab = {}
    for q in grid:
        R = run_q2_fc(Lh, PVh, q=q, price_plan=pmat, upto=31)
        tab[q] = float((R['pc'] + R['ec'])[7:31].sum())
    return min(tab, key=tab.get), tab


def add(tag, R, Ltag, PVtag, note='', q=None):
    rows.append(dict(tag=tag, q=q, plan=R['plan'], em=R['em'], total=R['total'], em_kwh=R['em_kwh'],
                     mae_L=metrics(PZ[Ltag], L_ACT)['mae_kw'],
                     mae_PV=metrics(PZ[PVtag], PV_ACT)['mae_kw'], note=note))
    print(f'  → {tag:42s} 计划 {R["plan"]:>12,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元'
          f' | Δ基准 {R["total"]-13848183:+,.0f}')


print('=== ① 问题二(固定电价) ===')
print('基准(论文口径, q=0.7)')
R = run_q2_fc(PZ[LB], PZ[PVB], q=0.7)
add('①基准', R, LB, PVB, '论文口径')

print('光伏预测替换(x 负荷=基线)')
qs, tab = tune_q(PZ[LB], PZ[PVS], None)
print('    1月 q 网格:', {k: round(v) for k, v in tab.items()}, '-> q =', qs)
R = run_q2_fc(PZ[LB], PZ[PVS], q=qs)
add('②光伏=堆叠, 经验分位缓冲', R, LB, PVS, f'q={qs}', qs)

print('负荷+光伏同时替换')
qs2, tab2 = tune_q(PZ[LEF], PZ[PVS], None)
print('    1月 q 网格:', {k: round(v) for k, v in tab2.items()}, '-> q =', qs2)
R = run_q2_fc(PZ[LEF], PZ[PVS], q=qs2)
add('③负荷=ef3 + 光伏=堆叠', R, LEF, PVS, f'q={qs2}', qs2)

print('条件分位缓冲(GBDT) × 光伏堆叠')
nb = {}
for q in (0.7, 0.75, 0.8, 0.85):
    B = MX.buffer_gbdt(PZ[LEF], PZ[PVS], q=q)
    nb[f'b{int(q*100)}'] = B
    R = run_q2_fc(PZ[LEF], PZ[PVS], q=q, buffer_mat=B)
    add(f'④GBDT条件分位缓冲 q={q}', R, LEF, PVS, '条件缓冲', q)
BEST_Q = min([r for r in rows if r['tag'].startswith('④')], key=lambda r: r['total'])
np.savez_compressed(os.path.join(DATA, 'buffer_gbdt.npz'), **nb)

print('=== ② 问题四之二(实时电价) ===')
R = run_q2_fc(PZ[LB], PZ[PVB], q=0.7, price_plan=RTP)
add('⑤Q4-2 基准(电价已知)', R, LB, PVB, '论文口径')
R = run_q2_fc(PZ[LEF], PZ[PVS], q=0.7, price_plan=RTP)
add('⑥Q4-2 负荷ef3+光伏堆叠, 电价已知', R, LEF, PVS)
for ptag, nm in [('P|堆叠(廓线+ef+GBDT)', '电价=堆叠预测'), ('P|廓线+自适应水平', '电价=廓线预测')]:
    qs3, tab3 = tune_q(PZ[LEF], PZ[PVS], PZ[ptag])
    print(f'    [{nm}] 1月 q 网格:', {k: round(v) for k, v in tab3.items()}, '-> q =', qs3)
    R = run_q2_fc(PZ[LEF], PZ[PVS], q=qs3, price_plan=PZ[ptag], price_bill=RTP)
    add(f'⑦Q4-2 {nm}(结算按实际电价)', R, LEF, PVS, nm, qs3)
    print(f'       电价预测 MAE = {metrics(PZ[ptag], RTP, scale=1.0)["mae_kw"]:.4f} 元/kWh')

print('=== ③ 问题三(固定电价, 0:00预报+滚动调整) ===')
R3 = run_q3_fc(PZ[LB], PZ[PVB], q=0.7)
print(f'  基准: {R3["total"]:,.0f} 元 (期望 14,047,116)')
R3b = run_q3_fc(PZ[LEF], PZ[PVS], q=qs2)
print(f'  负荷ef3+光伏堆叠: {R3b["total"]:,.0f} 元  (Δ {R3b["total"]-R3["total"]:+,.0f})')
rows.append(dict(tag='⑧问题三 负荷ef3+光伏堆叠', q=qs2, plan=R3b['plan'], em=R3b['em'],
                 total=R3b['total'], em_kwh=R3b['em_kwh'],
                 mae_L=metrics(PZ[LEF], L_ACT)['mae_kw'], mae_PV=metrics(PZ[PVS], PV_ACT)['mae_kw'],
                 note='含滚动调整'))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA, 'cost_final.csv'), index=False, encoding='utf-8-sig')
print('\n' + df[['tag', 'q', 'plan', 'em', 'total', 'mae_L', 'mae_PV']].to_string(index=False))
print(f'\n[最优] {BEST_Q["tag"]} = {BEST_Q["total"]:,.0f} 元')
print(f'[总耗时] {time.time()-t0:.0f}s -> data/cost_final.csv')
