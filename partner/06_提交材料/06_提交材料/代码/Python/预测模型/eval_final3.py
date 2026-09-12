# -*- coding: utf-8 -*-
"""最终端到端评价: 组合负荷模型 + 光伏堆叠 + 电价堆叠
   基线(论文口径): Q2 13,848,183 | Q3 14,047,116 | Q4-2 14,531,823 | Q4-3 14,721,611
   说明: 问题三的 0:00 计划使用注入的 PV 预测(基线为附件3 的 0:00 预报), 6/12/18 仍用附件3 更新预报
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from fc_common import (D, L_ACT, PV_ACT, FC0, RTP, PRICE, run_q2_fc, run_q3_fc, metrics,
                       buffer_from_resid, DATA, m2, n, DATES, IS_LOW)

PZ = dict(np.load(os.path.join(DATA, 'final_preds.npz')))
LC = np.load(os.path.join(DATA, 'load_combined.npz'))['L_comb']
PVS = PZ['PV|堆叠(mean4+ef2+lgbm+fc0)']
PS = PZ['P|堆叠(廓线+ef+GBDT)']
res = {}
t0 = time.time()


def tune_q(Lh, PVh, pmat=None, grid=(0.5, 0.6, 0.7, 0.8, 0.9), q3=False, buf_mat=None):
    """1 月(1.8-1.31)标定; q3=True 时用含调整费的管道"""
    tab = {}
    for q in grid:
        if q3:
            R = run_q3_fc(Lh, PVh, q=q, price_plan=pmat, upto=31, buf_mat=buf_mat)
            v = float((R['pc'] + R['dv'] + R['ec'])[7:31].sum())
        else:
            R = run_q2_fc(Lh, PVh, q=q, price_plan=pmat, upto=31, buffer_mat=buf_mat)
            v = float((R['pc'] + R['ec'])[7:31].sum())
        tab[q] = v
    return min(tab, key=tab.get), tab


def line(name, R, base=None):
    d = '' if base is None else f'  Δ {R["total"]-base:+,.0f} 元 ({(R["total"]-base)/base*100:+.2f}%)'
    print(f'{name:44s} {R["total"]:>14,.0f} 元{d}')
    return R['total']


print('=== 精度 ===')
for tag, A, B, sc in [('负荷', LC, L_ACT, 6), ('光伏', PVS, PV_ACT, 6), ('电价', PS, RTP, 1)]:
    m = metrics(A, B, scale=sc)
    print(f'  {tag}: MAE={m["mae_kw"]:.2f} {"kW" if sc==6 else "元/kWh"}  RMSE={m["rmse_kw"]:.2f}  '
          f'日总量MAPE={m["daily_mape"]:.2f}%')
e = np.abs(LC[m2:] - L_ACT[m2:]).mean(axis=1) * 6
print(f'  负荷分类 MAE: 周五六 {e[IS_LOW[m2:]].mean():.2f} kW / 其他 {e[~IS_LOW[m2:]].mean():.2f} kW')

print('\n=== ① 问题二(固定电价) ===')
R0 = run_q2_fc(D.Lb, D.Pb, q=0.7)
b = line('基线(论文口径)', R0)
q1, t1 = tune_q(LC, PVS)
print('   1月 q 网格:', {k: round(v) for k, v in t1.items()}, '-> q =', q1)
RN = run_q2_fc(LC, PVS, q=q1)
res['q2_base'] = b; res['q2_new'] = RN['total']
line(f'新模型组合(q={q1})', RN, b)
mm = ((RN['pc'] + RN['ec']) - (R0['pc'] + R0['ec']))[m2:]
mon = np.array([d.month for d in DATES[m2:]])
print('   逐月差异(元):', {int(k): int(mm[mon == k].sum()) for k in range(2, 13)})

print('\n=== ② 问题三(固定电价, 滚动调整) ===')
BUF_BASE = buffer_from_resid(D.Lb, D.Pb, 0.7)          # 论文口径的报童缓冲(基于近4日光伏基值残差)
R30 = run_q3_fc(D.Lb, FC0, q=0.7, buf_mat=BUF_BASE)
b3 = line('基线(0:00预报 + 全滚动, 论文口径)', R30)
q3s, t3 = tune_q(LC, PVS, q3=True)
print('   1月 q 网格(问题三管道):', {k: round(v) for k, v in t3.items()}, '-> 该管道最优 q =', q3s)
R3N = run_q3_fc(LC, PVS, q=q1)                          # ★ 统一用"问题二管道 1 月标定"的 q
res['q3_base'] = b3; res['q3_new'] = R3N['total']
line(f'新模型组合(统一 q={q1})', R3N, b3)
if q3s != q1:
    R3b = run_q3_fc(LC, PVS, q=q3s)
    res['q3_new_scenarioq'] = R3b['total']
    line(f'  (参考: 若按问题三管道 1 月标定 q={q3s})', R3b, b3)

print('\n=== ③ 问题四之二(实时电价) ===')
R40 = run_q2_fc(D.Lb, D.Pb, q=0.7, price_plan=RTP)
b42 = line('基线(电价已知)', R40)
R4N = run_q2_fc(LC, PVS, q=q1, price_plan=RTP)
res['q42_base'] = b42; res['q42_new'] = R4N['total']
line(f'新模型组合(电价已知, q={q1})', R4N, b42)
q4s, t4 = tune_q(LC, PVS, pmat=PS)
print('   1月 q 网格:', {k: round(v) for k, v in t4.items()}, '-> q =', q4s)
R4P = run_q2_fc(LC, PVS, q=q4s, price_plan=PS, price_bill=RTP)
res['q42_predprice'] = R4P['total']
line(f'新模型 + 电价用堆叠预测(结算按实际, q={q4s})', R4P, b42)

print('\n=== ④ 问题四之三(实时电价, 滚动) ===')
R430 = run_q3_fc(D.Lb, FC0, q=0.7, price_plan=RTP, buf_mat=BUF_BASE)
b43 = line('基线(0:00预报 + 全滚动, 论文口径)', R430)
R43N = run_q3_fc(LC, PVS, q=q1, price_plan=RTP)
res['q43_base'] = b43; res['q43_new'] = R43N['total']
line(f'新模型组合(统一 q={q1})', R43N, b43)

json.dump(res, open(os.path.join(DATA, 'final_cost.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\n[总耗时] {time.time()-t0:.0f}s')
