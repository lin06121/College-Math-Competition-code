# -*- coding: utf-8 -*-
"""阶段四: 端到端费用评价(因果管道) —— 预测模型替换后的全年费用
   基准: 问题二 13,848,183 元; 问题四之二 14,531,823 元
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from fc_common import D, L_ACT, PV_ACT, RTP, run_q2_fc, run_q3_fc, metrics, DATA
import models_extra as MX

PZ = dict(np.load(os.path.join(DATA, 'final_preds.npz')))
BASE = dict(L='L|基线(近4个同类日均值)', PV='PV|基线(近4日均值)', P='P|已知实时电价')
rows = []
t0 = time.time()


def tune_q(Lh, PVh, pmat=None, grid=(0.5, 0.6, 0.7, 0.8, 0.9), buf=None):
    tab = {}
    for q in grid:
        R = run_q2_fc(Lh, PVh, q=q, price_plan=pmat, upto=31, buffer_mat=None if buf is None else buf)
        tab[q] = float((R['pc'] + R['ec'])[7:31].sum())
    return min(tab, key=tab.get), tab


def run(tag, Ltag, PVtag, Ptag='P|已知实时电价', q=None, buf=None, pmat=None, jan=True,
        bill_actual=True):
    Lh = PZ[Ltag]; PVh = PZ[PVtag]; Ph = PZ[Ptag]
    is_known_price = (Ptag == BASE['P'])
    pplan = None if is_known_price else Ph
    pbill = RTP if (bill_actual and not is_known_price) else None
    t1 = time.time()
    qs_txt = ''
    if jan:
        qs, tab = tune_q(Lh, PVh, pmat=pplan)
        qs_txt = '1月 q 网格: ' + ' '.join(f'{k}:{v:,.0f}' for k, v in tab.items()) + f' -> 选 q={qs}'
        if q is None:
            q = qs
    R = run_q2_fc(Lh, PVh, q=q, price_plan=pplan, price_bill=pbill, buffer_mat=buf)
    mtL = metrics(Lh, L_ACT)['mae_kw']; mtPV = metrics(PVh, PV_ACT)['mae_kw']
    print(f'{tag:44s} q={q} 计划 {R["plan"]:>12,.0f} 紧急 {R["em"]:>10,.0f} 总 {R["total"]:>13,.0f} '
          f'| MAE {mtL:6.1f}/{mtPV:6.1f} | {time.time()-t1:.0f}s')
    print(f'    {qs_txt}')
    rows.append(dict(tag=tag, q=q, plan=R['plan'], em=R['em'], total=R['total'],
                     em_kwh=R['em_kwh'], mae_L=mtL, mae_PV=mtPV))
    return R


print('=== 基准(论文口径) ===')
run('① 基准: 同类日均值 + 近4日光伏  q=0.7', BASE['L'], BASE['PV'], q=0.7, jan=False)
print('=== 替换光伏预测 ===')
run('② 光伏=堆叠(136.9 kW)', BASE['L'], 'PV|堆叠(mean4+ef2+lgbm+fc0)')
run('③ 光伏=融合0.6(140.8 kW)', BASE['L'], 'PV|融合0.6xmean4+0.4xlgbm')
print('=== 同时替换负荷预测 ===')
run('④ 负荷=ef3 + 光伏=堆叠', 'L|误差反馈ef3', 'PV|堆叠(mean4+ef2+lgbm+fc0)')
print('=== 条件分位缓冲(GBDT) ===')
Lh = PZ[BASE['L']]; PVh = PZ['PV|堆叠(mean4+ef2+lgbm+fc0)']
t1 = time.time()
BUF = MX.buffer_gbdt(Lh, PVh, q=0.7)
from fc_common import buffer_from_resid
BUF_EMP = buffer_from_resid(Lh, PVh, 0.7)
cc = float(np.corrcoef(BUF[60:].ravel(), BUF_EMP[60:].ravel())[0, 1])
print(f'  [条件分位缓冲] 训练完成 {time.time()-t1:.0f}s | 平均裕度 {BUF.mean()*6:.1f} kW '
      f'(经验分位 {BUF_EMP.mean()*6:.1f} kW) | 两者相关 {cc:.3f}')
R = run_q2_fc(Lh, PVh, q=0.7, buffer_mat=BUF)
print(f'⑤ 条件分位缓冲(GBDT, q=0.7): 计划 {R["plan"]:,.0f} 紧急 {R["em"]:,.0f} 总 {R["total"]:,.0f}')
rows.append(dict(tag='⑤ 条件分位缓冲(GBDT,q0.7)', q=0.7, plan=R['plan'], em=R['em'], total=R['total'],
                 em_kwh=R['em_kwh'], mae_L=metrics(Lh, L_ACT)['mae_kw'], mae_PV=metrics(PVh, PV_ACT)['mae_kw']))
BUF8 = MX.buffer_gbdt(Lh, PVh, q=0.8)
R = run_q2_fc(Lh, PVh, q=0.8, buffer_mat=BUF8)
print(f'⑥ 条件分位缓冲(GBDT, q=0.8): 计划 {R["plan"]:,.0f} 紧急 {R["em"]:,.0f} 总 {R["total"]:,.0f}')
rows.append(dict(tag='⑥ 条件分位缓冲(GBDT,q0.8)', q=0.8, plan=R['plan'], em=R['em'], total=R['total'],
                 em_kwh=R['em_kwh'], mae_L=metrics(Lh, L_ACT)['mae_kw'], mae_PV=metrics(PVh, PV_ACT)['mae_kw']))
np.savez_compressed(os.path.join(DATA, 'buffer_gbdt.npz'), b07=BUF, b08=BUF8)

print('=== 问题四之二(实时电价) ===')
run('⑦ Q4-2 基准(电价已知)', BASE['L'], BASE['PV'], q=0.7, jan=False)
run('⑧ Q4-2 光伏=堆叠, 电价已知', BASE['L'], 'PV|堆叠(mean4+ef2+lgbm+fc0)')
run('⑨ Q4-2 光伏=堆叠, 电价用堆叠预测', BASE['L'], 'PV|堆叠(mean4+ef2+lgbm+fc0)', Ptag='P|堆叠(廓线+ef+GBDT)')
run('⑩ Q4-2 光伏=堆叠, 电价用廓线', BASE['L'], 'PV|堆叠(mean4+ef2+lgbm+fc0)', Ptag='P|廓线+自适应水平')

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA, 'cost_final.csv'), index=False, encoding='utf-8-sig')
print('\n' + df.to_string(index=False))
print(f'[总耗时] {time.time()-t0:.0f}s')
