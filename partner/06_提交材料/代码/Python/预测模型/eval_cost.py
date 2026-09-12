# -*- coding: utf-8 -*-
"""阶段二: 端到端费用评价 —— 把预测矩阵注入因果管道, 与论文口径(基线)对比
   Q2(固定电价): 基线 13,848,183 元 ; Q4-2(实时电价): 基线 14,531,823 元
   每个组合先在 1 月网格标定报童分位 q, 再跑全年
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from fc_common import (D, L_ACT, PV_ACT, PRICE, RTP, run_q2_fc, metrics, DATA, n)

PZ = np.load(os.path.join(DATA, 'preds.npz'))


def get(tag, target):
    key = f'{target}|{tag}'
    return PZ[key] if key in PZ else None


BASE_L = '基线:近4个同类日均值'
BASE_PV = '基线:近4日均值'


def tune_q(Lh, PVh, pmat=None, grid=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """1 月(1.8-1.31)网格标定: 返回 (q*, {q: cost})"""
    out = {}
    for q in grid:
        R = run_q2_fc(Lh, PVh, q=q, price_plan=pmat, upto=31)
        out[q] = float((R['pc'] + R['ec'])[7:31].sum())
    qs = min(out, key=out.get)
    return qs, out


def eval_combo(name, Lh, PVh, q=None, pmat=None, jan=True):
    if Lh is None:
        Lh = D.Lb
    if PVh is None:
        PVh = D.Pb
    t0 = time.time()
    qinfo = ''
    if jan:
        qs, tab = tune_q(Lh, PVh, pmat)
        qinfo = f'1月最优 q={qs} ' + ' '.join(f'{k}:{v:,.0f}' for k, v in tab.items())
        if q is None:
            q = qs
    R = run_q2_fc(Lh, PVh, q=q, price_plan=pmat)
    mae_l = metrics(Lh, L_ACT)['mae_kw']
    mae_pv = metrics(PVh, PV_ACT)['mae_kw']
    print(f'{name:38s} | q={q} | 计划 {R["plan"]:>12,.0f} 紧急 {R["em"]:>10,.0f} '
          f'总 {R["total"]:>13,.0f} | MAE L={mae_l:6.1f} PV={mae_pv:6.1f} | {time.time()-t0:.0f}s')
    print(f'    [{qinfo}]')
    return dict(name=name, q=q, plan=R['plan'], em=R['em'], total=R['total'],
                em_kwh=R['em_kwh'], mae_L=mae_l, mae_PV=mae_pv)


if __name__ == '__main__':
    rows = []
    print('=== 基线(论文口径) ===')
    rows.append(eval_combo('基线: 同类日平均 + 近4日光伏', None, None, q=0.7, jan=False))

    # 负荷侧候选
    for tag in ['指数加权同类日(8日,0.85)', '相似日kNN(5个同类日)', '岭回归(逐时段特征)',
                'LightGBM(逐时段特征)', '基线+LightGBM残差修正']:
        Lh = get(tag, 'L')
        if Lh is None:
            continue
        print(f'=== 负荷模型: {tag} (光伏用基线) ===')
        rows.append(eval_combo(f'L={tag} | PV=基线', Lh, None))

    # 光伏侧候选
    for tag in ['晴空指数模型(近10日k×包络)', '附件3 0:00预报', 'LightGBM(逐时段特征)',
                '基线+LightGBM残差修正']:
        PVh = get(tag, 'PV')
        if PVh is None:
            continue
        print(f'=== 光伏模型: {tag} (负荷用基线) ===')
        rows.append(eval_combo(f'L=基线 | PV={tag}', None, PVh))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, 'cost_table.csv'), index=False, encoding='utf-8-sig')
    print('\n' + df.to_string(index=False))
