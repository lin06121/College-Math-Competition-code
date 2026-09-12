# -*- coding: utf-8 -*-
"""自检: 主线自包含实现(fc_upgraded) 与 07_预测模型实验 的最终预测矩阵是否逐位一致"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import ROOT, D
import fc_upgraded as FU

REF = os.path.join(ROOT, '07_预测模型实验', 'data')
npz = dict(np.load(os.path.join(REF, 'final_preds.npz')))
lc = np.load(os.path.join(REF, 'load_combined.npz'))['L_comb']

L, info = FU.build_load()
PV = FU.build_pv()
P = FU.build_price()
m2 = D.m2
for nm, A, B, sc in [('负荷', L, lc, 6), ('光伏', PV, npz['PV|堆叠(mean4+ef2+lgbm+fc0)'], 6),
                     ('电价', P, npz['P|堆叠(廓线+ef+GBDT)'], 1)]:
    d = A[m2:] - B[m2:]
    print(f'{nm}: MAE差={np.abs(d).mean()*sc:.6f}  最大差={np.abs(d).max()*sc:.6f}  '
          f'完全一致={np.allclose(A, B, atol=1e-9)}')
