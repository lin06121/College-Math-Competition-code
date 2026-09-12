# -*- coding: utf-8 -*-
"""导出预测模型实验所需/可比对的矩阵(供 MATLAB 侧使用)
   输出目录: 07_预测模型实验/data/matlab_in/*.csv  (均为 365x144, 6 位小数)
   说明: MATLAB 侧无任何工具箱(无 LightGBM 等价物), 故两个"机器学习基模型"的预测
       以 CSV 形式作为输入序列提供(与附件3 的预报地位相同), 其余全部在 MATLAB 内实现。
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from fc_common import D, L_ACT, PV_ACT, FC0, FCADJ, RTP, PRICE, DATA, n, H
import models as M
import models_ml as ML
import models_extra as MX

OUT = os.path.join(DATA, 'matlab_in')
os.makedirs(OUT, exist_ok=True)


def save(name, A):
    pd.DataFrame(A).to_csv(os.path.join(OUT, name + '.csv'), index=False, header=False,
                           float_format='%.6f')
    print(f'  {name}.csv  {A.shape}')


print('=== 导出输入序列 ===')
save('L_act', L_ACT); save('PV_act', PV_ACT); save('RTP', RTP)
save('PRICE_fixed', np.broadcast_to(PRICE, (n, H)))
save('FC0', FC0)
for h in (6, 12, 18):
    save(f'FC{h}', FCADJ[h])

print('=== 导出基模型(PV) ===')
ef2 = MX.err_feedback(D.Pb, PV_ACT, k=2, damp=0.5, clip=(0.7, 1.3))
lgbm_pv = ML.ml_forecast('PV', 'lgbm')
save('pvbase_mean4', D.Pb); save('pvbase_ef2', ef2); save('pvbase_lgbm', lgbm_pv); save('pvbase_fc0', FC0)

print('=== 导出基模型(电价) ===')
prof = M.price_models()['类别×时段廓线+自适应水平']
pef = MX.err_feedback(prof, RTP, k=2, damp=0.5, clip=(0.6, 1.6))
lgbm_p = ML.ml_forecast('P', 'lgbm')
save('pbase_profile', prof); save('pbase_ef', pef); save('pbase_lgbm', lgbm_p)

print('=== 导出 Python 侧的最终预测(用于 MATLAB 自检比对) ===')
pvst = MX.causal_stack([D.Pb, ef2, lgbm_pv, FC0], PV_ACT, clip=(0, None))
pst = MX.causal_stack([prof, pef, lgbm_p], RTP, clip=(0.005, 3.0))
LC = np.load(os.path.join(DATA, 'load_combined.npz'))['L_comb']
save('ref_pv_stack', pvst); save('ref_price_stack', pst); save('ref_load_combined', LC)
json.dump(dict(n=n, H=H, note='365x144 matrices, rows=2025-01-01..12-31'),
          open(os.path.join(OUT, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('->', OUT)
