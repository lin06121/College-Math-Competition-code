# -*- coding: utf-8 -*-
"""生成最终候选预测矩阵(用于端到端费用评价), 保存到 data/final_preds.npz
   L : 基线(近4个同类日均值) / 误差反馈ef3 / 类别感知堆叠
   PV: 基线(近4日均值) / 堆叠(mean4+ef2+lgbm+fc0) / 融合0.6
   P : 已知实时电价 / 廓线 / 廓线+ef+GBDT堆叠
"""
import sys, io, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fc_common import D, L_ACT, PV_ACT, FC0, RTP, metrics, DATA, IS_LOW, H, n
import models as M
import models_ml as ML
import models_extra as MX

t0 = time.time()
out = {}

print('负荷 ...')
out['L|基线(近4个同类日均值)'] = D.Lb
out['L|误差反馈ef3'] = MX.err_feedback(D.Lb, L_ACT, k=3, damp=0.35)
lg = ML.ml_forecast('L', 'lgbm')
# 类别感知堆叠: 候选 + 类别交互项(让模型自行决定周五六/其他两类的最优权重)
X = np.stack([D.Lb, out['L|误差反馈ef3'], lg], axis=-1)
low = np.repeat(IS_LOW.astype(float), H).reshape(n, H)
Xa = np.concatenate([X, X * low[..., None], low[..., None]], axis=-1)
from sklearn.linear_model import Ridge
st = np.zeros((n, H))
for d0 in range(0, n, 14):
    d1 = min(n, d0 + 14); tr = np.arange(0, d0)
    if len(tr) < 40:
        st[d0:d1] = D.Lb[d0:d1]; continue
    m = Ridge(alpha=1.0).fit(Xa[tr].reshape(-1, Xa.shape[-1]), L_ACT[tr].reshape(-1))
    st[d0:d1] = m.predict(Xa[d0:d1].reshape(-1, Xa.shape[-1])).reshape(d1 - d0, H)
out['L|类别感知堆叠'] = np.clip(st, 0, None)

print('光伏 ...')
out['PV|基线(近4日均值)'] = D.Pb
ef2 = MX.err_feedback(D.Pb, PV_ACT, k=2, damp=0.5, clip=(0.7, 1.3))
plgbm = ML.ml_forecast('PV', 'lgbm')
pvst = MX.causal_stack([D.Pb, ef2, plgbm, FC0], PV_ACT, clip=(0, None))
out['PV|堆叠(mean4+ef2+lgbm+fc0)'] = pvst
out['PV|融合0.6xmean4+0.4xlgbm'] = ML.blend(D.Pb, plgbm, 0.6)

print('电价 ...')
prof = M.price_models()['类别×时段廓线+自适应水平']
pef = MX.err_feedback(prof, RTP, k=2, damp=0.5, clip=(0.6, 1.6))
out['P|已知实时电价'] = RTP
out['P|廓线+自适应水平'] = prof
out['P|堆叠(廓线+ef+GBDT)'] = MX.causal_stack([prof, pef, ML.ml_forecast('P', 'lgbm')], RTP,
                                              clip=(0.005, 3.0))

for k, v in out.items():
    tgt, tag = k.split('|')
    A = {'L': L_ACT, 'PV': PV_ACT, 'P': RTP}[tgt]
    mt = metrics(v, A)
    print(f'  {k:34s} MAE={mt["mae_kw"]:7.2f}  RMSE={mt["rmse_kw"]:7.2f}')
np.savez_compressed(os.path.join(DATA, 'final_preds.npz'), **out)
print(f'[耗时] {time.time()-t0:.0f}s -> data/final_preds.npz')
