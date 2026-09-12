# -*- coding: utf-8 -*-
"""净需求直接建模(N = L - PV): 与"分别建负载/光伏再做差"对照
   计划关注的量正是净需求, 直接对 N 建模可利用两噪声的相关结构。
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, ROOT
import fc_upgraded as FU
import fc_features as FF
import plan_core as PC
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import lightgbm as lgb

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
N_ACT = L_ACT - PV_ACT
DATA = os.path.join(ROOT, '04_建模求解', 'data')
t0 = time.time()

print('=== 1) 现有两个模型做差 ===')
L, _ = FU.build_load(); PV = FU.build_pv()
N_diff = L - PV
e = N_ACT[m2:] - N_diff[m2:]
print(f'  MAE(净需求) = {np.abs(e).mean()*6:.2f} kW | RMSE {np.sqrt((e**2).mean())*6:.2f} kW')

print('=== 2) 直接对净需求建模 ===')
X, names = FF.build_X('L')                     # 复用特征(时段/日历/滞后/滚动/附件3预报/电价)
Fm = X.reshape(n, H, -1)
F = Fm.shape[-1]
# 追加: 现有 L̂ 与 P̂V 作为特征(栈式)
Xall = np.concatenate([Fm, L[:, :, None], PV[:, :, None]], axis=-1)
F2 = Xall.shape[-1]


def causal_fit_predict(model_fn, Xm, Y, retrain=14):
    out = np.zeros((n, H))
    for d0 in range(0, n, retrain):
        d1 = min(n, d0 + retrain)
        tr = np.arange(0, d0)
        if len(tr) < 40:
            out[d0:d1] = Y[:max(1, d0)].mean(axis=0); continue
        m = model_fn()
        m.fit(Xm[tr].reshape(-1, Xm.shape[-1]), Y[tr].reshape(-1))
        out[d0:d1] = m.predict(Xm[d0:d1].reshape(-1, Xm.shape[-1])).reshape(d1 - d0, H)
    return out


cands = {}
cands['LGBM(净需求, 含L̂/P̂V特征)'] = causal_fit_predict(
    lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                              min_child_samples=40, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.9, verbose=-1, n_jobs=4), Xall, N_ACT)
cands['HGB(净需求)'] = causal_fit_predict(
    lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                          min_samples_leaf=40), Xall, N_ACT)
cands['岭回归(净需求)'] = causal_fit_predict(lambda: Ridge(alpha=1.0), Xall, N_ACT)
# 残差学习: 在 N_diff 基础上学残差
cands['LGBM(净需求残差)'] = N_diff + causal_fit_predict(
    lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              min_child_samples=40, verbose=-1, n_jobs=4), Xall, N_ACT - N_diff)
for k, v in cands.items():
    ee = N_ACT[m2:] - v[m2:]
    print(f'  {k:26s} MAE {np.abs(ee).mean()*6:7.2f} kW | RMSE {np.sqrt((ee**2).mean())*6:7.2f} kW')

print('=== 3) 直接净需求模型 + 两阶段随机规划(2 月快评) ===')
BEST = min(cands, key=lambda k: np.abs(N_ACT[m2:] - cands[k][m2:]).mean())
print(f'  最优净需求模型: {BEST}')
N_use = cands[BEST]
L_eff = N_use + PV          # 等价的"负载"输入(保持 plan_core 接口): P 覆盖 N_use, 光伏按 PV 计
# 注意: 计划用 (L_eff, PV) 与 (N_use) 等价: 净需求相同, 光伏侧只影响上网/充电的物理约束


def lstp_chain(Lb, PB, q=0.8, K=30, S=20, mode='tail', upto=None):
    from q2_methods2 import scen_resid, chain  # 复用情景构造
    return chain('M1', q=q, K=K, S=S, mode=mode, upto=upto)


R0 = lstp_chain(L, PV, upto=59)
print(f'  现有模型(差法) + 两阶段: 计划 {R0["plan"]:,.0f} + 紧急 {R0["em"]:,.0f} = {R0["total"]:,.0f} 元')
R1 = lstp_chain(L_eff, PV, upto=59)
print(f'  净需求直接建模     + 两阶段: 计划 {R1["plan"]:,.0f} + 紧急 {R1["em"]:,.0f} = {R1["total"]:,.0f} 元')
print(f'[耗时] {time.time()-t0:.0f}s')
np.savez_compressed(os.path.join(DATA, 'net_model_preds.npz'),
                    **{k.replace('(', '_').replace(')', '').replace(',', '').replace(' ', ''): v
                       for k, v in cands.items()}, N_diff=N_diff)
