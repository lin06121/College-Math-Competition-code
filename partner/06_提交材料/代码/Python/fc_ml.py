# -*- coding: utf-8 -*-
"""机器学习预测模型(因果滚动重训): 岭回归 / LightGBM / 基线+GBDT 残差修正
   约定: 预测第 d 天时, 只用 j <= d-1 的样本训练; 每 RETRAIN 天重训一次(预测块内用同一模型)
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from fc_shim import n, H, L_ACT, PV_ACT, RTP
from fc_features import build_X

TARGET_Y = {'L': L_ACT, 'PV': PV_ACT, 'P': RTP}
CLIP = {'L': (0.0, None), 'PV': (0.0, None), 'P': (0.0, 6.0)}
RETRAIN = 14


def _fit(kind, Xtr, ytr, params=None):
    if kind == 'ridge':
        from sklearn.linear_model import Ridge
        a = 1.0 if params is None else params.get('alpha', 1.0)
        m = Ridge(alpha=a, fit_intercept=True)
    elif kind == 'lgbm':
        import lightgbm as lgb
        pp = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                  subsample=0.9, subsample_freq=1, colsample_bytree=0.9, verbose=-1, n_jobs=4)
        if params:
            pp.update(params)
        m = lgb.LGBMRegressor(**pp)
    elif kind == 'hgb':
        from sklearn.ensemble import HistGradientBoostingRegressor
        pp = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40)
        if params:
            pp.update(params)
        m = HistGradientBoostingRegressor(**pp)
    else:
        raise ValueError(kind)
    m.fit(Xtr, ytr)
    return m


def ml_forecast(target='L', kind='lgbm', retrain=RETRAIN, base=None, params=None, seed=0):
    """返回 (n,H) 预测矩阵; base 非 None 时预测 base 的残差(混合模型)"""
    Xflat, names = build_X(target)
    X = Xflat.reshape(n, H, -1)
    Y = TARGET_Y[target].copy()
    if base is not None:
        Y = Y - base
    out = np.zeros((n, H))
    lo, hi = CLIP[target]
    for d0 in range(0, n, retrain):
        tr = np.arange(0, d0)
        if len(tr) < 30:
            pred_block = (base[d0:d0 + retrain] if base is not None else Y[d0:d0 + retrain].mean(axis=0))
            out[d0:d0 + retrain] = pred_block if base is not None else np.tile(
                np.zeros(H) if base is None else pred_block, (1, 1))
            if base is None:
                out[d0:d0 + retrain] = Y[:max(1, d0)].mean(axis=0)
            continue
        Xtr = X[tr].reshape(-1, X.shape[-1])
        ytr = Y[tr].reshape(-1)
        m = _fit(kind, Xtr, ytr, params)
        d1 = min(n, d0 + retrain)
        pr = m.predict(X[d0:d1].reshape(-1, X.shape[-1])).reshape(d1 - d0, H)
        if base is not None:
            pr = pr + base[d0:d1]
        out[d0:d1] = pr
    return np.clip(out, lo, hi)


def blend(A, B, w):
    """按权重 w 融合(A 权重 w)"""
    return w * A + (1 - w) * B
