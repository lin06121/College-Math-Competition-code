# -*- coding: utf-8 -*-
"""补充预测模型: 误差反馈修正 / 稳健中位数 / 递推平滑水平 / 因果堆叠(stacking) / 条件分位缓冲"""
import numpy as np
from fc_common import n, H, D, L_ACT, PV_ACT, FC0, RTP, PRICE, DOW, IS_LOW, DATES
from models_ml import _fit

RETRAIN = 14


# ---------- 误差反馈(前一/二日的实测与预测之比, 阻尼后校正基值) ----------
def err_feedback(base, A_act, k=2, damp=0.5, clip=(0.9, 1.12)):
    out = np.zeros((n, H))
    for j in range(n):
        hist = list(range(max(0, j - k), j))
        if not hist:
            out[j] = base[j]; continue
        r = np.array([A_act[q] / np.maximum(base[q], 1e-6) for q in hist]).mean(axis=0)
        r = np.clip(r, clip[0], clip[1])
        out[j] = base[j] * ((1 - damp) + damp * r)
    return out


def robust_median(A, K):
    """近 K 天同槽位中位数"""
    out = np.zeros((n, H))
    for j in range(n):
        lo = max(0, j - K)
        out[j] = np.median(A[lo:j], axis=0) if j > lo else A[0]
    return out


def recency_weighted(A, w):
    """近 len(w) 天的加权平均(w[0] 为最近一天权重)"""
    out = np.zeros((n, H)); w = np.asarray(w, float); w = w / w.sum()
    for j in range(n):
        hist = list(range(max(0, j - len(w)), j))
        if not hist:
            out[j] = A[0]; continue
        ww = w[-len(hist):]; ww = ww / ww.sum()
        out[j] = (A[hist] * ww[:, None]).sum(0)
    return out


def level_smooth(base, A_act, alpha=0.3, K=4):
    """递推平滑日水平: 用前一日实际总量/预测总量之比做指数平滑校正"""
    out = np.zeros((n, H)); s = 1.0
    for j in range(n):
        if j >= 1:
            r = A_act[j - 1].sum() / max(base[j - 1].sum(), 1e-6)
            s = alpha * r + (1 - alpha) * s
        out[j] = base[j] * np.clip(s, 0.85, 1.15)
    return out


# ---------- 因果堆叠 ----------
def causal_stack(cands, Y, retrain=RETRAIN, alpha=1.0, slot_feat=True, clip=(0.0, None)):
    """cands: list of (n,H); 逐块用 j<d0 的样本拟合岭回归(可选含时段/类别特征)"""
    F = len(cands)
    X = np.stack(cands, axis=-1)                      # (n,H,F)
    if slot_feat:
        t = np.tile((np.arange(H) + 1) / H, (n, 1))[..., None]
        low = np.repeat(IS_LOW.astype(float), H).reshape(n, H)[..., None]
        X = np.concatenate([X, t, low, t * low], axis=-1)
    Fn = X.shape[-1]
    out = np.zeros((n, H))
    from sklearn.linear_model import Ridge
    for d0 in range(0, n, retrain):
        d1 = min(n, d0 + retrain)
        tr = np.arange(0, d0)
        if len(tr) < 40:
            out[d0:d1] = cands[0][d0:d1]; continue
        m = Ridge(alpha=alpha).fit(X[tr].reshape(-1, Fn), Y[tr].reshape(-1))
        out[d0:d1] = m.predict(X[d0:d1].reshape(-1, Fn)).reshape(d1 - d0, H)
    return np.clip(out, clip[0], clip[1])


# ---------- 条件分位缓冲(GBDT quantile) ----------
def buffer_gbdt(L_hat, PV_hat, q=0.7, retrain=RETRAIN, W=30):
    """用 LightGBM 分位回归给出"随特征变化"的安全裕度(因果): 目标=净需求预测残差"""
    e = (L_ACT - L_hat) - (PV_ACT - PV_hat)
    t = np.arange(H) + 1
    feats = []
    for j in range(n):
        lo = max(0, j - W)
        vol = e[lo:j].std(axis=0) if j - lo >= 7 else np.zeros(H)     # 近30天同槽位波动
        vol7 = e[max(0, j - 7):j].std(axis=0) if j >= 7 else np.zeros(H)
        row = np.column_stack([
            np.full(H, IS_LOW[j], float), np.full(H, DOW[j], float), t / H,
            np.sin(2 * np.pi * t / H), np.cos(2 * np.pi * t / H),
            np.full(H, DATES[j].dayofyear / 365.0, float),
            vol * 6, vol7 * 6, L_hat[j] * 6, PV_hat[j] * 6,
        ])
        feats.append(row)
    X = np.stack(feats)                                   # (n,H,F)
    import lightgbm as lgb
    out = np.zeros((n, H))
    for d0 in range(0, n, retrain):
        d1 = min(n, d0 + retrain)
        tr = np.arange(0, d0)
        if len(tr) < 40:
            out[d0:d1] = np.quantile(e[:max(1, d0)], q, axis=0) if d0 > 0 else 0.0
            continue
        m = lgb.LGBMRegressor(objective='quantile', alpha=q, n_estimators=200, learning_rate=0.06,
                              num_leaves=15, min_child_samples=60, verbose=-1, n_jobs=4)
        m.fit(X[tr].reshape(-1, X.shape[-1]), e[tr].reshape(-1))
        out[d0:d1] = m.predict(X[d0:d1].reshape(-1, X.shape[-1])).reshape(d1 - d0, H)
    return out
