# -*- coding: utf-8 -*-
"""因果预测模型库(负荷/光伏/电价) —— 全部只用 d 日 0:00 之前的信息
   模型族:
     L: 基线同类日均值 | 指数加权同类日 | 岭回归 | LightGBM | 基线+GBDT残差修正 | 相似日kNN | 融合
     PV: 基线近4日均值 | 晴空指数模型 | 附件3 0:00预报 | LightGBM | 基线+GBDT残差 | 融合
     P : 附件4 实时电价(已知) | 类别×时段廓线+自适应水平 | 岭回归 | LightGBM
"""
import numpy as np
from fc_common import (n, H, D, L_ACT, PV_ACT, FC0, L_TYP, PV_TYP, PRICE, RTP, DOW, IS_LOW,
                       DATES, fourier)


# ============================ 特征构造 ============================
def _rolling_same_slot(A, k):
    """A: (n,H); 返回 (n,H): 第 j 行 = 前 k 天(j-k..j-1)同槽位均值(不足则用更少)"""
    out = np.zeros((n, H))
    for j in range(n):
        lo = max(0, j - k)
        out[j] = A[lo:j].mean(axis=0) if j > lo else A[0]
    return out


def _daily_level(A, k):
    """每日总量与前 k 天均值之比(反映近期用电水平)"""
    tot = A.sum(axis=1)
    out = np.ones(n)
    for j in range(n):
        lo = max(0, j - k)
        if j - lo >= 1:
            out[j] = tot[j - 1] / max(tot[lo:j].mean(), 1e-6)
    return out


def _cls_mean(A, K=4):
    """同类日(周五六/其他)近 K 个同槽位均值 -> (n,H)"""
    out = np.zeros((n, H))
    for j in range(n):
        hist = [q for q in range(j) if IS_LOW[q] == IS_LOW[j]]
        out[j] = A[hist[-K:]].mean(axis=0) if hist else A[0]
    return out


def build_X(target='L', extra_cols=()):
    """构造逐(日,槽)特征矩阵 (n*H, F) 与列名; 全部特征在 d 日 0:00 已知"""
    lag1 = np.vstack([L_ACT[0], L_ACT[:-1]])
    lag3 = np.vstack([np.repeat(L_ACT[:1], 1, axis=0)] * 0) if False else np.zeros((n, H))
    for j in range(n):
        lag3[j] = L_ACT[max(0, j - 3)]
    lag7 = np.zeros((n, H))
    for j in range(n):
        lag7[j] = L_ACT[max(0, j - 7)]
    r3 = _rolling_same_slot(L_ACT, 3)
    r7 = _rolling_same_slot(L_ACT, 7)
    c4 = _cls_mean(L_ACT, 4)
    lev1 = np.repeat(_daily_level(L_ACT, 1), H).reshape(n, H)
    lev7 = np.repeat(_daily_level(L_ACT, 7), H).reshape(n, H)

    pv_lag1 = np.vstack([PV_ACT[0], PV_ACT[:-1]])
    pv_r3 = _rolling_same_slot(PV_ACT, 3)
    pv_r5 = _rolling_same_slot(PV_ACT, 5)
    pv_r10 = _rolling_same_slot(PV_ACT, 10)

    t = (np.arange(H) + 1)
    hour = (t - 1) / 6.0
    t_mat = np.tile(t, (n, 1))
    hour_mat = np.tile(hour, (n, 1))
    sin1 = np.tile(np.sin(2 * np.pi * t / H), (n, 1)); cos1 = np.tile(np.cos(2 * np.pi * t / H), (n, 1))
    sin2 = np.tile(np.sin(4 * np.pi * t / H), (n, 1)); cos2 = np.tile(np.cos(4 * np.pi * t / H), (n, 1))
    islow = np.repeat(IS_LOW.astype(float), H).reshape(n, H)
    dow = np.repeat(DOW.astype(float), H).reshape(n, H)
    doy = np.array([d.dayofyear for d in DATES], float)
    doyf = np.tile(doy, (H, 1)).T
    siny = np.sin(2 * np.pi * doyf / 365.0); cosy = np.cos(2 * np.pi * doyf / 365.0)
    siny2 = np.sin(4 * np.pi * doyf / 365.0); cosy2 = np.cos(4 * np.pi * doyf / 365.0)

    base = dict(
        t=t_mat, sin1=sin1, cos1=cos1, sin2=sin2, cos2=cos2, hour=hour_mat,
        islow=islow, dow=dow, siny=siny, cosy=cosy, siny2=siny2, cosy2=cosy2,
        lag1=lag1, lag3=lag3, lag7=lag7, r3=r3, r7=r7, cls4=c4, lev1=lev1, lev7=lev7,
        pv_lag1=pv_lag1, pv_r3=pv_r3, pv_r5=pv_r5, pv_r10=pv_r10,
        fc0=FC0, fc0_tot=np.tile(FC0.sum(axis=1), (H, 1)).T,
        rtp_lag1=np.vstack([RTP[0], RTP[:-1]]), rtp_r3=_rolling_same_slot(RTP, 3),
        rtp_r7=_rolling_same_slot(RTP, 7),
        l_r3=np.tile(r3.sum(axis=1), (H, 1)).T, pv_r5_tot=np.tile(pv_r5.sum(axis=1), (H, 1)).T,
    )
    for c in extra_cols:
        base[c] = extra_cols[c] if isinstance(extra_cols, dict) else None
    names = list(base.keys())
    Fm = np.stack([np.asarray(base[k], float).ravel() for k in names], axis=1)
    return Fm, names


# ============================ 负荷模型 ============================
def load_models():
    out = {}
    out['基线:近4个同类日均值'] = D.Lb.copy()

    # 指数加权同类日(近 8 个同类日, 衰减 0.85)
    ew = np.zeros((n, H)); lam = 0.85
    for j in range(n):
        hist = [q for q in range(j) if IS_LOW[q] == IS_LOW[j]][-8:]
        if not hist:
            ew[j] = L_TYP
        else:
            w = lam ** np.arange(len(hist) - 1, -1, -1)
            ew[j] = (L_ACT[hist] * w[:, None]).sum(0) / w.sum()
    out['指数加权同类日(8日,0.85)'] = ew

    # 相似日 kNN: 与最近 2 天形状最相似的 4 个同类日加权平均
    kNN = np.zeros((n, H))
    for j in range(n):
        hist = [q for q in range(j) if IS_LOW[q] == IS_LOW[j]]
        if len(hist) < 2:
            kNN[j] = D.Lb[j]; continue
        ref = L_ACT[max(0, j - 2):j].mean(axis=0)          # 近 2 天平均形状
        sc, ww = [], []
        for q in hist:
            a = L_ACT[q]
            sc.append(np.abs(a - ref).mean() + 0.2 * abs(a.sum() - ref.sum()) / H)
        sc = np.array(sc)
        k = min(5, len(hist))
        idx = np.argsort(sc)[:k]
        w = 1.0 / (sc[idx] + 1e-6)
        kNN[j] = (L_ACT[np.array(hist)[idx]] * (w / w.sum())[:, None]).sum(0)
    out['相似日kNN(5个同类日)'] = kNN
    return out


# ============================ 光伏模型 ============================
def clearsky(d, win=60, q=0.97):
    """晴空包络: 近 win 天同槽位高分位(仅用 d 之前) + 跨槽位平滑"""
    lo = max(0, d - win)
    if d - lo < 10:
        return np.maximum(D.PV_typ.copy(), 1e-6)
    seg = PV_ACT[lo:d]
    cs = np.quantile(seg, q, axis=0)
    k = np.ones(7) / 7
    cs = np.convolve(np.concatenate([cs[:3], cs, cs[-3:]]), k, mode='same')[3:-3]
    return np.maximum(cs, 1e-6)


def pv_models():
    out = {}
    out['基线:近4日均值'] = D.Pb.copy()
    out['附件3 0:00预报'] = FC0.copy()

    csi = np.zeros((n, H))
    for j in range(n):
        cs = clearsky(j)
        hist = [q for q in range(max(0, j - 10), j)]
        ks, ws = [], []
        for q in hist:
            day = PV_ACT[q]
            mask = cs > 0.05 * cs.max()
            if mask.sum() < 10:
                continue
            ks.append(np.median(day[mask] / cs[mask])); ws.append(0.85 ** (j - q))
        if ks:
            k = float(np.average(ks, weights=ws))
        else:
            k = 0.6
        # 用附件3 的当日预报修正水平(取白天时段中位数比)
        mask = cs > 0.05 * cs.max()
        if mask.sum() >= 10 and FC0[j].max() > 1e-6:
            k_fc = float(np.median(FC0[j][mask] / cs[mask]))
            k = 0.5 * k + 0.5 * k_fc
        csi[j] = np.clip(k, 0, 1.3) * cs
    out['晴空指数模型(近10日k×包络)'] = csi
    return out


# ============================ 电价模型 ============================
def price_models():
    out = {}
    out['已知实时电价(附件4)'] = RTP.copy()

    # 类别×时段廓线 + 自适应水平(近 3 个同类日中位数)
    prof = np.zeros((n, H))
    for j in range(n):
        hist = [q for q in range(j) if IS_LOW[q] == IS_LOW[j]][-6:]
        if not hist:
            prof[j] = PRICE; continue
        base = np.median(RTP[hist], axis=0)
        lvl_hist = np.array([RTP[q].mean() for q in hist])
        lvl_now = RTP[hist[-1]].mean()
        scale = lvl_now / max(lvl_hist.mean(), 1e-6)
        prof[j] = np.clip(base * (0.5 + 0.5 * scale), 0.005, 3.0)
    out['类别×时段廓线+自适应水平'] = prof
    return out
