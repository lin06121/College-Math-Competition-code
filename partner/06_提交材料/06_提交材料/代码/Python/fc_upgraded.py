# -*- coding: utf-8 -*-
"""升级版预测模型(正式采用版, 与 07_预测模型实验 的结论一致, 自包含实现)
   L : 水平×形状分解(K=6) + 递推水平平滑, 按"周五六 / 其他五天"分类凸组合(1月标定权重)
   PV: 因果岭回归堆叠 [近4日均值, 误差反馈, LightGBM, 附件3 的 0:00 预报]
   P : 类别×时段廓线+自适应水平, 因果岭回归堆叠 [廓线, 误差反馈, LightGBM]
   约定: 预测第 d 天只用 j<d 的信息; ML 模型每 14 天用截至当时的样本重训
"""
import numpy as np
from causal_core import D, price as PRICE_FIX, PR as RTP_M  # noqa: F401

n, H = D.n, 144
LOW = np.array([d in {4, 5} for d in D.dow])
RETRAIN = 14
ALPHA = 1.0


# ---------- 工具 ----------
def _ridge_stack(cands, Y, retrain=RETRAIN, alpha=ALPHA, clip=(0.0, np.inf)):
    """因果岭回归堆叠(含时段/类别交互特征), 返回 (n,H)"""
    m = len(cands)
    X = np.zeros((n, H, m + 3))
    for k, C in enumerate(cands):
        X[:, :, k] = C
    t = (np.arange(H) + 1) / H
    X[:, :, m] = t
    X[:, :, m + 1] = LOW.astype(float)[:, None]
    X[:, :, m + 2] = t * LOW.astype(float)[:, None]
    F = X.shape[2]
    out = np.zeros((n, H))
    from sklearn.linear_model import Ridge
    for d0 in range(0, n, retrain):
        d1 = min(n, d0 + retrain)
        tr = np.arange(0, d0)
        if len(tr) < 40:
            out[d0:d1] = cands[0][d0:d1]
            continue
        mdl = Ridge(alpha=alpha).fit(X[tr].reshape(-1, F), Y[tr].reshape(-1))
        out[d0:d1] = mdl.predict(X[d0:d1].reshape(-1, F)).reshape(d1 - d0, H)
    return np.clip(out, clip[0], clip[1])


def _err_feedback(base, A, k=2, damp=0.5, clip=(0.7, 1.3)):
    out = np.zeros((n, H))
    for j in range(n):
        hist = list(range(max(0, j - k), j))
        if not hist:
            out[j] = base[j]; continue
        r = np.clip(np.mean([A[q] / np.maximum(base[q], 1e-6) for q in hist], axis=0), clip[0], clip[1])
        out[j] = base[j] * ((1 - damp) + damp * r)
    return out


def _lgbm(target, base=None, params=None):
    """LightGBM 因果滚动重训; target: 'L'|'PV'|'P'; base 非空则预测残差"""
    from fc_ml import ml_forecast
    return ml_forecast(target, 'lgbm', base=base, params=params)


# ---------- 负荷: 水平×形状 + 递推水平 + 分类凸组合 ----------
def build_load(Lb=None, tune=True):
    """返回 (L_hat, info); Lb 缺省用论文口径的"近4个同类日均值"基值"""
    L = D.L_act
    Lb = D.Lb if Lb is None else Lb
    K = 6
    SH = np.zeros((n, H)); lv = np.zeros(n)
    for j in range(n):
        hist = [q for q in range(j) if LOW[q] == LOW[j]]
        if hist:
            idx = hist[-K:]
            A = L[idx]
            SH[j] = (A / np.maximum(A.sum(axis=1, keepdims=True), 1e-6)).mean(axis=0)
            lv[j] = L[hist[-4:]].sum(axis=1).mean() if len(hist) >= 1 else Lb[j].sum()
        else:
            SH[j] = Lb[j] / max(Lb[j].sum(), 1e-6); lv[j] = Lb[j].sum()
    M_shape = SH * lv[:, None]
    M_lvl = np.zeros((n, H)); s = 1.0
    for j in range(n):
        if j >= 1:
            s = 0.35 * (L[j - 1].sum() / max(Lb[j - 1].sum(), 1e-6)) + 0.65 * s
        M_lvl[j] = Lb[j] * np.clip(s, 0.85, 1.15)
    M_ef = _err_feedback(Lb, L, k=3, damp=0.35, clip=(0.85, 1.15))
    cands = [Lb, M_shape, M_lvl, M_ef]
    names = ['基线mean4cls', '水平×形状K6', '递推水平lvl', '全局反馈ef3']
    grid = np.arange(0, 1 + 1e-9, 0.1)
    W = [(a, b, c, 1 - a - b - c) for a in grid for b in grid for c in grid if a + b + c <= 1 + 1e-9]
    out = np.zeros((n, H)); info = {}
    for cls in (False, True):
        rows = [j for j in range(7, 31) if LOW[j] == cls]          # 1 月标定段
        best = None
        for w in W:
            P = sum(wi * cands[k] for k, wi in enumerate(w))
            e = np.abs(P[rows] - L[rows]).mean()
            if best is None or e < best[0]:
                best = (e, w)
        info['周五六' if cls else '其他'] = dict(weights={nm: round(wi, 2) for nm, wi in zip(names, best[1])})
        P = sum(wi * cands[k] for k, wi in enumerate(best[1]))
        out[LOW == cls] = P[LOW == cls]
    return np.clip(out, 0, None), info


# ---------- 光伏 / 电价 ----------
def build_pv(verbose=False):
    Pb = D.Pb
    ef = _err_feedback(Pb, D.PV_act, k=2, damp=0.5, clip=(0.7, 1.3))
    gb = _lgbm('PV')
    st = _ridge_stack([Pb, ef, gb, D.fc0m], D.PV_act, clip=(0.0, np.inf))
    if verbose:
        for nm, A in [('近4日均值(基线)', Pb), ('误差反馈', ef), ('LightGBM', gb)]:
            print(f'    PV {nm:16s} MAE={np.abs(A[D.m2:] - D.PV_act[D.m2:]).mean()*6:7.2f} kW')
        print(f'    PV 岭回归堆叠          MAE={np.abs(st[D.m2:] - D.PV_act[D.m2:]).mean()*6:7.2f} kW')
    return st


def build_price(verbose=False):
    """类别×时段廓线+自适应水平 -> 岭回归堆叠"""
    R = D.PR
    prof = np.zeros((n, H))
    for j in range(n):
        hist = [q for q in range(j) if LOW[q] == LOW[j]]
        if not hist:
            prof[j] = PRICE_FIX; continue
        idx = hist[-6:]
        base = np.median(R[idx], axis=0)
        lvls = R[idx].mean(axis=1)
        scale = lvls[-1] / max(lvls.mean(), 1e-6)
        prof[j] = np.clip(base * (0.5 + 0.5 * scale), 0.005, 3.0)
    ef = _err_feedback(prof, R, k=2, damp=0.5, clip=(0.6, 1.6))
    gb = _lgbm('P')
    st = _ridge_stack([prof, ef, gb], R, clip=(0.005, 3.0))
    if verbose:
        for nm, A in [('廓线+自适应水平', prof), ('LightGBM', gb)]:
            print(f'    P  {nm:16s} MAE={np.abs(A[D.m2:] - R[D.m2:]).mean():.4f} 元/kWh')
        print(f'    P  岭回归堆叠          MAE={np.abs(st[D.m2:] - R[D.m2:]).mean():.4f} 元/kWh')
    return st


def build_all(verbose=True):
    L, info = build_load()
    PV = build_pv(verbose)
    P = build_price(verbose)
    if verbose:
        e = np.abs(L[D.m2:] - D.L_act[D.m2:]).mean(axis=1) * 6
        lo = LOW[D.m2:]
        print(f'    负荷 分类组合模型 MAE: 总 {e.mean():.2f} / 周五六 {e[lo].mean():.2f} / 其他 {e[~lo].mean():.2f} kW')
        for k, v in info.items():
            print(f'      [{k}] ' + ', '.join(f'{a}:{b}' for a, b in v['weights'].items()))
    return L, PV, P, info
