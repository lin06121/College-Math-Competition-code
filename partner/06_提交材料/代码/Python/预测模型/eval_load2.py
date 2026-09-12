# -*- coding: utf-8 -*-
"""负荷模型(二): 水平×形状分解 与 类别-时段面板岭回归
   目标: 在"周五六 / 其他"两类上都不劣于基线(近4个同类日均值)
"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fc_common import D, L_ACT, IS_LOW, DATA, n, H, DATES
import models_extra as MX

LOW = IS_LOW
m2 = D.m2


def mae_cls(v):
    e = np.abs(v[m2:] - L_ACT[m2:]).mean(axis=1) * 6
    return float(e.mean()), float(e[LOW[m2:]].mean()), float(e[~LOW[m2:]].mean())


# ---------- 模型 A: 水平×形状分解 ----------
def shape_by_class(base, A_act, K=6):
    """类别形状: 同类别日的归一化曲线均值(近 K 个同类日)"""
    sh = np.zeros((n, H))
    for j in range(n):
        hist = [q for q in range(j) if LOW[q] == LOW[j]][-K:]
        if not hist:
            sh[j] = base[j] / max(base[j].sum(), 1e-6); continue
        A = A_act[hist]
        sh[j] = (A / np.maximum(A.sum(axis=1, keepdims=True), 1e-6)).mean(axis=0)
    return sh


def level_gbdt(A_act, K=6, retrain=14):
    """日总量预测: LightGBM(特征=近期总量/同类日总量/日历)"""
    import lightgbm as lgb
    tot = A_act.sum(axis=1)
    F = []
    for j in range(n):
        hist = [q for q in range(j) if LOW[q] == LOW[j]][-4:]
        cls_tot = np.mean([tot[q] for q in hist]) if hist else tot[max(0, j - 1)]
        F.append([LOW[j], D.dates[j].dayofweek, D.dates[j].dayofyear / 365.0,
                  tot[j - 1] if j >= 1 else tot[0],
                  tot[max(0, j - 3):j].mean() if j >= 1 else tot[0],
                  tot[max(0, j - 7):j].mean() if j >= 1 else tot[0],
                  cls_tot, np.sin(2 * np.pi * D.dates[j].dayofyear / 365),
                  np.cos(2 * np.pi * D.dates[j].dayofyear / 365)])
    F = np.array(F, float)
    out = np.zeros(n)
    for d0 in range(0, n, retrain):
        d1 = min(n, d0 + retrain); tr = np.arange(0, d0)
        if len(tr) < 40:
            out[d0:d1] = tot[max(0, d0 - 1)]
            continue
        m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.06, num_leaves=15,
                              min_child_samples=20, verbose=-1, n_jobs=4)
        m.fit(F[tr], tot[tr])
        out[d0:d1] = m.predict(F[d0:d1])
    return out


# ---------- 模型 B: 类别-时段面板岭回归 ----------
def panel_ridge(A_act, retrain=14, alpha=3.0, resid_on=None):
    """特征: 同类4日均值 / 同类上一日 / 昨日 / 日总量比 / 时段与类别交互;
       残差模式: 以基线为基准学习偏差"""
    from sklearn.linear_model import Ridge
    cls4 = D.Lb
    prev_cls = np.zeros((n, H))
    for j in range(n):
        hist = [q for q in range(j) if LOW[q] == LOW[j]]
        prev_cls[j] = A_act[hist[-1]] if hist else cls4[j]
    lag1 = np.vstack([A_act[0], A_act[:-1]])
    lev = np.repeat(MX._daily_level if False else 0, 0)
    tot = A_act.sum(axis=1)
    levr = np.ones(n)
    for j in range(n):
        lo = max(0, j - 4)
        if j - lo >= 1:
            levr[j] = tot[j - 1] / max(tot[lo:j].mean(), 1e-6)
    t = (np.arange(H) + 1) / H
    X = np.stack([cls4, prev_cls, lag1, np.repeat(levr, H).reshape(n, H),
                  np.tile(t, (n, 1)),
                  cls4 * np.repeat(LOW.astype(float), H).reshape(n, H),
                  np.tile(t, (n, 1)) * np.repeat(LOW.astype(float), H).reshape(n, H)], axis=-1)
    Y = A_act if resid_on is None else (A_act - resid_on)
    out = np.zeros((n, H))
    for d0 in range(0, n, retrain):
        d1 = min(n, d0 + retrain); tr = np.arange(0, d0)
        if len(tr) < 40:
            out[d0:d1] = cls4[d0:d1]; continue
        m = Ridge(alpha=alpha).fit(X[tr].reshape(-1, X.shape[-1]), Y[tr].reshape(-1))
        pr = m.predict(X[d0:d1].reshape(-1, X.shape[-1])).reshape(d1 - d0, H)
        out[d0:d1] = pr + (0 if resid_on is None else resid_on[d0:d1])
    return np.clip(out, 0, None)


if __name__ == '__main__':
    base = D.Lb
    print(f'{"模型":30s} {"总MAE":>8s} {"周五六":>8s} {"其他":>8s}')
    a, b, c = mae_cls(base); print(f'{"基线 mean4cls":30s} {a:8.2f} {b:8.2f} {c:8.2f}')

    for K in (4, 6, 8):
        sh = shape_by_class(base, L_ACT, K=K)
        for lg in ('gbdt', 'cls'):
            lv = level_gbdt(L_ACT) if lg == 'gbdt' else np.array(
                [L_ACT[[q for q in range(j) if LOW[q] == LOW[j]][-4:]].sum(axis=1).mean()
                 if j else L_ACT[0].sum() for j in range(n)])
            v = sh * lv[:, None]
            tag = f'水平×形状(K={K},{lg})'
            a, b, c = mae_cls(v); print(f'{tag:30s} {a:8.2f} {b:8.2f} {c:8.2f}')

    for al in (1.0, 3.0, 10.0):
        v = panel_ridge(L_ACT, alpha=al)
        tag = f'面板岭回归(alpha={al})'
        a, b, c = mae_cls(v); print(f'{tag:30s} {a:8.2f} {b:8.2f} {c:8.2f}')
    for al in (1.0, 3.0, 10.0):
        v = panel_ridge(L_ACT, alpha=al, resid_on=base)
        tag = f'基线+面板岭残差(alpha={al})'
        a, b, c = mae_cls(v); print(f'{tag:30s} {a:8.2f} {b:8.2f} {c:8.2f}')
