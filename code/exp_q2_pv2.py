# -*- coding: utf-8 -*-
"""问题二：真正的"季节项"测试（前一版把季节窗与近期窗混淆了）。

只有 2025 一年数据，故"同季节窗 ±W 天"其实就等于"最近 W 天"——没有真正引入季节信息。
本脚本改用一个**真正的季节模型**：对每个时段分别拟合 DOY 谐波回归（因果、每 14 天重拟合），
得到"纯季节曲线" $S(d,t)$，再与近期 4 日均值做凸组合，考察是否有改善。

用法：PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/exp_q2_pv2.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_q2_pv as E   # noqa: E402  复用数据与管线

N, M2, PV, DATES = E.N, E.M2, E.PV, E.DATES


def seasonal_slot_curve(refit=14, n_harm=2, min_train=30):
    """逐时段的 DOY 谐波季节曲线（因果：每 refit 天用 d0 之前的样本重拟合）。"""
    doy = DATES.dayofyear.values.astype(float)
    cols = [np.ones_like(doy)]
    for k in range(1, n_harm + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    X = np.column_stack(cols)
    out = np.full((N, PV.shape[1]), np.nan)
    for d0 in range(0, N, refit):
        tr = np.arange(0, d0)
        if len(tr) < min_train:
            continue
        B, *_ = np.linalg.lstsq(X[tr], PV[tr], rcond=None)
        hi = min(N, d0 + refit)
        out[d0:hi] = X[d0:hi] @ B
    return np.maximum(out, 0.0)


SS = seasonal_slot_curve()


def recent4(i):
    return np.maximum(PV[max(0, i - 4):i].mean(axis=0), 0.0) if i > 0 else E.recent_mean(i, 4)


def blend(i, w):
    s = SS[i]
    if not np.isfinite(s).all():
        return recent4(i)
    return np.maximum(w * s + (1 - w) * recent4(i), 0.0)


def mae(fn):
    e = np.array([PV[i] - fn(i) for i in range(M2, N)])
    return np.abs(e).mean()


def main():
    print("=== 光伏 MAE（2.1--12.31，单位 kW）===")
    print(f"{'近4日均值（采用）':<26}{mae(recent4):>8.2f}")
    print(f"{'纯季节曲线 S(d,t)':<26}{mae(lambda i: SS[i] if np.isfinite(SS[i]).all() else recent4(i)):>8.2f}")
    for w in (0.1, 0.2, 0.3, 0.5):
        print(f"{'凸组合 w=' + str(w):<26}{mae(lambda i, w=w: blend(i, w)):>8.2f}")

    print("\n=== 问题二全年总费用 ===")
    base_fn = lambda i: recent4(i)
    base_err = E.compute_residuals_mate(E.DATES, E.LOAD, E.PV, E.NET, E.CLS)
    r0 = E.run_year(base_fn, base_err)
    print(f"{'近4日均值（采用）':<26}{r0['total']:>14,.0f} 元")
    for w in (0.1, 0.2, 0.3, 0.5):
        fn = lambda i, w=w: blend(i, w)
        err = E.resid_for(fn)
        r = E.run_year(fn, err)
        print(f"{'凸组合 w=' + str(w):<26}{r['total']:>14,.0f} 元  ({r['total'] - r0['total']:+,.0f})")


if __name__ == "__main__":
    main()
