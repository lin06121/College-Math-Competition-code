# -*- coding: utf-8 -*-
"""问题二：光伏预测加入“季节性”后是否有改善？

采用口径（V0）= 最近 4 个连续日均值。测以下变体（全部因果：只用目标日之前的信息）：
  V1 季节乘子   : 近 4 日均值 × [S(d) / mean(S(近4日))]，S 为日光伏电量的因果谐波季节曲线
  V2 季节凸组合 : w·季节基线 + (1-w)·近 4 日均值（w 在 1 月标定集上取最优）
  V3 近 8 日均值
  V4 近 4 日均值 × 近期趋势(近2日/近4日 电量比, 限幅)
指标：光伏 MAE（2.1--12.31）与问题二全年总费用（同一条计划+因果结算管线）。

用法：PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/exp_q2_pv.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SLOTS_PER_DAY, HOURS_PER_SLOT, STORAGE_INITIAL_SOC   # noqa: E402
from forecast_q2 import load_historical_matrices                         # noqa: E402
from method_q2 import (make_class_array, forecast_mate, compute_residuals_mate,
                       quantile_margin, causal_dispatch)                 # noqa: E402
from daily_solver import build_and_solve                                 # noqa: E402

DATES, LOAD, PV, PRICE_MAT = load_historical_matrices()
NET = LOAD - PV
N = len(DATES)
CLS = make_class_array(DATES)
D2I = {pd.Timestamp(d): i for i, d in enumerate(DATES)}
M2 = int(np.where(DATES >= pd.Timestamp("2025-02-01"))[0][0])
PRICE = PRICE_MAT[D2I[pd.Timestamp("2025-01-01")]]
TDS = pd.date_range("2025-01-01", "2025-12-31", freq="D")
KEEP = pd.Timestamp("2025-02-01")

E_DAILY = PV.sum(axis=1)                       # 日光伏电量（kWh）


# ---------------------------------------------------------------- 季节曲线
def seasonal_curve(refit=14, n_harm=2, min_train=30):
    """因果的日光伏电量季节曲线：每 refit 天用 d0 之前的样本重拟合谐波回归。"""
    doy = DATES.dayofyear.values.astype(float)
    cols = [np.ones_like(doy)]
    for k in range(1, n_harm + 1):
        cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
    X = np.column_stack(cols)
    out = np.full(N, np.nan)
    for d0 in range(0, N, refit):
        tr = np.arange(0, d0)
        if len(tr) < min_train:
            continue
        beta, *_ = np.linalg.lstsq(X[tr], E_DAILY[tr], rcond=None)
        hi = min(N, d0 + refit)
        out[d0:hi] = np.maximum(X[d0:hi] @ beta, 0.0)
    return out


S = seasonal_curve()


def recent_mean(i, k):
    prev = np.arange(N) < i
    idx = np.where(prev)[0]
    if len(idx) == 0:
        from method_q2 import load_fallback_profile
        return load_fallback_profile()[1].copy()
    return np.maximum(PV[idx[-k:]].mean(axis=0), 0.0)


def recent_mean_same_class_load(i, k=4):
    return forecast_mate(i, DATES, LOAD, PV, CLS)[0]


# ---------------------------------------------------------------- 变体
DOY = DATES.dayofyear.values.astype(float)


def season_window(i, W):
    """季节窗均值：历史中与目标日“同一季节”（DOY 距离 ≤ W，按年环绕）的日子均值。"""
    if i == 0:
        return recent_mean(i, 4)
    d = DOY[i]
    dist = np.abs(DOY[:i] - d)
    dist = np.minimum(dist, 365.25 - dist)
    sel = dist <= W
    if sel.sum() == 0:
        return recent_mean(i, 4)
    return np.maximum(PV[:i][sel].mean(axis=0), 0.0)


def pv_variants(i):
    prev = np.where(np.arange(N) < i)[0]
    base4 = recent_mean(i, 4)
    out = {"V0 近4日（采用）": base4}
    out["V3 近8日均值"] = recent_mean(i, 8)

    fac = 1.0
    if i >= 4 and np.isfinite(S[i]) and np.isfinite(S[i - 4:i]).all():
        fac = float(np.clip(S[i] / max(S[i - 4:i].mean(), 1e-6), 0.6, 1.6))
    out["V1 近4日×季节乘子"] = np.maximum(base4 * fac, 0.0)

    w15 = season_window(i, 15)
    w30 = season_window(i, 30)
    out["V5 季节窗均值 W=15"] = w15
    out["V6 季节窗均值 W=30"] = w30
    out["V7 凸组合 0.5×近4日+0.5×W15"] = 0.5 * base4 + 0.5 * w15
    return out


# ---------------------------------------------------------------- 全年管线
def resid_for(pv_fn):
    err = np.full((N, SLOTS_PER_DAY), np.nan)
    for i in range(N):
        lf = recent_mean_same_class_load(i)
        pf = pv_fn(i)
        err[i] = NET[i] - (lf - pf)
    return err


def run_year(pv_fn, err, q=0.8):
    e0 = STORAGE_INITIAL_SOC
    tot = plan_c = em_c = em_k = 0.0
    for td in TDS:
        i = D2I[td]
        lf = recent_mean_same_class_load(i)
        pf = pv_fn(i)
        b = quantile_margin(err, i, q)
        sol = build_and_solve(PRICE, lf + b, pf, e0=e0, cyclic=False,
                              allow_emergency=False, verbose=False)
        P = sol["P_plan_kW"]
        P_ch, P_dis, P_em, E, _ = causal_dispatch(NET[i], P, e0)
        if td >= KEEP:
            plan_c += float((P * PRICE).sum() * HOURS_PER_SLOT)
            em_c += float((P_em * PRICE).sum() * HOURS_PER_SLOT * 5.0)
            em_k += float(P_em.sum() * HOURS_PER_SLOT)
        e0 = E[-1]
    return dict(plan=plan_c, em=em_c, total=plan_c + em_c, em_kwh=em_k)


def main():
    names = list(pv_variants(30).keys())
    # 基线残差用官方函数，保证 V0 与论文口径完全一致
    errs = {names[0]: compute_residuals_mate(DATES, LOAD, PV, NET, CLS)}
    for nm in names[1:]:
        errs[nm] = resid_for(lambda i, nm=nm: pv_variants(i)[nm])

    print(f"{'变体':<20}{'光伏 MAE(kW)':>14}{'总费用(元)':>16}{'紧急购电费':>14}{'紧急电量(kWh)':>16}")
    for nm in names:
        e = errs[nm]
        mae = np.abs(e[M2:]).mean()
        r = run_year(lambda i, nm=nm: pv_variants(i)[nm], e)
        print(f"{nm:<20}{mae:>14.2f}{r['total']:>16,.0f}{r['em']:>14,.0f}{r['em_kwh']:>16,.0f}")


if __name__ == "__main__":
    main()
