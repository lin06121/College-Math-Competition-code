"""
问题 4：附件 4 实时电价与电价预报。

问题 4 假设 0:00 制定计划时**不知道**当天电价（实时波动），只能使用历史电价
预报当天电价曲线；实时结算仍按实际电价。本模块负责：
- 读取附件 4 实时电价矩阵（`matrices.npz['price_realtime']`）；
- 构建因果电价预报：某天各时段的预计电价 = 此前 window 天内同一时段的平均电价，
  历史不足时退化为附件 1 典型日曲线。
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_PROCESSED, SLOTS_PER_DAY, HOURS_PER_SLOT


def load_price_realtime(path: Path = DATA_PROCESSED / "matrices.npz") -> np.ndarray:
    """读取附件 4 实时电价矩阵 (365, 144) 元/kWh。"""
    z = np.load(path, allow_pickle=True)
    return np.asarray(z["price_realtime"], dtype=float)


def build_price_forecast(price_realtime: np.ndarray, typical_day: np.ndarray,
                         window: int = 14) -> np.ndarray:
    """
    因果电价预报：日前 0:00 可用的电价预报。

    forecast[d, t] = mean( price_realtime[d-W : d, t] )   （近 W 天同时段平均）
    历史不足（d < W）退化为附件 1 典型日曲线——它作为题目给定信息先验可得。
    """
    n_days = price_realtime.shape[0]
    forecast = np.empty_like(price_realtime)
    typical = np.asarray(typical_day, dtype=float)
    if typical.ndim == 2:
        typical = typical[0]                       # 允许传整年矩阵，取第一行
    for d in range(n_days):
        if d >= window:
            forecast[d] = price_realtime[d - window:d].mean(axis=0)
        else:
            forecast[d] = typical
    return forecast


def price_forecast_mae(price_realtime: np.ndarray, forecast: np.ndarray,
                       dates=None, start_date="2025-02-01") -> dict:
    """电价预报精度（2/1 起）。"""
    if dates is None:
        dates = np.arange(price_realtime.shape[0])
    keep = np.array([pd.Timestamp(d) >= pd.Timestamp(start_date) for d in dates])
    e = (forecast - price_realtime)[keep]
    return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                mbe=float(e.mean()))


if __name__ == "__main__":
    from forecast_q2 import load_historical_matrices
    dates, _, _, ptyp = load_historical_matrices()
    rp = load_price_realtime()
    fc = build_price_forecast(rp, ptyp)
    print("电价预报精度（2/1~12/31）:", price_forecast_mae(rp, fc, dates))