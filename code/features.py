"""
特征工程：日/月/季/全年统计、峰谷分析、周期性、预测误差等
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_PROCESSED, SLOTS_PER_DAY, HOURS_PER_SLOT, STORAGE_MAX_POWER,
    get_season, SEASON_ORDER, SEASON_LABELS,
)


def load_full_timeseries(path: Path = DATA_PROCESSED / "full_timeseries.parquet") -> pd.DataFrame:
    return pd.read_parquet(path)


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算每日特征"""
    daily2 = []
    for date, g in df.groupby("date"):
        load_peak_slot = int(g["slot"].iloc[g["load_kW"].values.argmax()])
        load_valley_slot = int(g["slot"].iloc[g["load_kW"].values.argmin()])
        pv_peak_slot = int(g["slot"].iloc[g["pv_actual_kW"].values.argmax()])
        price_peak_slot = int(g["slot"].iloc[g["price_realtime"].values.argmax()])
        price_valley_slot = int(g["slot"].iloc[g["price_realtime"].values.argmin()])

        # 有效发电时长：用日最大值的 5% 作为阈值
        pv_max = g["pv_actual_kW"].max()
        threshold = max(pv_max * 0.05, 10)
        gen_slots = (g["pv_actual_kW"] > threshold).sum()
        sunrise_slot = g.loc[g["pv_actual_kW"] > threshold, "slot"].min() if gen_slots > 0 else np.nan
        sunset_slot = g.loc[g["pv_actual_kW"] > threshold, "slot"].max() if gen_slots > 0 else np.nan

        # 预测误差
        valid = g[g["pv_actual_kW"] > 0]
        if len(valid) > 0:
            mae = (valid["pv_forecast_op_kW"] - valid["pv_actual_kW"]).abs().mean()
            rmse = np.sqrt(((valid["pv_forecast_op_kW"] - valid["pv_actual_kW"]) ** 2).mean())
            # sMAPE 避免 actual 接近 0 时爆炸
            smape = (2 * (valid["pv_forecast_op_kW"] - valid["pv_actual_kW"]).abs()
                     / (valid["pv_forecast_op_kW"].abs() + valid["pv_actual_kW"].abs())).mean() * 100
        else:
            mae = rmse = mape = np.nan

        # 典型日（附件 1）电价下的低价/高价时段套利空间
        low_slots = g["price_typical"] < g["price_typical"].quantile(0.25)
        high_slots = g["price_typical"] > g["price_typical"].quantile(0.75)
        theoretical_arbitrage_typical = (g.loc[high_slots, "price_typical"].mean() - g.loc[low_slots, "price_typical"].mean()) * STORAGE_MAX_POWER * HOURS_PER_SLOT if high_slots.any() and low_slots.any() else 0

        # 净负载缺电量（不考虑储能）
        shortage = g.loc[g["net_load_actual_kW"] > 0, "net_load_actual_kW"].sum() * HOURS_PER_SLOT
        # 弃光量（光伏超过负载+最大充电功率）
        curtailment = (g["pv_actual_kW"] - g["load_kW"] - STORAGE_MAX_POWER).clip(lower=0).sum() * HOURS_PER_SLOT

        daily2.append({
            "date": date,
            "month": g["month"].iloc[0],
            "dayofweek": g["dayofweek"].iloc[0],
            "is_weekend": g["is_weekend"].iloc[0],
            "season": g["season"].iloc[0],
            "load_sum_kWh": g["load_kW"].sum() * HOURS_PER_SLOT,
            "load_mean": g["load_kW"].mean(),
            "load_max": g["load_kW"].max(),
            "load_min": g["load_kW"].min(),
            "load_std": g["load_kW"].std(),
            "load_peak_slot": load_peak_slot,
            "load_valley_slot": load_valley_slot,
            "load_peak_valley_diff": g["load_kW"].max() - g["load_kW"].min(),
            "pv_sum_kWh": g["pv_actual_kW"].sum() * HOURS_PER_SLOT,
            "pv_mean": g["pv_actual_kW"].mean(),
            "pv_max": pv_max,
            "pv_peak_slot": pv_peak_slot,
            "pv_gen_slots": gen_slots,
            "pv_sunrise_slot": sunrise_slot,
            "pv_sunset_slot": sunset_slot,
            "price_realtime_mean": g["price_realtime"].mean(),
            "price_realtime_max": g["price_realtime"].max(),
            "price_realtime_min": g["price_realtime"].min(),
            "price_realtime_std": g["price_realtime"].std(),
            "price_peak_slot": price_peak_slot,
            "price_valley_slot": price_valley_slot,
            "price_peak_valley_diff": g["price_realtime"].max() - g["price_realtime"].min(),
            "net_load_sum_kWh": g["net_load_actual_kW"].sum() * HOURS_PER_SLOT,
            "net_load_max": g["net_load_actual_kW"].max(),
            "net_load_min": g["net_load_actual_kW"].min(),
            "net_load_positive_slots": (g["net_load_actual_kW"] > 0).sum(),
            "forecast_mae": mae,
            "forecast_rmse": rmse,
            "forecast_smape": smape,
            "theoretical_arbitrage_typical": theoretical_arbitrage_typical,
            "shortage_kWh": shortage,
            "curtailment_kWh": curtailment,
        })

    daily = pd.DataFrame(daily2)
    return daily


def compute_monthly_features(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = daily.groupby("month").agg(
        days=("date", "count"),
        load_sum_mean=("load_sum_kWh", "mean"),
        load_sum_std=("load_sum_kWh", "std"),
        pv_sum_mean=("pv_sum_kWh", "mean"),
        pv_sum_std=("pv_sum_kWh", "std"),
        price_mean=("price_realtime_mean", "mean"),
        price_std=("price_realtime_std", "mean"),
        forecast_mae_mean=("forecast_mae", "mean"),
        shortage_mean=("shortage_kWh", "mean"),
        curtailment_mean=("curtailment_kWh", "mean"),
    ).reset_index()
    return monthly


def compute_seasonal_features(daily: pd.DataFrame) -> pd.DataFrame:
    seasonal = daily.groupby("season").agg(
        days=("date", "count"),
        load_sum_mean=("load_sum_kWh", "mean"),
        pv_sum_mean=("pv_sum_kWh", "mean"),
        price_mean=("price_realtime_mean", "mean"),
        forecast_mae_mean=("forecast_mae", "mean"),
        shortage_mean=("shortage_kWh", "mean"),
        curtailment_mean=("curtailment_kWh", "mean"),
    ).reset_index()
    return seasonal


def compute_forecast_errors(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个 (date, slot) 的预测误差，并附 lead_time 信息"""
    err = df[["date", "slot", "month", "season", "dayofweek", "is_weekend",
              "lead_time_hours", "pv_actual_kW", "pv_forecast_op_kW"]].copy()
    err["error"] = err["pv_forecast_op_kW"] - err["pv_actual_kW"]
    err["abs_error"] = err["error"].abs()
    err["pct_error"] = err["abs_error"] / err["pv_actual_kW"].replace(0, np.nan)
    return err


def compute_slotwise_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """计算日内 144 slot 的平均曲线"""
    profiles = df.groupby("slot").agg(
        load_mean=("load_kW", "mean"),
        load_std=("load_kW", "std"),
        pv_mean=("pv_actual_kW", "mean"),
        pv_std=("pv_actual_kW", "std"),
        price_realtime_mean=("price_realtime", "mean"),
        price_realtime_std=("price_realtime", "std"),
        price_typical_mean=("price_typical", "mean"),
        net_load_mean=("net_load_actual_kW", "mean"),
    ).reset_index()
    return profiles


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    print("加载主表 ...")
    df = load_full_timeseries()

    print("计算日特征 ...")
    daily = compute_daily_features(df)
    daily.to_csv(DATA_PROCESSED / "daily_features.csv", index=False)

    print("计算月特征 ...")
    monthly = compute_monthly_features(daily)
    monthly.to_csv(DATA_PROCESSED / "monthly_features.csv", index=False)

    print("计算季节特征 ...")
    seasonal = compute_seasonal_features(daily)
    seasonal.to_csv(DATA_PROCESSED / "seasonal_features.csv", index=False)

    print("计算预测误差 ...")
    errors = compute_forecast_errors(df)
    errors.to_csv(DATA_PROCESSED / "forecast_errors.csv", index=False)

    print("计算日内平均曲线 ...")
    profiles = compute_slotwise_profiles(df)
    profiles.to_csv(DATA_PROCESSED / "slotwise_profiles.csv", index=False)

    print("特征文件已保存到:", DATA_PROCESSED)


if __name__ == "__main__":
    main()
