"""
问题 2 预测模块：每天 0:00 只能使用附件 2 中当天之前的历史数据，
采用“过去 K 个同星期几的曲线平均”预测当日负载与光伏。
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_PROCESSED, SLOTS_PER_DAY


def load_historical_matrices(path: Path = DATA_PROCESSED / "matrices.npz"):
    """读取全年负载、光伏、电价矩阵。"""
    data = np.load(path, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    load = data["load_kW"]
    pv = data["pv_actual_kW"]
    price = data["price_typical"]
    return dates, load, pv, price


def forecast_load_pv(target_date, dates, hist_load, hist_pv,
                     k: int = 4, min_samples: int = 1):
    """
    对目标日期，使用历史同星期几的平均曲线预测负载与光伏。

    参数
    ----------
    target_date : pd.Timestamp or str
        需要预测的日期。
    dates : pd.DatetimeIndex
        历史日期索引，长度 N。
    hist_load, hist_pv : ndarray, shape (N, SLOTS_PER_DAY)
        历史负载与光伏功率矩阵。
    k : int
        最多使用过去 K 个同星期几。
    min_samples : int
        最少需要几个同星期几样本；不足时会抛出错误（历史不应不足）。

    返回
    ----------
    load_f, pv_f : ndarray, shape (SLOTS_PER_DAY,)
    """
    target_date = pd.Timestamp(target_date)
    target_dow = target_date.dayofweek

    # 只取目标日期之前的历史
    mask_before = dates < target_date
    candidate_dates = dates[mask_before]
    candidate_load = hist_load[mask_before]
    candidate_pv = hist_pv[mask_before]

    # 同星期几
    mask_dow = candidate_dates.dayofweek == target_dow
    same_dow_load = candidate_load[mask_dow]
    same_dow_pv = candidate_pv[mask_dow]

    if len(same_dow_load) < min_samples:
        raise ValueError(
            f"{target_date.date()} 历史同星期几样本不足: "
            f"仅有 {len(same_dow_load)} 个，需要至少 {min_samples} 个"
        )

    # 取最近 K 个
    n_use = min(k, len(same_dow_load))
    load_f = same_dow_load[-n_use:].mean(axis=0)
    pv_f = same_dow_pv[-n_use:].mean(axis=0)

    # 光伏非负
    pv_f = np.maximum(pv_f, 0.0)

    return load_f, pv_f


def build_forecast_series(dates, hist_load, hist_pv,
                          start_date="2025-02-01",
                          end_date="2025-12-31",
                          k: int = 4):
    """
    为指定日期范围生成逐日预测，返回 DataFrame。

    返回 DataFrame 列：
    date, slot, load_forecast_kW, pv_forecast_kW
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    target_dates = pd.date_range(start, end, freq="D")

    records = []
    for td in target_dates:
        load_f, pv_f = forecast_load_pv(td, dates, hist_load, hist_pv, k=k)
        for slot in range(SLOTS_PER_DAY):
            records.append({
                "date": td,
                "slot": slot,
                "load_forecast_kW": load_f[slot],
                "pv_forecast_kW": pv_f[slot],
            })

    return pd.DataFrame(records)


def compute_forecast_errors(forecast_df: pd.DataFrame,
                            actual_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算预测误差（与实际值对比）。
    actual_df 需包含 date, slot, load_kW, pv_actual_kW。
    若未提供，则从 full_timeseries.parquet 读取。
    """
    if actual_df is None:
        actual_df = pd.read_parquet(DATA_PROCESSED / "full_timeseries.parquet")

    merged = forecast_df.merge(
        actual_df[["date", "slot", "load_kW", "pv_actual_kW"]],
        on=["date", "slot"],
        how="inner",
    )

    merged["load_error"] = merged["load_forecast_kW"] - merged["load_kW"]
    merged["pv_error"] = merged["pv_forecast_kW"] - merged["pv_actual_kW"]
    merged["load_abs_error"] = merged["load_error"].abs()
    merged["pv_abs_error"] = merged["pv_error"].abs()

    return merged


def main():
    dates, load, pv, price = load_historical_matrices()
    print(f"历史数据日期范围: {dates.min().date()} ~ {dates.max().date()}")

    forecast_df = build_forecast_series(dates, load, pv, k=4)
    out_file = DATA_PROCESSED / "forecast_q2.csv"
    forecast_df.to_csv(out_file, index=False)
    print(f"预测结果已保存: {out_file}, 共 {len(forecast_df)} 行")

    # 计算误差
    errors = compute_forecast_errors(forecast_df)
    load_mae = errors["load_abs_error"].mean()
    pv_mae = errors["pv_abs_error"].mean()
    print(f"负载预测 MAE: {load_mae:.2f} kW")
    print(f"光伏预测 MAE: {pv_mae:.2f} kW")


if __name__ == "__main__":
    main()
