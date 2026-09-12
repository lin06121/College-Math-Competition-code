"""
数据清洗、对齐与预报插值
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

from config import (
    DATA_PROCESSED, SLOTS_PER_DAY, HOURS_PER_SLOT,
    get_season, SEASON_ORDER, SEASON_LABELS,
)
from io_utils import read_attachment1, read_attachment2, read_attachment3, read_attachment4


def _hour_to_slot(hour: int, minute: int = 0) -> int:
    """把结束时刻 (hour, minute) 映射到 slot。hour=0, minute=0 表示次日 0:00，对应 slot 143。"""
    total_minutes = hour * 60 + minute
    slot = total_minutes // 10 - 1
    if slot < 0:
        # 只有 hour=0,minute=0 会到这里，表示 0:00+1
        slot = SLOTS_PER_DAY - 1
    return slot


def _slot_to_start_datetime(date: pd.Timestamp, slot: int) -> pd.Timestamp:
    """slot 起始时刻"""
    minutes = slot * 10
    return date + pd.Timedelta(minutes=minutes)


def interpolate_forecast(forecast_raw: pd.DataFrame) -> pd.DataFrame:
    """
    把附件 3 的整点预报线性插值到 10 分钟 slot。
    返回长表：issue_date, issue_time, issue_datetime, target_date, target_slot, lead_time_hours, forecast_kW
    """
    records = []
    grouped = forecast_raw.sort_values(["issue_date", "issue_time", "lead_hour"])
    for (issue_date, issue_time), g in grouped.groupby(["issue_date", "issue_time"]):
        g = g.sort_values("lead_hour").reset_index(drop=True)
        issue_dt = pd.Timestamp.combine(pd.to_datetime(issue_date), issue_time)

        # 24 个整点锚点
        anchors = []
        for _, row in g.iterrows():
            target_dt = issue_dt + pd.Timedelta(hours=row["lead_hour"])
            # 按附件 2/4 的列约定，整点 H:00 是区间 [H-1:00, H:00) 的结束；
            # 特别地，24:00（次日 0:00）属于前一天最后一个 slot（143）。
            if target_dt.hour == 0 and target_dt.minute == 0:
                target_date = (target_dt - timedelta(days=1)).date()
                slot = SLOTS_PER_DAY - 1
            else:
                target_date = target_dt.date()
                slot = _hour_to_slot(target_dt.hour, target_dt.minute)
            anchors.append({
                "target_dt": target_dt,
                "target_date": target_date,
                "target_slot": slot,
                "forecast": row["forecast_kW"],
                "lead_hour": row["lead_hour"],
            })

        # 在每对相邻锚点之间线性插值出 6 个 slot
        for i in range(len(anchors)):
            a = anchors[i]
            # 当前锚点本身
            lead_hours = (a["target_dt"] - issue_dt).total_seconds() / 3600.0
            records.append({
                "issue_date": issue_date,
                "issue_time": issue_time,
                "issue_datetime": issue_dt,
                "target_date": a["target_date"],
                "target_slot": a["target_slot"],
                "lead_time_hours": lead_hours,
                "forecast_kW": a["forecast"],
            })
            # 如果存在下一个锚点，插值中间 5 个 slot
            if i + 1 < len(anchors):
                b = anchors[i + 1]
                for k in range(1, 6):
                    interp_slot = a["target_slot"] + k
                    interp_date = a["target_date"]
                    if interp_slot >= SLOTS_PER_DAY:
                        # 跨日：0:00 锚点(slot 143) 之后的 5 个 slot 属于次日
                        interp_slot -= SLOTS_PER_DAY
                        interp_date = a["target_date"] + timedelta(days=1)
                    frac = k / 6.0
                    val = a["forecast"] + (b["forecast"] - a["forecast"]) * frac
                    target_dt = a["target_dt"] + pd.Timedelta(minutes=k * 10)
                    lead_hours = (target_dt - issue_dt).total_seconds() / 3600.0
                    records.append({
                        "issue_date": issue_date,
                        "issue_time": issue_time,
                        "issue_datetime": issue_dt,
                        "target_date": interp_date,
                        "target_slot": interp_slot,
                        "lead_time_hours": lead_hours,
                        "forecast_kW": val,
                    })

    df = pd.DataFrame(records)
    df["forecast_kW"] = df["forecast_kW"].clip(lower=0)
    return df


def build_operational_forecast(forecast_interp: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 (target_date, target_slot)，选择 issue_datetime 最新且不超过该 slot 起始时刻的起报。
    """
    # 计算每个目标 slot 的起始时刻
    forecast_interp = forecast_interp.copy()
    forecast_interp["target_date_dt"] = pd.to_datetime(forecast_interp["target_date"])
    forecast_interp["target_start"] = forecast_interp.apply(
        lambda r: _slot_to_start_datetime(r["target_date_dt"], r["target_slot"]), axis=1
    )

    # 只保留 issue_datetime <= target_start 的记录
    valid = forecast_interp[forecast_interp["issue_datetime"] <= forecast_interp["target_start"]].copy()

    # 按 (target_date, target_slot) 分组，取 issue_datetime 最大的
    idx = valid.groupby(["target_date", "target_slot"])["issue_datetime"].idxmax()
    op = valid.loc[idx].copy()
    op = op.sort_values(["target_date", "target_slot"]).reset_index(drop=True)
    op = op.rename(columns={
        "target_date": "date",
        "forecast_kW": "pv_forecast_op_kW",
    })
    op["date"] = pd.to_datetime(op["date"])
    # 附件 3 末尾的 2025-12-31 18:00 起报会覆盖到 2026-01-01，属于目标年之外，导出时剔除
    op = op[op["date"].dt.year == 2025].reset_index(drop=True)
    return op[["date", "target_slot", "issue_date", "issue_time", "lead_time_hours", "pv_forecast_op_kW"]]


def build_full_timeseries(att1, att2, att3_raw, att4) -> pd.DataFrame:
    """合并所有数据为统一长表"""
    # 附件 1：典型日曲线，按 slot 映射
    df_att1 = att1[["slot", "price", "load", "pv_forecast"]].rename(columns={
        "price": "price_typical",
        "load": "load_typical",
        "pv_forecast": "pv_forecast_typical",
    })

    # 附件 2
    df_load = att2["load"]
    df_pv = att2["pv"]
    df_att2 = df_load.merge(df_pv, on=["date", "slot"], how="outer")

    # 附件 4
    df_att4 = att4.rename(columns={"price_realtime": "price_realtime"})

    # 附件 3：插值 + operational
    forecast_interp = interpolate_forecast(att3_raw)
    op_forecast = build_operational_forecast(forecast_interp)
    op_forecast = op_forecast.rename(columns={"target_slot": "slot"})

    # 合并
    df = df_att2.merge(df_att4, on=["date", "slot"], how="outer")
    df = df.merge(df_att1, on="slot", how="left")
    df = df.merge(op_forecast, on=["date", "slot"], how="left")

    # 附件 3 从 2025-01-01 0:00 起报，2025-01-01 凌晨 00:00–00:50 缺少
    # 2024-12-31 18:00 的预报（数据边界）。该时段光伏为 0，用 0 填充；
    # 问题 2/3 的建模时段从 2025-02-01 开始，不受影响。
    n_missing = int(df["pv_forecast_op_kW"].isna().sum())
    if n_missing:
        missing_dates = df.loc[df["pv_forecast_op_kW"].isna(), "date"].unique()
        print(f"  提示: operational forecast 缺失 {n_missing} 个值，"
              f"日期范围 {pd.Timestamp(missing_dates.min()).date()} ~ {pd.Timestamp(missing_dates.max()).date()}，按 0 填充（仅数据边界凌晨时段）")
        df["pv_forecast_op_kW"] = df["pv_forecast_op_kW"].fillna(0.0)

    # 派生列
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek  # 0=周一
    df["is_weekend"] = df["dayofweek"].isin([5, 6])
    df["season"] = df["month"].map(get_season)
    df["season"] = pd.Categorical(df["season"], categories=SEASON_ORDER, ordered=True)

    # 净负载
    df["net_load_actual_kW"] = df["load_kW"] - df["pv_actual_kW"]
    df["net_load_forecast_kW"] = df["load_kW"] - df["pv_forecast_op_kW"]

    # 时段标签
    time_strs = []
    for s in df["slot"]:
        end_min = (s + 1) * 10
        h, m = divmod(end_min, 60)
        if h >= 24:
            label = "0:00+1"
        else:
            label = f"{h:02d}:{m:02d}"
        time_strs.append(label)
    df["time_str"] = time_strs

    return df


def save_matrices(df: pd.DataFrame, out_dir: Path = DATA_PROCESSED):
    """把关键变量 reshape 为 365×144 矩阵保存为 NPZ"""
    dates = sorted(df["date"].unique())
    assert len(dates) == 365, f"日期数不是 365: {len(dates)}"
    mats = {}
    for col in ["load_kW", "pv_actual_kW", "pv_forecast_op_kW", "price_realtime", "price_typical", "net_load_actual_kW"]:
        mat = df.pivot(index="date", columns="slot", values=col).values
        assert mat.shape == (365, SLOTS_PER_DAY), f"{col} 矩阵形状异常: {mat.shape}"
        mats[col] = mat
    np.savez(out_dir / "matrices.npz", **mats, dates=np.array(dates))


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    print("读取附件 1 ...")
    att1 = read_attachment1()
    print("读取附件 2 ...")
    att2 = read_attachment2()
    print("读取附件 4 ...")
    att4 = read_attachment4()
    print("读取附件 3 ...")
    att3_raw = read_attachment3()

    print("生成插值预报 ...")
    forecast_interp = interpolate_forecast(att3_raw)
    print("生成 operational forecast ...")
    op_forecast = build_operational_forecast(forecast_interp)

    print("合并主表 ...")
    df = build_full_timeseries(att1, att2, att3_raw, att4)

    # 校验
    assert len(df) == 365 * SLOTS_PER_DAY, f"主表行数异常: {len(df)}"
    assert df.groupby(["date", "slot"]).size().max() == 1, "主表 (date, slot) 存在重复"

    # 导出
    att1.to_csv(DATA_PROCESSED / "attachment1.csv", index=False)
    att2["load"].to_csv(DATA_PROCESSED / "load_pv_actual_load.csv", index=False)
    att2["pv"].to_csv(DATA_PROCESSED / "load_pv_actual_pv.csv", index=False)
    att4.to_csv(DATA_PROCESSED / "price_realtime.csv", index=False)
    forecast_interp.to_csv(DATA_PROCESSED / "forecast_interpolated.csv", index=False)
    op_forecast.to_csv(DATA_PROCESSED / "forecast_operational.csv", index=False)

    df.to_csv(DATA_PROCESSED / "full_timeseries.csv", index=False)
    df.to_parquet(DATA_PROCESSED / "full_timeseries.parquet", index=False)

    save_matrices(df)

    print("预处理完成，输出到:", DATA_PROCESSED)
    print("主表形状:", df.shape)
    print("主表列:", df.columns.tolist())


if __name__ == "__main__":
    main()
