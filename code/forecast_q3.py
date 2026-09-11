"""
问题 3：附件 3 光伏预报处理。

附件 3 每天 0:00/6:00/12:00/18:00 发布未来 24 小时整点光伏功率预报。
需要把整点预报插值到 10 分钟，并与当天 144 个时段对齐。

对齐规则：
  某时刻 s:00 发布的「预报 k 小时」是发布时刻之后**第 k 个整点 (s+k):00 的瞬时功率**，
  不是该小时的平均功率。在 10 分钟网格上，第 k 个整点对应时段 s*6+k*6−1
  （即结束于该整点的那个 10 分钟时段），相邻整点之间做**线性插值**。

  该结论由数据验证：对四个发布时刻分别扫描偏移，最佳偏移在三种情况下均为 +5
  （见 run_q3.py 的对齐回归测试）。若误用阶梯插值（把整点值摊到整个小时），
  在光伏中午前后每小时变化可达 800 kW 时会造出数百 kW 的假误差。

  超出当日 24:00 的预报属于次日，本模块置为 NaN（问题 3 只调整当日剩余时段）。
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    ATTACHMENT3, DATA_PROCESSED, SLOTS_PER_DAY,
    FORECAST_ISSUE_TIMES, FORECAST_ISSUE_SLOTS,
)


def load_attachment3(path: Path = ATTACHMENT3) -> pd.DataFrame:
    """读取附件 3，并把日期列 forward-fill 到每一行。"""
    df = pd.read_excel(path, sheet_name="Sheet1")
    df["日期"] = pd.to_datetime(df["日期"]).ffill()
    df["预报时刻"] = df["预报时刻"].astype(str)
    return df


def forecast_hours_to_slots(forecast_24h: np.ndarray, start_slot: int = 0,
                            prev_anchor: float = None) -> np.ndarray:
    """
    把 24 个**整点瞬时值**预报铺到当日 144 个 10 分钟时段上。

    预报 k 小时 = 发布时刻之后第 k 个整点 (start+k 小时) 的瞬时功率，对应时段
    start_slot + k*6 − 1（结束于该整点的 10 分钟时段）；相邻整点之间线性插值。

    发布时刻本身不是 24 个整点之一，首个整点在 +1 小时处，因此 [start, +1h) 这一段
    没有左端点。prev_anchor 传入**上一期预报的「预报 6 小时」**（即当前发布时刻那个
    整点的值），作为时段 start_slot−1 的锚点，从而首小时也能正常插值而不是取常数。
    prev_anchor 为 None 时退化为端点常数（首小时取预报 1 小时的值）。

    发布时刻之前的时段置 NaN。
    """
    assert len(forecast_24h) == 24
    xs = [start_slot + k * 6 - 1 for k in range(1, 25)]
    ys = [float(v) for v in forecast_24h]
    if prev_anchor is not None and np.isfinite(prev_anchor):
        xs = [start_slot - 1] + xs
        ys = [float(prev_anchor)] + ys
    grid = np.arange(SLOTS_PER_DAY, dtype=float)
    slots = np.clip(np.interp(grid, np.array(xs, float), np.array(ys, float)), 0.0, None)
    slots[:start_slot] = np.nan
    return slots


def build_forecast_matrix(path: Path = ATTACHMENT3) -> dict:
    """
    构建每天 4 个预报时刻的 144 时段光伏预报矩阵。

    返回 dict：
      dates: 日期列表
      issue_times: ['0:00','6:00','12:00','18:00']
      pv_forecast: ndarray (n_days, 4, 144)  单位 kW
    """
    df = load_attachment3(path)
    dates = pd.to_datetime(df["日期"].unique())
    issue_times = list(FORECAST_ISSUE_TIMES)
    n_days = len(dates)
    fc_cols = [f"预报{k}小时" for k in range(1, 25)]

    # 1) 原始 24 个整点值
    raw = np.full((n_days, len(issue_times), 24), np.nan)
    for d_idx, d in enumerate(dates):
        sub = df[df["日期"] == d]
        for it_idx, it in enumerate(issue_times):
            row = sub[sub["预报时刻"] == it]
            if row.empty:
                continue
            raw[d_idx, it_idx] = row.iloc[0][fc_cols].values.astype(float)

    # 2) 重建到 144 时段；用上一期「预报 6 小时」补当前发布时刻那个整点的锚点
    pv_forecast = np.full((n_days, len(issue_times), SLOTS_PER_DAY), np.nan)
    for d_idx in range(n_days):
        for it_idx, it in enumerate(issue_times):
            vals = raw[d_idx, it_idx]
            if np.isnan(vals).all():
                continue
            if it_idx > 0:
                anchor = raw[d_idx, it_idx - 1][5]
            elif d_idx > 0:
                anchor = raw[d_idx - 1, -1][5]      # 前一天 18:00 期的“预报 6 小时”
            else:
                anchor = None
            pv_forecast[d_idx, it_idx] = forecast_hours_to_slots(
                vals, start_slot=FORECAST_ISSUE_SLOTS[it], prev_anchor=anchor)

    return {
        "dates": dates,
        "issue_times": issue_times,
        "pv_forecast": pv_forecast,
        "raw_24h": raw,
    }


def debias_forecast_matrix(fc: dict, pv_actual: np.ndarray, dates,
                           d2i: dict, window: int = 20) -> np.ndarray:
    """
    用历史预报误差对附件 3 预报做**因果**去偏（bias correction）。

    对第 k 天、发布时刻 a、时段 t：
        bias(k,a,t) = mean( fc(j,a,t) − pv_actual(j,t) ) ,  j ∈ [k−window, k−1]
        去偏预报     = fc(k,a,t) − bias(k,a,t)

    只使用目标日之前已实现的预报误差（微网自有光伏出力与收到的预报都是可观测的），
    不使用任何未来信息。返回与 fc['pv_forecast'] 同形状的数组。
    """
    dates = pd.to_datetime(dates)
    fdates = pd.to_datetime(fc["dates"])
    assert len(fdates) == len(dates) and (fdates == dates).all(), \
        "预报日期与历史日期未对齐"

    n_days, n_it, n_slot = fc["pv_forecast"].shape
    err = np.full((n_days, n_it, n_slot), np.nan)
    for k in range(n_days):
        err[k] = fc["pv_forecast"][k] - pv_actual[k]

    out = fc["pv_forecast"].astype(float).copy()
    for k in range(1, n_days):
        past = err[max(0, k - window):k]
        if past.size == 0 or np.isnan(past).all():
            continue                       # 历史不足，不去偏
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(past, axis=0)   # 某些 (发布时刻, 时段) 可能整列无样本
        m = np.where(np.isnan(m), 0.0, m)
        out[k] = fc["pv_forecast"][k] - m
    return out


def get_forecast_for_date(date, issue_time, fc_dict):
    """获取某一天某个预报时刻的 144 时段光伏预报。"""
    d_idx = np.where(fc_dict["dates"] == pd.Timestamp(date))[0]
    if len(d_idx) == 0:
        raise ValueError(f"日期 {date} 不在附件 3 中")
    it_idx = fc_dict["issue_times"].index(issue_time)
    return fc_dict["pv_forecast"][d_idx[0], it_idx]


def main():
    fc = build_forecast_matrix()
    print(f"预报矩阵形状: {fc['pv_forecast'].shape}")
    print(f"日期范围: {fc['dates'].min().date()} ~ {fc['dates'].max().date()}")

    # 保存处理后的预报
    records = []
    for d_idx, d in enumerate(fc["dates"]):
        for it_idx, it in enumerate(fc["issue_times"]):
            for slot in range(SLOTS_PER_DAY):
                records.append({
                    "date": d,
                    "issue_time": it,
                    "slot": slot,
                    "pv_forecast_kW": fc["pv_forecast"][d_idx, it_idx, slot],
                })
    df = pd.DataFrame(records)
    out = DATA_PROCESSED / "forecast_q3_interpolated.csv"
    df.to_csv(out, index=False)
    print(f"已保存: {out}, 共 {len(df)} 行")


if __name__ == "__main__":
    main()
