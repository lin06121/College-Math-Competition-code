"""
读取 4 个附件的底层 IO 工具
"""
import pandas as pd
from pathlib import Path
from typing import Dict

from config import ATTACHMENT1, ATTACHMENT2, ATTACHMENT3, ATTACHMENT4, SLOTS_PER_DAY


def _slot_label_to_minute(label) -> int:
    """把列标签（datetime.time 或 '0:00+1' 字符串）转为 slot 编号 0..143。"""
    if isinstance(label, str):
        if label == "0:00+1":
            return SLOTS_PER_DAY - 1
        # 处理像 '0:10-0:20' 这种表头
        parts = label.split(":")
        if len(parts) >= 2:
            h = int(parts[0])
            m = int(parts[1].split("-")[0])
            # 结束时刻 -> slot = (h*60 + m) / 10 - 1
            slot = (h * 60 + m) // 10 - 1
            return slot
        raise ValueError(f"无法解析时段标签: {label}")
    # datetime.time 对象
    import datetime
    if isinstance(label, datetime.time):
        slot = (label.hour * 60 + label.minute) // 10 - 1
        return slot
    raise ValueError(f"无法解析时段标签类型: {type(label)} -> {label}")


def read_attachment1(path: Path = ATTACHMENT1) -> pd.DataFrame:
    """读取典型日数据，返回 144 行，列：slot, time_str, price, load, pv_forecast"""
    df = pd.read_excel(path, sheet_name="Sheet1", header=0)
    # 列名中文：时间、电价、小区负载、光伏发电预测功率
    df = df.rename(columns={
        df.columns[0]: "time_str",
        df.columns[1]: "price",
        df.columns[2]: "load",
        df.columns[3]: "pv_forecast",
    })
    # 解析 slot
    slots = []
    for v in df["time_str"]:
        if isinstance(v, str) and v == "0:00+1":
            slots.append(SLOTS_PER_DAY - 1)
        else:
            # 时间字符串格式可能为 "00:10:00" 或 "10:10"，使用 mixed 解析
            t = pd.to_datetime(str(v), format="mixed").time()
            slot = (t.hour * 60 + t.minute) // 10 - 1
            slots.append(slot)
    df["slot"] = slots
    df = df.sort_values("slot").reset_index(drop=True)
    assert len(df) == SLOTS_PER_DAY, f"附件1行数异常: {len(df)}"
    assert df["slot"].nunique() == SLOTS_PER_DAY, "附件1 slot 有重复"
    return df[["slot", "time_str", "price", "load", "pv_forecast"]]


def _wide_sheet_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """把附件 2/4 的宽表（日期 × 144 时段）转为长表"""
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    # 日期列可能含时间部分，统一只保留日期
    df["date"] = pd.to_datetime(df["date"]).dt.date
    time_cols = df.columns[1:].tolist()
    # melt
    df_long = df.melt(id_vars=["date"], value_vars=time_cols,
                      var_name="time_label", value_name=value_name)
    # 解析 slot
    df_long["slot"] = df_long["time_label"].apply(_slot_label_to_minute)
    df_long["date"] = pd.to_datetime(df_long["date"])
    df_long = df_long.sort_values(["date", "slot"]).reset_index(drop=True)
    return df_long[["date", "slot", value_name]]


def read_attachment2(path: Path = ATTACHMENT2) -> Dict[str, pd.DataFrame]:
    """读取全年负载与光伏实际功率，返回 dict: {'load': df, 'pv': df}"""
    xl = pd.ExcelFile(path)
    result = {}
    # 表名可能是中文
    sheet_load = "小区负载" if "小区负载" in xl.sheet_names else xl.sheet_names[0]
    sheet_pv = "光伏发电实际功率" if "光伏发电实际功率" in xl.sheet_names else xl.sheet_names[1]

    df_load = pd.read_excel(path, sheet_name=sheet_load, header=0)
    df_pv = pd.read_excel(path, sheet_name=sheet_pv, header=0)

    result["load"] = _wide_sheet_to_long(df_load, "load_kW")
    result["pv"] = _wide_sheet_to_long(df_pv, "pv_actual_kW")

    # 校验
    for name, df in result.items():
        assert len(df) == 365 * SLOTS_PER_DAY, f"附件2 {name} 长表行数异常: {len(df)}"
    return result


def read_attachment4(path: Path = ATTACHMENT4) -> pd.DataFrame:
    """读取全年实时电价"""
    df = pd.read_excel(path, sheet_name="Sheet1", header=0)
    df_long = _wide_sheet_to_long(df, "price_realtime")
    assert len(df_long) == 365 * SLOTS_PER_DAY, f"附件4 长表行数异常: {len(df_long)}"
    return df_long


def read_attachment3(path: Path = ATTACHMENT3) -> pd.DataFrame:
    """
    读取全年光伏预报。
    返回长表，列：issue_date, issue_time, lead_hour, target_datetime, forecast_kW
    其中 lead_hour = 1..24。
    """
    df = pd.read_excel(path, sheet_name="Sheet1", header=0)
    # 列名：日期、预报时刻、预报1小时、...、预报24小时
    df = df.rename(columns={df.columns[0]: "issue_date", df.columns[1]: "issue_time"})
    df["issue_date"] = pd.to_datetime(df["issue_date"]).ffill()
    df["issue_date"] = df["issue_date"].dt.date

    # 解析 issue_time 为 time 对象
    def _parse_issue_time(x):
        if isinstance(x, str):
            return pd.to_datetime(x, format="%H:%M").time()
        return x
    df["issue_time"] = df["issue_time"].apply(_parse_issue_time)

    records = []
    for _, row in df.iterrows():
        issue_date = pd.to_datetime(row["issue_date"])
        issue_time = row["issue_time"]
        issue_dt = pd.Timestamp.combine(issue_date, issue_time)
        for lead in range(1, 25):
            col = f"预报{lead}小时"
            target_dt = issue_dt + pd.Timedelta(hours=lead)
            records.append({
                "issue_date": issue_date.date(),
                "issue_time": issue_time,
                "lead_hour": lead,
                "target_datetime": target_dt,
                "forecast_kW": row[col],
            })
    return pd.DataFrame(records)
