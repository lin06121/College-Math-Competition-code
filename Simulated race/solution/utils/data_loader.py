# -*- coding: utf-8 -*-
"""
utils/data_loader.py — 通用数据加载
====================================
将附件 C题_数据附件.xlsx 的全部工作表读入一个 dict[str, pandas.DataFrame]，
统一 dtype/日期解析，供 problem1..4 复用。
"""

from __future__ import annotations

import pandas as pd

from config import INPUT_XLSX


# 需要解析为日期的列
_DATE_COLUMNS = {
    "historical_matches": ["date"],
    "time_slots": ["date"],
}


def load_all_sheets(path: str | None = None) -> dict[str, pd.DataFrame]:
    """读取 xlsx 所有工作表，返回 {sheet_name: DataFrame}。

    参数
    ----
    path : 输入 xlsx 路径，默认取 config.INPUT_XLSX。
    """
    path = str(path or INPUT_XLSX)

    # read_excel(sheet_name=None) 一次性读取全部表
    sheets: dict[str, pd.DataFrame] = pd.read_excel(
        path, sheet_name=None, engine="openpyxl"
    )

    # 统一解析日期列
    for sheet, cols in _DATE_COLUMNS.items():
        if sheet in sheets:
            for c in cols:
                if c in sheets[sheet].columns:
                    sheets[sheet][c] = pd.to_datetime(sheets[sheet][c])

    return sheets


def summarize(sheets: dict[str, pd.DataFrame]) -> None:
    """打印各表形状与前 2 行，便于快速核对数据格式（调试用）。"""
    for name, df in sheets.items():
        print(f"\n==== {name}: {df.shape} ====")
        print("列:", list(df.columns))
        print(df.head(2).to_string())


if __name__ == "__main__":
    data = load_all_sheets()
    summarize(data)