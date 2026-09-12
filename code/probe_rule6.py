# -*- coding: utf-8 -*-
"""复现同伴"规律六"预报误差表（写死 build_dataset.py L162-189 口径），并对照两套口径。
两个自由维度：
  窗口  : remaining(剩余当天)  vs  full24h(发布后24h,跨日)
  对齐  : +5(我方)             vs  +6(同伴按 slot=整点*6)
用同一份 附件2 实测、附件3 预报，2/1-12/31 与全年两套期间。
"""
import numpy as np, pandas as pd

BASE = "problem/C题/附件"
pv_slot = (pd.read_excel(f"{BASE}/附件2.xlsx", sheet_name="光伏发电实际功率")
             .set_index("日期\\时间"))       # 列=结束时刻
pv_slot.index = pd.to_datetime(pv_slot.index)
# 列标签是时段结束时刻：00:10=slot1=[0:00,0:10], 0:00+1=slot144
cols = list(pv_slot.columns)
slot_labels = {}
for i, c in enumerate(cols):
    t = c if isinstance(c, str) else str(c).split(" ")[0]
    if "+1" in t and t.startswith("0:00"):
        hh, mm = 24, 0
    else:
        hh, mm = int(t.split(":")[0]), int(t.split(":")[1])
    slot_labels[c] = (hh * 60 + mm) // 10          # 结束时刻 -> slot 号
pv = pv_slot.rename(columns=slot_labels)               # 列 -> slot 号
N = len(pv); H = 144
pv_arr = pv.to_numpy(dtype=float)                      # (date, slot)
dates = pv.index

a3 = pd.read_excel(f"{BASE}/附件3.xlsx", sheet_name="Sheet1")
a3["日期"] = a3["日期"].ffill()
a3["日期"] = pd.to_datetime(a3["日期"])
a3["预报时刻h"] = a3["预报时刻"].map({"0:00": 0, "6:00": 6, "12:00": 12, "18:00": 18})
irow = {r["日期"]: a3.index[a3["日期"] == r["日期"]].to_list() for r in []}
pmap = {}
for (d, hh), gr in a3.groupby(["日期", "预报时刻h"]):
    pmap[(d, hh)] = gr.iloc[0]

def date_pv(ad):
    """返回 pd.Timestamp 对应天在矩阵中的行号；跨年(2026)为 None"""
    try:
        return pv.index.get_loc(pd.Timestamp(ad))
    except KeyError:
        return None

def err_for(issue_date, h0, window):
    """附属力：单天单个发布时刻的所有 24 个预报误差(kW)，返回 (val[], target)，val=误差"""
    row = pmap[(pd.Timestamp(issue_date), h0)]
    errs = []
    for k in range(1, 25):
        # 绝对目标小时
        ad = issue_date + pd.Timedelta(hours=h0 + k)   # 目标整点
        ah = ad.hour
        if window == "full24h":
            keep = True                       # 发布后24h，全保留
        else:
            # 只保留"当天"剩余：ad 与 issue 同一天
            keep = ad.date() == issue_date.date()
        if not keep:
            continue
        di = date_pv(ad.date())
        if di is None:
            continue
        # 同伴对齐: 目标整点 ah -> slot ah*6（伙伴口径，列标签=结束时刻，ah*6 结束于 h:00）
        for (name, sl) in [("+6", ah * 6), ("+5", ah * 6 - 1)]:
            if 1 <= sl <= H:
                errs.append((name, float(row[f"预报{k}小时"]) - float(pv_arr[di, sl - 1])))
    return errs

from collections import defaultdict
for period in ["2.1-12.31", "全年"]:
    lo = dates >= pd.Timestamp("2025-02-01") if period == "2.1-12.31" else np.ones(len(dates), bool)
    print(f"\n===== 期间 {period} =====")
    for window in ["full24h", "remaining"]:
        print(f"--- 窗口 {window} ---")
        for align_ in ["+6", "+5"]:
            out = []
            for h0 in (0, 6, 12, 18):
                e = []
                for i in range(len(dates)):
                    if not lo[i]:
                        continue
                    e += [v for name, v in err_for(dates[i], h0, window) if name == align_]
                out.append((f"{h0}:00", np.abs(np.array(e)).mean()))
                print(f"   对齐{align_} {h0:>2}:00   MAE={np.abs(np.array(e)).mean():7.1f} kW   n={len(e)}")