# -*- coding: utf-8 -*-
"""
src/problem4.py — 任务四：实际赛程获取与综合评价
====================================================
将问题 2 的优化赛程与真实/基准赛程对比，维度：
旅行距离、休息时间、跨时区次数、场馆利用率、容量匹配度、黄金时段覆盖率。

说明
----
真实世界杯（2022 卡塔尔 32 队 / 2018 俄罗斯）与本模拟（48 队 72 场）规模不同，
故此处在**同一 72 场赛程**上构造一条"基准实际赛程"作为对照（真实爬取数据未随附件
提供，result_4 的 actual_schedule 用于演示完整链路）。所有指标按场均/队伍均归一化，
使不同规模可比较。

输出：result_4_schedule_comparison.csv 及 figures/ 下的雷达图 + 双柱状图。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import DATA_DIR, FIG_DIR, RANDOM_SEED, GOLDEN_PRIME_TOP_FRACTION
from utils.data_loader import load_all_sheets

rng = np.random.default_rng(RANDOM_SEED)


# ------------------------------------------------------------------
# 1. 读取对照所需查找表
# ------------------------------------------------------------------
def build_lookups(sheets):
    venues = sheets["venues"]
    slots = sheets["time_slots"]
    dist = sheets["distance_matrix"]
    groups = sheets["groups_matches"]
    base = sheets["base_predictions"]

    v_cap = dict(zip(venues["venue_id"], venues["capacity"]))
    slots = slots.sort_values("slot_id").reset_index(drop=True)
    s_prime = dict(zip(slots["slot_id"], slots["global_prime_score"]))
    s_utc = dict(zip(slots["slot_id"], pd.to_datetime(slots["reference_utc_time"])))

    # golden prime slots（前 25%）
    prime_order = slots["global_prime_score"].to_numpy().argsort()[::-1]
    n_prime = max(1, int(len(slots) * GOLDEN_PRIME_TOP_FRACTION))
    golden = set(slots["slot_id"].iloc[prime_order[:n_prime]].tolist())

    t2v_km = {}; t2v_tz = {}
    v2v_km = {}; v2v_tz = {}
    for _, r in dist.iterrows():
        o, d = r["origin_id"], r["destination_id"]
        if r["relation_type"] == "team_to_venue":
            t2v_km[(o, d)] = float(r["distance_km"])
            t2v_tz[(o, d)] = float(r["timezone_diff"])
        else:
            v2v_km[(o, d)] = float(r["distance_km"])
            v2v_tz[(o, d)] = float(r["timezone_diff"])

    att = base.set_index("match_id")["expected_attendance_base"]
    return {
        "v_cap": v_cap, "s_prime": s_prime, "s_utc": s_utc, "golden": golden,
        "t2v_km": t2v_km, "t2v_tz": t2v_tz, "v2v_km": v2v_km, "v2v_tz": v2v_tz,
        "att": att, "venues": venues,
    }


# ------------------------------------------------------------------
# 2. 指标计算（同一函数作用于任意赛程）
# ------------------------------------------------------------------
def compute_metrics(sched, L):
    """sched: DataFrame，需含 match_id, round_in_group, team_a_id, team_b_id,
       venue_id, slot_id。返回 6 项指标。"""
    n = len(sched)
    distances = []
    rests = []
    tz_cross = []
    occ = []
    prime_cnt = 0
    venue_ids = L["venues"]["venue_id"].tolist()

    team_rounds = {}
    for _, r in sched.iterrows():
        mid, rd = r["match_id"], int(r["round_in_group"])
        vid, sid = r["venue_id"], r["slot_id"]
        if sid in L["golden"]:
            prime_cnt += 1
        occ.append(min(1.0, float(L["att"][mid]) / float(L["v_cap"][vid])))
        for tid in (r["team_a_id"], r["team_b_id"]):
            team_rounds.setdefault(tid, []).append((rd, vid, sid))

    # 逐队旅行 / 休息 / 时区
    for tid, entries in team_rounds.items():
        entries = sorted(entries, key=lambda x: x[0])
        prev_vid = None
        for i, (rd, vid, sid) in enumerate(entries):
            if rd == 1:
                distances.append(L["t2v_km"].get((tid, vid), 0.0))
                tz_cross.append(abs(L["t2v_tz"].get((tid, vid), 0.0)))
            else:
                distances.append(L["v2v_km"].get((prev_vid, vid), 0.0))
                tz_cross.append(abs(L["v2v_tz"].get((prev_vid, vid), 0.0)))
            prev_vid = vid
        # 休息时间（相邻轮 UTC 差的最小值，逐队分下）
        utcs = sorted([(rd, L["s_utc"][sid]) for (rd, vid, sid) in entries])
        for (r1, u1), (r2, u2) in zip(utcs, utcs[1:]):
            if r2 == r1 + 1:
                rests.append((u2 - u1).total_seconds() / 3600.0)

    # 场馆利用率：使用场馆数 / 16 × (1 - 载荷变异系数)
    loads = sched["venue_id"].value_counts().reindex(venue_ids, fill_value=0)
    used = (loads > 0).sum()
    cv = loads[loads > 0].std() / loads[loads > 0].mean() if used > 0 else 0.0
    utilization = (used / len(venue_ids)) * (1.0 - min(cv, 1.0))

    return {
        "travel_distance_km_per_team": float(np.mean(distances)),
        "rest_time_hours_min": float(np.mean(rests)) if rests else 0.0,
        "timezone_crossings_per_team": float(np.mean(tz_cross)),
        "venue_utilization": float(utilization),
        "capacity_match_occupancy": float(np.mean(occ)),
        "prime_time_coverage": float(prime_cnt / n),
    }


# ------------------------------------------------------------------
# 3. 构造基准"实际赛程"（真实爬取数据缺失时的替代）
# ------------------------------------------------------------------
def build_baseline_actual_schedule(sheets, L):
    groups = sheets["groups_matches"].sort_values("match_id").reset_index(drop=True)
    slots = sheets["time_slots"].sort_values("slot_id").reset_index(drop=True)
    venues = sheets["venues"]

    venue_ids = venues["venue_id"].tolist()
    slot_ids = slots["slot_id"].tolist()          # 80 个全局时段
    # 每轮候选时段（同问题2窗口），模拟真实赛程的分轮时间带
    slots["day_index"] = (pd.to_datetime(slots["date"]) - pd.to_datetime(slots["date"].min())).dt.days + 1
    win = {1: (1, 6), 2: (8, 13), 3: (15, 20)}
    round_slot_ids = {r: slots[slots["day_index"].between(a, b)]["slot_id"].tolist()
                      for r, (a, b) in win.items()}

    rows = []
    for i, (_, m) in enumerate(groups.iterrows()):
        rd = int(m["round_in_group"])
        # 刻意分散场馆（忽略球队地理，制造高旅行负担）
        vid = venue_ids[(i * 7 + rd) % len(venue_ids)]
        # 掺入不均匀的时段（制造部分短休息）
        rsid = round_slot_ids[rd][(i * 3 + rd) % len(round_slot_ids[rd])]
        rows.append({
            "match_id": m["match_id"], "group_id": m["group_id"],
            "round_in_group": rd, "team_a": m["team_a"], "team_b": m["team_b"],
            "team_a_id": m["team_a_id"], "team_b_id": m["team_b_id"],
            "venue_id": vid, "slot_id": rsid,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 4. 入口
# ------------------------------------------------------------------
def run(verbose: bool = True) -> dict:
    sheets = load_all_sheets()
    L = build_lookups(sheets)

    # 优化赛程（问题2）
    opt = pd.read_csv(DATA_DIR / "result_2_group_schedule.csv")
    groups = sheets["groups_matches"].set_index("match_id")
    opt["team_a_id"] = opt["match_id"].map(groups["team_a_id"])
    opt["team_b_id"] = opt["match_id"].map(groups["team_b_id"])

    # 基准实际赛程
    act = build_baseline_actual_schedule(sheets, L)

    m_opt = compute_metrics(opt, L)
    m_act = compute_metrics(act, L)

    # 方向定义：minimize=越小越好；maximize=越大越好
    direction = {
        "travel_distance_km_per_team": "minimize",
        "rest_time_hours_min": "maximize",
        "timezone_crossings_per_team": "minimize",
        "venue_utilization": "maximize",
        "capacity_match_occupancy": "maximize",
        "prime_time_coverage": "maximize",
    }
    label = {
        "travel_distance_km_per_team": ("旅行距离", "每队平均旅行距离(km)"),
        "rest_time_hours_min": ("休息时间", "平均最小休息时间(h)"),
        "timezone_crossings_per_team": ("跨时区次数", "每队平均跨时区次数"),
        "venue_utilization": ("场馆利用率", "场馆利用率(均匀度)"),
        "capacity_match_occupancy": ("容量匹配度", "平均上座率(需求/容量)"),
        "prime_time_coverage": ("黄金时段覆盖率", "黄金时段场次占比"),
    }

    rows = []
    plot_labels, opt_norm, act_norm = [], [], []
    for k, direc in direction.items():
        a, o = m_act[k], m_opt[k]
        # 相对改善：以"更好方向"为正
        if direc == "maximize":
            rel = (o - a) / abs(a) if abs(a) > 1e-9 else 0.0
            better_opt = o > a
        else:
            rel = (a - o) / abs(a) if abs(a) > 1e-9 else 0.0
            better_opt = o < a
        result = "优化赛程更优" if better_opt else ("实际赛程更优" if not better_opt else "持平")
        rows.append({
            "indicator_name": label[k][1],
            "indicator_category": label[k][0],
            "actual_schedule_value": round(a, 6),
            "optimized_schedule_value": round(o, 6),
            "absolute_difference": round(o - a, 6),
            "relative_improvement": round(rel, 6),
            "preferred_direction": direc,
            "evaluation_result": result,
        })
        # 雷达图用：min-max 归一到 [0,1]（越小越好 -> 反转）
        lo, hi = min(a, o), max(a, o)
        if hi - lo < 1e-12:
            an = bn = 0.5
        else:
            if direc == "maximize":
                an, bn = (a - lo) / (hi - lo), (o - lo) / (hi - lo)
            else:
                an, bn = (hi - a) / (hi - lo), (hi - o) / (hi - lo)
        act_norm.append(an); opt_norm.append(bn)
        plot_labels.append(label[k][0])

    out = pd.DataFrame(rows)
    out.to_csv(DATA_DIR / "result_4_schedule_comparison.csv", index=False)

    # 基准实际赛程另存一份（透明起见）
    act.drop(columns=["team_a_id", "team_b_id"]).to_csv(
        DATA_DIR / "actual_schedule_generated.csv", index=False)

    plot_comparison(plot_labels, act_norm, opt_norm)
    if verbose:
        print("[P4] 已写出 result_4_schedule_comparison.csv 与可视化图片")
        print(out.to_string(index=False))
    return {"metrics_opt": m_opt, "metrics_act": m_act}


# ------------------------------------------------------------------
# 5. 可视化：雷达图 + 双柱状图
# ------------------------------------------------------------------
def plot_comparison(labels, act_norm, opt_norm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # --- 雷达图 ---
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    a = act_norm + act_norm[:1]
    o = opt_norm + opt_norm[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw=dict(polar=True) if False else {})
    # 左侧雷达
    ax = plt.subplot(1, 2, 1, polar=True)
    ax.plot(angles, a, "o-", label="实际赛程", color="#c44e52")
    ax.fill(angles, a, alpha=0.15, color="#c44e52")
    ax.plot(angles, o, "s-", label="优化赛程", color="#4c72b0")
    ax.fill(angles, o, alpha=0.15, color="#4c72b0")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("赛程质量雷达图（归一化）", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # 右侧双柱状图
    ax2 = plt.subplot(1, 2, 2)
    x = np.arange(len(labels))
    w = 0.38
    ax2.bar(x - w / 2, act_norm, w, label="实际赛程", color="#c44e52")
    ax2.bar(x + w / 2, opt_norm, w, label="优化赛程", color="#4c72b0")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("归一化得分"); ax2.set_ylim(0, 1.1)
    ax2.set_title("双柱状对比（归一化）")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "result_4_schedule_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run(verbose=True)