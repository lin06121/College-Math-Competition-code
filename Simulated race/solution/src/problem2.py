# -*- coding: utf-8 -*-
"""
src/problem2.py — 任务二：场馆与时段协同组合优化
====================================================
为 72 场小组赛同时安排 venue_id 与 slot_id，最大化目标函数

    Z2 = 0.25T + 0.25B + 0.15U + 0.10H - 0.08C - 0.07D - 0.06F - 0.04R

实现要点
--------
1. 组合规模大（72 × 候选 venue/slot），直接整数规划不可行，采用**遗传算法**。
2. 八项指标中 T/B/C/D 需在**当前种群全体**做 min-max 归一化（相对归一化，
   契合题面"坑点 2"）；U/H/F/R 本身已是 0-1 尺度（均值），不做种群归一化。
3. 硬约束（唯一性、休息时间、单日场馆容量、总场次、安保、黄金时段公平）
   通过分组建模的初始化 + 惩罚项保证。

单位与公式约定（详见 README.md）：
   T=基础票价×预测观众(人)；B=观众×单位观看价值×全球黄金值×赞助权重；
   C=场馆启用成本+运营成本+安保成本；D=跨轮旅行时间(小时)合计；
   F=0.5·黄金时段极差/3+0.5·大容量场馆极差/3；R=气候/上座/安保需求的均值。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (DATA_DIR, RANDOM_SEED, GA_POPULATION, GA_GENERATIONS,
                    GA_TOURNAMENT, GA_CROSSOVER_PROB, GA_MUTATION_PROB,
                    GA_ELITISM, REST_MIN_HOURS, GOLDEN_PRIME_TOP_FRACTION,
                    LARGE_CAPACITY_TOP_FRACTION, SECURITY_UNIT_COST_MUSD,
                    ROUND_DAY_WINDOWS, Z2_WEIGHTS)
from utils.data_loader import load_all_sheets
from utils.metrics import minmax_normalize

rng = np.random.default_rng(RANDOM_SEED)

_STAGE_BY_ROUND = {1: "Group_Match_R1", 2: "Group_Match_R2", 3: "Group_Match_R3"}


# ------------------------------------------------------------------
# 1. 数据准备
# ------------------------------------------------------------------
def build_data(sheets, pred_viewers):
    """把各表整理成可向量化的索引结构。pred_viewers: {match_id -> 人}。"""
    groups = sheets["groups_matches"].sort_values("match_id").reset_index(drop=True)
    venues = sheets["venues"]
    slots = sheets["time_slots"]
    dist = sheets["distance_matrix"]
    sec = sheets["security_requirements"]
    base = sheets["base_predictions"]
    ticket = sheets["ticket_broadcast"]

    groups = groups.merge(
        base[["match_id", "uncertainty_index", "attractiveness_index",
              "expected_attendance_base"]], on="match_id", how="left")
    groups = groups.merge(
        sec[["match_id", "required_security_level", "security_demand_score"]],
        on="match_id", how="left")
    groups["predicted_viewers"] = groups["match_id"].map(pred_viewers).astype(float)

    tb = ticket.set_index("match_stage")
    groups["base_price"] = groups["round_in_group"].map(
        _STAGE_BY_ROUND).map(lambda s: float(tb.loc[s, "base_ticket_price_usd"]))
    groups["unit_value"] = groups["round_in_group"].map(
        _STAGE_BY_ROUND).map(lambda s: float(tb.loc[s, "broadcast_unit_value_usd"]))
    groups["sponsor_weight"] = groups["round_in_group"].map(
        _STAGE_BY_ROUND).map(lambda s: float(tb.loc[s, "sponsor_weight"]))

    v_id2idx = {v: i for i, v in enumerate(venues["venue_id"])}
    v_info = {
        "id": venues["venue_id"].tolist(),
        "city": venues["city"].tolist(),
        "country": venues["country"].tolist(),
        "capacity": venues["capacity"].to_numpy(float),
        "security_level": venues["security_level"].to_numpy(int),
        "security_cost_index": venues["security_cost_index"].to_numpy(float),
        "climate_risk": venues["climate_risk"].to_numpy(float),
        "setup_cost": venues["setup_cost_musd"].to_numpy(float),
        "operation_cost": venues["operation_cost_musd_per_match"].to_numpy(float),
        "max_per_day": venues["max_matches_per_day"].to_numpy(int),
        "min_total": venues["min_total_matches"].to_numpy(int),
        "max_total": venues["max_total_matches"].to_numpy(int),
    }

    slots = slots.sort_values("slot_id").reset_index(drop=True)
    slots["utc_dt"] = pd.to_datetime(slots["reference_utc_time"])
    min_date = slots["date"].min()
    slots["day_index"] = (slots["date"] - min_date).dt.days + 1
    s_id2idx = {s: i for i, s in enumerate(slots["slot_id"])}
    s_info = {
        "id": slots["slot_id"].tolist(),
        "date": slots["date"].dt.strftime("%Y-%m-%d").tolist(),
        "kickoff": slots["reference_kickoff_time"].tolist(),
        "utc_dt": slots["utc_dt"].tolist(),
        "prime": slots["global_prime_score"].to_numpy(float),
        "day_index": slots["day_index"].to_numpy(int),
    }

    t2v = {}
    v2v = {}
    for _, r in dist.iterrows():
        o, d = r["origin_id"], r["destination_id"]
        if r["relation_type"] == "team_to_venue":
            t2v[(o, d)] = float(r["travel_time_hour"])
        else:
            v2v[(o, d)] = float(r["travel_time_hour"])

    n = len(groups)
    match_info = {
        "match_id": groups["match_id"].tolist(),
        "group_id": groups["group_id"].tolist(),
        "round": groups["round_in_group"].to_numpy(int),
        "team_a": groups["team_a"].tolist(),
        "team_b": groups["team_b"].tolist(),
        "team_a_id": groups["team_a_id"].tolist(),
        "team_b_id": groups["team_b_id"].tolist(),
        "required_sec": groups["required_security_level"].to_numpy(int),
        "sec_demand": groups["security_demand_score"].to_numpy(float),
        "uncertainty": groups["uncertainty_index"].to_numpy(float),
        "attractiveness": groups["attractiveness_index"].to_numpy(float),
        "attendance": groups["expected_attendance_base"].to_numpy(float),
        "viewers": groups["predicted_viewers"].to_numpy(float),
        "base_price": groups["base_price"].to_numpy(float),
        "unit_value": groups["unit_value"].to_numpy(float),
        "sponsor_weight": groups["sponsor_weight"].to_numpy(float),
    }
    valid_venues = [
        [i for i in range(len(v_info["id"]))
         if v_info["security_level"][i] >= match_info["required_sec"][m]]
        for m in range(n)
    ]
    # 每轮候选时段（全局 slot index）
    round_slots = {}
    round_local_map = {}
    for rd, (d0, d1) in ROUND_DAY_WINDOWS.items():
        idx = [i for i in range(len(s_info["id"]))
               if d0 <= s_info["day_index"][i] <= d1]
        round_slots[rd] = idx
        round_local_map[rd] = {g: loc for loc, g in enumerate(idx)}
    round_slot_idx = [round_slots[rd] for rd in match_info["round"]]

    # 每天 -> 全局 slot 下标（用于分组建模初始化）
    day_to_slots = {}
    for i in range(len(s_info["id"])):
        day_to_slots.setdefault(int(s_info["day_index"][i]), []).append(i)
    for d in day_to_slots:
        day_to_slots[d].sort()

    # 每组每轮的比赛下标
    group_round_matches = {}
    for m in range(n):
        group_round_matches.setdefault((match_info["group_id"][m], match_info["round"][m]), []).append(m)

    prime_order = np.argsort(s_info["prime"])[::-1]
    n_prime = max(1, int(len(s_info["id"]) * GOLDEN_PRIME_TOP_FRACTION))
    golden_slots = set(prime_order[:n_prime].tolist())
    cap_order = np.argsort(v_info["capacity"])[::-1]
    n_large = max(1, int(len(v_info["id"]) * LARGE_CAPACITY_TOP_FRACTION))
    large_venues = set(cap_order[:n_large].tolist())

    team_matches = {}
    for m in range(n):
        for tid in (match_info["team_a_id"][m], match_info["team_b_id"][m]):
            team_matches.setdefault(tid, {})[match_info["round"][m]] = m

    return {
        "n": n, "match_info": match_info, "valid_venues": valid_venues,
        "round_slot_idx": round_slot_idx, "round_local_map": round_local_map,
        "v_info": v_info, "s_info": s_info, "t2v": t2v, "v2v": v2v,
        "golden_slots": golden_slots, "large_venues": large_venues,
        "team_matches": team_matches, "day_to_slots": day_to_slots,
        "group_round_matches": group_round_matches,
        "v_id2idx": v_id2idx, "s_id2idx": s_id2idx,
    }


# ------------------------------------------------------------------
# 2. 原始指标与约束违规
# ------------------------------------------------------------------
def evaluate_raw(venues, slot_genes, data):
    mi = data["match_info"]
    vi = data["v_info"]
    si = data["s_info"]
    n = data["n"]

    slot_globals = np.array([data["round_slot_idx"][m][slot_genes[m]]
                             for m in range(n)], dtype=int)
    venue_globals = np.asarray(venues, dtype=int)

    T = B = C = D = 0.0
    U = H = R = 0.0
    used_venues = set()
    golden_cnt, large_cnt = {}, {}
    team_slot = []
    team_venue = {}
    pair_count, venue_day = {}, {}

    for m in range(n):
        vg = venue_globals[m]
        sg = slot_globals[m]
        used_venues.add(vg)

        T += mi["base_price"][m] * mi["viewers"][m]
        B += (mi["viewers"][m] * mi["unit_value"][m]
              * si["prime"][sg] * mi["sponsor_weight"][m])
        U += mi["uncertainty"][m]
        H += mi["attractiveness"][m] / 100.0
        C += vi["operation_cost"][vg] + SECURITY_UNIT_COST_MUSD \
            * mi["required_sec"][m] * vi["security_cost_index"][vg]

        occupancy = min(1.0, mi["attendance"][m] / vi["capacity"][vg])
        R += (vi["climate_risk"][vg] + occupancy + mi["sec_demand"][m]) / 3.0

        pair = (vg, sg)
        pair_count[pair] = pair_count.get(pair, 0) + 1
        day = int(si["day_index"][sg])
        venue_day[(vg, day)] = venue_day.get((vg, day), 0) + 1

        rd = mi["round"][m]
        for tid in (mi["team_a_id"][m], mi["team_b_id"][m]):
            golden_cnt[tid] = golden_cnt.get(tid, 0) + (1 if sg in data["golden_slots"] else 0)
            large_cnt[tid] = large_cnt.get(tid, 0) + (1 if vg in data["large_venues"] else 0)
            team_slot.append((tid, rd, si["utc_dt"][sg]))
            team_venue.setdefault(tid, {})[rd] = vg

    C += sum(vi["setup_cost"][vg] for vg in used_venues)

    for tid, rounds in data["team_matches"].items():
        prev = None
        for rd in (1, 2, 3):
            if rd not in rounds:
                continue
            vg = team_venue[tid][rd]
            if rd == 1:
                D += data["t2v"].get((tid, vi["id"][vg]), 0.0)
            else:
                D += data["v2v"].get((vi["id"][prev], vi["id"][vg]), 0.0)
            prev = vg

    golden_rng = max(golden_cnt.values()) - min(golden_cnt.values())
    large_rng = max(large_cnt.values()) - min(large_cnt.values())
    F = 0.5 * (golden_rng / 3.0) + 0.5 * (large_rng / 3.0)

    U /= n; H /= n; R /= n

    # ---- 约束违规 ----
    penalty = 0.0
    for (vg, sg), c in pair_count.items():
        if c > 1:
            penalty += 5.0 * (c - 1)

    team_slots_by_team = {}
    for (tid, rd, dt) in team_slot:
        team_slots_by_team.setdefault(tid, {})[rd] = dt
    for tid, ss in team_slots_by_team.items():
        for r1, r2 in ((1, 2), (2, 3)):
            if r1 in ss and r2 in ss:
                gap_h = (ss[r2] - ss[r1]).total_seconds() / 3600.0
                if gap_h < REST_MIN_HOURS:
                    penalty += 0.5 * (REST_MIN_HOURS - gap_h)

    for (vg, day), c in venue_day.items():
        cap = vi["max_per_day"][vg]
        if c > cap:
            penalty += 4.0 * (c - cap)

    venue_total = {}
    for m in range(n):
        venue_total[venue_globals[m]] = venue_total.get(venue_globals[m], 0) + 1
    for vg, c in venue_total.items():
        if c < vi["min_total"][vg]:
            penalty += 3.0 * (vi["min_total"][vg] - c)
        if c > vi["max_total"][vg]:
            penalty += 3.0 * (c - vi["max_total"][vg])

    for m in range(n):
        if vi["security_level"][venue_globals[m]] < mi["required_sec"][m]:
            penalty += 10.0

    if golden_rng > 2:
        penalty += 2.0 * (golden_rng - 2)

    return {
        "T": T, "B": B, "C": C, "D": D, "U": U, "H": H, "F": F, "R": R,
        "penalty": penalty, "slot_globals": slot_globals,
        "venue_globals": venue_globals,
    }


def population_z2(raw_list):
    T = np.array([r["T"] for r in raw_list])
    B = np.array([r["B"] for r in raw_list])
    C = np.array([r["C"] for r in raw_list])
    D = np.array([r["D"] for r in raw_list])
    U = np.array([r["U"] for r in raw_list])
    H = np.array([r["H"] for r in raw_list])
    F = np.array([r["F"] for r in raw_list])
    R = np.array([r["R"] for r in raw_list])
    pen = np.array([r["penalty"] for r in raw_list])

    Tn, Bn = minmax_normalize(T), minmax_normalize(B)
    Cn, Dn = minmax_normalize(C), minmax_normalize(D)

    z2 = (Z2_WEIGHTS["T"] * Tn + Z2_WEIGHTS["B"] * Bn + Z2_WEIGHTS["U"] * U
          + Z2_WEIGHTS["H"] * H - Z2_WEIGHTS["C"] * Cn - Z2_WEIGHTS["D"] * Dn
          - Z2_WEIGHTS["F"] * F - Z2_WEIGHTS["R"] * R)
    fit = z2 - pen
    return z2, fit


# ------------------------------------------------------------------
# 3. 遗传算法
# ------------------------------------------------------------------
def genetic_algorithm(data):
    n = data["n"]
    valid_venues = data["valid_venues"]
    round_slot_idx = data["round_slot_idx"]
    round_local_map = data["round_local_map"]
    day_to_slots = data["day_to_slots"]
    group_round_matches = data["group_round_matches"]
    mi = data["match_info"]

    window_start = {1: ROUND_DAY_WINDOWS[1][0], 2: ROUND_DAY_WINDOWS[2][0],
                    3: ROUND_DAY_WINDOWS[3][0]}

    def grouped_init():
        """分组建模初始化：每组随机一天偏移 k，同跨度跨轮放置，保证休息约束。"""
        v = np.empty(n, dtype=int)
        s = np.empty(n, dtype=int)
        for (gid, rd), ms in group_round_matches.items():
            k = rng.integers(0, 6)              # 偏移天数
            day = window_start[rd] + k
            day_slots = day_to_slots[day]        # 该日 4 个全局 slot
            slots_for_round = rng.choice(day_slots, size=len(ms), replace=False)
            for j, m in enumerate(ms):
                v[m] = rng.choice(valid_venues[m])
                s[m] = round_local_map[rd][int(slots_for_round[j])]
        return v, s

    def random_init():
        v = np.array([rng.choice(valid_venues[m]) for m in range(n)])
        s = rng.integers(0, len(round_slot_idx[0]), size=n)
        return v, s

    def mutate(v, s):
        v, s = v.copy(), s.copy()
        if rng.random() < GA_MUTATION_PROB:
            for m in range(n):
                if rng.random() < 0.04:
                    v[m] = rng.choice(valid_venues[m])
                if rng.random() < 0.04:
                    s[m] = rng.integers(0, len(round_slot_idx[m]))
        return v, s

    def crossover(v1, s1, v2, s2):
        if rng.random() < GA_CROSSOVER_PROB:
            mask = rng.random(n) < 0.5
            v1n = np.where(mask, v2, v1)
            v2n = np.where(mask, v1, v2)
            s1n = np.where(mask, s2, s1).astype(int)
            s2n = np.where(mask, s1, s2).astype(int)
            v1, v2, s1, s2 = v1n, v2n, s1n, s2n
        return (v1, s1), (v2, s2)

    pop = [grouped_init() for _ in range(GA_POPULATION)]

    best_ind, best_z2, best_pen = None, -np.inf, np.inf
    for gen in range(GA_GENERATIONS):
        raw_list = [evaluate_raw(v, s, data) for (v, s) in pop]
        z2, fit = population_z2(raw_list)

        for i in range(len(pop)):
            r = raw_list[i]
            if r["penalty"] < best_pen - 1e-9 or \
               (abs(r["penalty"] - best_pen) < 1e-9 and z2[i] > best_z2):
                best_ind, best_z2, best_pen = pop[i], z2[i], r["penalty"]

        order = np.argsort(fit)[::-1]
        new_pop = [pop[i] for i in order[:GA_ELITISM]]
        while len(new_pop) < GA_POPULATION:
            idx = rng.choice(GA_POPULATION, size=GA_TOURNAMENT, replace=False)
            p1 = pop[idx[np.argmax(fit[idx])]]
            idx = rng.choice(GA_POPULATION, size=GA_TOURNAMENT, replace=False)
            p2 = pop[idx[np.argmax(fit[idx])]]
            c1, c2 = crossover(p1[0], p1[1], p2[0], p2[1])
            new_pop.append(mutate(*c1))
            if len(new_pop) < GA_POPULATION:
                new_pop.append(mutate(*c2))
        pop = new_pop

    return best_ind, evaluate_raw(best_ind[0], best_ind[1], data), best_z2, best_pen


# ------------------------------------------------------------------
# 4. 入口
# ------------------------------------------------------------------
def run(pred_viewers: dict | None = None, verbose: bool = True) -> dict:
    sheets = load_all_sheets()
    if pred_viewers is None:
        df1 = pd.read_csv(DATA_DIR / "result_1_match_prediction.csv")
        pred_viewers = dict(zip(df1["match_id"], df1["predicted_tv_viewers"]))

    data = build_data(sheets, pred_viewers)
    best_ind, raw, best_z2, best_pen = genetic_algorithm(data)

    if verbose:
        print(f"[P2] GA 结束：Z2 = {best_z2:.6f}，约束违规 penalty = {best_pen:.4f}")

    write_schedule(best_ind, raw, data, best_z2, verbose)
    return {"z2": best_z2, "penalty": best_pen, "data": data,
            "best_ind": best_ind, "raw": raw}


def write_schedule(best_ind, raw, data, z2, verbose):
    mi = data["match_info"]
    vi = data["v_info"]
    si = data["s_info"]
    n = data["n"]

    venue_globals = raw["venue_globals"]
    slot_globals = raw["slot_globals"]

    rows = []
    for m in range(n):
        vg = venue_globals[m]
        sg = slot_globals[m]
        date, kickoff = si["date"][sg], si["kickoff"][sg]
        t_m = mi["base_price"][m] * mi["viewers"][m]
        b_m = (mi["viewers"][m] * mi["unit_value"][m]
               * si["prime"][sg] * mi["sponsor_weight"][m])
        travel_m = 0.0
        for tid in (mi["team_a_id"][m], mi["team_b_id"][m]):
            rd = mi["round"][m]
            if rd == 1:
                travel_m += data["t2v"].get((tid, vi["id"][vg]), 0.0)
            else:
                pvg = raw["venue_globals"][data["team_matches"][tid][rd - 1]]
                travel_m += data["v2v"].get((vi["id"][pvg], vi["id"][vg]), 0.0)
        occupancy = min(1.0, mi["attendance"][m] / vi["capacity"][vg])
        risk_m = (vi["climate_risk"][vg] + occupancy + mi["sec_demand"][m]) / 3.0

        rows.append({
            "match_id": mi["match_id"][m],
            "group_id": mi["group_id"][m],
            "round_in_group": mi["round"][m],
            "team_a": mi["team_a"][m],
            "team_b": mi["team_b"][m],
            "venue_id": vi["id"][vg],
            "slot_id": si["id"][sg],
            "city": vi["city"][vg],
            "country": vi["country"][vg],
            "reference_date": date,
            "reference_kickoff_time": kickoff,
            "local_datetime": f"{date} {kickoff}",
            "utc_datetime": si["utc_dt"][sg].strftime("%Y-%m-%d %H:%M"),
            "required_security_level": int(mi["required_sec"][m]),
            "expected_attendance": int(round(mi["attendance"][m])),
            "expected_tv_viewers": int(round(mi["viewers"][m])),
            "ticket_revenue_usd": round(t_m, 2),
            "broadcast_value_usd": round(b_m, 2),
            "travel_cost_index": round(travel_m, 4),
            "fairness_penalty": round(raw["F"], 6),
            "risk_index": round(risk_m, 6),
            "total_objective_value": "",
        })

    out = pd.DataFrame(rows)
    out.loc[0, "total_objective_value"] = round(z2, 6)
    cols = ["match_id", "group_id", "round_in_group", "team_a", "team_b",
            "venue_id", "slot_id", "city", "country", "reference_date",
            "reference_kickoff_time", "local_datetime", "utc_datetime",
            "required_security_level", "expected_attendance",
            "expected_tv_viewers", "ticket_revenue_usd", "broadcast_value_usd",
            "travel_cost_index", "fairness_penalty", "risk_index",
            "total_objective_value"]
    out = out[cols]
    out.to_csv(DATA_DIR / "result_2_group_schedule.csv", index=False)
    if verbose:
        print("[P2] 已写出 result_2_group_schedule.csv")


if __name__ == "__main__":
    run(verbose=True)