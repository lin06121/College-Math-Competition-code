# -*- coding: utf-8 -*-
"""
src/problem3.py — 任务三：前两轮反馈下的第三轮动态资源优化
=============================================================
流程
----
1. 状态更新：由前两轮真实结果(live_group_results)计算球队竞技状态 s_t，
   据此更新 24 场第三轮比赛的泊松进球均值 λ（clip [0.15,4.50]）。
2. 蒙特卡洛模拟 20000 次：
   - 每次对 24 场第三轮抽样泊松进球；
   - 叠加前两轮积分，按（积分>净胜球>总进球）排名，并列按等比例分配晋级名额；
   - 得每队晋级概率 p_t 与每场晋级重要性 Q_i。
3. 现场/转播反馈 f_N、f_V；更新吸引力 A'、风险 R'（含 R_stake 无激励、R_col 默契）。
4. 资源分配（背包问题变体）：为每场选转播(1-3)/安保(1-4)/交通(1-3)/票价调整 δ，
   满足每日高等级数量与预算约束，最大化 Z3。
5. 静态方案 vs 动态方案：静态决策用赛前数据优化、赛后保持决策但用更新数据重算；
   动态决策用更新数据重新优化。improvement_rate = (D-S)/|S|。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (DATA_DIR, RANDOM_SEED, MC_N_SIMS, POISSON_LAMBDA_MIN,
                    POISSON_LAMBDA_MAX, FEEDBACK_N_MIN, FEEDBACK_N_MAX,
                    FEEDBACK_V_MIN, FEEDBACK_V_MAX, TICKET_DELTA_CANDIDATES,
                    Z3_WEIGHTS)
from utils.data_loader import load_all_sheets
from utils.metrics import minmax_normalize, clamp

rng = np.random.default_rng(RANDOM_SEED)

# 阶段参数（第三轮）
_R3 = "Group_Match_R3"
_r3 = {"base_price": 110.0, "elasticity": -0.75, "unit_value": 0.052,
       "sponsor_weight": 1.1}


# ------------------------------------------------------------------
# 1. 数据准备
# ------------------------------------------------------------------
def build_p3_data(sheets):
    groups = sheets["groups_matches"]
    base = sheets["base_predictions"]
    sec = sheets["security_requirements"]
    live = sheets["live_group_results"]
    teams = sheets["teams"]
    membership = sheets["group_membership"]
    costs = sheets["dynamic_resource_costs"]
    limits = sheets["dynamic_resource_limits"]

    # 球队 id<->名
    id2name = dict(zip(teams["team_id"], teams["team_name"]))
    name2id = {v: k for k, v in id2name.items()}

    # 分组（group-major 排序，供蒙特卡洛 reshape）
    membership = membership.sort_values(["group_id", "team_id"]).reset_index(drop=True)
    team_ids = membership["team_id"].tolist()               # 48
    team_idx = {tid: i for i, tid in enumerate(team_ids)}
    group_of = membership["group_id"].tolist()              # 每队所属组
    n_groups = membership["group_id"].nunique()             # 12

    # ---- 前两轮逐队统计 ----
    stats = {tid: dict(pts=0, gf=0, ga=0, xgf=0.0, xga=0.0, red=0,
                       inj_sum=0, inj_n=0) for tid in team_ids}
    for _, r in live.iterrows():
        a_id = name2id[r["team_a"]]
        b_id = name2id[r["team_b"]]
        ga, gb = int(r["goals_a"]), int(r["goals_b"])
        for tid, gfor, gagg, xf, xa, red in (
                (a_id, ga, gb, r["xg_a"], r["xg_b"], r["red_cards_a"]),
                (b_id, gb, ga, r["xg_b"], r["xg_a"], r["red_cards_b"])):
            s = stats[tid]
            s["gf"] += gfor; s["ga"] += gagg
            s["xgf"] += float(xf); s["xga"] += float(xa); s["red"] += int(red)
            s["pts"] += 3 if gfor > gagg else (1 if gfor == gagg else 0)
            s["inj_sum"] += int(r["injury_impact_level"]); s["inj_n"] += 1

    # 竞技状态 s_t 与泊松乘子
    form_factor = {}
    for tid, s in stats.items():
        pts_comp = 0.5 * (s["pts"] / 6.0)
        xgd = s["xgf"] - s["xga"]
        xg_comp = 0.3 * ((xgd + 6.0) / 12.0)
        red_comp = 0.2 * (1.0 / (1.0 + s["red"]))
        s_t = pts_comp + xg_comp + red_comp          # [0,1]
        injury = 1.0 - 0.06 * (s["inj_sum"] / s["inj_n"])
        form_factor[tid] = (0.7 + 0.6 * s_t) * injury

    # ---- 24 场第三轮比赛 ----
    r3 = groups[groups["round_in_group"] == 3].sort_values("match_id").reset_index(drop=True)
    r3 = r3.merge(base[["match_id", "expected_goals_a", "expected_goals_b",
                        "uncertainty_index", "attractiveness_index",
                        "expected_attendance_base"]], on="match_id", how="left")
    r3 = r3.merge(sec[["match_id", "security_demand_score"]], on="match_id", how="left")

    lam_a, lam_b = [], []
    for _, r in r3.iterrows():
        la = float(r["expected_goals_a"]) * form_factor[name2id[r["team_a"]]]
        lb = float(r["expected_goals_b"]) * form_factor[name2id[r["team_b"]]]
        lam_a.append(float(np.clip(la, POISSON_LAMBDA_MIN, POISSON_LAMBDA_MAX)))
        lam_b.append(float(np.clip(lb, POISSON_LAMBDA_MIN, POISSON_LAMBDA_MAX)))

    # 每组 4 队的队标（group-major）
    group_teams = {}
    for i, tid in enumerate(team_ids):
        group_teams.setdefault(group_of[i], []).append(i)

    # 前两轮基础分（按 team_idx 顺序）
    base_pts = np.array([stats[team_ids[i]]["pts"] for i in range(len(team_ids))], float)
    base_gf = np.array([stats[team_ids[i]]["gf"] for i in range(len(team_ids))], float)
    base_ga = np.array([stats[team_ids[i]]["ga"] for i in range(len(team_ids))], float)

    # 第三轮每场的两队在 team_idx 中的下标
    a_idx = r3["team_a_id"].map(team_idx).to_numpy(int)
    b_idx = r3["team_b_id"].map(team_idx).to_numpy(int)
    match_group = r3["group_id"].tolist()

    # 每队第三轮对阵 (match, 是否为主队a)
    team3_match = np.full(len(team_ids), -1, int)
    team3_is_a = np.zeros(len(team_ids), bool)
    for m in range(len(r3)):
        team3_match[a_idx[m]] = m; team3_is_a[a_idx[m]] = True
        team3_match[b_idx[m]] = m; team3_is_a[b_idx[m]] = False

    # 资源成本表
    cost_tbl = {(str(rt), int(lv)): float(uv)
                for rt, lv, uv in zip(costs["resource_type"], costs["resource_level"],
                                      costs["unit_cost_index"])}
    dem_tbl = {}
    risk_tbl = {}
    for _, c in costs.iterrows():
        key = (str(c["resource_type"]), int(c["resource_level"]))
        dem_tbl[key] = float(c["demand_multiplier"])
        risk_tbl[key] = float(c["risk_multiplier"])

    return {
        "r3": r3, "team_ids": team_ids, "team_idx": team_idx, "group_of": group_of,
        "group_teams": group_teams, "n_groups": n_groups,
        "lam_a": np.array(lam_a), "lam_b": np.array(lam_b),
        "base_pts": base_pts, "base_gf": base_gf, "base_ga": base_ga,
        "a_idx": a_idx, "b_idx": b_idx, "match_group": match_group,
        "team3_match": team3_match, "team3_is_a": team3_is_a,
        "cost_tbl": cost_tbl, "dem_tbl": dem_tbl, "risk_tbl": risk_tbl,
        "limits": limits, "id2name": id2name, "name2id": name2id,
    }


# ------------------------------------------------------------------
# 2. 蒙特卡洛模拟
# ------------------------------------------------------------------
def monte_carlo(d):
    lam_a, lam_b = d["lam_a"], d["lam_b"]
    n_sims, n_match = MC_N_SIMS, len(lam_a)
    n_team = len(d["team_ids"])

    # 泊松抽样 + 逐队累加
    ga = rng.poisson(lam_a, size=(n_sims, n_match))
    gb = rng.poisson(lam_b, size=(n_sims, n_match))

    own = np.where(d["team3_is_a"][None, :], ga[:, d["team3_match"]],
                   gb[:, d["team3_match"]])
    opp = np.where(d["team3_is_a"][None, :], gb[:, d["team3_match"]],
                   ga[:, d["team3_match"]])
    pts = d["base_pts"][None, :] + 3 * (own > opp) + 1 * (own == opp)
    gf = d["base_gf"][None, :] + own
    ga_t = d["base_ga"][None, :] + opp
    gd = gf - ga_t

    # group-major reshape -> 排名
    N = n_sims * d["n_groups"]
    pts3 = pts.reshape(N, 4).astype(int)
    gd3 = gd.reshape(N, 4).astype(int)
    gf3 = gf.reshape(N, 4).astype(int)

    order = np.lexsort((-gf3, -gd3, -pts3), axis=1)          # 降序，积分为主
    pts_s = np.take_along_axis(pts3, order, axis=1)
    gd_s = np.take_along_axis(gd3, order, axis=1)
    gf_s = np.take_along_axis(gf3, order, axis=1)

    # 整数键判定并列（积分>净胜球>总进球）
    key = (pts_s.astype(np.int64) * 100_000_000
           + (gd_s + 50).astype(np.int64) * 1_000
           + gf_s.astype(np.int64))
    cutoff = key[:, 1]
    n_above = (key > cutoff[:, None]).sum(axis=1)
    cnt_cutoff = np.maximum((key == cutoff[:, None]).sum(axis=1), 1)
    frac = np.where(key > cutoff[:, None], 1.0,
                    np.where(key == cutoff[:, None],
                             (2.0 - n_above[:, None]) / cnt_cutoff[:, None], 0.0))

    advance_flat = np.zeros((N, 4))
    np.put_along_axis(advance_flat, order, frac, axis=1)
    advance = advance_flat.reshape(n_sims, d["n_groups"], 4)  # group-major 即 team_idx

    # p_t：每队晋级概率
    p_t = advance.mean(axis=0).reshape(n_team)

    # Q_i：每场晋级重要性 = 两队胜负对晋级概率的边际影响之和
    Q = np.zeros(n_match)
    for m in range(n_match):
        a, b = d["a_idx"][m], d["b_idx"][m]
        adv_a = advance[:, :, :].reshape(n_sims, n_team)[:, a]
        adv_b = advance.reshape(n_sims, n_team)[:, b]
        aw = ga[:, m] > gb[:, m]
        bw = gb[:, m] > ga[:, m]
        imp_a = adv_a[aw].mean() - adv_a[~aw].mean()
        imp_b = adv_b[bw].mean() - adv_b[~bw].mean()
        Q[m] = imp_a + imp_b
    Q_norm = Q / max(Q.max(), 1e-9)

    # R_col：默契风险 = 打平且两队同时晋级概率
    R_col = np.zeros(n_match)
    for m in range(n_match):
        a, b = d["a_idx"][m], d["b_idx"][m]
        adv_a = advance.reshape(n_sims, n_team)[:, a]
        adv_b = advance.reshape(n_sims, n_team)[:, b]
        draw = ga[:, m] == gb[:, m]
        R_col[m] = ((adv_a[draw] + adv_b[draw]) / 2.0).mean()

    return p_t, Q, Q_norm, R_col


# ------------------------------------------------------------------
# 3. 反馈系数与更新
# ------------------------------------------------------------------
def update_values(d, p_t, Q_norm, R_col, pred_viewers, f_N, f_V):
    r3 = d["r3"]
    # 第三轮逐场更新
    rows = []
    for i, (_, r) in enumerate(r3.iterrows()):
        mid = r["match_id"]
        a_id = d["name2id"][r["team_a"]]
        b_id = d["name2id"][r["team_b"]]
        pa = p_t[d["team_idx"][a_id]]
        pb = p_t[d["team_idx"][b_id]]
        qn = Q_norm[i]

        att_upd = float(r["expected_attendance_base"]) * f_N * (1 + 0.25 * qn)
        tv_upd = float(pred_viewers[mid]) * f_V * (1 + 0.30 * qn)
        # 吸引力（0-100 尺度）
        A_raw = float(r["attractiveness_index"]) * (1 + 0.30 * qn)
        # 无激励风险 / 默契风险
        stake_a = 4 * pa * (1 - pa)
        stake_b = 4 * pb * (1 - pb)
        R_stake = 1.0 - (stake_a + stake_b) / 2.0
        R_c = R_col[i]
        R_raw = float(np.clip(0.4 * R_stake + 0.4 * R_c
                              + 0.2 * float(r["security_demand_score"]), 0.0, 1.0))
        rows.append({
            "match_id": mid, "group_id": r["group_id"],
            "team_a": r["team_a"], "team_b": r["team_b"],
            "pa": pa, "pb": pb, "att_upd": att_upd, "tv_upd": tv_upd,
            "A_raw": A_raw, "R_stake": R_stake, "R_col": R_c, "R_raw": R_raw,
        })
    upd = pd.DataFrame(rows)
    return upd


# ------------------------------------------------------------------
# 4. 资源分配候选与价值
# ------------------------------------------------------------------
def _cost(rt, lv, d):
    return d["cost_tbl"][(str(rt), int(lv))]


def _demand(rt, lv, d):
    return d["dem_tbl"].get((rt, lv), 1.0)


def _risk(rt, lv, d):
    return d["risk_tbl"].get((rt, lv), 1.0)


def candidate_contributions(upd, d, mode):
    """为每场生成全部候选 (b,s,t,δ) 的原始价值。

    mode: 'dynamic' 使用更新后数据；'static' 使用赛前数据（无 Q/f，风险不含 stake/collusion）。
    返回：
      contributions : dict[match_idx] -> list of dict(raw values)
      best_by_match : 每场最优候选下标（供 greedy/静态决策）
    """
    n = len(upd)
    r3 = d["r3"].reset_index(drop=True)
    base_pred = pd.read_csv(DATA_DIR / "result_1_match_prediction.csv")
    pv = base_pred.set_index("match_id")["predicted_tv_viewers"]

    all_c = []
    for i in range(n):
        row = upd.iloc[i]
        mid = row["match_id"]
        if mode == "dynamic":
            att = row["att_upd"]; tvu = row["tv_upd"]
            A = row["A_raw"] / 100.0; R0 = row["R_raw"]
        else:
            att = float(r3["expected_attendance_base"].iloc[i])
            tvu = float(pv[mid])
            A = float(r3["attractiveness_index"].iloc[i]) / 100.0
            R0 = float(r3["security_demand_score"].iloc[i])
        cs = []
        for b in (1, 2, 3):
            for s in (1, 2, 3, 4):
                for t in (1, 2, 3):
                    for delta in TICKET_DELTA_CANDIDATES:
                        att_eff = att * (1 + _r3["elasticity"] * delta) \
                            * _demand("transport", t, d)
                        att_eff = max(att_eff, 0.0)
                        TV = _r3["base_price"] * (1 + delta) * att_eff
                        BV = tvu * _r3["unit_value"] * _demand("broadcast", b, d) \
                            * _r3["sponsor_weight"]
                        C = (_cost("broadcast", b, d) + _cost("security", s, d)
                             + _cost("transport", t, d))
                        R = R0 * _risk("security", s, d)
                        cs.append({"b": b, "s": s, "t": t, "delta": delta,
                                   "TV": TV, "BV": BV, "C": C, "R": R, "A": A})
        all_c.append(cs)
    return all_c


def greedy_allocate(all_c, upd, d, dates):
    """贪心 + 修复：为每场选一个候选，满足每日高等级数量与预算约束。

    返回每场最终候选下标与 per-match net contribution。
    """
    n = len(all_c)
    # 展平并做全局 min-max（各后续候选 min-max）
    TVs = np.array([c["TV"] for cs in all_c for c in cs])
    BVs = np.array([c["BV"] for cs in all_c for c in cs])
    Cs = np.array([c["C"] for cs in all_c for c in cs])
    TVn = minmax_normalize(TVs)
    BVn = minmax_normalize(BVs)
    Cn = minmax_normalize(Cs)

    # 组织回 per-match 候选（含归一化价值）
    idx = 0
    cand = []
    for i in range(n):
        row = []
        for c in all_c[i]:
            avg_A = c["A"]
            net = (Z3_WEIGHTS["TV"] * TVn[idx] + Z3_WEIGHTS["BV"] * BVn[idx]
                   + Z3_WEIGHTS["A"] * avg_A - Z3_WEIGHTS["C"] * Cn[idx]
                   - Z3_WEIGHTS["R"] * c["R"])
            row.append({"b": c["b"], "s": c["s"], "t": c["t"], "delta": c["delta"],
                        "net": net, "cost": c["C"], "TV": c["TV"], "BV": c["BV"],
                        "R": c["R"]})
            idx += 1
        cand.append(row)

    # 每日限制（按比赛日期）
    limits = d["limits"].set_index("reference_date")
    day_of = dates              # 每场对应的 reference_date

    # 贪心：初始全取最低成本候选(0 号若按 b,s,t 升序已是最低)，再逐场升级
    # 候选已按 b,s,t,δ 升序生成，故第 0 个候选 = (1,1,1,-0.15)，成本最低。
    chosen = [0] * n
    used_cost = {dt: 0.0 for dt in set(day_of)}
    used_high_b = {dt: 0 for dt in set(day_of)}
    used_high_s = {dt: 0 for dt in set(day_of)}
    used_high_t = {dt: 0 for dt in set(day_of)}
    for i in range(n):
        c = cand[i][0]
        dt = day_of[i]
        used_cost[dt] += c["cost"]
        # 基线低等级不计入 high

    # 升级迭代
    changed = True
    while changed:
        changed = False
        for i in range(n):
            dt = day_of[i]
            cur = chosen[i]
            best_nb = -1; best_gain = 0.0
            for j in range(len(cand[i])):
                if j == cur:
                    continue
                c = cand[i][j]
                c_lv = cand[i][cur]
                dcost = c["cost"] - c_lv["cost"]
                lim = limits.loc[dt]
                if used_cost[dt] + dcost > float(lim["daily_resource_budget_index"]):
                    continue
                # 高等级约束
                dh_b = (1 if c["b"] == 3 else 0) - (1 if c_lv["b"] == 3 else 0)
                dh_s = (1 if c["s"] >= 3 else 0) - (1 if c_lv["s"] >= 3 else 0)
                dh_t = (1 if c["t"] == 3 else 0) - (1 if c_lv["t"] == 3 else 0)
                if used_high_b[dt] + dh_b > int(lim["high_broadcast_capacity"]):
                    continue
                if used_high_s[dt] + dh_s > int(lim["high_security_capacity"]):
                    continue
                if used_high_t[dt] + dh_t > int(lim["enhanced_transport_capacity"]):
                    continue
                gain = c["net"] - cand[i][cur]["net"]
                if gain > best_gain:
                    best_gain, best_nb = gain, j
            if best_nb >= 0:
                c = cand[i][best_nb]; c_lv = cand[i][cur]
                used_cost[dt] += c["cost"] - c_lv["cost"]
                used_high_b[dt] += (1 if c["b"] == 3 else 0) - (1 if c_lv["b"] == 3 else 0)
                used_high_s[dt] += (1 if c["s"] >= 3 else 0) - (1 if c_lv["s"] >= 3 else 0)
                used_high_t[dt] += (1 if c["t"] == 3 else 0) - (1 if c_lv["t"] == 3 else 0)
                chosen[i] = best_nb
                changed = True
    return chosen, cand


# ------------------------------------------------------------------
# 5. 入口
# ------------------------------------------------------------------
def run(verbose: bool = True) -> dict:
    sheets = load_all_sheets()
    d = build_p3_data(sheets)

    base_pred = pd.read_csv(DATA_DIR / "result_1_match_prediction.csv")
    pred_viewers = dict(zip(base_pred["match_id"], base_pred["predicted_tv_viewers"]))

    # 蒙特卡洛
    p_t, Q, Q_norm, R_col = monte_carlo(d)
    if verbose:
        print(f"[P3] 蒙特卡洛完成（{MC_N_SIMS} 次）；平均晋级概率 "
              f"{p_t.mean():.4f}，晋级重要性均值 {Q.mean():.4f}")

    # 反馈系数 f_N / f_V（前两轮实际 vs 预期）
    live = sheets["live_group_results"]
    exp_att = sheets["base_predictions"]
    exp_att_map = exp_att.set_index("match_id")["expected_attendance_base"]
    f_N = float(clamp(np.array([live["attendance"].sum()
                                / live["match_id"].map(exp_att_map).sum()]),
                      FEEDBACK_N_MIN, FEEDBACK_N_MAX)[0])
    f_V = float(clamp(np.array([live["tv_viewers"].sum()
                                / live["match_id"].map(pred_viewers).sum()]),
                      FEEDBACK_V_MIN, FEEDBACK_V_MAX)[0])

    # 反馈与更新
    upd = update_values(d, p_t, Q_norm, R_col, pred_viewers, f_N, f_V)
    if verbose:
        print(f"[P3] 反馈系数 f_N = {f_N:.4f}，f_V = {f_V:.4f}")

    # 第三轮比赛日期（来自问题2赛程）
    sched = pd.read_csv(DATA_DIR / "result_2_group_schedule.csv")
    r3_matches = d["r3"]["match_id"].tolist()
    sched_r3 = sched[sched["match_id"].isin(r3_matches)].set_index("match_id")
    dates = [sched_r3.loc[m, "reference_date"] for m in r3_matches]

    # 动态分配（更新数据）
    all_c_dyn = candidate_contributions(upd, d, mode="dynamic")
    # 静态分配（赛前数据）
    all_c_sta = candidate_contributions(upd, d, mode="static")

    # 静态决策：用赛前归一化做贪心，但需在其自身候选池上打分
    chosen_sta, cand_sta = greedy_allocate(all_c_sta, upd, d, dates)
    chosen_dyn, cand_dyn = greedy_allocate(all_c_dyn, upd, d, dates)

    # 静态净值的重算：用更新后数据的归一化体系评估静态决策
    # （cand_dyn 是更新后数据的候选，含归一化 net；按静态决策下标取 net）
    static_net = [cand_dyn[i][chosen_sta[i]]["net"] for i in range(len(upd))]
    static_net_total = float(sum(static_net))

    # 动态净值（直接取 cand_dyn 的动态决策 net）
    dynamic_net = [cand_dyn[i][chosen_dyn[i]]["net"] for i in range(len(upd))]
    dynamic_net_total = float(sum(dynamic_net))

    improvement = (dynamic_net_total - static_net_total) / abs(static_net_total) \
        if abs(static_net_total) > 1e-9 else 0.0
    if verbose:
        print(f"[P3] 静态净和 = {static_net_total:.6f}，动态净和 = {dynamic_net_total:.6f}，"
              f"改善率 = {improvement:.4%}")

    write_output(d, upd, p_t, Q_norm, R_col, chosen_dyn, chosen_sta,
                 cand_dyn, static_net, dynamic_net, improvement, verbose)
    return {"static_net": static_net_total, "dynamic_net": dynamic_net_total,
            "improvement": improvement}


def write_output(d, upd, p_t, Q_norm, R_col, chosen_dyn, chosen_sta,
                 cand_dyn, static_net, dynamic_net, improvement, verbose):
    n = len(upd)
    r3 = d["r3"].reset_index(drop=True)
    base_pred = pd.read_csv(DATA_DIR / "result_1_match_prediction.csv")
    pv = base_pred.set_index("match_id")["predicted_tv_viewers"]

    rows = []
    for i in range(n):
        row = upd.iloc[i]
        cd = cand_dyn[i][chosen_dyn[i]]
        cs = cand_dyn[i][chosen_sta[i]]
        mid = row["match_id"]
        # 上报的票务/转播价值用更新后数据 + 所选等级
        TV_dyn = cd["TV"]; BV_dyn = cd["BV"]
        rows.append({
            "match_id": mid,
            "group_id": row["group_id"],
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "updated_p_team_a_advance": round(row["pa"], 6),
            "updated_p_team_b_advance": round(row["pb"], 6),
            "updated_expected_attendance": int(round(row["att_upd"])),
            "updated_expected_tv_viewers": int(round(row["tv_upd"])),
            "stakeless_risk": round(row["R_stake"], 6),
            "collusion_risk": round(row["R_col"], 6),
            "updated_attractiveness": round(row["A_raw"], 4),
            "recommended_broadcast_priority": cd["b"],
            "recommended_security_level": cd["s"],
            "recommended_transport_level": cd["t"],
            "recommended_ticket_adjustment": cd["delta"],
            "updated_ticket_revenue_usd": round(TV_dyn, 2),
            "updated_broadcast_value_usd": round(BV_dyn, 2),
            "resource_cost_index": round(cd["cost"], 4),
            "risk_exposure_index": round(cd["R"], 6),
            "static_net_value": round(static_net[i], 6),
            "dynamic_net_value": round(dynamic_net[i], 6),
            "improvement_rate": round(improvement, 6),
        })
    out = pd.DataFrame(rows)
    cols = ["match_id", "group_id", "team_a", "team_b",
            "updated_p_team_a_advance", "updated_p_team_b_advance",
            "updated_expected_attendance", "updated_expected_tv_viewers",
            "stakeless_risk", "collusion_risk", "updated_attractiveness",
            "recommended_broadcast_priority", "recommended_security_level",
            "recommended_transport_level", "recommended_ticket_adjustment",
            "updated_ticket_revenue_usd", "updated_broadcast_value_usd",
            "resource_cost_index", "risk_exposure_index",
            "static_net_value", "dynamic_net_value", "improvement_rate"]
    out = out[cols]
    out.to_csv(DATA_DIR / "result_3_dynamic_strategy.csv", index=False)
    if verbose:
        print("[P3] 已写出 result_3_dynamic_strategy.csv")


if __name__ == "__main__":
    run(verbose=True)