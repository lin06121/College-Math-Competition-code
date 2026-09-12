"""
问题 2 对齐实验：在**同一预测依据**与**同一可实行（因果）结算规则**下，
比较两类日前计划方法的优劣，并量化结算口径的影响。

对照维度：
  预测依据 F:
    F_mate —— 负载二分类（周五/六 vs 其余）取最近 4 个同类日；光伏取最近 4 个连续日
    F_ours —— 负载/光伏均取最近 4 个同星期几
  计划方法 M:
    M_nv(q) —— 净需求残差 q 分位作为安全裕度（报童）+ 确定性 MILP
    M_saa   —— 最近 m 天误差曲线构造情景的两阶段随机规划
    M_point —— 无裕度点预测
  结算方式 S:
    S_causal —— 因果控制规则（**可实行**）
    S_recour —— 当日全部实测下的最优再调度（**不可实行**，下界参考）
"""
import numpy as np
import pandas as pd

from config import (
    SLOTS_PER_DAY, HOURS_PER_SLOT, STORAGE_INITIAL_SOC,
)
from forecast_q2 import load_historical_matrices, forecast_load_pv
from daily_solver import build_and_solve
from stochastic_q2 import compute_net_forecast_errors, build_scenarios, solve_stochastic_day
from method_q2 import (
    make_class_array, forecast_mate, compute_residuals_mate, load_fallback_profile,
    quantile_margin, causal_dispatch, stiff_dispatch,
)


def run_year(forecast_kind, plan_kind, settle="causal",
             q=0.7, m_scenarios=10, k_forecast=4, storage=True,
             warm_start="2025-01-01", start_date="2025-02-01", end_date="2025-12-31",
             verbose=False):
    """
    跑一整年，返回汇总 dict。

    forecast_kind : 'mate'（同类日负荷 + 近 4 日光伏，采用）| 'ours'（同星期几）
                    | 'persist'（昨日外推）| 'typical'（附件 1 典型日）
    plan_kind     : 'nv'（报童裕度）| 'saa'（两阶段随机规划）| 'point'（无裕度点预测）
    settle        : 'causal'（因果实时平衡，可实行）| 'stiff'（僵硬执行，无日内平衡）
                    | 'recour'（事后最优再调度，不可实行下界）
    storage       : False 时储能容量置零（无储能对照）

    注：'stiff' 需要日前计划的充放电轨迹，故只与 plan_kind ∈ {'nv','point'} 搭配。
    """
    dates, load, pv, price_mat = load_historical_matrices()
    net = load - pv
    d2i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    tds = pd.date_range(pd.Timestamp(warm_start), pd.Timestamp(end_date), freq="D")
    keep_from = pd.Timestamp(start_date)
    price = price_mat[d2i[pd.Timestamp(warm_start)]]

    cls_arr = make_class_array(dates)
    typ_l, typ_p = load_fallback_profile()

    def forecast(i, td):
        if forecast_kind == "ours":
            if i < 7:                    # 预热期首周无同星期几样本，退化为典型日
                return typ_l.copy(), typ_p.copy()
            return forecast_load_pv(td, dates, load, pv, k=k_forecast)
        if forecast_kind == "persist":
            if i == 0:
                return typ_l.copy(), typ_p.copy()
            return load[i - 1].copy(), np.maximum(pv[i - 1], 0.0)
        if forecast_kind == "typical":
            return typ_l.copy(), typ_p.copy()
        return forecast_mate(i, dates, load, pv, cls_arr)

    # 残差与该预测依据一致（裕度需与实际偏差同分布）
    if forecast_kind == "ours":
        _, err = compute_net_forecast_errors(dates, net, k=k_forecast)
    elif forecast_kind == "mate":
        err = compute_residuals_mate(dates, load, pv, net, cls_arr)
    else:
        err = np.full((len(dates), SLOTS_PER_DAY), np.nan)
        for i in range(len(dates)):
            lf, pf = forecast(i, dates[i])
            err[i] = net[i] - (lf - pf)

    e0 = STORAGE_INITIAL_SOC
    r = dict(plan_cost=0.0, em_cost=0.0, em_kWh=0.0, plan_kWh=0.0, curt_kWh=0.0)

    for td in tds:
        i = d2i[td]
        load_f, pv_f = forecast(i, td)
        net_f = load_f - pv_f
        net_a = net[i]

        # ---- 日前计划 ----
        C_plan = D_plan = None
        if not storage:
            b = quantile_margin(err, i, q) if plan_kind == "nv" else np.zeros(SLOTS_PER_DAY)
            P_plan = np.maximum(net_f + b, 0.0)
            C_plan = np.zeros(SLOTS_PER_DAY)
            D_plan = np.zeros(SLOTS_PER_DAY)
        elif plan_kind == "saa":
            has_hist = len(np.where(
                (np.arange(len(err)) < i) & (~np.isnan(err).any(axis=1)))[0]) > 0
            if has_hist:
                idx, probs = build_scenarios(i, err, dates, m=m_scenarios)
                P_plan, _, _ = solve_stochastic_day(
                    price, net_f, err[idx], probs, e0=e0, verbose=False)
            else:                        # 预热期无历史误差样本，退回确定性 LP
                sol = build_and_solve(price, load_f, pv_f, e0=e0, cyclic=False,
                                      allow_emergency=False, verbose=False)
                P_plan, C_plan, D_plan = sol["P_plan_kW"], sol["P_ch_kW"], sol["P_dis_kW"]
        else:
            b = quantile_margin(err, i, q) if plan_kind == "nv" else np.zeros(SLOTS_PER_DAY)
            sol = build_and_solve(price, load_f + b, pv_f, e0=e0,
                                  cyclic=False, allow_emergency=False, verbose=False)
            P_plan = sol["P_plan_kW"]
            C_plan, D_plan = sol["P_ch_kW"], sol["P_dis_kW"]

        # ---- 结算 ----
        if settle == "causal":
            P_ch, P_dis, P_em, E, P_curt = causal_dispatch(net_a, P_plan, e0)
        elif settle == "stiff":
            assert C_plan is not None, "stiff 结算需要日前充放电轨迹（仅支持 nv/point 计划）"
            P_ch, P_dis, P_em, E, P_curt = stiff_dispatch(
                net_a, P_plan, C_plan, D_plan, e0)
        else:
            from stochastic_q2 import solve_recourse_day
            s0 = solve_recourse_day(price, net_a, P_plan, e0=e0, verbose=False)
            P_ch, P_dis, P_em, E, P_curt = (s0["P_ch_kW"], s0["P_dis_kW"],
                                            s0["P_em_kW"], s0["E_kWh"], s0["P_curt_kW"])

        e_next = E[-1]
        if td >= keep_from:                       # 1 月仅作储电量链条预热，不计入统计
            r["plan_cost"] += float((P_plan * price).sum() * HOURS_PER_SLOT)
            r["em_cost"] += float((P_em * price).sum() * HOURS_PER_SLOT * 5.0)
            r["em_kWh"] += float(P_em.sum() * HOURS_PER_SLOT)
            r["plan_kWh"] += float(P_plan.sum() * HOURS_PER_SLOT)
            r["curt_kWh"] += float(P_curt.sum() * HOURS_PER_SLOT)
        e0 = e_next

    r["total_cost"] = r["plan_cost"] + r["em_cost"]
    if verbose:
        print(f"  {forecast_kind}/{plan_kind}/{settle}: 总 {r['total_cost']:,.0f} 元")
    return r


if __name__ == "__main__":
    rows = []

    def add(tag, **kw):
        r = run_year(**kw)
        rows.append((tag, r))
        print(f"{tag:<44} 计划 {r['plan_cost']:>11,.0f}  紧急 {r['em_cost']:>10,.0f}  "
              f"总 {r['total_cost']:>12,.0f}  紧急电量 {r['em_kWh']:>9,.0f} kWh")

    print("=" * 118)
    print("A. 同一预测（同伴依据）+ 同一可实行结算（因果）：纯算法对比")
    print("=" * 118)
    for q in (0.6, 0.7, 0.75, 0.8, 0.9):
        add(f"报童裕度 q={q}   + 因果结算", forecast_kind="mate", plan_kind="nv",
            q=q, settle="causal")
    add("两阶段随机规划 SAA + 因果结算", forecast_kind="mate", plan_kind="saa",
        settle="causal")
    add("无裕度(点预测)      + 因果结算", forecast_kind="mate", plan_kind="point",
        settle="causal")

    print()
    print("=" * 118)
    print("B. 换成“同星期几”预测（本文原依据），其余不变")
    print("=" * 118)
    for q in (0.8, 0.85):
        add(f"报童裕度 q={q}   + 因果结算", forecast_kind="ours", plan_kind="nv",
            q=q, settle="causal")
    add("两阶段随机规划 SAA + 因果结算", forecast_kind="ours", plan_kind="saa",
        settle="causal")

    print()
    print("=" * 118)
    print("C. 结算口径的影响（不可实现的事后最优下界，仅供对照）")
    print("=" * 118)
    add("报童裕度 q=0.7  + 事后最优再调度", forecast_kind="mate", plan_kind="nv",
        q=0.7, settle="recour")
    add("两阶段随机规划 SAA + 事后最优再调度", forecast_kind="mate", plan_kind="saa",
        settle="recour")
