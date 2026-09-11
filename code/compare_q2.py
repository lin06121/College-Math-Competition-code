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
    make_class_array, forecast_mate, compute_residuals_mate,
    quantile_margin, causal_dispatch,
)


def run_year(forecast_kind, plan_kind, settle="causal",
             q=0.7, m_scenarios=10, k_forecast=4,
             start_date="2025-02-01", end_date="2025-12-31", verbose=False):
    """
    跑一整年，返回汇总 dict。

    forecast_kind : 'mate' | 'ours'
    plan_kind     : 'nv' | 'saa' | 'point'
    settle        : 'causal'（可实行）| 'recour'（不可实行下界）
    """
    dates, load, pv, price_mat = load_historical_matrices()
    net = load - pv
    d2i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    tds = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    price = price_mat[d2i[pd.Timestamp(start_date)]]

    cls_arr = make_class_array(dates)
    if forecast_kind == "ours":
        _, err = compute_net_forecast_errors(dates, net, k=k_forecast)
    else:
        err = compute_residuals_mate(dates, load, pv, net, cls_arr)

    def forecast(i, td):
        if forecast_kind == "ours":
            return forecast_load_pv(td, dates, load, pv, k=k_forecast)
        return forecast_mate(i, dates, load, pv, cls_arr)

    e0 = STORAGE_INITIAL_SOC
    r = dict(plan_cost=0.0, em_cost=0.0, em_kWh=0.0, plan_kWh=0.0, curt_kWh=0.0)

    for td in tds:
        i = d2i[td]
        load_f, pv_f = forecast(i, td)
        net_f = load_f - pv_f
        net_a = net[i]

        # ---- 日前计划 ----
        if plan_kind == "saa":
            idx, probs = build_scenarios(i, err, dates, m=m_scenarios)
            P_plan, _, _ = solve_stochastic_day(
                price, net_f, err[idx], probs, e0=e0, verbose=False)
        else:
            b = quantile_margin(err, i, q) if plan_kind == "nv" else np.zeros(SLOTS_PER_DAY)
            sol = build_and_solve(price, load_f + b, pv_f, e0=e0,
                                  cyclic=False, allow_emergency=False, verbose=False)
            P_plan = sol["P_plan_kW"]

        # ---- 结算 ----
        if settle == "causal":
            P_ch, P_dis, P_em, E, P_curt = causal_dispatch(net_a, P_plan, e0)
        else:
            from stochastic_q2 import solve_recourse_day
            s0 = solve_recourse_day(price, net_a, P_plan, e0=e0, verbose=False)
            P_ch, P_dis, P_em, E, P_curt = (s0["P_ch_kW"], s0["P_dis_kW"],
                                            s0["P_em_kW"], s0["E_kWh"], s0["P_curt_kW"])

        r["plan_cost"] += float((P_plan * price).sum() * HOURS_PER_SLOT)
        r["em_cost"] += float((P_em * price).sum() * HOURS_PER_SLOT * 5.0)
        r["em_kWh"] += float(P_em.sum() * HOURS_PER_SLOT)
        r["plan_kWh"] += float(P_plan.sum() * HOURS_PER_SLOT)
        r["curt_kWh"] += float(P_curt.sum() * HOURS_PER_SLOT)
        e0 = E[-1]

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
