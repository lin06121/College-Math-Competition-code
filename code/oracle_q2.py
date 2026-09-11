"""
问题 2 的预言机（oracle）信息价值分解：

1. 采用方案：负载/光伏均按 method_q2 预测；
2. PV 完美：光伏用实测值，负载仍用预测；
3. Load 完美：负载用实测值，光伏仍用预测；
4. 完全信息：负载与光伏均用实测值。

日前计划均加报童裕度 q=0.8，实时用因果结算，统一口径以量化信息价值。
"""
import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED, HOURS_PER_SLOT, STORAGE_INITIAL_SOC,
)
from forecast_q2 import load_historical_matrices
from daily_solver import build_and_solve
from method_q2 import (
    make_class_array, forecast_mate, compute_residuals_mate,
    quantile_margin, causal_dispatch,
)


def run_oracle(load_src, pv_src, q=0.8, start_date="2025-02-01", end_date="2025-12-31"):
    """
    load_src/pv_src : 'pred' | 'actual'
    返回 (plan_cost, em_cost, total_cost, em_kWh)
    """
    dates, load_mat, pv_mat, price_mat = load_historical_matrices()
    net_mat = load_mat - pv_mat
    cls_arr = make_class_array(dates)
    d2i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    tds = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    price = price_mat[d2i[pd.Timestamp(start_date)]]
    err = compute_residuals_mate(dates, load_mat, pv_mat, net_mat, cls_arr)

    e0 = STORAGE_INITIAL_SOC
    plan_cost = em_cost = em_kWh = plan_kWh = 0.0

    for td in tds:
        i = d2i[td]
        load_f, pv_f = forecast_mate(i, dates, load_mat, pv_mat, cls_arr)

        if load_src == "actual":
            load_plan = load_mat[i]            # 完美负载
        else:
            load_plan = load_f                 # 预测负载

        if pv_src == "actual":
            pv_plan = pv_mat[i]                # 完美光伏
        else:
            pv_plan = pv_f                     # 预测光伏

        net_plan = load_plan - pv_plan
        b = quantile_margin(err, i, q)
        sol = build_and_solve(price, load_plan + b, pv_plan, e0=e0,
                              cyclic=False, allow_emergency=False, verbose=False)
        P_plan = sol["P_plan_kW"]

        # 结算始终面对实际值
        net_actual = net_mat[i]
        P_ch, P_dis, P_em, E, P_curt = causal_dispatch(net_actual, P_plan, e0)

        plan_cost += float((P_plan * price).sum() * HOURS_PER_SLOT)
        em_cost += float((P_em * price).sum() * HOURS_PER_SLOT * 5.0)
        em_kWh += float(P_em.sum() * HOURS_PER_SLOT)
        plan_kWh += float(P_plan.sum() * HOURS_PER_SLOT)
        e0 = E[-1]

    return dict(plan_cost=plan_cost, em_cost=em_cost,
                total_cost=plan_cost + em_cost, em_kWh=em_kWh, plan_kWh=plan_kWh)


if __name__ == "__main__":
    cases = [
        ("采用方案（预测负载 + 预测光伏）", "pred", "pred"),
        ("PV 完美（预测负载 + 实测光伏）", "pred", "actual"),
        ("Load 完美（实测负载 + 预测光伏）", "actual", "pred"),
        ("完全信息（实测负载 + 实测光伏）", "actual", "actual"),
    ]
    results = []
    for tag, ls, ps in cases:
        r = run_oracle(ls, ps)
        results.append((tag, r))
        print(f"{tag:<32} 计划 {r['plan_cost']:>12,.0f}  紧急 {r['em_cost']:>10,.0f}  "
              f"总 {r['total_cost']:>12,.0f}  紧急电量 {r['em_kWh']:>9,.0f} kWh")

    baseline = results[0][1]["total_cost"]
    pv_only = results[1][1]["total_cost"]
    load_only = results[2][1]["total_cost"]
    full_info = results[3][1]["total_cost"]
    print()
    print(f"距完全信息下界: {baseline - full_info:,.0f} 元 ({100*(baseline-full_info)/baseline:.2f}%)")
    print(f"  其中光伏噪声价值: {baseline - pv_only:,.0f} 元 ({100*(baseline-pv_only)/baseline:.2f}%)")
    print(f"       负载噪声价值: {baseline - load_only:,.0f} 元 ({100*(baseline-load_only)/baseline:.2f}%)")
    print(f"       交互项:        {(baseline - full_info) - (baseline - pv_only) - (baseline - load_only):,.0f} 元")
