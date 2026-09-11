"""
问题 2 求解器：全年逐日两阶段随机规划。

流程（对 2025-02-01 至 2025-12-31 的每一天）：

    1. 0:00 用附件 2 中当天之前的历史数据（同星期几平均）预测当日负载与光伏；
    2. 用最近 m 个历史预测误差曲线构造净负载情景；
    3. 求解两阶段随机规划 -> 第一阶段决策：计划购电量 P_plan(t)；
    4. 当天实际负载/光伏实现后，求解实时 recourse ->
       最优储能充放电 P_ch、P_dis、紧急购电 P_em、终端 SOC；
    5. 终端 SOC 作为次日初始 SOC，逐日滚动。

同时保留确定性方案（P_plan 只按预测值优化、储能严格按计划执行）作为对照。
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_PROCESSED,
    SLOTS_PER_DAY,
    HOURS_PER_SLOT,
    STORAGE_INITIAL_SOC,
    KEY_DATES,
)
from forecast_q2 import load_historical_matrices, forecast_load_pv
from daily_solver import build_and_solve
from stochastic_q2 import (
    compute_net_forecast_errors,
    build_scenarios,
    solve_stochastic_day,
    solve_recourse_day,
)


# ----------------------------------------------------------------------------
# 确定性对照方案
# ----------------------------------------------------------------------------

def compute_emergency(P_plan, P_ch, P_dis, load_actual, pv_actual):
    """严格按计划执行时，由实际负载/光伏偏差导致的紧急购电功率。"""
    shortfall = load_actual + P_ch - P_plan - pv_actual - P_dis
    return np.maximum(shortfall, 0.0)


def solve_problem2_deterministic(start_date="2025-02-01", end_date="2025-12-31",
                                 forecast_k: int = 4, realtime: bool = False,
                                 save_dir: Path = None):
    """
    确定性对照方案：日前只按点预测优化 P_plan（不含 5 倍应急风险），
    实时阶段可选
        realtime=False：储能严格按计划充放电（只改紧急购电）；
        realtime=True ：储能按实际负载/光伏实时最优调度（recourse）。
    """
    dates, load_mat, pv_mat, price_mat = load_historical_matrices()
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    target_dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")

    price = price_mat[date_to_idx[pd.Timestamp(start_date)]]
    e0 = STORAGE_INITIAL_SOC
    records, daily = [], []

    for td in target_dates:
        idx = date_to_idx[td]
        load_a, pv_a = load_mat[idx], pv_mat[idx]
        net_a = load_a - pv_a
        load_f, pv_f = forecast_load_pv(td, dates, load_mat, pv_mat, k=forecast_k)

        sol = build_and_solve(price, load_f, pv_f, e0=e0,
                              cyclic=False, allow_emergency=False, verbose=False)
        P_plan = sol["P_plan_kW"]

        if realtime:
            s0 = solve_recourse_day(price, net_a, P_plan, e0=e0, verbose=False)
            P_ch, P_dis, P_em, E = (s0["P_ch_kW"], s0["P_dis_kW"],
                                    s0["P_em_kW"], s0["E_kWh"])
            P_curt = s0["P_curt_kW"]
        else:
            P_ch, P_dis = sol["P_ch_kW"], sol["P_dis_kW"]
            P_curt = sol["P_curt_kW"]
            E = sol["E_kWh"]
            P_em = compute_emergency(P_plan, P_ch, P_dis, load_a, pv_a)

        for t in range(SLOTS_PER_DAY):
            records.append({
                "date": td, "slot": t,
                "P_plan_kW": P_plan[t], "P_em_kW": P_em[t],
                "P_ch_kW": P_ch[t], "P_dis_kW": P_dis[t],
                "E_kWh": E[t], "P_curt_kW": P_curt[t],
                "load_forecast_kW": load_f[t], "pv_forecast_kW": pv_f[t],
                "load_actual_kW": load_a[t], "pv_actual_kW": pv_a[t],
                "price": price[t],
            })

        plan_cost = float((P_plan * price).sum() * HOURS_PER_SLOT)
        em_cost = float((P_em * price).sum() * HOURS_PER_SLOT * 5.0)
        daily.append({"date": td, "e0": e0, "e_end": E[-1],
                      "plan_cost": plan_cost, "em_cost": em_cost,
                      "total_cost": plan_cost + em_cost,
                      "plan_kWh": float(P_plan.sum() * HOURS_PER_SLOT),
                      "em_kWh": float(P_em.sum() * HOURS_PER_SLOT)})
        e0 = E[-1]

    result_df, summary_df = pd.DataFrame(records), pd.DataFrame(daily)
    if save_dir is not None:
        save_dir = Path(save_dir)
        tag = "det_rt" if realtime else "det"
        result_df.to_csv(save_dir / f"result2_{tag}_details.csv", index=False)
        summary_df.to_csv(save_dir / f"result2_{tag}_summary.csv", index=False)
    return result_df, summary_df


# ----------------------------------------------------------------------------
# 两阶段随机规划主求解
# ----------------------------------------------------------------------------

def solve_problem2(start_date="2025-02-01", end_date="2025-12-31",
                   forecast_k: int = 4, m_scenarios: int = 10,
                   verbose: bool = True, save_dir: Path = DATA_PROCESSED):
    """
    求解问题 2：全年逐日两阶段随机规划。

    返回
    ----------
    result_df : 逐时段明细（含计划购电、实际充放电、紧急购电、SOC 等）
    summary_df : 逐日汇总
    """
    dates, load_mat, pv_mat, price_mat = load_historical_matrices()
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    target_dates = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")

    # 净负载 = 负载 - 光伏；历史误差用于构造情景
    net_mat = load_mat - pv_mat
    _, net_err = compute_net_forecast_errors(dates, net_mat, k=forecast_k)

    price = price_mat[date_to_idx[pd.Timestamp(start_date)]]
    e0 = STORAGE_INITIAL_SOC
    records, daily = [], []

    for k_day, td in enumerate(target_dates):
        idx = date_to_idx[td]
        load_a, pv_a = load_mat[idx], pv_mat[idx]
        net_a = net_mat[idx]

        # 1) 日前预测（仅用当天之前的历史）
        load_f, pv_f = forecast_load_pv(td, dates, load_mat, pv_mat, k=forecast_k)
        net_f = load_f - pv_f

        # 2) 情景：最近 m 个历史预测误差
        scen_idx, probs = build_scenarios(idx, net_err, dates, m=m_scenarios)

        # 3) 两阶段随机规划 -> 计划购电量
        P_plan, _, obj = solve_stochastic_day(
            price, net_f, net_err[scen_idx], probs, e0=e0, verbose=False)

        # 4) 实时 recourse -> 实际运行
        s0 = solve_recourse_day(price, net_a, P_plan, e0=e0, verbose=False)
        P_ch, P_dis, P_em, E, em_cost = (
            s0["P_ch_kW"], s0["P_dis_kW"], s0["P_em_kW"], s0["E_kWh"], s0["em_cost"])
        P_curt = s0["P_curt_kW"]

        for t in range(SLOTS_PER_DAY):
            records.append({
                "date": td, "slot": t,
                "P_plan_kW": P_plan[t], "P_em_kW": P_em[t],
                "P_ch_kW": P_ch[t], "P_dis_kW": P_dis[t],
                "E_kWh": E[t], "P_curt_kW": P_curt[t],
                "load_forecast_kW": load_f[t], "pv_forecast_kW": pv_f[t],
                "load_actual_kW": load_a[t], "pv_actual_kW": pv_a[t],
                "price": price[t],
            })

        plan_cost = float((P_plan * price).sum() * HOURS_PER_SLOT)
        daily.append({
            "date": td, "e0": e0, "e_end": E[-1],
            "plan_cost": plan_cost, "em_cost": em_cost,
            "total_cost": plan_cost + em_cost,
            "plan_kWh": float(P_plan.sum() * HOURS_PER_SLOT),
            "em_kWh": float(P_em.sum() * HOURS_PER_SLOT),
            "stochastic_obj": obj,
        })
        e0 = E[-1]

        if verbose and (k_day % 30 == 0 or td == target_dates[-1]):
            print(f"  {td.date()}: 计划 {plan_cost:9.2f} 元, "
                  f"紧急 {em_cost:8.2f} 元, 紧急电量 {daily[-1]['em_kWh']:7.2f} kWh, "
                  f"e_end={e0:.1f}")

    result_df, summary_df = pd.DataFrame(records), pd.DataFrame(daily)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_dir / "result2_details.csv", index=False)
        summary_df.to_csv(save_dir / "result2_summary.csv", index=False)
    return result_df, summary_df


def get_key_date_results(result_df: pd.DataFrame, summary_df: pd.DataFrame):
    """提取 4 个指定日期的逐时段结果与日汇总。"""
    key_results = {}
    for d_str in KEY_DATES:
        d = pd.Timestamp(d_str)
        sub = result_df[result_df["date"] == d]
        if sub.empty:
            print(f"[警告] 指定日期 {d_str} 不在结果中")
            continue
        key_results[d_str] = {
            "df": sub.reset_index(drop=True),
            "summary": summary_df[summary_df["date"] == d].iloc[0].to_dict(),
        }
    return key_results


if __name__ == "__main__":
    import time
    t0 = time.time()
    result_df, summary_df = solve_problem2(verbose=True, save_dir=DATA_PROCESSED)
    print(f"\n全年求解完成，耗时 {time.time() - t0:.1f} s")
    print("全年统计:")
    print(f"  计划购电费: {summary_df['plan_cost'].sum():,.2f} 元")
    print(f"  紧急购电费: {summary_df['em_cost'].sum():,.2f} 元")
    print(f"  总费用:     {summary_df['total_cost'].sum():,.2f} 元")
    print(f"  计划购电量: {summary_df['plan_kWh'].sum():,.2f} kWh")
    print(f"  紧急购电量: {summary_df['em_kWh'].sum():,.2f} kWh")
