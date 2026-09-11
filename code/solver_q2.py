"""
问题 2 求解器（最终方案）。

结构：每天 0:00 制定计划 -> 当日实时结算 -> 储电量逐日滚动。

日前计划依据（见 method_q2.py）
    负载：周五/周六 vs 其余，二分类后取最近 K=4 个同类日平均；
    光伏：最近 K=4 个连续日平均；
    裕度：净需求残差的 q 分位（报童），q 在 1 月标定、2–12 月外样本评估。

实时结算
    因果（online）控制规则：只用当前实测净负载与当前储电量，先放电兜底、
    再充电吸纳，仍兜不住的缺口按 5 倍电价紧急购电——完全可实行。

对照方案
    method="saa"：日前改用两阶段随机规划（情景 = 最近 m 天误差曲线）；
    settle="recour"：用当日全部实测做最优再调度（不可实行，仅作下界参考）。
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATA_PROCESSED, SLOTS_PER_DAY, HOURS_PER_SLOT, STORAGE_INITIAL_SOC, KEY_DATES,
)
from forecast_q2 import load_historical_matrices
from daily_solver import build_and_solve
from stochastic_q2 import build_scenarios, solve_stochastic_day, solve_recourse_day
from method_q2 import (
    make_class_array, forecast_mate, compute_residuals_mate,
    quantile_margin, causal_dispatch,
)


# ----------------------------------------------------------------------------
# 单年模拟
# ----------------------------------------------------------------------------

def simulate_year(price, dates, load_mat, pv_mat, net_mat, err, cls_arr, d2i,
                  start_date, end_date, e0,
                  q=None, method="nv", settle="causal", m_scenarios=10):
    """
    从 start_date 到 end_date 逐日滚动，返回 (result_df, summary_df)。

    q        : 报童分位（method="nv" 时使用）
    method   : 'nv'（报童裕度）| 'saa'（两阶段随机规划）| 'point'（无裕度）
    settle   : 'causal'（可实行）| 'recour'（不可实行下界）
    """
    tds = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    records, daily = [], []

    for td in tds:
        i = d2i[td]
        load_a, pv_a = load_mat[i], pv_mat[i]
        net_a = net_mat[i]

        # ---- 日前预测与计划 ----
        load_f, pv_f = forecast_mate(i, dates, load_mat, pv_mat, cls_arr)
        net_f = load_f - pv_f

        if method == "saa":
            try:
                idx, probs = build_scenarios(i, err, dates, m=m_scenarios)
                P_plan, _, _ = solve_stochastic_day(
                    price, net_f, err[idx], probs, e0=e0, verbose=False)
            except ValueError:
                # 预热期历史误差不足，退化为点预测
                sol = build_and_solve(price, load_f, pv_f, e0=e0,
                                      cyclic=False, allow_emergency=False, verbose=False)
                P_plan = sol["P_plan_kW"]
        else:
            b = quantile_margin(err, i, q) if method == "nv" else np.zeros(SLOTS_PER_DAY)
            sol = build_and_solve(price, load_f + b, pv_f, e0=e0,
                                  cyclic=False, allow_emergency=False, verbose=False)
            P_plan = sol["P_plan_kW"]

        # ---- 实时结算 ----
        if settle == "causal":
            P_ch, P_dis, P_em, E, P_curt = causal_dispatch(net_a, P_plan, e0)
        else:
            s0 = solve_recourse_day(price, net_a, P_plan, e0=e0, verbose=False)
            P_ch, P_dis, P_em, E, P_curt = (s0["P_ch_kW"], s0["P_dis_kW"],
                                            s0["P_em_kW"], s0["E_kWh"], s0["P_curt_kW"])

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
        daily.append({
            "date": td, "e0": e0, "e_end": E[-1],
            "plan_cost": plan_cost, "em_cost": em_cost,
            "total_cost": plan_cost + em_cost,
            "plan_kWh": float(P_plan.sum() * HOURS_PER_SLOT),
            "em_kWh": float(P_em.sum() * HOURS_PER_SLOT),
            "curt_kWh": float(P_curt.sum() * HOURS_PER_SLOT),
        })
        e0 = E[-1]

    return pd.DataFrame(records), pd.DataFrame(daily)


# ----------------------------------------------------------------------------
# 报童分位标定
# ----------------------------------------------------------------------------

def calibrate_q(dates, load_mat, pv_mat, net_mat, err, cls_arr, d2i, price,
                cal_start="2025-01-08", cal_end="2025-01-31",
                q_grid=(0.5, 0.6, 0.7, 0.8, 0.9), e0=STORAGE_INITIAL_SOC,
                method="nv", settle="causal", verbose=True):
    """在标定窗口上网格搜索使全年（窗口内）总费用最低的报童分位 q。"""
    best_q, best_cost, table = None, np.inf, []
    for q in q_grid:
        _, s = simulate_year(price, dates, load_mat, pv_mat, net_mat, err,
                             cls_arr, d2i, cal_start, cal_end, e0,
                             q=q, method=method, settle=settle)
        cost = s["total_cost"].sum()
        table.append((q, s["plan_cost"].sum(), s["em_cost"].sum(), cost))
        if verbose:
            print(f"    q={q:.2f}  计划 {s['plan_cost'].sum():>10,.0f}  "
                  f"紧急 {s['em_cost'].sum():>10,.0f}  总 {cost:>11,.0f} 元")
        if cost < best_cost:
            best_q, best_cost = q, cost
    return best_q, pd.DataFrame(table, columns=["q", "plan_cost", "em_cost", "total_cost"])


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def solve_problem2(start_date="2025-02-01", end_date="2025-12-31",
                   method="nv", settle="causal", q=None,
                   warm_start="2025-01-01", m_scenarios=10,
                   verbose: bool = True, save_dir: Path = DATA_PROCESSED):
    """
    求解问题 2。

    q=None 时先在 1 月标定报童分位（1.8–1.31），再以标定值跑全年（1.1 起预热），
    输出 start_date ~ end_date 的结果（默认 2025-02-01 至 2025-12-31）。
    """
    dates, load_mat, pv_mat, price_mat = load_historical_matrices()
    net_mat = load_mat - pv_mat
    cls_arr = make_class_array(dates)
    d2i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    err = compute_residuals_mate(dates, load_mat, pv_mat, net_mat, cls_arr)
    price = price_mat[d2i[pd.Timestamp(warm_start)]]

    if method == "nv" and q is None:
        if verbose:
            print("在 2025-01-08 ~ 01-31 上标定报童分位 q：")
        q, table = calibrate_q(dates, load_mat, pv_mat, net_mat, err, cls_arr,
                               d2i, price, method=method, settle=settle)
        if verbose:
            print(f"  -> 标定结果 q* = {q}")

    # 从 warm_start 连续跑到 end_date，保证储电量链条完整
    result_df, summary_df = simulate_year(
        price, dates, load_mat, pv_mat, net_mat, err, cls_arr, d2i,
        warm_start, end_date, STORAGE_INITIAL_SOC,
        q=q, method=method, settle=settle, m_scenarios=m_scenarios)

    # 修剪到输出窗口
    keep = pd.Timestamp(start_date)
    result_df = result_df[result_df["date"] >= keep].reset_index(drop=True)
    summary_df = summary_df[summary_df["date"] >= keep].reset_index(drop=True)

    if verbose:
        print(f"\n输出窗口 {start_date} ~ {end_date}（{len(summary_df)} 天）")
        print(f"  计划购电费 {summary_df['plan_cost'].sum():,.2f} 元")
        print(f"  紧急购电费 {summary_df['em_cost'].sum():,.2f} 元")
        print(f"  总费用     {summary_df['total_cost'].sum():,.2f} 元")
        print(f"  紧急购电量 {summary_df['em_kWh'].sum():,.2f} kWh")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_dir / "result2_details.csv", index=False)
        summary_df.to_csv(save_dir / "result2_summary.csv", index=False)

    return result_df, summary_df


def get_key_date_results(result_df, summary_df, key_dates=KEY_DATES):
    """提取 4 个指定日期的逐时段结果与日汇总。"""
    out = {}
    for d in key_dates:
        sub = result_df[result_df["date"] == pd.Timestamp(d)]
        if sub.empty:
            print(f"[警告] 指定日期 {d} 不在结果中")
            continue
        out[d] = {"df": sub.reset_index(drop=True),
                  "summary": summary_df[summary_df["date"] == pd.Timestamp(d)].iloc[0].to_dict()}
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    solve_problem2(verbose=True, save_dir=DATA_PROCESSED)
    print(f"\n耗时 {time.time() - t0:.1f} s")
