"""
问题 2 一键运行：两阶段随机规划求解 -> 结果校验 -> 输出论文表格与 result2.xlsx。

用法：
    PYTHONUTF8=1 /d/devtool/anaconda/conda/python.exe code/run_q2.py
可选：
    在 code/ 目录下执行 `python run_q2.py --compare` 额外计算三种策略的对照。
"""
import sys
import time
import numpy as np
import pandas as pd

from config import DATA_PROCESSED, ROOT, HOURS_PER_SLOT, SLOTS_PER_DAY, STORAGE_INITIAL_SOC
from solver_q2 import solve_problem2, solve_problem2_deterministic
from format_result_q2 import format_all, print_paper_tables


def verify(result_df: pd.DataFrame, summary_df: pd.DataFrame, tol: float = 1e-3):
    """校验全年解：功率平衡、SOC 连续性、边界、充放电互斥、费用一致性。"""
    print("\n===== 结果校验 =====")
    ok = True

    # 1. 功率平衡（实际工况）
    bal = (result_df["P_plan_kW"] + result_df["P_em_kW"] + result_df["pv_actual_kW"]
           + result_df["P_dis_kW"] - result_df["load_actual_kW"]
           - result_df["P_ch_kW"] - result_df["P_curt_kW"])
    print(f"[{'OK' if bal.abs().max() < tol else '!!'}] 功率平衡最大偏差 {bal.abs().max():.3e} kW")
    ok &= bal.abs().max() < tol

    # 2. SOC 连续性：每日 e_end == 次日 e0
    s = summary_df.sort_values("date").reset_index(drop=True)
    gap = (s["e0"].values[1:] - s["e_end"].values[:-1])
    print(f"[{'OK' if np.abs(gap).max() < tol else '!!'}] 日间 SOC 连续性最大偏差 {np.abs(gap).max():.3e} kWh")
    ok &= np.abs(gap).max() < tol
    print(f"[{'OK' if abs(s['e0'].iloc[0] - STORAGE_INITIAL_SOC) < tol else '!!'}] "
          f"起始 SOC {s['e0'].iloc[0]:.2f} kWh（应为 6000）")

    # 3. SOC 边界与充放电互斥
    E = result_df["E_kWh"].values
    in_bounds = (E.min() >= 1200 - tol) and (E.max() <= 10800 + tol) and \
                (s["e_end"].min() >= 1200 - tol) and (s["e_end"].max() <= 10800 + tol)
    print(f"[{'OK' if in_bounds else '!!'}] SOC 范围 [{E.min():.2f}, {max(E.max(), s['e_end'].max()):.2f}] kWh")
    ok &= in_bounds

    overl = ((result_df["P_ch_kW"] > tol) & (result_df["P_dis_kW"] > tol)).sum()
    print(f"[{'OK' if overl == 0 else '!!'}] 同时充放电时段数 {overl}")
    ok &= (overl == 0)

    pmax = max(result_df["P_ch_kW"].max(), result_df["P_dis_kW"].max())
    print(f"[{'OK' if pmax <= 5000 + tol else '!!'}] 最大充放电功率 {pmax:.2f} kW（限值 5000）")
    ok &= pmax <= 5000 + tol

    # 4. 费用一致性
    plan = (result_df["P_plan_kW"] * result_df["price"] * HOURS_PER_SLOT).sum()
    em = (result_df["P_em_kW"] * result_df["price"] * HOURS_PER_SLOT * 5.0).sum()
    print(f"[OK] 全年计划购电费 {plan:,.2f} 元；紧急购电费 {em:,.2f} 元；"
          f"合计 {plan + em:,.2f} 元")
    return ok


def describe(result_df, summary_df):
    print("\n===== 全年统计 =====")
    print(f"  计划购电量: {summary_df['plan_kWh'].sum():>14,.2f} kWh")
    print(f"  紧急购电量: {summary_df['em_kWh'].sum():>14,.2f} kWh "
          f"（占计划购电量 {100 * summary_df['em_kWh'].sum() / summary_df['plan_kWh'].sum():.2f}%）")
    print(f"  计划购电费: {summary_df['plan_cost'].sum():>14,.2f} 元")
    print(f"  紧急购电费: {summary_df['em_cost'].sum():>14,.2f} 元")
    print(f"  总费用:     {summary_df['total_cost'].sum():>14,.2f} 元")
    print(f"  有紧急购电的天数: {(summary_df['em_kWh'] > 1e-6).sum()} / {len(summary_df)}")


def compare():
    """三种策略对照：确定性计划+严格执行 / 确定性计划+实时调整 / 随机规划+实时调整。"""
    print("\n===== 策略对照（计算中，约 2 分钟） =====")
    rows = []
    for label, kw in [("确定性计划 + 储能严格按计划执行", dict(realtime=False)),
                      ("确定性计划 + 实时调整储能", dict(realtime=True))]:
        _, s = solve_problem2_deterministic(save_dir=None, **kw)
        rows.append((label, s))

    _, s_main = solve_problem2(verbose=False, save_dir=None)
    rows.append(("两阶段随机规划 + 实时调整储能", s_main))

    print(f"  {'策略':<32}{'计划费(元)':>14}{'紧急费(元)':>14}{'总费用(元)':>14}{'紧急电量(kWh)':>14}")
    for label, s in rows:
        print(f"  {label:<30}{s['plan_cost'].sum():>14,.0f}{s['em_cost'].sum():>14,.0f}"
              f"{s['total_cost'].sum():>14,.0f}{s['em_kWh'].sum():>14,.0f}")
    return rows


if __name__ == "__main__":
    t0 = time.time()
    result_df, summary_df = solve_problem2(verbose=True, save_dir=DATA_PROCESSED)
    print(f"\n求解耗时 {time.time() - t0:.1f} s")

    ok = verify(result_df, summary_df)
    describe(result_df, summary_df)

    res = format_all(result_df, summary_df)
    print_paper_tables(result_df, summary_df)
    print(f"\nresult2.xlsx -> {res['result_xlsx']}")
    print(f"论文表格 -> {ROOT/'table1_q2.csv'} / table2_q2.csv / table3_q2.csv")
    print(f"紧急购电事件 -> {ROOT/'emergency_events_q2.csv'}")

    if "--compare" in sys.argv:
        compare()

    print("\n校验结果:", "全部通过" if ok else "存在问题，请检查上方 [!!] 项")
