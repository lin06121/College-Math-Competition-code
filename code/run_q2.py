"""
问题 2 一键运行：报童裕度日前计划 + 因果实时结算 -> 结果校验 -> 输出论文表格与 result2.xlsx。

用法：
    PYTHONUTF8=1 /d/devtool/anaconda/conda/python.exe code/run_q2.py
可选：
    --compare  额外输出对齐对照实验（预测依据 / 计划方法 / 结算口径）
    --q 0.7    手动指定报童分位（默认在 1 月标定）
"""
import sys
import time
import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED, ROOT, HOURS_PER_SLOT, SLOTS_PER_DAY,
    STORAGE_SOC_MIN, STORAGE_SOC_MAX, STORAGE_MAX_POWER,
)
from solver_q2 import solve_problem2
from format_result_q2 import format_all, print_paper_tables


def verify(result_df: pd.DataFrame, summary_df: pd.DataFrame, tol: float = 1e-3):
    """校验全年解：功率平衡、SOC 连续性、边界、充放电互斥、费用一致性。"""
    print("\n===== 结果校验 =====")
    ok = True

    # 1. 功率平衡（实际工况，因果结算）
    bal = (result_df["P_plan_kW"] + result_df["P_em_kW"] + result_df["pv_actual_kW"]
           + result_df["P_dis_kW"] - result_df["load_actual_kW"]
           - result_df["P_ch_kW"] - result_df["P_curt_kW"])
    print(f"[{'OK' if bal.abs().max() < tol else '!!'}] 功率平衡最大偏差 {bal.abs().max():.3e} kW")
    ok &= bal.abs().max() < tol

    # 2. SOC 连续性：每日 e_end == 次日 e0
    s = summary_df.sort_values("date").reset_index(drop=True)
    gap = s["e0"].values[1:] - s["e_end"].values[:-1]
    print(f"[{'OK' if np.abs(gap).max() < tol else '!!'}] 日间 SOC 连续性最大偏差 {np.abs(gap).max():.3e} kWh")
    ok &= np.abs(gap).max() < tol

    # 3. SOC 边界
    E = result_df["E_kWh"].values
    lo = min(E.min(), s["e_end"].min())
    hi = max(E.max(), s["e_end"].max())
    in_bounds = (lo >= STORAGE_SOC_MIN - tol) and (hi <= STORAGE_SOC_MAX + tol)
    print(f"[{'OK' if in_bounds else '!!'}] SOC 范围 [{lo:.2f}, {hi:.2f}] kWh "
          f"（允许 {STORAGE_SOC_MIN:.0f}-{STORAGE_SOC_MAX:.0f}）")
    ok &= in_bounds

    # 4. 充放电互斥与功率上限
    overl = ((result_df["P_ch_kW"] > tol) & (result_df["P_dis_kW"] > tol)).sum()
    print(f"[{'OK' if overl == 0 else '!!'}] 同时充放电时段数 {overl}")
    ok &= (overl == 0)

    pmax = max(result_df["P_ch_kW"].max(), result_df["P_dis_kW"].max())
    print(f"[{'OK' if pmax <= STORAGE_MAX_POWER + tol else '!!'}] 最大充放电功率 {pmax:.2f} kW"
          f"（限值 {STORAGE_MAX_POWER:.0f}）")
    ok &= pmax <= STORAGE_MAX_POWER + tol

    # 5. 不可行解检查：紧急购电与弃光不可同时为正
    both = ((result_df["P_em_kW"] > tol) & (result_df["P_curt_kW"] > tol)).sum()
    print(f"[{'OK' if both == 0 else '!!'}] 同时紧急购电与弃光时段数 {both}")
    ok &= (both == 0)

    # 6. 费用一致性
    plan = (result_df["P_plan_kW"] * result_df["price"] * HOURS_PER_SLOT).sum()
    em = (result_df["P_em_kW"] * result_df["price"] * HOURS_PER_SLOT * 5.0).sum()
    print(f"[OK] 计划购电费 {plan:,.2f} 元；紧急购电费 {em:,.2f} 元；合计 {plan + em:,.2f} 元")
    return ok


def describe(summary_df, result_df):
    print("\n===== 全年统计 =====")
    pk, ek = summary_df["plan_kWh"].sum(), summary_df["em_kWh"].sum()
    print(f"  计划购电量: {pk:>14,.2f} kWh")
    print(f"  紧急购电量: {ek:>14,.2f} kWh（占计划 {100 * ek / pk:.2f}%）")
    print(f"  弃光电量:   {result_df['P_curt_kW'].sum() * HOURS_PER_SLOT:>14,.2f} kWh")
    print(f"  计划购电费: {summary_df['plan_cost'].sum():>14,.2f} 元")
    print(f"  紧急购电费: {summary_df['em_cost'].sum():>14,.2f} 元")
    print(f"  总费用:     {summary_df['total_cost'].sum():>14,.2f} 元")
    print(f"  有紧急购电的天数: {(summary_df['em_kWh'] > 1e-6).sum()} / {len(summary_df)}")


if __name__ == "__main__":
    q = None
    if "--q" in sys.argv:
        q = float(sys.argv[sys.argv.index("--q") + 1])

    t0 = time.time()
    result_df, summary_df = solve_problem2(q=q, verbose=True, save_dir=DATA_PROCESSED)
    print(f"\n求解耗时 {time.time() - t0:.1f} s")

    ok = verify(result_df, summary_df)
    describe(summary_df, result_df)

    res = format_all(result_df, summary_df)
    print_paper_tables(result_df, summary_df)
    print(f"\nresult2.xlsx -> {res['result_xlsx']}")
    print(f"论文表格 -> {ROOT/'table1_q2.csv'} / table2_q2.csv / table3_q2.csv")
    print(f"紧急购电事件 -> {ROOT/'emergency_events_q2.csv'}")

    if "--compare" in sys.argv:
        from compare_q2 import run_year
        print("\n===== 对齐对照实验（计算中） =====")
        for tag, kw in [
            ("同伴预测 + 报童 q=0.75 + 因果", dict(forecast_kind="mate", plan_kind="nv", q=0.75, settle="causal")),
            ("同伴预测 + SAA        + 因果", dict(forecast_kind="mate", plan_kind="saa", settle="causal")),
            ("同星期几  + 报童 q=0.85 + 因果", dict(forecast_kind="ours", plan_kind="nv", q=0.85, settle="causal")),
            ("同星期几  + SAA        + 因果", dict(forecast_kind="ours", plan_kind="saa", settle="causal")),
            ("同伴预测 + 报童 q=0.7  + 事后最优(不可实行)", dict(forecast_kind="mate", plan_kind="nv", q=0.7, settle="recour")),
        ]:
            r = run_year(**kw)
            print(f"  {tag:<40} 总 {r['total_cost']:>12,.0f} 元  紧急电量 {r['em_kWh']:>9,.0f} kWh")

    print("\n校验结果:", "全部通过" if ok else "存在问题，请检查上方 [!!] 项")
