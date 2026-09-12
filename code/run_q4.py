"""
问题 4 一键运行：波动电价下重算问题 2、3。

信息结构（用户确认）：0:00 制定计划时**不知道**当天电价，只能用电价预报；
实时结算按附件 4 实际电价。

计划电价 = 近 W 日逐时段滚动均值（因果）；结算电价 = 附件 4 实际。
附带"电价信息价值"对照：计划分别用 预报 / 实际 / 附件1典型，结算一律用实际。

用法：PYTHONUTF8=1 "$PY" run_q4.py
"""
import sys
import time
import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED, ROOT, RESULT_TEMPLATE_DIR, SLOTS_PER_DAY, HOURS_PER_SLOT,
    STORAGE_SOC_MIN, STORAGE_SOC_MAX, STORAGE_MAX_POWER, STORAGE_INITIAL_SOC,
)
import solver_q2
import solver_q3
from forecast_q4 import load_price_realtime, build_price_forecast, price_forecast_mae
from forecast_q2 import load_historical_matrices
from method_q2 import make_class_array, compute_residuals_mate
from format_result_q2 import (
    make_paper_table1_q2, make_paper_table2_q2, make_paper_table3_q2,
    make_emergency_events, write_result2_xlsx,
)
from format_result_q3 import (
    make_paper_table1_q3, make_paper_table2_q3, make_paper_table3_q3,
    write_result3_xlsx,
)


# ----------------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------------

def verify_q2(result_df, summary_df, tol=1e-3):
    ok = True
    bal = (result_df["P_plan_kW"] + result_df["P_em_kW"] + result_df["P_dis_kW"]
           - result_df["P_ch_kW"] - result_df["P_curt_kW"]
           - (result_df["load_actual_kW"] - result_df["pv_actual_kW"]))
    ok &= bal.abs().max() < tol
    d = summary_df["e0"].values[1:] - summary_df["e_end"].values[:-1]
    ok &= np.abs(d).max() < tol
    E = result_df["E_kWh"]
    ok &= E.min() >= STORAGE_SOC_MIN - tol and E.max() <= STORAGE_SOC_MAX + tol
    both = ((result_df["P_ch_kW"] > tol) & (result_df["P_dis_kW"] > tol)).sum()
    ok &= both == 0
    pm = max(result_df["P_ch_kW"].max(), result_df["P_dis_kW"].max())
    ok &= pm <= STORAGE_MAX_POWER + tol
    print(f"  Q4-2 校验：功率平衡 {bal.abs().max():.2e} kW, SOC 连续 {np.abs(d).max():.2e}, "
          f"互斥对 {both}, 功率上限 {pm:.1f} -> {'OK' if ok else 'FAIL'}")
    return ok


def verify_q3(result_df, summary_df, tol=1e-3):
    ok = True
    net = result_df["load_actual_kW"] - result_df["pv_actual_kW"]
    bal = (result_df["P_adj_kW"] + result_df["P_em_kW"] + result_df["P_dis_kW"]
           - result_df["P_ch_kW"] - result_df["P_curt_kW"] - net)
    ok &= bal.abs().max() < tol
    d = summary_df["e0"].values[1:] - summary_df["e_end"].values[:-1]
    ok &= np.abs(d).max() < tol
    E = result_df["E_kWh"]
    ok &= E.min() >= STORAGE_SOC_MIN - tol and E.max() <= STORAGE_SOC_MAX + tol
    ok &= ((result_df["P_ch_kW"] > tol) & (result_df["P_dis_kW"] > tol)).sum() == 0
    pm = max(result_df["P_ch_kW"].max(), result_df["P_dis_kW"].max())
    ok &= pm <= STORAGE_MAX_POWER + tol
    bad = 0
    for _, g in result_df.groupby("date"):
        g = g.sort_values("slot")
        if not np.allclose(g["P_adj_kW"].values[:36], g["P_plan_kW"].values[:36]):
            bad += 1
    ok &= bad == 0
    print(f"  Q4-3 校验：功率平衡 {bal.abs().max():.2e} kW, SOC 连续 {np.abs(d).max():.2e}, "
          f"P_adj[0:36]=P_plan 不满足天数 {bad} -> {'OK' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------
# 电价信息价值对照
# ----------------------------------------------------------------------------

def price_info_value_q2(q, verbose=True):
    """Q4-2：计划分别用 预报/实际/典型，结算都用实际，隔离电价知识的代价。"""
    print(f"\n===== Q4-2 电价信息价值（q={q}，结算价=实际） =====")
    rows = {}
    for plan in ["forecast", "actual", "typical"]:
        _, s = solver_q2.solve_problem4_2(q=q, price_plan=plan, verbose=False,
                                          save_dir=None)
        rows[plan] = (s["total_cost"].sum(), s["em_kWh"].sum())
        print(f"  {plan:<10} 总费用 {rows[plan][0]:>14,.0f}   紧急电量 {rows[plan][1]:>8,.0f} kWh")
    base, fc = rows["actual"][0], rows["forecast"][0]
    print(f"  电价未知代价（预报 vs 已知）: {fc - base:+,.0f} 元 ({(fc-base)/base:+.2%})")
    return pd.DataFrame([(k, v[0], v[1]) for k, v in rows.items()],
                        columns=["price_plan", "total_cost", "em_kWh"])


def price_info_value_q3(q, lam, verbose=True):
    print(f"\n===== Q4-3 电价信息价值（q={q}, λ={lam}，结算价=实际） =====")
    rows = {}
    for plan in ["forecast", "actual", "typical"]:
        _, s = solver_q3.solve_problem4_3(q=q, term_value=lam, price_plan=plan,
                                          verbose=False, save_dir=None)
        rows[plan] = (s["total_cost"].sum(), s["em_kWh"].sum())
        print(f"  {plan:<10} 总费用 {rows[plan][0]:>14,.0f}   紧急电量 {rows[plan][1]:>8,.0f} kWh")
    base, fc = rows["actual"][0], rows["forecast"][0]
    print(f"  电价未知代价（预报 vs 已知）: {fc - base:+,.0f} 元 ({(fc-base)/base:+.2%})")
    return pd.DataFrame([(k, v[0], v[1]) for k, v in rows.items()],
                        columns=["price_plan", "total_cost", "em_kWh"])


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main(do_info=True):
    t0 = time.time()

    # ---- Q4-2：波动电价下重算问题 2 ----
    print("========== Q4-2（波动电价下重算问题 2） ==========")
    rd2, sd2 = solver_q2.solve_problem4_2(verbose=True)
    ok2 = verify_q2(rd2, sd2)
    write_result2_xlsx(rd2, sd2, out_path=ROOT / "result4-2.xlsx",
                       template_path=RESULT_TEMPLATE_DIR / "result4-2.xlsx")
    for name, tb in [("table1_q4_2.csv", make_paper_table1_q2(rd2, sd2)),
                     ("table2_q4_2.csv", make_paper_table2_q2(rd2, sd2)),
                     ("table3_q4_2.csv", make_paper_table3_q2(rd2))]:
        tb.to_csv(ROOT / name, index=False, encoding="utf-8-sig")
    make_emergency_events(rd2).to_csv(ROOT / "emergency_events_q4_2.csv",
                                      index=False, encoding="utf-8-sig")

    # ---- Q4-3：波动电价下重算问题 3 ----
    print("\n========== Q4-3（波动电价下重算问题 3） ==========")
    # λ 取 0.4：1 月标定在 0.4/0.5 间（差 0.04%），全年验证 0.4 更优且日终储电量健康
    #（0.5 会囤储：136 天顶 10800）；q 仍由 1 月标定。
    rd3, sd3 = solver_q3.solve_problem4_3(term_value=0.4, verbose=True)
    ok3 = verify_q3(rd3, sd3)
    write_result3_xlsx(rd3, sd3, out_path=ROOT / "result4-3.xlsx",
                       template_path=RESULT_TEMPLATE_DIR / "result4-3.xlsx")
    for name, tb in [("table1_q4_3.csv", make_paper_table1_q3(rd3, sd3)),
                     ("table2_q4_3.csv", make_paper_table2_q3(rd3, sd3)),
                     ("table3_q4_3.csv", make_paper_table3_q3(rd3, sd3))]:
        tb.to_csv(ROOT / name, index=False, encoding="utf-8-sig")
    make_emergency_events(rd3).to_csv(ROOT / "emergency_events_q4_3.csv",
                                      index=False, encoding="utf-8-sig")

    # ---- 电价预报精度与信息价值 ----
    dates, _, _, typ = load_historical_matrices()
    rp = load_price_realtime()
    fc = build_price_forecast(rp, typ, window=14)
    print("\n电价预报精度（近14日滚动均值）:")
    print("  ", price_forecast_mae(rp, fc, dates))

    if do_info:
        q2 = sd2.attrs.get("q", 0.8)
        q3 = sd3.attrs.get("q", 0.6)
        lam3 = sd3.attrs.get("lambda", 0.0)
        price_info_value_q2(q2).to_csv(ROOT / "price_info_q4_2.csv",
                                       index=False, encoding="utf-8-sig")
        price_info_value_q3(q3, lam3).to_csv(ROOT / "price_info_q4_3.csv",
                                             index=False, encoding="utf-8-sig")
    else:
        print("（跳过电价信息价值对照，用 run_q4.py::price_info_value_* 单独跑）")

    print("\n总耗时", round(time.time() - t0, 1), "s")
    print(f"\n校验结果：Q4-2 {'OK' if ok2 else 'FAIL'}，Q4-3 {'OK' if ok3 else 'FAIL'}")


if __name__ == "__main__":
    main(do_info="--no-info" not in sys.argv)