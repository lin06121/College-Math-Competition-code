# -*- coding: utf-8 -*-
"""问题 4-2 一键运行：波动电价下重算问题 2（提供版实现，交付用入口）

信息结构：0:00 制定计划时**不知道**当天电价
    · 计划价 = 因果电价预报（近 14 天同时段均值，`forecast_q4.build_price_forecast`；
      历史不足 14 天时退化为附件 1 典型日曲线）
    · 结算价 = 附件 4 实际电价（缺口按 5 倍价紧急购电）

产出：`result4-2.xlsx`（计划购电量 / 充放电量 / 紧急购电量 三表）
      + 论文用表 `table1_q4_2.csv` / `table2_q4_2.csv` / `table3_q4_2.csv`
      + `emergency_events_q4_2.csv`、`data/processed/result42_details|summary.csv`

用法：
    PYTHONUTF8=1 python preprocess.py     # 首次需先生成 data/processed
    PYTHONUTF8=1 python run_q4_2.py

说明：本入口由 run_q4.py 中 Q4-2 那一段单独抽出，调用与校验逻辑完全一致；
     问题 4-3（对应问题 3）的代码与结果见上一级 `代码\\Python\\`。
"""
import sys
import time

import numpy as np

from config import (ROOT, RESULT_TEMPLATE_DIR,
                    STORAGE_SOC_MIN, STORAGE_SOC_MAX, STORAGE_MAX_POWER)
import solver_q2
from format_result_q2 import (write_result2_xlsx, make_paper_table1_q2,
                              make_paper_table2_q2, make_paper_table3_q2,
                              make_emergency_events)
from forecast_q4 import load_price_realtime, build_price_forecast, price_forecast_mae
from forecast_q2 import load_historical_matrices


def verify_q2(result_df, summary_df, tol=1e-3):
    """与 run_q4.py 完全相同的校验：功率平衡 / SOC 连续与边界 / 充放电互斥 / 功率上限。"""
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


def main():
    t0 = time.time()
    rd2, sd2 = solver_q2.solve_problem4_2(verbose=True)
    ok = verify_q2(rd2, sd2)

    write_result2_xlsx(rd2, sd2, out_path=ROOT / "result4-2.xlsx",
                       template_path=RESULT_TEMPLATE_DIR / "result4-2.xlsx")
    for name, tb in [("table1_q4_2.csv", make_paper_table1_q2(rd2, sd2)),
                     ("table2_q4_2.csv", make_paper_table2_q2(rd2, sd2)),
                     ("table3_q4_2.csv", make_paper_table3_q2(rd2))]:
        tb.to_csv(ROOT / name, index=False, encoding="utf-8-sig")
    make_emergency_events(rd2).to_csv(ROOT / "emergency_events_q4_2.csv",
                                      index=False, encoding="utf-8-sig")

    dates, _, _, typ = load_historical_matrices()
    rp = load_price_realtime()
    fc = build_price_forecast(rp, typ, window=14)
    print("  电价预报精度（2/1 起）:", price_forecast_mae(rp, fc, dates))

    print(f"\n  result4-2.xlsx -> {ROOT / 'result4-2.xlsx'}")
    print(f"  总费用 {sd2['total_cost'].sum():,.2f} 元 | "
          f"计划 {sd2['plan_cost'].sum():,.2f} + 紧急 {sd2['em_cost'].sum():,.2f} | "
          f"紧急电量 {sd2['em_kWh'].sum():,.2f} kWh | 用时 {time.time() - t0:.0f}s")
    print("\n校验结果:", "全部通过" if ok else "存在问题，请检查上方 FAIL 项")


if __name__ == "__main__":
    main()
