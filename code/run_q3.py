"""
问题 3 一键运行：求解 → 校验 → 输出 result3.xlsx 与论文表 1/2/3。

用法：
    python run_q3.py                 # 标定 q + 全年求解 + 校验 + 出结果
    python run_q3.py --q 0.6         # 指定报童分位（标定结果为 0.6）
    python run_q3.py --ablate        # 追加“各预报时刻价值”对照
"""
# 1 月标定得到的报童分位（仅作默认值；main 会重新标定）
DEFAULT_Q = 0.6
import sys
import time
import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED, ROOT, SLOTS_PER_DAY, HOURS_PER_SLOT,
    STORAGE_SOC_MIN, STORAGE_SOC_MAX, STORAGE_MAX_POWER, STORAGE_INITIAL_SOC,
)
import solver_q3
from solver_q3 import (
    load_problem3_data, simulate_year_q3, solve_day_q3, compute_costs,
)
from format_result_q3 import format_all, print_paper_tables, settled_fee
from method_q2 import forecast_mate, quantile_margin


# ----------------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------------

def verify(result_df, summary_df, tol=1e-3, verbose=True):
    """问题 3 解的正确性校验，返回 (ok, 说明列表)。"""
    msgs, ok = [], True

    def chk(cond, text):
        nonlocal ok
        ok &= bool(cond)
        msgs.append(("OK " if cond else "!! ") + text)

    net = result_df["load_actual_kW"] - result_df["pv_actual_kW"]
    bal = (result_df["P_adj_kW"] + result_df["P_em_kW"] + result_df["P_dis_kW"]
           - result_df["P_ch_kW"] - result_df["P_curt_kW"] - net)
    chk(bal.abs().max() < tol, f"功率平衡 max|err| = {bal.abs().max():.3e} kW")

    d = summary_df["e0"].values[1:] - summary_df["e_end"].values[:-1]
    chk(np.abs(d).max() < tol, f"SOC 逐日连续性 max|err| = {np.abs(d).max():.3e} kWh")

    E = result_df["E_kWh"]
    chk(E.min() >= STORAGE_SOC_MIN - tol and E.max() <= STORAGE_SOC_MAX + tol,
        f"SOC 范围 [{E.min():.2f}, {E.max():.2f}] kWh")

    both = ((result_df["P_ch_kW"] > tol) & (result_df["P_dis_kW"] > tol)).sum()
    chk(both == 0, f"充放电互斥（同时充放对 = {both}）")
    pmax = max(result_df["P_ch_kW"].max(), result_df["P_dis_kW"].max())
    chk(pmax <= STORAGE_MAX_POWER + tol, f"充放电功率上限 {pmax:.2f} kW ≤ 5000")
    ec = ((result_df["P_em_kW"] > tol) & (result_df["P_curt_kW"] > tol)).sum()
    chk(ec == 0, f"紧急购电与弃光不同时发生（对 = {ec}）")

    # 分段结构：0:00 阶段未被任何调整改动
    bad = 0
    for _, g in result_df.groupby("date"):
        g = g.sort_values("slot")
        if not np.allclose(g["P_adj_kW"].values[:36], g["P_plan_kW"].values[:36]):
            bad += 1
    chk(bad == 0, f"P_adj[0:36] == P_plan[0:36]（不满足天数 = {bad}）")

    # 成本对账
    kwh = (result_df["P_adj_kW"] * result_df["price"] * HOURS_PER_SLOT).sum()
    chk(abs(kwh - summary_df["adj_purchase_cost"].sum()) < 1e-4,
        f"购电费对账 details={kwh:,.2f} vs summary={summary_df['adj_purchase_cost'].sum():,.2f}")

    if verbose:
        for m in msgs:
            print("  " + m)
    return ok, msgs


def check_alignment(verbose=True):
    """
    align 回归测试：附件 3 的「预报 k 小时」是发布时刻之后第 k 个整点的**瞬时值**，
    应落在 slot s0+(k-1)*6+5。对四个发布时刻分别做偏移扫描，断言最佳偏移均为 +5。

    若有人改回阶梯插值（把整点值摊到整个小时），或把偏移改错，此测试会失败。
    """
    from config import FORECAST_ISSUE_SLOTS
    data = load_problem3_data()
    dates, load_mat, pv_mat, _, fc, _, _, _, _, _ = data
    raw = fc["raw_24h"]
    ok, bests = True, {}
    for ai, it in enumerate(fc["issue_times"]):
        s0 = FORECAST_ISSUE_SLOTS[it]
        score = {}
        for off in range(-6, 13):
            errs = []
            for k in range(len(dates)):
                if pd.Timestamp(dates[k]) < pd.Timestamp("2025-02-01"):
                    continue
                for j in range(24):
                    s = s0 + j * 6 + off
                    if 0 <= s < SLOTS_PER_DAY and np.isfinite(raw[k, ai, j]):
                        errs.append(raw[k, ai, j] - pv_mat[k, s])
            score[off] = np.abs(np.array(errs)).mean()
        best = min(score, key=score.get)
        bests[it] = best
        ok &= (best == 5)
    if verbose:
        txt = ", ".join(f"{it}→{b:+d}" for it, b in bests.items())
        print(f"  {'OK ' if ok else '!! '}预报对齐回归：各期最佳偏移 {txt}（期望全部 +5）")
    return ok


def check_nonanticipativity(sample_days=3, verbose=True, q=DEFAULT_Q):
    """
    非预期性：把 12:00/18:00 的光伏预报改成乱值后，slot 0..72 的 P_adj 必须不变
    —— 即后段信息不会倒流影响已执行/已决定的时段。
    """
    data = load_problem3_data()
    dates, load_mat, pv_mat, price_mat, fc, cls_arr, err, d2i, f2i, net_mat = data
    rng = np.random.default_rng(0)
    worst = 0.0
    for td in [pd.Timestamp(x) for x in
               ["2025-03-20", "2025-06-21", "2025-12-21"][:sample_days]]:
        k, j = d2i[td], f2i[td]
        lf, _ = forecast_mate(k, dates, load_mat, pv_mat, cls_arr)
        lp = lf + quantile_margin(err, k, q)
        base = {it: fc["pv_forecast"][j, fc["issue_times"].index(it)]
                for it in fc["issue_times"]}
        sol_a = solve_day_q3(price_mat[k], load_mat[k], pv_mat[k], base, lp,
                             STORAGE_INITIAL_SOC)
        pert = dict(base)
        for it in ["12:00", "18:00"]:
            pert[it] = rng.uniform(0, 9000, SLOTS_PER_DAY)
        sol_b = solve_day_q3(price_mat[k], load_mat[k], pv_mat[k], pert, lp,
                             STORAGE_INITIAL_SOC)
        worst = max(worst, np.abs(sol_a["P_adj_kW"][:72]
                                  - sol_b["P_adj_kW"][:72]).max())
    ok = worst < 1e-6
    if verbose:
        print(f"  {'OK ' if ok else '!! '}非预期性：扰动后段预报，slot 0..72 的 P_adj "
              f"最大变化 = {worst:.3e} kW")
    return ok


def describe(result_df, summary_df):
    s = lambda c: summary_df[c].sum()
    print("\n===== 全年汇总（2025-02-01 ~ 12-31） =====")
    print(f"  实际购电费   {s('adj_purchase_cost'):>15,.2f} 元")
    print(f"  少买违约费   {s('breach_cost'):>15,.2f} 元")
    print(f"  多买超额费   {s('uplift_cost'):>15,.2f} 元")
    print(f"  紧急购电费   {s('em_cost'):>15,.2f} 元")
    print(f"  总费用       {s('total_cost'):>15,.2f} 元")
    print(f"  实际购电量   {s('adj_kWh'):>15,.2f} kWh（计划 {s('plan_kWh'):,.2f}）")
    print(f"  偏离量       {s('dev_kWh'):>15,.2f} kWh"
          f"（少买 {s('under_kWh'):,.2f} / 多买 {s('over_kWh'):,.2f}）")
    print(f"  紧急购电量   {s('em_kWh'):>15,.2f} kWh")
    print(f"  弃光电量     {s('curt_kWh'):>15,.2f} kWh")


def run_ablation(q=DEFAULT_Q):
    """各预报发布时刻的边际价值。"""
    data = load_problem3_data()
    sets = [("仅 0:00（无调整）", ("0:00",)),
            ("仅 +6:00", ("0:00", "6:00")),
            ("仅 +12:00", ("0:00", "12:00")),
            ("仅 +18:00", ("0:00", "18:00")),
            ("四个时刻全部", ("0:00", "6:00", "12:00", "18:00"))]
    print("\n===== 各预报时刻价值对照 =====")
    print(f"{'配置':<18}{'总费用':>13}{'购电费':>13}{'调整费':>10}"
          f"{'紧急费':>11}{'紧急电量':>10}{'偏离量':>10}")
    rows = []
    for tag, times in sets:
        _, s = simulate_year_q3(*data[:9], "2025-01-01", "2025-12-31",
                                STORAGE_INITIAL_SOC, q=q, issue_times=times,
                                debias=False)
        s = s[s["date"] >= pd.Timestamp("2025-02-01")]
        f = lambda c: s[c].sum()
        rows.append({"配置": tag, "总费用": f("total_cost"),
                     "购电费": f("adj_purchase_cost"), "调整费": f("adj_cost"),
                     "紧急费": f("em_cost"), "紧急电量": f("em_kWh"),
                     "偏离量": f("dev_kWh")})
        print(f"{tag:<18}{f('total_cost'):>13,.0f}{f('adj_purchase_cost'):>13,.0f}"
              f"{f('adj_cost'):>10,.0f}{f('em_cost'):>11,.0f}"
              f"{f('em_kWh'):>10,.0f}{f('dev_kWh'):>10,.0f}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "ablation_q3.csv", index=False, encoding="utf-8-sig")
    base = out.loc[0, "总费用"]
    print(f"\n基准（无调整）{base:,.0f} 元；全部调整 {out.iloc[-1]['总费用']:,.0f} 元，"
          f"节省 {base - out.iloc[-1]['总费用']:,.0f} 元 "
          f"({100*(base-out.iloc[-1]['总费用'])/base:.2f}%)")
    return out


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main():
    argv = sys.argv[1:]
    q = None
    if "--q" in argv:
        q = float(argv[argv.index("--q") + 1])

    t0 = time.time()
    if q is None:
        data = load_problem3_data()
        print("在 2025-01-08 ~ 01-31 上标定报童分位 q：")
        q, _ = solver_q3.calibrate_q_q3(data, verbose=True, debias=False)
        print(f"  -> 标定结果 q* = {q}")
    result_df, summary_df = solver_q3.solve_problem3(q=q, verbose=True)
    print(f"\n求解耗时 {time.time() - t0:.1f} s")

    print("\n===== 校验 =====")
    ok, _ = verify(result_df, summary_df)
    ok &= check_alignment()
    ok &= check_nonanticipativity(q=q)
    describe(result_df, summary_df)

    res = format_all(result_df, summary_df)
    print_paper_tables(result_df, summary_df)

    if "--ablate" in argv:
        run_ablation(q=q)

    print("\n输出文件：")
    for p in [res["result_xlsx"], ROOT / "table1_q3.csv", ROOT / "table2_q3.csv",
              ROOT / "table3_q3.csv", ROOT / "emergency_events_q3.csv",
              DATA_PROCESSED / "result3_details.csv",
              DATA_PROCESSED / "result3_summary.csv"]:
        print(f"  {p}")
    print(f"\n校验结果：{'全部通过' if ok else '存在问题'}")


if __name__ == "__main__":
    main()
