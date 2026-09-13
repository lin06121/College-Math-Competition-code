# -*- coding: utf-8 -*-
"""生成论文图（问题一典型日最优调度，三面板、图内不画标题）。

面板（共享 0--24 h 时间轴，自上而下）：
  (a) 电价曲线（谷、峰标注）；
  (b) 功率平衡：小区负载 / 光伏预测 / 购电功率，光伏富余区间以浅绿标出；
  (c) 储能充放电功率（充电为负、放电为正，左轴）与储电量 SOC（右轴）。

图题由论文 \\caption 给出，图内不含标题；数值与论文一致（59 482.7 kWh / 35 126.95 元）。

用法：PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/gen_fig_q1.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from solver_q1 import solve_problem1        # noqa: E402
from config import SLOTS_PER_DAY            # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GREEN, RED, GRAY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#888888"
CH, DIS = "#56B4E9", "#D55E00"

HOURS = (np.arange(SLOTS_PER_DAY) + 0.5) * 10 / 60.0


def _style(ax):
    ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    sol = solve_problem1()
    price, load, pv = sol["price"], sol["load"], sol["pv"]
    P, C, D = sol["P_plan_kW"], sol["P_ch_kW"], sol["P_dis_kW"]
    E = sol["E_kWh"]                                   # 长度 145

    fig, axes = plt.subplots(
        3, 1, figsize=(9.5, 8.0), sharex=True,
        gridspec_kw=dict(height_ratios=[1.0, 1.35, 1.35], hspace=0.18))

    # (a) 电价
    ax = axes[0]
    ax.plot(HOURS, price, color=BLUE, lw=1.6)
    ax.fill_between(HOURS, 0, price, color=BLUE, alpha=0.12)
    ax.set_ylabel("电价 (元/kWh)")
    p_min, p_max = price.min(), price.max()
    ax.text(HOURS[np.argmin(price)], p_min, f"谷 {p_min:.3f}",
            ha="center", va="top", fontsize=8, color=GRAY)
    ax.text(HOURS[np.argmax(price)], p_max, f"峰 {p_max:.3f}",
            ha="center", va="bottom", fontsize=8, color=RED)
    _style(ax)

    # (b) 功率平衡
    ax = axes[1]
    ax.plot(HOURS, load, color=BLUE, lw=1.5, label="小区负载")
    ax.plot(HOURS, pv, color=GREEN, lw=1.5, label="光伏预测")
    ax.plot(HOURS, P, color=ORANGE, lw=1.2, label="购电功率")
    surp = pv > load
    ax.fill_between(HOURS, load, pv, where=surp, color=GREEN, alpha=0.22,
                    label="光伏富余区间（可充电）")
    ax.set_ylabel("功率 (kW)")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    _style(ax)

    # (c) 储能充放电 + SOC
    ax = axes[2]
    ax.bar(HOURS, D, width=10 / 60, color=DIS, alpha=0.75, label="放电")
    ax.bar(HOURS, -C, width=10 / 60, color=CH, alpha=0.75, label="充电")
    ax.axhline(0, color="k", lw=0.7)
    cmax = max(C.max(), D.max())
    ax.set_ylabel("充放电功率 (kW)")
    ax.set_ylim(-cmax * 1.15, cmax * 1.22)
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(HOURS, E[1:] / 1000, color=GREEN, lw=1.5, label="储电量 SOC")
    ax2.axhline(10.8, color=RED, ls=":", lw=1)
    ax2.axhline(1.2, color=RED, ls=":", lw=1)
    ax2.set_ylabel("储电量 (MWh)")
    ax2.set_ylim(0, 12)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, ncol=2, loc="upper left")

    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(np.arange(0, 25, 4))
    axes[-1].set_xlabel("时刻 (h)")

    out = ROOT / "passage" / "figures" / "fig_q1_schedule.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("购电量 %.2f kWh, 购电费 %.2f 元" % (sol["E_plan_kWh"].sum(), sol["total_plan_cost"]))
    print("图已生成 ->", out)


if __name__ == "__main__":
    main()