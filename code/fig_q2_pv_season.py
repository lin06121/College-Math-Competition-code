# -*- coding: utf-8 -*-
"""图：为什么问题二的光伏预测不需要显式的季节项。

(a) 日光伏电量：实际 vs 近 4 日均值 vs 逐时段谐波季节曲线 —— 近 4 日均值已紧贴实际并自动跟随季节走势。
(b) 季节项权重 w 的扫描：随着 w 增大（越依赖季节气候态、越少用近期天气），
    光伏 MAE 与问题二全年费用迅速变差。

输出 passage/figures/fig_q2_pv_season.png
用法：PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/fig_q2_pv_season.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT                                    # noqa: E402
import exp_q2_pv as E                                      # noqa: E402
import exp_q2_pv2 as E2                                    # noqa: E402

OUT = ROOT / "passage" / "figures"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#999999"

# ---- 季节项权重扫描结果（由 exp_q2_pv2 计算）----
W = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
MAE = np.array([151.95, 151.29, 153.82, 159.13, 176.30, 240.85])
COST = np.array([13_678_365, 13_674_485, 13_684_393, 13_711_466, 13_806_443, 14_355_059])


def main():
    m2 = E.M2
    dates = E.DATES[m2:]
    act = E.E_DAILY[m2:] / 1e3                       # MWh
    r4 = np.array([E.PV[i - 4:i].sum(axis=1).mean() for i in range(m2, E.N)]) / 1e3
    ss_raw = E2.SS.sum(axis=1) / 1e3
    ss = ss_raw[m2:]

    ok = np.isfinite(ss)
    r_act = np.corrcoef(act, r4)[0, 1]
    r_sea = np.corrcoef(act[ok], ss[ok])[0, 1]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 4.0),
                               gridspec_kw=dict(width_ratios=[1.35, 1.0], wspace=0.22))

    # ---- (a) 机制 ----
    a.plot(dates, act, color=GRAY, lw=0.9, label="实际日光伏电量")
    a.plot(dates, r4, color=BLUE, lw=1.6, label=f"近 4 日均值（采用，$r={r_act:.3f}$）")
    a.plot(dates, ss, color=ORANGE, lw=1.8, ls="--",
           label=f"逐时段谐波季节曲线（$r={r_sea:.3f}$）")
    a.set_ylabel("日光伏电量 (MWh)")
    a.set_ylim(act.min() * 0.90, max(act.max(), np.nanmax(ss)) * 1.22)
    a.legend(frameon=False, fontsize=8, loc="upper left")
    a.grid(alpha=0.3, ls="--")
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)

    # ---- (b) 证据 ----
    b.plot(W, MAE, color=BLUE, lw=1.6, marker="o", ms=5)
    b.axhline(MAE[0], color=GRAY, ls=":", lw=1)
    for w, m, c in zip(W, MAE, COST):
        d = c - COST[0]
        if w == 0:
            b.annotate("采用", (w, m), textcoords="offset points", xytext=(-9, -3),
                       ha="right", va="center", fontsize=8.5, color=BLUE, fontweight="bold")
        else:
            b.annotate(f"{d/1e4:+.2f} 万", (w, m), textcoords="offset points", xytext=(0, 9),
                       ha="center", fontsize=8,
                       color=BLUE if abs(d) < 5000 else "#CC0000")
    b.set_xlabel("季节项权重 $w$（$w=0$ 为纯近期均值，$w=1$ 为纯季节曲线）")
    b.set_ylabel("光伏预测 MAE (kW)")
    b.set_xticks(W)
    b.set_xlim(-0.16, 1.06)
    b.set_ylim(MAE.min() * 0.94, MAE.max() * 1.12)
    b.grid(alpha=0.3, ls="--")
    b.spines["top"].set_visible(False)
    b.spines["right"].set_visible(False)

    # 图题与面板说明由论文 \caption 给出，图内不画标题
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / "fig_q2_pv_season.png"
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    main()
