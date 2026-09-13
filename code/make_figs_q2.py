# -*- coding: utf-8 -*-
"""问题二论文插图（新口径：同类日负荷+近4日光伏均值、报童 q*=0.8、因果实时平衡结算）。

生成 4 张图到 passage/figures/：
  fig_q2_exec    实时平衡控制的价值（同一计划、三种结算；费用分解 + 逐月紧急电量）
  fig_q2_basis   计划依据/方法/储能对照（总费用横向条形）
  fig_q2_annual  全年表现（逐日费用、逐月费用构成、逐月紧急电量）
  fig_q2_balance 因果平衡过程（两个代表性日期）

数据：data/processed/q2_counterfactual.csv、result2_summary.csv、result2_details.csv

用法：PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/make_figs_q2.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, ROOT, HOURS_PER_SLOT, SLOTS_PER_DAY  # noqa: E402

OUT = ROOT / "passage" / "figures"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"
CH, DIS, RED, GRAY = "#56B4E9", "#D55E00", "#CC0000", "#888888"
C_ADOPT = "#0072B2"

HOURS = (np.arange(SLOTS_PER_DAY) + 0.5) * 10 / 60.0


def _style(ax, xlim=False):
    ax.grid(alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlim:
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.set_xlabel("时刻 (h)")


# ---------------------------------------------------------------- fig_q2_exec
def fig_q2_exec(cf, summary):
    ex = cf[cf["group"] == "执行机制"].set_index("label")
    order = ["僵硬执行（无日内平衡）", "因果实时平衡（采用）", "事后最优再调度（下界）"]
    plan = [ex.loc[k, "plan_cost"] / 1e4 for k in order]
    em = [ex.loc[k, "em_cost"] / 1e4 for k in order]
    tot = [p + e for p, e in zip(plan, em)]
    emk = [ex.loc[k, "em_kWh"] / 1e3 for k in order]

    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.2], width_ratios=[1.05, 1.0],
                          hspace=0.07, wspace=0.30)
    atop = fig.add_subplot(gs[0, 0])
    abot = fig.add_subplot(gs[1, 0], sharex=atop)
    b = fig.add_subplot(gs[:, 1])
    x = np.arange(len(order))
    tl = ["僵硬执行", "因果平衡\n（采用）", "事后最优\n（下界）"]
    brk = 400.0                                   # 断轴位置（万元）

    # 左：总费用分解，断轴（下段示量级=0 基线，上段示差额）
    for ax in (atop, abot):
        ax.bar(x, plan, 0.55, color=BLUE, label="计划购电费")
        ax.bar(x, em, 0.55, bottom=plan, color=ORANGE, label="紧急购电费")
        ax.grid(alpha=0.3, ls="--")
        ax.spines["right"].set_visible(False)
    atop.set_ylim(min(plan) * 0.985, max(tot) * 1.02)
    abot.set_ylim(0, brk)
    atop.spines["bottom"].set_visible(False)
    abot.spines["top"].set_visible(False)
    atop.tick_params(labelbottom=False, bottom=False)
    for ax in (atop, abot):                       # 断轴斜切标记
        ax.plot([0, 0.012], [0, -0.02] if ax is atop else [0, 0.02],
                transform=ax.transAxes, color="k", clip_on=False, lw=0.9)
    for xi, t in zip(x, tot):
        atop.text(xi, t + 5, f"{t:,.0f}", ha="center", fontsize=9, fontweight="bold")
    save_amt = tot[0] - tot[1]
    atop.text(2.0, 1444, f"日内平衡省\n{save_amt:,.0f} 万元（{100*save_amt/tot[0]:.1f}%）",
              ha="center", va="top", fontsize=8.5, color=RED)
    atop.set_yticks([1300, 1350, 1400, 1450])
    atop.spines["top"].set_visible(False)
    abot.set_yticks([0, 200, 400])
    abot.set_xticks(x); abot.set_xticklabels(tl, fontsize=9)
    abot.set_ylabel("全年费用 (万元)")
    atop.legend(frameon=False, fontsize=8, loc="lower left",
                bbox_to_anchor=(0.0, 1.0), ncol=2)

    # 右：紧急购电量（该机制真正改变的量）
    b.bar(x, emk, 0.55, color=[ORANGE, GREEN, GRAY])
    for xi, v in zip(x, emk):
        b.text(xi, v + 4, f"{v:,.0f}", ha="center", fontsize=9)
    b.set_xticks(x); b.set_xticklabels(tl, fontsize=9)
    b.set_ylabel("紧急购电量 (MWh)")
    b.set_ylim(0, max(emk) * 1.20)
    _style(b)

    fig.tight_layout()
    p = OUT / "fig_q2_exec.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


# --------------------------------------------------------------- fig_q2_basis
def fig_q2_basis(cf):
    lab = {
        "同类日负荷+近4日光伏（采用）": "同类日负荷 + 近4日光伏（采用）",
        "同星期几": "同星期几",
        "附件1 典型日": "附件1 典型日曲线",
        "昨日外推": "昨日外推",
        "无裕度（点预测）": "无裕度（点预测）",
        "两阶段随机规划 SAA": "两阶段随机规划 SAA",
        "无储能": "无储能",
    }
    d = cf[cf["label"].isin(lab)].copy()
    d["name"] = d["label"].map(lab)
    d = d.sort_values("total_cost")
    base = float(cf.loc[cf["label"] == "同类日负荷+近4日光伏（采用）", "total_cost"].iloc[0])

    colors = [C_ADOPT if r == "同类日负荷+近4日光伏（采用）" else GRAY for r in d["label"]]
    colors = [ORANGE if r in ("无裕度（点预测）", "两阶段随机规划 SAA") else c
              for r, c in zip(d["label"], colors)]
    colors = [GREEN if r == "无储能" else c for r, c in zip(d["label"], colors)]

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.barh(y, d["total_cost"] / 1e4, 0.6, color=colors)
    for yi, (t, lab_) in zip(y, zip(d["total_cost"], d["label"])):
        pct = 100 * (t - base) / base
        s = "采用" if lab_ == "同类日负荷+近4日光伏（采用）" else f"{pct:+.1f}%"
        ax.text(t / 1e4 + 8, yi, f"{t/1e4:,.0f} 万元  ({s})", va="center", fontsize=8.5,
                fontweight="bold" if lab_ == "同类日负荷+近4日光伏（采用）" else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(d["name"], fontsize=9)
    ax.set_xlabel("全年总费用 (万元)")
    ax.set_xlim(0, d["total_cost"].max() / 1e4 * 1.30)
    ax.invert_yaxis()
    _style(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_ADOPT, label="本文采用"),
                       Patch(color=GRAY, label="预测依据对照"),
                       Patch(color=ORANGE, label="计划方法对照"),
                       Patch(color=GREEN, label="储能对照")],
              frameon=False, fontsize=8, loc="lower right", ncol=1)
    fig.tight_layout()
    p = OUT / "fig_q2_basis.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


# -------------------------------------------------------------- fig_q2_annual
def fig_q2_annual(summary, details):
    s = summary.sort_values("date").reset_index(drop=True)
    s["month"] = pd.to_datetime(s["date"]).dt.month
    mon = s.groupby("month").agg(plan=("plan_cost", "sum"), em=("em_cost", "sum"),
                                 emk=("em_kWh", "sum")).reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    a = ax[0]
    a.plot(np.arange(1, len(s) + 1), s["total_cost"] / 1e4, color=BLUE, lw=0.7)
    a.axhline(s["total_cost"].mean() / 1e4, color=ORANGE, ls="--", lw=1.2,
              label=f"日均 {s['total_cost'].mean()/1e4:.1f} 万元")
    a.set_xlabel("序日（2025-02-01 起）")
    a.set_ylabel("当日总费用 (万元)")
    a.legend(frameon=False, fontsize=8)
    _style(a)

    b = ax[1]
    x = np.arange(len(mon))
    b.bar(x, mon["plan"] / 1e4, 0.6, color=BLUE, label="计划购电费")
    b.bar(x, mon["em"] / 1e4, 0.6, bottom=mon["plan"] / 1e4, color=ORANGE, label="紧急购电费")
    b.set_xticks(x); b.set_xticklabels([f"{m}月" for m in mon["month"]], fontsize=8)
    b.set_ylabel("月度费用 (万元)")
    b.legend(frameon=False, fontsize=8)
    _style(b)

    c = ax[2]
    c.bar(x, mon["emk"] / 1e3, 0.6, color=GREEN)
    c.set_xticks(x); c.set_xticklabels([f"{m}月" for m in mon["month"]], fontsize=8)
    c.set_ylabel("紧急购电量 (MWh)")
    _style(c)

    fig.tight_layout()
    p = OUT / "fig_q2_annual.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


# ------------------------------------------------------------- fig_q2_balance
def fig_q2_balance(details, days=None):
    d = details.copy()
    d["date"] = pd.to_datetime(d["date"])
    if days is None:
        # 选两个代表日：弃光最多者（计划偏保守）与紧急购电最多者（实际高于计划）
        g = d.groupby("date").agg(curt=("P_curt_kW", "sum"), em=("P_em_kW", "sum"))
        days = [g["curt"].idxmax(), g["em"].idxmax()]

    fig, axes = plt.subplots(len(days), 1, figsize=(9.5, 6.4), sharex=True)
    if len(days) == 1:
        axes = [axes]
    for ax, day in zip(axes, days):
        sub = d[d["date"] == day].sort_values("slot")
        net_a = (sub["load_actual_kW"] - sub["pv_actual_kW"]).values
        net_f = (sub["load_forecast_kW"] - sub["pv_forecast_kW"]).values
        em = sub["P_em_kW"].values
        ax.fill_between(HOURS, 0, sub["P_dis_kW"].values, color=DIS, alpha=0.55, label="放电")
        ax.fill_between(HOURS, 0, -sub["P_ch_kW"].values, color=CH, alpha=0.55, label="充电")
        ax.plot(HOURS, net_a, color=BLUE, lw=1.6, label="实际净负荷")
        ax.plot(HOURS, net_f, color=GRAY, lw=1.2, ls="--", label="计划所用预测净负荷")
        if em.max() > 1e-6:
            ax.fill_between(HOURS, 0, em, color=RED, alpha=0.45, label="紧急购电")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel("功率 (kW)")
        ax.set_title(f"{day.date()}　（当日紧急购电 {em.sum()*HOURS_PER_SLOT:,.0f} kWh，"
                     f"弃光 {sub['P_curt_kW'].sum()*HOURS_PER_SLOT:,.0f} kWh）", fontsize=9)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    axes[-1].set_xlim(0, 24); axes[-1].set_xticks(np.arange(0, 25, 4))
    axes[-1].set_xlabel("时刻 (h)")
    fig.tight_layout()
    p = OUT / "fig_q2_balance.png"
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cf = pd.read_csv(DATA_PROCESSED / "q2_counterfactual.csv")
    summary = pd.read_csv(DATA_PROCESSED / "result2_summary.csv", parse_dates=["date"])
    details = pd.read_csv(DATA_PROCESSED / "result2_details.csv", parse_dates=["date"])
    for fn in (fig_q2_exec, fig_q2_annual):
        pass
    print(fig_q2_exec(cf, summary))
    print(fig_q2_basis(cf))
    print(fig_q2_annual(summary, details))
    print(fig_q2_balance(details))


if __name__ == "__main__":
    main()
