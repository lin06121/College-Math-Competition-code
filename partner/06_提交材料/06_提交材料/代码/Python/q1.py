# -*- coding: utf-8 -*-
"""Q1: 单典型日 计划购电策略 —— LP(精确) vs 贪心配对(快速) vs 无储能(基线)

绘图部分抽为模块级函数 plot_q1_schedule()，可被外部脚本 import 复用
（顶层可执行逻辑统一收在 main() 内，避免 import 时触发数据加载）。
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine import solve_day, settle_emergency, block_agg, P_MAX, SOC_MIN, SOC_MAX
from io_results import load_all, write_result1, ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# --- Okabe-Ito 色盲安全配色（见学术绘图图表库 §7）---
C_LOAD, C_PV, C_GRID = '#0072B2', '#009E73', '#E69F00'      # 负载 / 光伏 / 购电
C_CH, C_DIS = '#56B4E9', '#D55E00'                          # 充电 / 放电
C_SOC, C_PRICE, C_GRAY = '#0072B2', '#CC79A7', '#888888'    # SOC(独立面板) / 电价 / 参考线
C_SOC_M = '#333333'                                         # SOC(并入充放电面板时,深色以区别于浅蓝充电)

FIG = os.environ.get('Q1_FIG_DIR', os.path.join(ROOT, r'04_建模求解\fig'))
DATA = os.path.join(ROOT, r'04_建模求解\data')


def plot_q1_schedule(price, load, pv, P, C, D, E, out_path=None,
                     soc_min=SOC_MIN, soc_max=SOC_MAX, merge_soc=True):
    """问题一典型日最优调度（共享 0--24h 时间轴的小倍数面板）。

    参数
    ----
    price : (144,) 逐时段电价，元/kWh
    load, pv, P, C, D : (144,) 逐时段功率，单位 **kW**
                        （小区负载、光伏预测、购电、充电、放电）
    E : (145,) 逐时段边界储电量，单位 **kWh**（含 0:00 与 24:00 端点）
    out_path : 输出图片路径；缺省 FIG/fig_q1_schedule.png（FIG 可用环境变量 Q1_FIG_DIR 覆盖）
    merge_soc : False = 四面板（充放电、SOC 各自独立面板）；
                True  = 三面板（SOC 以右轴叠加到充放电面板，更紧凑）
    """
    price = np.asarray(price, float)
    load, pv, P, C, D = (np.asarray(a, float) for a in (load, pv, P, C, D))
    E = np.asarray(E, float)
    t = (np.arange(len(price)) + 0.5) * 10 / 60.0        # 时段中点(小时)
    xe = np.arange(len(E)) / 6.0                         # 储电量时刻(小时, 0..24)

    nrow = 3 if merge_soc else 4
    ratios = [1.0, 1.3, 1.35] if merge_soc else [1.0, 1.25, 1.0, 1.0]
    fig, ax = plt.subplots(nrow, 1, figsize=(9.5, 7.6 if merge_soc else 9.2),
                           sharex=True, gridspec_kw=dict(height_ratios=ratios))

    # (a) 电价：何时便宜（标注与正文一致的两个时段均值）
    a = ax[0]
    a.step(t, price, where='mid', color=C_PRICE, lw=1.2)
    a.fill_between(t, 0, price, step='mid', color=C_PRICE, alpha=0.10)
    for x0, tag in ((2.0, f'夜间均值 {price[:24].mean():.3f}'),
                    (18.0, f'晚高峰均值 {price[96:120].mean():.3f}')):
        a.annotate(tag, (x0, np.interp(x0, t, price)),
                   textcoords='offset points', xytext=(0, 7), ha='center',
                   fontsize=8, color=C_PRICE)
    a.set_ylabel('电价 (元/kWh)')

    # (b) 功率平衡：负载、光伏、购电；光伏富余区间高亮（缺口由购电面积本身体现）
    b = ax[1]
    b.fill_between(t, 0, P, color=C_GRID, alpha=0.35, label='购电功率')
    b.fill_between(t, load, np.maximum(load, pv), where=pv > load,
                   color=C_PV, alpha=0.12)
    b.plot(t, load, color=C_LOAD, lw=1.6, label='小区负载')
    b.plot(t, pv, color=C_PV, lw=1.6, label='光伏预测')
    i_sur = int((pv - load).argmax())
    b.annotate('光伏富余（免费充电）', (t[i_sur], pv[i_sur]),
               textcoords='offset points', xytext=(0, 6), ha='center',
               fontsize=8, color=C_PV)
    b.set_ylim(0, max(load.max(), pv.max(), P.max()) * 1.22)   # 顶部留白放图例
    b.set_ylabel('功率 (kW)')

    # (c) 充放电：低充高放
    c = ax[2]
    c.fill_between(t, 0, D, color=C_DIS, alpha=0.55,
                   label=f'放电 {D.sum() / 6:,.0f} kWh')
    c.fill_between(t, 0, -C, color=C_CH, alpha=0.55,
                   label=f'充电 {C.sum() / 6:,.0f} kWh')
    c.axhline(0, color='k', lw=0.8)
    c.set_ylabel('充/放电功率 (kW)')

    if merge_soc:
        # SOC 是充放电功率的积分，二者强相关，故允许双轴同图对照
        c2 = c.twinx()
        c2.plot(xe, E / 1000, color=C_SOC_M, lw=1.9, label='储电量 SOC（右轴）')
        c2.axhline(soc_max / 1000, color=C_GRAY, ls=':', lw=1)
        c2.axhline(soc_min / 1000, color=C_GRAY, ls=':', lw=1)
        vmax = max(C.max(), D.max())
        c.set_ylim(-vmax * 1.35, vmax * 1.35)              # 顶部留白放图例
        c2.set_ylim(0, soc_max / 1000 * 1.35)
        c2.set_ylabel('储电量 (MWh)', color=C_SOC_M)
        c2.tick_params(axis='y', colors=C_SOC_M)
        c2.spines['top'].set_visible(False)
        h1, l1 = c.get_legend_handles_labels()
        h2, l2 = c2.get_legend_handles_labels()
        c.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper right', ncol=1)
    else:
        # (d) 储电量轨迹
        d = ax[3]
        d.axhspan(soc_min / 1000, soc_max / 1000, color=C_GRAY, alpha=0.08)
        d.plot(xe, E / 1000, color=C_SOC, lw=1.6, label='储电量 SOC')
        d.axhline(soc_max / 1000, color=C_GRAY, ls=':', lw=1,
                  label=f'上限 {soc_max / 1000:.1f} MWh')
        d.axhline(soc_min / 1000, color=C_GRAY, ls=':', lw=1,
                  label=f'下限 {soc_min / 1000:.1f} MWh')
        d.annotate(f'0:00 = 24:00 = {E[0] / 1000:.1f} MWh',
                   xy=(0.02, E[0] / 1000), xycoords=('axes fraction', 'data'),
                   textcoords='offset points', xytext=(2, 8), fontsize=8, color=C_SOC)
        d.set_ylabel('储电量 (MWh)')

    # 统一样式：去上/右框线、浅灰网格、面板编号
    for k, a_ in enumerate(ax):
        a_.grid(alpha=0.3, ls='--')
        a_.spines['top'].set_visible(False)
        a_.spines['right'].set_visible(False)
        a_.text(0.008, 0.95, f'({"abcd"[k]})', transform=a_.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
    ax[1].legend(frameon=False, fontsize=8, loc='upper right', ncol=1)
    if not merge_soc:
        ax[2].legend(frameon=False, fontsize=8, loc='upper right', ncol=1)
        ax[3].legend(frameon=False, fontsize=8, loc='upper right', ncol=1)
    ax[-1].set_xlim(0, 24)
    ax[-1].set_xticks(np.arange(0, 25, 4))
    ax[-1].set_xlabel('时刻 (h)')
    fig.suptitle('问题一典型日最优调度：低充高放与光伏富余充电', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if out_path is None:
        out_path = os.path.join(FIG, 'fig_q1_schedule.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    day, typ, fc0, fca, pers, tmpl, const = load_all()
    price = typ['price_typ'].to_numpy()
    load = typ['load_typ'].to_numpy() / 6.0      # kW -> kWh/时段
    pv = typ['pvfc_typ'].to_numpy() / 6.0

    # ---------- M1: LP ----------
    t0 = time.time()
    res = solve_day(price, load, pv, soc0=6000.0, soc_end=6000.0)
    t_lp = time.time() - t0
    assert res['status'] == 'ok', res['status']
    P, C, D, E = res['P'], res['C'], res['D'], res['E']
    cost_lp = res['gross']
    print(f"[M1 LP] 状态={res['status']} 耗时={t_lp*1000:.1f}ms 购电费={cost_lp:.2f} 元")

    # ---------- M2: 贪心配对(忽略时序约束的松弛启发式) ----------
    net = load - pv
    P0 = np.maximum(net, 0)
    cost0 = float(np.sum(price * P0))
    surp = [(price[t], t, min(-net[t], P_MAX) * 0.81) for t in range(144) if net[t] < -1e-9]
    defi = [(price[t], t, min(net[t], P_MAX)) for t in range(144) if net[t] > 1e-9]
    surp.sort(); defi.sort(key=lambda x: -x[0])
    q_total = 0.0; saving = 0.0
    si = 0
    for (pd_, d, rem_d) in defi:
        while rem_d > 1e-9 and si < len(surp) and q_total < 4320.0 - 1e-9:
            ps_, s, rem_s = surp[si]
            q = min(rem_d, rem_s, 4320.0 - q_total)
            if q <= 1e-9:
                si += 1; continue
            saving += pd_ * q
            q_total += q; rem_d -= q
            surp[si] = (ps_, s, rem_s - q)
            if rem_s - q <= 1e-9:
                si += 1
        if q_total >= 4320.0 - 1e-9:
            break
    cost_greedy = cost0 - saving
    print(f"[M2 贪心] 无储能购电费={cost0:.2f} 元, 套利节约={saving:.2f} 元, 贪心购电费={cost_greedy:.2f} 元")

    # ---------- 对比与一致性 ----------
    print(f"[对比] LP={cost_lp:.2f} <= 贪心(松弛)={cost_greedy:.2f} <= 无储能={cost0:.2f} (元)")
    gap = cost_greedy - cost_lp
    print(f"[对比] 贪心对LP差距={gap:.2f} 元 ({gap/cost0*100:.2f}% 的无储能成本), 时序约束代价占比")

    # ---------- 论文表1: 指定时段购电量 ----------
    slot_of = {'10:00-10:10': 61, '12:00-12:10': 73, '14:00-14:10': 85,
               '16:00-16:10': 97, '18:00-18:10': 109, '20:00-20:10': 121}
    t1 = {k: round(float(P[v - 1]), 2) for k, v in slot_of.items()}
    print('[表1] 指定时段购电量(kWh):', t1)
    print(f'[表1] 全天购电量={P.sum():.2f} kWh, 全天购电费={cost_lp:.2f} 元')
    blk = block_agg(C), block_agg(D)
    t2 = {b: (round(float(blk[0][i]), 2), round(float(blk[1][i]), 2)) for i, b in enumerate(['0:00-4:00','4:00-8:00','8:00-12:00','12:00-16:00','16:00-20:00','20:00-24:00'])}
    print('[表2] 分时段充/放电量(kWh):', t2)
    print(f'[表2] 0:00储电量={E[0]:.2f} kWh, 24:00储电量={E[-1]:.2f} kWh')

    # ---------- 写结果文件 ----------
    write_result1('result1.xlsx', P, C, D, E[0], E[-1])

    # ---------- 图 ----------
    png = plot_q1_schedule(price, load * 6, pv * 6, P * 6, C * 6, D * 6, E)
    print('[图] 问题一调度图 ->', png)

    # ---------- 存档 ----------
    with open(os.path.join(DATA, 'q1_results.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(cost_lp=cost_lp, cost_greedy=cost_greedy, cost_nostorage=cost0,
                       t_lp_ms=t_lp * 1000, t1=t1, t2=t2, total_kwh=float(P.sum()),
                       soc0=float(E[0]), soc24=float(E[-1])), f, ensure_ascii=False, indent=1)
    print('Q1 DONE')


if __name__ == '__main__':
    main()
