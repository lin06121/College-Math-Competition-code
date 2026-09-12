# -*- coding: utf-8 -*-
"""Q1: 单典型日 计划购电策略 —— LP(精确) vs 贪心配对(快速) vs 无储能(基线)"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
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
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')
DATA = os.path.join(ROOT, r'04_建模求解\data')
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
t_axis = (np.arange(144) + 0.5) * 10 / 60
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax = axes[0]
ax.plot(t_axis, load * 6, color=BLUE, lw=1.5, label='小区负载')
ax.plot(t_axis, pv * 6, color=ORANGE, lw=1.5, label='光伏预测')
ax.plot(t_axis, P * 6, color=GREEN, lw=1.2, label='购电功率')
ax.plot(t_axis, D * 6, color=RED, lw=1.2, ls='--', label='放电功率')
ax.plot(t_axis, -C * 6, color=GRAY, lw=1.2, ls='--', label='充电功率(负)')
ax.set_ylabel('功率 (kW)'); ax.grid(alpha=0.3, ls='--')
ax.legend(ncol=3, frameon=False, fontsize=8)
ax.set_title('问题1 典型日最优调度 (LP): 功率曲线', fontsize=10)
ax2 = axes[1]
ax2.plot(t_axis, E[1:] / 1000, color=GREEN, lw=1.5, label='储电量 SOC')
ax2.axhline(SOC_MAX / 1000, color=RED, ls=':', lw=1, label='SOC上限 10.8 MWh')
ax2.axhline(SOC_MIN / 1000, color=GRAY, ls=':', lw=1, label='SOC下限 1.2 MWh')
ax2.set_ylabel('储电量 (MWh)'); ax2.set_xlabel('时刻 (h)')
ax2.grid(alpha=0.3, ls='--'); ax2.legend(frameon=False, fontsize=8)
ax2.set_xlim(0, 24)
ax2.set_title('储能设备储电量轨迹 (0:00与24:00均为6.0 MWh)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_q1_schedule.png')); plt.close(fig)

# ---------- 存档 ----------
with open(os.path.join(DATA, 'q1_results.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(cost_lp=cost_lp, cost_greedy=cost_greedy, cost_nostorage=cost0,
                   t_lp_ms=t_lp * 1000, t1=t1, t2=t2, total_kwh=float(P.sum()),
                   soc0=float(E[0]), soc24=float(E[-1])), f, ensure_ascii=False, indent=1)
print('Q1 DONE')
