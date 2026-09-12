# -*- coding: utf-8 -*-
"""EDA 图 + 常量 + 月度统计 (第二遍数据工程的验证性输出)"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUT = r'03_数据工程'
FIG = os.path.join(OUT, 'fig')
BLUE, ORANGE, GREEN, RED = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
GRAY = '#888888'

day = pd.read_parquet(os.path.join(OUT, 'data', 'day_slots.parquet'))
typ = pd.read_parquet(os.path.join(OUT, 'data', 'typ_day.parquet'))
fc0 = pd.read_parquet(os.path.join(OUT, 'data', 'fc_slots_plan.parquet'))
fce = pd.read_csv(os.path.join(OUT, 'data', 'fc_error_by_horizon.csv'))
fch = pd.read_parquet(os.path.join(OUT, 'data', 'fc_hourly.parquet'))

t_axis = np.arange(1, 145) * 10 / 60  # 小时

# ---- fig1 典型日 ----
fig, ax1 = plt.subplots(figsize=(9, 3.8))
ax1.plot(t_axis, typ['load_typ'] / 1000, color=BLUE, lw=1.6, label='小区负载')
ax1.plot(t_axis, typ['pvfc_typ'] / 1000, color=ORANGE, lw=1.6, label='光伏预测功率')
ax1.set_xlabel('时刻 (h)'); ax1.set_ylabel('功率 (MW)')
ax2 = ax1.twinx()
ax2.plot(t_axis, typ['price_typ'], color=GREEN, lw=1.2, ls='--', label='电价')
ax2.set_ylabel('电价 (元/kWh)')
ax1.grid(alpha=0.3, ls='--'); ax1.set_xlim(0, 24)
l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc='upper left', frameon=False, fontsize=8)
ax1.set_title('附件1 典型日: 电价 / 小区负载 / 光伏预测功率', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig1_typical_day.png')); plt.close(fig)

# ---- fig2 四季代表日 ----
fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True)
dates = ['2025-01-15', '2025-04-15', '2025-07-15', '2025-10-15']
titles = ['冬季 1月15日', '春季 4月15日', '夏季 7月15日', '秋季 10月15日']
for ax, ds, ti in zip(axes.ravel(), dates, titles):
    sub = day[day['date'] == pd.Timestamp(ds)]
    ax.plot(t_axis, sub['load_kw'] / 1000, color=BLUE, lw=1.2, label='负载')
    ax.plot(t_axis, sub['pv_kw'] / 1000, color=ORANGE, lw=1.2, label='光伏')
    ax.set_title(ti, fontsize=9); ax.grid(alpha=0.3, ls='--')
    ax.set_ylabel('功率 (MW)', fontsize=8)
axes[0, 0].legend(frameon=False, fontsize=8)
for ax in axes[1, :]:
    ax.set_xlabel('时刻 (h)')
fig.suptitle('附件2 四季代表日: 负载 vs 光伏实际功率', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig2_season_profiles.png')); plt.close(fig)

# ---- fig3 预报 vs 实测 (2025-06-21) ----
ds = '2025-06-21'
sub = day[day['date'] == pd.Timestamp(ds)]
f0 = fc0[fc0['date'] == pd.Timestamp(ds)]
fc_pts = fch[(fch['date'] == pd.Timestamp(ds)) & (fch['issue_hour'] == 0)]
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(t_axis, sub['pv_kw'] / 1000, color=BLUE, lw=1.5, label='光伏实际功率')
ax.plot(t_axis, f0['fc0_kw'] / 1000, color=ORANGE, lw=1.3, label='0:00发布预报(线性插值)')
ax.scatter(fc_pts['horizon'], fc_pts['fc_kw'] / 1000, color=RED, s=18, zorder=5, label='整点预报值')
ax.set_xlabel('时刻 (h)'); ax.set_ylabel('功率 (MW)')
ax.set_title(f'光伏预报 vs 实际 ({ds})', fontsize=10)
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, ls='--'); ax.set_xlim(0, 24)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig3_fc_vs_actual.png')); plt.close(fig)

# ---- fig4 预报误差 ----
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
iss = [0, 6, 12, 18]
buck = ['1-6h', '7-12h', '13-18h', '19-24h']
x = np.arange(4)
for ax, col, ti in zip(axes, ['mae_aligned', 'mae_hourly'],
                       ['误差口径: 与整点对齐时段实测', '误差口径: 与整点小时均值实测']):
    for j, h in enumerate(iss):
        sub = fce[fce['issue_hour'] == h].set_index('horizon_bucket').loc[buck]
        ax.plot(x, sub[col] / 1000, marker='o', ms=4, lw=1.4, label=f'{h}:00发布', color=[BLUE, ORANGE, GREEN, RED][j])
    ax.set_xticks(x); ax.set_xticklabels(buck); ax.set_xlabel('提前期')
    ax.set_ylabel('MAE (MW)'); ax.set_title(ti, fontsize=9)
    ax.grid(alpha=0.3, ls='--'); ax.legend(frameon=False, fontsize=8)
fig.suptitle('光伏预报误差: 按发布时刻与提前期', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig4_fc_error.png')); plt.close(fig)

# ---- fig5 电价热力图 ----
piv = day.pivot_table(index='date', columns='hour', values='price_q4', aggfunc='mean')
fig, ax = plt.subplots(figsize=(9, 4.6))
im = ax.imshow(piv.values, aspect='auto', cmap='cividis', interpolation='nearest')
ax.set_xticks(np.arange(24)); ax.set_xticklabels(np.arange(24), fontsize=7)
ax.set_yticks(np.arange(0, 365, 30))
ax.set_yticklabels([str(piv.index[i].date()) for i in range(0, 365, 30)], fontsize=7)
ax.set_xlabel('小时'); ax.set_ylabel('日期')
ax.set_title('附件4 波动电价: 日均小时电价热力图 (元/kWh)', fontsize=10)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig5_price_heatmap.png')); plt.close(fig)

# ---- 月度统计 ----
mstat = day.groupby('month').agg(load_energy_kwh=('load_kwh', 'sum'), pv_energy_kwh=('pv_kwh', 'sum'),
                                 price_q4_mean=('price_q4', 'mean'), days=('date', 'nunique'))
mstat['load_energy_kwh'] /= mstat['days']; mstat['pv_energy_kwh'] /= mstat['days']
mstat.to_csv(os.path.join(OUT, 'data', 'monthly_stats.csv'), encoding='utf-8-sig')
print('monthly mean load kWh:', mstat['load_energy_kwh'].round(0).tolist())
print('monthly mean pv kWh:', mstat['pv_energy_kwh'].round(0).tolist())

# ---- 常量 ----
const = {
    'cap_kwh': 12000.0, 'p_charge_kw': 5000.0, 'p_discharge_kw': 5000.0,
    'soc_min_kwh': 1200.0, 'soc_max_kwh': 10800.0, 'soc0_2025_01_01': 6000.0,
    'eta_charge': 0.9, 'eta_discharge': 0.9, 'eta_roundtrip': 0.81,
    'slot_min': 10, 'slots_per_day': 144,
    'max_charge_per_slot_kwh': 5000 * 10 / 60, 'max_discharge_per_slot_kwh': 5000 * 10 / 60,
    'emergency_mult': 5.0, 'penalty_surplus': 0.5, 'penalty_excess': 1.5,
}
with open(os.path.join(OUT, 'data', 'constants.json'), 'w', encoding='utf-8') as f:
    json.dump(const, f, ensure_ascii=False, indent=1)
print('const ok:', const['max_charge_per_slot_kwh'])
