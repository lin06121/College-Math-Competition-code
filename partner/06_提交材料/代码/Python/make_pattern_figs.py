# -*- coding: utf-8 -*-
"""论文规律证据图: 周内负荷/电价规律 + 负荷季节特征 + 电价日内结构"""
import sys, io, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io_results import load_all, ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')

day, typ, fc0, fca, pers, tmpl, const = load_all()
day = day.copy()
day['dow'] = day['date'].dt.dayofweek
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.5))
# 1) 周内负荷
g_l = day.groupby(['date', 'dow'])['load_kwh'].sum().groupby('dow').mean() / 1000
ax = axes[0][0]
ax.bar(names, g_l.to_numpy(), color=BLUE, alpha=0.9)
ax.axhline(g_l.to_numpy()[[0, 1, 2, 3, 6]].mean(), color=GRAY, ls='--', lw=1, label='其他五天均值')
ax.set_ylabel('日用电量 (MWh)'); ax.set_title('周内规律①: 周五/周六用电低 36.8%', fontsize=10)
ax.grid(alpha=0.3, axis='y', ls='--'); ax.legend(frameon=False, fontsize=8)
for k, v in enumerate(g_l.to_numpy()):
    ax.text(k, v + 2, f'{v:.0f}', ha='center', fontsize=8)
# 2) 周内电价
g_p = day.groupby('dow')['price_q4'].mean()
ax = axes[0][1]
ax.bar(names, g_p.to_numpy(), color=ORANGE, alpha=0.9)
ax.axhline(g_p.to_numpy()[[0, 1, 2, 3, 6]].mean(), color=GRAY, ls='--', lw=1, label='其他五天均值')
ax.set_ylabel('电价 (元/kWh)'); ax.set_title('周内规律②: 周五/周六电价低 23.3%', fontsize=10)
ax.grid(alpha=0.3, axis='y', ls='--'); ax.legend(frameon=False, fontsize=8)
for k, v in enumerate(g_p.to_numpy()):
    ax.text(k, v + 0.01, f'{v:.3f}', ha='center', fontsize=8)
# 3) 电价 月x星期 热力图
piv = day.pivot_table(index='month', columns='dow', values='price_q4', aggfunc='mean')
im = ax.axes if False else None
ax = axes[1][0]
im = ax.imshow(piv.to_numpy(), aspect='auto', cmap='cividis')
ax.set_xticks(range(7)); ax.set_xticklabels(names, fontsize=8)
ax.set_yticks(range(12)); ax.set_yticklabels([f'{k}月' for k in range(1, 13)], fontsize=7)
ax.set_title('电价周内规律全年一致 (逐月×星期)', fontsize=10)
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.ax.tick_params(labelsize=7)
# 4) 负荷季节 + 电价日内结构
ax = axes[1][1]
g_m = day.groupby(['date', 'month'])['load_kwh'].sum().groupby('month').mean() / 1000
ax.plot(range(1, 13), g_m.to_numpy(), marker='o', color=BLUE, label='日均负荷(MWh)')
ax2 = ax.twinx()
ax2.plot(typ.index, typ['price_typ'], color=GREEN, lw=1.1, ls='--', label='典型日电价(元/kWh)')
ax.set_xlabel('月份'); ax.set_ylabel('日用电量 (MWh)', color=BLUE)
ax2.set_ylabel('电价 (元/kWh)', color=GREEN)
ax.set_title('负荷季节特征(夏高冬低) 与 电价日内结构', fontsize=10)
ax.grid(alpha=0.3, ls='--')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc='upper right')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_weekly_patterns.png')); plt.close(fig)
print('fig_weekly_patterns.png saved')
