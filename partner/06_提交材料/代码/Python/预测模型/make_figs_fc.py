# -*- coding: utf-8 -*-
"""预测模型实验: 结果图(精度对比 / 预测曲线 / 费用对比与逐月节省)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fc_common import (D, L_ACT, PV_ACT, FC0, RTP, metrics, DATA, FIG, m2, IS_LOW, DATES, run_q2_fc)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
os.makedirs(FIG, exist_ok=True)

LC = np.load(os.path.join(DATA, 'load_combined.npz'))['L_comb']
PZ = dict(np.load(os.path.join(DATA, 'final_preds.npz')))
PVS = PZ['PV|堆叠(mean4+ef2+lgbm+fc0)']
PS = PZ['P|堆叠(廓线+ef+GBDT)']
cost = json.load(open(os.path.join(DATA, 'final_cost.json'), encoding='utf-8'))

# ---------- 图 1: 精度对比 ----------
eL_b = np.abs(D.Lb[m2:] - L_ACT[m2:]).mean(axis=1) * 6
eL_n = np.abs(LC[m2:] - L_ACT[m2:]).mean(axis=1) * 6
eP_b = np.abs(D.Pb[m2:] - PV_ACT[m2:]).mean(axis=1) * 6
eP_n = np.abs(PVS[m2:] - PV_ACT[m2:]).mean(axis=1) * 6
ep_b = np.abs(PZ['P|廓线+自适应水平'][m2:] - RTP[m2:]).mean(axis=1)
ep_n = np.abs(PS[m2:] - RTP[m2:]).mean(axis=1)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
x = np.arange(2); w = 0.36
ax[0].bar(x - w / 2, [D.Lb[m2:].__abs__().mean() * 0 + np.abs(D.Lb[m2:] - L_ACT[m2:]).mean() * 6,
                      np.abs(LC[m2:] - L_ACT[m2:]).mean() * 6], width=w, color=[GRAY, BLUE])
ax[0].set_xticks(x); ax[0].set_xticklabels(['基线\n(同类日均值)', '新模型\n(分类组合)'])
ax[0].set_ylabel('负荷 MAE (kW)'); ax[0].set_title('负荷预测精度')
for i, v in enumerate([np.abs(D.Lb[m2:] - L_ACT[m2:]).mean() * 6, np.abs(LC[m2:] - L_ACT[m2:]).mean() * 6]):
    ax[0].text(i, v + 2, f'{v:.1f}', ha='center', fontsize=9)
ax[0].grid(alpha=0.3, axis='y')

low = IS_LOW[m2:]
ax[1].bar([0, 1], [eL_b[low].mean(), eL_n[low].mean()], color=[GRAY, BLUE])
ax[1].bar([2.6, 3.6], [eL_b[~low].mean(), eL_n[~low].mean()], color=[GRAY, ORANGE])
ax[1].set_xticks([0, 1, 2.6, 3.6]); ax[1].set_xticklabels(['基线', '新模型', '基线', '新模型'], fontsize=8)
ax[1].text(0.5, 160, '周五/周六', ha='center', fontsize=10)
ax[1].text(3.1, 135, '其他五天', ha='center', fontsize=10)
for xx, vv in zip([0, 1, 2.6, 3.6], [eL_b[low].mean(), eL_n[low].mean(), eL_b[~low].mean(), eL_n[~low].mean()]):
    ax[1].text(xx, vv + 2, f'{vv:.1f}', ha='center', fontsize=9)
ax[1].set_ylabel('负荷 MAE (kW)'); ax[1].set_title('按周五六 / 其他五天分类'); ax[1].grid(alpha=0.3, axis='y')

ax[2].bar([0, 1], [np.abs(D.Pb[m2:] - PV_ACT[m2:]).mean() * 6, np.abs(PVS[m2:] - PV_ACT[m2:]).mean() * 6],
          color=[GRAY, GREEN])
for i, v in enumerate([np.abs(D.Pb[m2:] - PV_ACT[m2:]).mean() * 6, np.abs(PVS[m2:] - PV_ACT[m2:]).mean() * 6]):
    ax[2].text(i, v + 2, f'{v:.1f}', ha='center', fontsize=9)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(['基线\n(近4日均值)', '新模型\n(堆叠)'])
ax[2].set_ylabel('光伏 MAE (kW)'); ax[2].set_title('光伏预测精度'); ax[2].grid(alpha=0.3, axis='y')

ax[3].bar([0, 1], [ep_b.mean(), ep_n.mean()], color=[GRAY, RED])
for i, v in enumerate([ep_b.mean(), ep_n.mean()]):
    ax[3].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
ax[3].set_xticks([0, 1]); ax[3].set_xticklabels(['基线(同类日均价)', '新模型(堆叠)'])
ax[3].set_ylabel('电价 MAE (元/kWh)'); ax[3].set_title('实时电价预测精度'); ax[3].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_final.png')); plt.close()

# ---------- 图 2: 曲线对比(一周) ----------
i0 = int(np.where(DATES == pd.Timestamp('2025-06-16'))[0][0])
days = range(i0, i0 + 7)
tx = np.arange(7 * 144) * 10 / 60
fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for k in range(7):
    sl = slice(k * 144, (k + 1) * 144)
    ax[0].plot(tx[sl], L_ACT[i0 + k] * 6, color='k', lw=0.9)
    ax[0].plot(tx[sl], D.Lb[i0 + k] * 6, color=GRAY, lw=0.8, ls='--')
    ax[0].plot(tx[sl], LC[i0 + k] * 6, color=BLUE, lw=1.1)
    ax[1].plot(tx[sl], PV_ACT[i0 + k] * 6, color='k', lw=0.9)
    ax[1].plot(tx[sl], D.Pb[i0 + k] * 6, color=GRAY, lw=0.8, ls='--')
    ax[1].plot(tx[sl], PVS[i0 + k] * 6, color=GREEN, lw=1.1)
    ax[2].plot(tx[sl], RTP[i0 + k], color='k', lw=0.9)
    ax[2].plot(tx[sl], PS[i0 + k], color=RED, lw=1.0)
    for a in ax:
        a.axvline(k * 24, color=GRAY, lw=0.4, alpha=0.5)
ax[0].plot([], [], color='k', label='实际'); ax[0].plot([], [], color=GRAY, ls='--', label='基线预测')
ax[0].plot([], [], color=BLUE, label='新模型')
ax[0].legend(fontsize=8); ax[0].set_ylabel('负荷 (kW)'); ax[0].set_title('2025-06-16 至 06-22：负荷预测')
ax[1].set_ylabel('光伏 (kW)'); ax[1].set_title('光伏预测'); ax[1].legend(fontsize=8)
ax[2].set_ylabel('电价 (元/kWh)'); ax[2].set_title('实时电价预测'); ax[2].set_xlabel('时刻 (小时)')
for a in ax:
    a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_curves.png')); plt.close()

# ---------- 图 3: 费用对比 + 逐月节省 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
labels = ['问题二\n(固定电价)', '问题三\n(滚动调整)', '问题四之二\n(实时电价)', '问题四之三\n(实时+滚动)']
base = [cost['q2_base'], cost['q3_base'], cost['q42_base'], cost['q43_base']]
new = [cost['q2_new'], cost['q3_new'], cost['q42_new'], cost['q43_new']]
x = np.arange(4); w = 0.36
ax[0].bar(x - w / 2, np.array(base) / 1e4, width=w, color=GRAY, label='基线(论文口径)')
ax[0].bar(x + w / 2, np.array(new) / 1e4, width=w, color=BLUE, label='新预测模型')
for i in range(4):
    ax[0].text(i + w / 2, new[i] / 1e4 + 2, f'{(new[i]-base[i])/1000:.0f}千', ha='center', fontsize=8, color=RED)
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylabel('全年费用 (万元)'); ax[0].set_title('四类场景: 预测模型升级前后费用')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y')

R0 = run_q2_fc(D.Lb, D.Pb, q=0.7)
RN = run_q2_fc(LC, PVS, q=0.7)
dm = ((RN['pc'] + RN['ec']) - (R0['pc'] + R0['ec']))[m2:]
mon = np.array([d.month for d in DATES[m2:]])
bym = np.array([dm[mon == k].sum() / 1e4 for k in range(2, 13)])
ax[1].bar([f'{k}月' for k in range(2, 13)], bym, color=[GREEN if v < 0 else RED for v in bym])
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_ylabel('相对基线的费用变化 (万元)')
ax[1].set_title(f'问题二逐月差异(合计 {dm.sum()/1e4:.1f} 万元)')
ax[1].grid(alpha=0.3, axis='y'); ax[1].tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_cost.png')); plt.close()
print('figures ->', FIG)
print('图1/2/3 完成; 逐月差异(万元):', {int(k): round(float(dm[mon == k].sum() / 1e4), 1) for k in range(2, 13)})
