# -*- coding: utf-8 -*-
"""预测模型实验 · 多样化图表(参考《学术绘图图表库与LLM绘图提示模板》选型)
   哑铃图 / 雷达图 / 雨云图 / ECDF / 月×时段热力图 / 日历热力图 / 瀑布图 / 散点45°线 / 双轴组合
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from fc_common import (D, L_ACT, PV_ACT, FC0, RTP, PRICE, run_q2_fc, metrics, DATA, FIG,
                       m2, n, H, IS_LOW, DATES)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 160
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888', '#7B52AB'
os.makedirs(FIG, exist_ok=True)

LC = np.load(os.path.join(DATA, 'load_combined.npz'))['L_comb']
PZ = dict(np.load(os.path.join(DATA, 'final_preds.npz')))
PVS = PZ['PV|堆叠(mean4+ef2+lgbm+fc0)']
PS = PZ['P|堆叠(廓线+ef+GBDT)']
PPROF = PZ['P|廓线+自适应水平']
cost = json.load(open(os.path.join(DATA, 'final_cost.json'), encoding='utf-8'))

low = IS_LOW[m2:]
eL_b = np.abs(D.Lb[m2:] - L_ACT[m2:]).mean(axis=1) * 6
eL_n = np.abs(LC[m2:] - L_ACT[m2:]).mean(axis=1) * 6
eP_b = np.abs(D.Pb[m2:] - PV_ACT[m2:]).mean(axis=1) * 6
eP_n = np.abs(PVS[m2:] - PV_ACT[m2:]).mean(axis=1) * 6
ep_b = np.abs(PPROF[m2:] - RTP[m2:]).mean(axis=1)
ep_n = np.abs(PS[m2:] - RTP[m2:]).mean(axis=1)


def p95(a):
    return float(np.percentile(np.abs(a), 95))


# ============ 图 A: 哑铃图(基线 → 新模型, 各指标 MAE) ============
rows = [('负荷·周五六', eL_b[low].mean(), eL_n[low].mean(), 'kW'),
        ('负荷·其他五天', eL_b[~low].mean(), eL_n[~low].mean(), 'kW'),
        ('负荷·全窗口', eL_b.mean(), eL_n.mean(), 'kW'),
        ('光伏·全窗口', eP_b.mean(), eP_n.mean(), 'kW'),
        ('电价·全窗口', ep_b.mean(), ep_n.mean(), '元/kWh')]
fig, ax = plt.subplots(figsize=(9.6, 4.2))
y = np.arange(len(rows))[::-1]
for i, (nm, a, b, u) in enumerate(rows):
    yy = y[i]
    ax.plot([a, b], [yy, yy], color=GRAY, lw=3, alpha=0.5, zorder=1, solid_capstyle='round')
    ax.scatter(a, yy, s=130, color=GRAY, zorder=3, label='基线(论文口径)' if i == 0 else None)
    ax.scatter(b, yy, s=150, color=BLUE, zorder=3, marker='D', label='新预测模型' if i == 0 else None)
    ax.text(a, yy + 0.22, f'{a:.1f}', ha='center', fontsize=8, color=GRAY)
    ax.text(b, yy - 0.34, f'{b:.1f}', ha='center', fontsize=8, color=BLUE)
    ax.text(max(a, b) * 1.06, yy, f'{(b-a)/a*100:+.1f}%', va='center', fontsize=8, color=RED)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel('平均绝对误差 MAE(负荷/光伏: kW;电价: 元/kWh)')
ax.set_title('哑铃图: 预测模型升级前后的误差对比')
ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, axis='x')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_dumbbell.png')); plt.close()

# ============ 图 B: 雷达图(多指标归一化) ============
def radar_data(pred, act, scale=6.0):
    d = pred[m2:] - act[m2:]
    out = [np.abs(d).mean() * scale, np.sqrt((d ** 2).mean()) * scale, p95(d) * scale,
           np.abs(d.mean(axis=1)).mean() * scale,
           np.abs(pred[m2:].sum(axis=1) - act[m2:].sum(axis=1)).mean() / act[m2:].sum(axis=1).mean() * 100]
    return out


labels = ['MAE', 'RMSE', 'P95 误差', '日均偏差', '日总量 MAPE']
def radar_comp(base_pred, new_pred, act, title, fname, scale=6.0):
    B = np.array(radar_data(base_pred, act, scale)); N = np.array(radar_data(new_pred, act, scale))
    mx = np.maximum(B, N) * 1.15
    Bn, Nn = B / mx, N / mx
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    Bn = np.concatenate([Bn, Bn[:1]]); Nn = np.concatenate([Nn, Nn[:1]])
    fig = plt.figure(figsize=(5.2, 4.6)); ax = fig.add_subplot(111, polar=True)
    ax.plot(ang, Bn, color=GRAY, lw=2, label='基线')
    ax.fill(ang, Bn, color=GRAY, alpha=0.18)
    ax.plot(ang, Nn, color=BLUE, lw=2, label='新模型')
    ax.fill(ang, Nn, color=BLUE, alpha=0.22)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([]); ax.set_title(title, fontsize=10, pad=14)
    for i, (a, b) in enumerate(zip(B, N)):
        ax.annotate(f'{a:.0f}→{b:.0f}', xy=(ang[i], max(Bn[i], Nn[i])), fontsize=7, ha='center')
    ax.legend(loc='lower right', bbox_to_anchor=(1.18, -0.08), fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, fname)); plt.close()


radar_comp(D.Lb, LC, L_ACT, '负荷预测: 多指标雷达(越小越好)', 'fig_fc_radar_load.png')
radar_comp(D.Pb, PVS, PV_ACT, '光伏预测: 多指标雷达(越小越好)', 'fig_fc_radar_pv.png')
radar_comp(PPROF, PS, RTP, '电价预测: 多指标雷达(越小越好)', 'fig_fc_radar_price.png', scale=1.0)

# ============ 图 C: 雨云图(日误差分布: 半小提琴 + 箱线 + 抖动点) ============
def raincloud(ax, data_list, colors, labels, title, ylabel):
    for k, (d, c, lb) in enumerate(zip(data_list, colors, labels)):
        off = k * 1.0
        kde = gaussian_kde(d)
        yy = np.linspace(max(0, d.min()), d.max(), 200)
        xx = kde(yy); xx = xx / xx.max() * 0.36
        ax.fill_betweenx(yy, off - xx, off, color=c, alpha=0.35, lw=0)
        bp = ax.boxplot([d], positions=[off + 0.22], widths=0.16, vert=True, patch_artist=True,
                        showfliers=False, medianprops=dict(color='k', lw=1.2))
        bp['boxes'][0].set_facecolor(c); bp['boxes'][0].set_alpha(0.75)
        jit = (np.random.RandomState(0).rand(len(d)) - 0.5) * 0.16
        ax.scatter(np.full(len(d), off + 0.46) + jit, d, s=4, color=c, alpha=0.45, lw=0)
    ax.set_xticks([0.2, 1.2]); ax.set_xticklabels(labels)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(alpha=0.3, axis='y')


fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))
raincloud(ax[0], [eL_b[low], eL_n[low]], [GRAY, BLUE], ['基线', '新模型'],
          '周五六 日误差分布', '日 MAE (kW)')
raincloud(ax[1], [eL_b[~low], eL_n[~low]], [GRAY, ORANGE], ['基线', '新模型'],
          '其他五天 日误差分布', '日 MAE (kW)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_raincloud.png')); plt.close()

# ============ 图 D: ECDF 误差分布(尾部对比) ============
fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
for a, (bb, nn, ttl, u) in zip(ax, [(eL_b, eL_n, '负荷', 'kW'), (eP_b, eP_n, '光伏', 'kW'),
                                    (ep_b, ep_n, '电价', '元/kWh')]):
    for d, c, lb in [(bb, GRAY, '基线'), (nn, BLUE, '新模型')]:
        x = np.sort(d); y = np.arange(1, len(x) + 1) / len(x)
        a.plot(x, y, color=c, lw=1.8, label=lb)
    a.axvline(np.percentile(bb, 90), color=GRAY, ls=':', lw=1)
    a.axvline(np.percentile(nn, 90), color=BLUE, ls=':', lw=1)
    a.set_xlabel(f'日 MAE ({u})'); a.set_ylabel('累积概率 ECDF')
    a.set_title(f'{ttl}: 误差经验分布'); a.legend(fontsize=8, frameon=False); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_ecdf.png')); plt.close()

# ============ 图 E: 月×时段 误差改善热力图(光伏) ============
mon = np.array([d.month for d in DATES[m2:]])
hour = np.arange(H) // 6
imp = (np.abs(D.Pb[m2:] - PV_ACT[m2:]) - np.abs(PVS[m2:] - PV_ACT[m2:])).mean(axis=0)
imp_mat = np.zeros((12, 24))
for m in range(2, 13):
    sel = mon == m
    for h in range(24):
        imp_mat[m - 1, h] = (np.abs(D.Pb[m2:][sel] - PV_ACT[m2:][sel])[:, hour == h].mean()
                             - np.abs(PVS[m2:][sel] - PV_ACT[m2:][sel])[:, hour == h].mean()) * 6
fig, ax = plt.subplots(figsize=(11.5, 5.0))
v = np.nanmax(np.abs(imp_mat[1:]))
im = ax.imshow(imp_mat[1:], aspect='auto', cmap='RdYlGn', vmin=-v, vmax=v)
ax.set_yticks(range(11)); ax.set_yticklabels([f'{m}月' for m in range(2, 13)])
ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f'{h}时' for h in range(0, 24, 2)], fontsize=8)
ax.set_title('光伏预测误差改善热力图(基线 MAE − 新模型 MAE, kW; 绿色=改善)')
plt.colorbar(im, ax=ax, label='误差下降 (kW)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_heatmap.png')); plt.close()

# ============ 图 F: 逐日节省日历热力图 + 累计曲线 ============
R0 = run_q2_fc(D.Lb, D.Pb, q=0.7)
RN = run_q2_fc(LC, PVS, q=0.7)
save_day = (R0['pc'] + R0['ec'])[m2:] - (RN['pc'] + RN['ec'])[m2:]
dd = DATES[m2:]
fig, ax = plt.subplots(2, 1, figsize=(13, 5.6), gridspec_kw=dict(height_ratios=[1, 1.1]))
week = np.array([(d - dd[0]).days // 7 for d in dd]); dow = np.array([d.dayofweek for d in dd])
M = np.full((int(week.max()) + 1, 7), np.nan)
for k in range(len(dd)):
    M[week[k], dow[k]] = save_day[k]
v = np.nanmax(np.abs(M)) / 1000
im = ax[0].imshow(M.T, aspect='auto', cmap='RdYlGn', vmin=-v, vmax=v)
ax[0].set_yticks(range(7)); ax[0].set_yticklabels(['一', '二', '三', '四', '五', '六', '日'])
ax[0].set_xlabel('周序号(自 2025-02-01 起)'); ax[0].set_title('问题二 逐日费用差异日历(绿=新模型更省, 千元)')
plt.colorbar(im, ax=ax[0], label='日节省 (千元)')
cum = np.cumsum(save_day) / 1e4
ax[1].plot(dd, cum, color=BLUE, lw=1.8)
ax[1].fill_between(dd, 0, cum, color=BLUE, alpha=0.18)
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set_ylabel('累计节省 (万元)'); ax[1].set_xlabel('日期')
ax[1].set_title(f'累计节省曲线: 全年合计 {save_day.sum()/1e4:.1f} 万元')
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_calendar.png')); plt.close()

# ============ 图 G: 瀑布图(费用下降归因) ============
steps = [('基线(论文口径)', 13848183, GRAY),
         ('光伏换为岭回归堆叠', 13704757, BLUE),
         ('负荷换为全局误差反馈', 13671204, GREEN),
         ('负荷换为分类组合模型', 13655835, PURPLE)]
fig, ax = plt.subplots(figsize=(9.6, 4.6))
xs = np.arange(len(steps))
cur = steps[0][1]
ax.bar(0, cur / 1e4, color=GRAY, width=0.6)
ax.text(0, cur / 1e4 + 6, f'{cur/1e4:.1f}万', ha='center', fontsize=9)
for k in range(1, len(steps)):
    prev, now = steps[k - 1][1], steps[k][1]
    ax.bar(k, (prev - now) / 1e4, bottom=now / 1e4, color=steps[k][2], width=0.6)
    ax.plot([k - 1 + 0.3, k - 0.3], [prev / 1e4, prev / 1e4], color=GRAY, ls='--', lw=0.9)
    ax.text(k, prev / 1e4 + 6, f'−{(prev-now)/1e4:.2f}万', ha='center', fontsize=8.5, color=RED)
ax.bar(len(steps), steps[-1][1] / 1e4, color='#2C6E9B', width=0.6)
ax.text(len(steps), steps[-1][1] / 1e4 + 6, f'{steps[-1][1]/1e4:.1f}万', ha='center', fontsize=9)
ax.set_xticks(list(xs) + [len(steps)])
ax.set_xticklabels([s[0] for s in steps] + ['新模型合计'], fontsize=8, rotation=18)
ax.set_ylim(1350, 1395); ax.set_ylabel('问题二 全年费用 (万元)')
ax.set_title('瀑布图: 问题二费用下降的逐项归因')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_waterfall.png')); plt.close()

# ============ 图 H: 预测 vs 实际 散点(日总量, 含 45° 线与 R²) ============
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
for a, (bp, npd, act, ttl, u) in zip(ax, [
        (D.Lb, LC, L_ACT, '负荷 日总量', 'MWh'),
        (D.Pb, PVS, PV_ACT, '光伏 日总量', 'MWh'),
        (PPROF, PS, RTP, '电价 日均值', '元/kWh')]):
    xb = act[m2:].sum(axis=1) / 1000 if u == 'MWh' else act[m2:].mean(axis=1)
    yb = bp[m2:].sum(axis=1) / 1000 if u == 'MWh' else bp[m2:].mean(axis=1)
    yn = npd[m2:].sum(axis=1) / 1000 if u == 'MWh' else npd[m2:].mean(axis=1)
    a.scatter(xb, yb, s=8, color=GRAY, alpha=0.6, label='基线')
    a.scatter(xb, yn, s=8, color=BLUE, alpha=0.7, label='新模型')
    lo_, hi_ = min(xb.min(), yb.min(), yn.min()), max(xb.max(), yb.max(), yn.max())
    a.plot([lo_, hi_], [lo_, hi_], color=RED, ls='--', lw=1)
    r2b = 1 - ((yb - xb) ** 2).sum() / ((xb - xb.mean()) ** 2).sum()
    r2n = 1 - ((yn - xb) ** 2).sum() / ((xb - xb.mean()) ** 2).sum()
    a.set_xlabel(f'实际 ({u})'); a.set_ylabel(f'预测 ({u})')
    a.set_title(f'{ttl}: $R^2$ {r2b:.3f} → {r2n:.3f}', fontsize=10)
    a.legend(fontsize=8, frameon=False); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_scatter.png')); plt.close()

# ============ 图 I: 双轴组合(逐月费用柱 + 累计节省折线) ============
fig, ax = plt.subplots(figsize=(11, 4.4))
mons = list(range(2, 13))
base_m = np.array([(R0['pc'] + R0['ec'])[m2:][mon == m].sum() for m in mons]) / 1e4
new_m = np.array([(RN['pc'] + RN['ec'])[m2:][mon == m].sum() for m in mons]) / 1e4
x = np.arange(len(mons)); w = 0.38
ax.bar(x - w / 2, base_m, width=w, color=GRAY, label='基线')
ax.bar(x + w / 2, new_m, width=w, color=BLUE, label='新模型')
ax.set_xticks(x); ax.set_xticklabels([f'{m}月' for m in mons])
ax.set_ylabel('月度费用 (万元)'); ax.grid(alpha=0.3, axis='y')
ax2 = ax.twinx()
cm = np.cumsum(base_m - new_m)
ax2.plot(x, cm, color=RED, marker='o', ms=4, lw=1.6, label='累计节省(右轴)')
ax2.set_ylabel('累计节省 (万元)', color=RED); ax2.tick_params(axis='y', colors=RED)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8, frameon=False)
ax.set_title('问题二: 月度费用对比(柱) 与 累计节省(线)')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_cost2.png')); plt.close()

print('多样化图表完成 ->', FIG)
print('逐日节省: 合计 %.1f 万元, 最省一天 %.0f 元, 最差一天 %.0f 元' %
      (save_day.sum() / 1e4, save_day.max(), save_day.min()))
