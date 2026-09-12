# -*- coding: utf-8 -*-
"""升级预测模型口径下重绘论文插图:
   fig_q2_annual.png / fig_q2_basis.png / fig_q2_balance.png / fig_q3_rolling.png / fig_q4_fluct.png
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from causal_core import D, price, PR, ROOT, balance
from engine import solve_day
import fc_upgraded as FU
import upgraded_core as UC

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888', '#7B52AB'
FIG = os.path.join(ROOT, '04_建模求解', 'fig')
os.makedirs(FIG, exist_ok=True)
m2 = D.m2
Q = json.load(open(os.path.join(ROOT, r'04_建模求解\data\upgraded_all_numbers.json'),
                   encoding='utf-8'))['q_star']

print('构造预测 ...')
L, INFO = FU.build_load()
PV = FU.build_pv()
P = FU.build_price()

print('跑管道 ...')
R2 = UC.run_q2(L, PV, q=Q)
R2b = UC.run_q2(D.Lb, D.Pb, q=0.7)                       # 原依据(对照)
R2n = UC.run_q2(L, PV, q=0.0)                            # 无裕度
CB = UC.clairvoyant()
NS = UC.nostorage(L, PV, Q)
R3 = UC.run_q3(L, PV, q=Q)
R3v = {nm: UC.run_q3(L, PV, q=Q, adjust=at) for at, nm in
       [((), '仅0:00'), ((12,), '+12:00'), ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]}
R42 = UC.run_q2(L, PV, q=Q, pmat=PR)
R42b = UC.run_q2(D.Lb, D.Pb, q=0.7, pmat=PR)
R42p = UC.run_q2(L, PV, q=Q, pmat=P)
R43 = UC.run_q3(L, PV, q=Q, pmat=PR)

t_ax = np.arange(144) * 10 / 60
sel = int(np.where(D.dates == pd.Timestamp('2025-06-21'))[0][0])

# ---------- 图 1: 问题二全年 ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
cp = (R2['pc'] + R2['ec'])[m2:] / 1000
cpb = (R2b['pc'] + R2b['ec'])[m2:] / 1000
cns = np.zeros(len(cp))
for k, i in enumerate(range(m2, D.n)):
    B = UC.buffer_mat(L, PV, Q)
    P0 = np.maximum(L[i] + B[i] - PV[i], 0.0)
    Em = np.maximum(D.L_act[i] - D.PV_act[i] - P0, 0.0)
    cns[k] = (price @ P0 + 5 * price @ Em) / 1000
dates = D.dates[m2:]
ax[0].plot(dates, cp, color=BLUE, lw=0.9, label='采用(升级预测)')
ax[0].plot(dates, cpb, color=GRAY, lw=0.8, alpha=0.85, label='原依据(同类日+近4日)')
ax[0].plot(dates, cns, color=ORANGE, lw=0.8, alpha=0.8, label='无储能')
ax[0].axhline(CB / len(cp) / 1000, color=RED, ls='--', lw=1.0, label='天眼链式下界(日均)')
ax[0].set_xlabel('日期'); ax[0].set_ylabel('日费用 (千元)')
ax[0].set_title('问题二：全年逐日购电总费用'); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
mo = np.array([d.month for d in dates])
emk = np.array([float(R2['Em'][m2 + k].sum()) for k in range(len(dates))])
emk_b = np.array([float(R2b['Em'][m2 + k].sum()) for k in range(len(dates))])
w = 0.4
ax[1].bar(np.arange(2, 13) - w / 2, [emk[mo == m].sum() / 1000 for m in range(2, 13)], width=w,
          color=BLUE, label='升级预测')
ax[1].bar(np.arange(2, 13) + w / 2, [emk_b[mo == m].sum() / 1000 for m in range(2, 13)], width=w,
          color=GRAY, label='原依据')
ax[1].set_xticks(range(2, 13)); ax[1].set_xlabel('月份'); ax[1].set_ylabel('紧急购电量 (MWh)')
ax[1].set_title('紧急购电量按月汇总'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y')
ax[2].plot(t_ax, D.L_act[sel] * 6, color=BLUE, lw=1.2, label='实际负载')
ax[2].plot(t_ax, D.PV_act[sel] * 6, color=GREEN, lw=1.2, label='实际光伏')
ax[2].plot(t_ax, R2['P'][sel] * 6, color=ORANGE, lw=1.2, label='计划购电')
ax[2].plot(t_ax, (R2['D'][sel] - R2['C'][sel]) * 6, color=RED, lw=1.0, ls='--', label='储能净放电')
ax[2].set_xlabel('时刻'); ax[2].set_ylabel('功率 (kW)')
ax[2].set_title('2025-06-21 计划与实际'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_annual.png')); plt.close()

# ---------- 图 2: 预测模型对照 ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
lag1 = np.vstack([D.L_act[0], D.L_act[:-1]])
ax[0].plot(t_ax, D.L_act[sel] * 6, color='k', lw=1.0, ls=':', label='实际负载')
ax[0].plot(t_ax, D.Lb[sel] * 6, color=GRAY, lw=1.1, label='近4个同类日均值')
ax[0].plot(t_ax, L[sel] * 6, color=BLUE, lw=1.3, label='水平×形状+分类组合')
ax[0].plot(t_ax, lag1[sel] * 6, color=ORANGE, lw=0.9, alpha=0.8, label='昨日外推')
ax[0].set_xlabel('时刻'); ax[0].set_ylabel('负载 (kW)'); ax[0].set_title('2025-06-21 负载预测')
ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
names = ['基线\nmean4cls', '水平×形状', '递推水平', '全局反馈', '岭回归', 'LightGBM', '分类组合']
vals = [137.52, 135.84, 140.19, 138.68, 188.58, 159.32, 132.28]
cols = [GRAY, GRAY, GRAY, GRAY, GRAY, GRAY, BLUE]
ax[1].bar(names, vals, color=cols)
ax[1].axhline(137.52, color=RED, ls='--', lw=1, label='原依据 137.52 kW')
for i, v in enumerate(vals):
    ax[1].text(i, v + 2, f'{v:.1f}', ha='center', fontsize=8)
ax[1].set_ylabel('负载 MAE (kW)'); ax[1].set_title('负载预测模型对照')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y'); ax[1].tick_params(axis='x', labelsize=7)
pvn = ['基线\nmean4', '附件3\n0:00预报', '晴空指数', 'LightGBM', '岭回归堆叠']
pvv = [151.95, 206.65, 213.33, 151.24, 136.85]
ax[2].bar(pvn, pvv, color=[GRAY, GRAY, GRAY, GRAY, GREEN])
ax[2].axhline(151.95, color=RED, ls='--', lw=1, label='原依据 151.95 kW')
for i, v in enumerate(pvv):
    ax[2].text(i, v + 3, f'{v:.1f}', ha='center', fontsize=8)
ax[2].set_ylabel('光伏 MAE (kW)'); ax[2].set_title('光伏预测模型对照')
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis='y'); ax[2].tick_params(axis='x', labelsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_basis.png')); plt.close()

# ---------- 图 3: 实时平衡控制过程 ----------
fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
for k, (d_, tag) in enumerate([('2025-03-20', '计划与实际接近，储能小幅往返'), ('2025-09-23', '实际高于计划，储能兜底后仍少量紧急')]):
    i = int(np.where(D.dates == pd.Timestamp(d_))[0][0])
    net = D.L_act[i] - D.PV_act[i]
    ax[k].plot(t_ax, net * 6, color='k', lw=1.3, label='实际净需求 $L-PV$')
    ax[k].plot(t_ax, R2['P'][i] * 6, color=BLUE, lw=1.2, label='计划购电 $P^0$')
    ax[k].plot(t_ax, (net - R2['P'][i]) * 6, color=GRAY, lw=1.0, ls='--', label='计划口径缺口')
    ax[k].plot(t_ax, (R2['D'][i] - R2['C'][i]) * 6, color=GREEN, lw=1.1, label='实时储能净放电')
    ax[k].plot(t_ax, R2['Em'][i] * 6, color=RED, lw=1.6, label='紧急购电(5 倍价)')
    ax[k].set_ylabel('功率 (kW)'); ax[k].set_title(f'{d_}：{tag}')
    ax[k].legend(fontsize=7, ncol=3); ax[k].grid(alpha=0.3)
ax[1].set_xlabel('时刻')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_balance.png')); plt.close()

# ---------- 图 4: 问题三滚动对照 ----------
fig, ax = plt.subplots(2, 2, figsize=(12, 7.6))
nm = ['仅0:00', '+12:00', '+6,+12', '全滚动']
base = R2['total']
marg = [-0.30, -0.30, 16.28, 15.88]      # 万元(相对"仅0:00"), 用于显示
vals = [R3v['仅0:00']['total'] / 1e4, R3v['+12:00']['total'] / 1e4, R3v['+6,+12']['total'] / 1e4,
        R3v['全滚动']['total'] / 1e4]
ax[0, 0].bar(nm, vals, color=[BLUE, GREEN, GRAY, GRAY])
for i, v in enumerate(vals):
    ax[0, 0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=8)
ax[0, 0].set_ylabel('全年费用 (万元)'); ax[0, 0].set_title('问题三：滚动方案费用')
ax[0, 0].grid(alpha=0.3, axis='y')
ax[0, 1].bar(nm, [-m for m in marg], color=[GREEN, RED, RED])
ax[0, 1].axhline(0, color='k', lw=0.8)
ax[0, 1].set_ylabel('相对"仅0:00"的变化 (万元，正=更省)')
ax[0, 1].set_title('各方案边际价值：仅 12:00 略有正收益')
ax[0, 1].grid(alpha=0.3, axis='y')
parts = np.array([[R3v[k]['plan'], R3v[k]['dev'], R3v[k]['em']] for k in nm]) / 1e4
bot = np.zeros(len(nm))
for j, (lab, c) in enumerate([('计划购电费', BLUE), ('调整相关费', ORANGE), ('紧急购电费', RED)]):
    ax[1, 0].bar(nm, parts[:, j], bottom=bot, label=lab, color=c); bot += parts[:, j]
ax[1, 0].set_ylabel('费用 (万元)'); ax[1, 0].set_title('问题三：费用结构分解')
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3, axis='y')
ax[1, 1].plot(t_ax, R3v['仅0:00']['P0'][sel] * 6, color=GRAY, lw=1.2, label='0:00 计划')
ax[1, 1].plot(t_ax, R3v['+12:00']['Pf'][sel] * 6, color=GREEN, lw=1.4, label='12:00 调整后')
ax[1, 1].plot(t_ax, D.L_act[sel] * 6 - D.PV_act[sel] * 6, color='k', lw=1.0, ls=':', label='实际净需求')
ax[1, 1].set_xlabel('时刻'); ax[1, 1].set_ylabel('功率 (kW)')
ax[1, 1].set_title('2025-06-21 计划与调整'); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q3_rolling.png')); plt.close()

# ---------- 图 5: 问题四 ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
labs = ['Q4-2\n升级预测', 'Q4-2\n原依据', 'Q4-2\n预测电价', 'Q4-2\n无储能', 'Q4-3\n全滚动']
v = [R42['total'], R42b['total'], R42p['total'], UC.nostorage(L, PV, Q, pmat=PR), R43['total']]
ax[0].bar(labs, np.array(v) / 1e4, color=[GREEN, GRAY, BLUE, ORANGE, PURPLE])
ax[0].axhline(UC.clairvoyant(pmat=PR) / 1e4, color=RED, ls='--', lw=1, label='天眼下界')
for i, x in enumerate(v):
    ax[0].text(i, x / 1e4 + 3, f'{x/1e4:.1f}', ha='center', fontsize=8)
ax[0].set_ylabel('全年费用 (万元)'); ax[0].set_title('问题四：各方案费用')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y'); ax[0].tick_params(axis='x', labelsize=7)
ax[1].bar(['Q4-3 全滚动', 'Q4-2 不调整'], [R43['total'] / 1e4, R42['total'] / 1e4],
          color=[PURPLE, GREEN])
for i, x in enumerate([R43['total'], R42['total']]):
    ax[1].text(i, x / 1e4 + 2, f'{x/1e4:.1f}', ha='center', fontsize=9)
ax[1].set_ylabel('全年费用 (万元)')
ax[1].set_title(f'波动电价下调整的价值为负 ({(R43["total"]-R42["total"])/1e4:+.1f} 万元)')
ax[1].grid(alpha=0.3, axis='y')
p4 = PR[sel]
ax[2].plot(t_ax, p4, color=ORANGE, lw=1.3)
ax2 = ax[2].twinx()
ax2.plot(t_ax, R43['Pf'][sel] * 6, color=BLUE, lw=1.2, label='全滚动调整后购电')
ax2.plot(t_ax, (R43['D'][sel] - R43['C'][sel]) * 6, color=GREEN, lw=1.0, ls='--', label='储能净放电')
ax[2].set_xlabel('时刻'); ax[2].set_ylabel('实时电价 (元/kWh)', color=ORANGE)
ax2.set_ylabel('功率 (kW)'); ax2.legend(fontsize=7, loc='upper left')
ax[2].set_title('2025-06-21 电价与调度(问题四之三)'); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q4_fluct.png')); plt.close()
print('figures ->', FIG)
