# -*- coding: utf-8 -*-
"""口径 C(0:00 一次性决策 + 照计划执行) 下的论文插图"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from causal_core import D, price, PR, ROOT
import fc_upgraded as FU
import plan_core as PC

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888', '#7B52AB'
FIG = os.path.join(ROOT, '04_建模求解', 'fig')
os.makedirs(FIG, exist_ok=True)
m2 = D.m2
NU = json.load(open(os.path.join(ROOT, r'04_建模求解\data\plan_all_numbers.json'), encoding='utf-8'))
Q = NU['q_star']
print('构造预测 ...')
L, INFO = FU.build_load(); PV = FU.build_pv(); P = FU.build_price()
print('跑管道 ...')
R2 = PC.run_q2(L, PV, q=Q)
R2b = PC.run_q2(D.Lb, D.Pb, q=Q)                      # 原依据(对照)
R3 = PC.run_q3(L, PV, q=Q, adjust=(12,))
R42 = PC.run_q2(L, PV, q=Q, pmat=PR)
R43 = PC.run_q3(L, PV, q=Q, pmat=PR, adjust=(12,))
CB = PC.clairvoyant(); NS = PC.nostorage(L, PV, Q)
t_ax = np.arange(144) * 10 / 60
sel = int(np.where(D.dates == pd.Timestamp('2025-06-21'))[0][0])
dates = D.dates[m2:]

# 图 1: 问题二全年
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
cp = (R2['pc'] + R2['ec'])[m2:] / 1000
cpb = (R2b['pc'] + R2b['ec'])[m2:] / 1000
cns = np.zeros(len(cp)); B = PC.buffer_mat(L, PV, Q)
for k, i in enumerate(range(m2, D.n)):
    P0 = np.maximum(L[i] + B[i] - PV[i], 0.0)
    Em = np.maximum(D.L_act[i] - D.PV_act[i] - P0, 0.0)
    cns[k] = (price @ P0 + 5 * price @ Em) / 1000
ax[0].plot(dates, cp, color=BLUE, lw=0.9, label='采用(升级预测)')
ax[0].plot(dates, cpb, color=GRAY, lw=0.8, alpha=0.85, label='原依据(同类日+近4日)')
ax[0].plot(dates, cns, color=ORANGE, lw=0.8, alpha=0.8, label='无储能')
ax[0].axhline(CB / len(cp) / 1000, color=RED, ls='--', lw=1.0, label='天眼链式下界(日均)')
ax[0].set_xlabel('日期'); ax[0].set_ylabel('日费用 (千元)')
ax[0].set_title('问题二：全年逐日购电总费用'); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
mo = np.array([d.month for d in dates])
emk = np.array([float(R2['Em'][m2 + k].sum()) for k in range(len(dates))])
emkb = np.array([float(R2b['Em'][m2 + k].sum()) for k in range(len(dates))])
w = 0.4
ax[1].bar(np.arange(2, 13) - w / 2, [emk[mo == m].sum() / 1000 for m in range(2, 13)], width=w,
          color=BLUE, label='升级预测')
ax[1].bar(np.arange(2, 13) + w / 2, [emkb[mo == m].sum() / 1000 for m in range(2, 13)], width=w,
          color=GRAY, label='原依据')
ax[1].set_xticks(range(2, 13)); ax[1].set_xlabel('月份'); ax[1].set_ylabel('紧急购电量 (MWh)')
ax[1].set_title('紧急购电量按月汇总'); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y')
ax[2].plot(t_ax, D.L_act[sel] * 6, color=BLUE, lw=1.2, label='实际负载')
ax[2].plot(t_ax, D.PV_act[sel] * 6, color=GREEN, lw=1.2, label='实际光伏')
ax[2].plot(t_ax, R2['P'][sel] * 6, color=ORANGE, lw=1.2, label='计划购电')
ax[2].plot(t_ax, (R2['D'][sel] - R2['C'][sel]) * 6, color=RED, lw=1.0, ls='--', label='储能净放电(计划)')
ax[2].set_xlabel('时刻'); ax[2].set_ylabel('功率 (kW)')
ax[2].set_title('2025-06-21 计划与实际'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_annual.png')); plt.close()

# 图 2: 预测模型对照
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
lag1 = np.vstack([D.L_act[0], D.L_act[:-1]])
ax[0].plot(t_ax, D.L_act[sel] * 6, color='k', lw=1.0, ls=':', label='实际负载')
ax[0].plot(t_ax, D.Lb[sel] * 6, color=GRAY, lw=1.1, label='近4个同类日均值')
ax[0].plot(t_ax, L[sel] * 6, color=BLUE, lw=1.3, label='水平×形状+分类组合')
ax[0].plot(t_ax, lag1[sel] * 6, color=ORANGE, lw=0.9, alpha=0.8, label='昨日外推')
ax[0].set_xlabel('时刻'); ax[0].set_ylabel('负载 (kW)'); ax[0].set_title('2025-06-21 负载预测')
ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
names = ['基线\nmean4cls', '水平×形状', '递推水平', '全局反馈', '岭回归', 'LightGBM', '分类组合']
vals = [137.52, 135.84, 140.19, 138.68, 188.58, 159.32, 132.28]
ax[1].bar(names, vals, color=[GRAY] * 6 + [BLUE])
ax[1].axhline(137.52, color=RED, ls='--', lw=1, label='原依据 137.52 kW')
for i, v in enumerate(vals):
    ax[1].text(i, v + 2, f'{v:.1f}', ha='center', fontsize=8)
ax[1].set_ylabel('负载 MAE (kW)'); ax[1].set_title('负载预测模型对照')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis='y'); ax[1].tick_params(axis='x', labelsize=7)
pvn = ['基线\nmean4', '附件3\n0:00预报', '晴空指数', 'LightGBM', '岭回归堆叠']
pvv = [151.95, 206.65, 213.33, 151.24, 136.85]
ax[2].bar(pvn, pvv, color=[GRAY] * 4 + [GREEN])
ax[2].axhline(151.95, color=RED, ls='--', lw=1, label='原依据 151.95 kW')
for i, v in enumerate(pvv):
    ax[2].text(i, v + 3, f'{v:.1f}', ha='center', fontsize=8)
ax[2].set_ylabel('光伏 MAE (kW)'); ax[2].set_title('光伏预测模型对照')
ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis='y'); ax[2].tick_params(axis='x', labelsize=7)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_basis.png')); plt.close()

# 图 3: "照计划执行"机制(计划 vs 实际 -> 缺口 -> 紧急购电)
fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.4), sharex=True)
for k, (d_, tag) in enumerate([('2025-06-21', '计划与实际接近：仅少量紧急购电'),
                               ('2025-09-23', '实际高于计划：缺口按 5 倍电价紧急购电')]):
    i = int(np.where(D.dates == pd.Timestamp(d_))[0][0])
    net = D.L_act[i] - D.PV_act[i]
    plan_net = R2['P'][i] + R2['D'][i] - R2['C'][i]
    ax[k].plot(t_ax, net * 6, color='k', lw=1.3, label='实际净需求 $L-PV$')
    ax[k].plot(t_ax, plan_net * 6, color=BLUE, lw=1.3, label='计划供给 $P^0+D^0-C^0$')
    ax[k].fill_between(t_ax, plan_net * 6, net * 6, where=(net > plan_net), color=RED, alpha=0.25,
                       label='缺口→5 倍紧急购电')
    ax[k].fill_between(t_ax, plan_net * 6, net * 6, where=(net < plan_net), color=GREEN, alpha=0.2,
                       label='富余→弃光')
    ax[k].set_ylabel('功率 (kW)'); ax[k].set_title(f'{d_}：{tag}')
    ax[k].legend(fontsize=7, ncol=2); ax[k].grid(alpha=0.3)
ax[1].set_xlabel('时刻')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_exec.png')); plt.close()

# 图 4: 问题三各调整方案
fig, ax = plt.subplots(2, 2, figsize=(12.4, 7.8))
opts = NU['q3_options']
nm = ['仅0:00', '+6:00', '+12:00', '+18:00', '+6,+12', '+12,+18', '全滚动']
vals = [opts[k]['total'] / 1e4 for k in nm]
cols = [GREEN if k == '+12:00' else GRAY for k in nm]
ax[0, 0].bar(nm, vals, color=cols)
for i, v in enumerate(vals):
    ax[0, 0].text(i, v + 0.6, f'{v:.1f}', ha='center', fontsize=7.5)
ax[0, 0].set_ylabel('全年费用 (万元)'); ax[0, 0].set_title('问题三：各调整方案费用(绿=采用)')
ax[0, 0].tick_params(axis='x', labelsize=7); ax[0, 0].grid(alpha=0.3, axis='y')
base = opts['仅0:00']['total']
ax[0, 1].bar(nm, [(base - opts[k]['total']) / 1e4 for k in nm],
             color=[GREEN if k == '+12:00' else (RED if base - opts[k]['total'] < 0 else GRAY) for k in nm])
ax[0, 1].axhline(0, color='k', lw=0.8)
ax[0, 1].set_ylabel('相对不调整的节省 (万元)')
ax[0, 1].set_title('调整的边际价值：只有 12:00 为正')
ax[0, 1].tick_params(axis='x', labelsize=7); ax[0, 1].grid(alpha=0.3, axis='y')
parts = np.array([[opts[k]['plan'], opts[k]['dev'], opts[k]['em']] for k in nm]) / 1e4
bot = np.zeros(len(nm))
for j, (lab, c) in enumerate([('计划购电费', BLUE), ('调整相关费', ORANGE), ('紧急购电费', RED)]):
    ax[1, 0].bar(nm, parts[:, j], bottom=bot, label=lab, color=c); bot += parts[:, j]
ax[1, 0].set_ylabel('费用 (万元)'); ax[1, 0].set_title('问题三：费用结构分解')
ax[1, 0].legend(fontsize=8); ax[1, 0].tick_params(axis='x', labelsize=7); ax[1, 0].grid(alpha=0.3, axis='y')
ax[1, 1].plot(t_ax, R3['P0'][sel] * 6, color=GRAY, lw=1.3, label='0:00 计划购电')
ax[1, 1].plot(t_ax, R3['Pf'][sel] * 6, color=GREEN, lw=1.4, label='12:00 调整后计划')
ax[1, 1].plot(t_ax, (D.L_act[sel] - D.PV_act[sel]) * 6, color='k', lw=1.0, ls=':', label='实际净需求')
ax[1, 1].axvline(12, color=RED, lw=0.8, ls='--')
ax[1, 1].set_xlabel('时刻'); ax[1, 1].set_ylabel('功率 (kW)')
ax[1, 1].set_title('2025-06-21 计划与 12:00 调整'); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q3_rolling.png')); plt.close()

# 图 5: 问题四
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
labs = ['Q4-2\n升级预测', 'Q4-2\n原依据', 'Q4-2\n预测电价', 'Q4-2\n无储能', 'Q4-3\n12:00调整']
R42p = PC.run_q2(L, PV, q=Q, pmat=P)
v = [R42['total'], R2b['total'], R42p['total'], PC.nostorage(L, PV, Q, pmat=PR), R43['total']]
ax[0].bar(labs, np.array(v) / 1e4, color=[GREEN, GRAY, BLUE, ORANGE, PURPLE])
ax[0].axhline(PC.clairvoyant(pmat=PR) / 1e4, color=RED, ls='--', lw=1, label='天眼下界')
for i, x in enumerate(v):
    ax[0].text(i, x / 1e4 + 3, f'{x/1e4:.1f}', ha='center', fontsize=7.5)
ax[0].set_ylabel('全年费用 (万元)'); ax[0].set_title('问题四：各方案费用')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y'); ax[0].tick_params(axis='x', labelsize=7)
o43 = NU['q43_options']
nm4 = list(o43.keys())
ax[1].bar(nm4, [o43[k]['total'] / 1e4 for k in nm4],
          color=[GREEN if k == '+12:00' else GRAY for k in nm4])
for i, k in enumerate(nm4):
    ax[1].text(i, o43[k]['total'] / 1e4 + 2, f'{o43[k]["total"]/1e4:.1f}', ha='center', fontsize=8)
ax[1].set_ylabel('全年费用 (万元)'); ax[1].set_title('问题四之三：调整方案对比')
ax[1].tick_params(axis='x', labelsize=7); ax[1].grid(alpha=0.3, axis='y')
p4 = PR[sel]
ax[2].plot(t_ax, p4, color=ORANGE, lw=1.3)
ax2 = ax[2].twinx()
ax2.plot(t_ax, R43['Pf'][sel] * 6, color=BLUE, lw=1.2, label='12:00 调整后购电')
ax2.plot(t_ax, (R43['D'][sel] - R43['C'][sel]) * 6, color=GREEN, lw=1.0, ls='--', label='储能净放电(计划)')
ax[2].set_xlabel('时刻'); ax[2].set_ylabel('实时电价 (元/kWh)', color=ORANGE)
ax2.set_ylabel('功率 (kW)'); ax2.legend(fontsize=7, loc='upper left')
ax[2].set_title('2025-06-21 电价与调度(问题四之三)'); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q4_fluct.png')); plt.close()
print('figures ->', FIG)
