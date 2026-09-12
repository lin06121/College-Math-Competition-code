# -*- coding: utf-8 -*-
"""口径 A(最终) 论文插图重绘:
   fig_q2_annual / fig_q2_basis / fig_q2_balance / fig_q2_exec / fig_q3_rolling / fig_q4_fluct
   数据来源: A_all_numbers.json(数字) + A_core(逐时轨迹)
"""
import sys, io, os, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from causal_core import D, price, PR, ROOT
import A_core as AC

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888', '#7B52AB'
FIG = os.path.join(ROOT, '04_建模求解', 'fig')
PAPER_FIG = os.path.join(ROOT, '05_论文与结果', '论文', 'figures')
os.makedirs(FIG, exist_ok=True)
A = json.load(open(os.path.join(ROOT, r'04_建模求解\data\A_all_numbers.json'), encoding='utf-8'))
M = A['main']; Q = M['q_star']; Q3 = float(M.get('q3_star', Q)); n, m2 = D.n, D.m2
L, PV, P_RT = AC.L, AC.PV, AC.P_RT
import price_fc as PF
P_HAT, _, _PD = PF.build_price()
P_ROLL = PF.build_price_rolling(P_HAT, damp=0.7)
t_ax = np.arange(144) * 10 / 60
sel = int(np.where(D.dates == pd.Timestamp('2025-06-21'))[0][0])
print('跑管道 ...')
R2 = AC.run_q2(q=Q)                                  # 采用方案(问题二)
R2b = AC.run_q2(D.Lb, D.Pb, q=Q)                     # 原依据(同类日均值+近4日光伏)对照
R30 = AC.run_q3(q=Q3, adjust=(), use_adjbuf=True)     # 问题三: 不调整
R312 = AC.run_q3(q=Q3, adjust=(6, 12, 18), use_adjbuf=True)  # 问题三: 12:00 调整(采用)
R42 = AC.run_q2(q=Q, pmat=P_HAT, bill=PR)            # 问题四之二(预测电价定计划、实际电价结算)
R42b = AC.run_q2(D.Lb, D.Pb, q=Q, pmat=P_HAT, bill=PR)   # 问题四之二 原依据对照
R4312 = AC.run_q3(q=Q3, pmat=P_HAT, bill=PR, pmat_adj=P_ROLL, adjust=(6, 12, 18), use_adjbuf=True)   # 问题四之三: 全滚动(采用)
print(f'  采用 Q2 {R2["total"]:,.0f} | 原依据 {R2b["total"]:,.0f} | Q3 +12 {R312["total"]:,.0f} '
      f'| Q4-2(预测电价) {R42["total"]:,.0f} | Q4-3 全滚动 {R4312["total"]:,.0f}')

# ---------- 图 1: 问题二全年 ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
cp = (R2['pc'] + R2['ec'])[m2:] / 1000
cpb = (R2b['pc'] + R2b['ec'])[m2:] / 1000
cns = np.zeros(len(cp))
for k, i in enumerate(range(m2, n)):
    b = AC.buffer_of(L, PV, Q, i)
    P0 = np.maximum(L[i] + b - PV[i], 0.0)
    Em = np.maximum(D.L_act[i] - D.PV_act[i] - P0, 0.0)
    cns[k] = (price @ P0 + 5 * price @ Em) / 1000
dates = D.dates[m2:]
ax[0].plot(dates, cp, color=BLUE, lw=0.9, label='采用(升级预测)')
ax[0].plot(dates, cpb, color=GRAY, lw=0.8, alpha=0.85, label='原依据(同类日+近4日)')
ax[0].plot(dates, cns, color=ORANGE, lw=0.8, alpha=0.8, label='无储能')
ax[0].axhline(A['cmp_天眼下界'] / len(cp) / 1000, color=RED, ls='--', lw=1.0, label='天眼链式下界(日均)')
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
cols = [GRAY] * 6 + [BLUE]
ax[1].bar(names, vals, color=cols)
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

# ---------- 图 3: 平衡响应规则过程 ----------
fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
for k, (d_, tag) in enumerate([('2025-03-20', '实际低于计划：富余先减放、再多充（无紧急购电）'),
                               ('2025-09-23', '实际高于计划：缺额先减充、再多放，仍缺才 5 倍紧急购电')]):
    i = int(np.where(D.dates == pd.Timestamp(d_))[0][0])
    net = D.L_act[i] - D.PV_act[i]
    ax[k].plot(t_ax, net * 6, color='k', lw=1.3, label='实际净需求 $L-PV$')
    ax[k].plot(t_ax, R2['P'][i] * 6, color=BLUE, lw=1.2, label='计划购电 $P^0$')
    ax[k].plot(t_ax, (net - R2['P'][i]) * 6, color=GRAY, lw=1.0, ls='--', label='计划口径缺口')
    ax[k].plot(t_ax, (R2['D'][i] - R2['C'][i]) * 6, color=GREEN, lw=1.1, label='最终储能净放电')
    ax[k].plot(t_ax, R2['Em'][i] * 6, color=RED, lw=1.6, label='紧急购电(5 倍价)')
    ax[k].set_ylabel('功率 (kW)'); ax[k].set_title(f'{d_}：{tag}')
    ax[k].legend(fontsize=7, ncol=3); ax[k].grid(alpha=0.3)
ax[1].set_xlabel('时刻')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_balance.png')); plt.close()

# ---------- 图 4: 同一计划的三种执行机制 ----------
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
labs = ['僵硬执行计划\n(无日内平衡)', '实时平衡响应规则\n(采用)', '完美信息再调度\n(不可实行)']
E = [A['exec_asplanned'], A['exec_balance'], A['exec_perfect']]
tot = [e['total'] / 1e4 for e in E]
pl = [e['plan'] / 1e4 for e in E]
em = [e['em'] / 1e4 for e in E]
xp = np.arange(3)
ax[0].bar(xp, pl, color=BLUE, label='计划购电费')
ax[0].bar(xp, em, bottom=pl, color=RED, label='紧急购电费')
for i, t in enumerate(tot):
    ax[0].text(i, t + 8, f'{t:.1f}', ha='center', fontsize=9)
ax[0].set_xticks(xp); ax[0].set_xticklabels(labs, fontsize=8)
ax[0].set_ylabel('全年费用 (万元)'); ax[0].set_title('同一 0:00 计划、三种执行机制的全年费用')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y')
gam = (E[0]['total'] - E[1]['total']) / 1e4
ax[0].annotate('', xy=(0.5, tot[1] + 30), xytext=(0.5, tot[0] + 30),
               arrowprops=dict(arrowstyle='<->', color='k', lw=1.0))
ax[0].text(0.62, (tot[0] + tot[1]) / 2 + 30, f'响应规则价值\n{gam:.1f} 万元', fontsize=8)
ax[1].plot(dates, R2['Em'][m2:].sum(axis=1), color=BLUE, lw=0.8, label='采用(实时平衡)')
ax[1].set_xlabel('日期'); ax[1].set_ylabel('日紧急购电量 (kWh)')
ax[1].set_title(f'逐日紧急购电量：采用方案合计 {R2["em_kwh"]/1000:.0f} MWh')
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_exec.png')); plt.close()

# ---------- 图 5: 问题三各调整方案 ----------
opts = A['q3_options']
nm = ['仅0:00', '+6:00', '+12:00\n(采用)', '+18:00', '+6,+12', '+12,+18', '全滚动']
keys = ['仅0:00', '+6:00', '+12:00', '+18:00', '+6,+12', '+12,+18', '全滚动']
fig, ax = plt.subplots(2, 2, figsize=(12.4, 7.8))
vals = [opts[k]['total'] / 1e4 for k in keys]
ax[0, 0].bar(nm, vals, color=[GRAY, ORANGE, GREEN, ORANGE, GRAY, GRAY, GRAY])
for i, v in enumerate(vals):
    ax[0, 0].text(i, max(vals) * 0.02 + v, f'{v:.1f}', ha='center', fontsize=8)
ax[0, 0].set_ylim(min(vals) * 0.96, max(vals) * 1.03)
ax[0, 0].set_ylabel('全年费用 (万元)'); ax[0, 0].set_title('问题三：各调整方案全年费用')
ax[0, 0].grid(alpha=0.3, axis='y'); ax[0, 0].tick_params(axis='x', labelsize=7)
base = opts['仅0:00']['total']
marg = [(base - opts[k]['total']) / 1e4 for k in keys]
cols2 = [GREEN if m > 0 else RED for m in marg]
ax[0, 1].bar(range(len(keys)), marg, color=cols2)
ax[0, 1].axhline(0, color='k', lw=0.8)
for i, m in enumerate(marg):
    ax[0, 1].text(i, m + (0.8 if m > 0 else -1.6), f'{m:+.1f}', ha='center', fontsize=8)
ax[0, 1].set_xticks(range(len(keys))); ax[0, 1].set_xticklabels(nm, fontsize=7)
ax[0, 1].set_ylabel('相对"仅0:00"的节约 (万元)')
ax[0, 1].set_title('边际价值：12:00 调整最有效，6:00 调整反而亏损')
ax[0, 1].grid(alpha=0.3, axis='y')
parts = np.array([[opts[k]['plan'], opts[k]['dev'], opts[k]['em']] for k in keys]) / 1e4
bot = np.zeros(len(keys))
for j, (lab, c) in enumerate([('计划购电费', BLUE), ('调整相关费', ORANGE), ('紧急购电费', RED)]):
    ax[1, 0].bar(range(len(keys)), parts[:, j], bottom=bot, label=lab, color=c); bot += parts[:, j]
ax[1, 0].set_xticks(range(len(keys))); ax[1, 0].set_xticklabels(nm, fontsize=7)
ax[1, 0].set_ylabel('费用 (万元)'); ax[1, 0].set_title('问题三：费用结构分解（调整费换紧急费）')
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3, axis='y')
ax[1, 1].plot(t_ax, R30['P0'][sel] * 6, color=GRAY, lw=1.2, label='0:00 计划')
ax[1, 1].plot(t_ax, R312['Pf'][sel] * 6, color=GREEN, lw=1.4, label='12:00 调整后')
ax[1, 1].plot(t_ax, D.L_act[sel] * 6 - D.PV_act[sel] * 6, color='k', lw=1.0, ls=':', label='实际净需求')
ax[1, 1].axvline(12, color=RED, ls='--', lw=0.9)
ax[1, 1].text(12.15, ax[1, 1].get_ylim()[1] * 0.85, '12:00 调整时刻', fontsize=7, color=RED)
ax[1, 1].set_xlabel('时刻'); ax[1, 1].set_ylabel('功率 (kW)')
ax[1, 1].set_title('2025-06-21 计划与 12:00 调整'); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q3_rolling.png')); plt.close()

# ---------- 图 6: 问题四 ----------
o43 = A['q43_options']
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
labs = ['Q4-2\n预测电价\n甲(采用)', 'Q4-2\n预测电价\n乙(对照)', 'Q4-2\n已知实际电价\n(上界,不可达)',
        'Q4-2\n原依据', 'Q4-2\n无储能', 'Q4-3\n全滚动\n(采用)']
v = [M['q42']['total'], M['q42']['price_yi'], M['q42']['price_known_bound'],
     R42b['total'], A['q4_nostorage'], o43['全滚动']['total']]
ax[0].bar(labs, np.array(v) / 1e4, color=[GREEN, GRAY, BLUE, PURPLE, ORANGE, GREEN])
for i, x in enumerate(v):
    ax[0].text(i, x / 1e4 + 5, f'{x/1e4:.1f}', ha='center', fontsize=7.5)
ax[0].set_ylabel('全年费用 (万元)'); ax[0].set_title('问题四：各方案全年费用')
ax[0].axhline(A['q4_bound'] / 1e4, color=RED, ls='--', lw=1, label='天眼下界')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y'); ax[0].tick_params(axis='x', labelsize=6.5)
d43 = (o43['仅0:00']['total'] - o43['全滚动']['total']) / 1e4
ax[1].bar(['Q4-3 仅 0:00\n(不调整)', 'Q4-3 仅 12:00', 'Q4-3 全滚动\n(采用)'],
          [o43['仅0:00']['total'] / 1e4, o43['+12:00']['total'] / 1e4, o43['全滚动']['total'] / 1e4],
          color=[GRAY, PURPLE, GREEN])
_v3 = [o43['仅0:00']['total'], o43['+12:00']['total'], o43['全滚动']['total']]
for i, x in enumerate(_v3):
    ax[1].text(i, x / 1e4 + 2, f'{x/1e4:.1f}', ha='center', fontsize=9)
ax[1].set_ylabel('全年费用 (万元)')
ax[1].set_title(f'波动电价下 全时刻滚动的价值 {(_v3[0]-_v3[2])/1e4:.1f} 万元')
ax[1].grid(alpha=0.3, axis='y'); ax[1].tick_params(axis='x', labelsize=8)
p4 = PR[sel]
ax[2].plot(t_ax, p4, color=ORANGE, lw=1.3)
ax2 = ax[2].twinx()
ax2.plot(t_ax, R4312['Pf'][sel] * 6, color=BLUE, lw=1.2, label='逐段调整后购电')
ax2.plot(t_ax, (R4312['D'][sel] - R4312['C'][sel]) * 6, color=GREEN, lw=1.0, ls='--', label='储能净放电')
ax[2].set_xlabel('时刻'); ax[2].set_ylabel('实时电价 (元/kWh)', color=ORANGE)
ax2.set_ylabel('功率 (kW)'); ax2.legend(fontsize=7, loc='upper left')
ax[2].set_title('2025-06-21 实时电价与调度(问题四之三)'); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q4_fluct.png')); plt.close()

# ---------- 复制到论文目录 ----------
for f in ['fig_q2_annual.png', 'fig_q2_basis.png', 'fig_q2_balance.png', 'fig_q2_exec.png',
          'fig_q3_rolling.png', 'fig_q4_fluct.png']:
    shutil.copyfile(os.path.join(FIG, f), os.path.join(PAPER_FIG, f))
print('figures ->', FIG, '&', PAPER_FIG)
