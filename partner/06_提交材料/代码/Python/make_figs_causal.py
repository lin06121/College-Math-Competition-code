# -*- coding: utf-8 -*-
"""按"因果口径"(0:00 计划 + 日内实时平衡控制)重绘论文图:
   fig_q2_annual.png / fig_q2_basis.png / fig_q3_rolling.png / fig_q4_fluct.png / fig_q2_balance.png
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from causal_core import (D, price, PR, run_q2, run_q3, clairvoyant, nostorage, buffer_of, balance,
                         ROOT, P_MAX)
from engine import solve_day, merge_spans, span_label

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')
os.makedirs(FIG, exist_ok=True)
m2 = D.m2
dates = D.dates[m2:]
res = {}

# ---------------- 问题二 ----------------
R2 = run_q2(q=0.7, reserve=1200.0)
R2n = run_q2(q=0.0, reserve=1200.0)
CB = clairvoyant()
NS = nostorage(q=0.7)
res['q2'] = dict(adopted=R2['total'], plan=R2['plan'], em=R2['em'], nomargin=R2n['total'],
                 nostorage=NS, bound=CB, em_kwh=float(R2['Em'][m2:].sum()))
cp = (R2['pc'] + R2['ec'])[m2:]
cpn = (R2n['pc'] + R2n['ec'])[m2:]
cns = np.zeros(len(dates)); cbo = np.zeros(len(dates))
for k, i in enumerate(range(m2, D.n)):
    P0 = np.maximum(D.Lb[i] + buffer_of(i, 0.7) - D.Pb[i], 0.0)
    Em = np.maximum(D.L_act[i] - D.PV_act[i] - P0, 0.0)
    cns[k] = float(price @ P0) + float(5 * price @ Em)
cbo[:] = CB / len(dates)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
mo = np.array([d.month for d in dates])
ax[0].plot(dates, cp / 1000, color=BLUE, lw=0.9, label='采用方案(同类日+报童+实时平衡)')
ax[0].plot(dates, cpn / 1000, color=ORANGE, lw=0.8, alpha=0.8, label='同类日(无报童裕度)')
ax[0].plot(dates, cns / 1000, color=GRAY, lw=0.8, alpha=0.8, label='无储能')
ax[0].axhline(CB / len(dates) / 1000, color=RED, ls='--', lw=1.0, label='天眼链式下界(日均)')
ax[0].set_xlabel('日期'); ax[0].set_ylabel('日费用 (千元)')
ax[0].set_title('问题二：全年逐日购电总费用'); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
emk = np.array([float(R2['Em'][m2 + k].sum()) for k in range(len(dates))])
bym = np.array([emk[mo == m].sum() for m in range(2, 13)])
ax[1].bar(range(2, 13), bym / 1000, color=ORANGE)
ax[1].set_xticks(range(2, 13)); ax[1].set_xticklabels([f'{m}' for m in range(2, 13)])
ax[1].set_xlabel('月份'); ax[1].set_ylabel('紧急购电量 (MWh)')
ax[1].set_title('问题二：紧急购电量按月汇总'); ax[1].grid(alpha=0.3, axis='y')
s = 144 * 10 // 60
hd = [f'{t // 6}:{(t % 6) * 10:02d}' for t in range(144)]
sel = np.where(D.dates == pd.Timestamp('2025-06-21'))[0][0]
t_ax = np.arange(144) * 10 / 60
ax[2].plot(t_ax, D.L_act[sel] * 6, color=BLUE, lw=1.2, label='实际负载')
ax[2].plot(t_ax, D.PV_act[sel] * 6, color=GREEN, lw=1.2, label='实际光伏')
ax[2].plot(t_ax, R2['P'][sel] * 6, color=ORANGE, lw=1.2, label='计划购电')
ax[2].plot(t_ax, (R2['D'][sel] - R2['C'][sel]) * 6, color=RED, lw=1.0, ls='--', label='储能净放电')
ax[2].set_xlabel('时刻'); ax[2].set_ylabel('功率 (kW)')
ax[2].set_title('2025-06-21 计划与实际'); ax[2].legend(fontsize=7); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_annual.png')); plt.close()

# ---------------- 问题二: 计划依据对比 ----------------
Lb_per = D.L_act[np.r_[0, np.arange(D.n - 1)]]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(D.Lb[sel] * 6, color=BLUE, lw=1.4, label='同类星期平均(本文)')
ax[0].plot(Lb_per[sel] * 6, color=ORANGE, lw=1.2, label='昨日外推')
ax[0].plot(D.L_typ * 6, color=GRAY, lw=1.0, label='附件1 典型日曲线')
ax[0].plot(D.L_act[sel] * 6, color='k', lw=1.0, ls=':', label='实际负载')
ax[0].set_xlabel('时刻'); ax[0].set_ylabel('负载 (kW)'); ax[0].set_title('2025-06-21 负载依据对比')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
errs = [np.abs(D.Lb[m2:] - D.L_act[m2:]).mean() * 6, np.abs(Lb_per[m2:] - D.L_act[m2:]).mean() * 6,
        np.abs(D.L_typ - D.L_act[m2:]).mean() * 6]
ax[1].bar(['同类星期平均', '昨日外推', '附件1典型日'], errs, color=[BLUE, ORANGE, GRAY])
for i, e in enumerate(errs):
    ax[1].text(i, e + 8, f'{e:.1f}', ha='center', fontsize=9)
ax[1].set_ylabel('负载预测 MAE (kW)'); ax[1].set_title('三种计划依据的精度(2.1-12.31)')
ax[1].grid(alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_basis.png')); plt.close()
res['q2_basis_mae'] = errs

# ---------------- 问题三 ----------------
q3 = {}
for at, name in [((), '仅用0:00预报'), ((6,), '+6:00'), ((12,), '+12:00'), ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]:
    q3[name] = run_q3(adjust=at)
res['q3'] = {k: dict(tot=v['total'], plan=v['plan'], dev=v['dev'], em=v['em']) for k, v in q3.items()}
names = list(q3.keys())
fig, ax = plt.subplots(2, 2, figsize=(12, 7.6))
vals = [q3[k]['total'] / 1e4 for k in names]
ax[0, 0].bar(names, vals, color=[GRAY, GRAY, BLUE, GRAY, GREEN])
ax[0, 0].set_ylabel('全年费用 (万元)'); ax[0, 0].set_title('问题三：滚动方案全年费用对比')
for i, v in enumerate(vals):
    ax[0, 0].text(i, v + 0.6, f'{v:.1f}', ha='center', fontsize=8)
ax[0, 0].grid(alpha=0.3, axis='y'); ax[0, 0].tick_params(axis='x', labelsize=8)
base = q3['仅用0:00预报']['total']
marg = [base - q3[k]['total'] for k in names[1:]]
ax[0, 1].bar(['+6:00', '+12:00', '+6,+12', '全滚动'], np.array(marg) / 1e4,
             color=[GRAY, BLUE, GRAY, GREEN])
ax[0, 1].axhline(0, color='k', lw=0.8)
ax[0, 1].set_ylabel('相对仅0:00预报的节约 (万元)')
ax[0, 1].set_title('滚动调整的边际价值'); ax[0, 1].grid(alpha=0.3, axis='y')
parts = np.array([[q3[k]['plan'], q3[k]['dev'], q3[k]['em']] for k in names]) / 1e4
bot = np.zeros(len(names))
for j, (lab, c) in enumerate([('计划购电费', BLUE), ('调整相关费', ORANGE), ('紧急购电费', RED)]):
    ax[1, 0].bar(names, parts[:, j], bottom=bot, label=lab, color=c)
    bot += parts[:, j]
ax[1, 0].set_ylabel('费用 (万元)'); ax[1, 0].set_title('问题三：费用结构分解')
ax[1, 0].legend(fontsize=8); ax[1, 0].tick_params(axis='x', labelsize=7); ax[1, 0].grid(alpha=0.3, axis='y')
sell = np.where(D.dates == pd.Timestamp('2025-06-21'))[0][0]
R30 = q3['仅用0:00预报']; R3F = q3['全滚动']
ax[1, 1].plot(t_ax, R30['P0'][sell] * 6, color=GRAY, lw=1.2, label='0:00 计划购电')
ax[1, 1].plot(t_ax, R3F['Pf'][sell] * 6, color=GREEN, lw=1.4, label='全滚动调整后购电')
ax[1, 1].plot(t_ax, D.L_act[sell] * 6 - D.PV_act[sell] * 6, color='k', lw=1.0, ls=':', label='实际净需求')
ax[1, 1].set_xlabel('时刻'); ax[1, 1].set_ylabel('功率 (kW)')
ax[1, 1].set_title('2025-06-21 计划与滚动调整'); ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q3_rolling.png')); plt.close()

# ---------------- 问题四 ----------------
R42 = run_q2(p_matrix=PR, q=0.7, reserve=1200.0)
R42n = run_q2(p_matrix=PR, q=0.0, reserve=1200.0)
R43 = run_q3(p_matrix=PR, adjust=(6, 12, 18))
R430 = run_q3(p_matrix=PR, adjust=())
CB4 = clairvoyant(p_matrix=PR); NS4 = nostorage(q=0.7, p_matrix=PR)
res['q4'] = dict(q42=R42['total'], q42_plan=R42['plan'], q42_em=R42['em'], q42_nomargin=R42n['total'],
                 q43=R43['total'], q43_plan=R43['plan'], q43_dev=R43['dev'], q43_em=R43['em'],
                 q43_zero=R430['total'], nostorage=NS4, bound=CB4)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
labs = ['Q4-2\n采用', 'Q4-2\n无裕度', 'Q4-2\n无储能', 'Q4-3\n仅0:00', 'Q4-3\n全滚动']
v = [R42['total'], R42n['total'], NS4, R430['total'], R43['total']]
ax[0].bar(labs, np.array(v) / 1e4, color=[GREEN, GRAY, ORANGE, GRAY, BLUE])
ax[0].axhline(CB4 / 1e4, color=RED, ls='--', lw=1.0, label='天眼下界')
for i, x in enumerate(v):
    ax[0].text(i, x / 1e4 + 3, f'{x/1e4:.1f}', ha='center', fontsize=8)
ax[0].set_ylabel('全年费用 (万元)'); ax[0].set_title('问题四：波动电价下各方案费用')
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis='y'); ax[0].tick_params(axis='x', labelsize=7)
ax[1].bar(['仅0:00预报', '全滚动'], [R430['total'] / 1e4, R43['total'] / 1e4], color=[GRAY, BLUE])
for i, x in enumerate([R430['total'], R43['total']]):
    ax[1].text(i, x / 1e4 + 3, f'{x/1e4:.1f}', ha='center', fontsize=9)
ax[1].set_ylabel('全年费用 (万元)'); ax[1].set_title(f'Q4-3 滚动调整价值 {(R430["total"]-R43["total"])/1e4:.1f} 万元')
ax[1].grid(alpha=0.3, axis='y')
p4 = PR[sell]
ax[2].plot(t_ax, p4, color=ORANGE, lw=1.3)
ax2 = ax[2].twinx()
ax2.plot(t_ax, R43['Pf'][sell] * 6, color=BLUE, lw=1.2, label='全滚动调整后购电')
ax2.plot(t_ax, (R43['D'][sell] - R43['C'][sell]) * 6, color=GREEN, lw=1.0, ls='--', label='储能净放电')
ax[2].set_xlabel('时刻'); ax[2].set_ylabel('实时电价 (元/kWh)', color=ORANGE)
ax2.set_ylabel('功率 (kW)')
ax[2].set_title('2025-06-21 电价与调度(问题四之三)')
ax2.legend(fontsize=7, loc='upper left'); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q4_fluct.png')); plt.close()

# ---------------- 实时平衡控制过程图(新增) ----------------
fig, ax = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
for k, (sel_d, tag) in enumerate([('2025-03-20', '计划优于实际(储能吸收富余)'), ('2025-09-23', '实际高于计划(储能兜底→紧急)')]):
    i = np.where(D.dates == pd.Timestamp(sel_d))[0][0]
    net = D.L_act[i] - D.PV_act[i]
    ax[k].plot(t_ax, net * 6, color='k', lw=1.3, label='实际净需求 $L-PV$')
    ax[k].plot(t_ax, R2['P'][i] * 6, color=BLUE, lw=1.2, label='计划购电 $P^0$')
    ax[k].plot(t_ax, (D.L_act[i] - D.PV_act[i] - R2['P'][i]) * 6, color=GRAY, lw=1.0, ls='--',
               label='计划口径缺口')
    ax[k].plot(t_ax, (R2['D'][i] - R2['C'][i]) * 6, color=GREEN, lw=1.1, label='实时储能净放电')
    ax[k].plot(t_ax, R2['Em'][i] * 6, color=RED, lw=1.6, label='紧急购电(5 倍价)')
    ax[k].set_ylabel('功率 (kW)'); ax[k].set_title(f'{sel_d}：{tag}')
    ax[k].legend(fontsize=7, ncol=3); ax[k].grid(alpha=0.3)
ax[1].set_xlabel('时刻')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_q2_balance.png')); plt.close()

json.dump(res, open(os.path.join(ROOT, r'04_建模求解\data\fig_numbers_causal.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: float(o)))
print('figures saved ->', FIG)
