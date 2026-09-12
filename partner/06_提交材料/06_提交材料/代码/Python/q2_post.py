# -*- coding: utf-8 -*-
"""Q2 收尾: 用已定案的M7(q*=0.6)重算链条 -> 写result2.xlsx + 图 + json
(完整五法对比数字来自上次运行日志, 此处仅重算M1/M7/M3以生成交付文件与图)"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from engine import solve_day, merge_spans, span_label, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, write_result2, dates_of_results, ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')
DATA = os.path.join(ROOT, r'04_建模求解\data')
EPS = 1e-4

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
v_const = 0.9 * price.mean()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
L_prev = pers.pivot(index='date', columns='slot', values='load_prev_kw').sort_index().to_numpy() / 6.0
PV_prev = pers.pivot(index='date', columns='slot', values='pv_prev_kw').sort_index().to_numpy() / 6.0
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
n = len(dates365)

def settlement(L, PV, P0, soc0, v):
    h = 144; p = price
    iC, iD, iE, iS = 0, h, 2 * h, 3 * h
    N = 4 * h + (h + 1)
    c = np.zeros(N); c[iE:iE + h] = 5.0 * p; c[iC:iC + h] = EPS; c[iD:iD + h] = EPS; c[iS + h] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        row = np.zeros(N); row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        row = np.zeros(N); row[iS + t] = -1; row[iS + t + 1] = 1
        row[iC + t] = -ETA_C; row[iD + t] = 1.0 / ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iS:iS + h + 1] = SOC_MIN; ub[iS:iS + h + 1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success
    return (res.x[iE:iE + h], res.x[iC:iC + h], res.x[iD:iD + h], res.x[iS + h], float(5.0 * p @ res.x[iE:iE + h]))

def chain(basis_L, basis_PV):
    soc0 = 6000.0
    P_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    s0 = np.zeros(n); s24 = np.zeros(n); cp = np.zeros(n); E_m = np.zeros((n, 144)); ce = np.zeros(n)
    for i in range(n):
        v = 0.0 if i == n - 1 else v_const
        P0 = solve_day(price, basis_L[i], basis_PV[i], soc0, soc_end=None, v=v)['P']
        E, C, D, s24r, cei = settlement(L_act[i], PV_act[i], P0, soc0, v)
        P_m[i] = P0; C_m[i] = C; D_m[i] = D; E_m[i] = E; cp[i] = float(price @ P0); ce[i] = cei
        s0[i] = soc0; s24[i] = s24r; soc0 = s24r
    return P_m, C_m, D_m, s0, s24, cp, E_m, ce

r = (L_act - PV_act) - (L_prev - PV_prev)
q = np.zeros((n, 144))
for i in range(n):
    lo = max(0, i - 30)
    q[i] = np.quantile(r[lo:i], 0.6, axis=0) if i - lo >= 7 else 0.0
L_buf = L_prev + q
t0 = time.time()
M1 = chain(L_prev, PV_prev)
M7 = chain(L_buf, PV_prev)
# 天眼
soc0 = 6000.0; cp3 = np.zeros(n)
for i in range(n):
    v = 0.0 if i == n - 1 else v_const
    rr = solve_day(price, L_act[i], PV_act[i], soc0, soc_end=None, v=v)
    cp3[i] = rr['gross']; soc0 = rr['E'][-1]
print(f'重算耗时={time.time()-t0:.1f}s; M1总={ (M1[5]+M1[7])[np.where(dates365>=pd.Timestamp("2025-02-01"))[0][0]:].sum():,.0f}; M7总={ (M7[5]+M7[7])[np.where(dates365>=pd.Timestamp("2025-02-01"))[0][0]:].sum():,.0f}')

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
totM1 = M1[5] + M1[7]; totM7 = M7[5] + M7[7]
# 上次全量日志中的其他方法 (2.1-12.31)
TOTS = dict(M1=float(totM1[m:].sum()), M2=22847374.0, M4=23988876.0, M5=19595720.0,
            M6=30584132.0, M7=float(totM7[m:].sum()), M3=float(cp3[m:].sum()))

res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in dates365])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
P_r, C_r, D_r = M7[0][i0:i1], M7[1][i0:i1], M7[2][i0:i1]
s0_r, s24_r, cp_r, E_r = M7[3][i0:i1], M7[4][i0:i1], M7[5][i0:i1], M7[6][i0:i1]
emerg = []
for i in range(len(res_dates)):
    idx = np.where(E_r[i] > 1e-6)[0] + 1
    emerg.append([(span_label(a, b), float(E_r[i][a - 1:b].sum())) for a, b in merge_spans(idx)])
write_result2('result2.xlsx', res_dates, P_r, cp_r, s0_r, s24_r, C_r, D_r, emerg)

spec = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
slot_of = {'10:00-10:10': 61, '12:00-12:10': 73, '14:00-14:10': 85, '16:00-16:10': 97, '18:00-18:10': 109, '20:00-20:10': 121}
paper = {}
for ds in spec:
    k = list(dates365).index(pd.Timestamp(ds)) - i0
    paper[ds] = dict(
        t1={lab: round(float(P_r[k, v - 1]), 2) for lab, v in slot_of.items()},
        total=round(float(P_r[k].sum()), 2), cost=round(float(cp_r[k]), 2),
        blocks=[(round(float(C_r[k, b * 24:(b + 1) * 24].sum()), 2), round(float(D_r[k, b * 24:(b + 1) * 24].sum()), 2)) for b in range(6)],
        soc0=round(float(s0_r[k]), 2), soc24=round(float(s24_r[k]), 2),
        emerg=[(span_label(a, b), round(float(E_r[k][a - 1:b].sum()), 2)) for a, b in merge_spans(np.where(E_r[k] > 1e-6)[0] + 1)])
print('[论文表]', json.dumps(paper, ensure_ascii=False, indent=1))

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
ax = axes[0]
ax.plot(dates365, totM1, color=GRAY, lw=0.8, label='M1 外推计划')
ax.plot(dates365, totM7, color=BLUE, lw=0.9, label='M7 外推+q*报童(采用)')
ax.plot(dates365, cp3, color=ORANGE, lw=0.8, label='M3 天眼下界')
ax.set_title('全年逐日购电总费用 (元)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7); ax.tick_params(axis='x', rotation=45, labelsize=6)
ax = axes[1]
em_m = pd.Series(E_r.sum(axis=1), index=dates365[i0:i1]).groupby(dates365[i0:i1].month).sum()
em_1 = pd.Series(M1[6].sum(axis=1), index=dates365).groupby(dates365.month).sum()
x = np.arange(len(em_m)); w = 0.4
ax.bar(x - w / 2, em_1.to_numpy(), w, color=GRAY, alpha=0.8, label='M1 紧急电量')
ax.bar(x + w / 2, em_m.to_numpy(), w, color=RED, alpha=0.85, label='M7 紧急电量')
ax.set_xticks(x); ax.set_xticklabels([f'{k}月' for k in em_m.index], fontsize=6)
ax.set_title('紧急购电量按月汇总 (kWh)', fontsize=9); ax.grid(alpha=0.3, axis='y', ls='--')
ax.legend(frameon=False, fontsize=7)
ax = axes[2]
k = list(dates365).index(pd.Timestamp('2025-06-21'))
t_axis = (np.arange(144) + 0.5) / 6
ax.plot(t_axis, L_act[k] * 6, color=BLUE, lw=1.0, label='实际负载')
ax.plot(t_axis, PV_act[k] * 6, color=ORANGE, lw=1.0, label='实际光伏')
ax.plot(t_axis, P_r[k - i0] * 6, color=GREEN, lw=1.0, label='计划购电(M7)')
ax.fill_between(t_axis, 0, E_r[k - i0] * 6, color=RED, alpha=0.5, label='紧急购电')
ax.set_title('2025-06-21 计划 vs 实际 (kW)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_q2_annual.png')); plt.close(fig)

with open(os.path.join(DATA, 'q2_results.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(best='M7', tots=TOTS, em_m7_kwh=float(E_r.sum()),
                   em_m1_kwh=float(M1[6][m:].sum()), paper=paper), f, ensure_ascii=False, indent=1)
print('Q2 POST DONE')
