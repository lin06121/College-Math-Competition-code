# -*- coding: utf-8 -*-
"""Q4: 波动电价(附件4)下重算Q2与Q3
result4-2 = Q2机制 + 附件4电价 (M1外推 / M7报童q* / M4无储能 / M3天眼)
result4-3 = Q3机制 + 附件4电价 (仅0:00 / 全滚动6/12/18 / 边际价值 / 天眼)
"""
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
from io_results import load_all, write_result2, write_result3, dates_of_results, ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')
DATA = os.path.join(ROOT, r'04_建模求解\data')
EPS = 1e-4

day, typ, fc0, fca, pers, tmpl, const = load_all()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
PR = day.pivot(index='date', columns='slot', values='price_q4').sort_index().to_numpy()
fc0m = fc0.pivot(index='date', columns='slot', values='fc0_kw').sort_index().to_numpy() / 6.0
fca_m = {h: fca[fca.issue_hour == h].pivot(index='date', columns='slot', values='fc_kw').sort_index().reindex(columns=range(1, 145)).to_numpy() / 6.0 for h in (6, 12, 18)}
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
n = len(dates365)
# 计划依据升级(与Q2/Q3一致): 负载=近4个同类日平均; 光伏=近3日均值
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
dow_q4 = np.array([d.dayofweek for d in dates365])
L_prev = np.zeros_like(L_act)
for i in range(n):
    low_set = {4, 5}
    hist = [j for j in range(i) if (dow_q4[j] in low_set) == (dow_q4[i] in low_set)]
    L_prev[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
PV_prev = np.zeros_like(PV_act)
for i in range(n):
    hist = list(range(max(0, i - 3), i))
    PV_prev[i] = PV_act[hist].mean(axis=0) if hist else PV_typ
v_next = np.zeros(n)
for i in range(n - 1):
    v_next[i] = 0.9 * PR[i + 1].mean()

def settlement(L, PV, P0, soc0, v, p):
    h = 144
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

def chain_q2(basis_L, basis_PV):
    soc0 = 6000.0
    P_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    s0 = np.zeros(n); s24 = np.zeros(n); cp = np.zeros(n); E_m = np.zeros((n, 144)); ce = np.zeros(n)
    for i in range(n):
        p = PR[i]
        P0 = solve_day(p, basis_L[i], basis_PV[i], soc0, soc_end=None, v=v_next[i])['P']
        E, C, D, s24r, cei = settlement(L_act[i], PV_act[i], P0, soc0, v_next[i], p)
        P_m[i] = P0; C_m[i] = C; D_m[i] = D; E_m[i] = E
        cp[i] = float(p @ P0); ce[i] = cei
        s0[i] = soc0; s24[i] = s24r; soc0 = s24r
    return P_m, C_m, D_m, s0, s24, cp, E_m, ce

# q* 调优 (1月, 波动电价下)
r = (L_act - PV_act) - (L_prev - PV_prev)
def buf(qlevel):
    q = np.zeros((n, 144))
    for i in range(n):
        lo = max(0, i - 30)
        q[i] = np.quantile(r[lo:i], qlevel, axis=0) if i - lo >= 7 else 0.0
    return L_prev + q
t0 = time.time()
jan_cost = {}
for ql in [0.5, 0.6, 0.7, 0.8]:
    Mq = chain_q2(buf(ql), PV_prev)
    jan_m = np.where(dates365 >= pd.Timestamp('2025-01-08'))[0][0]
    jan_cost[ql] = (Mq[5] + Mq[7])[jan_m:].sum()
q_star = min(jan_cost, key=jan_cost.get)
print(f'[Q4-2 q*调优] 1月验证: { {k: round(v) for k, v in jan_cost.items()} } -> q*={q_star}')
M1 = chain_q2(L_prev, PV_prev)
M7 = chain_q2(buf(q_star), PV_prev)
P4 = np.maximum(buf(0.8) - PV_prev, 0)
cp4 = (P4 * PR).sum(axis=1)
E4 = np.maximum(L_act - PV_act - P4, 0)
ce4 = (5.0 * E4 * PR).sum(axis=1)
soc0 = 6000.0; cp3 = np.zeros(n)
for i in range(n):
    rr = solve_day(PR[i], L_act[i], PV_act[i], soc0, soc_end=None, v=v_next[i])
    cp3[i] = rr['gross']; soc0 = rr['E'][-1]
t_q42 = time.time() - t0

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
tot = lambda M: (M[5] + M[7])[m:].sum()
print(f'===== Q4-2 (波动电价, 2.1-12.31) =====')
print(f'M1 外推: 计划费={M1[5][m:].sum():,.0f} + 紧急费={M1[7][m:].sum():,.0f} = 总{tot(M1):,.0f} 元 | 紧急电量={M1[6][m:].sum():,.0f} kWh')
print(f'M7 报童q*={q_star}: 计划费={M7[5][m:].sum():,.0f} + 紧急费={M7[7][m:].sum():,.0f} = 总{tot(M7):,.0f} 元 | 紧急电量={M7[6][m:].sum():,.0f} kWh')
print(f'M4 无储能: 总={cp4[m:].sum()+ce4[m:].sum():,.0f} 元; M3 天眼: 总={cp3[m:].sum():,.0f} 元 | 耗时={t_q42:.1f}s')
adopt = M7 if tot(M7) < tot(M1) else M1
print(f'[Q4-2 采用] {"M7" if adopt is M7 else "M1"}; 储能价值(M4-采用)={(cp4[m:].sum()+ce4[m:].sum())-tot(adopt):,.0f} 元')

res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in dates365])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
emerg = []
for i in range(len(res_dates)):
    idx = np.where(adopt[6][i0 + i] > 1e-6)[0] + 1
    emerg.append([(span_label(a, b), float(adopt[6][i0 + i][a - 1:b].sum())) for a, b in merge_spans(idx)])
write_result2('result4-2.xlsx', res_dates, adopt[0][i0:i1], adopt[5][i0:i1], adopt[3][i0:i1], adopt[4][i0:i1],
              adopt[1][i0:i1], adopt[2][i0:i1], emerg, template='result4-2.xlsx')

# ---------- Q4-3: 滚动调整 + 波动电价 ----------
def run_rolling(adjust_times):
    soc0 = 6000.0
    P0_m = np.zeros((n, 144)); Pf_m = np.zeros((n, 144))
    C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    s0 = np.zeros(n); s24 = np.zeros(n)
    plan_c = np.zeros(n); dev_c = np.zeros(n); E_m = np.zeros((n, 144)); em_c = np.zeros(n)
    for i in range(n):
        p = PR[i]; v = v_next[i]
        r0 = solve_day(p, L_prev[i], fc0m[i], soc0, soc_end=None, v=v)
        P0 = r0['P']; E0 = r0['E']
        P = P0.copy(); C = r0['C'].copy(); D = r0['D'].copy(); E = E0.copy()
        S = np.zeros(144); X = np.zeros(144)
        for h0, fcv in [(36, fca_m[6]), (72, fca_m[12]), (108, fca_m[18])]:
            if (h0 == 36 and 6 not in adjust_times) or (h0 == 72 and 12 not in adjust_times) or (h0 == 108 and 18 not in adjust_times):
                continue
            rr = solve_day(p[h0:], L_prev[i, h0:], fcv[i, h0:], E[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = rr['P'], rr['C'], rr['D']
            E = np.concatenate([E[:h0 + 1], rr['E'][1:]])
            S[h0:], X[h0:] = rr['S'], rr['X']
        E_, C_, D_, s24r, emc = settlement(L_act[i], PV_act[i], P, soc0, v, p)
        P0_m[i], Pf_m[i], C_m[i], D_m[i] = P0, P, C_, D_
        s0[i] = soc0; s24[i] = s24r
        plan_c[i] = float(p @ P0)
        dev_c[i] = float(np.sum(-0.5 * p * S + 1.5 * p * X))
        E_m[i] = E_; em_c[i] = emc
        soc0 = s24r
    return P0_m, Pf_m, C_m, D_m, s0, s24, plan_c, dev_c, E_m, em_c

t0 = time.time()
R2 = run_rolling(())
R1 = run_rolling((6, 12, 18))
t_q43 = time.time() - t0
totR = lambda M: (M[6] + M[7] + M[9])[m:].sum()
print(f'===== Q4-3 (波动电价滚动, 2.1-12.31) =====')
print(f'仅0:00: 计划费={R2[6][m:].sum():,.0f} + 调整费={R2[7][m:].sum():,.0f} + 紧急费={R2[9][m:].sum():,.0f} = 总{totR(R2):,.0f} 元 | 紧急电量={R2[8][m:].sum():,.0f} kWh')
print(f'全滚动: 计划费={R1[6][m:].sum():,.0f} + 调整费={R1[7][m:].sum():,.0f} + 紧急费={R1[9][m:].sum():,.0f} = 总{totR(R1):,.0f} 元 | 紧急电量={R1[8][m:].sum():,.0f} kWh')
print(f'[Q4-3 结论] 滚动节费={totR(R2)-totR(R1):,.0f} 元 ({(totR(R2)-totR(R1))/totR(R2)*100:.2f}%) | 耗时={t_q43:.1f}s')

S = np.maximum(R1[0] - R1[1], 0); X = np.maximum(R1[1] - R1[0], 0)
adj_cost = np.zeros(n)
for i in range(n):
    adj_cost[i] = float(PR[i] @ (R1[0][i] - S[i]) + 0.5 * PR[i] @ S[i] + 1.5 * PR[i] @ X[i])
emerg = []
for i in range(len(res_dates)):
    idx = np.where(R1[8][i0 + i] > 1e-6)[0] + 1
    emerg.append([(span_label(a, b), float(R1[8][i0 + i][a - 1:b].sum())) for a, b in merge_spans(idx)])
write_result3('result4-3.xlsx', res_dates, R1[0][i0:i1], R1[6][i0:i1], R1[1][i0:i1], adj_cost[i0:i1],
              R1[4][i0:i1], R1[5][i0:i1], R1[2][i0:i1], R1[3][i0:i1], emerg, template='result4-3.xlsx')

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
ax = axes[0]
ax.plot(dates365, M1[5] + M1[7], color=GRAY, lw=0.8, label='Q4-2: M1 外推')
ax.plot(dates365, adopt[5] + adopt[7], color=BLUE, lw=0.9, label='Q4-2: 采用方法')
ax.plot(dates365, cp3, color=ORANGE, lw=0.8, label='Q4-2: 天眼下界')
ax.set_title('波动电价下全年逐日购电总费用 (元)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7); ax.tick_params(axis='x', rotation=45, labelsize=6)
ax = axes[1]
totR2 = R2[6] + R2[7] + R2[9]; totR1 = R1[6] + R1[7] + R1[9]
ax.plot(dates365, totR2, color=GRAY, lw=0.8, label='Q4-3: 仅0:00')
ax.plot(dates365, totR1, color=BLUE, lw=0.9, label='Q4-3: 全滚动')
ax.set_title('Q4-3 滚动调整对比 (元)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7); ax.tick_params(axis='x', rotation=45, labelsize=6)
ax = axes[3 - 1]
k = list(dates365).index(pd.Timestamp('2025-06-21'))
t_axis = (np.arange(144) + 0.5) / 6
ax.plot(t_axis, L_act[k] * 6, color=BLUE, lw=1.0, label='实际负载')
ax.plot(t_axis, PV_act[k] * 6, color=ORANGE, lw=1.0, label='实际光伏')
ax.plot(t_axis, R1[0][k] * 6, color=GREEN, lw=1.0, label='0:00计划购电')
ax.plot(t_axis, R1[1][k] * 6, color=RED, lw=1.0, ls='--', label='滚动调整后购电')
ax.set_title('2025-06-21 波动电价调度 (kW)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_q4_fluct.png')); plt.close(fig)

with open(os.path.join(DATA, 'q4_results.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(q42=dict(m1=float(tot(M1)), m7=float(tot(M7)), m4=float(cp4[m:].sum() + ce4[m:].sum()),
                            m3=float(cp3[m:].sum()), q_star=q_star, adopt=('M7' if adopt is M7 else 'M1')),
                   q43=dict(r0=float(totR(R2)), roll=float(totR(R1)), em_kwh=float(R1[8][m:].sum()),
                            dev=float(R1[7][m:].sum()), em_cost=float(R1[9][m:].sum())),
                   t_q42=t_q42, t_q43=t_q43), f, ensure_ascii=False, indent=1)
print('Q4 DONE')
