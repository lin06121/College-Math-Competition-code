# -*- coding: utf-8 -*-
"""Q3: 0/6/12/18时滚动预报调整 + 违约/超额计价 + 5x紧急购电
机制: 0:00用0:00发布预报+负载外推制定计划P0; 6/12/18时用最新预报调整剩余时段(储能继承SOC);
     计费: min(P0,Pfin)按基准价, 计划>调整的差额付50%违约, 调整>计划超出按1.5倍, 缺口5x紧急
方法对比: M1 全滚动(0,6,12,18) / M2 仅0:00 / 边际价值(只+6 / +6+12 / 全) / M3 天眼下界
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
from io_results import load_all, write_result3, dates_of_results, ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')
DATA = os.path.join(ROOT, r'04_建模求解\data')
EPS = 1e-4

def settlement(L, PV, P0, soc0, v, price):
    """实时储能再调度: 最小化紧急购电 (与Q2同一机制)"""
    h = 144; p = price
    iC, iD, iE, iS = 0, h, 2 * h, 3 * h
    N = 4 * h + (h + 1)
    c = np.zeros(N)
    c[iE:iE + h] = 5.0 * p
    c[iC:iC + h] = EPS; c[iD:iD + h] = EPS
    c[iS + h] = -v
    Aub, bub = [], []
    Aeq, beq = [], []
    for t in range(h):
        row = np.zeros(N)
        row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        row = np.zeros(N)
        row[iS + t] = -1; row[iS + t + 1] = 1
        row[iC + t] = -ETA_C; row[iD + t] = 1.0 / ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iS:iS + h + 1] = SOC_MIN; ub[iS:iS + h + 1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success
    E = res.x[iE:iE + h]; C = res.x[iC:iC + h]; D = res.x[iD:iD + h]; S = res.x[iS:iS + h + 1]
    return E, C, D, S[-1], float(5.0 * p @ E)

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
v_const = 0.9 * price.mean()

L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
fc0m = fc0.pivot(index='date', columns='slot', values='fc0_kw').sort_index().to_numpy() / 6.0
fca_m = {(h): fca[fca.issue_hour == h].pivot(index='date', columns='slot', values='fc_kw').sort_index().reindex(columns=range(1, 145)).to_numpy() / 6.0 for h in (6, 12, 18)}
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
n = len(dates365)
# 计划依据升级: 负载=近4个同类日(周五六/其他)平均 (周内规律, 与Q2一致)
L_typ = typ['load_typ'].to_numpy() / 6.0
dow_q3 = np.array([d.dayofweek for d in dates365])
L_prev = np.zeros_like(L_act)
for i in range(n):
    low_set = {4, 5}   # 周五/周六
    hist = [j for j in range(i) if (dow_q3[j] in low_set) == (dow_q3[i] in low_set)]
    L_prev[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
H6, H12, H18 = 36, 72, 108   # 0-based 调整起始slot: 6:00=slot37->idx36, 12:00=73->72, 18:00=109->108

def run_rolling(adjust_times=(6, 12, 18)):
    """返回 P0,Pfin,Cfin,Dfin,Efin(145),soc0,soc24, plan_cost, dev_cost, E_emerg, em_cost"""
    soc0 = 6000.0
    P0_m = np.zeros((n, 144)); Pf_m = np.zeros((n, 144))
    C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    Etr = np.zeros((n, 145))
    s0 = np.zeros(n); s24 = np.zeros(n)
    plan_c = np.zeros(n); dev_c = np.zeros(n); E_m = np.zeros((n, 144)); em_c = np.zeros(n)
    for i in range(n):
        v = 0.0 if i == n - 1 else v_const
        r0 = solve_day(price, L_prev[i], fc0m[i], soc0, soc_end=None, v=v)
        P0 = r0['P']; C0 = r0['C']; D0 = r0['D']; E0 = r0['E']
        P = P0.copy(); C = C0.copy(); D = D0.copy(); E = E0.copy()
        S = np.zeros(144); X = np.zeros(144)
        if 6 in adjust_times:
            h0 = H6
            r = solve_day(price[h0:], L_prev[i, h0:], fca_m[6][i, h0:], E0[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = r['P'], r['C'], r['D']
            E = np.concatenate([E[:h0 + 1], r['E'][1:]])
            S[h0:], X[h0:] = r['S'], r['X']
        if 12 in adjust_times:
            h0 = H12
            r = solve_day(price[h0:], L_prev[i, h0:], fca_m[12][i, h0:], E[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = r['P'], r['C'], r['D']
            E = np.concatenate([E[:h0 + 1], r['E'][1:]])
            S[h0:], X[h0:] = r['S'], r['X']
        if 18 in adjust_times:
            h0 = H18
            r = solve_day(price[h0:], L_prev[i, h0:], fca_m[18][i, h0:], E[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = r['P'], r['C'], r['D']
            E = np.concatenate([E[:h0 + 1], r['E'][1:]])
            S[h0:], X[h0:] = r['S'], r['X']
        # 结算: 最终购电计划锁定, 储能按实际再调度, 缺口=紧急 (与Q2同机制)
        E_, C_, D_, s24r, emc = settlement(L_act[i], PV_act[i], P, soc0, v, price)
        P0_m[i], Pf_m[i], C_m[i], D_m[i] = P0, P, C_, D_
        Etr[i] = np.zeros(145)   # 仅占位(SOC轨迹以结算为准)
        s0[i] = soc0; s24[i] = s24r
        plan_c[i] = float(price @ P0)
        dev_c[i] = float(np.sum(-0.5 * price * S + 1.5 * price * X))
        E_m[i] = E_; em_c[i] = emc
        soc0 = s24r
    return P0_m, Pf_m, C_m, D_m, Etr, s0, s24, plan_c, dev_c, E_m, em_c

t0 = time.time()
M1 = run_rolling((6, 12, 18))
M2 = run_rolling(())
M6 = run_rolling((6,))
M12 = run_rolling((6, 12))
M3c = None  # 天眼
# 天眼: 实际数据LP
soc0 = 6000.0; cp3 = np.zeros(n)
for i in range(n):
    v = 0.0 if i == n - 1 else v_const
    r = solve_day(price, L_act[i], PV_act[i], soc0, soc_end=None, v=v)
    cp3[i] = r['gross']; soc0 = r['E'][-1]
t_all = time.time() - t0

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
def totM(M_):
    return M_[7] + M_[8] + M_[10]
def line(name, M_):
    t = totM(M_)[m:].sum()
    print(f'{name}: 计划费={M_[7][m:].sum():,.0f} + 调整费={M_[8][m:].sum():,.0f} + 紧急费={M_[10][m:].sum():,.0f} = 总{t:,.0f} 元 | 紧急电量={M_[9][m:].sum():,.0f} kWh')
print('===== Q3 2.1-12.31 各滚动方案 =====')
line('仅0:00计划        ', M2)
line('+6:00调整         ', M6)
line('+6:00,+12:00      ', M12)
line('全滚动(6/12/18)   ', M1)
print(f'天眼下界: {cp3[m:].sum():,.0f} 元 | 耗时={t_all:.1f}s')
tot1 = totM(M1)[m:].sum(); tot2 = totM(M2)[m:].sum()
t6 = totM(M6)[m:].sum(); t12 = totM(M12)[m:].sum()
print(f'[结论] 全滚动对仅0:00节费={tot2-tot1:,.0f} 元 ({(tot2-tot1)/tot2*100:.2f}%); 距天眼={tot1-cp3[m:].sum():,.0f} 元')
print(f'[边际价值] 6:00调整带来={tot2-t6:,.0f} 元; 12:00再带来={t6-t12:,.0f} 元; 18:00再带来={t12-tot1:,.0f} 元')

# ---------- 采用全滚动写 result3 ----------
P0, Pf, C, D, Etr, s0, s24, plan_c, dev_c, E_m, em_c = M1
res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in dates365])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
# 调整购电费(按偏差计价, 不含紧急): min(P0,Pf)基准 + 0.5*S + 1.5*X
S = np.maximum(P0 - Pf, 0); X = np.maximum(Pf - P0, 0)
adj_cost = (price * (P0 - S) + 0.5 * price * S + 1.5 * price * X).sum(axis=1)
tot_cost = plan_c + dev_c + em_c
emerg = []
for i in range(len(res_dates)):
    idx = np.where(E_m[i0 + i] > 1e-6)[0] + 1
    emerg.append([(span_label(s0_, s1), float(E_m[i0 + i][s0_ - 1:s1].sum())) for s0_, s1 in merge_spans(idx)])
write_result3('result3.xlsx', res_dates, P0[i0:i1], plan_c[i0:i1], Pf[i0:i1], adj_cost[i0:i1],
              s0[i0:i1], s24[i0:i1], C[i0:i1], D[i0:i1], emerg)

# ---------- 论文表 ----------
spec = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
slot_of = {'10:00-10:10': 61, '12:00-12:10': 73, '14:00-14:10': 85, '16:00-16:10': 97, '18:00-18:10': 109, '20:00-20:10': 121}
paper = {}
for ds in spec:
    k = list(dates365).index(pd.Timestamp(ds))
    paper[ds] = dict(
        t1={lab: round(float(P0[k, v - 1]), 2) for lab, v in slot_of.items()},
        total=round(float(P0[k].sum()), 2), cost=round(float(plan_c[k]), 2),
        t2_blocks=[(round(float(C[k, b * 24:(b + 1) * 24].sum()), 2), round(float(D[k, b * 24:(b + 1) * 24].sum()), 2)) for b in range(6)],
        soc0=round(float(s0[k]), 2), soc24=round(float(s24[k]), 2),
        t3=[(span_label(a, b), round(float(E_m[k][a - 1:b].sum()), 2)) for a, b in merge_spans(np.where(E_m[k] > 1e-6)[0] + 1)],
        adj_total=round(float(Pf[k].sum()), 2), dev_cost=round(float(dev_c[k]), 2), tot_cost=round(float(tot_cost[k]), 2))
print('[论文表]', json.dumps(paper, ensure_ascii=False, indent=1))

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
totM2 = totM(M2); totM1 = totM(M1)
ax = axes[0]
ax.plot(dates365, totM2, color=GRAY, lw=0.8, label='仅0:00计划')
ax.plot(dates365, totM1, color=BLUE, lw=0.9, label='全滚动(6/12/18)')
ax.plot(dates365, cp3, color=ORANGE, lw=0.8, label='天眼下界')
ax.set_title('全年逐日购电总费用 (元)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7); ax.tick_params(axis='x', rotation=45, labelsize=6)
ax = axes[1]
labels = ['仅0:00', '+6:00', '+6:00\n+12:00', '全滚动']
vals = [tot2, t6, t12, tot1]
ax.bar(labels, np.array(vals) / 1e6, color=[GRAY, '#56B4E9', '#0072B2', BLUE])
ax.set_ylabel('全年总费用 (百万元)'); ax.set_title('滚动调整的边际价值', fontsize=9)
ax.grid(alpha=0.3, axis='y', ls='--')
for kk, vv in enumerate(vals):
    ax.text(kk, vv / 1e6 + 0.1, f'{vv/1e6:.2f}', ha='center', fontsize=7)
ax = axes[2]
k = list(dates365).index(pd.Timestamp('2025-06-21'))
t_axis = (np.arange(144) + 0.5) / 6
ax.plot(t_axis, L_act[k] * 6, color=BLUE, lw=1.0, label='实际负载')
ax.plot(t_axis, PV_act[k] * 6, color=ORANGE, lw=1.0, label='实际光伏')
ax.plot(t_axis, P0[k] * 6, color=GREEN, lw=1.0, label='0:00计划购电')
ax.plot(t_axis, Pf[k] * 6, color=RED, lw=1.0, ls='--', label='滚动调整后购电')
ax.set_title('2025-06-21 计划 vs 调整 (kW)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_q3_rolling.png')); plt.close(fig)

with open(os.path.join(DATA, 'q3_results.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(tot_m1=float(tot1), tot_m2=float(tot2), tot_m3=float(cp3[m:].sum()),
                   dev_cost=float(dev_c[m:].sum()), em_cost=float(em_c[m:].sum()),
                   plan_cost=float(plan_c[m:].sum()), em_kwh=float(E_m[m:].sum()),
                   marginal={'6h': float(tot2 - t6), '12h': float(t6 - t12), '18h': float(t12 - tot1)},
                   t_all=t_all, paper=paper), f, ensure_ascii=False, indent=1)
print('Q3 DONE')
