# -*- coding: utf-8 -*-
"""用 m4 光伏基值复算报童分位 q 的全年最优值(1月标定 + 2-12月外样本评估)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

EPS = 1e-4
day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365]); n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0; PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
Lb = np.zeros_like(L_act); Pb = np.zeros_like(PV_act)      # m4: 近4日均值
for i in range(n):
    low = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low) == (dow[i] in low)]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
    hist2 = list(range(max(0, i - 4), i))
    Pb[i] = PV_act[hist2].mean(axis=0) if hist2 else PV_typ
eNet = (L_act - Lb) - (PV_act - Pb)

def settlement(L, PV, P0, soc0, v):
    h = 144; p = price
    iC, iD, iE, iS = 0, h, 2*h, 3*h; N = 4*h + (h+1)
    c = np.zeros(N); c[iE:iE+h] = 5*p; c[iC:iC+h] = EPS; c[iD:iD+h] = EPS; c[iS+h] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        row = np.zeros(N); row[iE+t], row[iC+t], row[iD+t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        row = np.zeros(N); row[iS+t] = -1; row[iS+t+1] = 1
        row[iC+t] = -ETA_C; row[iD+t] = 1/ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC+h] = P_MAX; ub[iD:iD+h] = P_MAX
    lb[iS:iS+h+1] = SOC_MIN; ub[iS:iS+h+1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                bounds=list(zip(lb, ub)), method='highs')
    E = r.x[iE:iE+h]; C = r.x[iC:iC+h]; D = r.x[iD:iD+h]; S = r.x[iS:iS+h+1]
    return E, C, D, S[-1], float(5*p @ E)

def chain(q, i_from=1, i_to=365):
    soc = 6000.0; v = 0.9 * price.mean()
    cp = np.zeros(n); ce = np.zeros(n)
    for i in range(i_to):
        b = np.quantile(eNet[max(0, i-30):i], q, axis=0) if i >= 7 else np.zeros(144)
        P0 = solve_day(price, Lb[i] + b, Pb[i], soc, soc_end=None, v=v)['P']
        E, C, D, s24, cei = settlement(L_act[i], PV_act[i], P0, soc, v)
        cp[i] = price @ P0; ce[i] = cei; soc = s24
    return cp, ce

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
print('q    1月费用(8-31日)   2-12月总费用')
res = {}
for q in [0.5, 0.6, 0.7, 0.8]:
    cp, ce = chain(q)
    jan = (cp + ce)[7:31].sum(); annual = (cp + ce)[m:].sum()
    res[q] = (jan, annual)
    print(f'{q:.1f}  {jan:12,.0f}   {annual:14,.0f}')
best_jan = min(res, key=lambda k: res[k][0]); best_ann = min(res, key=lambda k: res[k][1])
print(f'[1月标定最优] q*={best_jan};  [全年最优] q*={best_ann}')
print(f'[敏感性] 全年 q=0.6 与 q=0.7 之差 = {res[0.6][1]-res[0.7][1]:+,.0f} 元 ({(res[0.6][1]-res[0.7][1])/res[0.7][1]*100:+.3f}%)')

# 若 m4 下 1月标定最优与交付版 q=0.7 不同, 输出对应全年差异
with open(os.path.join(ROOT, r'04_建模求解\data\q_tune_m4.json'), 'w', encoding='utf-8') as f:
    json.dump({str(k): {'jan': v[0], 'annual': v[1]} for k, v in res.items()}, f, ensure_ascii=False, indent=1)
