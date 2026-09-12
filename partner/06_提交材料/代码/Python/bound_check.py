# -*- coding: utf-8 -*-
"""严格定界: 在"采用方案(实时平衡, 因果)"的同一 SOC 链条与同一计划 P0 下, 逐日求
   "应急最小化下界"(仅最小化 5p*E, 储能自由调度, 终值自由) -> 年紧急费合法下界
   同时给出非因果完美信息再调度(带残值项)的费用, 作为对照。
"""
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
Lb = np.zeros_like(L_act); Pb = np.zeros_like(PV_act)
for i in range(n):
    hist = [j for j in range(i) if (dow[j] in {4,5}) == (dow[i] in {4,5})]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
    h2 = list(range(max(0, i-4), i)); Pb[i] = PV_act[h2].mean(axis=0) if h2 else PV_typ
eNet = (L_act - Lb) - (PV_act - Pb)
m2 = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
Q = 0.6

def storage_lp(L, PV, P0, soc0, obj_kind):
    """储能调度 LP: obj_kind='em' 只最小化 5p*E(下界); 'em_v' 含残值 -v*E_end(对照)"""
    h = 144; iE, iC, iD, iS = 0, h, 2*h, 3*h; N = 3*h + (h+1)
    v = 0.9 * price.mean()
    c = np.zeros(N); c[iE:iE+h] = 5*price
    if obj_kind == 'em_v':
        c[iC:iC+h] = EPS; c[iD:iD+h] = EPS; c[iS+h] = -v
    Aub, bub = [], []
    for t in range(h):
        r = np.zeros(N); r[iE+t], r[iC+t], r[iD+t] = -1, 1, -1
        Aub.append(r); bub.append(PV[t] + P0[t] - L[t])
    Aeq, beq = [], []
    for t in range(h):
        r = np.zeros(N); r[iS+t] = -1; r[iS+t+1] = 1; r[iC+t] = -ETA_C; r[iD+t] = 1/ETA_D
        Aeq.append(r); beq.append(0.0)
    r_ = np.zeros(N); r_[iS] = 1; Aeq.append(r_); beq.append(soc0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC+h] = P_MAX; ub[iD:iD+h] = P_MAX
    lb[iS:iS+h+1] = SOC_MIN; ub[iS:iS+h+1] = SOC_MAX
    lb[iS] = ub[iS] = soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    return res.x[iE:iE+h], res.x[iC:iC+h], res.x[iD:iD+h], res.x[iS+h], float(5*price @ res.x[iE:iE+h])

def bal(P0, Cpl, Dpl, soc0, L, PV):
    h = 144; C = Cpl.astype(float).copy(); D = Dpl.astype(float).copy(); Em = np.zeros(h); soc = soc0
    for t in range(h):
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        if gap > 1e-9:
            extra = min(gap, P_MAX, max(0.0, (soc - SOC_MIN) * ETA_D)); D[t] += extra; gap -= extra
        elif gap < -1e-9:
            extra = min(-gap, P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C)); C[t] += extra; gap += extra
        Em[t] = max(0.0, gap); soc = float(min(max(soc + ETA_C*C[t] - D[t]/ETA_D, SOC_MIN), SOC_MAX))
    return C, D, soc, float(5*price @ Em)

# 第一步: 采用方案(实时平衡)链条, 记录每日 P0 与 soc0
recs = []
soc = 6000.0; v = 0.9*price.mean()
for i in range(n):
    b = np.quantile(eNet[max(0, i-30):i], Q, axis=0) if i >= 7 else np.zeros(144)
    r0 = solve_day(price, Lb[i]+b, Pb[i], soc, soc_end=None, v=v)
    P0, C0, D0 = r0['P'], r0['C'], r0['D']
    CB, DB, socB, cB = bal(P0, C0, D0, soc, L_act[i], PV_act[i])
    recs.append((i, P0.copy(), soc, float(price @ P0), cB))
    soc = socB

plan = sum(r[3] for r in recs if r[0] >= m2)
emB = sum(r[4] for r in recs if r[0] >= m2)
# 第二步: 同一 (P0, soc0) 下的下界与对照
emBound = 0.0; emV = 0.0
for (i, P0, soc0, pc, cB) in recs:
    if i < m2: continue
    _, _, _, _, eBd = storage_lp(L_act[i], PV_act[i], P0, soc0, 'em')
    _, _, _, _, eV = storage_lp(L_act[i], PV_act[i], P0, soc0, 'em_v')
    emBound += eBd; emV += eV
print(f'q={Q} 全链同一口径(同一 P0 与 soc0):')
print(f'  计划购电费                       = {plan:,.0f} 元')
print(f'  采用方案 紧急费(实时平衡, 因果)   = {emB:,.0f} 元  -> 总 {plan+emB:,.0f} 元')
print(f'  完美信息应急下界 紧急费           = {emBound:,.0f} 元  -> 总 {plan+emBound:,.0f} 元')
print(f'  完美信息+残值项 紧急费(对照)      = {emV:,.0f} 元  -> 总 {plan+emV:,.0f} 元')
print(f'  [检验] 采用方案 - 下界 = {emB-emBound:,.0f} 元 ({((emB-emBound)/max(emBound,1e-9)*100):.1f}% of 下界)  (应为 ≥0)')
