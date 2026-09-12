# -*- coding: utf-8 -*-
"""一致性校验: 逐日验证 因果平衡策略 的紧急费 >= 同期(同 P0, 同 soc0)的 LP 应急最小下界
   (理论上必然成立; 若不成立说明实现仍有漏洞)"""
import sys, io, os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
d = day.pivot(index='date', columns='slot', values='load_kw').sort_index()
L_act = d.to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
dow = np.array([x.dayofweek for x in d.index]); n = 365
L_typ = typ['load_typ'].to_numpy() / 6.0; PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
Lb = np.zeros_like(L_act); Pb = np.zeros_like(PV_act)
for i in range(n):
    hist = [j for j in range(i) if (dow[j] in {4,5}) == (dow[i] in {4,5})]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
    h2 = list(range(max(0, i-4), i)); Pb[i] = PV_act[h2].mean(axis=0) if h2 else PV_typ
eNet = (L_act - Lb) - (PV_act - Pb)

def balance(P0, Cpl, Dpl, soc0, L, PV, reserve=SOC_MIN):
    h = len(P0); C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        c_ok = min(max(float(Cpl[t]), 0.0), P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C))
        d_ok = min(max(float(Dpl[t]), 0.0), P_MAX, ETA_D * max(0.0, soc - reserve + ETA_C * c_ok))
        C[t], D[t] = c_ok, d_ok
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        if gap > 1e-9:
            cut = min(C[t], gap); C[t] -= cut; gap -= cut
            if gap > 1e-9:
                avail = ETA_D * max(0.0, soc - reserve + ETA_C * C[t]) - D[t]
                extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, avail)); D[t] += extra; gap -= extra
        elif gap < -1e-9:
            cut = min(D[t], -gap); D[t] -= cut; gap += cut
            if gap < -1e-9:
                room = max(0.0, (SOC_MAX - soc + D[t] / ETA_D)) / ETA_C - C[t]
                extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, room)); C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
    return C, D, soc, Em

def lp_bound(L, PV, P0, soc0):
    h = 144; iE, iC, iD, iS = 0, h, 2*h, 3*h; N = 3*h + (h+1)
    c = np.zeros(N); c[iE:iE+h] = 5*price
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
    lb[iS:iS+h+1] = SOC_MIN; ub[iS:iS+h+1] = SOC_MAX; lb[iS] = ub[iS] = soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    return float(res.fun)

Q = 0.7; v = 0.9 * price.mean(); soc = 6000.0
viol = 0; worst = 0.0; totB = 0.0; totL = 0.0
for i in range(31, 365, 7):          # 每隔 7 天抽检(约 48 天)
    b = np.quantile(eNet[max(0, i-30):i], Q, axis=0)
    r0 = solve_day(price, Lb[i]+b, Pb[i], soc, soc_end=None, v=v)
    P0, C0, D0 = r0['P'], r0['C'], r0['D']
    C, D, sEnd, Em = balance(P0, C0, D0, soc, L_act[i], PV_act[i])
    cB = float(5*price @ Em); cL = lp_bound(L_act[i], PV_act[i], P0, soc)
    totB += cB; totL += cL
    if cB < cL - 1e-6:
        viol += 1; worst = max(worst, cL - cB)
        print(f'  day {i}: 策略 {cB:,.1f} < 下界 {cL:,.1f}  (差 {cL-cB:,.1f})')
    soc = sEnd
print(f'抽检天数 ~48: 违反"策略>=下界"的天数 = {viol} (最大违反 {worst:,.2f} 元)')
print(f'抽检合计: 因果平衡紧急费 = {totB:,.0f} 元 | LP 应急下界 = {totL:,.0f} 元 | 差距 = {totB-totL:,.0f} 元 ({(totB-totL)/max(totL,1)*100:.1f}%)')
