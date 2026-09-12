# -*- coding: utf-8 -*-
"""同一 q 口径下的三方对比(问题二):
   (A) 执行计划(因果, 无日内调整) | (B) 实时平衡控制(因果, 采用) | (C) 完美信息再调度(非因果上界)
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

def buf_of(i, q):
    lo = max(0, i - 30)
    return np.quantile(eNet[lo:i], q, axis=0) if i - lo >= 7 else np.zeros(144)

def perfect(L, PV, P0, soc0, v, p):
    h = 144; iC, iD, iE, iS = 0, h, 2*h, 3*h; N = 4*h + (h+1)
    c = np.zeros(N); c[iE:iE+h] = 5*p; c[iC:iC+h] = EPS; c[iD:iD+h] = EPS; c[iS+h] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        r_ = np.zeros(N); r_[iE+t], r_[iC+t], r_[iD+t] = -1, 1, -1
        Aub.append(r_); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        r_ = np.zeros(N); r_[iS+t] = -1; r_[iS+t+1] = 1; r_[iC+t] = -ETA_C; r_[iD+t] = 1/ETA_D
        Aeq.append(r_); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC+h] = P_MAX; ub[iD:iD+h] = P_MAX
    lb[iS:iS+h+1] = SOC_MIN; ub[iS:iS+h+1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                bounds=list(zip(lb, ub)), method='highs')
    return r.x[iE:iE+h], r.x[iS+h], float(5*p @ r.x[iE:iE+h])

def bal(P0, Cpl, Dpl, soc0, L, PV, reserve=SOC_MIN):
    h = len(P0)
    C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        # ① 计划量裁剪为物理可行
        c_ok = min(max(float(Cpl[t]), 0.0), P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C))
        d_ok = min(max(float(Dpl[t]), 0.0), P_MAX, ETA_D * max(0.0, soc - reserve + ETA_C * c_ok))
        C[t], D[t] = c_ok, d_ok
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        # ② 缺额: 先减充, 再增放
        if gap > 1e-9:
            cut = min(C[t], gap); C[t] -= cut; gap -= cut
            if gap > 1e-9:
                avail = ETA_D * max(0.0, soc - reserve + ETA_C * C[t]) - D[t]
                extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, avail))
                D[t] += extra; gap -= extra
        # ③ 富余: 先减放, 再增充
        elif gap < -1e-9:
            cut = min(D[t], -gap); D[t] -= cut; gap += cut
            if gap < -1e-9:
                room = max(0.0, (SOC_MAX - soc + D[t] / ETA_D)) / ETA_C - C[t]
                extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, room))
                C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
        assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6, ('SOC越界', t, soc)
        assert C[t] <= P_MAX + 1e-9 and D[t] <= P_MAX + 1e-9, ('功率越界', t)
        assert min(C[t], D[t]) <= 1e-9, ('同时充放', t)
    return C, D, soc, Em
