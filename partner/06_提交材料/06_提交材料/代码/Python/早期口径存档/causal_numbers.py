# -*- coding: utf-8 -*-
"""因果口径下的论文对照数字: Q3 滚动方案边际价值 + Q2 依据对比 + 无储能基准"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365]); n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0; PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
fc0m = fc0.pivot(index='date', columns='slot', values='fc0_kw').sort_index().to_numpy() / 6.0
fca_m = {h: fca[fca.issue_hour == h].pivot(index='date', columns='slot', values='fc_kw').sort_index().reindex(columns=range(1, 145)).to_numpy() / 6.0 for h in (6, 12, 18)}
m2 = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
Lb_dow = np.zeros_like(L_act); Pb_m4 = np.zeros_like(PV_act); Lb_per = np.zeros_like(L_act)
for i in range(n):
    hist = [j for j in range(i) if (dow[j] in {4,5}) == (dow[i] in {4,5})]
    Lb_dow[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
    h2 = list(range(max(0, i-4), i)); Pb_m4[i] = PV_act[h2].mean(axis=0) if h2 else PV_typ
    Lb_per[i] = L_act[i-1] if i > 0 else L_typ

def balance(P0, Cpl, Dpl, soc0, L, PV):
    h = len(P0); C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        c_ok = min(max(float(Cpl[t]),0.0), P_MAX, max(0.0,(SOC_MAX-soc)/ETA_C))
        d_ok = min(max(float(Dpl[t]),0.0), P_MAX, ETA_D*max(0.0, soc-SOC_MIN+ETA_C*c_ok))
        C[t], D[t] = c_ok, d_ok
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        if gap > 1e-9:
            cut = min(C[t], gap); C[t] -= cut; gap -= cut
            if gap > 1e-9:
                avail = ETA_D*max(0.0, soc-SOC_MIN+ETA_C*C[t]) - D[t]
                extra = min(gap, max(0.0,P_MAX-D[t]), max(0.0,avail)); D[t] += extra; gap -= extra
        elif gap < -1e-9:
            cut = min(D[t], -gap); D[t] -= cut; gap += cut
            if gap < -1e-9:
                room = max(0.0,(SOC_MAX-soc+D[t]/ETA_D))/ETA_C - C[t]
                extra = min(-gap, max(0.0,P_MAX-C[t]), max(0.0,room)); C[t] += extra; gap += extra
        Em[t] = max(0.0, gap); soc = soc + ETA_C*C[t] - D[t]/ETA_D
    return C, D, soc, Em

def run_q2(LB, PB, q, v_mode='typ'):
    soc = 6000.0; v = 0.9*price.mean()
    tot = 0.0; plan = 0.0; em = 0.0
    eNet = (L_act - LB) - (PV_act - PB)
    for i in range(n):
        b = np.quantile(eNet[max(0,i-30):i], q, axis=0) if i >= 7 else np.zeros(144)
        r0 = solve_day(price, LB[i]+b, PB[i], soc, soc_end=None, v=(v if i < n-1 else 0.0))
        C, D, sEnd, Em = balance(r0['P'], r0['C'], r0['D'], soc, L_act[i], PV_act[i])
        if i >= m2:
            plan += float(price @ r0['P']); em += float(5*price @ Em)
        soc = sEnd
    return plan, em, plan + em

def run_nostorage(LB, PB, q):
    eNet = (L_act - LB) - (PV_act - PB); tot = 0.0
    for i in range(n):
        b = np.quantile(eNet[max(0,i-30):i], q, axis=0) if i >= 7 else np.zeros(144)
        P0 = np.maximum(LB[i] + b - PB[i], 0.0)
        Em = np.maximum(L_act[i] - PV_act[i] - P0, 0.0)
        if i >= m2:
            tot += float(price @ P0) + float(5*price @ Em)
    return tot

print('=== 问题二 因果口径下的方法对比 ===')
p, e, t = run_q2(Lb_dow, Pb_m4, 0.7); print(f'M8 同类日平均+近4日光伏+q0.7(采用): 计划 {p:,.0f} + 紧急 {e:,.0f} = {t:,.0f} 元')
p, e, t = run_q2(Lb_per, Pb_m4, 0.6); print(f'M1 昨日外推+q0.6:                 计划 {p:,.0f} + 紧急 {e:,.0f} = {t:,.0f} 元')
print(f'M4 无储能(同类日+报童):           {run_nostorage(Lb_dow, Pb_m4, 0.7):,.0f} 元')

print('\n=== 问题三 因果口径下滚动方案对比 ===')
def run_q3(adjust_times):
    soc = 6000.0; v = 0.9*price.mean(); tot = 0.0; planT = devT = emT = 0.0
    H = [36, 72, 108]; fh = [6, 12, 18]
    for i in range(n):
        vv = v if i < n-1 else 0.0
        eNet = (L_act - Lb_dow) - (PV_act - Pb_m4)
        b = np.quantile(eNet[max(0,i-30):i], 0.7, axis=0) if i >= 7 else np.zeros(144)
        Lbar = Lb_dow[i] + b
        r0 = solve_day(price, Lbar, fc0m[i], soc, soc_end=None, v=vv)
        P, C, D = r0['P'], r0['C'], r0['D']; SOC = r0['E']; Pinit = P.copy()
        for h0, hh in zip(H, fh):
            if hh not in adjust_times: continue
            r = solve_day(price[h0:], Lbar[h0:], fca_m[hh][i, h0:], SOC[h0], soc_end=None, v=vv, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0+1], r['E'][1:]])
        C2, D2, sEnd, Em = balance(P, C, D, soc, L_act[i], PV_act[i])
        if i >= m2:
            planT += float(price @ Pinit)
            devT += float(np.sum(-0.5*price*np.maximum(Pinit-P,0) + 1.5*price*np.maximum(P-Pinit,0)))
            emT += float(5*price @ Em)
        soc = sEnd
    return planT, devT, emT, planT+devT+emT
for at, name in [((), '仅0:00'), ((6,), '+6:00'), ((6,12), '+6,+12'), ((6,12,18), '全滚动(采用)')]:
    p, d, e, t = run_q3(at)
    print(f'{name:14s}: 计划 {p:,.0f} + 调整 {d:,.0f} + 紧急 {e:,.0f} = {t:,.0f} 元')
