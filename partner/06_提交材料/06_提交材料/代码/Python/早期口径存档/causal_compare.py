# -*- coding: utf-8 -*-
"""结算层因果性对比:
   (A) 执行计划(储能按计划充放电, 完全因果)
   (B) 因果再调度(每个整点用"已观测偏差"更新剩余时段预测后重优化, 仅用当前及历史信息)
   (C) 完美信息再调度(全天实际数据一次优化; 现行论文口径, 上界)
对比 Q2 / Q3 / Q4-2 / Q4-3 的全年费用。
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
PR = day.pivot(index='date', columns='slot', values='price_q4').sort_index().to_numpy()
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365]); n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0; PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
Lb = np.zeros_like(L_act); Pb = np.zeros_like(PV_act)
for i in range(n):
    low = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low) == (dow[i] in low)]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
    h2 = list(range(max(0, i - 4), i))
    Pb[i] = PV_act[h2].mean(axis=0) if h2 else PV_typ
eNet = (L_act - Lb) - (PV_act - Pb)
fc0m = fc0.pivot(index='date', columns='slot', values='fc0_kw').sort_index().to_numpy() / 6.0
fca_m = {h: fca[fca.issue_hour == h].pivot(index='date', columns='slot', values='fc_kw').sort_index().reindex(columns=range(1, 145)).to_numpy() / 6.0 for h in (6, 12, 18)}
m2 = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]

def settle(L, PV, P0, Cfix, Dfix, soc0, v, p):
    """(A) 执行计划: 储能按 Cfix/Dfix 执行, 缺口按 5 倍紧急购电"""
    short = L - PV - P0 - Dfix + Cfix
    E = np.maximum(short, 0.0)
    soc = soc0
    # SOC 轨迹(用于次日链条): E_{t+1} = E_t + 0.9C - D/0.9
    return E, soc + np.cumsum(ETA_C * Cfix - Dfix / ETA_D)[-1], float(5 * p @ E)

def settle_perfect(L, PV, P0, soc0, v, p):
    """(C) 完美信息再调度(现行口径)"""
    h = 144
    iC, iD, iE, iS = 0, h, 2 * h, 3 * h; N = 4 * h + (h + 1)
    c = np.zeros(N); c[iE:iE + h] = 5 * p; c[iC:iC + h] = EPS; c[iD:iD + h] = EPS; c[iS + h] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        row = np.zeros(N); row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        row = np.zeros(N); row[iS + t] = -1; row[iS + t + 1] = 1
        row[iC + t] = -ETA_C; row[iD + t] = 1 / ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iS:iS + h + 1] = SOC_MIN; ub[iS:iS + h + 1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                bounds=list(zip(lb, ub)), method='highs')
    return r.x[iE:iE + h], r.x[iS + h], float(5 * p @ r.x[iE:iE + h])

def settle_causal(L, PV, Lbase, PVbase, P0, soc0, v, p, horizon_step=6):
    """(B) 因果再调度: 每 horizon_step 个时段用“已观测偏差”更新剩余预测后重优化(仅历史+当前信息)"""
    h = 144
    C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = soc0
    k = 0
    while k < h:
        k2 = min(k + horizon_step, h)
        # 已观测偏差(截至 k-1)用于修正剩余时段预测; 无观测时用基值
        Lf = Lbase[k2:].copy() if k2 < h else np.zeros(0)
        PVf = PVbase[k2:].copy() if k2 < h else np.zeros(0)
        if k > 0:
            dev_L = float(np.mean(L[:k] - Lbase[:k])); dev_PV = float(np.mean(PV[:k] - PVbase[:k]))
            if k2 < h:
                Lf = Lf + dev_L; PVf = np.maximum(PVf + dev_PV, 0)
        if k2 > k:
            hh = k2 - k
            iC, iD, iS = 0, hh, 2 * hh; N = 2 * hh + (hh + 1)
            c = np.zeros(N); c[iC:iC + hh] = EPS; c[iD:iD + hh] = EPS
            if k2 == h:
                c[iS + hh] = -v
            Aeq = np.zeros((hh + 1, N)); beq = np.zeros(hh + 1)
            for t in range(hh):
                Aeq[t, iS + t] = -1; Aeq[t, iS + t + 1] = 1
                Aeq[t, iC + t] = -ETA_C; Aeq[t, iD + t] = 1 / ETA_D
            Aeq[hh, iS] = 1; beq[hh] = soc
            lb = np.zeros(N); ub = np.full(N, np.inf)
            ub[iC:iC + hh] = P_MAX; ub[iD:iD + hh] = P_MAX
            lb[iS:iS + hh + 1] = SOC_MIN; ub[iS:iS + hh + 1] = SOC_MAX
            lb[iS] = ub[iS] = soc
            r = linprog(c, A_eq=Aeq, b_eq=beq, bounds=list(zip(lb, ub)), method='highs')
            C[k:k2] = r.x[iC:iC + hh]; D[k:k2] = r.x[iD:iD + hh]
            soc = soc + float(np.sum(ETA_C * C[k:k2] - D[k:k2] / ETA_D))
        Em[k:k2] = np.maximum(L[k:k2] - PV[k:k2] - P0[k:k2] - D[k:k2] + C[k:k2], 0)
        k = k2
    return Em, soc, float(5 * p @ Em)

def run_q2(mode):
    soc = 6000.0; tot = 0.0; totA = 0.0; totC = 0.0
    v = 0.9 * price.mean()
    for i in range(n):
        bq = np.quantile(eNet[max(0, i - 30):i], 0.7, axis=0) if i >= 7 else np.zeros(144)
        res = solve_day(price, Lb[i] + bq, Pb[i], soc, soc_end=None, v=v)
        P0, C0, D0 = res['P'], res['C'], res['D']
        plan = float(price @ P0)
        # (A)
        EA, socA, cEA = settle(L_act[i], PV_act[i], P0, C0, D0, soc, v, price)
        # (B)
        EB, socB, cEB = settle_causal(L_act[i], PV_act[i], Lb[i], Pb[i], P0, soc, v, price)
        # (C)
        EC, socC, cEC = settle_perfect(L_act[i], PV_act[i], P0, soc, v, price)
        if i >= m2 - 1:
            totA += price @ P0 + cEA; totC += price @ P0 + cEC
        if i == n - 1:
            break
        soc = socC        # 链条统一用完美信息版(三者差别仅结算层)
    return totA, totC

def run_q3(use_causal_exec=True):
    """Q3: 0/6/12/18 预测驱动; 最终执行(因果) 或 完美信息再调度"""
    soc = 6000.0; v = 0.9 * price.mean()
    tot_causal = 0.0; tot_perfect = 0.0
    H = [36, 72, 108]
    for i in range(n):
        bq = np.quantile(eNet[max(0, i - 30):i], 0.7, axis=0) if i >= 7 else np.zeros(144)
        Lbar = Lb[i] + bq
        res = solve_day(price, Lbar, fc0m[i], soc, soc_end=None, v=v)
        P, C, D, E = res['P'], res['C'], res['D'], res['E']
        Pinit = P.copy()
        for h0, hh in zip(H, [6, 12, 18]):
            fc = fca_m[hh][i, h0:]
            r = solve_day(price[h0:], Lbar[h0:], fc, E[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = r['P'], r['C'], r['D']
            E = np.concatenate([E[:h0 + 1], r['E'][1:]])
        plan = float(price @ Pinit)
        dev = float(np.sum(-0.5 * price * np.maximum(Pinit - P, 0) + 1.5 * price * np.maximum(P - Pinit, 0)))
        EA, socA, cEA = settle(L_act[i], PV_act[i], P, C, D, soc, v, price)
        EC, socC, cEC = settle_perfect(L_act[i], PV_act[i], P, soc, v, price)
        if i >= m2 - 1:
            tot_causal += plan + dev + cEA; tot_perfect += plan + dev + cEC
        if i == n - 1:
            break
        soc = socC
    return tot_causal, tot_perfect

tA, tC = run_q2('q2')
print(f'问题二 全年(2.1-12.31): (A)执行计划 = {tA:,.0f} 元 | (C)完美信息再调度 = {tC:,.0f} 元 | 差 {(tA-tC):,.0f} 元 ({(tA-tC)/tC*100:.2f}%)')
t3A, t3C = run_q3()
print(f'问题三 全年: (A)按最终计划执行 = {t3A:,.0f} 元 | (C)完美信息再调度 = {t3C:,.0f} 元 | 差 {(t3A-t3C):,.0f} 元 ({(t3A-t3C)/t3C*100:.2f}%)')
with open(os.path.join(ROOT, r'04_建模求解\data\causal_compare.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(q2_execute=tA, q2_perfect=tC, q3_execute=t3A, q3_perfect=t3C), f, ensure_ascii=False, indent=1)
