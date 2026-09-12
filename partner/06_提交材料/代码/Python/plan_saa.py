# -*- coding: utf-8 -*-
"""口径 C 下的两阶段随机规划(SAA)对照
   计划: P0 = SAA(近 20 日实测情景, 最小化"计划费 + 期望紧急费");
        储能计划 = 在 P0 固定下, 按对负载/光伏的预测最优安排充放电(与采用方法同一条 LP, 只是购电量给定)
   执行: 照计划执行 -> 缺口按 5 倍电价紧急购电
   失效根源: 情景内储能自由度被重复利用 -> P0 被系统性压低 -> 真实日缺口巨大
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import linprog
from engine import solve_day, ETA_C, ETA_D, P_MAX, SOC_MIN, SOC_MAX
from causal_core import D, price, ROOT
import fc_upgraded as FU
import plan_core as PC

h = 144
n = D.n
m2 = D.m2
EPS = 1e-4
NS = 20
DATA = os.path.join(ROOT, '04_建模求解', 'data')


def saa_plan(L_scen, PV_scen, soc0, v, p=price):
    """两阶段随机规划: 决策 P0, 情景内独立调度储能(自由度可重复利用 —— 失效根源)"""
    S = L_scen.shape[0]
    nP = h; per = 3 * h + (h + 1); N = nP + S * per
    off = lambda s, k: nP + s * per + k
    c = np.zeros(N); c[:nP] = p
    for s in range(S):
        c[off(s, 2 * h):off(s, 3 * h)] += 5.0 * p / S
        c[off(s, 0):off(s, 2 * h)] += EPS / S
        c[off(s, 3 * h) + h] += -v / S
    Aeq, beq, Aub, bub = [], [], [], []
    for s in range(S):
        for t in range(h):
            row = np.zeros(N)
            row[off(s, 2 * h) + t] = -1.0; row[off(s, h) + t] = 1.0
            row[off(s, 0) + t] = -1.0; row[t] = -1.0
            Aub.append(row); bub.append(PV_scen[s, t] - L_scen[s, t])
        for t in range(h):
            row = np.zeros(N)
            row[off(s, 3 * h) + t] = -1.0; row[off(s, 3 * h) + t + 1] = 1.0
            row[off(s, 0) + t] = -ETA_C; row[off(s, h) + t] = 1 / ETA_D
            Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    for s in range(S):
        ub[off(s, 0):off(s, 2 * h)] = P_MAX
        lb[off(s, 3 * h):off(s, 3 * h) + h + 1] = SOC_MIN
        ub[off(s, 3 * h):off(s, 3 * h) + h + 1] = SOC_MAX
        lb[off(s, 3 * h)], ub[off(s, 3 * h)] = soc0, soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success, res.message
    return res.x[:nP]


def planned_storage(p, load_v, pv_v, P0, soc0, v):
    """购电量已由 SAA 给定 P0: 按预测安排储能以最小化"预测口径的紧急购电", 并计入期末残值"""
    hh = h
    iC, iD, iE, iS = 0, hh, 2 * hh, 3 * hh
    N = 4 * hh + (hh + 1)
    c = np.zeros(N); c[iC:iC + hh] = EPS; c[iD:iD + hh] = EPS
    c[iE:iE + hh] = 5.0 * p; c[iS + hh] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(hh):
        row = np.zeros(N); row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(pv_v[t] + P0[t] - load_v[t])
    for t in range(hh):
        row = np.zeros(N); row[iS + t], row[iS + t + 1] = -1, 1
        row[iC + t], row[iD + t] = -ETA_C, 1 / ETA_D
        Aeq.append(row); beq.append(0.0)
    row = np.zeros(N); row[iS] = 1; Aeq.append(row); beq.append(soc0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + hh] = P_MAX; ub[iD:iD + hh] = P_MAX
    lb[iS:iS + hh + 1] = SOC_MIN; ub[iS:iS + hh + 1] = SOC_MAX
    lb[iS] = ub[iS] = soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success, res.message
    return res.x[iC:iC + hh], res.x[iD:iD + hh]


if __name__ == '__main__':
    t0 = time.time()
    v_const = 0.9 * price.mean()
    L_UP, _ = FU.build_load(); PV_UP = FU.build_pv()
    soc = 6000.0; plan = em = eQ = 0.0
    for i in range(n):
        v = v_const if i < n - 1 else 0.0
        S = min(NS, i)
        if S >= 2:
            P0 = saa_plan(D.L_act[i - S:i], D.PV_act[i - S:i], soc, v)
        else:
            P0 = solve_day(price, D.Lb[i], D.Pb[i], soc, soc_end=None, v=v)['P']
        C, Dv = planned_storage(price, L_UP[i], PV_UP[i], P0, soc, v)
        em_i, curt_i, sEnd = PC.execute_as_planned(P0, C, Dv, soc, D.L_act[i], D.PV_act[i])
        if i >= m2:
            plan += float(price @ P0); em += float(5 * price @ em_i); eQ += float(em_i.sum())
        soc = sEnd
        if (i + 1) % 60 == 0:
            print(f'  day {i+1}/{n} 累计 {plan+em:,.0f} 元 {time.time()-t0:.0f}s', flush=True)
    print(f'SAA(口径C): 计划 {plan:,.0f} + 紧急 {em:,.0f} = 总 {plan+em:,.0f} 元 | 紧急电量 {eQ:,.0f} kWh | {time.time()-t0:.0f}s')
    json.dump(dict(plan=float(plan), em=float(em), total=float(plan + em), em_kwh=float(eQ)),
              open(os.path.join(DATA, 'plan_saa.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
