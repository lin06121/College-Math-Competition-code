# -*- coding: utf-8 -*-
"""M6 两阶段随机规划(SAA) 在因果实时平衡结算下的表现
   情景 = 最近 20 天实测(负载, 光伏), 等概率; 决策 P0 最小化 "计划费 + 期望紧急费"
   (情景内储能自由度按各情景独立分配 —— 这正是失效根源)
   评价: 用同一套实时平衡控制口径结算真实日 (与 M8 可比)
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy.optimize import linprog
from causal_core import D, price, balance, buffer_of, n, ROOT, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from engine import solve_day

h = 144
EPS = 1e-4
NS = 20


def saa_plan(L_scen, PV_scen, soc0, v, p=price):
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
            row[off(s, 2 * h) + t] = -1.0
            row[off(s, h) + t] = 1.0
            row[off(s, 0) + t] = -1.0
            row[t] = -1.0
            Aub.append(row); bub.append(PV_scen[s, t] - L_scen[s, t])
        for t in range(h):
            row = np.zeros(N)
            row[off(s, 3 * h) + t] = -1.0; row[off(s, 3 * h) + t + 1] = 1.0
            row[off(s, 0) + t] = -ETA_C; row[off(s, h) + t] = 1.0 / ETA_D
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


def main():
    v_const = 0.9 * price.mean()
    soc = 6000.0; plan = em = eQ = 0.0
    t0 = time.time()
    for i in range(n):
        v = v_const if i < n - 1 else 0.0
        S = min(NS, i)
        if S >= 2:
            P0 = saa_plan(D.L_act[i - S:i], D.PV_act[i - S:i], soc, v)
        else:
            P0 = solve_day(price, D.Lb[i], D.Pb[i], soc, soc_end=None, v=v)['P']
        C, Dd, sEnd, Em = balance(P0, np.zeros(h), np.zeros(h), soc, D.L_act[i], D.PV_act[i])
        if i >= D.m2:
            plan += float(price @ P0); em += float(5 * price @ Em); eQ += float(Em.sum())
        soc = sEnd
        if (i + 1) % 50 == 0:
            print(f'  day {i+1}/{n}  累计 {plan+em:,.0f} 元  用时 {time.time()-t0:.0f}s', flush=True)
    print(f'M6 两阶段SAA(因果实时平衡结算): 计划 {plan:,.0f} + 紧急 {em:,.0f} = 总 {plan+em:,.0f} 元 | 紧急电量 {eQ:,.0f} kWh | 用时 {time.time()-t0:.0f}s')
    out = os.path.join(ROOT, r'04_建模求解\data\causal_saa.json')
    json.dump(dict(plan=plan, em=em, tot=plan + em, eq=eQ), open(out, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('saved ->', out)


if __name__ == '__main__':
    main()
