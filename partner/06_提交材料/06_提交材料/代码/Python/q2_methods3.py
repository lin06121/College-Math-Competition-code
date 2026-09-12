# -*- coding: utf-8 -*-
"""第二问计划方法大扫: 期望型两阶段 / 尾部情景 / CVaR / 鲁棒(min-max) / 情景规模
   先在 2 月快评, 选出最优后跑全年
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, ROOT
import fc_upgraded as FU
import plan_core as PC
from lstp import plan_lstp

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
DATA = os.path.join(ROOT, '04_建模求解', 'data')
L, _ = FU.build_load()
PV = FU.build_pv()


def scen(i, K=30, S=20, mode='recent'):
    days = list(range(max(0, i - K), i))
    if not days:
        return []
    if mode == 'recent':
        idx = days[-S:]
    elif mode == 'tail':
        score = [float((L_ACT[j] - L[j]).sum() - (PV_ACT[j] - PV[j]).sum()) for j in days]
        order = list(np.argsort(score)[::-1])
        half = max(1, S // 2)
        idx = [days[k] for k in order[:half]] + [days[k] for k in order[half:2 * half]]
    elif mode == 'tail75':
        score = [float((L_ACT[j] - L[j]).sum() - (PV_ACT[j] - PV[j]).sum()) for j in days]
        order = list(np.argsort(score)[::-1])
        idx = [days[k] for k in order[:max(1, int(0.25 * len(days)))] + order[int(0.75 * len(days)):]]
    return [(L_ACT[j] - L[j], PV_ACT[j] - PV[j]) for j in sorted(set(idx))]


def chain(plan='lstp', q=0.8, **kw):
    upto = kw.pop('upto', None) or n
    B = PC.buffer_mat(L, PV, q)
    soc = 6000.0; pc = em = eQ = 0.0
    for i in range(upto):
        p = price
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        if plan == 'buf':
            r = PC.solve_day(p, L[i] + B[i], PV[i], soc, soc_end=None, v=v)
            P, C, Dv = r['P'], r['C'], r['D']
        else:
            rs = scen(i, **{k: v2 for k, v2 in kw.items() if k in ('K', 'S', 'mode')})
            if not rs:
                r = PC.solve_day(p, L[i] + B[i], PV[i], soc, soc_end=None, v=v)
                P, C, Dv = r['P'], r['C'], r['D']
            else:
                o = plan_lstp(p, L[i], PV[i], rs, soc, v, S=len(rs),
                              risk=kw.get('risk', 'mean'), alpha=kw.get('alpha', 0.9),
                              lam=kw.get('lam', 1.0))
                P, C, Dv = o['P'], o['C'], o['D']
        e, curt, sEnd = PC.execute_as_planned(P, C, Dv, soc, L_ACT[i], PV_ACT[i])
        if i >= m2:
            pc += float(p @ P); em += float(5 * p @ e); eQ += float(e.sum())
        soc = sEnd
    return dict(plan=pc, em=em, total=pc + em, em_kwh=eQ)


if __name__ == '__main__':
    t0 = time.time()
    tests = [('M0 分位裕度 q=0.8', dict(plan='buf')),
             ('M1 期望型 S20K30', dict(K=30, S=20, mode='recent')),
             ('M6 尾部情景 S20K30', dict(K=30, S=20, mode='tail')),
             ('M6b 尾部75% S20K60', dict(K=60, S=20, mode='tail75')),
             ('M8 期望型 S30K60', dict(K=60, S=30, mode='recent')),
             ('M12 鲁棒 min-max', dict(K=30, S=20, mode='tail', risk='robust')),
             ('M13 CVaR90 λ=0.5', dict(K=30, S=20, mode='tail', risk='cvar', alpha=0.9, lam=0.5)),
             ('M14 CVaR90 λ=1', dict(K=30, S=20, mode='tail', risk='cvar', alpha=0.9, lam=1.0))]
    res = {}
    for tag, kw in tests:
        R = chain(upto=59, **kw)
        res[tag] = R
        print(f'  {tag:22s}: 计划 {R["plan"]:>11,.0f} + 紧急 {R["em"]:>9,.0f} = {R["total"]:>12,.0f} 元 '
              f'| 紧急电量 {R["em_kwh"]:>9,.0f} kWh | {time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(os.path.join(DATA, 'q2_methods_feb2.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
    best = min(res, key=lambda k: res[k]['total'])
    print(f'\n[2 月最优] {best}: {res[best]["total"]:,.0f} 元')
