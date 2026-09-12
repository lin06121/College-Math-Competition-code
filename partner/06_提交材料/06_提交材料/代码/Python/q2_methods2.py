# -*- coding: utf-8 -*-
"""第二问计划方法进阶对照(2 月快评):
   M0 现行: 预测 + 逐时段分位裕度
   M1 两阶段随机规划(残差情景)
   M4 两阶段 + 残值系数放大(v 倍数)
   M5 两阶段 + 相似日情景(按光伏日总量/负载水平挑选历史相似日)
   M6 两阶段 + 尾部加权情景(优先取净残差最差的日子)
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, PR, ROOT
import fc_upgraded as FU
import plan_core as PC
from lstp import plan_lstp

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
DATA = os.path.join(ROOT, '04_建模求解', 'data')
L, INFO = FU.build_load()
PV = FU.build_pv()
FC0 = D.fc0m


def scen_resid(i, K=30, S=20, mode='recent'):
    lo = max(0, i - K)
    days = list(range(lo, i))
    if not days:
        return []
    if mode == 'recent':
        idx = days[-S:] if len(days) > S else days
    elif mode == 'analog':                     # 相似日: 光伏日总量 + 负载水平最接近今天的
        key_now = np.array([PV[i].sum(), L[i].sum(), FC0[i].sum()])
        d = []
        for j in days:
            kj = np.array([PV[j].sum(), L[j].sum(), FC0[j].sum()])
            d.append(np.abs(kj[0] - key_now[0]) / max(key_now[0], 1) + np.abs(kj[1] - key_now[1]) / max(key_now[1], 1)
                     + np.abs(kj[2] - key_now[2]) / max(key_now[2], 1))
        idx = [days[k] for k in np.argsort(d)[:S]]
    elif mode == 'tail':                       # 尾部加权: 净残差(负荷-光伏)最大的 S/2 天 + 随机 S/2 天
        score = [(L_ACT[j] - L[j]).sum() - (PV_ACT[j] - PV[j]).sum() for j in days]
        order = np.argsort(score)[::-1]
        half = max(1, S // 2)
        idx = [days[k] for k in order[:half]] + [days[k] for k in order[half:half * 2]]
    else:
        idx = days[-S:]
    return [(L_ACT[j] - L[j], PV_ACT[j] - PV[j]) for j in sorted(idx)]


def chain(method, q=0.8, K=30, S=20, vmult=1.0, mode='recent', upto=None, pmat=None):
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    B = PC.buffer_mat(L, PV, q)
    soc = 6000.0; pc = em = eQ = 0.0
    for i in range(upto):
        p = pm[i]
        v0 = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        v = v0 * vmult
        if method == 'M0':
            r = PC.solve_day(p, L[i] + B[i], PV[i], soc, soc_end=None, v=v0)
            P, C, Dv = r['P'], r['C'], r['D']
        else:
            rs = scen_resid(i, K=K, S=S, mode=mode)
            if not rs:
                r = PC.solve_day(p, L[i] + B[i], PV[i], soc, soc_end=None, v=v0)
                P, C, Dv = r['P'], r['C'], r['D']
            else:
                out = plan_lstp(p, L[i], PV[i], rs, soc, v, S=len(rs))
                P, C, Dv = out['P'], out['C'], out['D']
        e, curt, sEnd = PC.execute_as_planned(P, C, Dv, soc, L_ACT[i], PV_ACT[i])
        if i >= m2:
            pc += float(p @ P); em += float(5 * p @ e); eQ += float(e.sum())
        soc = sEnd
    return dict(plan=pc, em=em, total=pc + em, em_kwh=eQ)


if __name__ == '__main__':
    t0 = time.time()
    tests = [('M0 现行(分位裕度)', dict(method='M0')),
             ('M1 两阶段', dict(method='M1')),
             ('M4 两阶段 v×2', dict(method='M1', vmult=2.0)),
             ('M4b 两阶段 v×3', dict(method='M1', vmult=3.0)),
             ('M5 相似日情景', dict(method='M1', mode='analog')),
             ('M6 尾部加权情景', dict(method='M1', mode='tail')),
             ('M7 相似日+v×2', dict(method='M1', mode='analog', vmult=2.0)),
             ('M8 两阶段 S=30,K=60', dict(method='M1', S=30, K=60))]
    res = {}
    for tag, kw in tests:
        R = chain(upto=59, **kw)
        res[tag] = R
        print(f'  {tag:20s}: 计划 {R["plan"]:>11,.0f} + 紧急 {R["em"]:>9,.0f} = {R["total"]:>12,.0f} 元 '
              f'| 紧急电量 {R["em_kwh"]:>9,.0f} kWh | {time.time()-t0:.0f}s', flush=True)
    json.dump(res, open(os.path.join(DATA, 'q2_methods_feb.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1, default=float)
