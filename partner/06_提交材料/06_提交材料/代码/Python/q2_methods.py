# -*- coding: utf-8 -*-
"""第二问: 多种计划方法的对照(口径 C: 0:00 决策 + 照计划执行 + 5 倍紧急购电)
   M0 预测+逐时段分位裕度(现行)
   M1 两阶段随机规划(残差情景, 第二阶段无储能自由度)
   M2 两阶段随机规划 + 情景放大系数
   M3 两阶段随机规划 + 分位裕度混合
   先在 2 月快评, 再跑全年
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


def chain(method, q=0.8, S=20, K=30, w=1.0, upto=None, pmat=None, mix_q=0.0):
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    B = PC.buffer_mat(L, PV, q)
    soc = 6000.0
    pc = em = 0.0; eQ = 0.0
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        if method == 'M0':                                   # 预测 + 分位裕度
            r = PC.solve_day(p, L[i] + B[i], PV[i], soc, soc_end=None, v=v)
            P, C, Dv = r['P'], r['C'], r['D']
        else:
            resid = [(L_ACT[j] - L[j], PV_ACT[j] - PV[j]) for j in range(max(0, i - K), i)]
            if not resid:
                r = PC.solve_day(p, L[i] + B[i], PV[i], soc, soc_end=None, v=v)
                P, C, Dv = r['P'], r['C'], r['D']
            else:
                if method == 'M3':                           # 两阶段 + 分位裕度混合
                    out = plan_lstp(p, L[i] + mix_q * B[i], PV[i], resid, soc, v, S=S, w_extra=w)
                else:
                    out = plan_lstp(p, L[i], PV[i], resid, soc, v, S=S, w_extra=w)
                P, C, Dv = out['P'], out['C'], out['D']
        e, curt, sEnd = PC.execute_as_planned(P, C, Dv, soc, L_ACT[i], PV_ACT[i])
        if i >= m2:
            pc += float(p @ P); em += float(5 * p @ e); eQ += float(e.sum())
        soc = sEnd
    return dict(plan=pc, em=em, total=pc + em, em_kwh=eQ)


if __name__ == '__main__':
    t0 = time.time()
    print('=== 2 月快评(前 59 天, 结果窗口内 2.1-2.28) ===')
    for tag, kw in [('M0 分位裕度 q=0.8', dict(method='M0')),
                    ('M1 两阶段 S=20,K=30', dict(method='M1', S=20, K=30)),
                    ('M1b 两阶段 S=30,K=60', dict(method='M1', S=30, K=60)),
                    ('M2 两阶段 w=1.2', dict(method='M2', S=20, K=30, w=1.2)),
                    ('M3 两阶段+0.5裕度', dict(method='M3', S=20, K=30, mix_q=0.5))]:
        R = chain(upto=59, **kw)
        print(f'  {tag:22s}: 计划 {R["plan"]:>11,.0f} + 紧急 {R["em"]:>9,.0f} = {R["total"]:>12,.0f} 元 '
              f'| 紧急电量 {R["em_kwh"]:>9,.0f} kWh | {time.time()-t0:.0f}s')
