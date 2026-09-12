# -*- coding: utf-8 -*-
"""补充对照: (1) 各光伏计划依据的精度 (2) 计划基值统一后的 问题三/四之三 结果"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, PR, balance, buffer_of, run_q3, n, ROOT
from engine import solve_day

# ---------- 光伏依据精度(结果窗口 2.1-12.31) ----------
m2 = D.m2
for name, A in [('0:00 预报(附件3)', D.fc0m), ('近 4 日均值', D.Pb), ('昨日实测', D.PV_act[np.r_[0, np.arange(n - 1)]]),
                ('附件1 典型日', np.broadcast_to(D.PV_typ, D.PV_act.shape))]:
    err = np.abs(A[m2:] - D.PV_act[m2:])
    print(f'  {name:16s}: 光伏 MAE = {err.mean():7.2f} kWh/时段 ({err.mean()*6:8.1f} kW)')
for name, A in [('同类日平均(4日)', D.Lb), ('昨日实测', D.L_act[np.r_[0, np.arange(n - 1)]]),
                ('附件1 典型日', np.broadcast_to(D.L_typ, D.L_act.shape))]:
    err = np.abs(A[m2:] - D.L_act[m2:])
    print(f'  {name:16s}: 负载 MAE = {err.mean():7.2f} kWh/时段 ({err.mean()*6:8.1f} kW)')

# ---------- 计划基值统一(0:00 用同类日+近4日均值)的 问题三/四之三 ----------
def run_q3_uni(p_matrix=None, adjust=(6, 12, 18), q=0.7, reserve=1200.0):
    """与 run_q3 相同, 但 0:00 计划不用预报而用历史统计基值(与问题二一致)"""
    soc = 6000.0; H = [36, 72, 108]; fh = [6, 12, 18]
    pc = dv = ec = 0.0
    for i in range(n):
        pday = price if p_matrix is None else p_matrix[i]
        v = (0.9 * (price.mean() if p_matrix is None else p_matrix[min(i + 1, n - 1)].mean())) if i < n - 1 else 0.0
        Lbar = D.Lb[i] + buffer_of(i, q)
        r0 = solve_day(pday, Lbar, D.Pb[i], soc, soc_end=None, v=v)
        P, C, Dd, SOC = r0['P'], r0['C'], r0['D'], r0['E']; Pinit = P.copy()
        for h0, hh in zip(H, fh):
            if hh not in adjust: continue
            r = solve_day(pday[h0:], Lbar[h0:], D.fca_m[hh][i, h0:], SOC[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], Dd[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0 + 1], r['E'][1:]])
        C2, D2, sEnd, Em = balance(P, C, Dd, soc, D.L_act[i], D.PV_act[i], reserve)
        if i >= m2:
            pc += float(pday @ Pinit)
            dv += float(np.sum(-0.5 * pday * np.maximum(Pinit - P, 0) + 1.5 * pday * np.maximum(P - Pinit, 0)))
            ec += float(5 * pday @ Em)
        soc = sEnd
    return pc, dv, ec, pc + dv + ec

print('\n=== 计划基值统一(0:00 用历史统计基值) ===')
p, d, e, t = run_q3_uni()
print(f'  Q3  全滚动(基值统一): 计划 {p:,.0f} + 调整 {d:,.0f} + 紧急 {e:,.0f} = {t:,.0f} 元')
p2, d2, e2, t2 = run_q3_uni(p_matrix=PR)
print(f'  Q4-3 全滚动(基值统一): 计划 {p2:,.0f} + 调整 {d2:,.0f} + 紧急 {e2:,.0f} = {t2:,.0f} 元')
out = os.path.join(ROOT, r'04_建模求解\data\causal_unibasis.json')
json.dump(dict(q3_uni=t, q43_uni=t2, q3_uni_parts=[p, d, e], q43_uni_parts=[p2, d2, e2]),
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved ->', out)
