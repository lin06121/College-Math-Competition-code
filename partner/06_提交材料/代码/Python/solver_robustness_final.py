# -*- coding: utf-8 -*-
"""求解器稳健性对照(口径 A 最终版): 对偶单纯形(highs-ds) vs 内点法(highs-ipm)
   -> data/solver_robustness_final.json
   说明: 内点法返回的是"最优面"上的内点, 可能出现 C_t,D_t 同时为正的退化表示
   (物理上无意义, 但费用相同)。为使两种求解器可以走同一条结算管道,
   这里在结算前把同时充放的部分按 min(C,D) 抵消(标准退化投影)。
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import engine
import causal_core as cc
import A_core as AC

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
Q = 0.7
t0 = time.time()
out = {}


def tolerant_balance(P0, Cpl, Dpl, soc0, L, PV, reserve=cc.SOC_MIN):
    """与 cc.balance 相同, 但在裁剪后先抵消同时充放(退化投影), 不加断言"""
    x = np.minimum(np.maximum(Cpl, 0.0), np.maximum(Dpl, 0.0))
    Cpl = Cpl - x; Dpl = Dpl - x
    return cc.balance(P0, Cpl, Dpl, soc0, L, PV, reserve=reserve)


orig_solve = engine.solve_day
orig_bal = AC.balance
for tag, meth in [('ds', 'highs-ds'), ('ipm', 'highs-ipm')]:
    AC.solve_day = lambda *a, _m=meth, **k: orig_solve(*a, method=_m, **k)
    AC.balance = tolerant_balance
    r2 = AC.run_q2(q=Q)
    r3 = AC.run_q3(q=Q, adjust=(12,), use_adjbuf=True)
    out[tag] = dict(q2=float(r2['total']), q3=float(r3['total']))
    print(f'  {meth:10s}: 问题二 {r2["total"]:>13,.0f} 元 | 问题三(12:00) {r3["total"]:>13,.0f} 元')
AC.solve_day = orig_solve
AC.balance = orig_bal
d2 = (out['ipm']['q2'] - out['ds']['q2']) / out['ds']['q2'] * 100
d3 = (out['ipm']['q3'] - out['ds']['q3']) / out['ds']['q3'] * 100
out['diff_pct'] = dict(q2=float(d2), q3=float(d3))
print(f'  相对差: 问题二 {d2:+.5f}% | 问题三 {d3:+.5f}%')
json.dump(out, open(os.path.join(DATA, 'solver_robustness_final.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'[完成] {time.time()-t0:.0f}s')
