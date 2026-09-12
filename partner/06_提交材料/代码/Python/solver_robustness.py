# -*- coding: utf-8 -*-
"""Q3 数值稳健性: 换用内点法(highs-ipm)重解, 检验"多重最优解经 SOC 链条传播"造成的漂移量"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import engine
from scipy.optimize import linprog as _lp
import causal_core as cc

orig = engine.linprog
import numpy as _np


def ipm_lp(*a, **kw):
    kw['method'] = 'highs-ipm'
    return orig(*a, **kw)


# 同时替换 engine 与 causal_core 里引用的 solve_day 内部调用
engine_linprog = engine.linprog
engine.linprog = ipm_lp
# 内点法解不是顶点解, 可能出现微小"同时充放"; 预算清掉(等价于删去纯损耗项)
_sd = engine.solve_day


def solve_day_clean(*a, **kw):
    r = _sd(*a, **kw)
    if r.get('status') == 'ok':
        m = np.minimum(r['C'], r['D'])
        if m.max() > 1e-12:
            r['C'] = r['C'] - m; r['D'] = r['D'] - m
            print(f'  [清洗] 最大同时充放 {m.max():.4f} kWh')
        r['C'] = np.maximum(r['C'], 0.0); r['D'] = np.maximum(r['D'], 0.0)
    return r


engine.solve_day = solve_day_clean
cc.solve_day = solve_day_clean
tot = {}
from causal_core import run_q3, run_q2
R = run_q3()
print(f'Q3 内点法重解: 计划 {R["plan"]:,.0f} + 调整 {R["dev"]:,.0f} + 紧急 {R["em"]:,.0f} = {R["total"]:,.0f} 元')
tot['q3_ipm'] = R['total']
R2 = run_q2()
print(f'Q2 内点法重解: 计划 {R2["plan"]:,.0f} + 紧急 {R2["em"]:,.0f} = {R2["total"]:,.0f} 元')
tot['q2_ipm'] = R2['total']
engine.linprog = engine_linprog
print('[对照] Q2 dual-simplex = 13,848,183 元; Q3 dual-simplex = 14,047,116 元')
json.dump(tot, open(os.path.join(cc.ROOT, r'04_建模求解\data\solver_robustness.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
