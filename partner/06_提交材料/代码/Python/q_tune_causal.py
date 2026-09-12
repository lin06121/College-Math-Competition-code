# -*- coding: utf-8 -*-
"""问题二 报童分位 q 与储能保护下限 reserve 的 1 月标定(因果口径)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, run_q2, ROOT

JAN = 31
rows = {}
print('=== 1 月标定(因果-实时平衡结算, 计划费+紧急费) ===')
for q in (0.5, 0.6, 0.7, 0.8):
    for rsv in (1200.0, 2200.0, 4000.0):
        R = run_q2(q=q, reserve=rsv, upto=JAN)
        jan = float((R['pc'] + R['ec'])[7:JAN].sum())
        rows[f'q{q}_r{int(rsv)}'] = jan
        print(f'  q={q:.1f}  保护下限={rsv:6.0f} kWh -> 1月(1.8-1.31) {jan:>12,.0f} 元'
              f'  | 紧急费 {R["ec"][7:JAN].sum():>10,.0f} 元')
best = min(rows, key=rows.get)
print(f'[标定结果] 最优 = {best} ({rows[best]:,.0f} 元)')
json.dump(rows, open(os.path.join(ROOT, r'04_建模求解\data\q_tune_causal.json'), 'w',
                     encoding='utf-8'), ensure_ascii=False, indent=1)
