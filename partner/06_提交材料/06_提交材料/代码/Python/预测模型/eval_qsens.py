# -*- coding: utf-8 -*-
"""稳健性: 新模型下报童分位 q 的敏感性(问题二、问题四之二)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fc_common import D, RTP, run_q2_fc, DATA

LC = np.load(os.path.join(DATA, 'load_combined.npz'))['L_comb']
PZ = dict(np.load(os.path.join(DATA, 'final_preds.npz')))
PVS = PZ['PV|堆叠(mean4+ef2+lgbm+fc0)']
out = {}
print('=== 问题二(固定电价) 新模型在不同 q 下的全年费用 ===')
for q in (0.6, 0.65, 0.7, 0.75, 0.8):
    R = run_q2_fc(LC, PVS, q=q)
    out[f'q2_q{q}'] = R['total']
    print(f'  q={q}: {R["total"]:,.0f} 元 (计划 {R["plan"]:,.0f} + 紧急 {R["em"]:,.0f})')
print('=== 问题四之二(实时电价) ===')
for q in (0.6, 0.7, 0.8):
    R = run_q2_fc(LC, PVS, q=q, price_plan=RTP)
    out[f'q42_q{q}'] = R['total']
    print(f'  q={q}: {R["total"]:,.0f} 元')
json.dump(out, open(os.path.join(DATA, 'q_sensitivity_new.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved -> data/q_sensitivity_new.json')
