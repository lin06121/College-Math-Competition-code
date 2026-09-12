# -*- coding: utf-8 -*-
"""口径 C 下问题三/四之三 调整方案对照(含"调整时刻带报童裕度"与"不带"两种)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import ROOT, PR
import fc_upgraded as FU
import plan_core as PC

DATA = os.path.join(ROOT, '04_建模求解', 'data')
Q = json.load(open(os.path.join(DATA, 'plan_numbers.json'), encoding='utf-8'))['q_star']
L, _ = FU.build_load()
PV = FU.build_pv()
out = {}
for tag, buf in [('带裕度', True), ('不带裕度', False)]:
    print(f'===== 调整时刻{tag} =====')
    print('--- 问题三(固定电价) ---')
    for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((18,), '+18:00'),
                   ((6, 12), '+6,+12'), ((12, 18), '+12,+18'), ((6, 12, 18), '全滚动')]:
        R = PC.run_q3(L, PV, q=Q, adjust=at, adj_buffer_on=buf)
        out[f'{tag}|q3|{nm}'] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']),
                                     total=float(R['total']))
        print(f'  {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
    print('--- 问题四之三(实时电价) ---')
    for at, nm in [((), '仅0:00'), ((12,), '+12:00'), ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]:
        R = PC.run_q3(L, PV, q=Q, pmat=PR, adjust=at, adj_buffer_on=buf)
        out[f'{tag}|q43|{nm}'] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']),
                                      total=float(R['total']))
        print(f'  {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
json.dump(out, open(os.path.join(DATA, 'plan_q3_options.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved -> data/plan_q3_options.json')
