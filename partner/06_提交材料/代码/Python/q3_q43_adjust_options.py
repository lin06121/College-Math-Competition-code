# -*- coding: utf-8 -*-
"""问题四之三(实时电价) 各调整时刻对照 + 问题三/四之三"仅 12:00"方案是否更优"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import ROOT, PR
import fc_upgraded as FU
import upgraded_core as UC

DATA = os.path.join(ROOT, '04_建模求解', 'data')
Q = json.load(open(os.path.join(DATA, 'upgraded_all_numbers.json'), encoding='utf-8'))['q_star']
L, _ = FU.build_load()
PV = FU.build_pv()
out = {}
print('=== 问题三(固定电价) ===')
for at, nm in [((), '仅0:00'), ((12,), '+12:00'), ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]:
    R = UC.run_q3(L, PV, q=Q, adjust=at)
    out['q3_' + nm] = dict(total=float(R['total']), plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']))
    print(f'  {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
print('=== 问题四之三(实时电价) ===')
for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((18,), '+18:00'),
               ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]:
    R = UC.run_q3(L, PV, q=Q, adjust=at, pmat=PR)
    out['q43_' + nm] = dict(total=float(R['total']), plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']))
    print(f'  {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
json.dump(out, open(os.path.join(DATA, 'q3_q43_adjust_options.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved -> data/q3_q43_adjust_options.json')
