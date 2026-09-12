# -*- coding: utf-8 -*-
"""三种执行/结算方式对照(问题二, q=0.7, 保护下限=物理下限)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from causal_core import run_q2, ROOT

out = {}
for mode, name in [('asplanned', '僵硬执行计划'), ('causal', '实时平衡控制(采用)'), ('perfect', '完美信息再调度')]:
    R = run_q2(mode=mode, q=0.7, reserve=1200.0)
    out[mode] = dict(plan=R['plan'], em=R['em'], tot=R['total'])
    print(f'{name:16s}: 计划 {R["plan"]:>12,.0f} + 紧急 {R["em"]:>11,.0f} = {R["total"]:>13,.0f} 元')

q3 = run_q2(mode='causal', q=0.7, reserve=1200.0)
print(f'[对照] 实时平衡相对僵硬执行节约 {out["asplanned"]["tot"]-out["causal"]["tot"]:,.0f} 元 '
      f'({(out["asplanned"]["tot"]-out["causal"]["tot"])/out["asplanned"]["tot"]*100:.2f}%)')
json.dump(out, open(os.path.join(ROOT, r'04_建模求解\data\q2_exec_modes.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
