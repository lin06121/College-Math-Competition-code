# -*- coding: utf-8 -*-
"""补充: Python 侧问题三/四之三 在 q=0.7 下的全年费用(便于与 MATLAB 的 fc 模式直接对照)
   —— MATLAB 的 1 月标定规则是基于问题二管道选 q, 因此其 Q3/Q4-3 使用 q=0.7
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fc_common import D, FC0, RTP, run_q3_fc, DATA

LC = np.load(os.path.join(DATA, 'load_combined.npz'))['L_comb']
PZ = dict(np.load(os.path.join(DATA, 'final_preds.npz')))
PVS = PZ['PV|堆叠(mean4+ef2+lgbm+fc0)']
out = {}
for tag, kw in [('q3_q0.7', dict()), ('q3_q0.6', dict()), ('q43_q0.7', dict(price_plan=RTP)),
                ('q43_q0.6', dict(price_plan=RTP))]:
    q = 0.7 if '0.7' in tag else 0.6
    R = run_q3_fc(LC, PVS, q=q, **kw)
    out[tag] = R['total']
    print(f'  {tag}: {R["total"]:,.0f} 元  (计划 {R["plan"]:,.0f} + 调整 {R["dev"]:,.0f} + 紧急 {R["em"]:,.0f})')
json.dump(out, open(os.path.join(DATA, 'q3_q43_q07.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('saved -> data/q3_q43_q07.json')
