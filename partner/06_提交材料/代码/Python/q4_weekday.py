# -*- coding: utf-8 -*-
"""问题四: 采用方案的周内充放电规律(验证 LP 是否自动利用'周五六低价')"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, PR, run_q2, ROOT

R42 = run_q2(p_matrix=PR, q=0.7, reserve=1200.0)
R2 = run_q2(q=0.7, reserve=1200.0)
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
out = {}
for tag, R in [('Q4-2(实时电价)', R42), ('Q2(基准电价)', R2)]:
    net = (R['C'] - R['D'])[D.m2:]
    dw = D.dow[D.m2:]
    row = [float(net[dw == k].sum()) / max(1, int((dw == k).sum())) for k in range(7)]
    out[tag] = row
    print(f'{tag}: ' + '  '.join(f'{names[k]} {row[k]:+8.0f}' for k in range(7)) + '  (日均净充电 kWh)')
dwp = PR[D.m2:]
print('实时电价周内均值(元/kWh): ' +
      '  '.join(f'{names[k]} {dwp[D.dow[D.m2:] == k].mean():.4f}' for k in range(7)))
json.dump(out, open(os.path.join(ROOT, r'04_建模求解\data\q4_weekday.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
