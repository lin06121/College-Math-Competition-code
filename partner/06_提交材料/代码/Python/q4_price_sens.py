# -*- coding: utf-8 -*-
"""灵敏度: 0:00 不能获知当日实时电价, 只能用"周内类别均价"预测电价(问题四之二)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, PR, run_q2, ROOT

low = np.array([d.dayofweek in {4, 5} for d in D.dates])
PHAT = np.zeros_like(PR)
for i in range(D.n):
    hist = [j for j in range(i) if low[j] == low[i]]
    PHAT[i] = PR[hist[-30:]].mean(axis=0) if len(hist) >= 5 else PR[:max(1, i)].mean(axis=0)
print('预测电价误差: MAE = %.4f 元/kWh' % np.abs(PHAT[D.m2:] - PR[D.m2:]).mean())
R = run_q2(p_matrix=PHAT, q=0.7, reserve=1200.0, bill_matrix=PR)
B = run_q2(p_matrix=PR, q=0.7, reserve=1200.0)
print(f'已知实时电价(采用): 总 {B["total"]:,.0f} 元')
print(f'仅知周内规律预测:   总 {R["total"]:,.0f} 元  ({(R["total"]-B["total"])/B["total"]*100:+.2f}%)')
json.dump(dict(known=B['total'], pattern=R['total'], mae=float(np.abs(PHAT[D.m2:] - PR[D.m2:]).mean())),
          open(os.path.join(ROOT, r'04_建模求解\data\q4_price_sens.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
