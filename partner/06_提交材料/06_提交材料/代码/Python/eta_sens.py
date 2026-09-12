# -*- coding: utf-8 -*-
"""效率口径灵敏度(与问题一敏感性同一口径): 单程 0.9/0.9(采用) vs 往返 0.9(单程 sqrt(0.9))
   计划与实时平衡控制使用同一组效率参数"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import engine
import causal_core as cc
from causal_core import D, price, buffer_of, ROOT
from engine import solve_day

m2 = D.m2


def run(eta_c, eta_d, upto=None):
    upto = D.n if upto is None else upto
    old = (engine.ETA_C, engine.ETA_D, cc.ETA_C, cc.ETA_D)
    engine.ETA_C, engine.ETA_D = cc.ETA_C, cc.ETA_D = eta_c, eta_d
    try:
        soc = 6000.0; tot = 0.0
        for i in range(upto):
            v = 0.9 * price.mean() if i < D.n - 1 else 0.0
            buf = buffer_of(i, 0.7)
            r = solve_day(price, D.Lb[i] + buf, D.Pb[i], soc, soc_end=None, v=v)
            C, Dd, sEnd, Em = cc.balance(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
            if i >= m2:
                tot += float(price @ r['P']) + float(5 * price @ Em)
            soc = sEnd
        return tot
    finally:
        engine.ETA_C, engine.ETA_D, cc.ETA_C, cc.ETA_D = old


a = run(0.9, 0.9, upto=59)
b = run(0.9 ** 0.5, 0.9 ** 0.5, upto=59)
print(f'2月 单程 0.9/0.9(采用)     : {a:,.0f} 元')
print(f'2月 往返 0.9(单程 0.9487) : {b:,.0f} 元  ({(b-a)/a*100:+.2f}%)')
c = run(0.9, 0.9)
d = run(0.9 ** 0.5, 0.9 ** 0.5)
print(f'全年 单程 0.9/0.9(采用)    : {c:,.0f} 元')
print(f'全年 往返 0.9(单程 0.9487): {d:,.0f} 元  ({(d-c)/c*100:+.2f}%)')
json.dump(dict(feb_single=a, feb_roundtrip=b, year_single=c, year_roundtrip=d),
          open(os.path.join(ROOT, r'04_建模求解\data\eta_sens.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
