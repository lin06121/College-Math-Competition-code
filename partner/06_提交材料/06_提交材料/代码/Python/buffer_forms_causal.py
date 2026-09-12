# -*- coding: utf-8 -*-
"""安全裕度(缓冲)形式的对照实验 —— 因果口径统一重算
   基准: 近30天净残差 q=0.7 经验分位(采用)
   对照: 正态参数化(按小时 sigma)、负荷/光伏噪声分别取分位后叠加、60/90 天窗、逐月滚动 q*
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from scipy import stats
from causal_core import D, price, balance, n, ROOT
from engine import solve_day

price_v = price
m2 = D.m2
hour_of = np.arange(144) // 6
eL = D.L_act - D.Lb
eP = D.PV_act - D.Pb
eN = eL - eP
mon = np.array([d.month for d in D.dates])


def buf_of(mode, i, q=0.7, W=30):
    lo = max(0, i - W)
    if i - lo < 7:
        return np.zeros(144)
    if mode == 'net':
        return np.quantile(eN[lo:i], q, axis=0)
    if mode == 'split':
        return np.quantile(eL[lo:i], q, axis=0) + np.quantile(-eP[lo:i], q, axis=0)
    if mode == 'normal':
        z = stats.norm.ppf(q); b = np.zeros(144)
        for h in range(24):
            b[hour_of == h] = z * eN[lo:i][:, hour_of == h].ravel().std()
        return b
    raise ValueError(mode)


def chain(mode, q=0.7, W=30, q_by_month=None):
    soc = 6000.0; tot = 0.0
    for i in range(n):
        v = 0.9 * price_v.mean() if i < n - 1 else 0.0
        qq = q if q_by_month is None else q_by_month.get(mon[i], q)
        b = buf_of(mode, i, qq, W)
        r = solve_day(price_v, D.Lb[i] + b, D.Pb[i], soc, soc_end=None, v=v)
        C, Dd, sEnd, Em = balance(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
        if i >= m2:
            tot += float(price_v @ r['P']) + float(5 * price_v @ Em)
        soc = sEnd
    return tot


res = {}
base = chain('net'); res['采用 经验分位 q=0.7 (30天窗)'] = base
print(f'  采用 经验分位 q=0.7 (30天窗) : {base:,.0f} 元')
for tag, kw in [('正态参数化(按小时 σ, z0.7)', dict(mode='normal')),
                ('负荷/光伏噪声分别取分位叠加', dict(mode='split')),
                ('经验分位 60 天窗', dict(mode='net', W=60)),
                ('经验分位 90 天窗', dict(mode='net', W=90))]:
    x = chain(**kw); res[tag] = x
    print(f'  {tag:28s}: {x:,.0f} 元  (差 {x-base:+,.0f}, {(x-base)/base*100:+.2f}%)')
# 逐月滚动重调 q*
q_by_month = {1: 0.7}
for mm in range(2, 13):
    seg = np.where(mon == mm - 1)[0]
    best = (np.inf, 0.7)
    for ql in (0.5, 0.6, 0.7, 0.8):
        soc = 6000.0; tot = 0.0
        for i in range(seg[0], seg[-1] + 1):
            v = 0.9 * price_v.mean() if i < n - 1 else 0.0
            r = solve_day(price_v, D.Lb[i] + buf_of('net', i, ql), D.Pb[i], soc, soc_end=None, v=v)
            C, Dd, sEnd, Em = balance(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
            tot += float(price_v @ r['P']) + float(5 * price_v @ Em); soc = sEnd
        if tot < best[0]:
            best = (tot, ql)
    q_by_month[mm] = best[1]
print('  逐月滚动 q* =', q_by_month)
x = chain('net', q_by_month=q_by_month); res['逐月滚动重调 q*'] = x
print(f'  逐月滚动重调 q*              : {x:,.0f} 元  (差 {x-base:+,.0f}, {(x-base)/base*100:+.2f}%)')
json.dump(dict(base=base, rows=res, q_by_month=q_by_month),
          open(os.path.join(ROOT, r'04_建模求解\data\buffer_forms_causal.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
