# -*- coding: utf-8 -*-
"""校验论文表 1/2/3 与 paper_tables.json(即结果文件) 逐数字一致"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from causal_core import ROOT

P = os.path.join(ROOT, r'05_论文与结果\论文')
T = json.load(open(os.path.join(P, 'paper_tables.json'), encoding='utf-8'))
spec = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']


def fmt(v):
    s = f'{float(v):.2f}'
    if float(v) >= 10000:
        a, b = s.split('.')
        out = ''
        while len(a) > 3:
            out = r'\,' + a[-3:] + out; a = a[:-3]
        s = a + out + '.' + b
    return s


missing = []
for f, key in [('sec6_q2.tex', 'q2'), ('sec7_q3.tex', 'q3'), ('sec8_q4.tex', 'q4_3')]:
    src = open(os.path.join(P, 'sections', f), encoding='utf-8').read()
    for ds in spec:
        d = T[key][ds]
        vals = list(d['t1'].values()) + [d['total'], d['cost']] + \
               [x for blk in d['blocks'] for x in blk] + [d['soc0'], d['soc24']] + [d['em_total']]
        if 'adj_total' in d:
            vals += [d['adj_total'], d['adj_cost']]
        for v in vals:
            t = fmt(v)
            if t not in src:
                missing.append((f, ds, t))
print('未在 tex 中找到的数字条数:', len(missing))
for m in missing[:25]:
    print('  ', m)
