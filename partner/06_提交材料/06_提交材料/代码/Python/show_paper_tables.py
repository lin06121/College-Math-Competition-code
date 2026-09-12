# -*- coding: utf-8 -*-
"""打印论文表 1/2/3 所需数据(便于逐项核对)"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
P = r'05_论文与结果\论文\paper_tables.json'
T = json.load(open(P, encoding='utf-8'))
D4 = ('2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21')
for key in ('q2', 'q3', 'q4_3'):
    print(f'===== {key} =====')
    for ds in D4:
        d = T[key][ds]
        t1 = ' '.join(f"{k.split('-')[0]}={v:.2f}" for k, v in d['t1'].items() if v > 0)
        blocks = ' '.join(f"({a:.2f},{b:.2f})" for a, b in d['blocks'])
        print(f"  {ds}: t1[{t1}]")
        print(f"      total={d['total']:.2f} cost={d['cost']:.2f}")
        print(f"      blocks={blocks}")
        print(f"      soc0={d['soc0']:.2f} soc24={d['soc24']:.2f} em_total={d['em_total']:.2f} {d['emerg']}")
        if 'adj_total' in d:
            print(f"      adj_total={d['adj_total']:.2f} adj_cost={d['adj_cost']:.2f}")
