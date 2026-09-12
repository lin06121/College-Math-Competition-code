# -*- coding: utf-8 -*-
"""从已生成的 result*.xlsx 提取论文表1/2/3 数据 -> 05_论文与结果/论文/paper_tables.json"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from io_results import load_all, ROOT
sys.path.insert(0, os.path.join(ROOT, r'04_建模求解\scripts'))

OUT = os.path.join(ROOT, r'05_论文与结果')
PDIR = os.path.join(OUT, '论文')
slot_of = {'10:00-10:10': 61, '12:00-12:10': 73, '14:00-14:10': 85,
           '16:00-16:10': 97, '18:00-18:10': 109, '20:00-20:10': 121}
spec = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
blocks = ['0:00-4:00', '4:00-8:00', '8:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00']
res = {}

# ---------- result1 ----------
p1 = pd.read_excel(os.path.join(OUT, 'result1.xlsx'), sheet_name='计划购电量', header=None)
P1 = p1.iloc[1:, 1].astype(float).to_numpy()
c1 = pd.read_excel(os.path.join(OUT, 'result1.xlsx'), sheet_name='充放电量', header=None)
res['q1'] = dict(
    t1={k: round(float(P1[v - 1]), 2) for k, v in slot_of.items()},
    total=round(float(P1.sum()), 2),
    blocks=[[round(float(c1.iloc[1 + b, 1]), 2), round(float(c1.iloc[1 + b, 2]), 2)] for b in range(6)],
    soc0=float(c1.iloc[1, 4]), soc24=float(c1.iloc[2, 4]))

def read_plan(fn, sheet='计划购电量'):
    d = pd.read_excel(os.path.join(OUT, fn), sheet_name=sheet, header=0)
    d.columns = ['date'] + [f'c{i}' for i in range(1, 145)] + ['tot', 'cost']
    return d

def read_cd(fn, sheet='充放电量'):
    d = pd.read_excel(os.path.join(OUT, fn), sheet_name=sheet, header=0)
    d.columns = ['date', 'block', 'chg', 'dis', 't', 'soc']
    d['date'] = pd.to_datetime(d['date']).ffill()
    return d

def read_em(fn):
    d = pd.read_excel(os.path.join(OUT, fn), sheet_name='紧急购电量', header=0)
    d.columns = ['date', 'span', 'kwh']
    d['date'] = pd.to_datetime(d['date']).ffill()
    return d

def extract(fn):
    pl = read_plan(fn); cd = read_cd(fn); em = read_em(fn)
    out = {}
    for ds in spec:
        dt = pd.Timestamp(ds)
        row = pl[pl['date'] == dt].iloc[0]
        pv = [float(row[f'c{i}']) for i in range(1, 145)]
        blk = cd[cd['date'] == dt]
        segs = em[(em['date'] == dt) & (em['span'].notna())]
        out[ds] = dict(
            t1={k: round(pv[v - 1], 2) for k, v in slot_of.items()},
            total=round(float(row['tot']), 2), cost=round(float(row['cost']), 2),
            blocks=[[round(float(b['chg']), 2), round(float(b['dis']), 2)] for _, b in blk.iterrows()],
            soc0=round(float(blk['soc'].iloc[0]), 2), soc24=round(float(blk['soc'].iloc[1]), 2),
            emerg=[[str(s['span']), round(float(s['kwh']), 2)] for _, s in segs.iterrows()],
            em_total=round(float(segs['kwh'].sum()), 2))
    return out

res['q2'] = extract('result2.xlsx')
res['q4_2'] = extract('result4-2.xlsx')
res['q3'] = extract('result3.xlsx')
res['q4_3'] = extract('result4-3.xlsx')
# 调整购电量汇总(指定日期)
for key, fn in [('q3', 'result3.xlsx'), ('q4_3', 'result4-3.xlsx')]:
    adj = read_plan(fn, '调整购电量')
    for ds in spec:
        dt = pd.Timestamp(ds)
        row = adj[adj['date'] == dt].iloc[0]
        res[key][ds]['adj_total'] = round(float(row['tot']), 2)
        res[key][ds]['adj_cost'] = round(float(row['cost']), 2)

with open(os.path.join(PDIR, 'paper_tables.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print('Q1:', json.dumps(res['q1'], ensure_ascii=False))
print('Q2 2025-03-20:', json.dumps(res['q2']['2025-03-20'], ensure_ascii=False))
print('Q2 2025-06-21:', json.dumps(res['q2']['2025-06-21'], ensure_ascii=False))
print('Q2 2025-09-23:', json.dumps(res['q2']['2025-09-23'], ensure_ascii=False))
print('Q2 2025-12-21:', json.dumps(res['q2']['2025-12-21'], ensure_ascii=False))
print('Q3 2025-03-20:', json.dumps(res['q3']['2025-03-20'], ensure_ascii=False))
print('Q3 2025-06-21:', json.dumps(res['q3']['2025-06-21'], ensure_ascii=False))
print('Q4_3 2025-06-21:', json.dumps(res['q4_3']['2025-06-21'], ensure_ascii=False))
print('paper_tables.json written')
