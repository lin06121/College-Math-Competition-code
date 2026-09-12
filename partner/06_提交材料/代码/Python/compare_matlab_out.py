# -*- coding: utf-8 -*-
"""MATLAB 版结果文件 vs Python 版结果文件 逐表比对(费用合计 + 购电量合计)"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from causal_core import ROOT

A = os.path.join(ROOT, '05_论文与结果')
B = os.path.join(A, 'matlab_out')
FILES = ['result1.xlsx', 'result2.xlsx', 'result3.xlsx', 'result4-2.xlsx', 'result4-3.xlsx']


def plan_sums(path):
    out = {}
    for sh in ('计划购电量', '调整购电量'):
        try:
            d = pd.read_excel(path, sheet_name=sh, header=0)
        except Exception:
            continue
        if d.shape[1] != 147:
            print(f'    ({sh}: 形状 {d.shape}, 跳过)')
            continue
        d.columns = ['date'] + [f'c{i}' for i in range(1, 145)] + ['tot', 'cost']
        d = d[pd.to_datetime(d['date'], errors='coerce').notna()]
        out[sh] = (float(d['tot'].sum()), float(d['cost'].sum()))
    try:
        em = pd.read_excel(path, sheet_name='紧急购电量', header=0)
        em.columns = ['date', 'span', 'kwh']
        out['紧急购电量'] = float(em['kwh'].fillna(0).sum())
    except Exception:
        pass
    return out


for f in FILES:
    pa, pb = os.path.join(A, f), os.path.join(B, f)
    if not os.path.exists(pb):
        print(f'{f}: MATLAB 版缺失'); continue
    sa, sb = plan_sums(pa), plan_sums(pb)
    print(f'--- {f} ---')
    for k in sa:
        va, vb = sa[k], sb.get(k, (None, None))
        if isinstance(va, tuple):
            print(f'   {k}: Python tot={va[0]:,.2f} cost={va[1]:,.0f} | MATLAB tot={vb[0]:,.2f} cost={vb[1]:,.0f} '
                  f'| Δcost={vb[1]-va[1]:+,.0f}')
        else:
            print(f'   {k}: Python={va:,.2f} | MATLAB={vb:,.2f} | Δ={vb-va:+,.2f}')
