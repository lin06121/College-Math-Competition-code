# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np

base = r'C题\附件'
out = []

# ---- 附件1 ----
a1 = pd.read_excel(os.path.join(base, '附件1.xlsx'))
out.append(f'[附件1] shape={a1.shape} cols={list(a1.columns)}')
out.append(f'  time range: {a1.iloc[0,0]} -> {a1.iloc[-1,0]}')
out.append(f'  nulls: {a1.isna().sum().to_dict()}')
out.append(f'  price min/max={a1.电价.min():.4f}/{a1.电价.max():.4f}')
out.append(f'  load min/max={a1.小区负载.min():.1f}/{a1.小区负载.max():.1f}')
out.append(f'  pv min/max={a1.光伏发电预测功率.min():.1f}/{a1.光伏发电预测功率.max():.1f}')

# ---- 附件2 ----
xl2 = pd.ExcelFile(os.path.join(base, '附件2.xlsx'))
for sh in xl2.sheet_names:
    d = xl2.parse(sh)
    out.append(f'[附件2/{sh}] shape={d.shape}')
    out.append(f'  date range: {d.iloc[0,0]} -> {d.iloc[-1,0]}  ndays={len(d)}')
    out.append(f'  nulls total: {int(d.isna().sum().sum())}')
    out.append(f'  any negative: {int((d.select_dtypes("number")<0).sum().sum())}')
    out.append(f'  last col name: {d.columns[-1]}')

# ---- 附件3 ----
a3 = pd.read_excel(os.path.join(base, '附件3.xlsx'))
out.append(f'[附件3] shape={a3.shape}')
out.append(f'  rows per day expected: 1460, actual={len(a3)}')
out.append(f'  date range: {a3.日期.iloc[0]} -> {a3.日期.iloc[-1]}')
out.append(f'  nulls: {a3.isna().sum().sum()}')
out.append(f'  issue times unique: {sorted(a3.预报时刻.dropna().unique())}')
out.append(f'  any negative: {int((a3.select_dtypes("number")<0).sum().sum())}')

# ---- 附件4 ----
a4 = pd.read_excel(os.path.join(base, '附件4.xlsx'))
out.append(f'[附件4] shape={a4.shape}')
out.append(f'  date range: {a4.iloc[0,0]} -> {a4.iloc[-1,0]}  ndays={len(a4)}')
out.append(f'  nulls total: {int(a4.isna().sum().sum())}')
num = a4.select_dtypes('number')
out.append(f'  price min/max = {num.min().min():.4f}/{num.max().max():.4f}')

# compare 附件1 price with each day of 附件4 (nearest match)
p1 = a1['电价'].values
best = (None, 1e9)
for i in range(len(a4)):
    p4 = a4.iloc[i, 1:].values.astype(float)
    if len(p4) != len(p1):
        continue
    err = np.abs(p4 - p1).sum()
    if err < best[1]:
        best = (a4.iloc[i,0], err)
out.append(f'  附件1 price vs 附件4 closest day: {best[0]} err_sum={best[1]:.3f} (exact iff ~0)')

# ---- result templates ----
for fn in ['result1.xlsx', 'result2.xlsx', 'result3.xlsx', 'result4-2.xlsx', 'result4-3.xlsx']:
    xl = pd.ExcelFile(os.path.join(base, '附件5', fn))
    out.append(f'[{fn}] sheets={xl.sheet_names}')
    for sh in xl.sheet_names:
        d = xl.parse(sh, header=0)
        out.append(f'  {sh}: shape={d.shape}')

# result2 date range & ncols
r2 = pd.read_excel(os.path.join(base, '附件5', 'result2.xlsx'), sheet_name='计划购电量')
out.append(f'[result2/计划购电量] ncols={r2.shape[1]} dates {r2.iloc[0,0]} -> {r2.iloc[-1,0]} ndays={len(r2)}')
out.append(f'  first 3 cols: {list(r2.columns[:3])}')
out.append(f'  last 3 cols: {list(r2.columns[-3:])}')

with open('data_checks.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
