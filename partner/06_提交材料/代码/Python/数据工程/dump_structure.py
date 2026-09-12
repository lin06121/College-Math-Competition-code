# -*- coding: utf-8 -*-
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd

base = r'C题\附件'
files = ['附件1.xlsx', '附件2.xlsx', '附件3.xlsx', '附件4.xlsx',
         r'附件5\result1.xlsx', r'附件5\result2.xlsx', r'附件5\result3.xlsx',
         r'附件5\result4-2.xlsx', r'附件5\result4-3.xlsx']

out = []
for fn in files:
    path = os.path.join(base, fn)
    xl = pd.ExcelFile(path)
    out.append(f'\n########## {fn}  sheets={xl.sheet_names} ##########')
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None, nrows=8)
        out.append(f'--- sheet [{sh}] shape(first8rows)={df.shape}')
        for i, row in df.iterrows():
            vals = [('' if pd.isna(v) else str(v)) for v in row.tolist()]
            out.append(f'  r{i}: ' + ' | '.join(vals))

with open('attachments_structure.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('written', len(out))
