# -*- coding: utf-8 -*-
"""数据加载 + 结果xlsx写入(按附件5模板)"""
import os, json
import numpy as np
import pandas as pd
import openpyxl

ROOT = r'D:\Study\国赛\国赛'
TMPL = os.path.join(ROOT, r'02_赛题数据\附件\附件5')
OUT = os.path.join(ROOT, r'05_论文与结果')
DATA = os.path.join(ROOT, r'03_数据工程\data')

def load_all():
    day = pd.read_parquet(os.path.join(DATA, 'day_slots.parquet'))
    typ = pd.read_parquet(os.path.join(DATA, 'typ_day.parquet'))
    fc0 = pd.read_parquet(os.path.join(DATA, 'fc_slots_plan.parquet'))
    fca = pd.read_parquet(os.path.join(DATA, 'fc_slots_adj.parquet'))
    pers = pd.read_parquet(os.path.join(DATA, 'persistence.parquet'))
    tmpl = json.load(open(os.path.join(DATA, 'template_map.json'), encoding='utf-8'))
    const = json.load(open(os.path.join(DATA, 'constants.json'), encoding='utf-8'))
    return day, typ, fc0, fca, pers, tmpl, const

def dates_of_results():
    r2 = pd.read_excel(os.path.join(TMPL, 'result2.xlsx'), sheet_name='计划购电量', header=0)
    return pd.to_datetime(r2.iloc[:, 0]).tolist()   # 334天: 2025-02-01..2025-12-31

def _clear(wb, sheet, keep_header=True):
    ws = wb[sheet]
    start = 2 if keep_header else 1
    if ws.max_row >= start:
        for row in ws.iter_rows(min_row=start, max_row=ws.max_row):
            for c in row:
                c.value = None
    return ws

def write_result1(dst, P, C, D, soc0, soc24):
    """P/C/D: 长度144 (kWh)"""
    wb = openpyxl.load_workbook(os.path.join(TMPL, 'result1.xlsx'))
    ws = wb['计划购电量']
    for j, v in enumerate(P):
        ws.cell(row=2 + j, column=2, value=round(float(v), 2))
    ws2 = wb['充放电量']
    blocks = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
    for b, (s0, s1) in enumerate(blocks):
        ws2.cell(row=2 + b, column=2, value=round(float(C[s0:s1].sum()), 2))
        ws2.cell(row=2 + b, column=3, value=round(float(D[s0:s1].sum()), 2))
    ws2.cell(row=2, column=5, value=round(float(soc0), 2))
    ws2.cell(row=3, column=5, value=round(float(soc24), 2))
    os.makedirs(OUT, exist_ok=True)
    wb.save(os.path.join(OUT, dst))
    print('written', dst)

def _fill_plan_sheet(ws, dates, M, tot, cost, n=144):
    """M: ndarray (nday, 144) kWh; tot/cost: 每日全天购电量/购电费"""
    for i in range(len(dates)):
        ws.cell(row=2 + i, column=1, value=dates[i])
        for j in range(n):
            ws.cell(row=2 + i, column=2 + j, value=round(float(M[i, j]), 2))
        ws.cell(row=2 + i, column=2 + n, value=round(float(tot[i]), 2))
        ws.cell(row=2 + i, column=3 + n, value=round(float(cost[i]), 2))

def write_result2like(template, dst, dates, M_plan, cost, soc0_arr, soc24_arr, C, D, emerg):
    """通用: result2/result4-2 三表结构. cost=每日计划购电费"""
    write_result2(dst, dates, M_plan, cost, soc0_arr, soc24_arr, C, D, emerg)

def write_result2(dst, dates, P_plan, plan_cost, soc0_arr, soc24_arr, C, D, emerg, template='result2.xlsx'):
    wb = openpyxl.load_workbook(os.path.join(TMPL, template))
    ws = _clear(wb, '计划购电量')
    for i in range(len(dates)):
        ws.cell(row=2 + i, column=1, value=dates[i])
        for j in range(144):
            ws.cell(row=2 + i, column=2 + j, value=round(float(P_plan[i, j]), 2))
        ws.cell(row=2 + i, column=146, value=round(float(P_plan[i].sum()), 2))
        ws.cell(row=2 + i, column=147, value=round(float(plan_cost[i]), 2))
    ws2 = _clear(wb, '充放电量')
    blocks = ['0:00-4:00', '4:00-8:00', '8:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00']
    r = 2
    for i in range(len(dates)):
        for b in range(6):
            ws2.cell(row=r, column=1, value=dates[i] if b == 0 else None)
            ws2.cell(row=r, column=2, value=blocks[b])
            ws2.cell(row=r, column=3, value=round(float(C[i, b * 24:(b + 1) * 24].sum()), 2))
            ws2.cell(row=r, column=4, value=round(float(D[i, b * 24:(b + 1) * 24].sum()), 2))
            ws2.cell(row=r, column=5, value='0:00' if b == 0 else ('24:00' if b == 1 else None))
            ws2.cell(row=r, column=6, value=round(float(soc0_arr[i]), 2) if b == 0 else (round(float(soc24_arr[i]), 2) if b == 1 else None))
            r += 1
    ws3 = _clear(wb, '紧急购电量')
    r = 2
    for i in range(len(dates)):
        segs = emerg[i]
        if len(segs) == 0:
            ws3.cell(row=r, column=1, value=dates[i]); r += 1
        for k, (label, kwh) in enumerate(segs):
            ws3.cell(row=r, column=1, value=dates[i] if k == 0 else None)
            ws3.cell(row=r, column=2, value=label)
            ws3.cell(row=r, column=3, value=round(float(kwh), 2))
            r += 1
    os.makedirs(OUT, exist_ok=True)
    wb.save(os.path.join(OUT, dst))
    print('written', dst)

def write_result3(dst, dates, P0, plan_cost, P_final, tot_cost, soc0_arr, soc24_arr, C, D, emerg, template='result3.xlsx'):
    wb = openpyxl.load_workbook(os.path.join(TMPL, template))
    ws = _clear(wb, '计划购电量')
    for i in range(len(dates)):
        ws.cell(row=2 + i, column=1, value=dates[i])
        for j in range(144):
            ws.cell(row=2 + i, column=2 + j, value=round(float(P0[i, j]), 2))
        ws.cell(row=2 + i, column=146, value=round(float(P0[i].sum()), 2))
        ws.cell(row=2 + i, column=147, value=round(float(plan_cost[i]), 2))
    ws2 = _clear(wb, '调整购电量')
    for i in range(len(dates)):
        ws2.cell(row=2 + i, column=1, value=dates[i])
        for j in range(144):
            ws2.cell(row=2 + i, column=2 + j, value=round(float(P_final[i, j]), 2))
        ws2.cell(row=2 + i, column=146, value=round(float(P_final[i].sum()), 2))
        ws2.cell(row=2 + i, column=147, value=round(float(tot_cost[i]), 2))
    ws3 = _clear(wb, '充放电量')
    blocks = ['0:00-4:00', '4:00-8:00', '8:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00']
    r = 2
    for i in range(len(dates)):
        for b in range(6):
            ws3.cell(row=r, column=1, value=dates[i] if b == 0 else None)
            ws3.cell(row=r, column=2, value=blocks[b])
            ws3.cell(row=r, column=3, value=round(float(C[i, b * 24:(b + 1) * 24].sum()), 2))
            ws3.cell(row=r, column=4, value=round(float(D[i, b * 24:(b + 1) * 24].sum()), 2))
            ws3.cell(row=r, column=5, value='0:00' if b == 0 else ('24:00' if b == 1 else None))
            ws3.cell(row=r, column=6, value=round(float(soc0_arr[i]), 2) if b == 0 else (round(float(soc24_arr[i]), 2) if b == 1 else None))
            r += 1
    ws4 = _clear(wb, '紧急购电量')
    r = 2
    for i in range(len(dates)):
        segs = emerg[i]
        if len(segs) == 0:
            ws4.cell(row=r, column=1, value=dates[i]); r += 1
        for k, (label, kwh) in enumerate(segs):
            ws4.cell(row=r, column=1, value=dates[i] if k == 0 else None)
            ws4.cell(row=r, column=2, value=label)
            ws4.cell(row=r, column=3, value=round(float(kwh), 2))
            r += 1
    os.makedirs(OUT, exist_ok=True)
    wb.save(os.path.join(OUT, dst))
    print('written', dst)

def _plan_cost_row(dt, P_row):
    """计划购电费: 用附件1固定电价曲线(由调用方注入更稳妥, 这里占位)"""
    return float(P_row.sum() * 0)  # 占位, 实际值在q3.py计算后直接传入
