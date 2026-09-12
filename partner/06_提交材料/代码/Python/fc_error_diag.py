# -*- coding: utf-8 -*-
"""预测误差结构诊断: 找出"哪些天/哪类天"贡献了大部分净需求误差 -> 针对性改进"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from causal_core import D, ROOT
import fc_upgraded as FU

n, H = D.n, 144
m2 = D.m2
L = D.L_act; PV = D.PV_act          # kWh/时段
Lh, _ = FU.build_load()
PVh = FU.build_pv()
eL = L - Lh; eP = PV - PVh; eN = eL - eP
dates = D.dates

dL = np.abs(eL).mean(axis=1) * 6      # 日 MAE (kW)
dP = np.abs(eP).mean(axis=1) * 6
dN = np.abs(eN).mean(axis=1) * 6
print('=== 总体(2.1-12.31) ===')
print(f'  负载 MAE {dL[m2:].mean():.2f} | 光伏 {dP[m2:].mean():.2f} | 净需求 {dN[m2:].mean():.2f} kW')
print('\n=== 按月 ===')
mon = np.array([d.month for d in dates])
for mm in range(2, 13):
    sel = (mon == mm); sel[:m2] = False
    print(f'  {mm:2d}月: 负载 {dL[sel].mean():6.1f} | 光伏 {dP[sel].mean():6.1f} | 净需求 {dN[sel].mean():6.1f} kW'
          f'  (最大日 {dN[sel].max():7.1f})')
print('\n=== 净需求误差最大的 12 天 ===')
order = np.argsort(dN[m2:])[::-1][:12] + m2
for i in order:
    print(f'  {dates[i].date()} ({["一","二","三","四","五","六","日"][dates[i].dayofweek]}): '
          f'净需求 MAE {dN[i]:7.1f} kW | 负载 {dL[i]:6.1f} | 光伏 {dP[i]:6.1f} | '
          f'实际/预测日电量 负载 {L[i].sum()/1e3:6.1f}/{Lh[i].sum()/1e3:6.1f} MWh, '
          f'光伏 {PV[i].sum()/1e3:5.1f}/{PVh[i].sum()/1e3:5.1f} MWh')
tot = dN[m2:].sum()
top10 = np.sort(dN[m2:])[::-1][:int(0.1 * (n - m2))].sum()
print(f'\n  误差最大的 10% 天数({int(0.1*(n-m2))} 天)贡献了净需求误差总量的 {top10/tot*100:.1f}%')
print(f'  误差最大的 20% 天贡献 {np.sort(dN[m2:])[::-1][:int(0.2*(n-m2))].sum()/tot*100:.1f}%')
# 节假日效应: 春节 2025-01-28~02-04, 清明 4-4~4-6, 劳动 5-1~5-5, 端午 5-31~6-2, 国庆中秋 10-1~10-8
hol = [('2025-01-28', '2025-02-04', '春节'), ('2025-04-04', '2025-04-06', '清明'),
       ('2025-05-01', '2025-05-05', '劳动节'), ('2025-05-31', '2025-06-02', '端午'),
       ('2025-10-01', '2025-10-08', '国庆中秋')]
print('\n=== 节假日 vs 平日(净需求日 MAE) ===')
mask_hol = np.zeros(n, bool)
for a, b, nm in hol:
    sel = (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    mask_hol |= sel
    s = sel.copy(); s[:m2] = False
    if s.sum():
        print(f'  {nm:6s} {a}~{b}: {s.sum():2d} 天, 净需求 MAE {dN[s].mean():7.1f} kW (负载 {dL[s].mean():6.1f})')
s = (~mask_hol).copy(); s[:m2] = False
print(f'  其余平日: {s.sum()} 天, 净需求 MAE {dN[s].mean():7.1f} kW (负载 {dL[s].mean():6.1f})')
json.dump(dict(mae=dict(load=float(dL[m2:].mean()), pv=float(dP[m2:].mean()), net=float(dN[m2:].mean())),
               top10_share=float(top10 / tot), hol_days=int(mask_hol[m2:].sum())),
          open(os.path.join(ROOT, r'04_建模求解\data\fc_error_diag.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
