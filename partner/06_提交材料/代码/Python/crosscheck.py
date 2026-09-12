# -*- coding: utf-8 -*-
"""独立复核: 对偶单纯形(highs-ds) vs 内点法(highs-ipm) 对比 + 导出 MATLAB 校验数据"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from engine import P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

DATA = os.path.join(ROOT, r'04_建模求解\data')
OUT = os.path.join(ROOT, r'04_建模求解\data\matlab_verify')
os.makedirs(OUT, exist_ok=True)

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365])
n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
Lb = np.zeros_like(L_act)
for i in range(n):
    low_set = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low_set) == (dow[i] in low_set)]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
Pb = np.zeros_like(PV_act)
for i in range(n):
    hist = list(range(max(0, i - 4), i))
    Pb[i] = PV_act[hist].mean(axis=0) if hist else PV_typ
eN = (L_act - Lb) - (PV_act - Pb)
buf = np.zeros((n, 144))
for i in range(n):
    lo = max(0, i - 30)
    if i - lo >= 7:
        buf[i] = np.quantile(eN[lo:i], 0.7, axis=0)

def solve(price_v, load_v, pv_v, soc0, soc_end=None, v=0.0, method='highs'):
    h = len(price_v)
    iC, iD, iE = h, 2 * h, 3 * h
    N = 4 * h + 1
    c = np.zeros(N); c[:h] = price_v; c[iE + h] = -v
    Aub, bub = [], []
    for t in range(h):
        row = np.zeros(N); row[t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(pv_v[t] - load_v[t])
    Aeq, beq = [], []
    for t in range(h):
        row = np.zeros(N); row[iE + t], row[iE + t + 1] = -1, 1
        row[iC + t], row[iD + t] = -ETA_C, 1.0 / ETA_D
        Aeq.append(row); beq.append(0.0)
    if soc_end is not None:
        row = np.zeros(N); row[iE + h] = 1
        Aeq.append(row); beq.append(soc_end)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iE:iE + h + 1] = SOC_MIN; ub[iE:iE + h + 1] = SOC_MAX
    lb[iE], ub[iE] = soc0, soc0
    r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                bounds=list(zip(lb, ub)), method=method)
    assert r.success, r.message
    return r

print('=== 1) 问题一: 两种算法对比 ===')
r_ds = solve(price, L_typ, PV_typ, 6000.0, soc_end=6000.0, method='highs-ds')
r_ipm = solve(price, L_typ, PV_typ, 6000.0, soc_end=6000.0, method='highs-ipm')
print(f'  对偶单纯形 购电费 = {r_ds.fun:,.4f} 元')
print(f'  内点法     购电费 = {r_ipm.fun:,.4f} 元')
print(f'  相对偏差 = {abs(r_ds.fun-r_ipm.fun)/r_ds.fun*100:.2e} %  (一致即互证最优)')
P_ds = r_ds.x[:144]; P_ipm = r_ipm.x[:144]
print(f'  购电量最大差异 = {np.abs(P_ds-P_ipm).max():.6f} kWh')

print('=== 2) 问题二前 7 天(2.1-2.7): 两种算法对比 ===')
feb = np.where((dates365 >= pd.Timestamp('2025-02-01')) & (dates365 <= pd.Timestamp('2025-02-07')))[0]
soc = 6000.0
rows = []
for i in feb:
    v = 0.9 * price.mean()
    a = solve(price, Lb[i] + buf[i], Pb[i], soc, v=v, method='highs-ds')
    b = solve(price, Lb[i] + buf[i], Pb[i], soc, v=v, method='highs-ipm')
    rows.append((str(dates365[i].date()), a.fun, b.fun, abs(a.fun - b.fun) / a.fun * 100))
    soc = a.x[3 * 144 + 144]
for r in rows:
    print(f'  {r[0]}: ds={r[1]:,.4f}  ipm={r[2]:,.4f}  偏差={r[3]:.2e}%')

# 导出 MATLAB 校验数据
np.savetxt(os.path.join(OUT, 'q1_price.csv'), price, delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(OUT, 'q1_load.csv'), L_typ, delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(OUT, 'q1_pv.csv'), PV_typ, delimiter=',', fmt='%.6f')
m2_load = np.vstack([Lb[i] + buf[i] for i in feb])
m2_pv = np.vstack([Pb[i] for i in feb])
m2_act_L = np.vstack([L_act[i] for i in feb])
m2_act_PV = np.vstack([PV_act[i] for i in feb])
np.savetxt(os.path.join(OUT, 'q2_load_basis.csv'), m2_load, delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(OUT, 'q2_pv_basis.csv'), m2_pv, delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(OUT, 'q2_load_act.csv'), m2_act_L, delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(OUT, 'q2_pv_act.csv'), m2_act_PV, delimiter=',', fmt='%.6f')
with open(os.path.join(OUT, 'reference_results.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(q1_cost_ds=float(r_ds.fun), q1_cost_ipm=float(r_ipm.fun),
                   q1_plan_kwh=float(P_ds.sum()),
                   q2_feb1_7_ds=[float(x[1]) for x in rows],
                   q2_feb1_7_ipm=[float(x[2]) for x in rows]), f, ensure_ascii=False, indent=1)
print('导出 MATLAB 校验数据 ->', OUT)
