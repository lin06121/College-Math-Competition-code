# -*- coding: utf-8 -*-
"""依据细化实验:
E1) 光伏基值窗口: m2/m3/m4/m5/m7/m10/加权w3/EMA  (MAE + 全年链条成本, q*按1月调优)
E2) 电价周内规律(已验证): 周五六低23.3% -> 已由Q4已知电价假设自动利用(证明: 各星期充放电量)
    + 稳健性: "0:00仅知周内电价规律"变体 vs 已知电价
E3) 噪声单独建模(参考: 分时Stacking光伏预测思想):
    V0 净残差q0.7/30天(现行)  V1 净残差/60天  V2 净残差/90天
    V3 负荷噪声与光伏噪声分开设缓冲(qL,qP在1月网格调优, 60天窗)
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

DATA = os.path.join(ROOT, r'04_建模求解\data')
EPS = 1e-4
day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
PR = day.pivot(index='date', columns='slot', values='price_q4').sort_index().to_numpy()
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365])
n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
L_prev = np.zeros_like(L_act)
for i in range(n):
    low_set = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low_set) == (dow[i] in low_set)]
    L_prev[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ

def pv_basis(kind):
    B = np.zeros_like(PV_act)
    for i in range(n):
        if kind == 'ema':
            if i == 0:
                B[i] = PV_typ; continue
            alpha = 0.4
            B[i] = alpha * PV_act[i - 1] + (1 - alpha) * (B[i - 1] if i > 1 else PV_typ)
        elif kind == 'w3':
            if i >= 3:
                w = np.array([0.5, 0.3, 0.2])
                B[i] = (PV_act[i - 3:i] * w[:, None]).sum(axis=0)
            elif i > 0:
                B[i] = PV_act[:i].mean(axis=0)
            else:
                B[i] = PV_typ
        else:
            w = int(kind[1:])
            hist = list(range(max(0, i - w), i))
            B[i] = PV_act[hist].mean(axis=0) if hist else PV_typ
    return B

def settlement(L, PV, P0, soc0, v, p):
    h = 144
    iC, iD, iE, iS = 0, h, 2 * h, 3 * h
    N = 4 * h + (h + 1)
    c = np.zeros(N); c[iE:iE + h] = 5.0 * p; c[iC:iC + h] = EPS; c[iD:iD + h] = EPS; c[iS + h] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        row = np.zeros(N); row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        row = np.zeros(N); row[iS + t] = -1; row[iS + t + 1] = 1
        row[iC + t] = -ETA_C; row[iD + t] = 1.0 / ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iS:iS + h + 1] = SOC_MIN; ub[iS:iS + h + 1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success
    return (res.x[iE:iE + h], res.x[iC:iC + h], res.x[iD:iD + h], res.x[iS + h], float(5.0 * p @ res.x[iE:iE + h]))

def chain(PB, buf_spec, rng=None, price_mode='typ', price_hat=None, want_cd=False):
    """buf_spec: (mode, qL, qP, window) mode in {'net','sep','none'}; price_mode: 'typ'|'act'|'hat'"""
    idx = np.arange(n) if rng is None else np.arange(*rng)
    eL = L_act - L_prev
    eP = PV_act - PB
    soc0 = 6000.0
    cp = np.zeros(n); ce = np.zeros(n)
    C_d = np.zeros((n, 144)); D_d = np.zeros((n, 144))
    for i in idx:
        p = price if price_mode != 'act' else PR[i]
        ph = price_hat[i] if price_mode == 'hat' else p
        v = 0.9 * p.mean() if i < n - 1 else 0.0
        buf = np.zeros(144)
        if buf_spec[0] == 'net':
            lo = max(0, i - buf_spec[3])
            if i - lo >= 7:
                buf = np.quantile(eL[lo:i] - eP[lo:i], buf_spec[1], axis=0)
        elif buf_spec[0] == 'sep':
            lo = max(0, i - buf_spec[3])
            if i - lo >= 7:
                buf = np.quantile(eL[lo:i], buf_spec[1], axis=0) - np.quantile(eP[lo:i], buf_spec[2], axis=0)
        P0 = solve_day(ph, L_prev[i] + buf, PB[i], soc0, soc_end=None, v=v)['P']
        E, C, D, s24r, cei = settlement(L_act[i], PV_act[i], P0, soc0, v, p)
        cp[i] = float(p @ P0); ce[i] = cei
        C_d[i] = C; D_d[i] = D
        soc0 = s24r
    if want_cd:
        return cp, ce, C_d, D_d
    return cp, ce

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
jan_m = np.where(dates365 >= pd.Timestamp('2025-01-08'))[0][0]

def tune_and_eval(PB, buf_spec, label):
    """1月调q -> 全年成本"""
    jan_cost = {}
    if buf_spec[0] == 'sep':
        grid = [(ql, qp) for ql in [0.5, 0.6, 0.7, 0.8] for qp in [0.5, 0.6, 0.7, 0.8]]
        for (ql, qp) in grid:
            cp, ce = chain(PB, ('sep', ql, qp, buf_spec[3]), rng=(0, 31))
            jan_cost[(ql, qp)] = (cp + ce)[jan_m:31].sum()
        qL, qP = min(jan_cost, key=jan_cost.get)
    else:
        for ql in [0.5, 0.6, 0.7, 0.8]:
            cp, ce = chain(PB, ('net', ql, 0.0, buf_spec[3]), rng=(0, 31))
            jan_cost[ql] = (cp + ce)[jan_m:31].sum()
        qL, qP = min(jan_cost, key=jan_cost.get), 0.0
    cp, ce = chain(PB, (buf_spec[0], qL, qP, buf_spec[3]))
    tot = float((cp + ce)[m:].sum())
    print(f'[{label}] q=(qL={qL},qP={qP}) 1月调优, 2.1-12.31总费={tot:,.0f} 元 (紧急费={ce[m:].sum():,.0f})')
    return tot, (qL, qP)

t0 = time.time()
print('===== E1: 光伏基值窗口 =====')
pb = {k: pv_basis(k) for k in ['m2', 'm3', 'm4', 'm5', 'm7', 'm10', 'w3', 'ema']}
for k, B in pb.items():
    mae = np.abs(PV_act - B).mean()
    print(f'  {k}: 全年MAE={mae:.2f} kWh/时段')
best_e1 = {}
for k in ['m3', 'm4', 'm5', 'm7', 'w3', 'ema']:
    best_e1[k] = tune_and_eval(pb[k], ('net', 0.7, 0.0, 30), f'E1 {k}')
k_pv = min(best_e1, key=lambda k: best_e1[k][0])
print(f'[E1 结论] 最优光伏基值 = {k_pv} ({best_e1[k_pv][0]:,.0f} 元)')
PB_star = pb[k_pv]

print('===== E3: 噪声单独建模 =====')
e3 = {}
e3['V0 net/30'] = tune_and_eval(PB_star, ('net', 0.7, 0.0, 30), 'E3 V0 net/30')
e3['V1 net/60'] = tune_and_eval(PB_star, ('net', 0.7, 0.0, 60), 'E3 V1 net/60')
e3['V2 net/90'] = tune_and_eval(PB_star, ('net', 0.7, 0.0, 90), 'E3 V2 net/90')
e3['V3 sep/60'] = tune_and_eval(PB_star, ('sep', 0.7, 0.7, 60), 'E3 V3 sep/60')
best_e3 = min(e3, key=lambda k: e3[k][0])
print(f'[E3 结论] 最优缓冲方案 = {best_e3} ({e3[best_e3][0]:,.0f} 元)')

print('===== E2b: Q4 电价预测变体 (0:00仅知周内电价规律) =====')
# 电价预测: 同类日(周五六/其他)近4个平均
PR_hat = np.zeros_like(PR)
for i in range(n):
    low_set = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low_set) == (dow[i] in low_set)]
    PR_hat[i] = PR[hist[-4:]].mean(axis=0) if hist else PR[0]
def chain_q4(PB, buf_spec, pmode, want_cd=False):
    idx = np.arange(n)
    eL = L_act - L_prev; eP = PV_act - PB
    soc0 = 6000.0; cp = np.zeros(n); ce = np.zeros(n)
    C_d = np.zeros((n, 144)); D_d = np.zeros((n, 144))
    for i in idx:
        p = PR[i]
        ph = PR[i] if pmode == 'known' else PR_hat[i]
        v = 0.9 * PR[i + 1].mean() if i < n - 1 else 0.0
        buf = np.zeros(144)
        lo = max(0, i - 30)
        if i - lo >= 7 and buf_spec[0] == 'net':
            buf = np.quantile(eL[lo:i] - eP[lo:i], buf_spec[1], axis=0)
        P0 = solve_day(ph, L_prev[i] + buf, PB[i], soc0, soc_end=None, v=v)['P']
        E, C, D, s24r, cei = settlement(L_act[i], PV_act[i], P0, soc0, v, p)
        cp[i] = float(p @ P0); ce[i] = cei
        C_d[i] = C; D_d[i] = D
        soc0 = s24r
    if want_cd:
        return cp, ce, C_d, D_d
    return cp, ce
for pmode in ['known', 'hat']:
    jan_cost = {}
    for ql in [0.5, 0.6, 0.7, 0.8]:
        cp, ce = chain_q4(PB_star, ('net', ql, 0.0, 30), pmode)
        cpj, cej = cp, ce
        jan_cost[ql] = (cpj + cej)[jan_m:31].sum()
    qs = min(jan_cost, key=jan_cost.get)
    cp, ce = chain_q4(PB_star, ('net', qs, 0.0, 30), pmode)
    print(f'  Q4-2 {pmode} (q*={qs}): 总费={(cp+ce)[m:].sum():,.0f} 元 (计划费={cp[m:].sum():,.0f}, 紧急费={ce[m:].sum():,.0f})')

# 各星期充放电量 (已知电价方案, 用于证明周内规律被自动利用)
cp, ce, C_d, D_d = chain_q4(PB_star, ('net', 0.7, 0.0, 30), 'known', want_cd=True)
print('===== 已知电价方案: 各星期日均充/放电量 (kWh) =====')
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
for k in range(7):
    print(f'  {names[k]}: 充电={C_d[dow == k].mean(axis=0).sum():.0f} 放电={D_d[dow == k].mean(axis=0).sum():.0f}')

print(f'[总耗时] {time.time()-t0:.1f}s')
with open(os.path.join(DATA, 'basis_refine.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(e1={k: [v[0], list(map(float, v[1]))] for k, v in best_e1.items()}, k_pv=k_pv,
                   e3={k: [v[0], list(map(float, v[1]))] for k, v in e3.items()}, best_e3=best_e3), f, ensure_ascii=False, indent=1)
print('BASIS REFINE DONE')
