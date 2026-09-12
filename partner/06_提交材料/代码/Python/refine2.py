# -*- coding: utf-8 -*-
"""依据细化收尾:
1) V3 修正: 光伏噪声用正确分位方向 qP∈{0.3,0.4,0.5,0.6} (buf = qL(eL) - qP(eP), qP=0.3等价q70(-eP))
2) 采用 m4 光伏基值 + V0 缓冲, 重出 result2.xlsx / result4-2.xlsx
3) 各星期充放电统计 (证明周内电价规律被自动利用)
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from engine import solve_day, merge_spans, span_label, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, write_result2, dates_of_results, ROOT

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
PV_b = np.zeros_like(PV_act)   # m4: 近4日均值
for i in range(n):
    hist = list(range(max(0, i - 4), i))
    PV_b[i] = PV_act[hist].mean(axis=0) if hist else PV_typ

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

def chain_full(PVb, buf_spec, pmode='typ', rng=None, want_cd=False):
    idx = np.arange(n) if rng is None else np.arange(*rng)
    eL = L_act - L_prev; eP = PV_act - PVb
    soc0 = 6000.0
    P_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    s0 = np.zeros(n); s24 = np.zeros(n); cp = np.zeros(n); E_m = np.zeros((n, 144)); ce = np.zeros(n)
    for i in idx:
        if pmode == 'typ':
            p = price; ph = price
            v = 0.9 * price.mean() if i < n - 1 else 0.0
        else:
            p = PR[i]; ph = PR[i]
            v = 0.9 * PR[i + 1].mean() if i < n - 1 else 0.0
        buf = np.zeros(144)
        lo = max(0, i - 30)
        if i - lo >= 7:
            if buf_spec[0] == 'net':
                buf = np.quantile(eL[lo:i] - eP[lo:i], buf_spec[1], axis=0)
            elif buf_spec[0] == 'sep':
                buf = np.quantile(eL[lo:i], buf_spec[1], axis=0) - np.quantile(eP[lo:i], buf_spec[2], axis=0)
        P0 = solve_day(ph, L_prev[i] + buf, PVb[i], soc0, soc_end=None, v=v)['P']
        E, C, D, s24r, cei = settlement(L_act[i], PV_act[i], P0, soc0, v, p)
        P_m[i] = P0; C_m[i] = C; D_m[i] = D; E_m[i] = E
        cp[i] = float(p @ P0); ce[i] = cei
        s0[i] = soc0; s24[i] = s24r; soc0 = s24r
    out = (P_m, C_m, D_m, s0, s24, cp, E_m, ce)
    return out

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
jan_m = np.where(dates365 >= pd.Timestamp('2025-01-08'))[0][0]

# 1) V3 修正网格
print('===== E3 V3 修正: 光伏噪声分位 qP∈{0.3,0.4,0.5,0.6} =====')
jan_cost = {}
for ql in [0.6, 0.7, 0.8]:
    for qp in [0.3, 0.4, 0.5, 0.6]:
        Mq = chain_full(PV_b, ('sep', ql, qp, 60), rng=(0, 31))
        jan_cost[(ql, qp)] = (Mq[5] + Mq[7])[jan_m:31].sum()
ql, qp = min(jan_cost, key=jan_cost.get)
Mv3 = chain_full(PV_b, ('sep', ql, qp, 60))
tot_v3 = float((Mv3[5] + Mv3[7])[m:].sum())
print(f'[E3 V3修正] 最优(qL,qP)=({ql},{qp}), 全年={tot_v3:,.0f} 元 (对比V0=13,503,095)')

# 2) 采用 m4 + V0(net/30, q=0.7) 重出结果
t0 = time.time()
M2 = chain_full(PV_b, ('net', 0.7, 0.0, 30), pmode='typ')
tot2 = float((M2[5] + M2[7])[m:].sum())
M4 = chain_full(PV_b, ('net', 0.7, 0.0, 30), pmode='q4', want_cd=True)
tot4 = float((M4[5] + M4[7])[m:].sum())
print(f'[采用] m4+V0: Q2全年={tot2:,.0f} 元 (原m3=13,512,561); Q4-2全年={tot4:,.0f} 元 (原=14,196,919); 耗时={time.time()-t0:.1f}s')

res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in dates365])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
for M_, dst, tmpl in [(M2, 'result2.xlsx', 'result2.xlsx'), (M4, 'result4-2.xlsx', 'result4-2.xlsx')]:
    emerg = []
    for i in range(len(res_dates)):
        idx = np.where(M_[6][i0 + i] > 1e-6)[0] + 1
        emerg.append([(span_label(a, b), float(M_[6][i0 + i][a - 1:b].sum())) for a, b in merge_spans(idx)])
    write_result2(dst, res_dates, M_[0][i0:i1], M_[5][i0:i1], M_[3][i0:i1], M_[4][i0:i1],
                  M_[1][i0:i1], M_[2][i0:i1], emerg, template=tmpl)

# 3) 各星期充放电 (Q4-2 采用方案)
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
print('===== Q4-2 各星期日均充/放电量 (kWh) =====')
for k in range(7):
    ck = M4[1][dow == k].sum(axis=1).mean(); dk = M4[2][dow == k].sum(axis=1).mean()
    print(f'  {names[k]}: 充电={ck:,.0f} 放电={dk:,.0f} (净充={ck-dk:+,.0f})')

with open(os.path.join(DATA, 'basis_refine.json'), 'w', encoding='utf-8') as f:
    old = {}
    try:
        old = json.load(open(os.path.join(DATA, 'basis_refine.json'), encoding='utf-8'))
    except Exception:
        pass
    old.update(dict(v3_corrected=dict(qL=ql, qP=qp, total=tot_v3), adopt=dict(pv_basis='m4',
                   tot_q2=tot2, tot_q4_2=tot4)))
    json.dump(old, f, ensure_ascii=False, indent=1)
print('REFINE2 DONE')
