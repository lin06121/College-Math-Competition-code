# -*- coding: utf-8 -*-
"""因果口径公共核心: 数据装载 + 实时平衡控制结算 + 计划链(问题二/三/四)
   计划模型(0:00 只用历史信息) 与 结算模型(日内实时平衡控制) 分开建模, 两层独立。
   供 rebuild_causal.py / causal_*.py 复用。
"""
import os
import numpy as np
import pandas as pd
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

__all__ = ['D', 'price', 'PR', 'balance', 'buffer_of', 'run_q2', 'run_q3', 'clairvoyant',
           'P_MAX', 'SOC_MIN', 'SOC_MAX', 'ETA_C', 'ETA_D', 'ROOT']


class _Data:
    pass


D = _Data()
day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
D.price = price
D.L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
D.PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
D.PR = day.pivot(index='date', columns='slot', values='price_q4').sort_index().to_numpy()
D.dates = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
D.dow = np.array([d.dayofweek for d in D.dates])
D.n = len(D.dates)
D.L_typ = typ['load_typ'].to_numpy() / 6.0
D.PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
D.fc0m = fc0.pivot(index='date', columns='slot', values='fc0_kw').sort_index().to_numpy() / 6.0
D.fca_m = {h: fca[fca.issue_hour == h].pivot(index='date', columns='slot', values='fc_kw')
           .sort_index().reindex(columns=range(1, 145)).to_numpy() / 6.0 for h in (6, 12, 18)}
PR = D.PR
n = D.n

D.Lb = np.zeros_like(D.L_act); D.Pb = np.zeros_like(D.PV_act)
for i in range(n):
    low = {4, 5}
    hist = [j for j in range(i) if (D.dow[j] in low) == (D.dow[i] in low)]
    D.Lb[i] = D.L_act[hist[-4:]].mean(axis=0) if hist else D.L_typ
    h2 = list(range(max(0, i - 4), i))
    D.Pb[i] = D.PV_act[h2].mean(axis=0) if h2 else D.PV_typ
D.eNet = (D.L_act - D.Lb) - (D.PV_act - D.Pb)
D.m2 = np.where(D.dates >= pd.Timestamp('2025-02-01'))[0][0]


def buffer_of(i, q):
    """第 i 天的报童安全裕度: 只用近 30 天已实现的净需求预测残差"""
    lo = max(0, i - 30)
    if i - lo < 7 or q <= 0:
        return np.zeros(144)
    return np.quantile(D.eNet[lo:i], q, axis=0)


def balance(P0, Cpl, Dpl, soc0, L, PV, reserve=SOC_MIN):
    """实时平衡控制结算(因果): ①裁剪计划充放为物理可行 ②缺额先减充再增放 ③富余先减放再增充
       兜不住的缺口 = 5 倍电价紧急购电; 兜不住的富余 = 弃光"""
    h = len(P0)
    C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        c_ok = min(max(float(Cpl[t]), 0.0), P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C))
        d_ok = min(max(float(Dpl[t]), 0.0), P_MAX, ETA_D * max(0.0, soc - reserve + ETA_C * c_ok))
        C[t], D[t] = c_ok, d_ok
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        if gap > 1e-9:
            cut = min(C[t], gap); C[t] -= cut; gap -= cut
            if gap > 1e-9:
                avail = ETA_D * max(0.0, soc - reserve + ETA_C * C[t]) - D[t]
                extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, avail))
                D[t] += extra; gap -= extra
        elif gap < -1e-9:
            cut = min(D[t], -gap); D[t] -= cut; gap += cut
            if gap < -1e-9:
                room = max(0.0, (SOC_MAX - soc + D[t] / ETA_D)) / ETA_C - C[t]
                extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, room))
                C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
        assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6, ('SOC越界', t, soc)
        assert C[t] <= P_MAX + 1e-9 and D[t] <= P_MAX + 1e-9, ('功率越界', t)
        assert min(C[t], D[t]) <= 1e-9, ('同时充放', t)
    return C, D, soc, Em


def perfect(L_act, PV, P0, soc0, v, p):
    """完美信息再调度(非因果参照): 锁定同一计划 P0, 用当日实测最优调度储能
       若 plan=实测则等于联合最优; 这里作为"给定计划的应急费下界"使用"""
    from scipy.optimize import linprog
    h = 144
    iC, iD, iE, iS = 0, h, 2 * h, 3 * h
    N = 4 * h + (h + 1)
    c = np.zeros(N); c[iE:iE + h] = 5.0 * p; c[iC:iC + h] = 1e-4; c[iD:iD + h] = 1e-4; c[iS + h] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        row = np.zeros(N); row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L_act[t])
    for t in range(h):
        row = np.zeros(N); row[iS + t] = -1; row[iS + t + 1] = 1; row[iC + t] = -ETA_C; row[iD + t] = 1 / ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iS:iS + h + 1] = SOC_MIN; ub[iS:iS + h + 1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    r = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                bounds=list(zip(lb, ub)), method='highs')
    assert r.success, r.message
    return r.x[iE:iE + h], float(r.x[iS + h]), float(5 * p @ r.x[iE:iE + h])


def run_q2(p_matrix=None, q=0.7, reserve=1200.0, upto=None, mode='causal', bill_matrix=None):
    """问题二: 0:00 计划 + 日内结算
       p_matrix: 0:00 计划阶段使用的电价预期; bill_matrix: 实际结算电价(默认同 p_matrix)
       mode='causal' 实时平衡控制 | 'asplanned' 僵硬执行计划(偏差全部紧急/弃光) | 'perfect' 完美信息再调度"""
    upto = n if upto is None else upto
    soc = 6000.0
    P_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144)); E_m = np.zeros((n, 144))
    s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = price if p_matrix is None else p_matrix[i]
        bill = p if bill_matrix is None else bill_matrix[i]
        v = (0.9 * (price.mean() if p_matrix is None else p_matrix[min(i + 1, n - 1)].mean())) if i < n - 1 else 0.0
        buf = buffer_of(i, q)
        r = solve_day(p, D.Lb[i] + buf, D.Pb[i], soc, soc_end=None, v=v)
        if mode == 'causal':
            C, Dd, sEnd, Em = balance(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i], reserve)
        elif mode == 'asplanned':
            C, Dd = r['C'].copy(), r['D'].copy()
            Em = np.maximum(0.0, D.L_act[i] - (D.PV_act[i] + r['P'] + Dd - C))
            sEnd = float(soc + ETA_C * C.sum() - Dd.sum() / ETA_D)
        elif mode == 'perfect':
            Em, sEnd, _ = perfect(L_act=D.L_act[i], PV=D.PV_act[i], P0=r['P'], soc0=soc, v=v, p=p)
            C, Dd = r['C'], r['D']
        P_m[i] = r['P']; C_m[i] = C; D_m[i] = Dd; E_m[i] = Em
        pc[i] = float(bill @ r['P']); ec[i] = float(5 * bill @ Em)
        s0[i] = soc; s24[i] = sEnd; soc = sEnd
        if i + 1 >= upto:
            break
    return dict(P=P_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, ec=ec,
                total=float((pc + ec)[D.m2:upto].sum()), plan=float(pc[D.m2:upto].sum()),
                em=float(ec[D.m2:upto].sum()))


def run_q3(p_matrix=None, q=0.7, reserve=1200.0, adjust=(6, 12, 18), upto=None):
    """问题三: 0:00 计划 + 滚动调整(仅用已发布预报) + 日内实时平衡结算"""
    upto = n if upto is None else upto
    soc = 6000.0; H = [36, 72, 108]; fh = [6, 12, 18]
    P0_m = np.zeros((n, 144)); Pf_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    E_m = np.zeros((n, 144)); s0 = np.zeros(n); s24 = np.zeros(n)
    pc = np.zeros(n); dv = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        pday = price if p_matrix is None else p_matrix[i]
        v = (0.9 * (price.mean() if p_matrix is None else p_matrix[min(i + 1, n - 1)].mean())) if i < n - 1 else 0.0
        buf = buffer_of(i, q); Lbar = D.Lb[i] + buf
        r0 = solve_day(pday, Lbar, D.fc0m[i], soc, soc_end=None, v=v)
        P, C, Dd, SOC = r0['P'], r0['C'], r0['D'], r0['E']
        Pinit = P.copy()
        for h0, hh in zip(H, fh):
            if hh not in adjust:
                continue
            fc = D.fca_m[hh][i, h0:].copy()
            r = solve_day(pday[h0:], Lbar[h0:], fc, SOC[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], Dd[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0 + 1], r['E'][1:]])
        C2, D2, sEnd, Em = balance(P, C, Dd, soc, D.L_act[i], D.PV_act[i], reserve)
        P0_m[i] = Pinit; Pf_m[i] = P; C_m[i] = C2; D_m[i] = D2; E_m[i] = Em
        pc[i] = float(pday @ Pinit)
        dv[i] = float(np.sum(-0.5 * pday * np.maximum(Pinit - P, 0) + 1.5 * pday * np.maximum(P - Pinit, 0)))
        ec[i] = float(5 * pday @ Em)
        s0[i] = soc; s24[i] = sEnd; soc = sEnd
        if i + 1 >= upto:
            break
    return dict(P0=P0_m, Pf=Pf_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, dv=dv, ec=ec,
                plan=float(pc[D.m2:upto].sum()), dev=float(dv[D.m2:upto].sum()),
                em=float(ec[D.m2:upto].sum()), total=float((pc + dv + ec)[D.m2:upto].sum()))


def clairvoyant(p_matrix=None, upto=None):
    """天眼链式下界: 允许用当日实测制定计划(不可达), 储能链一致"""
    upto = n if upto is None else upto
    soc = 6000.0; tot = 0.0
    for i in range(upto):
        p = price if p_matrix is None else p_matrix[i]
        v = (0.9 * (price.mean() if p_matrix is None else p_matrix[min(i + 1, n - 1)].mean())) if i < n - 1 else 0.0
        r = solve_day(p, D.L_act[i], D.PV_act[i], soc, soc_end=None, v=v)
        if i >= D.m2:
            tot += float(p @ r['P'])
        soc = r['E'][-1]
    return tot


def nostorage(q=0.7, p_matrix=None, upto=None):
    """无储能对照: P0 = max(净需求计划,0), 缺口 = 5 倍紧急"""
    upto = n if upto is None else upto
    tot = 0.0
    for i in range(upto):
        p = price if p_matrix is None else p_matrix[i]
        buf = buffer_of(i, q)
        P0 = np.maximum(D.Lb[i] + buf - D.Pb[i], 0.0)
        Em = np.maximum(D.L_act[i] - D.PV_act[i] - P0, 0.0)
        if i >= D.m2:
            tot += float(p @ P0) + float(5 * p @ Em)
    return tot
