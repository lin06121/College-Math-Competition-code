# -*- coding: utf-8 -*-
"""升级版预测模型口径的公共核心: 预测构造 + 计划/滚动/结算管道 + 缓冲 + 指标
   (与 07_预测模型实验 的结论一致; 预测构造见 fc_upgraded.py)
"""
import numpy as np
from engine import solve_day
from causal_core import D, price, PR, balance

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT, RTP = D.L_act, D.PV_act, D.PR
FCADJ = D.fca_m
LOW = np.array([d in {4, 5} for d in D.dow])


def buffer_mat(Lh, PVh, q, W=30):
    """近 W 天净需求残差的逐时段经验分位(只用历史)"""
    e = (L_ACT - Lh) - (PV_ACT - PVh)
    B = np.zeros((n, H))
    for i in range(n):
        lo = max(0, i - W)
        if i - lo >= 7 and q > 0:
            B[i] = np.quantile(e[lo:i], q, axis=0)
    return B


def run_q2(Lh, PVh, q, pmat=None, upto=None, buffer_mat_in=None, asplanned=False, perfect=False):
    """0:00 计划(注入预测) + 日内结算
       pmat=None -> 附件1 固定电价; asplanned=True -> 僵硬执行; perfect=True -> 完美信息再调度"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    B = buffer_mat(Lh, PVh, q) if buffer_mat_in is None else buffer_mat_in
    soc = 6000.0
    P = np.zeros((n, H)); C = np.zeros((n, H)); Dm = np.zeros((n, H)); Em = np.zeros((n, H))
    s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        r = solve_day(p, Lh[i] + B[i], PVh[i], soc, soc_end=None, v=v)
        if perfect:
            from causal_core import perfect as _perfect
            e, sEnd, _ = _perfect(L_ACT[i], PV_ACT[i], r['P'], soc, v, p)
            c, d = r['C'], r['D']
        elif asplanned:
            c, d = r['C'].copy(), r['D'].copy()
            e = np.maximum(0.0, L_ACT[i] - (PV_ACT[i] + r['P'] + d - c))
            sEnd = float(soc + 0.9 * c.sum() - d.sum() / 0.9)
        else:
            c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        P[i], C[i], Dm[i], Em[i] = r['P'], c, d, e
        pc[i] = float(p @ r['P']); ec[i] = float(5 * p @ e)
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P=P, C=C, D=Dm, Em=Em, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(Em[m2:upto].sum()))


def run_q3(Lh, PVh, q, pmat=None, adjust=(6, 12, 18), upto=None):
    """0:00 计划 + 滚动调整(附件3 更新预报) + 实时平衡"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    B = buffer_mat(Lh, PVh, q)
    soc = 6000.0; HH = [36, 72, 108]; fh = [6, 12, 18]
    P0 = np.zeros((n, H)); Pf = np.zeros((n, H)); C = np.zeros((n, H)); Dm = np.zeros((n, H))
    Em = np.zeros((n, H)); s0 = np.zeros(n); s24 = np.zeros(n)
    pc = np.zeros(n); dv = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        Lbar = Lh[i] + B[i]
        r0 = solve_day(p, Lbar, PVh[i], soc, soc_end=None, v=v)
        P, Cc, Dd, SOC = r0['P'], r0['C'], r0['D'], r0['E']
        Pinit = P.copy()
        for h0, hh in zip(HH, fh):
            if hh not in adjust:
                continue
            r = solve_day(p[h0:], Lbar[h0:], FCADJ[hh][i, h0:], SOC[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], Cc[h0:], Dd[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0 + 1], r['E'][1:]])
        c, d, sEnd, e = balance(P, Cc, Dd, soc, L_ACT[i], PV_ACT[i])
        P0[i], Pf[i], C[i], Dm[i], Em[i] = Pinit, P, c, d, e
        pc[i] = float(p @ Pinit)
        dv[i] = float(np.sum(-0.5 * p * np.maximum(Pinit - P, 0) + 1.5 * p * np.maximum(P - Pinit, 0)))
        ec[i] = float(5 * p @ e)
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P0=P0, Pf=Pf, C=C, D=Dm, Em=Em, s0=s0, s24=s24, pc=pc, dv=dv, ec=ec,
                plan=float(pc[m2:upto].sum()), dev=float(dv[m2:upto].sum()),
                em=float(ec[m2:upto].sum()), total=float((pc + dv + ec)[m2:upto].sum()),
                em_kwh=float(Em[m2:upto].sum()))


def nostorage(Lh, PVh, q, pmat=None, upto=None):
    """无储能对照: P0 = max(计划净需求,0), 缺口按 5 倍紧急购电"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    B = buffer_mat(Lh, PVh, q)
    tot = 0.0
    for i in range(upto):
        P0 = np.maximum(Lh[i] + B[i] - PVh[i], 0.0)
        Em = np.maximum(L_ACT[i] - PV_ACT[i] - P0, 0.0)
        if i >= m2:
            tot += float(pm[i] @ P0) + float(5 * pm[i] @ Em)
    return tot


def clairvoyant(pmat=None, upto=None):
    """天眼链式下界(用当日实测制定计划, 不可达)"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    soc = 6000.0; tot = 0.0
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        r = solve_day(p, L_ACT[i], PV_ACT[i], soc, soc_end=None, v=v)
        if i >= m2:
            tot += float(p @ r['P'])
        soc = r['E'][-1]
    return tot


def metrics(A_hat, A_act, scale=6.0):
    d = A_hat[m2:] - A_act[m2:]
    return dict(mae=float(np.abs(d).mean() * scale), rmse=float(np.sqrt((d ** 2).mean()) * scale),
                p95=float(np.percentile(np.abs(d), 95) * scale),
                mape=float(np.abs(A_hat[m2:].sum(axis=1) - A_act[m2:].sum(axis=1)).mean()
                           / A_act[m2:].sum(axis=1).mean() * 100))
