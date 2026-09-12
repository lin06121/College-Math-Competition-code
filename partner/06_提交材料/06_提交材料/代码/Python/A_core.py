# -*- coding: utf-8 -*-
"""口径 A 公共核心: 预测构造 + 报童缓冲 + 计划 LP + 实时平衡控制(0:00 一次定下的响应规则)
   · 问题二: 0:00 计划 P⁰,C⁰,D⁰；日内按 0:00 定下的平衡规则执行
      ①按当前 SOC 裁剪计划充放 → ②缺额先减充、再增放 → ③富余先减放、再增充 → 兜不住才 5 倍紧急购电
      （规则只用当前实测与当前 SOC，无需未来信息、不重新优化计划）
   · 问题三: 0:00 计划 + 在指定时刻用最新预报增量调整（并按该预报口径重设报童裕度）
"""
import numpy as np
from engine import solve_day
from causal_core import D, price, PR, balance
import fc_upgraded as FU
import upgraded_core as UC

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
FCADJ = D.fca_m
LOW = np.array([d in {4, 5} for d in D.dow])

L, LOAD_INFO = FU.build_load()
PV = FU.build_pv()
P_RT = FU.build_price()


def buffer_of(Lh, PVh, q, i, W=30):
    lo = max(0, i - W)
    if i - lo < 7 or q <= 0:
        return np.zeros(H)
    e = (L_ACT[lo:i] - Lh[lo:i]) - (PV_ACT[lo:i] - PVh[lo:i])
    return np.quantile(e, q, axis=0)


def adj_buffer(Lh, PVh, hh, h0, q, i, W=30):
    """调整时刻的裕度: 用"该时刻最新预报"的历史残差经验分位(只用 j<i)"""
    lo = max(0, i - W)
    if i - lo < 7 or q <= 0:
        return np.zeros(H - h0)
    e = (L_ACT[lo:i, h0:] - Lh[lo:i, h0:]) - (PV_ACT[lo:i, h0:] - FCADJ[hh][lo:i, h0:])
    return np.quantile(e, q, axis=0)


def run_q2(Lh=None, PVh=None, q=0.7, pmat=None, upto=None, buffer_mat_in=None, bill=None):
    Lh = L if Lh is None else Lh
    PVh = PV if PVh is None else PVh
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    soc = 6000.0
    P_m = np.zeros((n, H)); C_m = np.zeros((n, H)); D_m = np.zeros((n, H)); E_m = np.zeros((n, H))
    s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        b = buffer_of(Lh, PVh, q, i) if buffer_mat_in is None else buffer_mat_in[i]
        r = solve_day(p, Lh[i] + b, PVh[i], soc, soc_end=None, v=v)
        c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        P_m[i], C_m[i], D_m[i], E_m[i] = r['P'], c, d, e
        b = (pm if bill is None else bill)[i]
        pc[i] = float(b @ r['P']); ec[i] = float(5 * b @ e)
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P=P_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(E_m[m2:upto].sum()))


def run_q3(Lh=None, PVh=None, q=0.7, pmat=None, adjust=(12,), upto=None, use_adjbuf=True,
           bill=None, pmat_adj=None):
    """问题三: 委托给 upgraded_core.run_q3(正式口径: 逐段滚动 + 实测 SOC + 0:00 计划为偏差基准 +
    因果去偏预报; use_adjbuf=True 时在调整时刻按该时刻预报重设报童裕度)"""
    Lh = L if Lh is None else Lh
    PVh = PV if PVh is None else PVh
    return UC.run_q3(Lh, PVh, q=q, pmat=pmat, adjust=adjust, upto=upto,
                     debias=True, use_adjbuf=use_adjbuf, bill=bill, pmat_adj=pmat_adj)


def nostorage(Lh=None, PVh=None, q=0.7, pmat=None, upto=None):
    Lh = L if Lh is None else Lh
    PVh = PV if PVh is None else PVh
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    tot = 0.0
    for i in range(upto):
        P0 = np.maximum(Lh[i] + buffer_of(Lh, PVh, q, i) - PVh[i], 0.0)
        Em = np.maximum(L_ACT[i] - PV_ACT[i] - P0, 0.0)
        if i >= m2:
            tot += float(pm[i] @ P0) + float(5 * pm[i] @ Em)
    return tot


def clairvoyant(pmat=None, upto=None):
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
                mape=float(np.abs(A_hat[m2:].sum(axis=1) - A_act[m2:].sum(axis=1)).mean()
                           / A_act[m2:].sum(axis=1).mean() * 100))
