# -*- coding: utf-8 -*-
"""口径 C（最终定案）：0:00 一次性决策 + 计划照原样执行（无任何日内再调度）
   · 问题二：0:00 决定 计划购电量 P0 与充放电计划 C0,D0 -> 全天按计划执行；
              实际与计划的缺口 max(0, L-(PV+P0+D0-C0)) 按 5 倍电价紧急购电；富余弃光。
   · 问题三：0:00 计划 + 在指定时刻用最新预报对剩余时段做增量优化（付 0.5/1.5 倍偏差费），
              之后同样按最终计划照原样执行。
   · SOC 递推 E_{t+1}=E_t+0.9C-D/0.9；储能是唯一有损耗的环节（单程 90%），
     光伏/网购电直供负载无损耗。
"""
import numpy as np
from engine import solve_day, ETA_C, ETA_D, SOC_MIN, SOC_MAX
from causal_core import D, price, PR

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
FCADJ = D.fca_m
LOW = np.array([d in {4, 5} for d in D.dow])


def buffer_mat(Lh, PVh, q, W=30):
    """近 W 天净需求预测残差的逐时段经验分位(只用历史)"""
    e = (L_ACT - Lh) - (PV_ACT - PVh)
    B = np.zeros((n, H))
    for i in range(n):
        lo = max(0, i - W)
        if i - lo >= 7 and q > 0:
            B[i] = np.quantile(e[lo:i], q, axis=0)
    return B


def execute_as_planned(P, C, Dv, soc0, L, PV):
    """按计划照原样执行: 返回 (紧急购电量, 弃光量, 日末SOC)"""
    sup = PV + P + Dv - C                                    # 可供负载的电量(无效率系数)
    em = np.maximum(0.0, L - sup)                            # 缺口 -> 5 倍紧急购电
    curt = np.maximum(0.0, sup - L)                          # 富余 -> 弃光
    sEnd = float(soc0 + ETA_C * C.sum() - Dv.sum() / ETA_D)
    assert SOC_MIN - 1e-6 <= sEnd <= SOC_MAX + 1e-6, ('SOC越界', sEnd)
    return em, curt, sEnd


def run_q2(Lh, PVh, q=0.7, pmat=None, upto=None, buffer_mat_in=None):
    """问题二: 0:00 计划 + 照计划执行"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    B = buffer_mat(Lh, PVh, q) if buffer_mat_in is None else buffer_mat_in
    soc = 6000.0
    P = np.zeros((n, H)); C = np.zeros((n, H)); Dm = np.zeros((n, H)); Em = np.zeros((n, H))
    Curt = np.zeros((n, H)); s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        r = solve_day(p, Lh[i] + B[i], PVh[i], soc, soc_end=None, v=v)
        em, curt, sEnd = execute_as_planned(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        P[i], C[i], Dm[i], Em[i], Curt[i] = r['P'], r['C'], r['D'], em, curt
        pc[i] = float(p @ r['P']); ec[i] = float(5 * p @ em)
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P=P, C=C, D=Dm, Em=Em, Curt=Curt, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(Em[m2:upto].sum()),
                curt_kwh=float(Curt[m2:upto].sum()))


def adj_buffer(Lh, PVh, hh, h0, q, i, W=30):
    """调整时刻的报童裕度: 用"该时刻最新预报"的历史残差经验分位(只用 j<i)"""
    lo = max(0, i - W)
    if i - lo < 7 or q <= 0:
        return np.zeros(H - h0)
    e = (L_ACT[lo:i, h0:] - Lh[lo:i, h0:]) - (PV_ACT[lo:i, h0:] - FCADJ[hh][lo:i, h0:])
    return np.quantile(e, q, axis=0)


def run_q3(Lh, PVh, q=0.7, pmat=None, adjust=(6, 12, 18), upto=None, fc_adj=None,
           adj_buffer_on=True):
    """问题三: 0:00 计划 + 指定时刻增量调整(带该时刻预报口径的报童裕度) + 照最终计划执行"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    fc_adj = FCADJ if fc_adj is None else fc_adj
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
        P, Cc, Dd = r0['P'].copy(), r0['C'].copy(), r0['D'].copy()
        SOC = r0['E'].copy()
        Pinit = P.copy()
        for h0, hh in zip(HH, fh):
            if hh not in adjust:
                continue
            Ladj = Lh[i, h0:] + (adj_buffer(Lh, PVh, hh, h0, q, i) if adj_buffer_on else 0.0)
            r = solve_day(p[h0:], Ladj, fc_adj[hh][i, h0:], SOC[h0], soc_end=None, v=v,
                          base_P=P[h0:])
            P[h0:], Cc[h0:], Dd[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0 + 1], r['E'][1:]])   # 继承上一次调整后的储能轨迹
        em, curt, sEnd = execute_as_planned(P, Cc, Dd, soc, L_ACT[i], PV_ACT[i])
        P0[i], Pf[i], C[i], Dm[i], Em[i] = Pinit, P, Cc, Dd, em
        pc[i] = float(p @ Pinit)
        dv[i] = float(np.sum(-0.5 * p * np.maximum(Pinit - P, 0) + 1.5 * p * np.maximum(P - Pinit, 0)))
        ec[i] = float(5 * p @ em)
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P0=P0, Pf=Pf, C=C, D=Dm, Em=Em, s0=s0, s24=s24, pc=pc, dv=dv, ec=ec,
                plan=float(pc[m2:upto].sum()), dev=float(dv[m2:upto].sum()),
                em=float(ec[m2:upto].sum()), total=float((pc + dv + ec)[m2:upto].sum()),
                em_kwh=float(Em[m2:upto].sum()))


def nostorage(Lh, PVh, q, pmat=None, upto=None):
    """无储能: 0:00 买 max(预测净需求,0), 缺口 5 倍紧急"""
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
    """完全信息下界: 用当日实测制定计划(不可达), 同样受 SOC 链约束"""
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
