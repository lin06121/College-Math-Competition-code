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


def _causal_debias(arr, actual, start_slot=0, window=20):
    """用目标日前同一预报链的误差做因果去偏；绝不读取目标日或未来日。"""
    arr = np.asarray(arr, dtype=float)
    out = arr.copy()
    for i in range(n):
        if i == 0:
            continue
        lo = max(0, i - window)
        hist = arr[lo:i, start_slot:] - actual[lo:i, start_slot:]
        bias = np.nanmean(hist, axis=0)
        bias = np.where(np.isfinite(bias), bias, 0.0)
        out[i, start_slot:] = arr[i, start_slot:] - bias
    return out


# 问题三使用因果去偏预报。问题二的PV_UP仍保持其独立的堆叠预测口径。
FC0_Q3 = _causal_debias(D.fc0m, D.PV_act, 0)
FCADJ_Q3 = {h: _causal_debias(D.fca_m[h], D.PV_act, h * 6) for h in (6, 12, 18)}


def buffer_mat(Lh, PVh, q, W=30):
    """近 W 天净需求残差的逐时段经验分位(只用历史)"""
    e = (L_ACT - Lh) - (PV_ACT - PVh)
    B = np.zeros((n, H))
    for i in range(n):
        lo = max(0, i - W)
        if i - lo >= 7 and q > 0:
            B[i] = np.quantile(e[lo:i], q, axis=0)
    return B


def run_q2(Lh, PVh, q, pmat=None, upto=None, buffer_mat_in=None, asplanned=False, perfect=False,
           bill=None):
    """0:00 计划(注入预测) + 日内结算
       pmat: 0:00 计划用的电价预期(问题四=预测电价); bill: 结算用的实际电价(默认同 pmat)
       asplanned=True -> 僵硬执行; perfect=True -> 完美信息再调度"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    bl = pm if bill is None else bill
    B = buffer_mat(Lh, PVh, q) if buffer_mat_in is None else buffer_mat_in
    soc = 6000.0
    P = np.zeros((n, H)); C = np.zeros((n, H)); Dm = np.zeros((n, H)); Em = np.zeros((n, H))
    s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        b = bl[i]
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
        pc[i] = float(b @ r['P']); ec[i] = float(5 * b @ e)      # 结算按实际电价
        s0[i], s24[i], soc = soc, sEnd, sEnd
        if i + 1 >= upto:
            break
    return dict(P=P, C=C, D=Dm, Em=Em, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(Em[m2:upto].sum()))


def adj_buffer_of(Lh, PVh, hh, s, q, i, fc_all, W=30):
    """调整时刻(发布时点 hh, 从槽位 s 起)的报童裕度:
    用"该时刻最新预报链"的历史残差经验分位(只用 j<i 的历史), 即"随预报重新标定裕度"。"""
    lo = max(0, i - W)
    if i - lo < 7 or q <= 0:
        return np.zeros(H - s)
    e = (L_ACT[lo:i, s:] - Lh[lo:i, s:]) - (PV_ACT[lo:i, s:] - fc_all[hh][lo:i, s:])
    return np.quantile(e, q, axis=0)


def run_q3(Lh, PVh, q, pmat=None, adjust=(6, 12, 18), upto=None,
           debias=True, use_adjbuf=True, adj_q=None, bill=None, pmat_adj=None):
    """问题三正式管道: 0:00 计划 + 逐段滚动调整 + 逐段因果结算。

    · 逐段推进(0-6-12-18-24 时): 每段以**上一段实际执行后**的 SOC 为起点, 段末真实 SOC 传给下一段;
    · 每次调整的偏差基准恒为 **0:00 原始计划** Pinit(base_P=Pinit[s:]), 即偏差始终相对计划报出量;
    · debias=True: 问题三使用因果去偏预报(FC0_Q3 / FCADJ_Q3, 只用目标日之前的同一预报链误差);
    · use_adjbuf=True: 调整时刻按"该时刻预报"的历史残差重设报童裕度; 否则沿用 0:00 的裕度;
    · 最新一次调整的计划充放电量覆盖其后所有时段(与"锁定最新计划"一致)。
    """
    upto = n if upto is None else upto
    adj_q = q if adj_q is None else adj_q
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    bl = pm if bill is None else bill          # 结算电价(问题四=附件4 实际电价)
    padj = {} if pmat_adj is None else pmat_adj  # {6: 矩阵, 12:…, 18:…} 各发布时刻更新后的价格预测
    B = buffer_mat(Lh, PVh, q)
    soc = 6000.0; fh = [6, 12, 18]; bounds = [0, 36, 72, 108, 144]
    P0 = np.zeros((n, H)); Pf = np.zeros((n, H)); C = np.zeros((n, H)); Dm = np.zeros((n, H))
    Em = np.zeros((n, H)); s0 = np.zeros(n); s24 = np.zeros(n)
    pc = np.zeros(n); dv = np.zeros(n); ec = np.zeros(n)
    fc0_all = FC0_Q3 if debias else D.fc0m
    fcadj_all = FCADJ_Q3 if debias else FCADJ
    for i in range(upto):
        p = pm[i]
        b = bl[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        Lbar = Lh[i] + B[i]
        r0 = solve_day(p, Lbar, fc0_all[i], soc, soc_end=None, v=v)
        Pinit = r0['P'].copy()
        P = Pinit.copy()
        c_plan = r0['C'].copy(); d_plan = r0['D'].copy()
        c = np.zeros(H); d = np.zeros(H); e = np.zeros(H)
        actual_soc = soc
        for k in range(len(bounds) - 1):
            s, eidx = bounds[k], bounds[k + 1]
            if k > 0 and fh[k - 1] in adjust:
                hh = fh[k - 1]
                Ladj = (Lh[i, s:] + adj_buffer_of(Lh, PVh, hh, s, adj_q, i, fcadj_all)
                        if use_adjbuf else Lbar[s:])
                p_obj = padj[hh][i] if hh in padj else p     # 该时刻更新后的价格预测(计划用)
                r = solve_day(p_obj[s:], Ladj, fcadj_all[hh][i, s:],
                              actual_soc, soc_end=None, v=v, base_P=Pinit[s:])
                P[s:] = r['P']
                c_plan[s:] = r['C']; d_plan[s:] = r['D']
            cs, ds, actual_soc, es = balance(P[s:eidx], c_plan[s:eidx], d_plan[s:eidx],
                                             actual_soc, L_ACT[i, s:eidx], PV_ACT[i, s:eidx])
            c[s:eidx], d[s:eidx], e[s:eidx] = cs, ds, es
        P0[i], Pf[i], C[i], Dm[i], Em[i] = Pinit, P, c, d, e
        pc[i] = float(b @ Pinit)                     # 结算: 实际电价 × 计划购电量
        dv[i] = float(np.sum(-0.5 * b * np.maximum(Pinit - P, 0) + 1.5 * b * np.maximum(P - Pinit, 0)))
        ec[i] = float(5 * b @ e)
        s0[i], s24[i], soc = soc, actual_soc, actual_soc
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
