# -*- coding: utf-8 -*-
"""
求解引擎：单日 LP（计划模式 / 调整模式）+ 结算 + 通用工具
时间轴：slot 1..144 = 当天第1..144个10min区间 [(s-1)*10, s*10) min
变量单位：全部 kWh（功率kW×1/6h）；价格 元/kWh
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ETA_C = 0.9            # 充电效率(单程, 已定案)
ETA_D = 0.9            # 放电效率(单程)
P_MAX = 5000 * 10 / 60 # 充/放电功率上限折合每时段电量 kWh = 833.333
SOC_MIN, SOC_MAX = 1200.0, 10800.0
SLOTS = 144

def solve_day(price, load, pv, soc0, soc_end=None, v=0.0, base_P=None, t0=0, method='highs'):
    """
    单日(或剩余时段)LP。
    price/load/pv: 长度 h 的数组(元/kWh, kWh/时段, kWh/时段)
    soc0: 起始SOC(kWh);  soc_end: None=自由(终值入目标 -v*E_end) 或 数值=固定终值
    v: 终值系数(元/kWh, 已含放电效率折算后的价值)
    base_P: None=计划模式(目标 min sum p*P - v*E_end)
            数组=调整模式(计划量已锁定base_P, 增量目标 sum[-0.5p*S + 1.5p*X] - v*E_end)
    t0: 该段对应全局slot起点(0-based), 仅用于日志
    返回 dict: P,C,D(长h), E(长h+1, E[0]=soc0), S,X(长h), gross, inc_cost, status
    method: scipy.linprog 算法('highs' 对偶单纯形 / 'highs-ipm' 内点法), 仅用于求解器稳健性对照
    """
    h = len(price)
    soc0 = float(min(max(soc0, SOC_MIN), SOC_MAX))   # 夹取起始 SOC, 避免浮点越界导致误判不可行
    p = np.asarray(price, float)
    l = np.asarray(load, float)
    g = np.asarray(pv, float)
    adj = base_P is not None
    b0 = np.asarray(base_P, float) if adj else None

    nP, nC, nD, nE = h, h, h, h + 1
    nS, nX = (h, h) if adj else (0, 0)
    n = nP + nC + nD + nE + nS + nX
    iP, iC, iD = 0, h, 2 * h
    iE = 3 * h
    iS, iX = iE + nE, iE + nE + h

    c = np.zeros(n)
    if not adj:
        c[iP:iP + h] = p
    else:
        c[iS:iS + h] = -0.5 * p
        c[iX:iX + h] = 1.5 * p
    c[iE + h] = -v  # E_end

    Aeq, beq = [], []
    Aub, bub = [], []
    # 功率平衡(允许弃光): PV + P + D - C >= L  <=>  -P + C - D <= PV - L
    for t in range(h):
        row = np.zeros(n)
        row[iP + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(g[t] - l[t])
    # SOC 递推: E[t+1] = E[t] + eta_c*C[t] - D[t]/eta_d
    for t in range(h):
        row = np.zeros(n)
        row[iE + t] = -1; row[iE + t + 1] = 1
        row[iC + t] = -ETA_C; row[iD + t] = 1.0 / ETA_D
        Aeq.append(row); beq.append(0.0)
    if soc_end is not None:
        row = np.zeros(n); row[iE + h] = 1
        Aeq.append(row); beq.append(float(soc_end))
    if adj:
        # P + S - X = base_P (仅当有调整时; 无调整时段通过bounds固定P)
        for t in range(h):
            row = np.zeros(n)
            row[iP + t], row[iS + t], row[iX + t] = 1, 1, -1
            Aeq.append(row); beq.append(b0[t])

    lb = np.zeros(n); ub = np.full(n, np.inf)
    for t in range(h):
        ub[iC + t] = P_MAX; ub[iD + t] = P_MAX
        if adj and b0[t] >= 0:  # 未纳入调整的时段: P固定=base
            pass  # 全部时段都参与调整时无固定; 由调用方裁剪时段
    for t in range(h + 1):
        lb[iE + t], ub[iE + t] = SOC_MIN, SOC_MAX
    lb[iE], ub[iE] = soc0, soc0
    if adj:
        # 若某时段不调整则锁定: 由调用方传入 base_P 为 None 的时段 -> 我们要求全部参与
        pass

    res = linprog(c, A_ub=np.array(Aub) if Aub else None, b_ub=np.array(bub) if bub else None,
                  A_eq=np.array(Aeq) if Aeq else None, b_eq=np.array(beq) if beq else None,
                  bounds=list(zip(lb, ub)), method=method)
    if not res.success:
        return dict(status=res.message, P=None, C=None, D=None, E=None, S=None, X=None,
                    gross=None, inc_cost=None)
    P = res.x[iP:iP + h]; C = res.x[iC:iC + h]; D = res.x[iD:iD + h]
    E = res.x[iE:iE + h + 1]
    S = res.x[iS:iS + h] if adj else np.zeros(h)
    X = res.x[iX:iX + h] if adj else np.zeros(h)
    gross = float(np.sum(p * P))
    inc_cost = float(np.sum(-0.5 * p * S + 1.5 * p * X)) if adj else gross
    return dict(status='ok', P=P, C=C, D=D, E=E, S=S, X=X, gross=gross, inc_cost=inc_cost,
                obj=float(res.fun))

def settle_emergency(L_act, PV_act, P, C, D):
    """按计划执行的结算: 缺口 = max(0, L_act - PV_act - P - D + C)"""
    short = np.asarray(L_act) - np.asarray(PV_act) - np.asarray(P) - np.asarray(D) + np.asarray(C)
    return np.maximum(short, 0.0)

def merge_spans(idx):
    """把一组slot序号(1-based, 已排序)合并为连续跨度列表 [(start_slot, end_slot)]"""
    if len(idx) == 0:
        return []
    spans = []
    s = p = idx[0]
    for x in idx[1:]:
        if x == p + 1:
            p = x
        else:
            spans.append((s, p)); s = p = x
    spans.append((s, p))
    return spans

def span_label(s0, s1):
    """slot s0..s1 (1-based) -> 'HH:MM-HH:MM' (区间起点~终点)"""
    def hm(slot):  # slot起点分钟
        m = (slot - 1) * 10
        return f"{m//60}:{m%60:02d}"
    return f"{hm(s0)}-{hm(s1 + 1)}"

def block_agg(x, blocks=6):
    """144时段向量 -> 每4小时块(24时段)求和, 返回长度blocks数组"""
    x = np.asarray(x)
    return np.array([x[b * 24:(b + 1) * 24].sum() for b in range(blocks)])
