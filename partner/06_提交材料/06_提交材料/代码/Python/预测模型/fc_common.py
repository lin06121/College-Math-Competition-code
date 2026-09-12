# -*- coding: utf-8 -*-
"""预测模型实验 · 公共设施
   - 数据: 复用 causal_core 的 D(实测/典型日/预报) 与 price/PR
   - 口径: 全部预测只用 d 日 0:00 之前可得的信息(因果), 不修改论文/提交代码
   - 端到端评价: 与 causal_core.run_q2 完全同构的管道, 但预测矩阵由外部注入
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, '04_建模求解', 'scripts'))

from causal_core import (D, price, PR, balance, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D)  # noqa: E402
from engine import solve_day  # noqa: E402

n = D.n
H = 144
OUT = os.path.join(ROOT, '07_预测模型实验')
DATA = os.path.join(OUT, 'data')
FIG = os.path.join(OUT, 'fig')
m2 = D.m2
DATES = D.dates
DOW = D.dow
IS_LOW = np.array([d in {4, 5} for d in DOW])          # 周五/周六 = 低负荷/低价类
L_ACT = D.L_act                                        # kWh/10min
PV_ACT = D.PV_act
FC0 = D.fc0m                                           # 附件3 0:00 发布预报
FCADJ = D.fca_m                                        # 6/12/18 时预报
L_TYP = D.L_typ
PV_TYP = D.PV_typ
PRICE = price                                          # 固定基准电价曲线(附件1)
RTP = PR                                               # 附件4 实时电价

SLOT = np.arange(1, H + 1)
SLOT_F = SLOT / H


def fourier(x, K, period=1.0):
    """返回 [sin(2πkx), cos(2πkx)] 共 2K 列"""
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * x / period))
        cols.append(np.cos(2 * np.pi * k * x / period))
    return np.column_stack(cols)


# ---------------- 通用: 残差缓冲 + 端到端管道 ----------------
def buffer_from_resid(L_hat, PV_hat, q, W=30):
    """近 W 天净需求预测残差的经验分位(逐时段), 只用历史 -> 返回 (n,144)"""
    e = (L_ACT - L_hat) - (PV_ACT - PV_hat)
    B = np.zeros((n, H))
    for i in range(n):
        lo = max(0, i - W)
        if i - lo >= 7 and q > 0:
            B[i] = np.quantile(e[lo:i], q, axis=0)
    return B


def run_q2_fc(L_hat, PV_hat, q=0.7, reserve=1200.0, price_plan=None, price_bill=None,
              upto=None, buffer_mat=None):
    """与 causal_core.run_q2 同构: 0:00 计划(注入预测) + 日内实时平衡控制
       price_plan: 计划阶段电价 (n,144) 或 None=固定电价; price_bill: 结算电价"""
    upto = n if upto is None else upto
    pm = price_plan if price_plan is not None else np.broadcast_to(PRICE, (n, H))
    bm = price_bill if price_bill is not None else pm
    B = buffer_from_resid(L_hat, PV_hat, q) if buffer_mat is None else buffer_mat
    soc = 6000.0
    pc = np.zeros(n)
    ec = np.zeros(n)
    Em_all = np.zeros((n, H))
    for i in range(upto):
        p = pm[i]
        if price_plan is None:
            v = 0.9 * PRICE.mean() if i < n - 1 else 0.0
        else:
            nxt = pm[min(i + 1, n - 1)]
            v = 0.9 * nxt.mean() if i < n - 1 else 0.0
        r = solve_day(p, L_hat[i] + B[i], PV_hat[i], soc, soc_end=None, v=v)
        C, Dd, sEnd, Em = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i], reserve)
        pc[i] = float(bm[i] @ r['P'])
        ec[i] = float(5 * bm[i] @ Em)
        Em_all[i] = Em
        soc = sEnd
        if i + 1 >= upto:
            break
    return dict(plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(Em_all[m2:upto].sum()),
                pc=pc, ec=ec)


def run_q3_fc(L_hat, PV_hat, q=0.7, reserve=1200.0, price_plan=None, adjust=(6, 12, 18),
              fc_adj=None, upto=None, buf_mat=None):
    """0:00 计划(注入预测) + 滚动调整 + 实时平衡"""
    upto = n if upto is None else upto
    pm = price_plan if price_plan is not None else np.broadcast_to(PRICE, (n, H))
    fc_adj = FCADJ if fc_adj is None else fc_adj
    B = buffer_from_resid(L_hat, PV_hat, q) if buf_mat is None else buf_mat
    soc = 6000.0
    HH = [36, 72, 108]
    fh = [6, 12, 18]
    pc = np.zeros(n); dv = np.zeros(n); ec = np.zeros(n); ek = np.zeros(n)
    for i in range(upto):
        pday = pm[i]
        v = (0.9 * PRICE.mean() if price_plan is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        Lbar = L_hat[i] + B[i]
        r0 = solve_day(pday, Lbar, PV_hat[i], soc, soc_end=None, v=v)
        P, C, Dd, SOC = r0['P'], r0['C'], r0['D'], r0['E']
        Pinit = P.copy()
        for h0, hh in zip(HH, fh):
            if hh not in adjust:
                continue
            r = solve_day(pday[h0:], Lbar[h0:], fc_adj[hh][i, h0:], SOC[h0], soc_end=None, v=v,
                          base_P=P[h0:])
            P[h0:], C[h0:], Dd[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0 + 1], r['E'][1:]])
        C2, D2, sEnd, Em = balance(P, C, Dd, soc, L_ACT[i], PV_ACT[i], reserve)
        pc[i] = float(pday @ Pinit)
        dv[i] = float(np.sum(-0.5 * pday * np.maximum(Pinit - P, 0) + 1.5 * pday * np.maximum(P - Pinit, 0)))
        ec[i] = float(5 * pday @ Em)
        ek[i] = float(Em.sum())
        soc = sEnd
        if i + 1 >= upto:
            break
    return dict(plan=float(pc[m2:upto].sum()), dev=float(dv[m2:upto].sum()),
                em=float(ec[m2:upto].sum()), total=float((pc + dv + ec)[m2:upto].sum()),
                em_kwh=float(ek[m2:upto].sum()), pc=pc, dv=dv, ec=ec)


# ---------------- 指标 ----------------
def metrics(A_hat, A_act, tag='', scale=6.0, daily_act=None):
    """逐时段 kWh 矩阵(或价格矩阵) -> MAE/RMSE(乘 scale 还原 kW) + 日总量 MAPE(%)"""
    d = A_hat[m2:] - A_act[m2:]
    mae = np.abs(d).mean() * scale
    rmse = np.sqrt((d ** 2).mean()) * scale
    d_hat = A_hat[m2:].sum(axis=1)
    d_act = A_act[m2:].sum(axis=1)
    mape = np.abs(d_hat - d_act).mean() / d_act.mean() * 100
    return dict(tag=tag, mae_kw=float(mae), rmse_kw=float(rmse), daily_mape=float(mape))
