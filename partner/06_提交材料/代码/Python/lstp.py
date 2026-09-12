# -*- coding: utf-8 -*-
"""两阶段随机规划计划器(LSTP): 0:00 决策 P,C,D; 执行阶段照计划执行、缺口按 5 倍电价购电
   与之前的 SAA 的关键区别: **第二阶段不含储能自由度**(储能按 Stage-1 的计划执行),
   因此不存在"情景内储能自由度被重复利用"的乐观偏差。
   情景: 以当日预测为条件, 叠加历史"预测残差"配对样本 L_s = L̂+ΔL_s, PV_s = P̂V+ΔPV_s
   目标: min Σp·P + (1/S)Σ_s Σ 5p·e_s − v·E_end
"""
import numpy as np
from scipy.optimize import linprog

H = 144
P_MAX = 833.3333
SOC_MIN, SOC_MAX = 1200.0, 10800.0
ETA_C = ETA_D = 0.9
EPS = 1e-4


def plan_lstp(price, Lhat, PVhat, resid, soc0, v, S=20, use_buffer=None, w_extra=1.0,
              risk='mean', alpha=0.9, lam=1.0, rng=None):
    """risk: 'mean'(期望) | 'cvar'(均值+λ·CVaR_α) | 'robust'(最坏情景)"""
    K = len(resid)
    idx = np.linspace(max(0, K - S), K - 1, min(S, K)).astype(int) if K >= S else np.arange(K)
    sc = []
    for k in idx:
        dL, dPV = resid[k]
        Ls = np.maximum(Lhat + dL * w_extra, 0.0)
        PVs = np.maximum(PVhat + dPV * w_extra, 0.0)
        sc.append((Ls, PVs))
    S = len(sc)
    iC, iD, iE = H, 2 * H, 3 * H
    nE = H + 1
    iEm = 4 * H + 1
    nExtra = 0 if risk == 'mean' else (1 if risk == 'robust' else 1 + S)
    iX = iEm + S * H                     # 附加变量区
    N = iX + nExtra
    off_em = lambda s: iEm + s * H
    c = np.zeros(N)
    c[0:H] = price
    w = 1.0 / S if risk == 'mean' else (1.0 - lam) / S if risk == 'cvar' else 0.0
    for s in range(S):
        c[off_em(s):off_em(s) + H] = 5.0 * price * w
    c[iC:iC + H] += EPS; c[iD:iD + H] += EPS
    c[iE + H] = -v
    if risk == 'robust':
        c[iX] = 1.0                                   # + t (最坏情景紧急费)
    elif risk == 'cvar':
        c[iX] = lam                                  # + λ·t
        c[iX + 1:iX + 1 + S] = lam / ((1 - alpha) * S)   # + λ/((1-α)S)·Σ z_s
    Aeq, beq, Aub, bub = [], [], [], []
    for t in range(H):                                    # SOC 递推(Stage-1)
        row = np.zeros(N); row[iE + t], row[iE + t + 1] = -1, 1
        row[iC + t], row[iD + t] = -ETA_C, 1 / ETA_D
        Aeq.append(row); beq.append(0.0)
    row = np.zeros(N); row[iE] = 1; Aeq.append(row); beq.append(soc0)
    for s in range(S):                                    # 情景约束: 缺口 <= 紧急购电
        Ls, PVs = sc[s]
        for t in range(H):
            row = np.zeros(N)
            row[off_em(s) + t] = -1.0
            row[t], row[iC + t], row[iD + t] = -1.0, 1.0, -1.0
            Aub.append(row); bub.append(PVs[t] - Ls[t])
    if risk == 'robust':                                  # t >= Σ 5p·e_s
        for s in range(S):
            row = np.zeros(N); row[iX] = -1.0
            for t in range(H):
                row[off_em(s) + t] = 5.0 * price[t]
            Aub.append(row); bub.append(0.0)
    elif risk == 'cvar':                                  # z_s >= cost_s - t
        for s in range(S):
            row = np.zeros(N); row[iX + 1 + s] = -1.0; row[iX] = -1.0
            for t in range(H):
                row[off_em(s) + t] = 5.0 * price[t]
            Aub.append(row); bub.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + H] = P_MAX; ub[iD:iD + H] = P_MAX
    lb[iE:iE + nE] = SOC_MIN; ub[iE:iE + nE] = SOC_MAX
    lb[iE] = ub[iE] = soc0
    if risk != 'mean':
        lb[iX] = -np.inf
        if risk == 'cvar':
            lb[iX + 1:iX + 1 + S] = 0.0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success, res.message
    P = np.maximum(res.x[0:H], 0)
    C = res.x[iC:iC + H]; Dv = res.x[iD:iD + H]
    E = res.x[iE:iE + nE]
    exp_em = float(sum(5.0 * price @ res.x[off_em(s):off_em(s) + H] for s in range(S)) / S)
    return dict(P=P, C=C, D=Dv, E=E, obj=float(res.fun), exp_em=exp_em,
                plan_cost=float(price @ P))


def make_resid(L_act, PV_act, Lhat_mat, PVhat_mat, upto, K=30, per_day=False):
    """历史残差样本: (ΔL, ΔPV) 逐日配对; per_day=True 时按"同类日"分组返回 dict"""
    lo = max(0, upto - K)
    res = [(L_act[j] - Lhat_mat[j], PV_act[j] - PVhat_mat[j]) for j in range(lo, upto)]
    return res
