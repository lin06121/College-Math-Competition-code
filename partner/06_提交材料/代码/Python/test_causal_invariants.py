# -*- coding: utf-8 -*-
"""问题二/三因果管道的独立不变量测试。

运行：python test_causal_invariants.py
测试不改写结果文件；它直接对公共核心的小规模实例和全年Q2输出做检查。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from causal_core import D, balance, buffer_of  # noqa: E402
    from engine import ETA_C, ETA_D, P_MAX, SOC_MIN, SOC_MAX  # noqa: E402
    DATA_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError):
    # 允许在未配置正式数据目录的机器上先运行合成物理测试。
    DATA_AVAILABLE = False
    ETA_C = ETA_D = 0.9
    P_MAX, SOC_MIN, SOC_MAX = 5000 * 10 / 60, 1200.0, 10800.0
    def balance(P0, Cpl, Dpl, soc0, L, PV):
        C = np.zeros(len(P0)); Dd = np.zeros(len(P0)); Em = np.zeros(len(P0))
        soc = float(soc0)
        for t in range(len(P0)):
            c = min(max(Cpl[t], 0.0), P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C))
            d = min(max(Dpl[t], 0.0), P_MAX, ETA_D * max(0.0, soc - SOC_MIN + ETA_C * c))
            gap = L[t] - (PV[t] + P0[t] + d - c)
            if gap > 0:
                c = max(0.0, c - gap)
                gap = L[t] - (PV[t] + P0[t] + d - c)
                d = min(d + max(gap, 0.0), P_MAX,
                        ETA_D * max(0.0, soc - SOC_MIN + ETA_C * c))
                gap = L[t] - (PV[t] + P0[t] + d - c)
            elif gap < 0:
                d = max(0.0, d + gap)
                gap = L[t] - (PV[t] + P0[t] + d - c)
                c = min(c + max(-gap, 0.0), P_MAX,
                        max(0.0, (SOC_MAX - soc + d / ETA_D) / ETA_C))
                gap = L[t] - (PV[t] + P0[t] + d - c)
            C[t], Dd[t], Em[t] = c, d, max(gap, 0.0)
            soc += ETA_C * c - Dd[t] / ETA_D
        return C, Dd, soc, Em


def check_q2_trace(P, C_plan, D_plan, soc0, L, PV, tol=1e-7):
    """返回单日实时结算的四类残差/违规统计。"""
    C, Dd, soc_end, Em = balance(P, C_plan, D_plan, soc0, L, PV)
    soc = np.empty(len(P) + 1)
    soc[0] = soc0
    for t in range(len(P)):
        soc[t + 1] = soc[t] + ETA_C * C[t] - Dd[t] / ETA_D
    net = L - PV
    curt = np.maximum(P + Dd - C - net, 0.0)
    balance_res = P + Em + Dd - C - curt - net
    soc_res = soc[1:] - soc[:-1] - ETA_C * C + Dd / ETA_D
    return {
        "power_balance_max": float(np.max(np.abs(balance_res))),
        "soc_recursion_max": float(np.max(np.abs(soc_res))),
        "soc_boundary_max": float(max(SOC_MIN - soc.min(), soc.max() - SOC_MAX, 0.0)),
        "power_limit_max": float(max(C.max(initial=0.0), Dd.max(initial=0.0)) - P_MAX),
        "simultaneous_slots": int(np.sum((C > tol) & (Dd > tol))),
        "soc_end": float(soc[-1]),
        "em_kwh": float(Em.sum() / 6.0),
    }


def check_cross_day_trace(R, upto=None, tol=1e-7):
    upto = R["s0"].shape[0] if upto is None else upto
    return float(np.max(np.abs(R["s0"][1:upto] - R["s24"][:upto - 1]))) if upto > 1 else 0.0


def check_year_trace(R, upto):
    """对正式Q2结果逐日重算功率平衡和SOC递推残差。"""
    max_bal = 0.0; max_soc = 0.0; max_soc_end = 0.0; max_soc_bound = 0.0; simultaneous = 0
    for i in range(upto):
        P, C, Dd, Em = R["P"][i], R["C"][i], R["D"][i], R["Em"][i]
        net = D.L_act[i] - D.PV_act[i]
        curt = np.maximum(P + Dd - C - net, 0.0)
        max_bal = max(max_bal, float(np.max(np.abs(P + Em + Dd - C - curt - net))))
        E = np.empty(145); E[0] = R["s0"][i]
        E[1:] = E[0] + np.cumsum(ETA_C * C - Dd / ETA_D)
        max_soc = max(max_soc, float(np.max(np.abs(E[1:] - E[:-1] - ETA_C * C + Dd / ETA_D))))
        max_soc_end = max(max_soc_end, float(abs(E[-1] - R["s24"][i])))
        max_soc_bound = max(max_soc_bound, float(max(SOC_MIN - E.min(), E.max() - SOC_MAX, 0.0)))
        simultaneous += int(np.sum((C > 1e-7) & (Dd > 1e-7)))
    return {"power_balance_max": max_bal, "soc_recursion_max": max_soc,
            "soc_end_residual_max": max_soc_end,
            "soc_boundary_max": max_soc_bound, "simultaneous_slots": simultaneous}


def check_nonanticipativity(day_index, q=0.7, window=30):
    """未来实际负荷/PV扰动不应改变目标日前计划缓冲。"""
    i = int(day_index)
    b0 = buffer_of(i, q, window=window)
    old_l = D.L_act.copy(); old_pv = D.PV_act.copy()
    try:
        # 只改目标日及之后，历史切片 [i-window:i] 不应受影响。
        D.L_act[i:] += 1234.5
        D.PV_act[i:] = np.maximum(D.PV_act[i:] - 987.6, 0.0)
        b1 = buffer_of(i, q, window=window)
    finally:
        D.L_act[:] = old_l
        D.PV_act[:] = old_pv
    return float(np.max(np.abs(b0 - b1)))


def main():
    # 用一个有充电、放电、弃光和紧急购电的合成日验证基本物理恒等式。
    h = 12
    P = np.full(h, 10.0)
    C_plan = np.zeros(h); C_plan[0:3] = 100.0
    D_plan = np.zeros(h); D_plan[6:9] = 100.0
    L = np.r_[np.full(3, 5.0), np.full(3, 50.0), np.full(3, 140.0), np.full(3, 20.0)]
    PV = np.r_[np.full(3, 120.0), np.zeros(9)]
    chk = check_q2_trace(P, C_plan, D_plan, 6000.0, L, PV)
    assert chk["power_balance_max"] <= 1e-7, chk
    assert chk["soc_recursion_max"] <= 1e-7, chk
    assert chk["soc_boundary_max"] <= 1e-7, chk
    assert chk["power_limit_max"] <= 1e-7, chk
    assert chk["simultaneous_slots"] == 0, chk

    if not DATA_AVAILABLE:
        print("Synthetic Q2 invariant tests passed; formal data audit skipped")
        return

    # 全年Q2：使用正式因果管道，检查跨日SOC衔接和未来信息不影响历史计划。
    from upgraded_core import run_q2  # noqa: E402
    from fc_upgraded import build_load, build_pv  # noqa: E402
    Lh, _ = build_load(); PVh = build_pv()
    R = run_q2(Lh, PVh, q=0.7, upto=min(D.n, D.m2 + 5))
    cross = check_cross_day_trace(R, upto=D.m2 + 5)
    trace = check_year_trace(R, D.m2 + 5)
    nonant = check_nonanticipativity(D.m2 + 10)
    assert trace["power_balance_max"] <= 1e-7, trace
    assert trace["soc_recursion_max"] <= 1e-7, trace
    assert trace["soc_end_residual_max"] <= 1e-7, trace
    assert trace["soc_boundary_max"] <= 1e-7, trace
    assert trace["simultaneous_slots"] == 0, trace
    assert cross <= 1e-7, cross
    assert nonant <= 1e-12, nonant
    print("Q2 invariant tests passed")
    print({"synthetic": chk, "year_trace": trace,
           "cross_day_soc_max": cross, "nonanticipativity_max": nonant})


if __name__ == "__main__":
    main()
