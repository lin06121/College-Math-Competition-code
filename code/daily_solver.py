"""
通用单日储能经济调度 MILP 求解器。

支持：
- 固定/自由终止 SOC
- 是否启用紧急购电变量 P_em（5 倍电价惩罚）
- 日前阶段仅优化计划购电，或同时优化紧急购电
- 自动允许光伏弃光（curtailment），保证模型可行性
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from config import (
    SLOTS_PER_DAY,
    HOURS_PER_SLOT,
    STORAGE_MAX_POWER,
    STORAGE_EFFICIENCY,
    STORAGE_SOC_MIN,
    STORAGE_SOC_MAX,
)


def build_and_solve(price, load, pv, e0,
                    eta=STORAGE_EFFICIENCY,
                    p_max=STORAGE_MAX_POWER,
                    e_min=STORAGE_SOC_MIN,
                    e_max=STORAGE_SOC_MAX,
                    cyclic=False,
                    e_end=None,
                    allow_emergency=False,
                    dt=HOURS_PER_SLOT,
                    verbose=False):
    """
    用 scipy.optimize.milp 求解单日储能经济调度。

    参数
    ----------
    price, load, pv : ndarray, shape (SLOTS_PER_DAY,)
        电价（元/kWh）、负载（kW）、光伏（kW）。
    e0 : float
        初始 SOC（kWh）。
    cyclic : bool
        若为 True，强制 E(0) = E(n) = e0。
    e_end : float or None
        若指定，强制 E(n) = e_end；与 cyclic 互斥。
    allow_emergency : bool
        若为 True，增加紧急购电变量 P_em，并在目标中加入 5 倍电价惩罚。

    返回
    ----------
    dict，包含 P_plan, P_em, P_ch, P_dis, P_curt, E, cost 等。
    """
    n = SLOTS_PER_DAY
    assert len(price) == len(load) == len(pv) == n
    if cyclic and e_end is not None:
        raise ValueError("cyclic 与 e_end 不能同时指定")

    # 变量排布：
    # [P_plan(0..n-1), P_em(0..n-1)[可选],
    #  P_ch(0..n-1), P_dis(0..n-1), P_curt(0..n-1), E(0..n), u(0..n-1)]
    i_plan = 0
    offset = n
    if allow_emergency:
        i_em = offset
        offset += n
    else:
        i_em = None
    i_ch = offset
    i_dis = offset + n
    i_curt = offset + 2 * n
    i_e = offset + 3 * n
    i_u = offset + 3 * n + (n + 1)
    n_vars = offset + 3 * n + (n + 1) + n

    # 目标函数：c^T x
    c = np.zeros(n_vars)
    c[i_plan:i_plan + n] = price * dt  # 计划购电费用
    if allow_emergency:
        c[i_em:i_em + n] = 5.0 * price * dt  # 紧急购电 5 倍惩罚
    # P_ch, P_dis, P_curt, E, u 的目标系数均为 0

    # 变量上下界
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    # P_plan, P_em：购电功率无上界（负载峰值约 6000 kW，留足余量）
    lb[i_plan:i_plan + n] = 0.0
    ub[i_plan:i_plan + n] = 20000.0
    if allow_emergency:
        lb[i_em:i_em + n] = 0.0
        ub[i_em:i_em + n] = 20000.0
    # P_ch, P_dis
    lb[i_ch:i_ch + n] = 0.0
    ub[i_ch:i_ch + n] = p_max
    lb[i_dis:i_dis + n] = 0.0
    ub[i_dis:i_dis + n] = p_max
    # P_curt：弃光功率
    lb[i_curt:i_curt + n] = 0.0
    ub[i_curt:i_curt + n] = np.inf
    # E
    lb[i_e:i_e + n + 1] = e_min
    ub[i_e:i_e + n + 1] = e_max
    # u binary
    lb[i_u:i_u + n] = 0.0
    ub[i_u:i_u + n] = 1.0

    integrality = np.zeros(n_vars, dtype=int)
    integrality[i_u:i_u + n] = 1  # integer, bounds [0,1] => binary

    constraints = []

    # 1. 功率平衡：P_plan + P_em + PV + P_dis = Load + P_ch + P_curt
    #    => P_plan + P_em - P_ch + P_dis - P_curt = Load - PV
    A_balance = np.zeros((n, n_vars))
    for t in range(n):
        A_balance[t, i_plan + t] = 1.0
        if allow_emergency:
            A_balance[t, i_em + t] = 1.0
        A_balance[t, i_ch + t] = -1.0
        A_balance[t, i_dis + t] = 1.0
        A_balance[t, i_curt + t] = -1.0
    b_balance = load - pv
    constraints.append(LinearConstraint(A_balance, lb=b_balance, ub=b_balance))

    # 2. 储能动态：E(t+1) = E(t) + η*P_ch*dt - P_dis/η*dt
    A_dyn = np.zeros((n, n_vars))
    for t in range(n):
        A_dyn[t, i_e + t + 1] = 1.0
        A_dyn[t, i_e + t] = -1.0
        A_dyn[t, i_ch + t] = -eta * dt
        A_dyn[t, i_dis + t] = (1.0 / eta) * dt
    constraints.append(LinearConstraint(A_dyn, lb=0.0, ub=0.0))

    # 3. 充放电互斥
    A_ch = np.zeros((n, n_vars))
    for t in range(n):
        A_ch[t, i_ch + t] = 1.0
        A_ch[t, i_u + t] = -p_max
    constraints.append(LinearConstraint(A_ch, lb=-np.inf, ub=0.0))

    A_dis = np.zeros((n, n_vars))
    for t in range(n):
        A_dis[t, i_dis + t] = 1.0
        A_dis[t, i_u + t] = p_max
    constraints.append(LinearConstraint(A_dis, lb=-np.inf, ub=p_max))

    # 4. 初始 SOC
    A_e0 = np.zeros((1, n_vars))
    A_e0[0, i_e] = 1.0
    constraints.append(LinearConstraint(A_e0, lb=e0, ub=e0))

    # 5. 终止 SOC
    A_en = np.zeros((1, n_vars))
    A_en[0, i_e + n] = 1.0
    if cyclic:
        constraints.append(LinearConstraint(A_en, lb=e0, ub=e0))
    elif e_end is not None:
        constraints.append(LinearConstraint(A_en, lb=e_end, ub=e_end))
    # 否则 E(n) 由 e_min/e_max 边界自由优化

    bounds = Bounds(lb, ub)

    if verbose:
        n_bin = n
        n_cont = n_vars - n_bin
        print(f"变量数: {n_vars} (连续 {n_cont}, 整数/二进制 {n_bin})")
        print("开始调用 scipy.optimize.milp ...")

    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

    if not res.success:
        raise RuntimeError(f"MILP 求解失败: {res.message}")

    x = res.x
    P_plan = x[i_plan:i_plan + n]
    P_em = x[i_em:i_em + n] if allow_emergency else np.zeros(n)
    P_ch = x[i_ch:i_ch + n]
    P_dis = x[i_dis:i_dis + n]
    P_curt = x[i_curt:i_curt + n]
    E = x[i_e:i_e + n + 1]
    u = x[i_u:i_u + n]

    E_plan_kWh = P_plan * dt
    E_em_kWh = P_em * dt
    E_ch_kWh = P_ch * dt
    E_dis_kWh = P_dis * dt
    E_curt_kWh = P_curt * dt

    total_plan_cost = float((E_plan_kWh * price).sum())
    total_em_cost = float((E_em_kWh * price * 5.0).sum()) if allow_emergency else 0.0
    total_cost = float(res.fun)

    return {
        "success": True,
        "status": res.status,
        "message": res.message,
        "P_plan_kW": P_plan,
        "P_em_kW": P_em,
        "P_ch_kW": P_ch,
        "P_dis_kW": P_dis,
        "P_curt_kW": P_curt,
        "E_kWh": E,
        "u": u,
        "E_plan_kWh": E_plan_kWh,
        "E_em_kWh": E_em_kWh,
        "E_ch_kWh": E_ch_kWh,
        "E_dis_kWh": E_dis_kWh,
        "E_curt_kWh": E_curt_kWh,
        "total_plan_cost": total_plan_cost,
        "total_em_cost": total_em_cost,
        "total_cost": total_cost,
    }


def verify_solution(sol, price, load, pv,
                    eta=STORAGE_EFFICIENCY,
                    e0=None,
                    e_end_expected=None,
                    dt=HOURS_PER_SLOT,
                    tol=1e-4):
    """校验解是否满足功率平衡、储能动态、SOC 等约束。"""
    n = SLOTS_PER_DAY
    P_plan = sol["P_plan_kW"]
    P_em = sol["P_em_kW"]
    P_ch = sol["P_ch_kW"]
    P_dis = sol["P_dis_kW"]
    P_curt = sol["P_curt_kW"]
    E = sol["E_kWh"]

    ok = True
    # 功率平衡
    balance_err = np.abs(P_plan + P_em + pv + P_dis - load - P_ch - P_curt)
    if balance_err.max() > tol:
        print(f"[警告] 功率平衡最大误差: {balance_err.max():.6f} kW")
        ok = False
    else:
        print(f"[OK] 功率平衡最大误差: {balance_err.max():.6f} kW")

    # 储能动态
    dyn_err = np.abs(E[1:] - E[:-1] - eta * P_ch * dt + P_dis / eta * dt)
    if dyn_err.max() > tol:
        print(f"[警告] 储能动态最大误差: {dyn_err.max():.6f} kWh")
        ok = False
    else:
        print(f"[OK] 储能动态最大误差: {dyn_err.max():.6f} kWh")

    # SOC 边界
    if E.min() < STORAGE_SOC_MIN - tol or E.max() > STORAGE_SOC_MAX + tol:
        print(f"[警告] SOC 越界: [{E.min():.2f}, {E.max():.2f}]")
        ok = False
    else:
        print(f"[OK] SOC 范围: [{E.min():.2f}, {E.max():.2f}] kWh")

    # 初始 SOC
    if e0 is not None and abs(E[0] - e0) > tol:
        print(f"[警告] 初始 SOC 不为 {e0}: E[0]={E[0]:.2f}")
        ok = False
    else:
        if e0 is not None:
            print(f"[OK] E[0]={e0:.2f} kWh")

    # 终止 SOC
    if e_end_expected is not None and abs(E[-1] - e_end_expected) > tol:
        print(f"[警告] 终止 SOC 不为 {e_end_expected}: E[-1]={E[-1]:.2f}")
        ok = False
    else:
        if e_end_expected is not None:
            print(f"[OK] E[144]={e_end_expected:.2f} kWh")

    # 无同时充放电
    overlap = (P_ch > tol) & (P_dis > tol)
    if overlap.any():
        print(f"[警告] 存在 {overlap.sum()} 个同时充放电时段")
        ok = False
    else:
        print(f"[OK] 无同时充放电")

    # 功率边界
    if P_ch.max() > STORAGE_MAX_POWER + tol or P_dis.max() > STORAGE_MAX_POWER + tol:
        print(f"[警告] 充放电功率越界")
        ok = False
    else:
        print(f"[OK] 充放电功率未越界")

    print(f"\n统计:")
    print(f"  全天计划购电费: {sol['total_plan_cost']:.2f} 元")
    if sol.get("total_em_cost", 0) > 0:
        print(f"  全天紧急购电费: {sol['total_em_cost']:.2f} 元")
    print(f"  全天总费用: {sol['total_cost']:.2f} 元")
    print(f"  总充电量: {sol['E_ch_kWh'].sum():.2f} kWh")
    print(f"  总放电量: {sol['E_dis_kWh'].sum():.2f} kWh")
    if sol.get("E_curt_kWh") is not None and sol["E_curt_kWh"].sum() > 0:
        print(f"  总弃光量: {sol['E_curt_kWh'].sum():.2f} kWh")

    return ok
