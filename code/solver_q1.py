"""
问题 1 求解器：典型日确定性储能经济调度（MILP）

目标：min Σ price(t) * P_grid(t) * Δt
约束：功率平衡、储能动态、充放电互斥、SOC 边界、日循环 E(0)=E(144)=6000
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds

from config import (
    DATA_PROCESSED,
    SLOTS_PER_DAY,
    HOURS_PER_SLOT,
    STORAGE_MAX_CAPACITY,
    STORAGE_MAX_POWER,
    STORAGE_EFFICIENCY,
    STORAGE_SOC_MIN,
    STORAGE_SOC_MAX,
    STORAGE_INITIAL_SOC,
)


def build_and_solve(price, load, pv,
                    eta=STORAGE_EFFICIENCY,
                    p_max=STORAGE_MAX_POWER,
                    e_min=STORAGE_SOC_MIN,
                    e_max=STORAGE_SOC_MAX,
                    e0=STORAGE_INITIAL_SOC,
                    dt=HOURS_PER_SLOT,
                    verbose=False):
    """
    用 scipy.optimize.milp 求解问题 1。

    参数均为长度 SLOTS_PER_DAY 的 1-D ndarray。
    返回 dict，包含 P_grid, P_ch, P_dis, E, cost 等。
    """
    n = SLOTS_PER_DAY
    assert len(price) == len(load) == len(pv) == n

    # 变量排布：
    # [P_grid(0..n-1), P_ch(0..n-1), P_dis(0..n-1), E(0..n), u(0..n-1)]
    i_grid = 0
    i_ch = n
    i_dis = 2 * n
    i_e = 3 * n
    i_u = 3 * n + (n + 1)
    n_vars = 5 * n + 1

    # 目标函数：c^T x = Σ price * P_grid * dt
    c = np.zeros(n_vars)
    c[i_grid:i_grid + n] = price * dt

    # 变量上下界
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    # P_ch, P_dis
    lb[i_ch:i_ch + n] = 0.0
    ub[i_ch:i_ch + n] = p_max
    lb[i_dis:i_dis + n] = 0.0
    ub[i_dis:i_dis + n] = p_max
    # E
    lb[i_e:i_e + n + 1] = e_min
    ub[i_e:i_e + n + 1] = e_max
    # u binary -> bounds [0,1], integrality 会在下面设
    lb[i_u:i_u + n] = 0.0
    ub[i_u:i_u + n] = 1.0

    integrality = np.zeros(n_vars, dtype=int)
    integrality[i_u:i_u + n] = 1  # integer, bounds [0,1] => binary

    constraints = []

    # 1. 功率平衡：P_grid + PV + P_dis = Load + P_ch
    #    => P_grid - P_ch + P_dis = Load - PV
    A_balance = np.zeros((n, n_vars))
    for t in range(n):
        A_balance[t, i_grid + t] = 1.0
        A_balance[t, i_ch + t] = -1.0
        A_balance[t, i_dis + t] = 1.0
    b_balance = load - pv
    constraints.append(LinearConstraint(A_balance, lb=b_balance, ub=b_balance))

    # 2. 储能动态：E(t+1) = E(t) + η*P_ch*dt - P_dis/η*dt
    #    => E(t+1) - E(t) - η*dt*P_ch + (1/η)*dt*P_dis = 0
    A_dyn = np.zeros((n, n_vars))
    for t in range(n):
        A_dyn[t, i_e + t + 1] = 1.0
        A_dyn[t, i_e + t] = -1.0
        A_dyn[t, i_ch + t] = -eta * dt
        A_dyn[t, i_dis + t] = (1.0 / eta) * dt
    constraints.append(LinearConstraint(A_dyn, lb=0.0, ub=0.0))

    # 3. 充放电互斥
    # P_ch(t) <= 5000 * u(t)  =>  P_ch - 5000*u <= 0
    A_ch = np.zeros((n, n_vars))
    for t in range(n):
        A_ch[t, i_ch + t] = 1.0
        A_ch[t, i_u + t] = -p_max
    constraints.append(LinearConstraint(A_ch, lb=-np.inf, ub=0.0))

    # P_dis(t) <= 5000 * (1 - u(t))  =>  P_dis + 5000*u <= 5000
    A_dis = np.zeros((n, n_vars))
    for t in range(n):
        A_dis[t, i_dis + t] = 1.0
        A_dis[t, i_u + t] = p_max
    constraints.append(LinearConstraint(A_dis, lb=-np.inf, ub=p_max))

    # 4. 日循环：E(0) = E(n) = e0
    A_e0 = np.zeros((1, n_vars))
    A_e0[0, i_e] = 1.0
    constraints.append(LinearConstraint(A_e0, lb=e0, ub=e0))

    A_en = np.zeros((1, n_vars))
    A_en[0, i_e + n] = 1.0
    constraints.append(LinearConstraint(A_en, lb=e0, ub=e0))

    bounds = Bounds(lb, ub)

    if verbose:
        print(f"变量数: {n_vars} (连续 {n_vars - n}, 整数/二进制 {n})")
        print("开始调用 scipy.optimize.milp ...")

    res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

    if not res.success:
        raise RuntimeError(f"MILP 求解失败: {res.message}")

    x = res.x
    P_grid = x[i_grid:i_grid + n]
    P_ch = x[i_ch:i_ch + n]
    P_dis = x[i_dis:i_dis + n]
    E = x[i_e:i_e + n + 1]
    u = x[i_u:i_u + n]

    # 计算能量量（kWh）
    E_grid_kWh = P_grid * dt
    E_ch_kWh = P_ch * dt
    E_dis_kWh = P_dis * dt
    total_cost = float(res.fun)
    total_grid_kWh = float(E_grid_kWh.sum())

    return {
        "success": True,
        "status": res.status,
        "message": res.message,
        "P_grid_kW": P_grid,
        "P_ch_kW": P_ch,
        "P_dis_kW": P_dis,
        "E_kWh": E,
        "u": u,
        "E_grid_kWh": E_grid_kWh,
        "E_ch_kWh": E_ch_kWh,
        "E_dis_kWh": E_dis_kWh,
        "total_cost": total_cost,
        "total_grid_kWh": total_grid_kWh,
    }


def verify_solution(sol, price, load, pv,
                    eta=STORAGE_EFFICIENCY,
                    e0=STORAGE_INITIAL_SOC,
                    dt=HOURS_PER_SLOT,
                    tol=1e-4):
    """校验解是否满足所有约束，返回 True/False 并打印统计。"""
    n = SLOTS_PER_DAY
    P_grid = sol["P_grid_kW"]
    P_ch = sol["P_ch_kW"]
    P_dis = sol["P_dis_kW"]
    E = sol["E_kWh"]

    ok = True
    # 功率平衡
    balance_err = np.abs(P_grid + pv + P_dis - load - P_ch)
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

    # 初末 SOC
    if abs(E[0] - e0) > tol or abs(E[-1] - e0) > tol:
        print(f"[警告] 初末 SOC 不为 {e0}: E[0]={E[0]:.2f}, E[-1]={E[-1]:.2f}")
        ok = False
    else:
        print(f"[OK] E[0]=E[144]={e0:.2f} kWh")

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
    print(f"  全天购电量: {sol['total_grid_kWh']:.2f} kWh")
    print(f"  全天购电费: {sol['total_cost']:.2f} 元")
    print(f"  总充电量: {sol['E_ch_kWh'].sum():.2f} kWh")
    print(f"  总放电量: {sol['E_dis_kWh'].sum():.2f} kWh")
    net_stored = eta * sol['E_ch_kWh'].sum() - sol['E_dis_kWh'].sum() / eta
    print(f"  储能实际净变化: {net_stored:.2f} kWh (应为 0)")

    return ok


def solve_problem1(attachment1_path: Path = None, verbose: bool = False):
    """读取附件 1 并求解问题 1。"""
    if attachment1_path is None:
        attachment1_path = DATA_PROCESSED / "attachment1.csv"
    df = pd.read_csv(attachment1_path)
    df = df.sort_values("slot").reset_index(drop=True)

    price = df["price"].values
    load = df["load"].values
    pv = df["pv_forecast"].values
    time_str = df["time_str"].values

    sol = build_and_solve(price, load, pv, verbose=verbose)
    sol["time_str"] = time_str
    sol["slot"] = df["slot"].values
    sol["price"] = price
    sol["load"] = load
    sol["pv"] = pv
    return sol


if __name__ == "__main__":
    sol = solve_problem1(verbose=True)
    print("\n=== 问题 1 求解结果校验 ===")
    verify_solution(sol, sol["price"], sol["load"], sol["pv"])
