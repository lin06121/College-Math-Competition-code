"""
问题 2 的两阶段随机规划核心求解模块。

模型（每天 0:00 制定计划，当天实时运行）：

第一阶段（日前，0:00）
    决策：计划购电功率 P_plan(t)，t = 1..144。
    目标：购买计划电量的确定费用 + 5 倍紧急购电费用的**期望值**。
    信息：只能使用附件 2 中当天之前的历史数据做预测。

第二阶段（实时，recourse）
    信息：当天实际负载 / 光伏（情景已实现）。
    决策：储能充放电 P_ch(t)、P_dis(t)，以及紧急购电 P_em(t)。
    目标：最小化 5 倍紧急购电费用。

两阶段随机规划的数学形式：

    min_{P_plan}  Σ_t c(t)·P_plan(t)·Δt
                  + Σ_s π_s · Q(P_plan, ξ_s)

    Q(P_plan, ξ_s) = min Σ_t 5·c(t)·P_em_s(t)·Δt
        s.t. P_plan(t) + P_em_s(t) + PV_s(t) + P_dis_s(t)
                 = Load_s(t) + P_ch_s(t) + P_curt_s(t)
             E_s(t+1) = E_s(t) + η·P_ch_s(t)·Δt − P_dis_s(t)/η·Δt
             1200 ≤ E_s(t) ≤ 10800,  0 ≤ P_ch_s, P_dis_s ≤ 5000
             E_s(0) = e0

其中 ξ_s 为第 s 个净负载情景（由历史预测误差生成），π_s 为其概率。
P_curt 为自由弃电（零成本），保证模型可行。

由于 P_ch/P_dis 在最优解中不会同时为正（η<1 时同时充放只会浪费能量），
模型整体退化为**线性规划**，无需引入 0-1 变量即可求得最优解。
"""
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

from config import (
    SLOTS_PER_DAY,
    HOURS_PER_SLOT,
    STORAGE_MAX_POWER,
    STORAGE_EFFICIENCY,
    STORAGE_SOC_MIN,
    STORAGE_SOC_MAX,
)


# ----------------------------------------------------------------------------
# 1. 历史预测误差 -> 情景生成
# ----------------------------------------------------------------------------

def compute_net_forecast_errors(dates, net_actual, k: int = 4):
    """
    对每个日期，用"过去 k 个同星期几净负载曲线平均"做预测，
    返回预测误差矩阵 error[i] = net_actual[i] - net_forecast[i]。

    对历史样本不足的日期返回 np.nan 行。
    """
    dates = pd.to_datetime(dates)
    n_days = len(dates)
    errors = np.full((n_days, SLOTS_PER_DAY), np.nan)
    forecasts = np.full((n_days, SLOTS_PER_DAY), np.nan)
    dow = np.array([d.dayofweek for d in dates])

    for i in range(n_days):
        hist_idx = np.where((np.arange(n_days) < i) & (dow == dow[i]))[0]
        if len(hist_idx) == 0:
            continue
        use = hist_idx[-k:]
        fc = net_actual[use].mean(axis=0)
        forecasts[i] = fc
        errors[i] = net_actual[i] - fc

    return forecasts, errors


def build_scenarios(target_idx, errors, dates, m: int = 10):
    """
    为目标日期收集最近 m 个可用的历史误差曲线作为情景。

    返回
    ----------
    scen_idx : ndarray, 历史日期下标（按时间升序）
    probs : ndarray, 等概率
    """
    dates = pd.to_datetime(dates)
    valid = np.where(
        (np.arange(len(dates)) < target_idx)
        & (~np.isnan(errors).any(axis=1))
    )[0]
    if len(valid) == 0:
        raise ValueError("没有可用的历史误差样本")
    use = valid[-m:]
    probs = np.full(len(use), 1.0 / len(use))
    return use, probs


# ----------------------------------------------------------------------------
# 2. 单日两阶段随机 LP
# ----------------------------------------------------------------------------

def _build_day_lp(price, net_scenarios, probs, e0,
                  p_plan_fixed=None,
                  terminal_value=0.0,
                  dt=HOURS_PER_SLOT,
                  eta=STORAGE_EFFICIENCY,
                  p_max=STORAGE_MAX_POWER,
                  e_min=STORAGE_SOC_MIN,
                  e_max=STORAGE_SOC_MAX,
                  p_plan_ub=20000.0,
                  emergency_ub=20000.0):
    """
    构建单日两阶段随机规划（或实时 recourse）的 LP。

    变量排布：
        [P_plan(n)] + 每个情景 [P_ch(n), P_dis(n), P_em(n), P_curt(n), E(n+1)]

    p_plan_fixed 给出时，P_plan 被固定（用于实时 recourse 阶段），
    且其目标系数置 0（计划费用已确定）。

    terminal_value : float
        终端储电量的边际价值 λ（元/kWh），目标中加入 −λ·E(144)，
        用于近似"把电留到明天"的跨日价值。λ=0 即自由终端。

    返回 (c, A, lb_ineq, ub_ineq, bounds) 。
    """
    n = SLOTS_PER_DAY
    S = len(net_scenarios)
    per_scen = 4 * n + (n + 1)          # P_ch, P_dis, P_em, P_curt, E
    nv = n + S * per_scen

    c = np.zeros(nv)

    # --- 目标函数 ---
    if p_plan_fixed is None:
        c[0:n] = price * dt                       # 计划购电费用
    for s in range(S):
        b = n + s * per_scen
        i_em = b + 2 * n
        c[i_em:i_em + n] = probs[s] * 5.0 * price * dt   # 期望紧急购电费用
        if terminal_value:
            # 终端储电量的跨日价值（按情景概率加权）
            c[b + 4 * n + n] -= probs[s] * terminal_value

    # --- 变量上下界 ---
    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)

    if p_plan_fixed is None:
        lb[0:n] = 0.0
        ub[0:n] = p_plan_ub
    else:
        lb[0:n] = p_plan_fixed
        ub[0:n] = p_plan_fixed

    for s in range(S):
        b = n + s * per_scen
        i_ch, i_dis, i_em, i_curt, i_e = b, b + n, b + 2 * n, b + 3 * n, b + 4 * n
        lb[i_ch:i_ch + n], ub[i_ch:i_ch + n] = 0.0, p_max
        lb[i_dis:i_dis + n], ub[i_dis:i_dis + n] = 0.0, p_max
        lb[i_em:i_em + n], ub[i_em:i_em + n] = 0.0, emergency_ub
        lb[i_curt:i_curt + n], ub[i_curt:i_curt + n] = 0.0, np.inf
        lb[i_e:i_e + n + 1], ub[i_e:i_e + n + 1] = e_min, e_max

    # --- 约束（稀疏） ---
    rows, cols, vals = [], [], []
    lb_ineq, ub_ineq = [], []
    r = 0

    def add(row_idx, col_idx, value):
        rows.append(row_idx)
        cols.append(col_idx)
        vals.append(value)

    for s in range(S):
        b = n + s * per_scen
        i_ch, i_dis, i_em, i_curt, i_e = b, b + n, b + 2 * n, b + 3 * n, b + 4 * n
        net = net_scenarios[s]

        # 功率平衡：P_plan + P_em - P_ch + P_dis - P_curt = net
        for t in range(n):
            add(r, t, 1.0)
            add(r, i_em + t, 1.0)
            add(r, i_ch + t, -1.0)
            add(r, i_dis + t, 1.0)
            add(r, i_curt + t, -1.0)
            lb_ineq.append(net[t])
            ub_ineq.append(net[t])
            r += 1

        # 储能动态：E(t+1) - E(t) - η·Δt·P_ch + (Δt/η)·P_dis = 0
        for t in range(n):
            add(r, i_e + t + 1, 1.0)
            add(r, i_e + t, -1.0)
            add(r, i_ch + t, -eta * dt)
            add(r, i_dis + t, (dt / eta))
            lb_ineq.append(0.0)
            ub_ineq.append(0.0)
            r += 1

        # 初始 SOC
        add(r, i_e, 1.0)
        lb_ineq.append(e0)
        ub_ineq.append(e0)
        r += 1

    A = csr_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(r, nv))
    return c, A, np.array(lb_ineq), np.array(ub_ineq), Bounds(lb, ub)


def _extract_solution(res, S, price, dt):
    """把 milp 解拆成结构化字段。"""
    n = SLOTS_PER_DAY
    per_scen = 4 * n + (n + 1)
    x = res.x

    P_plan = x[0:n]
    scen = []
    for s in range(S):
        b = n + s * per_scen
        scen.append({
            "P_ch_kW": x[b:b + n],
            "P_dis_kW": x[b + n:b + 2 * n],
            "P_em_kW": x[b + 2 * n:b + 3 * n],
            "P_curt_kW": x[b + 3 * n:b + 4 * n],
            "E_kWh": x[b + 4 * n:b + 5 * n + 1],
        })
    return P_plan, scen


def solve_stochastic_day(price, net_forecast, scen_errors, probs, e0,
                         terminal_value=0.0, dt=HOURS_PER_SLOT, verbose=False):
    """
    求解单日两阶段随机规划，返回第一阶段决策 P_plan 与各情景第二阶段解。
    """
    net_scenarios = [net_forecast + e for e in scen_errors]
    c, A, lb_ineq, ub_ineq, bounds = _build_day_lp(
        price, net_scenarios, probs, e0, p_plan_fixed=None,
        terminal_value=terminal_value, dt=dt)

    res = milp(c=c, constraints=LinearConstraint(A, lb_ineq, ub_ineq),
               bounds=bounds)
    if not res.success:
        raise RuntimeError(f"随机规划求解失败: {res.message}")

    P_plan, scen = _extract_solution(res, len(net_scenarios), price, dt)
    if verbose:
        print(f"  随机规划：{len(net_scenarios)} 情景，目标值 {res.fun:.2f} 元")
    return P_plan, scen, float(res.fun)


def solve_recourse_day(price, net_actual, P_plan, e0,
                       terminal_value=0.0, dt=HOURS_PER_SLOT, verbose=False):
    """
    实时 recourse：给定计划购电 P_plan 与当日实际净负载，优化储能充放电，
    最小化 5 倍紧急购电费用。返回实际充放电量、紧急购电量与终端 SOC。
    """
    c, A, lb_ineq, ub_ineq, bounds = _build_day_lp(
        price, [net_actual], np.array([1.0]), e0,
        p_plan_fixed=P_plan, terminal_value=terminal_value, dt=dt)

    res = milp(c=c, constraints=LinearConstraint(A, lb_ineq, ub_ineq),
               bounds=bounds)
    if not res.success:
        raise RuntimeError(f"实时 recourse 求解失败: {res.message}")

    _, scen = _extract_solution(res, 1, price, dt)
    s0 = scen[0]
    s0["em_cost"] = float((s0["P_em_kW"] * price).sum() * dt * 5.0)
    if verbose:
        print(f"  实时 recourse：紧急购电费 {s0['em_cost']:.2f} 元")
    return s0


def evaluate_plan_realization(price, net_actual, P_plan, e0, dt=HOURS_PER_SLOT):
    """
    在给定计划 P_plan 下，用一组实际净负载评估（实时储能最优调度）。
    返回 (P_ch, P_dis, P_em, E, em_cost)。
    """
    s0 = solve_recourse_day(price, net_actual, P_plan, e0, dt=dt)
    return (s0["P_ch_kW"], s0["P_dis_kW"], s0["P_em_kW"],
            s0["E_kWh"], s0["em_cost"])
