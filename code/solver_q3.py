"""
问题 3 求解器：多预报时刻滚动调整购电策略。

信息结构（附件 3 每天 0:00 / 6:00 / 12:00 / 18:00 发布未来 24 小时整点光伏预报）
--------------------------------------------------------------------------------
0:00  —— 用 0:00 光伏预报 + 负载历史预测（含报童裕度），制定全天计划 P_plan；
        slot 0..35 随即按实际负载/光伏因果执行（储能兜底），得到 slot 36 实际储电量。
6:00  —— 用 6:00 光伏预报，对 slot 36..143 重新优化调整购电量 P_adj；
        slot 36..71 按实际值因果执行，得到 slot 72 实际储电量。
12:00 —— 用 12:00 光伏预报，对 slot 72..143 再优化；执行 slot 72..107。
18:00 —— 用 18:00 光伏预报，对 slot 108..143 再优化；执行 slot 108..143。

关键规则：每一次调整都以 **0:00 的计划购电量 P_plan 为基准**衡量偏差（题面：
计划量高于调整量的部分按当时电价的 50% 计违约，调整量高于计划量的超出部分按
当时电价的 1.5 倍计）。字面口径下只按实际购入量付全价，偏差部分少买按 50%、
多买按 150% 计价，因此调整优化的目标（去掉常数项 c·P_plan）是最小化

    Σ_t [ 1.5·c(t)·P_pos(t) − 0.5·c(t)·P_neg(t) ]·Δt ,
    P_adj(t) = P_plan(t) + P_pos(t) − P_neg(t) ,

费用（slot t）：
    实际购电费  c(t)·P_adj(t)·Δt
    违约（少买）0.5·c(t)·max(P_plan−P_adj, 0)·Δt
    超额（多买）0.5·c(t)·max(P_adj−P_plan, 0)·Δt   （合计单价 1.5 倍）
    紧急购电费  5·c(t)·P_em(t)·Δt

result3.xlsx 输出：
  计划购电量：0:00 制定的 P_plan（全天 144 时段）
  调整购电量：最终执行的 P_adj（分段由各调整时刻决定）
  充放电量：  实际执行的 P_ch / P_dis（因果结算）
  紧急购电量：实际 P_em
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds

from config import (
    DATA_PROCESSED, SLOTS_PER_DAY, HOURS_PER_SLOT,
    STORAGE_INITIAL_SOC, STORAGE_MAX_POWER, STORAGE_EFFICIENCY,
    STORAGE_SOC_MIN, STORAGE_SOC_MAX, KEY_DATES,
    FORECAST_ISSUE_TIMES, FORECAST_ISSUE_SLOTS,
)
from forecast_q2 import load_historical_matrices
from forecast_q3 import build_forecast_matrix, debias_forecast_matrix
from forecast_q4 import load_price_realtime, build_price_forecast
from method_q2 import (
    make_class_array, forecast_mate, quantile_margin,
    compute_residuals_mate, causal_dispatch,
)
from daily_solver import build_and_solve

ETA = STORAGE_EFFICIENCY
PMAX = STORAGE_MAX_POWER
EMIN, EMAX = STORAGE_SOC_MIN, STORAGE_SOC_MAX

ISSUE_TIMES = list(FORECAST_ISSUE_TIMES)
ISSUE_SLOTS = dict(FORECAST_ISSUE_SLOTS)


# ----------------------------------------------------------------------------
# 1. 调整阶段优化（滑动窗口 MILP）
# ----------------------------------------------------------------------------

def _adjustment_lp(price, net_f, P_plan, e0, start_slot,
                   eta=ETA, p_max=PMAX, e_min=EMIN, e_max=EMAX,
                   dt=HOURS_PER_SLOT, term_value=0.0, p_adj_max=20000.0,
                   over_coef=1.5, under_coef=-0.5):
    """
    对 slot start_slot..143 重新优化购电量调整量。

    P_adj = P_plan + P_pos − P_neg（基准恒为 0:00 的计划 P_plan）

    结算口径（字面读法）：
        费用(t) = c·P_adj + 0.5c·|P_adj − P_plan| + 5c·P_em
    代入 P_adj 并去掉常数项 c·P_plan：
        min Σ [ 1.5·c·P_pos − 0.5·c·P_neg ]·Δt
    （少买一节电净省 0.5c，故 P_neg 的目标系数为负。）

    约束：功率平衡 P_pos − P_neg − P_ch + P_dis − P_curt = net_f − P_plan
          储能动态、SOC 边界、充放电互斥、初始 SOC = e0、0 ≤ P_adj ≤ p_adj_max

    net_f : 调整时刻可得的最新净负载预测（负载预测 − 该时刻光伏预报），(144,)
    term_value : 终端储电量价值 λ（元/kWh），目标中加 −λ·E(144)；0 表示不估值。
    返回 (P_adj, P_ch, P_dis, P_curt, E)，均为全天 144 长度（start_slot 之前
    的时段对 P_adj 填 P_plan，对其余量填 0 / NaN）。
    """
    n_total = SLOTS_PER_DAY
    n = n_total - start_slot

    # 变量：[P_pos, P_neg, P_ch, P_dis, P_curt, E, u]
    i_pos = 0
    i_neg = n
    i_ch = 2 * n
    i_dis = 3 * n
    i_curt = 4 * n
    i_e = 5 * n
    i_u = 5 * n + (n + 1)
    nv = 6 * n + (n + 1)

    c = np.zeros(nv)
    for j, t in enumerate(range(start_slot, n_total)):
        c[i_pos + j] = over_coef * price[t] * dt
        c[i_neg + j] = under_coef * price[t] * dt
    if term_value:
        c[i_e + n] = -term_value          # −λ·E(144)

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)
    ub[i_ch:i_ch + n] = p_max
    ub[i_dis:i_dis + n] = p_max
    lb[i_e:i_e + n + 1] = e_min
    ub[i_e:i_e + n + 1] = e_max
    ub[i_u:i_u + n] = 1.0

    integrality = np.zeros(nv, dtype=int)
    integrality[i_u:i_u + n] = 1

    constraints = []

    # 功率平衡：P_pos − P_neg − P_ch + P_dis − P_curt = net_f − P_plan
    A_bal = np.zeros((n, nv))
    for j in range(n):
        A_bal[j, i_pos + j] = 1.0
        A_bal[j, i_neg + j] = -1.0
        A_bal[j, i_ch + j] = -1.0
        A_bal[j, i_dis + j] = 1.0
        A_bal[j, i_curt + j] = -1.0
    b_bal = net_f[start_slot:] - P_plan[start_slot:]
    constraints.append(LinearConstraint(A_bal, lb=b_bal, ub=b_bal))

    # 调整量边界：0 ≤ P_adj = P_plan + P_pos − P_neg ≤ p_adj_max
    A_adj = np.zeros((n, nv))
    for j in range(n):
        A_adj[j, i_pos + j] = 1.0
        A_adj[j, i_neg + j] = -1.0
    adj_lo = -P_plan[start_slot:]
    adj_hi = p_adj_max - P_plan[start_slot:]
    constraints.append(LinearConstraint(A_adj, lb=adj_lo, ub=adj_hi))

    # 储能动态：E(j+1) − E(j) − η·P_ch·Δt + P_dis/η·Δt = 0
    A_dyn = np.zeros((n, nv))
    for j in range(n):
        A_dyn[j, i_e + j + 1] = 1.0
        A_dyn[j, i_e + j] = -1.0
        A_dyn[j, i_ch + j] = -eta * dt
        A_dyn[j, i_dis + j] = (1.0 / eta) * dt
    constraints.append(LinearConstraint(A_dyn, lb=0.0, ub=0.0))

    # 初始 SOC
    A_e0 = np.zeros((1, nv))
    A_e0[0, i_e] = 1.0
    constraints.append(LinearConstraint(A_e0, lb=e0, ub=e0))

    # 充放电互斥：P_ch ≤ p_max·u ； P_dis ≤ p_max·(1−u)
    A_ch = np.zeros((n, nv))
    A_dis = np.zeros((n, nv))
    for j in range(n):
        A_ch[j, i_ch + j] = 1.0
        A_ch[j, i_u + j] = -p_max
        A_dis[j, i_dis + j] = 1.0
        A_dis[j, i_u + j] = p_max
    constraints.append(LinearConstraint(A_ch, lb=-np.inf, ub=0.0))
    constraints.append(LinearConstraint(A_dis, lb=-np.inf, ub=p_max))

    res = milp(c=c, constraints=constraints,
               bounds=Bounds(lb, ub), integrality=integrality)
    if not res.success:
        raise RuntimeError(f"调整阶段 MILP 求解失败: {res.message}")

    x = res.x
    P_adj = P_plan.copy()
    P_adj[start_slot:] = P_plan[start_slot:] + x[i_pos:i_pos + n] - x[i_neg:i_neg + n]

    P_ch = np.zeros(n_total)
    P_ch[start_slot:] = x[i_ch:i_ch + n]
    P_dis = np.zeros(n_total)
    P_dis[start_slot:] = x[i_dis:i_dis + n]
    P_curt = np.zeros(n_total)
    P_curt[start_slot:] = x[i_curt:i_curt + n]
    E = np.full(n_total + 1, np.nan)
    E[start_slot:] = x[i_e:i_e + n + 1]

    return P_adj, P_ch, P_dis, P_curt, E


# ----------------------------------------------------------------------------
# 2. 单日模拟
# ----------------------------------------------------------------------------

def solve_day_q3(price, load_actual, pv_actual, pv_fc, load_for_plan, e0,
                 issue_times=ISSUE_TIMES, use_actual_for_plan=False,
                 return_snapshots=False, term_value=0.0,
                 over_coef=1.5, under_coef=-0.5):
    """
    模拟问题 3 的一天（滚动调整 + 实际执行）。

    参数
    ----------
    price        : (144,) 电价（附件 1 典型日）
    load_actual  : (144,) 实际负载 kW
    pv_actual    : (144,) 实际光伏 kW
    pv_fc        : dict {发布时刻: (144,) 光伏预报}，来自附件 3
    load_for_plan: (144,) 负载预测（已含报童裕度）
    e0           : 当日 0:00 储电量 kWh
    use_actual_for_plan : 若 True，0:00 计划直接用实际负载（仅 oracle 对照用）

    返回
    ----------
    dict，含 P_plan_kW, P_adj_kW, P_ch_kW, P_dis_kW, P_em_kW, P_curt_kW,
    E_kWh, load_forecast_kW, pv_forecast_kW(0:00), snapshots(可选)
    """
    n = SLOTS_PER_DAY
    net_actual = load_actual - pv_actual
    plan_load = load_actual if use_actual_for_plan else load_for_plan

    # ---- 0:00 计划（全天）----
    sol0 = build_and_solve(price, plan_load, pv_fc["0:00"], e0=e0,
                           cyclic=False, allow_emergency=False,
                           term_value=term_value, verbose=False)
    P_plan = sol0["P_plan_kW"]

    # ---- 分段滚动：调整 → 执行 ----
    P_adj = P_plan.copy()
    P_ch = np.zeros(n)
    P_dis = np.zeros(n)
    P_em = np.zeros(n)
    P_curt = np.zeros(n)
    E = np.zeros(n + 1)
    E[0] = e0
    e_run = e0

    boundaries = [ISSUE_SLOTS[it] for it in issue_times] + [n]
    snapshots = {"0:00": P_plan.copy()} if return_snapshots else None

    for k, it in enumerate(issue_times):
        s = ISSUE_SLOTS[it]
        if k > 0:
            # 调整：以 0:00 计划为基准，用该时刻最新光伏预报重优化剩余时段
            net_f = plan_load - pv_fc[it]
            P_adj_new, _, _, _, _ = _adjustment_lp(
                price, net_f, P_plan, e_run, s, term_value=term_value,
                over_coef=over_coef, under_coef=under_coef)
            P_adj[s:] = P_adj_new[s:]
            if return_snapshots:
                snapshots[it] = P_adj.copy()

        # 执行本段：只用当前实测，因果兜底
        e_idx = boundaries[k + 1]
        if e_idx > s:
            ch, dis, em, Eseg, curt = causal_dispatch(
                net_actual[s:e_idx], P_adj[s:e_idx], e_run)
            P_ch[s:e_idx] = ch
            P_dis[s:e_idx] = dis
            P_em[s:e_idx] = em
            P_curt[s:e_idx] = curt
            E[s:e_idx + 1] = Eseg
            e_run = Eseg[-1]
    E[0] = e0

    out = {
        "P_plan_kW": P_plan,
        "P_adj_kW": P_adj,
        "P_ch_kW": P_ch,
        "P_dis_kW": P_dis,
        "P_em_kW": P_em,
        "P_curt_kW": P_curt,
        "E_kWh": E,
        "load_forecast_kW": plan_load,
        "pv_forecast_kW": pv_fc["0:00"],
    }
    if return_snapshots:
        out["snapshots"] = snapshots
    return out


# ----------------------------------------------------------------------------
# 3. 费用核算
# ----------------------------------------------------------------------------

def compute_costs(price, P_plan, P_adj, P_em, dt=HOURS_PER_SLOT):
    """
    按题面结算规则计算问题 3 各项费用（元）。

    字面口径：只按实际购入量付全价，偏差部分少买按 50%、多买按 150% 计价：

        费用(t) = c·P_adj
                + 0.5·c·max(P_plan − P_adj, 0)   # 计划高于调整：少买，违约电价 50%
                + 0.5·c·max(P_adj − P_plan, 0)   # 调整高于计划：多买部分总价 1.5 倍
                + 5·c·P_em

    plan_cost（Σc·P_plan）仅作参考，不进入总费用。
    """
    adj_purchase_cost = float((P_adj * price * dt).sum())
    under = np.maximum(P_plan - P_adj, 0.0)
    over = np.maximum(P_adj - P_plan, 0.0)
    breach_cost = float((0.5 * price * under * dt).sum())
    uplift_cost = float((0.5 * price * over * dt).sum())
    em_cost = float((P_em * price * 5.0 * dt).sum())
    return {
        "plan_cost": float((P_plan * price * dt).sum()),
        "adj_purchase_cost": adj_purchase_cost,
        "breach_cost": breach_cost,
        "uplift_cost": uplift_cost,
        "adj_cost": breach_cost + uplift_cost,
        "em_cost": em_cost,
        "total_cost": adj_purchase_cost + breach_cost + uplift_cost + em_cost,
        "plan_kWh": float(P_plan.sum() * dt),
        "adj_kWh": float(P_adj.sum() * dt),
        "dev_kWh": float((under.sum() + over.sum()) * dt),
        "over_kWh": float(over.sum() * dt),
        "under_kWh": float(under.sum() * dt),
        "em_kWh": float(P_em.sum() * dt),
        "curt_kWh": 0.0,
    }


# ----------------------------------------------------------------------------
# 4. 全年滚动模拟
# ----------------------------------------------------------------------------

def simulate_year_q3(dates, load_mat, pv_mat, price_mat, fc, cls_arr, err,
                     d2i, f2i, start_date, end_date, e0,
                     q=0.8, issue_times=ISSUE_TIMES, verbose=False,
                     term_value=0.0, over_coef=1.5, under_coef=-0.5,
                     debias=False, debias_window=20,
                     price_settle_mat=None):
    """从 start_date 到 end_date 逐日滚动，返回 (result_df, summary_df)。

    price_mat : 计划用价格矩阵 (N,144)。问题 4 未知电价时传**电价预报**矩阵。
    price_settle_mat : 结算用实际价格矩阵 (N,144)。None 时用 price_mat。
    """
    tds = pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")
    records, daily = [], []
    s_mat = (None if price_settle_mat is None else np.asarray(price_settle_mat))

    pv_all = (debias_forecast_matrix(fc, pv_mat, dates, d2i, debias_window)
              if debias else fc["pv_forecast"])

    for td in tds:
        i = d2i[td]
        fi = f2i[td]
        load_a, pv_a = load_mat[i], pv_mat[i]
        price = price_mat[i]                        # 计划用（预报）价
        sp = s_mat[i] if s_mat is not None else price   # 结算用（实际）价

        load_f, _ = forecast_mate(i, dates, load_mat, pv_mat, cls_arr)
        margin = quantile_margin(err, i, q) if q is not None else np.zeros(SLOTS_PER_DAY)
        load_for_plan = load_f + margin

        pv_fc = {it: pv_all[fi, fc["issue_times"].index(it)]
                 for it in fc["issue_times"]}

        sol = solve_day_q3(price, load_a, pv_a, pv_fc, load_for_plan, e0,
                           issue_times=issue_times, term_value=term_value,
                           over_coef=over_coef, under_coef=under_coef)
        costs = compute_costs(sp, sol["P_plan_kW"], sol["P_adj_kW"], sol["P_em_kW"])
        costs["curt_kWh"] = float(sol["P_curt_kW"].sum() * HOURS_PER_SLOT)

        for t in range(SLOTS_PER_DAY):
            records.append({
                "date": td, "slot": t,
                "P_plan_kW": sol["P_plan_kW"][t],
                "P_adj_kW": sol["P_adj_kW"][t],
                "P_ch_kW": sol["P_ch_kW"][t],
                "P_dis_kW": sol["P_dis_kW"][t],
                "P_em_kW": sol["P_em_kW"][t],
                "P_curt_kW": sol["P_curt_kW"][t],
                "E_kWh": sol["E_kWh"][t],
                "load_forecast_kW": sol["load_forecast_kW"][t],
                "pv_forecast_kW": sol["pv_forecast_kW"][t],
                "load_actual_kW": load_a[t],
                "pv_actual_kW": pv_a[t],
                "price": sp[t],
            })

        daily.append({"date": td, "e0": e0, "e_end": sol["E_kWh"][-1], **costs})
        e0 = sol["E_kWh"][-1]

    return pd.DataFrame(records), pd.DataFrame(daily)


# ----------------------------------------------------------------------------
# 5. 主流程
# ----------------------------------------------------------------------------

def load_problem3_data():
    """一次性加载问题 3 所需的全部静态数据。"""
    dates, load_mat, pv_mat, price_mat = load_historical_matrices()
    fc = build_forecast_matrix()
    d2i = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    fdates = [pd.Timestamp(d) for d in fc["dates"]]
    fpos = {d: i for i, d in enumerate(fdates)}
    f2i = {d: fpos[d] for d in dates if d in fpos}
    cls_arr = make_class_array(dates)
    net_mat = load_mat - pv_mat
    err = compute_residuals_mate(dates, load_mat, pv_mat, net_mat, cls_arr)
    return dates, load_mat, pv_mat, price_mat, fc, cls_arr, err, d2i, f2i, net_mat


def _trim(df, keep):
    return df[df["date"] >= pd.Timestamp(keep)].reset_index(drop=True)


def summarize(summary_df, tag=""):
    """打印汇总。"""
    s = lambda col: summary_df[col].sum()
    print(f"\n{tag}（{len(summary_df)} 天）")
    print(f"  实际购电费 {s('adj_purchase_cost'):>14,.2f} 元")
    print(f"  少买违约费 {s('breach_cost'):>14,.2f} 元")
    print(f"  多买超额费 {s('uplift_cost'):>14,.2f} 元")
    print(f"  紧急购电费 {s('em_cost'):>14,.2f} 元")
    print(f"  总费用     {s('total_cost'):>14,.2f} 元")
    print(f"  实际购电量 {s('adj_kWh'):>14,.2f} kWh  （计划 {s('plan_kWh'):,.2f}）")
    print(f"  偏离量     {s('dev_kWh'):>14,.2f} kWh  "
          f"（少买 {s('under_kWh'):,.2f} / 多买 {s('over_kWh'):,.2f}）")
    print(f"  紧急购电量 {s('em_kWh'):>14,.2f} kWh")


def calibrate_q_q3(data, cal_start="2025-01-08", cal_end="2025-01-31",
                   warm_start="2025-01-01",
                   q_grid=(0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9),
                   lam_grid=(0.0, 0.3, 0.4, 0.45, 0.5),
                   issue_times=ISSUE_TIMES, verbose=True,
                   debias=False, debias_window=20,
                   price_settle_mat=None):
    """
    在标定窗口上**联合**网格搜索 (报童分位 q, 终端储能价值 λ)。

    λ 是日终储电量 E(144) 的影子价格（元/kWh）：给目标加 −λ·E(144)。
    λ 过小时储能机会成本被忽略、日终被放空；过大则退化为"日终充满"的病态解，
    故必须标定（本模型阈值 ≈0.425，平台 [0.43, 0.46]）。

    返回 (best_q, best_lam, 明细 DataFrame)。
    """
    dates, load_mat, pv_mat, price_mat, fc, cls_arr, err, d2i, f2i, net_mat = data
    best_q = best_lam = None
    best_cost, table = np.inf, []
    for lam in lam_grid:
        for q in q_grid:
            _, s = simulate_year_q3(
                dates, load_mat, pv_mat, price_mat, fc, cls_arr, err, d2i, f2i,
                warm_start, cal_end, STORAGE_INITIAL_SOC, q=q,
                issue_times=issue_times, term_value=lam,
                debias=debias, debias_window=debias_window,
                price_settle_mat=price_settle_mat)
            s = _trim(s, cal_start)
            cost = s["total_cost"].sum()
            table.append((lam, q, s["adj_purchase_cost"].sum(), s["adj_cost"].sum(),
                          s["em_cost"].sum(), cost))
            if verbose:
                print(f"    λ={lam:<5} q={q:<5} 购电 {s['adj_purchase_cost'].sum():>10,.0f}  "
                      f"调整 {s['adj_cost'].sum():>8,.0f}  紧急 {s['em_cost'].sum():>8,.0f}  "
                      f"总 {cost:>11,.0f} 元")
            if cost < best_cost:
                best_q, best_lam, best_cost = q, lam, cost
    col = ["lam", "q", "adj_purchase_cost", "adj_cost", "em_cost", "total_cost"]
    return best_q, best_lam, pd.DataFrame(table, columns=col)


def solve_problem3(start_date="2025-02-01", end_date="2025-12-31",
                   q=None, warm_start="2025-01-01",
                   issue_times=ISSUE_TIMES, verbose=True,
                   save_dir=DATA_PROCESSED, term_value=None, data=None,
                   debias=False, debias_window=20):
    """
    求解问题 3。

    q=None 或 term_value=None 时先在 1 月**联合标定** (报童分位 q, 终端储能价值 λ)，
    再以标定值跑全年（2025-01-01 预热），输出 start_date ~ end_date 的结果。
    debias=False（默认）直接使用附件 3 预报；置 True 可启用因果去偏
    （见 forecast_q3.debias_forecast_matrix），全年约再省 0.46%，但多一层后处理。
    """
    if data is None:
        data = load_problem3_data()
    dates, load_mat, pv_mat, price_mat, fc, cls_arr, err, d2i, f2i, net_mat = data

    if q is None or term_value is None:
        if verbose:
            print("在 2025-01-08 ~ 01-31 上联合标定 (q, λ)：")
        q_c, lam_c, _ = calibrate_q_q3(data, verbose=verbose,
                                       debias=debias, debias_window=debias_window)
        if q is None:
            q = q_c
        if term_value is None:
            term_value = lam_c
        if verbose:
            print(f"  -> 标定结果 q* = {q}，λ* = {term_value}")

    result_df, summary_df = simulate_year_q3(
        dates, load_mat, pv_mat, price_mat, fc, cls_arr, err, d2i, f2i,
        warm_start, end_date, STORAGE_INITIAL_SOC, q=q, issue_times=issue_times,
        term_value=term_value, debias=debias, debias_window=debias_window)

    result_df = _trim(result_df, start_date)
    summary_df = _trim(summary_df, start_date)

    if verbose:
        print(f"\n输出窗口 {start_date} ~ {end_date}，q={q}，λ={term_value}")
        summarize(summary_df)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_dir / "result3_details.csv", index=False)
        summary_df.to_csv(save_dir / "result3_summary.csv", index=False)

    return result_df, summary_df


def solve_problem4_3(start_date="2025-02-01", end_date="2025-12-31",
                     price_plan="forecast", forecast_window=14,
                     q=None, term_value=None, warm_start="2025-01-01",
                     issue_times=ISSUE_TIMES, verbose=True,
                     save_dir=DATA_PROCESSED, data=None, debias=False):
    """
    问题 4 子问：波动电价下重算问题 3（0:00 不知当日电价）。

    日前计划与各次调整都按**电价预报** price_plan 优化（预报=近 W 日逐时段均值），
    结算一律按附件 4 **实际电价**。q、λ 缺省时在 1 月联合标定。
    """
    if data is None:
        data = load_problem3_data()
    dates, load_mat, pv_mat, price_mat, fc, cls_arr, err, d2i, f2i, net_mat = data
    p_actual = load_price_realtime()

    if price_plan == "forecast":
        p_plan_mat = build_price_forecast(p_actual, price_mat, window=forecast_window)
    elif price_plan == "actual":
        p_plan_mat = p_actual
    elif price_plan == "typical":
        p_plan_mat = np.tile(price_mat, (len(dates), 1))
    else:
        raise ValueError(f"未知 price_plan: {price_plan}")

    data4 = (dates, load_mat, pv_mat, p_plan_mat, fc, cls_arr, err,
             d2i, f2i, net_mat)

    if q is None or term_value is None:
        if verbose:
            print(f"在 2025-01-08 ~ 01-31 上联合标定 (q, λ)"
                  f"（计划价={price_plan}，结算价=实际）：")
        q_c, lam_c, _ = calibrate_q_q3(data4, verbose=verbose, debias=debias,
                                       price_settle_mat=p_actual)
        if q is None:
            q = q_c
        if term_value is None:
            term_value = lam_c
        if verbose:
            print(f"  -> 标定结果 q*={q}，λ*={term_value}")

    result_df, summary_df = simulate_year_q3(
        *data4[:9], warm_start, end_date, STORAGE_INITIAL_SOC, q=q,
        issue_times=issue_times, term_value=term_value, debias=debias,
        price_settle_mat=p_actual)

    result_df = result_df[result_df["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
    summary_df = summary_df[summary_df["date"] >= pd.Timestamp(start_date)].reset_index(drop=True)

    if verbose:
        print(f"\n输出窗口 {start_date} ~ {end_date}，q={q}，λ={term_value}")
        summarize(summary_df)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(save_dir / "result43_details.csv", index=False)
        summary_df.to_csv(save_dir / "result43_summary.csv", index=False)

    summary_df.attrs["q"] = q
    summary_df.attrs["lambda"] = term_value
    return result_df, summary_df


if __name__ == "__main__":
    import time
    t0 = time.time()
    solve_problem3(verbose=True)
    print(f"\n耗时 {time.time() - t0:.1f} s")
