"""
问题 2 采用的日前计划与实时结算方法。

日前计划依据：
  负载 —— 按“周五/周六 vs 其余”二分类，取最近 K_load 个同类日实测均值；
  光伏 —— 取最近 K_pv 个连续日实测均值（光伏受天气驱动，日间持续性远强于周内周期性）；
  裕度 —— 净需求残差的 q 分位，q 由报童临界分位数标定。
实时结算：
  因果（online）控制规则，每时段只用当前实测净负载与当前储电量，不使用任何未来实测。
"""
import numpy as np
import pandas as pd

from config import (
    SLOTS_PER_DAY, HOURS_PER_SLOT,
    STORAGE_MAX_POWER, STORAGE_EFFICIENCY, STORAGE_SOC_MIN, STORAGE_SOC_MAX,
    DATA_PROCESSED,
)

CBAR = STORAGE_MAX_POWER * HOURS_PER_SLOT      # 每时段最大充/放电量 kWh (833.33)
ETA = STORAGE_EFFICIENCY
EMIN, EMAX = STORAGE_SOC_MIN, STORAGE_SOC_MAX

_FALLBACK = {}


# ----------------------------------------------------------------------------
# 1. 预测依据
# ----------------------------------------------------------------------------

def make_class_array(dates):
    """周内二分类：周五(4)、周六(5) 记为 0（低负荷），其余记为 1（高负荷）。"""
    dts = pd.to_datetime(dates)
    return np.array([0 if d.dayofweek in (4, 5) else 1 for d in dts])


def load_fallback_profile(attachment1_path=None):
    """首日无历史时退化为附件 1 的典型日曲线。"""
    if "arr" not in _FALLBACK:
        if attachment1_path is None:
            attachment1_path = DATA_PROCESSED / "attachment1.csv"
        df = pd.read_csv(attachment1_path).sort_values("slot")
        _FALLBACK["arr"] = (df["load"].values.astype(float),
                            df["pv_forecast"].values.astype(float))
    return _FALLBACK["arr"]


def forecast_mate(i, dates, load, pv, cls_arr, k_load=4, k_pv=4):
    """
    同伴预测依据：
    - 负载：最近 k_load 个同类日（周五/周六 与 其余 两类）的平均；
    - 光伏：最近 k_pv 个连续日的平均；
    - 历史不足时退化为附件 1 典型日曲线。
    """
    before = np.arange(len(dates)) < i
    cand = np.where(before & (cls_arr == cls_arr[i]))[0]
    prev = np.where(before)[0]

    if len(cand) == 0:
        load_f = load_fallback_profile()[0].copy()
    else:
        load_f = load[cand[-k_load:]].mean(axis=0)

    if len(prev) == 0:
        pv_f = load_fallback_profile()[1].copy()
    else:
        pv_f = np.maximum(pv[prev[-k_pv:]].mean(axis=0), 0.0)

    return load_f, pv_f


def compute_residuals_mate(dates, load, pv, net, cls_arr, k_load=4, k_pv=4):
    """同伴预测依据下的净负载残差 e_j(t) = 实际净负载 − 预测净负载（历史不足处为 NaN）。"""
    n_days = len(dates)
    err = np.full((n_days, SLOTS_PER_DAY), np.nan)
    for i in range(n_days):
        before = np.arange(n_days) < i
        cand = np.where(before & (cls_arr == cls_arr[i]))[0]
        prev = np.where(before)[0]
        if len(cand) < 1 or len(prev) < k_pv:
            continue
        net_f = load[cand[-k_load:]].mean(axis=0) - np.maximum(
            pv[prev[-k_pv:]].mean(axis=0), 0.0)
        err[i] = net[i] - net_f
    return err


# ----------------------------------------------------------------------------
# 2. 报童安全裕度
# ----------------------------------------------------------------------------

def quantile_margin(err, i, q, window=30):
    """目标日 i 之前 window 天内净需求残差的逐时段 q 分位（kW）。"""
    valid = np.where(
        (np.arange(len(err)) < i) & (~np.isnan(err).any(axis=1)))[0]
    if len(valid) == 0:
        return np.zeros(SLOTS_PER_DAY)
    return np.quantile(err[valid[-window:]], q, axis=0)


def newsvendor_q(prices, window=5):
    """
    报童临界分位数：缺货代价 c_u = 5p − p = 4p，超量代价 c_o = p，
    q* = c_u/(c_u+c_o) = 0.8。电价同比例缩放不影响该比值，故与价格无关。
    """
    return 4.0 / 5.0


# ----------------------------------------------------------------------------
# 3. 因果（online）结算规则
# ----------------------------------------------------------------------------

def causal_dispatch(net_actual_kW, P_plan_kW, e0):
    """
    逐时段因果控制：只用当前时刻实测净负载与当前储电量。

    记 g = (实际净负载 − 计划购电) × Δt（kWh/时段）：
      g > 0 实际比计划更缺电 —— 放电兜底，仍不足的部分为紧急购电；
      g < 0 实际比计划更富余 —— 充电吸纳，仍吸纳不下的部分弃光。
    完美预测时 g 恰等于计划充放电量，规则自动复现日前计划。

    返回 (P_ch, P_dis, P_em, E, P_curt)，功率口径 kW / 储电量 kWh。
    传入数组长度可变：既可用于整天（144），也可用于日内某一段。
    """
    n = len(net_actual_kW)
    P_ch = np.zeros(n)
    P_dis = np.zeros(n)
    P_em = np.zeros(n)
    P_curt = np.zeros(n)
    E = np.zeros(n + 1)
    E[0] = e0

    for t in range(n):
        g = (net_actual_kW[t] - P_plan_kW[t]) * HOURS_PER_SLOT     # kWh
        if g >= 0:                                  # 缺电：放电兜底
            d = max(min(g, CBAR, ETA * (E[t] - EMIN)), 0.0)
            E[t + 1] = E[t] - d / ETA
            P_dis[t] = d / HOURS_PER_SLOT
            P_em[t] = (g - d) / HOURS_PER_SLOT
        else:                                       # 富余：充电吸纳
            c = max(min(-g, CBAR, (EMAX - E[t]) / ETA), 0.0)
            E[t + 1] = E[t] + ETA * c
            P_ch[t] = c / HOURS_PER_SLOT
            P_curt[t] = (-g - c) / HOURS_PER_SLOT   # 吸纳不下的富余 → 弃光

    return P_ch, P_dis, P_em, E, P_curt


def verify_causal(P_ch, P_dis, P_em, E, P_curt, net_actual_kW, P_plan_kW, tol=1e-6):
    """校验因果结算结果满足功率平衡、SOC 边界与充放电互斥。"""
    # 净负载已含 −PV，故平衡为 P_plan + P_em + P_dis − P_ch − P_curt = net_actual
    bal = P_plan_kW + P_em + P_dis - P_ch - P_curt - net_actual_kW
    ok = np.abs(bal).max() < tol
    ok &= (E.min() >= EMIN - tol) and (E.max() <= EMAX + tol)
    ok &= not ((P_ch > tol) & (P_dis > tol)).any()
    return bool(ok), float(np.abs(bal).max())
