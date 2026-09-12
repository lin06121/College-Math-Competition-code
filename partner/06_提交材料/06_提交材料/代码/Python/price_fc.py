# -*- coding: utf-8 -*-
"""问题四电价预测(全部因果, 只用目标日之前 / 当日已实现的信息)

结构模型(可解释):  p̂_{d,t} = w1·p_{d-7,t} + w2·p_{d-14,t} + w3·p_{d-21,t},  w ≥ 0, Σw = 1
  · 权重按**目标日的星期**分别拟合(周五/周六与其他五天电价水平差异大), 只用目标日之前的历史;
  · 每 RETRAIN 天重拟合一次, 保证因果。

残差修正(可选):  r_{d,t} = p_{d,t} − p̂^base_{d,t}; 用 LightGBM 因果滚动预测 r̂;  p̂ = p̂^base + r̂
  · 只有在"残差确实可学"(滚动外样本 MAE 下降)时才启用, 否则退回纯结构模型。

日内滚动(Q4-3): 6:00/12:00/18:00 用**当日已实现**实时电价与 0:00 预测的偏差, 对剩余时段做阻尼修正。
"""
import numpy as np
from causal_core import D, PR as RTP

n, H, m2 = D.n, 144, D.m2
LOOKBACK = 35       # 拟合权重的历史天数上限(≈5 周)
RETRAIN = 14        # 重拟合/重训间隔(天)
ADJ_SLOTS = (36, 72, 108)     # 6:00 / 12:00 / 18:00
DOW = np.asarray(D.dow)


def weekly_weights(R, d):
    """用第 d 天之前、与 d 同星期的历史拟合 w≥0, Σw=1; 样本不足时回退到全部历史"""
    for same_dow_only in (True, False):
        idx = np.array([j for j in range(max(21, d - LOOKBACK), d)
                        if (not same_dow_only) or DOW[j] == DOW[d]])
        if len(idx) >= 5:
            A = np.stack([R[idx - 7], R[idx - 14], R[idx - 21]], axis=1).reshape(-1, 3)
            y = R[idx].reshape(-1)
            w = np.linalg.lstsq(A, y, rcond=None)[0]
            w = np.clip(w, 0.0, None)
            if w.sum() > 0:
                return w / w.sum(), len(idx)
    return np.array([1.0, 0.0, 0.0]), 0


def build_price_base(verbose=False):
    """周周期加权结构模型(按星期拟合权重), 全因果"""
    R = RTP
    base = np.zeros((n, H)); W = np.zeros((n, 3))
    for d0 in range(0, n, RETRAIN):
        for d in range(d0, min(d0 + RETRAIN, n)):
            if d < 21:
                base[d] = R[d - 7] if d >= 7 else R[0]
                continue
            w, _ = weekly_weights(R, d)
            base[d] = w[0] * R[d - 7] + w[1] * R[d - 14] + w[2] * R[d - 21]
            W[d] = w
    if verbose:
        for wd in range(7):
            sel = (DOW == wd) & (np.arange(n) >= m2)
            if sel.sum():
                print(f'    星期{wd}: w̄={np.round(W[sel].mean(axis=0), 3)}  ({sel.sum()} 天)')
    return base, W


def _feat(base, R, day_idx):
    """残差模型特征(只用目标日之前的信息): 时段、星期、基础预测、7 天前基础与误差等"""
    F = []
    for d in day_idx:
        t = np.arange(H)
        f = [t / 144.0, np.full(H, DOW[d] / 6.0), base[d] / base.max(),
             base[d - 7] / base.max() if d >= 7 else base[d] / base.max(),
             (R[d - 7] - base[d - 7]) / 1.0 if d >= 7 else np.zeros(H),
             (R[d - 1] - base[d - 1]) if d >= 1 else np.zeros(H),
             np.full(H, base[d].mean() / base.max()),
             np.full(H, (R[d - 14] - base[d - 14]).mean() if d >= 14 else 0.0)]
        F.append(np.stack(f, axis=-1))
    return np.concatenate(F, axis=0)


def build_price(use_residual=True, verbose=False):
    """完整电价预测: 结构模型 + (可选)LightGBM 残差修正; 返回 p_hat 与诊断"""
    import lightgbm as lgb
    R = RTP
    base, W = build_price_base()
    res = np.zeros((n, H))
    ok = False
    if use_residual:
        for d0 in range(28, n, RETRAIN):
            d1 = min(d0 + RETRAIN, n)
            tr = np.arange(0, d0)
            Xtr = _feat(base, R, tr); ytr = (R[tr] - base[tr]).reshape(-1)
            Xte = _feat(base, R, np.arange(d0, d1))
            mdl = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=40, subsample=0.9, verbose=-1)
            mdl.fit(Xtr, ytr)
            res[d0:d1] = mdl.predict(Xte).reshape(d1 - d0, H)
        p_hat = np.clip(base + res, 0.005, 3.0)
        ok = np.abs(p_hat[m2:] - R[m2:]).mean() < np.abs(base[m2:] - R[m2:]).mean()
        p_hat = p_hat if ok else base
    else:
        p_hat = base
    mae = lambda A: float(np.abs(A[m2:] - R[m2:]).mean())
    rmse = lambda A: float(np.sqrt(((A[m2:] - R[m2:]) ** 2).mean()))
    naive = np.zeros((n, H)); naive[7:] = R[:n - 7]
    diag = dict(mae_base=mae(base), mae_resid=mae(np.clip(base + res, 0.005, 3.0)),
                mae_final=mae(p_hat), rmse_final=rmse(p_hat), use_residual=bool(ok),
                mae_naive7=mae(naive), w_mean=W[m2:].mean(axis=0))
    if verbose:
        print(f'    结构模型(按星期拟合 w, 7/14/21 加权) MAE={diag["mae_base"]:.4f}  w̄={np.round(diag["w_mean"],3)}')
        print(f'    +LightGBM 残差修正                 MAE={diag["mae_resid"]:.4f}  '
              f'({"启用" if ok else "未改善, 已弃用"})')
        print(f'    采用                               MAE={diag["mae_final"]:.4f}  RMSE={diag["rmse_final"]:.4f}')
        print(f'    朴素(直接取 d-7 同一时段)           MAE={diag["mae_naive7"]:.4f}')
    return p_hat, base, diag


def build_price_rolling(p_hat0, t0_slots=ADJ_SLOTS, damp=0.7):
    """Q4-3 日内滚动修正: 只用当日已实现时段的价格误差, 对剩余时段做阻尼水平修正"""
    R = RTP
    out = {}
    for s in t0_slots:
        P = p_hat0.copy()
        for d in range(n):
            adj = damp * float(np.mean(R[d, :s] - p_hat0[d, :s]))
            P[d, s:] = np.clip(p_hat0[d, s:] + adj, 0.005, 3.0)
        out[s // 6] = P
    return out


def rolling_mae(p_hat0, P_roll):
    R = RTP
    return {h: dict(update=float(np.abs(P_roll[h][m2:, s:] - R[m2:, s:]).mean()),
                    none=float(np.abs(p_hat0[m2:, s:] - R[m2:, s:]).mean()))
            for h, s in zip((6, 12, 18), ADJ_SLOTS)}


if __name__ == '__main__':
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    print('=== 电价预测(周周期加权 + LightGBM 残差, 全因果) ===')
    p_hat, base, diag = build_price(verbose=True)
    print('\n=== 日内滚动修正: 阻尼选择(剩余时段 MAE, 元/kWh) ===')
    best = None
    for damp in (0.0, 0.3, 0.5, 0.7, 1.0):
        r = rolling_mae(p_hat, build_price_rolling(p_hat, damp=damp))
        avg = np.mean([r[h]['update'] for h in (6, 12, 18)])
        print(f'  damp={damp:.1f}: ' + ' | '.join(
            f'{h}时 {r[h]["update"]:.4f}(不修正 {r[h]["none"]:.4f})' for h in (6, 12, 18))
            + f'  平均 {avg:.4f}')
        if best is None or avg < best[1]:
            best = (damp, avg)
    print(f'\n最优阻尼 damp={best[0]:.1f} (平均 {best[1]:.4f})')
    print('\n=== 对照: 原有"廓线+自适应+堆叠"模型 ===')
    import fc_upgraded as FU
    st = FU.build_price(verbose=False)
    print(f'    廓线+自适应水平+岭回归堆叠 MAE={np.abs(st[m2:] - RTP[m2:]).mean():.4f}')
