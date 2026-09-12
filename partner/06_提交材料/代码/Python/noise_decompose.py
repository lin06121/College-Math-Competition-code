# -*- coding: utf-8 -*-
"""噪声/季节性 建模再评估 (参考: 考虑预测误差不确定性的源-荷-广义储低碳经济动态优化调度)
A) 残差季节偏置: 负载(dow2K4)与光伏(m4)依据误差的逐月均值±std
B) 误差分布: 净残差分小时偏度/峰度, 经验分位 vs 正态分位缓冲对比
C) 预言机分解: 完美光伏(O_PV) / 完美负载(O_L) 与 M8、M3 的差距 -> 光伏噪声 vs 负荷噪声的信息价值
D) 逐月滚动分位 q*(月) vs 常数0.7
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy import stats
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_core import balance

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')
DATA = os.path.join(ROOT, r'04_建模求解\data')
EPS = 1e-4

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365])
n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
Lb = np.zeros_like(L_act)
for i in range(n):
    low_set = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low_set) == (dow[i] in low_set)]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
Pb = np.zeros_like(PV_act)   # m4
for i in range(n):
    hist = list(range(max(0, i - 4), i))
    Pb[i] = PV_act[hist].mean(axis=0) if hist else PV_typ
eL = L_act - Lb
eP = PV_act - Pb
eN = eL - eP
mon = np.array([d.month for d in dates365])

# ============ A) 残差季节偏置 ============
print('===== A. 残差逐月均值(偏置)与标准差 (kWh/时段) =====')
for name, e in [('负载误差 eL', eL), ('光伏误差 eP', eP)]:
    rows = []
    for mm in range(1, 13):
        rows.append(f'{mm}月:{e[mon==mm].mean():+.1f}±{e[mon==mm].std():.0f}')
    print(f'  {name}: ' + '  '.join(rows))
print(f'  全年: eL 均值={eL.mean():+.2f}, eP 均值={eP.mean():+.2f}')

# ============ B) 误差分布 ============
print('===== B. 净残差分布 (按小时) =====')
hour_of = (np.arange(144) // 6)
skew_h = np.array([stats.skew(eN[:, hour_of == h].ravel()) for h in range(24)])
kurt_h = np.array([stats.kurtosis(eN[:, hour_of == h].ravel()) for h in range(24)])
sig_h = np.array([eN[:, hour_of == h].ravel().std() for h in range(24)])
print(f'  偏度: 均值={skew_h.mean():+.2f} (范围 {skew_h.min():+.2f}~{skew_h.max():+.2f})')
print(f'  峰度: 均值={kurt_h.mean():+.2f} (正态=0; 范围 {kurt_h.min():+.2f}~{kurt_h.max():+.2f})')
print(f'  标准差: 均值={sig_h.mean():.1f}, 最大小时={np.argmax(sig_h)}时({sig_h.max():.1f})')

# ============ 结算/链条 ============
def settlement(L, PV, P0, soc0, v):
    h = 144; p = price
    iC, iD, iE, iS = 0, h, 2 * h, 3 * h
    N = 4 * h + (h + 1)
    c = np.zeros(N); c[iE:iE + h] = 5.0 * p; c[iC:iC + h] = EPS; c[iD:iD + h] = EPS; c[iS + h] = -v
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        row = np.zeros(N); row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        row = np.zeros(N); row[iS + t] = -1; row[iS + t + 1] = 1
        row[iC + t] = -ETA_C; row[iD + t] = 1.0 / ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iS:iS + h + 1] = SOC_MIN; ub[iS:iS + h + 1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success
    return (res.x[iE:iE + h], res.x[iC:iC + h], res.x[iD:iD + h], res.x[iS + h], float(5.0 * p @ res.x[iE:iE + h]))

def chain(LB, PB_, buf_mode, qlevel, rng=None):
    """buf_mode: 'net' | 'load' | 'pv' | 'normal_hour' | 'monthly'(按月调q, qlevel忽略)"""
    idx = np.arange(n) if rng is None else np.arange(*rng)
    soc0 = 6000.0
    cp = np.zeros(n); ce = np.zeros(n)
    q_month = {}
    for i in idx:
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        buf = np.zeros(144)
        lo = max(0, i - 30)
        if i - lo >= 7:
            if buf_mode == 'net':
                buf = np.quantile(eN[lo:i], qlevel, axis=0)
            elif buf_mode == 'load':
                buf = np.quantile(eL[lo:i], qlevel, axis=0)
            elif buf_mode == 'pv':
                buf = np.quantile(-eP[lo:i], qlevel, axis=0)
            elif buf_mode == 'normal_hour':
                z = stats.norm.ppf(qlevel)
                buf = np.zeros(144)
                for h in range(24):
                    seg = eN[lo:i][:, hour_of == h].ravel()
                    buf[hour_of == h] = z * seg.std()
            elif buf_mode == 'monthly':
                if mon[i] == 1:
                    buf = np.quantile(eN[lo:i], 0.7, axis=0)
                else:
                    if mon[i] not in q_month:
                        if mon[i] == 2:
                            q_month[mon[i]] = 0.7   # 2月用1月调出的0.7
                        else:
                            pm = mon[i] - 1
                            best_q, best_c = 0.7, np.inf
                            seg = np.where(mon == pm)[0]
                            for ql in [0.5, 0.6, 0.7, 0.8]:
                                cpj, cej = chain(LB, PB_, 'net', ql, rng=(seg[0], seg[-1] + 1))
                                cj = (cpj + cej)[seg[0]:seg[-1] + 1].sum()
                                if cj < best_c:
                                    best_c, best_q = cj, ql
                            q_month[mon[i]] = best_q
                    buf = np.quantile(eN[lo:i], q_month[mon[i]], axis=0)
        P0 = solve_day(price, LB[i] + buf, PB_[i], soc0, soc_end=None, v=v)['P']
        Cb, Db, s24r, Em = balance(P0, np.zeros(144), np.zeros(144), soc0, L_act[i], PV_act[i])
        cp[i] = float(price @ P0); ce[i] = float(5.0 * price @ Em); soc0 = s24r
    return cp, ce

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
t0 = time.time()
print('===== C. 预言机分解 =====')
r_M8 = chain(Lb, Pb, 'net', 0.7);           tot_M8 = (r_M8[0] + r_M8[1])[m:].sum()
r_PV = chain(Lb, PV_act, 'load', 0.7);      tot_PV = (r_PV[0] + r_PV[1])[m:].sum()
r_L = chain(L_act, Pb, 'pv', 0.7);          tot_L = (r_L[0] + r_L[1])[m:].sum()
soc0 = 6000.0; cp3 = np.zeros(n)
for i in range(n):
    rr = solve_day(price, L_act[i], PV_act[i], soc0, soc_end=None, v=(0.9 * price.mean() if i < n - 1 else 0.0))
    cp3[i] = rr['gross']; soc0 = rr['E'][-1]
tot_M3 = cp3[m:].sum()
print(f'  M8(负荷+光伏均含噪声) = {tot_M8:,.0f} 元')
print(f'  O_PV(光伏完美)        = {tot_PV:,.0f} 元  -> 光伏噪声价值 = {tot_M8-tot_PV:,.0f} 元')
print(f'  O_L (负荷完美)        = {tot_L:,.0f} 元  -> 负荷噪声价值 = {tot_M8-tot_L:,.0f} 元')
print(f'  M3(全完美下界)        = {tot_M3:,.0f} 元')
print(f'  [结论] 信息价值分配: 光伏噪声占 {(tot_M8-tot_PV)/max(tot_M8-tot_M3,1)*100:.0f}%, 负荷噪声占 {(tot_M8-tot_L)/max(tot_M8-tot_M3,1)*100:.0f}%')
# 按月缺口
gap_m = pd.Series((r_M8[0] + r_M8[1]) - cp3, index=dates365).groupby(mon).sum()
print('  按月 距天眼缺口(元):', {int(k): int(v) for k, v in gap_m.items()})

print('===== D. 缓冲方案对比 =====')
r_N = chain(Lb, Pb, 'normal_hour', 0.7);    tot_N = (r_N[0] + r_N[1])[m:].sum()
r_M = chain(Lb, Pb, 'monthly', 0.7);        tot_M = (r_M[0] + r_M[1])[m:].sum()
print(f'  经验分位(现行)      = {tot_M8:,.0f} 元')
print(f'  正态参数化(按小时σ) = {tot_N:,.0f} 元 (差 {tot_N-tot_M8:+,.0f})')
print(f'  逐月滚动q*          = {tot_M:,.0f} 元 (差 {tot_M-tot_M8:+,.0f})')
print(f'[总耗时] {time.time()-t0:.1f}s')

# ============ 图 ============
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
ax = axes[0]
mm_ = np.arange(1, 13)
lmean = [eL[mon == k].mean() for k in mm_]; lstd = [eL[mon == k].std() for k in mm_]
pmean = [eP[mon == k].mean() for k in mm_]; pstd = [eP[mon == k].std() for k in mm_]
ax.errorbar(mm_, lmean, yerr=lstd, fmt='o-', color=BLUE, ms=3, lw=1, label='负载误差 eL')
ax.errorbar(mm_, pmean, yerr=pstd, fmt='s--', color=ORANGE, ms=3, lw=1, label='光伏误差 eP')
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xlabel('月份'); ax.set_ylabel('误差均值±标准差 (kWh/时段)')
ax.set_title('残差季节偏置检验: 全年无明显系统性偏置', fontsize=9)
ax.grid(alpha=0.3, ls='--'); ax.legend(frameon=False, fontsize=7)
ax = axes[1]
vals = [tot_M8, tot_PV, tot_L, tot_M3]
ax.bar(['M8 两噪声', 'O_PV 光伏完美', 'O_L 负荷完美', 'M3 全完美'], np.array(vals) / 1e6,
       color=[BLUE, GREEN, ORANGE, GRAY])
ax.set_ylabel('全年总费用 (百万元)')
ax.set_title('预言机分解: 光伏噪声价值小, 负荷噪声价值大', fontsize=9)
ax.grid(alpha=0.3, axis='y', ls='--')
for k, v in enumerate(vals):
    ax.text(k, v / 1e6 + 0.08, f'{v/1e6:.2f}', ha='center', fontsize=7)
ax = axes[2]
ax.bar([f'{k}月' for k in mm_], gap_m.to_numpy() / 1e4, color=RED, alpha=0.85)
ax.set_ylabel('距天眼缺口 (万元)')
ax.set_title('信息价值按月分布 (春季/雨季最大)', fontsize=9)
ax.grid(alpha=0.3, axis='y', ls='--'); ax.tick_params(labelsize=6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_noise_decompose.png')); plt.close(fig)

with open(os.path.join(DATA, 'noise_decompose.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(A=dict(eL_mean=float(eL.mean()), eP_mean=float(eP.mean())),
                   B=dict(skew=float(skew_h.mean()), kurt=float(kurt_h.mean()), sig_max_h=int(np.argmax(sig_h))),
                   C=dict(M8=tot_M8, O_PV=tot_PV, O_L=tot_L, M3=tot_M3,
                          pv_share=float((tot_M8 - tot_PV) / max(tot_M8 - tot_M3, 1)),
                          load_share=float((tot_M8 - tot_L) / max(tot_M8 - tot_M3, 1)),
                          gap_month={int(k): float(v) for k, v in gap_m.items()}),
                   D=dict(empirical=tot_M8, normal_hour=tot_N, monthly_q=tot_M)), f, ensure_ascii=False, indent=1)
print('NOISE DECOMPOSE DONE')
