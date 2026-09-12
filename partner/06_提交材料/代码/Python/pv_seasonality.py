# -*- coding: utf-8 -*-
"""光伏季节特征验证 + 季节趋势修正测试
A) 季节性证据: 月度统计/日序列平滑曲线/正弦拟合/月度日曲线形态/有效日照时长
B) 对计划依据的影响: m3(近3日均值) vs m3+动量趋势, 按月MAE + 全年链条成本对比
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import linprog
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

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
PV_E = PV_act.sum(axis=1)   # 日光伏电量 kWh
doy = np.array([d.dayofyear for d in dates365])

# ============ A) 季节性证据 ============
mon = dates365.month
g = pd.Series(PV_E, index=dates365).groupby(mon).agg(['mean', 'std'])
print('===== A. 月度统计 (日均光伏电量, kWh) =====')
print(g.round(0).to_string())
ratio = g['mean'].max() / g['mean'].min()
print(f'[A] 最高月/最低月 = {ratio:.2f} 倍; 月度均值标准差 = {g["mean"].std():.0f} kWh ({g["mean"].std()/g["mean"].mean()*100:.1f}% of 年均)')

# 正弦拟合: E(doy) = a + b*sin(2*pi*(doy - phi)/365 + pi/2 相位处理)
def sine(d, a, b, c):
    return a + b * np.sin(2 * np.pi * (d - c) / 365)
p0 = [PV_E.mean(), 10000, 100]
popt, _ = curve_fit(sine, doy, PV_E, p0=p0, maxfev=20000)
a, b, c = popt
fit = sine(doy, *popt)
ss_res = ((PV_E - fit) ** 2).sum(); ss_tot = ((PV_E - PV_E.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
peak_doy = (c + 91.25) % 365 if c + 91.25 <= 365 else c - 273.75
peak_date = pd.Timestamp('2025-01-01') + pd.Timedelta(days=int(peak_doy) - 1)
print(f'[A] 正弦拟合: 均值a={a:.0f} kWh, 振幅b={b:.0f} kWh ({(2*b)/(a)*100 if a>0 else 0:.1f}% 峰谷摆幅), 峰值日≈{peak_date.date()}, R²={r2:.3f}')

# 有效日照时长(数据代理): 每日 PV>1%峰值的时段数
thr = PV_act.max() * 0.01
sun_h = (PV_act > thr).sum(axis=1) / 6.0
corr = np.corrcoef(PV_E, sun_h)[0, 1]
print(f'[A] 有效日照时长: 全年均值={sun_h.mean():.1f} h/日, 冬={sun_h[mon==1].mean():.1f} 夏={sun_h[mon==8].mean():.1f}; 与日光伏电量相关系数={corr:.3f}')

# 月度日曲线形态 (1/4/7/10月均值, kW)
fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.5))
t_axis = (np.arange(144) + 0.5) / 6
ax0 = axes[0][0]
ax0.plot(doy, PV_E / 1000, '.', color=GRAY, ms=2, alpha=0.5, label='每日光伏电量')
smooth = pd.Series(PV_E).rolling(7, center=True).mean()
ax0.plot(doy, smooth / 1000, color=BLUE, lw=1.6, label='7日滑动平均')
ax0.plot(doy, fit / 1000, color=RED, lw=1.4, ls='--', label=f'正弦拟合 (R²={r2:.2f})')
ax0.set_xlabel('年积日'); ax0.set_ylabel('日光伏电量 (MWh)')
ax0.set_title('全年日光伏电量与正弦拟合', fontsize=9)
ax0.grid(alpha=0.3, ls='--'); ax0.legend(frameon=False, fontsize=7)
ax1 = axes[0][1]
monthly = pd.Series(PV_E, index=dates365).groupby(mon).mean()
ax1.bar([f'{m}月' for m in range(1, 13)], monthly / 1000, color=ORANGE, alpha=0.9)
ax1.set_ylabel('日均光伏电量 (MWh)'); ax1.set_title('月度均值 (6月雨季凹陷)', fontsize=9)
ax1.grid(alpha=0.3, axis='y', ls='--'); ax1.tick_params(labelsize=7)
for k, v in enumerate(monthly):
    ax1.text(k, v / 1000 + 0.6, f'{v/1000:.0f}', ha='center', fontsize=6.5)
ax2 = axes[1][0]
for m, color, lab in [(1, '#56B4E9', '1月'), (4, GREEN, '4月'), (7, RED, '7月'), (10, '#CC79A7', '10月')]:
    prof = PV_act[mon == m].mean(axis=0) * 6
    ax2.plot(t_axis, prof / 1000, color=color, lw=1.3, label=lab)
ax2.set_xlabel('时刻 (h)'); ax2.set_ylabel('功率 (MW)')
ax2.set_title('四季日曲线形态 (日出/日落时刻与峰值高度均变化)', fontsize=9)
ax2.grid(alpha=0.3, ls='--'); ax2.legend(frameon=False, fontsize=7)
ax3 = axes[1][1]
sun_m = pd.Series(sun_h, index=dates365).groupby(mon).mean()
ax3.plot(range(1, 13), sun_m, marker='o', color=GREEN)
ax3.set_xlabel('月份'); ax3.set_ylabel('有效日照时长 (h/日)')
ax3.set_title('有效日照时长 (PV>1%峰值)', fontsize=9); ax3.grid(alpha=0.3, ls='--')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_pv_seasonality.png')); plt.close(fig)
print('[A] 图已存 fig_pv_seasonality.png')

# ============ B) 对计划依据的影响 ============
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
L_prev = np.zeros_like(L_act)
for i in range(n):
    low_set = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low_set) == (dow[i] in low_set)]
    L_prev[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ

def pv_m3():
    B = np.zeros_like(PV_act)
    for i in range(n):
        hist = list(range(max(0, i - 3), i))
        B[i] = PV_act[hist].mean(axis=0) if hist else PV_typ
    return B

def pv_trend(alpha=0.5):
    B = np.zeros_like(PV_act)
    for i in range(n):
        if i >= 6:
            m3 = PV_act[i - 3:i].mean(axis=0)
            m3p = PV_act[i - 6:i - 3].mean(axis=0)
            B[i] = np.maximum(m3 + alpha * (m3 - m3p), 0.0)
        elif i >= 3:
            B[i] = PV_act[i - 3:i].mean(axis=0)
        elif i > 0:
            B[i] = PV_act[i - 1]
        else:
            B[i] = PV_typ
    return B

PB_m3 = pv_m3()
PB_tr = pv_trend(0.5)
# 按月 MAE
print('===== B. 光伏依据按月 MAE (kWh/时段) =====')
for name, B in [('m3', PB_m3), ('m3+趋势', PB_tr)]:
    mae_m = pd.Series(np.abs(PV_act - B).mean(axis=1), index=dates365).groupby(mon).mean()
    print(f'{name}: ' + ' '.join([f'{m}月{mae_m[m]:.1f}' for m in range(1, 13)]) + f' | 全年={np.abs(PV_act-B).mean():.2f}')

# 链条成本对比 (负载依据固定=dow2K4, q*按1月调优)
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
    return (res.x[iE:iE + h], res.x[iC:iC + h], res.x[iD:iD + h], res.x[iS + h], float(5.0 * p @ res.x[iE:iE + h]))

def chain(PB, qlevel, rng=None):
    idx = np.arange(n) if rng is None else np.arange(*rng)
    r = (L_act - PV_act) - (L_prev - PB)
    q = np.zeros(144); soc0 = 6000.0
    cp = np.zeros(n); ce = np.zeros(n)
    for i in idx:
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        if qlevel is not None:
            lo = max(0, i - 30)
            q = np.quantile(r[lo:i], qlevel, axis=0) if i - lo >= 7 else np.zeros(144)
        P0 = solve_day(price, L_prev[i] + q, PB[i], soc0, soc_end=None, v=v)['P']
        E, C, D, s24r, cei = settlement(L_act[i], PV_act[i], P0, soc0, v)
        cp[i] = float(price @ P0); ce[i] = cei; soc0 = s24r
    return cp, ce

t0 = time.time()
res = {}
for name, PB in [('m3', PB_m3), ('m3+趋势', PB_tr)]:
    jan_cost = {}
    for ql in [0.5, 0.6, 0.7, 0.8]:
        cp, ce = chain(PB, ql, rng=(0, 31))
        jan_m = np.where(dates365 >= pd.Timestamp('2025-01-08'))[0][0]
        jan_cost[ql] = (cp + ce)[jan_m:31].sum()
    q_star = min(jan_cost, key=jan_cost.get)
    cp, ce = chain(PB, q_star)
    m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
    res[name] = dict(q_star=q_star, total=float((cp + ce)[m:].sum()),
                     plan=float(cp[m:].sum()), em=float(ce[m:].sum()))
    print(f'[B] {name}: q*={q_star} (1月调优), 2.1-12.31总费={(cp+ce)[m:].sum():,.0f} 元')
print(f'[B] 趋势修正节费 = {res["m3"]["total"]-res["m3+趋势"]["total"]:,.0f} 元 ({(res["m3"]["total"]-res["m3+趋势"]["total"])/res["m3"]["total"]*100:.2f}%) | 耗时={time.time()-t0:.1f}s')

with open(os.path.join(DATA, 'pv_seasonality.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(monthly_mean={int(k): float(v) for k, v in monthly.items()},
                   sine=dict(a=float(a), b=float(b), peak=str(peak_date.date()), r2=float(r2)),
                   ratio=float(ratio), corr_sun=float(corr), basis_cost=res), f, ensure_ascii=False, indent=1)
print('PV SEASONALITY DONE')
