# -*- coding: utf-8 -*-
"""Q2: 全年逐日 计划购电(0:00制定) + 实时储能再调度 + 5x紧急购电
机制: 0:00锁定计划购电量P0(基准价); 当日储能按实际数据再调度; 缺口=紧急购电(5x)
方法对比:
  M1 外推(d-1)计划        M2 典型日曲线计划
  M4 无储能(外推+报童裕度) M5 外推+滚动报童裕度(80%分位)
  M6 两阶段随机规划SAA(近10日情景)   M3 天眼下界(不可达)
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
from engine import solve_day, merge_spans, span_label, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, write_result2, dates_of_results, ROOT

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
BLUE, ORANGE, GREEN, RED, GRAY = '#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888'
FIG = os.path.join(ROOT, r'04_建模求解\fig')
DATA = os.path.join(ROOT, r'04_建模求解\data')
EPS = 1e-4   # 打破退化的小惩罚 (元/kWh, 数值可忽略)

day, typ, fc0, fca, pers, tmpl, const = load_all()
price_typ = typ['price_typ'].to_numpy()
v_const = 0.9 * price_typ.mean()

def pivot(col):
    return day.pivot(index='date', columns='slot', values=col).sort_index() / 6.0
L_act = pivot('load_kw'); PV_act = pivot('pv_kw')
L_prev = pers.pivot(index='date', columns='slot', values='load_prev_kw').sort_index().to_numpy() / 6.0
PV_prev = pers.pivot(index='date', columns='slot', values='pv_prev_kw').sort_index().to_numpy() / 6.0
L_act_np, PV_act_np = L_act.to_numpy(), PV_act.to_numpy()
L_typ = np.broadcast_to(typ['load_typ'].to_numpy() / 6.0, L_act.shape)
PV_typ = np.broadcast_to(typ['pvfc_typ'].to_numpy() / 6.0, L_act.shape)
dates365 = L_act.index
n = len(dates365)

# ---------- 实时结算: 储能再调度, 最小化紧急购电 ----------
def settlement(L, PV, P0, soc0, v):
    h = 144; p = price_typ
    nC, nD, nE, nS = h, h, h, h + 1
    iC, iD, iE, iS = 0, h, 2 * h, 3 * h
    N = nC + nD + nE + nS
    c = np.zeros(N)
    c[iE:iE + h] = 5.0 * p
    c[iC:iC + h] = EPS; c[iD:iD + h] = EPS
    c[iS + h] = -v
    Aeq, beq = [], []
    Aub, bub = [], []
    for t in range(h):
        # 允许弃光: PV+P0+E+D-C >= L  <=>  -E+C-D <= PV+P0-L
        row = np.zeros(N)
        row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(PV[t] + P0[t] - L[t])
    for t in range(h):
        row = np.zeros(N)
        row[iS + t] = -1; row[iS + t + 1] = 1
        row[iC + t] = -ETA_C; row[iD + t] = 1.0 / ETA_D
        Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = P_MAX; ub[iD:iD + h] = P_MAX
    lb[iS:iS + h + 1] = SOC_MIN; ub[iS:iS + h + 1] = SOC_MAX
    lb[iS], ub[iS] = soc0, soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                  A_eq=np.array(Aeq) if Aeq else None, b_eq=np.array(beq) if beq else None,
                  bounds=list(zip(lb, ub)), method='highs')
    if not res.success:
        return None
    E = res.x[iE:iE + h]; C = res.x[iC:iC + h]; D = res.x[iD:iD + h]; S = res.x[iS:iS + h + 1]
    return E, C, D, S[-1], float(5.0 * p @ E)

# ---------- SAA 两阶段: 选P0使期望(计划费+紧急费)最小 ----------
def saa_plan(L_scen, PV_scen, soc0, v, price=None):
    """L_scen/PV_scen: (S,144) 情景; 返回 P0(144)"""
    p = price_typ if price is None else price
    S, h = L_scen.shape
    nP = h
    per = 3 * h + (h + 1)          # C,D,E,SOC 每情景
    N = nP + S * per
    def off(s, k):
        return nP + s * per + k
    c = np.zeros(N)
    c[:nP] = p
    for s in range(S):
        c[off(s, 2 * h):off(s, 3 * h)] += 5.0 * p / S
        c[off(s, 0):off(s, 2 * h)] += EPS / S
        c[off(s, 3 * h) + h] += -v / S
    Aeq, beq = [], []
    Aub, bub = [], []
    for s in range(S):
        for t in range(h):
            # 情景内允许弃光: PV+P0+E+D-C >= L
            row = np.zeros(N)
            row[t] = -1.0
            row[off(s, 2 * h) + t] = -1.0
            row[off(s, h) + t] = 1.0
            row[off(s, 0) + t] = -1.0
            Aub.append(row); bub.append(PV_scen[s, t] - L_scen[s, t])
        for t in range(h):
            row = np.zeros(N)
            row[off(s, 3 * h) + t] = -1.0; row[off(s, 3 * h) + t + 1] = 1.0
            row[off(s, 0) + t] = -ETA_C; row[off(s, h) + t] = 1.0 / ETA_D
            Aeq.append(row); beq.append(0.0)
    lb = np.zeros(N); ub = np.full(N, np.inf)
    for s in range(S):
        ub[off(s, 0):off(s, 2 * h)] = P_MAX
        lb[off(s, 3 * h):off(s, 3 * h) + h + 1] = SOC_MIN
        ub[off(s, 3 * h):off(s, 3 * h) + h + 1] = SOC_MAX
        lb[off(s, 3 * h)], ub[off(s, 3 * h)] = soc0, soc0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub),
                  A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=list(zip(lb, ub)), method='highs')
    assert res.success, res.message
    return res.x[:nP]

# ---------- 滚动报童裕度 ----------
net_basis = L_prev - PV_prev
r = net_basis  # placeholder
r = (L_act_np - PV_act_np) - (L_prev - PV_prev)
W = 30
q80 = np.zeros((n, 144))
for i in range(n):
    lo = max(0, i - W)
    q80[i] = np.quantile(r[lo:i], 0.8, axis=0) if i - lo >= 7 else 0.0
L_buf = L_prev + q80

# ---------- 方法链条 ----------
def chain_M(plan_fn, SAA=False):
    soc0 = 6000.0
    P_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    soc0s = np.zeros(n); soc24s = np.zeros(n)
    cp = np.zeros(n); E_m = np.zeros((n, 144)); ce = np.zeros(n)
    for i in range(n):
        v = 0.0 if i == n - 1 else v_const
        if SAA:
            S = min(20, i)
            if S >= 2:
                P0 = saa_plan(L_act_np[i - S:i], PV_act_np[i - S:i], soc0, v)
            else:
                P0 = solve_day(price_typ, L_buf[i], PV_prev[i], soc0, soc_end=None, v=v)['P']
        else:
            Lb, Pb = plan_fn(i)
            P0 = solve_day(price_typ, Lb, Pb, soc0, soc_end=None, v=v)['P']
        out = settlement(L_act_np[i], PV_act_np[i], P0, soc0, v)
        assert out is not None, (i, 'settlement infeasible')
        E, C, D, s24, cei = out
        P_m[i], C_m[i], D_m[i] = P0, C, D
        E_m[i] = E; cp[i] = float(price_typ @ P0); ce[i] = cei
        soc0s[i] = soc0; soc24s[i] = s24
        soc0 = s24
    return P_m, C_m, D_m, soc0s, soc24s, cp, E_m, ce

M1 = chain_M(lambda i: (L_prev[i], PV_prev[i]))
M2 = chain_M(lambda i: (L_typ[i], PV_typ[i]))
M5 = chain_M(lambda i: (L_buf[i], PV_prev[i]))
t0 = time.time()
M6 = chain_M(None, SAA=True)
t_saa = time.time() - t0
# 天眼: 直接用实际数据计划(无紧急)
def chain_clair():
    soc0 = 6000.0; cp = np.zeros(n); P_m = np.zeros((n, 144))
    for i in range(n):
        v = 0.0 if i == n - 1 else v_const
        r_ = solve_day(price_typ, L_act_np[i], PV_act_np[i], soc0, soc_end=None, v=v)
        P_m[i] = r_['P']; cp[i] = r_['gross']; soc0 = r_['E'][-1]
    return P_m, cp
P3, cp3 = chain_clair()
# 无储能(外推+报童): P0=max(net_buf,0), 缺口=紧急
P4 = np.maximum(L_buf - PV_prev, 0)
cp4 = P4 @ price_typ
E4 = np.maximum(L_act_np - PV_act_np - P4, 0)
ce4 = 5.0 * E4 @ price_typ

def tot(M):
    return M[5].sum() + M[7].sum()
m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
def line(name, M):
    print(f'{name}: 计划费={M[5][m:].sum():,.0f} + 紧急费={M[7][m:].sum():,.0f} = 总{M[5][m:].sum()+M[7][m:].sum():,.0f} 元 | 紧急电量={M[6][m:].sum():,.0f} kWh')
print(f'[参数] v={v_const:.4f}; 实时结算=储能再调度; SAA情景数=近10日')
print('===== 2.1-12.31 (334天) =====')
line('M1 外推计划        ', M1)
line('M2 典型日计划      ', M2)
line('M5 外推+报童裕度   ', M5)
line('M6 两阶段SAA(近20日情景)', M6)
print(f'M4 无储能(外推+报童): 计划费={cp4[m:].sum():,.0f} + 紧急费={ce4[m:].sum():,.0f} = 总{cp4[m:].sum()+ce4[m:].sum():,.0f} 元 | 紧急电量={E4[m:].sum():,.0f} kWh')
print(f'M3 天眼下界: 总{cp3[m:].sum():,.0f} 元')
print(f'[性能] SAA链耗时={t_saa:.1f}s')

# ---------- M7: 分位水平调优的报童 (q* 用1月验证集挑选) ----------
def chain_buf_q(qlevel):
    qb = np.zeros((n, 144))
    for i in range(n):
        lo = max(0, i - W)
        qb[i] = np.quantile(r[lo:i], qlevel, axis=0) if i - lo >= 7 else 0.0
    Lq = L_prev + qb
    return chain_M(lambda i, Lq=Lq: (Lq[i], PV_prev[i]))
q_grid = [0.5, 0.6, 0.7, 0.8]
jan_cost = {}
for q in q_grid:
    Mq = chain_buf_q(q)
    jan_m = np.where(dates365 >= pd.Timestamp('2025-01-08'))[0][0]  # 窗口满7天后
    jan_cost[q] = Mq[5][jan_m:].sum() + Mq[7][jan_m:].sum()
q_star = min(jan_cost, key=jan_cost.get)
print(f'[M7调优] 1月验证总费用: { {k: round(v) for k, v in jan_cost.items()} } -> 选 q*={q_star}')
M7 = chain_buf_q(q_star)
line('M7 报童(q*调优)    ', M7)

tots = {k: (v[5][m:].sum() + v[7][m:].sum()) for k, v in [('M1', M1), ('M2', M2), ('M5', M5), ('M6', M6), ('M7', M7)]}
tots['M4'] = cp4[m:].sum() + ce4[m:].sum()
tots['M3'] = cp3[m:].sum()
adoptable = {k: v for k, v in tots.items() if k in ('M1', 'M2', 'M5', 'M6', 'M7')}
best = min(adoptable, key=adoptable.get)
adopt = {'M1': M1, 'M2': M2, 'M5': M5, 'M6': M6, 'M7': M7}[best]
print(f'[结论] 可采纳方法中最优={best} ({tots[best]:,.0f} 元); 对M1节费={tots["M1"]-tots[best]:,.0f}; 储能价值(M4-{best})={tots["M4"]-tots[best]:,.0f}; 距天眼下界={tots[best]-tots["M3"]:,.0f} 元')

# ---------- 采用方法写结果 ----------
P_r, C_r, D_r = adopt[0], adopt[1], adopt[2]
s0_r, s24_r, cp_r, E_r = adopt[3], adopt[4], adopt[5], adopt[6]
res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in dates365])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
emerg = []
for i in range(len(res_dates)):
    idx = np.where(E_r[i0 + i] > 1e-6)[0] + 1
    emerg.append([(span_label(s0, s1), float(E_r[i0 + i][s0 - 1:s1].sum())) for s0, s1 in merge_spans(idx)])
write_result2('result2.xlsx', res_dates, P_r[i0:i1], cp_r[i0:i1], s0_r[i0:i1], s24_r[i0:i1], C_r[i0:i1], D_r[i0:i1], emerg)

# ---------- 论文表 ----------
spec = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
slot_of = {'10:00-10:10': 61, '12:00-12:10': 73, '14:00-14:10': 85, '16:00-16:10': 97, '18:00-18:10': 109, '20:00-20:10': 121}
paper = {}
for ds in spec:
    k = list(dates365).index(pd.Timestamp(ds))
    paper[ds] = dict(
        t1={lab: round(float(P_r[k, v - 1]), 2) for lab, v in slot_of.items()},
        total=round(float(P_r[k].sum()), 2), cost=round(float(cp_r[k]), 2),
        blocks=[(round(float(C_r[k, b * 24:(b + 1) * 24].sum()), 2), round(float(D_r[k, b * 24:(b + 1) * 24].sum()), 2)) for b in range(6)],
        soc0=round(float(s0_r[k]), 2), soc24=round(float(s24_r[k]), 2),
        emerg=[(span_label(s0, s1), round(float(E_r[k][s0 - 1:s1].sum()), 2)) for s0, s1 in merge_spans(np.where(E_r[k] > 1e-6)[0] + 1)])
print('[论文表]', json.dumps(paper, ensure_ascii=False, indent=1))

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
ax = axes[0]
ax.plot(dates365, tot(M1), color=GRAY, lw=0.8, label='M1 外推计划')
ax.plot(dates365, tot(M6), color=BLUE, lw=0.9, label='M6 两阶段SAA')
ax.plot(dates365, cp3, color=ORANGE, lw=0.8, label='M3 天眼下界')
ax.set_title('全年逐日购电总费用 (元)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7); ax.tick_params(axis='x', rotation=45, labelsize=6)
ax = axes[1]
em_m = pd.Series(E_r.sum(axis=1), index=dates365).groupby(dates365.month).sum()
em_1 = pd.Series(M1[6].sum(axis=1), index=dates365).groupby(dates365.month).sum()
x = np.arange(len(em_m)); w = 0.4
ax.bar(x - w / 2, em_1.to_numpy(), w, color=GRAY, alpha=0.8, label='M1 紧急电量')
ax.bar(x + w / 2, em_m.to_numpy(), w, color=RED, alpha=0.85, label=f'{best} 紧急电量')
ax.set_xticks(x); ax.set_xticklabels([f'{k}月' for k in em_m.index], fontsize=6)
ax.set_title('紧急购电量按月汇总 (kWh)', fontsize=9); ax.grid(alpha=0.3, axis='y', ls='--')
ax.legend(frameon=False, fontsize=7)
ax = axes[2]
k = list(dates365).index(pd.Timestamp('2025-06-21'))
t_axis = (np.arange(144) + 0.5) / 6
ax.plot(t_axis, L_act_np[k] * 6, color=BLUE, lw=1.0, label='实际负载')
ax.plot(t_axis, PV_act_np[k] * 6, color=ORANGE, lw=1.0, label='实际光伏')
ax.plot(t_axis, P_r[k] * 6, color=GREEN, lw=1.0, label='计划购电')
ax.fill_between(t_axis, 0, E_r[k] * 6, color=RED, alpha=0.5, label='紧急购电')
ax.set_title('2025-06-21 计划 vs 实际 (kW)', fontsize=9); ax.grid(alpha=0.3, ls='--')
ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_q2_annual.png')); plt.close(fig)

with open(os.path.join(DATA, 'q2_results.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(best=best, tots={k: float(v) for k, v in tots.items()}, t_saa=t_saa,
                   em_best_kwh=float(E_r[m:].sum()), paper=paper), f, ensure_ascii=False, indent=1)
print('Q2 DONE')
