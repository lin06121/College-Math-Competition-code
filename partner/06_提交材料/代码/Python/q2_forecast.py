# -*- coding: utf-8 -*-
"""Q2 计划依据升级: 同类星期(dow)滚动平均 + PV基值选优 + 报童q*重调
1) 用数据验证"周五/周六用电显著偏低"
2) 比较计划依据: 昨日外推 vs dow2类/7类近K个同类日平均; PV: 昨日 vs 近2/3日均值 (1月MAE选参)
3) 全年链条成本对比 (2.1-12.31): M1 / M7(旧采用) / M8(新) / M3天眼
4) M8若胜出 -> 重写 result2.xlsx
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
EPS = 1e-4

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365])   # 0=周一 ... 6=周日
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
n = len(dates365)

# ---------- 1) 验证周内规律 ----------
g = pd.Series(L_act.sum(axis=1), index=dates365).groupby(dow).mean()  # kWh/日
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
print('[周内规律] 日均负载电量(MWh):', {names[k]: round(float(g[k]) / 1000, 1) for k in range(7)})
low = sorted(np.argsort(g.to_numpy())[:2])
print(f'[周内规律] 用电最低两天 = {names[low[0]]}/{names[low[1]]}; 低日比高日低 { (1-g[low[0]]/np.mean([g[k] for k in range(7) if k not in low]))*100:.1f}%')

# ---------- 2) 计划依据构造 ----------
def dow_basis(K, nclass):
    """近K个同类日平均; nclass=7(按星期) 或 2(周五六 vs 其他)"""
    B = np.zeros_like(L_act)
    for i in range(n):
        if nclass == 7:
            hist = [j for j in range(i) if dow[j] == dow[i]]
        else:
            low_set = set(low)
            hist = [j for j in range(i) if (dow[j] in low_set) == (dow[i] in low_set)]
        if len(hist) == 0:
            B[i] = L_typ
        else:
            B[i] = L_act[hist[-K:]].mean(axis=0)
    return B

def pv_basis(kind):
    B = np.zeros_like(PV_act)
    for i in range(n):
        if kind == 'pers':
            B[i] = PV_act[i - 1] if i > 0 else PV_typ
        else:
            w = int(kind[-1])
            hist = list(range(max(0, i - w), i))
            B[i] = PV_act[hist].mean(axis=0) if hist else PV_typ
    return B

# 1月 MAE 选参
jan = dates365.month == 1
load_cands = {'pers': None, 'dow7K2': dow_basis(2, 7), 'dow7K4': dow_basis(4, 7), 'dow7K8': dow_basis(8, 7), 'dow2K4': dow_basis(4, 2)}
pv_cands = {'pers': pv_basis('pers'), 'm2': pv_basis('m2'), 'm3': pv_basis('m3')}
print('[选参] 1月 MAE (kWh/时段):')
best = {}
for lk, LB in load_cands.items():
    if LB is None:
        LB = np.roll(L_act, 1, axis=0); LB[0] = L_typ
    for pk, PB in pv_cands.items():
        mae_l = np.abs(L_act[jan] - LB[jan]).mean()
        mae_p = np.abs(PV_act[jan] - PB[jan]).mean()
        best[(lk, pk)] = mae_l + mae_p
        print(f'  load={lk:7s} pv={pk:4s}: 负载MAE={mae_l:6.1f} 光伏MAE={mae_p:6.1f} 合计={mae_l+mae_p:6.1f}')
lk_star, pk_star = min(best, key=best.get)
print(f'[选参] 最优组合: load={lk_star}, pv={pk_star} (1月合计MAE={best[(lk_star,pk_star)]:.1f})')

# ---------- 3) 结算与链条 ----------
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

def chain(LB, PB, qlevel=None, rng=None):
    """qlevel=None 表示不加裕度; rng=日期序号范围"""
    idx = np.arange(n) if rng is None else np.arange(*rng)
    r = (L_act - PV_act) - (LB - PB)
    q = np.zeros(144)
    soc0 = 6000.0
    P_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    s0 = np.zeros(n); s24 = np.zeros(n); cp = np.zeros(n); E_m = np.zeros((n, 144)); ce = np.zeros(n)
    for i in idx:
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        if qlevel is not None:
            lo = max(0, i - 30)
            q = np.quantile(r[lo:i], qlevel, axis=0) if i - lo >= 7 else np.zeros(144)
        P0 = solve_day(price, LB[i] + q, PB[i], soc0, soc_end=None, v=v)['P']
        E, C, D, s24r, cei = settlement(L_act[i], PV_act[i], P0, soc0, v)
        P_m[i] = P0; C_m[i] = C; D_m[i] = D; E_m[i] = E
        cp[i] = float(price @ P0); ce[i] = cei
        s0[i] = soc0; s24[i] = s24r; soc0 = s24r
    return P_m, C_m, D_m, s0, s24, cp, E_m, ce

LB_new = load_cands[lk_star] if lk_star != 'pers' else np.roll(L_act, 1, axis=0)
if lk_star == 'pers':
    LB_new[0] = L_typ
PB_new = pv_cands[pk_star]
# q* 在1月调优
jan_cost = {}
for ql in [0.5, 0.6, 0.7, 0.8]:
    Mq = chain(LB_new, PB_new, qlevel=ql, rng=(0, 31))
    jan_m = np.where(dates365 >= pd.Timestamp('2025-01-08'))[0][0]
    jan_cost[ql] = (Mq[5] + Mq[7])[jan_m:31].sum()
q_star = min(jan_cost, key=jan_cost.get)
print(f'[q*调优] 1月: { {k: round(v) for k, v in jan_cost.items()} } -> q*={q_star}')

LB_pers = np.roll(L_act, 1, axis=0); LB_pers[0] = L_typ
PB_pers = pv_basis('pers')
t0 = time.time()
M1 = chain(LB_pers, PB_pers)
M7 = chain(LB_pers, PB_pers, qlevel=0.6)
M8 = chain(LB_new, PB_new, qlevel=q_star)
soc0 = 6000.0; cp3 = np.zeros(n)
for i in range(n):
    rr = solve_day(price, L_act[i], PV_act[i], soc0, soc_end=None, v=(0.9 * price.mean() if i < n - 1 else 0.0))
    cp3[i] = rr['gross']; soc0 = rr['E'][-1]
print(f'[性能] 三链+天眼耗时={time.time()-t0:.1f}s')

m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
def show(name, M):
    print(f'{name}: 计划费={M[5][m:].sum():,.0f} + 紧急费={M[7][m:].sum():,.0f} = 总{(M[5]+M[7])[m:].sum():,.0f} 元 | 紧急电量={M[6][m:].sum():,.0f} kWh')
show('M1 昨日外推(无裕度)   ', M1)
show('M7 昨日外推+q60(旧采用)', M7)
show(f'M8 {lk_star}/{pk_star}+q{q_star}(新)', M8)
print(f'M3 天眼下界: {cp3[m:].sum():,.0f} 元')
t1, t7, t8 = (M1[5] + M1[7])[m:].sum(), (M7[5] + M7[7])[m:].sum(), (M8[5] + M8[7])[m:].sum()
print(f'[结论] M8对M7再节费={t7-t8:,.0f} 元 ({(t7-t8)/t7*100:.2f}%); 对M1节费={t1-t8:,.0f} 元; 距天眼={t8-cp3[m:].sum():,.0f} 元')
adopt = M8 if t8 < t7 else M7
print(f'[采用] {"M8(同类星期平均)" if adopt is M8 else "M7"}')

# ---------- 写 result2 ----------
res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in dates365])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
P_r, C_r, D_r = adopt[0][i0:i1], adopt[1][i0:i1], adopt[2][i0:i1]
s0_r, s24_r, cp_r, E_r = adopt[3][i0:i1], adopt[4][i0:i1], adopt[5][i0:i1], adopt[6][i0:i1]
emerg = []
for i in range(len(res_dates)):
    idx = np.where(E_r[i] > 1e-6)[0] + 1
    emerg.append([(span_label(a, b), float(E_r[i][a - 1:b].sum())) for a, b in merge_spans(idx)])
write_result2('result2.xlsx', res_dates, P_r, cp_r, s0_r, s24_r, C_r, D_r, emerg)

# ---------- 论文表 (采用方法, 4个指定日期) ----------
spec = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
slot_of = {'10:00-10:10': 61, '12:00-12:10': 73, '14:00-14:10': 85, '16:00-16:10': 97, '18:00-18:10': 109, '20:00-20:10': 121}
paper = {}
for ds in spec:
    k = list(dates365).index(pd.Timestamp(ds)) - i0
    paper[ds] = dict(
        t1={lab: round(float(P_r[k, v - 1]), 2) for lab, v in slot_of.items()},
        total=round(float(P_r[k].sum()), 2), cost=round(float(cp_r[k]), 2),
        blocks=[(round(float(C_r[k, b * 24:(b + 1) * 24].sum()), 2), round(float(D_r[k, b * 24:(b + 1) * 24].sum()), 2)) for b in range(6)],
        soc0=round(float(s0_r[k]), 2), soc24=round(float(s24_r[k]), 2),
        emerg=[(span_label(a, b), round(float(E_r[k][a - 1:b].sum()), 2)) for a, b in merge_spans(np.where(E_r[k] > 1e-6)[0] + 1)])
print('[论文表]', json.dumps(paper, ensure_ascii=False, indent=1))

# ---------- 图: 依据对比 ----------
fig, ax = plt.subplots(figsize=(9, 3.6))
labels = ['昨日外推', '近2同类日', '近4同类日', '近8同类日', '近4同类(五/六分类)']
maes = [np.abs(L_act[jan] - (np.roll(L_act, 1, axis=0)[jan])).mean()] + \
       [np.abs(L_act[jan] - load_cands[k][jan]).mean() for k in ['dow7K2', 'dow7K4', 'dow7K8', 'dow2K4']]
ax.bar(labels, maes, color=[GRAY, '#56B4E9', BLUE, '#0072B2', GREEN])
ax.set_ylabel('1月负载预测 MAE (kWh/时段)')
ax.set_title('计划依据对比: 同类星期平均显著优于昨日外推', fontsize=10)
ax.grid(alpha=0.3, axis='y', ls='--')
for k, v in enumerate(maes):
    ax.text(k, v + 1, f'{v:.1f}', ha='center', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_q2_basis.png')); plt.close(fig)

try:
    prev = json.load(open(os.path.join(DATA, 'q2_results.json'), encoding='utf-8'))
except Exception:
    prev = {}
prev.update(dict(best=('M8' if adopt is M8 else 'M7'), basis=dict(load=lk_star, pv=pk_star, q=q_star),
                 tots_new=dict(M1=float(t1), M7=float(t7), M8=float(t8), M3=float(cp3[m:].sum())),
                 paper=paper))
with open(os.path.join(DATA, 'q2_results.json'), 'w', encoding='utf-8') as f:
    json.dump(prev, f, ensure_ascii=False, indent=1)
print('Q2 FORECAST UPGRADE DONE')
