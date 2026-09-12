# -*- coding: utf-8 -*-
"""敏感性分析(部分数据: Q1典型日 + 2月28天)
1) 充放电效率: 单程0.9/0.9 vs 往返0.9(单程sqrt0.9)
2) 日末残值v: 0.9*次日均价 / v=0 / 日末SOC=起始(等式)
3) Q3预报插值: 线性 vs 分段常数
4) Q3惩罚记账: B(退还50%) vs A(计划全额+额外罚)
5) Q2结算方式: 储能再调度 vs 按计划执行
6) 报童分位q: 0.5/0.6/0.7/0.8 (2月外样本)
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

DATA = os.path.join(ROOT, r'04_建模求解\data')
EPS = 1e-4

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
L_prev = pers.pivot(index='date', columns='slot', values='load_prev_kw').sort_index().to_numpy() / 6.0
PV_prev = pers.pivot(index='date', columns='slot', values='pv_prev_kw').sort_index().to_numpy() / 6.0
fc0m = fc0.pivot(index='date', columns='slot', values='fc0_kw').sort_index().to_numpy() / 6.0
fca_m = {h: fca[fca.issue_hour == h].pivot(index='date', columns='slot', values='fc_kw').sort_index().reindex(columns=range(1, 145)).to_numpy() / 6.0 for h in (6, 12, 18)}
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
rows = []

def add(scenario, metric, value, note=''):
    rows.append(dict(情景=scenario, 指标=metric, 数值=round(float(value), 2), 说明=note))

# ============ 1) 效率 (Q1典型日) ============
import engine as eng
base_q1 = solve_day(price, L_typ, PV_typ, 6000.0, soc_end=6000.0)['gross']
eng.ETA_C = eng.ETA_D = np.sqrt(0.9)
q1_rt = solve_day(price, L_typ, PV_typ, 6000.0, soc_end=6000.0)['gross']
eng.ETA_C = eng.ETA_D = 0.9
add('Q1 效率: 单程0.9/0.9 (基准)', '典型日购电费(元)', base_q1)
add('Q1 效率: 往返0.9 (单程√0.9)', '典型日购电费(元)', q1_rt, f'变化 {q1_rt-base_q1:+.0f} 元 ({(q1_rt-base_q1)/base_q1*100:+.2f}%)')

# ============ 2) 残值 v (Q2式, 2月, M1口径) ============
feb = (dates365.month == 2)
def q2_feb(vmode):
    soc0 = 6000.0; tot = 0.0
    for i in np.where(feb)[0]:
        v = 0.9 * price.mean() if vmode == 'v_next' else 0.0
        if vmode == 'eq':
            r0 = solve_day(price, L_prev[i], PV_prev[i], soc0, soc_end=soc0)
        else:
            r0 = solve_day(price, L_prev[i], PV_prev[i], soc0, soc_end=None, v=v)
        E_ = np.maximum(L_act[i] - PV_act[i] - r0['P'] - r0['D'] + r0['C'], 0)
        tot += float(price @ r0['P'] + 5.0 * price @ E_)
        soc0 = r0['E'][-1]
    return tot
t_vn = q2_feb('v_next'); t_v0 = q2_feb('v0'); t_eq = q2_feb('eq')
add('Q2 残值: v=0.9*次日均价 (基准)', '2月总费用(元)', t_vn)
add('Q2 残值: v=0 (日末清空)', '2月总费用(元)', t_v0, f'变化 {t_v0-t_vn:+.0f} ({(t_v0-t_vn)/t_vn*100:+.2f}%)')
add('Q2 残值: 日末SOC=日初 (等式)', '2月总费用(元)', t_eq, f'变化 {t_eq-t_vn:+.0f} ({(t_eq-t_vn)/t_vn*100:+.2f}%)')

# ============ 3) Q3 插值口径 (2月, 仅0:00计划+按计划执行结算) ============
# 分段常数: 时段s取"该小时末整点"预报值
fc0_pc = np.zeros_like(fc0m)
for i in range(n):
    for s in range(144):
        h = s // 6 + 1
        fc0_pc[i, s] = fc0m[i, min(h * 6 - 1, 143)] if h <= 23 else fc0m[i, 143]
def q3_feb(use_pc):
    soc0 = 6000.0; tot = 0.0
    f = fc0_pc if use_pc else fc0m
    for i in np.where(feb)[0]:
        r0 = solve_day(price, L_prev[i], f[i], soc0, soc_end=None, v=0.9 * price.mean())
        E_ = np.maximum(L_act[i] - PV_act[i] - r0['P'] - r0['D'] + r0['C'], 0)
        tot += float(price @ r0['P'] + 5.0 * price @ E_)
        soc0 = r0['E'][-1]
    return tot
t_lin = q3_feb(False); t_pc = q3_feb(True)
add('Q3 插值: 线性 (基准)', '2月总费用(元)', t_lin)
add('Q3 插值: 分段常数', '2月总费用(元)', t_pc, f'变化 {t_pc-t_lin:+.0f} ({(t_pc-t_lin)/t_lin*100:+.2f}%)')

# ============ 4) Q3 惩罚记账 A vs B (2月, 全滚动) ============
def q3_feb_accounting(reading_A):
    soc0 = 6000.0; tot = 0.0
    for i in np.where(feb)[0]:
        v = 0.9 * price.mean()
        r0 = solve_day(price, L_prev[i], fc0m[i], soc0, soc_end=None, v=v)
        P0, E0 = r0['P'], r0['E']
        P, C, D, E = P0.copy(), r0['C'].copy(), r0['D'].copy(), E0.copy()
        S = np.zeros(144); X = np.zeros(144)
        for h0, fcv in [(36, fca_m[6]), (72, fca_m[12]), (108, fca_m[18])]:
            rr = solve_day(price[h0:], L_prev[i, h0:], fcv[i, h0:], E[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = rr['P'], rr['C'], rr['D']
            E = np.concatenate([E[:h0 + 1], rr['E'][1:]])
            S[h0:], X[h0:] = rr['S'], rr['X']
        E_ = np.maximum(L_act[i] - PV_act[i] - P - D + C, 0)
        if reading_A:
            tot += float(price @ P0 + 0.5 * price @ S + 1.5 * price @ X + 5.0 * price @ E_)
        else:
            tot += float(price @ (P0 - S) + 0.5 * price @ S + 1.5 * price @ X + 5.0 * price @ E_)
        soc0 = E[-1]
    return tot
t_B = q3_feb_accounting(False); t_A = q3_feb_accounting(True)
add('Q3 记账: B(计划>调整退50%) (基准)', '2月总费用(元)', t_B)
add('Q3 记账: A(计划全额+另罚)', '2月总费用(元)', t_A, f'变化 {t_A-t_B:+.0f} ({(t_A-t_B)/t_B*100:+.2f}%)')

# ============ 5) Q2 结算方式 (2月) ============
def q2_feb_settle(redispatch):
    soc0 = 6000.0; tot = 0.0
    for i in np.where(feb)[0]:
        v = 0.9 * price.mean()
        r0 = solve_day(price, L_prev[i], PV_prev[i], soc0, soc_end=None, v=v)
        if redispatch:
            h = 144
            iC, iD, iE, iS = 0, h, 2 * h, 3 * h
            N = 4 * h + (h + 1)
            c = np.zeros(N); c[iE:iE + h] = 5.0 * price; c[iC:iC + h] = EPS; c[iD:iD + h] = EPS; c[iS + h] = -v
            Aub, bub, Aeq, beq = [], [], [], []
            for t in range(h):
                row = np.zeros(N); row[iE + t], row[iC + t], row[iD + t] = -1, 1, -1
                Aub.append(row); bub.append(PV_act[i, t] + r0['P'][t] - L_act[i, t])
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
            Em = res.x[iE:iE + h]; s_end = res.x[iS + h]
            tot += float(price @ r0['P'] + 5.0 * price @ Em)
            soc0 = s_end
        else:
            Em = np.maximum(L_act[i] - PV_act[i] - r0['P'] - r0['D'] + r0['C'], 0)
            tot += float(price @ r0['P'] + 5.0 * price @ Em)
            soc0 = r0['E'][-1]
    return tot
t_rd = q2_feb_settle(True); t_as = q2_feb_settle(False)
add('Q2 结算: 储能再调度 (基准)', '2月总费用(元)', t_rd)
add('Q2 结算: 按计划执行', '2月总费用(元)', t_as, f'变化 {t_as-t_rd:+.0f} ({(t_as-t_rd)/t_rd*100:+.2f}%)')

# ============ 6) 报童分位 q (2月, 滚动30天) ============
r_ = (L_act - PV_act) - (L_prev - PV_prev)
def q2_feb_q(qlevel):
    soc0 = 6000.0; tot = 0.0
    q = np.zeros(144)
    for i in range(n):
        lo = max(0, i - 30)
        if i - lo >= 7:
            q = np.quantile(r_[lo:i], qlevel, axis=0)
        if feb[i]:
            Lb = L_prev[i] + q
            v = 0.9 * price.mean()
            r0 = solve_day(price, Lb, PV_prev[i], soc0, soc_end=None, v=v)
            Em = np.maximum(L_act[i] - PV_act[i] - r0['P'] - r0['D'] + r0['C'], 0)
            tot += float(price @ r0['P'] + 5.0 * price @ Em)
            soc0 = r0['E'][-1]
    return tot
for ql in [0.5, 0.6, 0.7, 0.8]:
    add(f'Q2 报童分位 q={ql}', '2月总费用(元)', q2_feb_q(ql))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA, 'sensitivity.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 60)
print(df.to_string(index=False))
print('SENS DONE')
