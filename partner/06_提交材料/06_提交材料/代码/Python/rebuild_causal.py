# -*- coding: utf-8 -*-
"""因果版重算(修正): 实时平衡控制结算 —— 逐时段用当前实测填补缺口/吸收富余
   因果性: 时段 t 的决策只用 t 时刻及之前的实测(读表)与计划量, 不使用未来实测
   分开建模: 计划模型(0:00 定 P0 与储能计划) + 结算模型(实时平衡) 两层独立
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D, merge_spans, span_label
from io_results import load_all, write_result2, write_result3, dates_of_results, ROOT

DATA = os.path.join(ROOT, r'04_建模求解\data')
day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
PR = day.pivot(index='date', columns='slot', values='price_q4').sort_index().to_numpy()
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365]); n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0; PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
fc0m = fc0.pivot(index='date', columns='slot', values='fc0_kw').sort_index().to_numpy() / 6.0
fca_m = {h: fca[fca.issue_hour == h].pivot(index='date', columns='slot', values='fc_kw').sort_index().reindex(columns=range(1, 145)).to_numpy() / 6.0 for h in (6, 12, 18)}
Lb = np.zeros_like(L_act); Pb = np.zeros_like(PV_act)
for i in range(n):
    low = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low) == (dow[i] in low)]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
    h2 = list(range(max(0, i - 4), i))
    Pb[i] = PV_act[h2].mean(axis=0) if h2 else PV_typ
eNet = (L_act - Lb) - (PV_act - Pb)
m2 = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]

def buffer_of(i, q):
    lo = max(0, i - 30)
    return np.quantile(eNet[lo:i], q, axis=0) if i - lo >= 7 else np.zeros(144)

def balance(P0, Cpl, Dpl, soc0, L, PV, reserve=SOC_MIN):
    h = len(P0)
    C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        # ① 计划量裁剪为物理可行
        c_ok = min(max(float(Cpl[t]), 0.0), P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C))
        d_ok = min(max(float(Dpl[t]), 0.0), P_MAX, ETA_D * max(0.0, soc - reserve + ETA_C * c_ok))
        C[t], D[t] = c_ok, d_ok
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        # ② 缺额: 先减充, 再增放
        if gap > 1e-9:
            cut = min(C[t], gap); C[t] -= cut; gap -= cut
            if gap > 1e-9:
                avail = ETA_D * max(0.0, soc - reserve + ETA_C * C[t]) - D[t]
                extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, avail))
                D[t] += extra; gap -= extra
        # ③ 富余: 先减放, 再增充
        elif gap < -1e-9:
            cut = min(D[t], -gap); D[t] -= cut; gap += cut
            if gap < -1e-9:
                room = max(0.0, (SOC_MAX - soc + D[t] / ETA_D)) / ETA_C - C[t]
                extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, room))
                C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
        assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6, ('SOC越界', t, soc)
        assert C[t] <= P_MAX + 1e-9 and D[t] <= P_MAX + 1e-9, ('功率越界', t)
        assert min(C[t], D[t]) <= 1e-9, ('同时充放', t)
    return C, D, soc, Em

def run_q2(p_matrix=None, q=0.7, reserve=1200.0, upto=n):
    soc = 6000.0
    P_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144)); E_m = np.zeros((n, 144))
    s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = price if p_matrix is None else p_matrix[i]
        if p_matrix is None:
            v = 0.9 * price.mean() if i < n - 1 else 0.0        # 固定电价: 日末残值 = 0.9*日均价
        else:
            v = 0.9 * p_matrix[min(i + 1, n - 1)].mean() if i < n - 1 else 0.0
        buf = buffer_of(i, q)
        r = solve_day(p, Lb[i] + buf, Pb[i], soc, soc_end=None, v=v)
        C, D, sEnd, Em = balance(r['P'], r['C'], r['D'], soc, L_act[i], PV_act[i], reserve)
        P_m[i] = r['P']; C_m[i] = C; D_m[i] = D; E_m[i] = Em
        pc[i] = float(p @ r['P']); ec[i] = float(5 * p @ Em)
        s0[i] = soc; s24[i] = sEnd; soc = sEnd
        if i + 1 >= upto: break
    return dict(P=P_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, ec=ec,
                total=float((pc + ec)[m2:upto].sum()), plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()))

print('=== 1月调优(因果-实时平衡结算) ===')
best = (np.inf, None)
for q in [0.6, 0.7, 0.8]:
    for rsv in [1200.0, 4000.0]:
        R = run_q2(q=q, reserve=rsv, upto=31)
        jan = float((R['pc'] + R['ec'])[7:31].sum())
        print(f'  q={q:.1f} 保护下限={rsv:.0f} -> 1月费用 {jan:,.0f} 元')
        if jan < best[0]: best = (jan, (q, rsv))
q_star, rsv_star = best[1]
print(f'[采用] q*={q_star}, 储能保护下限={rsv_star:.0f} kWh')

t0 = time.time()
R2 = run_q2(q=q_star, reserve=rsv_star)
print(f'\n问题二(因果-实时平衡, 2.1-12.31): 计划费 {R2["plan"]:,.0f} + 紧急费 {R2["em"]:,.0f} = 总 {R2["total"]:,.0f} 元 ({time.time()-t0:.1f}s)')

def run_q3(p_matrix=None, q=0.7, reserve=1200.0):
    soc = 6000.0; H = [36, 72, 108]; fh = [6, 12, 18]
    P0_m = np.zeros((n, 144)); Pf_m = np.zeros((n, 144)); C_m = np.zeros((n, 144)); D_m = np.zeros((n, 144))
    E_m = np.zeros((n, 144)); s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); dv = np.zeros(n); ec = np.zeros(n)
    for i in range(n):
        pday = price if p_matrix is None else p_matrix[i]
        if p_matrix is None:
            v = 0.9 * price.mean() if i < n - 1 else 0.0
        else:
            v = 0.9 * p_matrix[min(i + 1, n - 1)].mean() if i < n - 1 else 0.0
        buf = buffer_of(i, q); Lbar = Lb[i] + buf
        r0 = solve_day(pday, Lbar, fc0m[i], soc, soc_end=None, v=v)
        P, C, D, SOC = r0['P'], r0['C'], r0['D'], r0['E']
        Pinit = P.copy()
        for h0, hh in zip(H, fh):
            fc = fca_m[hh][i, h0:].copy()
            r = solve_day(pday[h0:], Lbar[h0:], fc, SOC[h0], soc_end=None, v=v, base_P=P[h0:])
            P[h0:], C[h0:], D[h0:] = r['P'], r['C'], r['D']
            SOC = np.concatenate([SOC[:h0 + 1], r['E'][1:]])
        C2, D2, sEnd, Em = balance(P, C, D, soc, L_act[i], PV_act[i], reserve)
        P0_m[i] = Pinit; Pf_m[i] = P; C_m[i] = C2; D_m[i] = D2; E_m[i] = Em
        pc[i] = float(pday @ Pinit)
        dv[i] = float(np.sum(-0.5 * pday * np.maximum(Pinit - P, 0) + 1.5 * pday * np.maximum(P - Pinit, 0)))
        ec[i] = float(5 * pday @ Em)
        s0[i] = soc; s24[i] = sEnd; soc = sEnd
    return dict(P0=P0_m, Pf=Pf_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, dv=dv, ec=ec,
                total=float((pc + dv + ec)[m2:].sum()))

R3 = run_q3(q=q_star, reserve=rsv_star)
print(f'问题三(因果-实时平衡): 计划 {R3["pc"][m2:].sum():,.0f} + 调整 {R3["dv"][m2:].sum():,.0f} + 紧急 {R3["ec"][m2:].sum():,.0f} = 总 {R3["total"]:,.0f} 元')
R42 = run_q2(p_matrix=PR, q=q_star, reserve=rsv_star)
print(f'问题四之二(波动电价, 因果): 总 {R42["total"]:,.0f} 元')
R43 = run_q3(p_matrix=PR, q=q_star, reserve=rsv_star)
print(f'问题四之三(波动电价, 因果): 总 {R43["total"]:,.0f} 元')

# ---- 写结果文件 ----
res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in dates365])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
seg = lambda E, i: [(span_label(a, b), float(E[i][a - 1:b].sum())) for a, b in merge_spans(np.where(E[i] > 1e-6)[0] + 1)]
write_result2('result2.xlsx', res_dates, R2['P'][i0:i1], R2['pc'][i0:i1], R2['s0'][i0:i1], R2['s24'][i0:i1],
              R2['C'][i0:i1], R2['D'][i0:i1], [seg(R2['Em'], i0 + k) for k in range(len(res_dates))])
write_result3('result3.xlsx', res_dates, R3['P0'][i0:i1], R3['pc'][i0:i1], R3['Pf'][i0:i1], (R3['pc'] + R3['dv'])[i0:i1],
              R3['s0'][i0:i1], R3['s24'][i0:i1], R3['C'][i0:i1], R3['D'][i0:i1],
              [seg(R3['Em'], i0 + k) for k in range(len(res_dates))])
write_result2('result4-2.xlsx', res_dates, R42['P'][i0:i1], R42['pc'][i0:i1], R42['s0'][i0:i1], R42['s24'][i0:i1],
              R42['C'][i0:i1], R42['D'][i0:i1], [seg(R42['Em'], i0 + k) for k in range(len(res_dates))],
              template='result4-2.xlsx')
write_result3('result4-3.xlsx', res_dates, R43['P0'][i0:i1], R43['pc'][i0:i1], R43['Pf'][i0:i1], (R43['pc'] + R43['dv'])[i0:i1],
              R43['s0'][i0:i1], R43['s24'][i0:i1], R43['C'][i0:i1], R43['D'][i0:i1],
              [seg(R43['Em'], i0 + k) for k in range(len(res_dates))], template='result4-3.xlsx')
with open(os.path.join(DATA, 'causal_rebuild.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(q_star=q_star, reserve=rsv_star, q2=R2['total'], q2_plan=R2['plan'], q2_em=R2['em'],
                   q2_em_kwh=float(R2['Em'][m2:].sum()), q3=R3['total'], q42=R42['total'], q43=R43['total']),
              f, ensure_ascii=False, indent=1)
print('\n因果版(实时平衡结算) 5 个结果文件已更新')
