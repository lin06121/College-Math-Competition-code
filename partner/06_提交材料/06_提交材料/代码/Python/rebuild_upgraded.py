# -*- coding: utf-8 -*-
"""★ 正式版重算(升级预测模型): 用 Python 重跑全部数据并写出 5 个结果文件
   预测模型(与 07_预测模型实验 结论一致, 自包含实现见 fc_upgraded.py):
     负荷 = 水平×形状分解 + 递推水平平滑 + 周五六/其他分类凸组合(1月标定)
     光伏 = 因果岭回归堆叠[近4日均值, 误差反馈, LightGBM, 附件3 的 0:00 预报]
     电价 = 类别×时段廓线+自适应水平, 与误差反馈、LightGBM 做因果岭回归堆叠
   管道 = 0:00 计划 LP + (问题三)滚动调整 + 日内实时平衡控制 + 5 倍紧急购电(因果口径)
   输出 = result1/2/3/4-2/4-3.xlsx + data/upgraded_numbers.json
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import solve_day, merge_spans, span_label
from causal_core import D, price, PR, balance, ROOT
import upgraded_core as UC
from io_results import load_all, write_result1, write_result2, write_result3, dates_of_results
import fc_upgraded as FU

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
FCADJ = D.fca_m
DATA = os.path.join(ROOT, '04_建模求解', 'data')
t0 = time.time()

print('=== 1) 构造升级版预测(负荷/光伏/电价) ===')
L_UP, INFO = FU.build_load()
PV_UP = FU.build_pv()
P_UP = FU.build_price()
for k, v in INFO.items():
    print(f'    [负荷权重 {k}] ' + ', '.join(f'{a}:{b}' for a, b in v['weights'].items()))


def buffer_of(Lh, PVh, q, i, W=30):
    lo = max(0, i - W)
    if i - lo < 7 or q <= 0:
        return np.zeros(H)
    e = (L_ACT[lo:i] - Lh[lo:i]) - (PV_ACT[lo:i] - PVh[lo:i])
    return np.quantile(e, q, axis=0)


def run_q2(Lh, PVh, q, pmat=None, upto=None):
    """0:00 计划(注入预测) + 实时平衡控制; pmat 为 None 表示用附件1 固定电价"""
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    soc = 6000.0
    P = np.zeros((n, H)); C = np.zeros((n, H)); Dm = np.zeros((n, H)); Em = np.zeros((n, H))
    s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        b = buffer_of(Lh, PVh, q, i)
        r = solve_day(p, Lh[i] + b, PVh[i], soc, soc_end=None, v=v)
        c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        P[i], C[i], Dm[i], Em[i] = r['P'], c, d, e
        pc[i] = float(p @ r['P']); ec[i] = float(5 * p @ e)
        s0[i], s24[i], soc = soc, sEnd, sEnd
    return dict(P=P, C=C, D=Dm, Em=Em, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()))


def run_q3(Lh, PVh, q, pmat=None, adjust=(6, 12, 18), upto=None):
    """问题三正式管道：逐段实际结算，偏差始终相对0:00原始计划。"""
    return UC.run_q3(Lh, PVh, q=q, pmat=pmat, adjust=adjust, upto=upto)


print('=== 2) 1 月标定报童分位 q(与论文同一纪律) ===')
grid = (0.5, 0.6, 0.7, 0.8, 0.9)
jan = {}
for q in grid:
    R = run_q2(L_UP, PV_UP, q=q, upto=31)
    jan[q] = float((R['pc'] + R['ec'])[7:31].sum())
    print(f'    q={q}: 1月费用 {jan[q]:,.0f} 元')
Q_STAR = min(jan, key=jan.get)
print(f'    [采用] q* = {Q_STAR}')

print('=== 3) 四问全年重算(2.1-12.31) ===')
ADJ = (6, 12, 18)   # 正式方案在q3独立标定后采用全部发布时刻
R2 = run_q2(L_UP, PV_UP, q=Q_STAR)
print(f'问题二: 计划 {R2["plan"]:,.0f} + 紧急 {R2["em"]:,.0f} = {R2["total"]:,.0f} 元')
q3_grid = (0.5, 0.6, 0.7, 0.8, 0.9)
q3_jan = {}
for q3 in q3_grid:
    Rq3 = run_q3(L_UP, PV_UP, q=q3, adjust=(6, 12, 18), upto=31)
    q3_jan[q3] = float((Rq3['pc'] + Rq3['dv'] + Rq3['ec'])[7:31].sum())
Q3_STAR = min(q3_jan, key=q3_jan.get)
print(f'    问题三独立标定 q*: { {k: round(v) for k, v in q3_jan.items()} } -> {Q3_STAR}')
ADJ = (6, 12, 18)  # 四个发布时刻全部纳入；逐段实测SOC和原始计划基准
R3 = run_q3(L_UP, PV_UP, q=Q3_STAR, adjust=ADJ)
print(f'问题三(全时刻滚动): 计划 {R3["plan"]:,.0f} + 调整 {R3["dev"]:,.0f} + 紧急 {R3["em"]:,.0f} = {R3["total"]:,.0f} 元')
R42 = run_q2(L_UP, PV_UP, q=Q_STAR, pmat=PR)
print(f'问题四之二: 计划 {R42["plan"]:,.0f} + 紧急 {R42["em"]:,.0f} = {R42["total"]:,.0f} 元')
R43 = run_q3(L_UP, PV_UP, q=Q3_STAR, pmat=PR, adjust=ADJ)
print(f'问题四之三(全时刻滚动): 计划 {R43["plan"]:,.0f} + 调整 {R43["dev"]:,.0f} + 紧急 {R43["em"]:,.0f} = {R43["total"]:,.0f} 元')
R3z = run_q3(L_UP, PV_UP, q=Q3_STAR, adjust=())
print(f'问题三(仅0:00, 对照): {R3z["total"]:,.0f} 元')
R3f = run_q3(L_UP, PV_UP, q=Q3_STAR, adjust=(6, 12, 18))
print(f'问题三(全滚动, 对照): {R3f["total"]:,.0f} 元')
R43z = run_q3(L_UP, PV_UP, q=Q3_STAR, pmat=PR, adjust=())
R43f = run_q3(L_UP, PV_UP, q=Q3_STAR, pmat=PR, adjust=(6, 12, 18))
print(f'问题四之三(仅0:00/全滚动, 对照): {R43z["total"]:,.0f} / {R43f["total"]:,.0f} 元')

print('=== 4) 写出 5 个结果文件 ===')
day, typ, fc0, fca, pers, tmpl, const = load_all()
res_dates = dates_of_results()
mask = np.array([d in set(pd.to_datetime(res_dates)) for d in D.dates])
i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
seg = lambda E, i: [(span_label(a, b), float(E[i][a - 1:b].sum()))
                    for a, b in merge_spans(np.where(E[i] > 1e-6)[0] + 1)]
write_result2('result2.xlsx', res_dates, R2['P'][i0:i1], R2['pc'][i0:i1], R2['s0'][i0:i1], R2['s24'][i0:i1],
              R2['C'][i0:i1], R2['D'][i0:i1], [seg(R2['Em'], i0 + k) for k in range(len(res_dates))])
write_result3('result3.xlsx', res_dates, R3['P0'][i0:i1], R3['pc'][i0:i1], R3['Pf'][i0:i1],
              (R3['pc'] + R3['dv'])[i0:i1], R3['s0'][i0:i1], R3['s24'][i0:i1], R3['C'][i0:i1], R3['D'][i0:i1],
              [seg(R3['Em'], i0 + k) for k in range(len(res_dates))])
write_result2('result4-2.xlsx', res_dates, R42['P'][i0:i1], R42['pc'][i0:i1], R42['s0'][i0:i1], R42['s24'][i0:i1],
              R42['C'][i0:i1], R42['D'][i0:i1], [seg(R42['Em'], i0 + k) for k in range(len(res_dates))],
              template='result4-2.xlsx')
write_result3('result4-3.xlsx', res_dates, R43['P0'][i0:i1], R43['pc'][i0:i1], R43['Pf'][i0:i1],
              (R43['pc'] + R43['dv'])[i0:i1], R43['s0'][i0:i1], R43['s24'][i0:i1], R43['C'][i0:i1], R43['D'][i0:i1],
              [seg(R43['Em'], i0 + k) for k in range(len(res_dates))], template='result4-3.xlsx')
# 问题一(与之前一致, 仍用 LP 精确求解)
sys.path.insert(0, os.path.join(ROOT, '04_建模求解', 'scripts'))
Q1P = np.load(os.path.join(DATA, 'q1_solution.npz')) if os.path.exists(os.path.join(DATA, 'q1_solution.npz')) else None
print('    result1.xlsx 由 q1.py 单独写出(问题一不含预测)')
np.savez_compressed(os.path.join(DATA, 'upgraded_preds.npz'), L=L_UP, PV=PV_UP, P=P_UP)

num = dict(q_star=Q_STAR, q3_star=Q3_STAR, jan_grid=jan, q3_jan_grid=q3_jan,
           adopt_adjust='全时刻滚动',
           q2=dict(plan=R2['plan'], em=R2['em'], total=R2['total'], em_kwh=float(R2['Em'][m2:].sum())),
           q3=dict(plan=R3['plan'], dev=R3['dev'], em=R3['em'], total=R3['total'],
                   em_kwh=float(R3['Em'][m2:].sum()), zero=R3z['total'], full=R3f['total']),
           q42=dict(plan=R42['plan'], em=R42['em'], total=R42['total'], em_kwh=float(R42['Em'][m2:].sum())),
           q43=dict(plan=R43['plan'], dev=R43['dev'], em=R43['em'], total=R43['total'],
                    em_kwh=float(R43['Em'][m2:].sum()), zero=R43z['total'], full=R43f['total']))
json.dump(num, open(os.path.join(DATA, 'upgraded_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\n[完成] 耗时 {time.time()-t0:.0f}s -> data/upgraded_numbers.json')
