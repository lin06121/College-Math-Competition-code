# -*- coding: utf-8 -*-
"""第二问正式重算(口径 C + 尾部情景两阶段随机规划) -> 写出 result2.xlsx
   并在 2 月微调情景规模/尾部比例
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import merge_spans, span_label
from causal_core import D, price, PR, ROOT
from io_results import load_all, write_result2, dates_of_results
import fc_upgraded as FU
import plan_core as PC
from lstp import plan_lstp

n, H = D.n, 144
m2 = D.m2
L_ACT, PV_ACT = D.L_act, D.PV_act
DATA = os.path.join(ROOT, '04_建模求解', 'data')
L, _ = FU.build_load()
PV = FU.build_pv()


def scen(i, K=30, frac=0.5):
    """尾部情景(与 2 月扫描中的 M6 同构): 取窗口内"净残差日总量"最大的 S 天作为情景"""
    days = list(range(max(0, i - K), i))
    if not days:
        return []
    S = max(1, int(round(frac * len(days))))
    score = [float((L_ACT[j] - L[j]).sum() - (PV_ACT[j] - PV[j]).sum()) for j in days]
    order = list(np.argsort(score)[::-1])
    idx = sorted(days[k] for k in order[:S])
    return [(L_ACT[j] - L[j], PV_ACT[j] - PV[j]) for j in idx]


def chain(pmat=None, K=30, frac=0.5, upto=None, q_extra=0.0, ret_out=False):
    upto = n if upto is None else upto
    pm = np.broadcast_to(price, (n, H)) if pmat is None else pmat
    B = PC.buffer_mat(L, PV, 0.8) if q_extra > 0 else None
    soc = 6000.0
    P_m = np.zeros((n, H)); C_m = np.zeros((n, H)); D_m = np.zeros((n, H))
    E_m = np.zeros((n, H)); s0 = np.zeros(n); s24 = np.zeros(n); pc = np.zeros(n); ec = np.zeros(n)
    for i in range(upto):
        p = pm[i]
        v = (0.9 * price.mean() if pmat is None else 0.9 * pm[min(i + 1, n - 1)].mean()) if i < n - 1 else 0.0
        rs = scen(i, K=K, frac=frac)
        if not rs:
            r = PC.solve_day(p, L[i], PV[i], soc, soc_end=None, v=v)
            P, C, Dv = r['P'], r['C'], r['D']
        else:
            o = plan_lstp(p, L[i] + (q_extra * B[i] if B is not None else 0.0), PV[i], rs, soc, v, S=len(rs))
            P, C, Dv = o['P'], o['C'], o['D']
        e, curt, sEnd = PC.execute_as_planned(P, C, Dv, soc, L_ACT[i], PV_ACT[i])
        P_m[i], C_m[i], D_m[i], E_m[i] = P, C, Dv, e
        pc[i] = float(p @ P); ec[i] = float(5 * p @ e)
        s0[i], s24[i], soc = soc, sEnd, sEnd
    return dict(P=P_m, C=C_m, D=D_m, Em=E_m, s0=s0, s24=s24, pc=pc, ec=ec,
                plan=float(pc[m2:upto].sum()), em=float(ec[m2:upto].sum()),
                total=float((pc + ec)[m2:upto].sum()), em_kwh=float(E_m[m2:upto].sum()))


if __name__ == '__main__':
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == 'tune':
        print('=== 2 月微调: 情景窗口 K 与情景天数比例 frac ===')
        best = (1e18, None)
        for K in (30, 60):
            for frac in (0.2, 0.3, 0.4, 0.5, 0.7):
                R = chain(K=K, frac=frac, upto=59)
                print(f'  K={K:3d} frac={frac:.2f} (S={max(1,int(round(frac*K))):2d}): {R["total"]:>12,.0f} 元'
                      f' (计划 {R["plan"]:,.0f} + 紧急 {R["em"]:,.0f}) | {time.time()-t0:.0f}s', flush=True)
                if R['total'] < best[0]:
                    best = (R['total'], (K, frac))
        print(f'[最优] K={best[1][0]}, frac={best[1][1]} -> {best[0]:,.0f} 元')
        json.dump(dict(K=best[1][0], frac=best[1][1], feb_total=best[0]),
                  open(os.path.join(DATA, 'q2_best_cfg.json'), 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        sys.exit(0)
    K, frac = 30, 0.5
    if os.path.exists(os.path.join(DATA, 'q2_best_cfg.json')):
        cfg = json.load(open(os.path.join(DATA, 'q2_best_cfg.json'), encoding='utf-8'))
        K, frac = cfg['K'], cfg['frac']
    print(f'=== 全年重算(尾部情景两阶段, K={K}, frac={frac}) ===')
    R2 = chain(K=K, frac=frac)
    print(f'问题二: 计划 {R2["plan"]:,.0f} + 紧急 {R2["em"]:,.0f} = {R2["total"]:,.0f} 元'
          f' | 紧急电量 {R2["em_kwh"]:,.0f} kWh | {time.time()-t0:.0f}s')
    R42 = chain(pmat=PR, K=K, frac=frac)
    print(f'问题四之二: 计划 {R42["plan"]:,.0f} + 紧急 {R42["em"]:,.0f} = {R42["total"]:,.0f} 元')
    day, typ, fc0, fca, pers, tmpl, const = load_all()
    res_dates = dates_of_results()
    mask = np.array([d in set(pd.to_datetime(res_dates)) for d in D.dates])
    i0 = np.where(mask)[0][0]; i1 = np.where(mask)[0][-1] + 1
    seg = lambda E, i: [(span_label(a, b), float(E[i][a - 1:b].sum()))
                        for a, b in merge_spans(np.where(E[i] > 1e-6)[0] + 1)]
    write_result2('result2.xlsx', res_dates, R2['P'][i0:i1], R2['pc'][i0:i1], R2['s0'][i0:i1], R2['s24'][i0:i1],
                  R2['C'][i0:i1], R2['D'][i0:i1], [seg(R2['Em'], i0 + k) for k in range(len(res_dates))])
    write_result2('result4-2.xlsx', res_dates, R42['P'][i0:i1], R42['pc'][i0:i1], R42['s0'][i0:i1],
                  R42['s24'][i0:i1], R42['C'][i0:i1], R42['D'][i0:i1],
                  [seg(R42['Em'], i0 + k) for k in range(len(res_dates))], template='result4-2.xlsx')
    json.dump(dict(K=K, frac=frac, q2=dict(plan=R2['plan'], em=R2['em'], total=R2['total'],
                                           em_kwh=R2['em_kwh']),
                   q42=dict(plan=R42['plan'], em=R42['em'], total=R42['total'],
                            em_kwh=R42['em_kwh'])),
              open(os.path.join(DATA, 'q2_lstp_results.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('written result2.xlsx / result4-2.xlsx')
