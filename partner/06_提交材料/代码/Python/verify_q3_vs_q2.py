# -*- coding: utf-8 -*-
"""验证: (1) 效率口径(只有进出储能有损耗) (2) 问题三比问题二贵的原因分解
        (3) "免费调整"反事实 (4) 各时刻附件3预报 相对 0:00 依据的真实信息量
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, PR, ROOT, balance
from engine import solve_day, ETA_C, ETA_D
import fc_upgraded as FU
import upgraded_core as UC

DATA = os.path.join(ROOT, '04_建模求解', 'data')
m2 = D.m2
L, INFO = FU.build_load()
PV = FU.build_pv()
Q = json.load(open(os.path.join(DATA, 'upgraded_all_numbers.json'), encoding='utf-8'))['q_star']

print('=' * 78)
print('(1) 效率口径自检: 是否"只有进出储能有损耗, PV/网购直供零损耗"')
print('=' * 78)
Eb = UC.buffer_mat(L, PV, Q)
soc = 6000.0
totC = totD = totEm = totP = totPV = totL = curt = 0.0
max_viol = 0.0
for i in range(m2, D.n):
    p = price
    v = 0.9 * price.mean() if i < D.n - 1 else 0.0
    r = solve_day(p, L[i] + Eb[i], PV[i], soc, soc_end=None, v=v)
    c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
    # 直接供给部分(无效率系数): PV + P + 紧急 + D - C - L = 富余(弃光); 应为 >= 0
    surplus = (D.PV_act[i] + r['P'] + e + d - c) - D.L_act[i]
    max_viol = max(max_viol, -surplus.min())                 # 供给不足量(应为 0)
    curt += float(np.maximum(surplus, 0).sum())
    totC += float(c.sum()); totD += float(d.sum())
    totP += float(r['P'].sum()); totPV += float(D.PV_act[i].sum())
    totL += float(D.L_act[i].sum()); totEm += float(e.sum())
    soc = sEnd
loss_C = totC * (1 / ETA_C - 0) * 0 + totC * (1 - ETA_C)          # 充电侧损耗 = (1-0.9)C
loss_D = totD * (1 / ETA_D - 1)                                   # 放电侧损耗 = (1/0.9-1)D
print(f'  供给充足率检查: 最大"供不应求"量 = {max_viol:.6e} kWh  (0 表示 PV/网购直供无损耗地满足负载)')
print(f'  全年(2.1-12.31) 负载 {totL/1e6:.3f} GWh | 光伏 {totPV/1e6:.3f} GWh | 网购 {totP/1e6:.3f} GWh(含紧急 {totEm/1e3:.1f} MWh)')
print(f'  通过储能: 充电 {totC/1e6:.3f} GWh -> 入 SOC {totC*ETA_C/1e6:.3f} GWh; 放电吐给负载 {totD/1e6:.3f} GWh (SOC 取用 {totD/ETA_D/1e6:.3f} GWh)')
print(f'  储能耗损 = {loss_C/1e3:,.0f} + {loss_D/1e3:,.0f} = {(loss_C+loss_D)/1e3:,.0f} MWh'
      f'  (占网购电量 {(loss_C+loss_D)/totP*100:.2f}%)')
print(f'  弃光 {curt/1e6:.3f} GWh (富余光伏, 无损耗地丢弃)')
print('  结论: 平衡式只有 PV+P+D-C>=L(无效率系数) -> PV/网购直供负载零损耗;')
print('        损耗只出现在 E_{t+1}=E_t+0.9C-D/0.9 的储能路径上(单程 90%, 往返 81%)')

print()
print('=' * 78)
print('(2) 问题二 vs 问题三(统一 q=0.7, 同一 0:00 依据) 费用分解')
print('=' * 78)
R2 = UC.run_q2(L, PV, q=Q)
R3z = UC.run_q3(L, PV, q=Q, adjust=())
R3f = UC.run_q3(L, PV, q=Q, adjust=(6, 12, 18))
R3_12 = UC.run_q3(L, PV, q=Q, adjust=(12,))
R3_6 = UC.run_q3(L, PV, q=Q, adjust=(6,))
R3_18 = UC.run_q3(L, PV, q=Q, adjust=(18,))
S = np.maximum(R3f['P0'] - R3f['Pf'], 0)      # 下调计划量
X = np.maximum(R3f['Pf'] - R3f['P0'], 0)      # 上调计划量
S_kwh = float(S[m2:].sum()); X_kwh = float(X[m2:].sum())
S_cost = float(sum(0.5 * price @ S[i] for i in range(m2, D.n)))
X_cost = float(sum(1.5 * price @ X[i] for i in range(m2, D.n)))
print(f'  问题二(0:00 计划 + 实时平衡)              : {R2["total"]:>13,.0f} 元'
      f'  (计划 {R2["plan"]:,.0f} + 紧急 {R2["em"]:,.0f})')
print(f'  问题三·仅 0:00 计划(不做任何调整)          : {R3z["total"]:>13,.0f} 元  <-- 与问题二完全相同(信息集与管道一致)')
print(f'  问题三·全滚动(6/12/18)                   : {R3f["total"]:>13,.0f} 元'
      f'  (计划 {R3f["plan"]:,.0f} + 调整 {R3f["dev"]:,.0f} + 紧急 {R3f["em"]:,.0f})')
print(f'  ---- 差异分解(全滚动 - 不调整) ----')
print(f'   计划购电费变化 : {R3f["plan"]-R3z["plan"]:>+12,.0f} 元')
print(f'   紧急购电费变化 : {R3f["em"]-R3z["em"]:>+12,.0f} 元   (调整带来的收益: 少买紧急电)')
print(f'   调整相关费     : {R3f["dev"]-R3z["dev"]:>+12,.0f} 元   (下调 {S_kwh:,.0f} kWh 退 50%, 上调 {X_kwh:,.0f} kWh 付 1.5 倍)')
print(f'       其中 下调退费 -{S_cost:,.0f} 元 / 上调加价 +{X_cost:,.0f} 元')
print(f'   合计净变化     : {R3f["total"]-R3z["total"]:>+12,.0f} 元  => {"更贵" if R3f["total"]>R3z["total"] else "更便宜"}')
print(f'  ---- 单时刻调整(各自单独启用) ----')
for nm, R in [('+6:00', R3_6), ('+12:00', R3_12), ('+18:00', R3_18), ('全滚动', R3f)]:
    print(f'   {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f}'
          f' = {R["total"]:>13,.0f} 元  (相对不调整 {R["total"]-R3z["total"]:>+9,.0f})')

print()
print('=' * 78)
print('(3) "免费调整"反事实: 偏差费系数置 0 时, 调整到底值多少钱')
print('=' * 78)
import engine
import causal_core
_orig = engine.solve_day


def solve_day_freefee(price_v, load_v, pv_v, soc0, soc_end=None, v=0.0, base_P=None, t0=0):
    """把调整模式的目标系数改为 0(即把偏差费视作免费, 只保留残值项) —— 用于反事实"""
    if base_P is None:
        return _orig(price_v, load_v, pv_v, soc0, soc_end, v, base_P)
    h = len(price_v)
    nP, nC, nD, nE = h, h, h, h + 1
    nS, nX = h, h
    N = nP + nC + nD + nE + nS + nX
    iC, iD, iE, iS, iX = nP, nP + nC, nP + nC + nD, nP + nC + nD + nE, nP + nC + nD + nE + h
    f = np.zeros(N); f[iE + h] = -v                     # 只需残值项
    Aub, bub, Aeq, beq = [], [], [], []
    for t in range(h):
        row = np.zeros(N); row[t], row[iC + t], row[iD + t] = -1, 1, -1
        Aub.append(row); bub.append(pv_v[t] - load_v[t])
    for t in range(h):
        row = np.zeros(N); row[iE + t], row[iE + t + 1] = -1, 1
        row[iC + t], row[iD + t] = -0.9, 1 / 0.9
        Aeq.append(row); beq.append(0.0)
    r = 0; Aeq.append(np.zeros(N)); beq.append(0.0)
    Aeq[-1][iE] = 1; beq[-1] = soc0
    if soc_end is not None:
        Aeq.append(np.zeros(N)); Aeq[-1][iE + h] = 1; beq.append(soc_end)
    for t in range(h):
        row = np.zeros(N); row[t], row[iS + t], row[iX + t] = 1, 1, -1
        Aeq.append(row); beq.append(base_P[t])
    lb = np.zeros(N); ub = np.full(N, np.inf)
    ub[iC:iC + h] = 833.3333; ub[iD:iD + h] = 833.3333
    lb[iE:iE + h + 1] = 1200; ub[iE:iE + h + 1] = 10800
    lb[iE] = ub[iE] = soc0
    from scipy.optimize import linprog
    res = linprog(f, A_ub=np.array(Aub), b_ub=np.array(bub), A_eq=np.array(Aeq), b_eq=np.array(beq),
                  bounds=list(zip(lb, ub)), method='highs')
    assert res.success
    return dict(status='ok', P=res.x[:h], C=res.x[iC:iC + h], D=res.x[iD:iD + h],
                E=res.x[iE:iE + h + 1], S=res.x[iS:iS + h], X=res.x[iX:iX + h],
                gross=float(price_v @ res.x[:h]), inc_cost=0.0)


engine.solve_day = solve_day_freefee
UC.solve_day = solve_day_freefee
R3_free = UC.run_q3(L, PV, q=Q, adjust=(6, 12, 18))
engine.solve_day = _orig
UC.solve_day = _orig
print(f'  免费调整下的最优计划(同样按真实 5 倍紧急电价结算):')
print(f'    计划 {R3_free["plan"]:,.0f} + 紧急 {R3_free["em"]:,.0f} = {R3_free["plan"]+R3_free["em"]:,.0f} 元'
      f'   (偏差费按题面规则本应付 {R3_free["dev"]:,.0f} 元)')
print(f'    => 若调整免费: 总费用 {R3_free["plan"]+R3_free["em"]:,.0f} 元, 比不调整({R3z["total"]:,.0f})'
      f' 省 {R3z["total"]-(R3_free["plan"]+R3_free["em"]):,.0f} 元')
print(f'    => 计入偏差费后: {R3_free["total"]:,.0f} 元, 反而比不调整贵 {R3_free["total"]-R3z["total"]:,.0f} 元')
print('    (说明: 调整本身是有价值的 —— 能少买紧急电; 但题面 0.5/1.5 倍的偏差计价吃掉了这部分收益)')

print()
print('=' * 78)
print('(4) 各时刻附件3预报 vs 我们的 0:00 依据: 后发预报的"新增信息量"有多大')
print('=' * 78)
FCADJ = D.fca_m
for h0, hh in [(36, 6), (72, 12), (108, 18)]:
    seg = slice(h0, 144)
    e_fcadj = np.abs(FCADJ[hh][m2:, seg] - D.PV_act[m2:, seg]).mean() * 6
    e_ours = np.abs(PV[m2:, seg] - D.PV_act[m2:, seg]).mean() * 6
    e_fc0 = np.abs(D.fc0m[m2:, seg] - D.PV_act[m2:, seg]).mean() * 6
    print(f'  {hh}:00 之后时段({h0}-144)光伏 MAE: 我们的 0:00 依据 {e_ours:6.1f} kW | '
          f'附件3 的 {hh}:00 预报 {e_fcadj:6.1f} kW | 附件3 的 0:00 预报 {e_fc0:6.1f} kW')
print('  说明: 我们的 0:00 依据本身就是"统计 + 附件3 预报"的堆叠, 已经吸收了后发预报的多数信息;')
print('        因此 6/12/18 时的新预报对"改变已锁定计划"的边际信息量有限, 而改动要付偏差费。')
