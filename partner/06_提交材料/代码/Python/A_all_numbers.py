# -*- coding: utf-8 -*-
"""口径 A(最终) 下论文所需全套数字 -> data/A_all_numbers.json"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from engine import solve_day
from causal_core import D, price, PR, ROOT, balance
import A_core as AC

n, H, m2 = AC.n, AC.H, AC.m2
L, PV = AC.L, AC.PV
P_RT = AC.P_RT                                    # 乙: 廓线+自适应+堆叠(对照)
import price_fc as PF
P_HAT, _, PD_PRICE = PF.build_price()             # 甲: 周周期加权+LightGBM 残差(采用)
P_ROLL = PF.build_price_rolling(P_HAT, damp=0.7)  # 6/12/18 滚动修正
_PM_INFO = dict(mae_jia=float(PD_PRICE['mae_final']), mae_yi=float(AC.metrics(P_RT, D.PR, 1)['mae']),
                mae_base=float(PD_PRICE['mae_base']), mae_naive7=float(PD_PRICE['mae_naive7']))
L_ACT, PV_ACT = AC.L_ACT, AC.PV_ACT
DATA = os.path.join(ROOT, '04_建模求解', 'data')
t0 = time.time()
A = json.load(open(os.path.join(DATA, 'A_numbers.json'), encoding='utf-8'))
Q = A['q_star']
Q3 = float(A.get('q3_star', Q))   # 问题三独立标定(1 月, 全时刻滚动口径)
out = dict(main=A, load_weights=AC.LOAD_INFO, price_model=_PM_INFO)

print('=== 预测精度 ===')
for nm, Aha, Aac, sc in [('负荷', L, L_ACT, 6), ('光伏', PV, PV_ACT, 6), ('电价', P_RT, D.PR, 1)]:
    out['mae_' + nm] = dict(new=AC.metrics(Aha, Aac, sc),
                            base=AC.metrics(D.Lb if nm == '负荷' else (D.Pb if nm == '光伏' else P_RT), Aac, sc))
    print(f'  {nm}: 新 {out["mae_"+nm]["new"]["mae"]:.3f} | 基线 {out["mae_"+nm]["base"]["mae"]:.3f}')
e = np.abs(L[m2:] - L_ACT[m2:]).mean(axis=1) * 6
eb = np.abs(D.Lb[m2:] - L_ACT[m2:]).mean(axis=1) * 6
lo = AC.LOW[m2:]
out['load_class'] = dict(new=dict(total=float(e.mean()), low=float(e[lo].mean()), high=float(e[~lo].mean())),
                         base=dict(total=float(eb.mean()), low=float(eb[lo].mean()), high=float(eb[~lo].mean())))

print('=== 问题三 各调整方案(带调整时刻裕度) ===')
opts = {}
for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((18,), '+18:00'),
               ((6, 12), '+6,+12'), ((12, 18), '+12,+18'), ((6, 12, 18), '全滚动')]:
    R = AC.run_q3(q=Q3, adjust=at, use_adjbuf=True)
    opts[nm] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']), total=float(R['total']))
    print(f'  {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
out['q3_options'] = opts
print('=== 问题四之三 各调整方案(电价已知, pmat=附件4 实时电价) ===')
o43 = {}
for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((18,), '+18:00'), ((6, 12, 18), '全滚动')]:
    R = AC.run_q3(q=Q3, pmat=P_HAT, bill=PR, pmat_adj=P_ROLL, adjust=at, use_adjbuf=True)
    o43[nm] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']), total=float(R['total']))
    print(f'  {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
out['q43_options'] = o43
o43p = {}
for at, nm in [((), '仅0:00'), ((12,), '+12:00'), ((6, 12, 18), '全滚动')]:
    R = AC.run_q3(q=Q3, pmat=P_RT, bill=PR, adjust=at, use_adjbuf=True)   # 乙 电价模型对照
    o43p[nm] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']), total=float(R['total']))
    print(f'  [预测电价口径] {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} '
          f'+ 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
out['q43_options_predprice'] = o43p

print('=== 对照方法(问题二) ===')
out['cmp_典型日'] = float(AC.run_q2(np.broadcast_to(D.L_typ, (n, H)),
                                 np.broadcast_to(D.PV_typ, (n, H)), q=Q)['total'])
out['cmp_原依据'] = float(AC.run_q2(D.Lb, D.Pb, q=Q)['total'])   # 同类日均值+近4日光伏均值(旧依据)
out['cmp_无储能'] = float(AC.nostorage(q=Q))
out['cmp_天眼下界'] = float(AC.clairvoyant())
print(f'  典型日 {out["cmp_典型日"]:,.0f} | 原依据 {out["cmp_原依据"]:,.0f} | '
      f'无储能 {out["cmp_无储能"]:,.0f} | 天眼下界 {out["cmp_天眼下界"]:,.0f}')
# 昨日外推(自身基值)
Lp = L_ACT[np.r_[0, np.arange(n - 1)]]; PVp = PV_ACT[np.r_[0, np.arange(n - 1)]]
for tag, qq in [('M7_q0.6', 0.6), ('M5_q0.8', 0.8), ('M1_q0', 0.0)]:
    out['cmp_昨日外推_' + tag] = float(AC.run_q2(Lp, PVp, q=qq)['total'])
    print(f'  昨日外推 {tag}: {out["cmp_昨日外推_"+tag]:,.0f}')

print('=== 执行方式对照(同一计划的三种执行机制) ===')
import engine
def run_exec(mode):
    soc = 6000.0; pc = em = 0.0
    for i in range(n):
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        r = solve_day(price, L[i] + AC.buffer_of(L, PV, Q, i), PV[i], soc, soc_end=None, v=v)
        if mode == 'asplanned':
            c, d = r['C'].copy(), r['D'].copy()
            e = np.maximum(0.0, L_ACT[i] - (PV_ACT[i] + r['P'] + d - c))
            sEnd = float(soc + 0.9 * c.sum() - d.sum() / 0.9)
        else:
            c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        if i >= m2:
            pc += float(price @ r['P']); em += float(5 * price @ e)
        soc = sEnd
    return pc, em
for mode, nm in [('asplanned', '僵硬执行计划'), ('balance', '实时平衡控制(采用)')]:
    pc, em = run_exec(mode)
    out['exec_' + mode] = dict(plan=pc, em=em, total=pc + em)
    print(f'  {nm:16s}: 计划 {pc:>12,.0f} + 紧急 {em:>10,.0f} = {pc+em:>13,.0f} 元')
from causal_core import perfect as _perfect
soc = 6000.0; pc = em = 0.0
for i in range(n):
    v = 0.9 * price.mean() if i < n - 1 else 0.0
    r = solve_day(price, L[i] + AC.buffer_of(L, PV, Q, i), PV[i], soc, soc_end=None, v=v)
    e, sEnd, _ = _perfect(L_ACT[i], PV_ACT[i], r['P'], soc, v, price)
    if i >= m2:
        pc += float(price @ r['P']); em += float(5 * price @ e)
    soc = sEnd
out['exec_perfect'] = dict(plan=pc, em=em, total=pc + em)
print(f'  {"完美信息再调度":16s}: 计划 {pc:>12,.0f} + 紧急 {em:>10,.0f} = {pc+em:>13,.0f} 元')

print('=== 灵敏度 ===')
sens = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    sens[f'q{int(q*10)}'] = float(AC.run_q2(q=q)['total'])
import causal_core as _cc
old = (engine.ETA_C, engine.ETA_D, _cc.ETA_C, _cc.ETA_D)
engine.ETA_C = engine.ETA_D = _cc.ETA_C = _cc.ETA_D = 0.9 ** 0.5   # 往返 0.9 ⇔ 单程 sqrt(0.9)
sens['eta_roundtrip'] = float(AC.run_q2(q=Q)['total'])
engine.ETA_C, engine.ETA_D, _cc.ETA_C, _cc.ETA_D = old
def feb(mode):
    tot = 0.0; soc = 6000.0
    for i in range(59):
        v = 0.0 if mode == 'zero' else (0.9 * price.mean() if i < n - 1 else 0.0)
        r = solve_day(price, L[i] + AC.buffer_of(L, PV, Q, i), PV[i], soc,
                      soc_end=(soc if mode == 'eq' else None), v=v)
        c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        if mode == 'eq':
            sEnd = soc
        if i >= m2:
            tot += float(price @ r['P']) + float(5 * price @ e)
        soc = sEnd
    return tot
sens['feb_base'] = feb('std'); sens['feb_v0'] = feb('zero'); sens['feb_socEq'] = feb('eq')
sens['price_pred'] = float(AC.run_q2(q=Q, pmat=P_HAT, bill=PR)['total'])   # 采用: 预测电价定计划
sens['price_known'] = float(AC.run_q2(q=Q, pmat=PR, bill=PR)['total'])     # 上界: 0:00 已知实际电价(不可达)
R3 = AC.run_q3(q=Q3, adjust=(6, 12, 18), use_adjbuf=True)
sens['rule_A'] = float(R3['plan'] + R3['em']
                       + sum(1.5 * price @ np.maximum(R3['Pf'][i] - R3['P0'][i], 0) for i in range(m2, n)))
sens['rule_B'] = float(R3['total'])
def run_with_B(B):
    soc = 6000.0; pc = em = 0.0
    for i in range(n):
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        r = solve_day(price, L[i] + B[i], PV[i], soc, soc_end=None, v=v)
        c, d, sEnd, e = balance(r['P'], r['C'], r['D'], soc, L_ACT[i], PV_ACT[i])
        if i >= m2:
            pc += float(price @ r['P']); em += float(5 * price @ e)
        soc = sEnd
    return pc + em
e_ = (L_ACT - L) - (PV_ACT - PV)
sens['buf'] = dict(net=float(AC.run_q2(q=Q)['total']))
for tag, mode, W in [('norm', 'normal', 30), ('split', 'split', 30), ('w60', 'net', 60), ('w90', 'net', 90)]:
    B = np.zeros((n, H))
    for i in range(n):
        l2 = max(0, i - W)
        if i - l2 < 7:
            continue
        if mode == 'net':
            B[i] = np.quantile(e_[l2:i], Q, axis=0)
        elif mode == 'split':
            B[i] = (np.quantile((L_ACT - L)[l2:i], Q, axis=0)
                    + np.quantile(-(PV_ACT - PV)[l2:i], Q, axis=0))
        else:
            from scipy import stats
            z = stats.norm.ppf(Q); hh = np.arange(H) // 6
            for k in range(24):
                B[i][hh == k] = z * e_[l2:i][:, hh == k].ravel().std()
    sens['buf'][tag] = float(run_with_B(B))
out['sens'] = sens
print('  q 全年:', {k: round(v) for k, v in sens.items() if k.startswith('q')})
print(f'  往返0.9 {sens["eta_roundtrip"]:,.0f} | 2月 基准 {sens["feb_base"]:,.0f} / v0 {sens["feb_v0"]:,.0f} / SOC等式 {sens["feb_socEq"]:,.0f}')
print(f'  预测电价 {sens["price_pred"]:,.0f} | 计价A/B {sens["rule_A"]:,.0f} / {sens["rule_B"]:,.0f}')
print('  缓冲形式:', {k: round(v) for k, v in sens['buf'].items()})
print('=== 预言机分解 ===')
out['oracle_pv'] = float(AC.run_q2(L, PV_ACT, q=Q)['total'])
out['oracle_load'] = float(AC.run_q2(L_ACT, PV, q=Q)['total'])
print(f'  O_PV {out["oracle_pv"]:,.0f} | O_L {out["oracle_load"]:,.0f}')
print('=== 问题四 ===')
out['q4_nostorage'] = float(AC.nostorage(q=Q, pmat=PR))
out['q4_bound'] = float(AC.clairvoyant(pmat=PR))
out['q4_nostorage_predprice'] = float(AC.nostorage(q=Q, pmat=P_HAT))
R42 = AC.run_q2(q=Q, pmat=P_HAT, bill=PR)   # 问题四之二 采用(预测电价定计划、实际电价结算)
R42p = AC.run_q2(q=Q, pmat=PR, bill=PR)     # 上界(0:00 已知实际电价, 不可达)
R42y = AC.run_q2(q=Q, pmat=P_RT, bill=PR)   # 乙 电价模型对照
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
dw = D.dow[m2:]
for tag, R in [('weekday', R42), ('weekday_predprice', R42p), ('weekday_q2', AC.run_q2(q=Q))]:
    net = (R['C'] - R['D'])[m2:]
    out[tag] = [float(net[dw == k].sum()) / max(1, int((dw == k).sum())) for k in range(7)]
print(f'  Q4 无储能 {out["q4_nostorage"]:,.0f} | Q4 下界 {out["q4_bound"]:,.0f}')
print('  Q4-2 周内净充电:', ' '.join(f'{names[k]}{out["weekday"][k]:+.0f}' for k in range(7)))
json.dump(out, open(os.path.join(DATA, 'A_all_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1, default=float)
print(f'[完成] {time.time()-t0:.0f}s -> data/A_all_numbers.json')
