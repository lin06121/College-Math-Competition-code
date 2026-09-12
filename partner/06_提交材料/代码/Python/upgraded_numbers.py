# -*- coding: utf-8 -*-
"""升级版预测模型口径下的全部论文数字(与结果文件同源)
   输出: 04_建模求解/data/upgraded_all_numbers.json
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, PR, ROOT
from engine import solve_day
import fc_upgraded as FU
import upgraded_core as UC

DATA = os.path.join(ROOT, '04_建模求解', 'data')
t0 = time.time()
out = {}

print('=== 预测构造 ===')
L_UP, INFO = FU.build_load()
PV_UP = FU.build_pv(verbose=True)
P_UP = FU.build_price(verbose=True)
np.savez_compressed(os.path.join(DATA, 'upgraded_preds.npz'), L=L_UP, PV=PV_UP, P=P_UP)
for nm, A, B, sc in [('负荷', L_UP, D.L_act, 6), ('光伏', PV_UP, D.PV_act, 6), ('电价', P_UP, D.PR, 1)]:
    m = UC.metrics(A, B, sc)
    m.update(baseline=UC.metrics(D.Lb if nm == '负荷' else (D.Pb if nm == '光伏' else P_UP), B, sc))
    out[f'mae_{nm}'] = m
    print(f'  {nm}: 新模型 MAE={m["mae"]:.3f} RMSE={m["rmse"]:.3f} P95={m["p95"]:.3f} MAPE={m["mape"]:.2f}%')
e = np.abs(L_UP[D.m2:] - D.L_act[D.m2:]).mean(axis=1) * 6
lo = UC.LOW[D.m2:]
out['load_class'] = dict(total=float(e.mean()), low=float(e[lo].mean()), high=float(e[~lo].mean()))
eb = np.abs(D.Lb[D.m2:] - D.L_act[D.m2:]).mean(axis=1) * 6
out['load_class_base'] = dict(total=float(eb.mean()), low=float(eb[lo].mean()), high=float(eb[~lo].mean()))
print(f'  负荷分类: 新 {out["load_class"]}  基线 {out["load_class_base"]}')
out['load_weights'] = INFO

print('=== 报童分位标定 ===')
jan = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    R = UC.run_q2(L_UP, PV_UP, q=q, upto=31)
    jan[str(q)] = float((R['pc'] + R['ec'])[7:31].sum())
Q = float(min(jan, key=lambda k: jan[k]))
out['q_tune'] = jan; out['q_star'] = Q
print(f'  {jan} -> q*={Q}')

print('=== 四问主结果(与结果文件一致) ===')
R2 = UC.run_q2(L_UP, PV_UP, q=Q)
R3 = UC.run_q3(L_UP, PV_UP, q=Q)
R42 = UC.run_q2(L_UP, PV_UP, q=Q, pmat=PR)
R43 = UC.run_q3(L_UP, PV_UP, q=Q, pmat=PR)
for k, R in [('q2', R2), ('q3', R3), ('q42', R42), ('q43', R43)]:
    out[k] = {kk: float(R[kk]) for kk in ('plan', 'em', 'total', 'em_kwh') if kk in R}
    if 'dev' in R:
        out[k]['dev'] = float(R['dev'])
    print(f'  {k}: {out[k]}')

print('=== 执行方式三种(问题二) ===')
for tag, kw in [('asplanned', dict(asplanned=True)), ('causal', dict()), ('perfect', dict(perfect=True))]:
    R = UC.run_q2(L_UP, PV_UP, q=Q, **kw)
    out[f'exec_{tag}'] = dict(plan=float(R['plan']), em=float(R['em']), total=float(R['total']))
    print(f'  {tag}: 计划 {R["plan"]:,.0f} + 紧急 {R["em"]:,.0f} = {R["total"]:,.0f}')

print('=== 对照方法(问题二) ===')
def cand(name, Lh, PVh, q=Q):
    R = UC.run_q2(Lh, PVh, q=q)
    out[f'cmp_{name}'] = float(R['total'])
    print(f'  {name}: {R["total"]:,.0f}')
    return R
cand('采用(升级预测)', L_UP, PV_UP)
cand('典型日曲线', np.broadcast_to(D.L_typ, (D.n, 144)), np.broadcast_to(D.PV_typ, (D.n, 144)))
out['cmp_无储能'] = UC.nostorage(L_UP, PV_UP, Q)
print(f'  无储能: {out["cmp_无储能"]:,.0f}')
out['cmp_天眼下界'] = UC.clairvoyant()
print(f'  天眼下界: {out["cmp_天眼下界"]:,.0f}')
# 昨日外推 / 典型日 等按各自口径(不随升级变化)保留原值
old = json.load(open(os.path.join(DATA, 'causal_basis.json'), encoding='utf-8'))['rows']
for k in ('M7', 'M5', 'M1', 'M2'):
    if k in old:
        out[f'cmp_{k}_旧口径保留'] = old[k]

print('=== 灵敏度 ===')
# (1) 效率口径: 往返 0.9 (单程 sqrt(0.9)) —— 计划与结算同步替换
import engine, causal_core
def with_eta(fn, ec_):
    old_ = (engine.ETA_C, engine.ETA_D, causal_core.ETA_C, causal_core.ETA_D, UC.balance.__globals__['ETA_C'], UC.balance.__globals__['ETA_D'])
    engine.ETA_C = engine.ETA_D = causal_core.ETA_C = causal_core.ETA_D = ec_
    UC.balance.__globals__['ETA_C'] = UC.balance.__globals__['ETA_D'] = ec_
    try:
        return fn()
    finally:
        engine.ETA_C, engine.ETA_D, causal_core.ETA_C, causal_core.ETA_D = old_[0], old_[1], old_[2], old_[3]
        UC.balance.__globals__['ETA_C'], UC.balance.__globals__['ETA_D'] = old_[4], old_[5]
out['sens_eta_roundtrip'] = float(with_eta(lambda: UC.run_q2(L_UP, PV_UP, q=Q)['total'], 0.9 ** 0.5))
print(f'  往返0.9: {out["sens_eta_roundtrip"]:,.0f}')
# (2) 2 月: 残值 v=0 / 日末 SOC=日初
def feb(mode):
    tot = 0.0; soc = 6000.0
    B = UC.buffer_mat(L_UP, PV_UP, Q)
    for i in range(59):
        v = 0.9 * price.mean() if i < D.n - 1 else 0.0
        if mode == 'zero':
            v = 0.0
        end = soc if mode == 'eq' else None
        r = solve_day(price, L_UP[i] + B[i], PV_UP[i], soc, soc_end=end, v=v)
        if mode == 'eq':
            from causal_core import balance as _bal
            c, d, _, e = _bal(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
            sEnd = soc
        else:
            from causal_core import balance as _bal
            c, d, sEnd, e = _bal(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
        if i >= D.m2:
            tot += float(price @ r['P']) + float(5 * price @ e)
        soc = sEnd
    return tot
out['sens_feb_base'] = feb('std'); out['sens_feb_v0'] = feb('zero'); out['sens_feb_socEq'] = feb('eq')
print(f'  2月: 基准 {out["sens_feb_base"]:,.0f} / v=0 {out["sens_feb_v0"]:,.0f} / SOC等式 {out["sens_feb_socEq"]:,.0f}')
# (3) q 网格(全年)
out['sens_q_year'] = {}
for q in (0.5, 0.6, 0.7, 0.8, 0.9):
    out['sens_q_year'][str(q)] = float(UC.run_q2(L_UP, PV_UP, q=q)['total'])
print(f'  q 全年: {out["sens_q_year"]}')
# (4) 结算方式
out['sens_settle_asplanned'] = out['exec_asplanned']['total']
out['sens_settle_causal'] = out['exec_causal']['total']
out['sens_settle_perfect'] = out['exec_perfect']['total']
# (5) 电价信息: 计划用预测电价, 结算按实际
out['sens_price_pred'] = float(UC.run_q2(L_UP, PV_UP, q=Q, pmat=P_UP)['total'])
print(f'  计划用预测电价(按实际结算): {out["sens_price_pred"]:,.0f}')
# (6) 偏差计价口径 A/B(问题三)
S = (R3['P0'] - R3['Pf']).clip(min=0)
half_S = float(sum(price @ S[i] for i in range(D.m2, D.n))) * 0.5
out['sens_price_rule_A'] = float(R3['total'] + half_S)
print(f'  计价口径 A: {out["sens_price_rule_A"]:,.0f} (B={R3["total"]:,.0f})')
# (7) 保护下限(1月) —— 通过加缓冲实现近似? 直接跳过(论文保留原表并加注)
out['sens_buffer_forms_note'] = '见 upgraded_buffer_forms 脚本'

print('=== 天眼分解(升级预测) ===')
eL = D.L_act - L_UP
eP = D.PV_act - PV_UP
# O_PV: 计划阶段光伏用实测, 负荷用预测
out['oracle_pv'] = float(UC.run_q2(L_UP, D.PV_act, q=Q)['total'])
out['oracle_load'] = float(UC.run_q2(D.L_act, PV_UP, q=Q)['total'])
print(f'  O_PV={out["oracle_pv"]:,.0f}  O_L={out["oracle_load"]:,.0f}  下界={out["cmp_天眼下界"]:,.0f}')

json.dump(out, open(os.path.join(DATA, 'upgraded_all_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'\n[完成] {time.time()-t0:.0f}s -> data/upgraded_all_numbers.json')
