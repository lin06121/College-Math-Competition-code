# -*- coding: utf-8 -*-
"""统一"因果口径"下的问题二方法对比表(全部方法使用同一实时平衡控制结算)
   口径: 0:00 只用历史信息定计划; 日内按实时平衡控制(裁剪/减充/增放/减放/增充)兜底;
        兜不住的缺口按 5 倍电价紧急购电; 弃光允许。
   方法: M1 昨日外推 | M2 附件1典型日 | M5 昨日外推+q0.8 | M7 昨日外推+q0.6
         M8 同类日平均+近4日光伏+q0.7(采用) | M4 无储能 | M3 天眼链式下界
   输出: 04_建模求解/data/causal_basis.json
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import solve_day, P_MAX, SOC_MIN, SOC_MAX, ETA_C, ETA_D
from io_results import load_all, ROOT

day, typ, fc0, fca, pers, tmpl, const = load_all()
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365]); n = len(dates365)
L_typ1 = typ['load_typ'].to_numpy() / 6.0
PV_typ1 = typ['pvfc_typ'].to_numpy() / 6.0
L_typ = np.broadcast_to(L_typ1, L_act.shape).copy()
PV_typ = np.broadcast_to(PV_typ1, PV_act.shape).copy()
L_per = pers.pivot(index='date', columns='slot', values='load_prev_kw').sort_index().to_numpy() / 6.0
PV_per = pers.pivot(index='date', columns='slot', values='pv_prev_kw').sort_index().to_numpy() / 6.0
m2 = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]
v_const = 0.9 * price.mean()

Lb_dow = np.zeros_like(L_act); Pb_m4 = np.zeros_like(PV_act)
for i in range(n):
    hist = [j for j in range(i) if (dow[j] in {4, 5}) == (dow[i] in {4, 5})]
    Lb_dow[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ1
    h2 = list(range(max(0, i - 4), i)); Pb_m4[i] = PV_act[h2].mean(axis=0) if h2 else PV_typ1
eNet = (L_act - Lb_dow) - (PV_act - Pb_m4)


def buf(i, q):
    lo = max(0, i - 30)
    return np.quantile(eNet[lo:i], q, axis=0) if i - lo >= 7 else np.zeros(144)


def balance(P0, Cpl, Dpl, soc0, L, PV):
    """实时平衡控制(因果): 只用当前/历史状态"""
    h = len(P0); C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        c_ok = min(max(float(Cpl[t]), 0.0), P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C))
        d_ok = min(max(float(Dpl[t]), 0.0), P_MAX, ETA_D * max(0.0, soc - SOC_MIN + ETA_C * c_ok))
        C[t], D[t] = c_ok, d_ok
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        if gap > 1e-9:                       # 缺额: 先减充, 再增放
            cut = min(C[t], gap); C[t] -= cut; gap -= cut
            if gap > 1e-9:
                avail = ETA_D * max(0.0, soc - SOC_MIN + ETA_C * C[t]) - D[t]
                extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, avail)); D[t] += extra; gap -= extra
        elif gap < -1e-9:                    # 富余: 先减放, 再增充, 剩余弃光
            cut = min(D[t], -gap); D[t] -= cut; gap += cut
            if gap < -1e-9:
                room = max(0.0, (SOC_MAX - soc + D[t] / ETA_D)) / ETA_C - C[t]
                extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, room)); C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
        assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6
        assert C[t] <= P_MAX + 1e-9 and D[t] <= P_MAX + 1e-9
    return C, D, soc, Em


def buffer_from(eMat, i, q):
    """该方法的报童裕度: 用该方法自身的预测残差经验分位(近 30 天)"""
    lo = max(0, i - 30)
    if i - lo < 7 or q <= 0:
        return np.zeros(144)
    return np.quantile(eMat[lo:i], q, axis=0)


def chain(GLB, GPV, q, eMat=None):
    """GLB/GPV: 逐日计划依据(L,PV); q: 报童分位; eMat: 该方法自身的残差矩阵"""
    eMat = eNet if eMat is None else eMat
    soc = 6000.0
    plan = em = eQ = 0.0
    for i in range(n):
        b = buffer_from(eMat, i, q)
        Lb = GLB[i] + b
        r0 = solve_day(price, Lb, GPV[i], soc, soc_end=None, v=(v_const if i < n - 1 else 0.0))
        C, D, sEnd, Em = balance(r0['P'], r0['C'], r0['D'], soc, L_act[i], PV_act[i])
        if i >= m2:
            plan += float(price @ r0['P']); em += float(5 * price @ Em); eQ += float(Em.sum())
        soc = sEnd
    return plan, em, eQ


def chain_typ():
    """附件1 典型日曲线的报童裕度: 残差相对典型曲线"""
    soc = 6000.0; plan = em = eQ = 0.0
    eT = (L_act - L_typ) - (PV_act - PV_typ)
    for i in range(n):
        b = buffer_from(eT, i, 0.7)
        r0 = solve_day(price, L_typ[i] + b, PV_typ[i], soc, soc_end=None, v=(v_const if i < n - 1 else 0.0))
        C, D, sEnd, Em = balance(r0['P'], r0['C'], r0['D'], soc, L_act[i], PV_act[i])
        if i >= m2:
            plan += float(price @ r0['P']); em += float(5 * price @ Em); eQ += float(Em.sum())
        soc = sEnd
    return plan, em, eQ


def nostorage(GLB, GPV, q, eMat=None):
    eMat = eNet if eMat is None else eMat
    tot = 0.0
    for i in range(n):
        b = buffer_from(eMat, i, q)
        P0 = np.maximum(GLB[i] + b - GPV[i], 0.0)
        Em = np.maximum(L_act[i] - PV_act[i] - P0, 0.0)
        if i >= m2:
            tot += float(price @ P0) + float(5 * price @ Em)
    return tot


def clairvoyant():
    """天眼链式下界: 用当日实测做计划(储能链一致), 不可达下界"""
    soc = 6000.0; tot = 0.0
    for i in range(n):
        r = solve_day(price, L_act[i], PV_act[i], soc, soc_end=None, v=(v_const if i < n - 1 else 0.0))
        if i >= m2:
            tot += float(price @ r['P'])
        soc = r['E'][-1]
    return tot


res = {}
print('===== 问题二：统一因果口径(0:00计划 + 日内实时平衡控制 + 5倍紧急) =====')
E_PER = (L_act - L_per) - (PV_act - PV_per)      # 外推法自身残差
t0 = time.time()
for key, name, GLB, GPV, q, eM in [
        ('M8', '同类日平均+近4日光伏+q0.7(采用)', Lb_dow, Pb_m4, 0.7, eNet),
        ('M7', '昨日外推+报童q0.6', L_per, PV_per, 0.6, E_PER),
        ('M5', '昨日外推+报童q0.8', L_per, PV_per, 0.8, E_PER),
        ('M1', '昨日外推(无裕度)', L_per, PV_per, 0.0, E_PER)]:
    plan, em, eQ = chain(GLB, GPV, q, eM)
    res[key] = dict(plan=plan, em=em, tot=plan + em, eq=eQ, label=name)
    print(f'{key} {name:34s}: 计划 {plan:>12,.0f} + 紧急 {em:>11,.0f} = {plan+em:>13,.0f} 元 | 紧急电量 {eQ:>10,.0f} kWh')
plan, em, eQ = chain_typ()
res['M2'] = dict(plan=plan, em=em, tot=plan + em, eq=eQ, label='附件1典型日曲线+报童q0.7')
print(f'M2 {"附件1典型日曲线+报童q0.7":34s}: 计划 {plan:>12,.0f} + 紧急 {em:>11,.0f} = {plan+em:>13,.0f} 元 | 紧急电量 {eQ:>10,.0f} kWh')
res['M4'] = dict(plan=None, em=None, tot=nostorage(Lb_dow, Pb_m4, 0.7), label='无储能(同类日+报童q0.7)')
print(f'M4 {"无储能(同类日+报童q0.7)":34s}: 总 {res["M4"]["tot"]:>13,.0f} 元')
res['M3'] = dict(tot=clairvoyant(), label='天眼链式下界(不可达)')
print(f'M3 {"天眼链式下界":36s}: 总 {res["M3"]["tot"]:>13,.0f} 元')
print(f'[耗时] {time.time()-t0:.1f}s')

out = os.path.join(ROOT, r'04_建模求解\data\causal_basis.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(dict(price_mean=float(price.mean()), v=float(v_const), rows=res),
          open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved ->', out)
