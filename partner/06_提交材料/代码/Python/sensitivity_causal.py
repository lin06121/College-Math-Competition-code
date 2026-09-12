# -*- coding: utf-8 -*-
"""统一因果口径下的灵敏度分析(问题二):
   ①2月: 残值系数 v=0、日末 SOC=日初、计划的充放电效率口径
   ②全年: 光伏基值窗口(3/4/5/7/10 日/加权/EMA)、报童分位、计划依据
   ③问题三: 偏差计价口径 A/B、结算方式(僵硬执行 vs 实时平衡)
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import (D, price, PR, run_q2, run_q3, buffer_of, balance, n, ROOT)
from engine import solve_day, ETA_C, ETA_D, P_MAX, SOC_MIN, SOC_MAX

m2 = D.m2; JAN = 31; FEB_END = 59          # 2 月 = 索引 31..58
res = {}
t0 = time.time()

# ---------------- ① 2 月: 残值 / SOC 等式 / 效率口径 ----------------
print('=== ① 2 月灵敏度(因果口径) ===')


def run_feb(v_mode='std', eta=False, upto=FEB_END):
    soc = 6000.0; tot = 0.0
    for i in range(upto):
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        if v_mode == 'zero':
            v = 0.0
        buf = buffer_of(i, 0.7)
        kw = dict(soc_end=(soc if v_mode == 'eq' else None), v=v)
        if eta:      # 计划阶段按"往返 0.9"(单程 sqrt(0.9)) 建模, 结算仍用单程 0.9
            import engine
            old = (engine.ETA_C, engine.ETA_D)
            engine.ETA_C = engine.ETA_D = 0.9 ** 0.5
            r = solve_day(price, D.Lb[i] + buf, D.Pb[i], soc, **kw)
            engine.ETA_C, engine.ETA_D = old
        else:
            r = solve_day(price, D.Lb[i] + buf, D.Pb[i], soc, **kw)
        C, Dd, sEnd, Em = balance(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
        if i >= m2:
            tot += float(price @ r['P']) + float(5 * price @ Em)
        if v_mode == 'eq':
            sEnd = soc
        soc = sEnd
    return tot


base_feb = run_feb('std')
res['feb_base'] = base_feb
print(f'  基准(2 月): {base_feb:,.0f} 元')
for tag, kw in [('残值系数 v=0', dict(v_mode='zero')), ('日末 SOC=日初', dict(v_mode='eq')),
                ('计划按往返 0.9 效率', dict(eta=True))]:
    x = run_feb(**kw)
    res[f'feb_{tag}'] = x
    print(f'  {tag:20s}: {x:,.0f} 元  ({(x-base_feb)/base_feb*100:+.2f}%)')

# ---------------- ② 全年: 光伏基值窗口 / 报童分位 / 计划依据 ----------------
print('\n=== ② 全年灵敏度(因果口径) ===')
y = np.arange(n)


def run_year(PB, q=0.7, LAB=None, upto=n):
    LAB = D.Lb if LAB is None else LAB
    eNet = (D.L_act - LAB) - (D.PV_act - PB)
    soc = 6000.0; tot = 0.0
    for i in range(upto):
        v = 0.9 * price.mean() if i < n - 1 else 0.0
        lo = max(0, i - 30)
        b = np.quantile(eNet[lo:i], q, axis=0) if (i - lo >= 7 and q > 0) else np.zeros(144)
        r = solve_day(price, LAB[i] + b, PB[i], soc, soc_end=None, v=v)
        C, Dd, sEnd, Em = balance(r['P'], r['C'], r['D'], soc, D.L_act[i], D.PV_act[i])
        if i >= m2:
            tot += float(price @ r['P']) + float(5 * price @ Em)
        soc = sEnd
    return tot


def pv_window(K, weighted=False, ema=False):
    out = np.zeros_like(D.PV_act)
    for i in range(n):
        h = list(range(max(0, i - K), i))
        if not h:
            out[i] = D.PV_typ; continue
        A = D.PV_act[h]
        if weighted:
            w = np.arange(1, len(h) + 1, dtype=float); out[i] = (A * w[:, None]).sum(0) / w.sum()
        elif ema:
            a = 2 / (K + 1); out[i] = A[-1] if len(h) == 1 else None
            s = A[0].copy()
            for k in range(1, len(h)):
                s = a * A[k] + (1 - a) * s
            out[i] = s
        else:
            out[i] = A.mean(0)
    return out


base = run_year(D.Pb, 0.7)
res['year_base'] = base
print(f'  基准(采用, 4 日均值+q0.7): {base:,.0f} 元')
for K in (3, 5, 7, 10):
    x = run_year(pv_window(K), 0.7); res[f'pv{K}'] = x
    print(f'  光伏基值 {K:2d} 日均值 : {x:,.0f} 元 ({(x-base)/base*100:+.2f}%)')
for tag, kw in [('加权(近期权重高)', dict(weighted=True)), ('EMA', dict(ema=True))]:
    x = run_year(pv_window(4, **kw), 0.7); res[f'pv_{tag}'] = x
    print(f'  光伏基值 {tag:14s}: {x:,.0f} 元 ({(x-base)/base*100:+.2f}%)')
for q in (0.5, 0.6, 0.8):
    x = run_year(D.Pb, q); res[f'q{q}'] = x
    print(f'  报童分位 q={q:.1f}          : {x:,.0f} 元 ({(x-base)/base*100:+.2f}%)')

# ---------------- ③ 问题三: 偏差计价口径 / 结算方式 ----------------
print('\n=== ③ 问题三灵敏度(因果口径) ===')
R3 = run_q3(adjust=(6, 12, 18))
S = np.maximum(R3['P0'] - R3['Pf'], 0.0)
half_S = float(sum(price @ S[i] for i in range(m2, n))) * 0.5
res['q3_B'] = R3['total']; res['q3_A'] = R3['total'] + half_S
print(f'  计价口径 B(差额, 采用): {R3["total"]:,.0f} 元')
print(f'  计价口径 A(计划全额+罚): {R3["total"]+half_S:,.0f} 元  (+{half_S/R3["total"]*100:.2f}%)')
json.dump(res, open(os.path.join(ROOT, r'04_建模求解\data\sensitivity_causal.json'), 'w',
                    encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'[耗时] {time.time()-t0:.1f}s')
