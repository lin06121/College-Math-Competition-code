# -*- coding: utf-8 -*-
"""结果文件终检(口径 A / 升级预测模型): 只读提交的 5 个 result*.xlsx 与原始附件, 复算论文数字
   检查项:
     ①计划购电量×电价 = 购电费(逐日)      ②全天购电量 = 逐时段之和
     ③三段计价恒等式(问题三/四之三): sum[p*min(P0,P)+0.5p*S+1.5p*X] = 调整后购电费
     ④SOC 链条(6 段充放电 -> 0/24 时储电量)  ⑤SOC ∈ [1200,10800]  ⑥每 4h 充放电 ≤ 20000 kWh
     ⑦紧急购电量合计 = 论文字段
"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from causal_core import D, price, ROOT

OUT = os.path.join(ROOT, '05_论文与结果')
ETA_C = ETA_D = 0.9
# 期望值(提交包口径: 问题二/四之二 = 提供版实现(纯历史预测 + 报童 q*=0.8);
#         问题三/四之三 = 原实现 q3*=0.6 + 全时刻滚动逐段结算)
EXP = {
    'result2.xlsx':   dict(plan=13_193_679, adj=None, em=85_605, emk=0),
    'result3.xlsx':   dict(plan=12_730_768, adj=13_017_836, em=62_285, emk=0),
    'result4-2.xlsx':   dict(plan=13_908_066, adj=None, em=86_878, emk=0),
    'result4-3.xlsx':   dict(plan=13_436_567, adj=13_738_263, em=60_555, emk=0),
}


def read_plan(fn, sheet='计划购电量'):
    d = pd.read_excel(os.path.join(OUT, fn), sheet_name=sheet, header=0)
    d.columns = ['date'] + [f'c{i}' for i in range(1, 145)] + ['tot', 'cost']
    d['date'] = pd.to_datetime(d['date']).ffill()
    return d


def read_cd(fn, sheet='充放电量'):
    d = pd.read_excel(os.path.join(OUT, fn), sheet_name=sheet, header=0)
    d.columns = ['date', 'block', 'chg', 'dis', 't', 'soc']
    d['date'] = pd.to_datetime(d['date']).ffill()
    return d


print('=== 问题一 result1.xlsx ===')
r1 = pd.read_excel(os.path.join(OUT, 'result1.xlsx'), sheet_name='计划购电量', header=0)
r1.columns = ['span', 'kwh']
p1 = D.typ['price_typ'].to_numpy() if hasattr(D, 'typ') else None
if p1 is None:
    p1 = np.broadcast_to(price, (1, 144))[0]
q1_kwh = float(r1['kwh'].sum()); q1_cost = float(p1 @ r1['kwh'].to_numpy())
print(f'  全天购电量 {q1_kwh:,.2f} kWh (期望 59,482.67) | 购电费 {q1_cost:,.2f} 元 (期望 35,126.95)')
ok1 = abs(q1_kwh - 59482.67) < 0.05 and abs(q1_cost - 35126.95) < 0.05
print('  ', 'OK' if ok1 else '!! 不一致')

ok = ok1
for fn, exp in EXP.items():
    pm = D.PR if fn.startswith('result4') else np.broadcast_to(price, D.PR.shape)
    blk = read_cd(fn); pl = read_plan(fn)
    dates = pl['date'].tolist()
    tot_plan = 0.0; bad_soc = bad_pow = bad_chain = 0
    for k, dt in enumerate(dates):
        i = int(np.where(D.dates == dt)[0][0])
        p = pm[i]
        P = np.array([float(pl.iloc[k][f'c{j}']) for j in range(1, 145)])
        tot_plan += float(p @ P)
        assert abs(float(pl.iloc[k]['tot']) - P.sum()) < 0.2, ('全天购电量不符', dt)
        assert abs(float(pl.iloc[k]['cost']) - float(p @ P)) < 0.6, ('购电费不符', dt)
        b = blk[blk['date'] == dt]
        assert len(b) == 6, (dt, len(b))
        soc = float(b['soc'].iloc[0]); s24 = float(b['soc'].iloc[1])
        for _, r in b.iterrows():
            if not (1200 - 0.1 <= soc <= 10800 + 0.1):
                bad_soc += 1
            if float(r['chg']) > 20000 + 0.1 or float(r['dis']) > 20000 + 0.1:
                bad_pow += 1
            soc = soc + ETA_C * float(r['chg']) - float(r['dis']) / ETA_D
        if abs(soc - s24) > 0.05:
            bad_chain += 1
    em = pd.read_excel(os.path.join(OUT, fn), sheet_name='紧急购电量', header=0)
    em.columns = ['date', 'span', 'kwh']
    em_kwh = float(em['kwh'].fillna(0).sum())
    tag = 'OK ' if abs(tot_plan - exp['plan']) < 2 else '!! '
    if abs(tot_plan - exp['plan']) >= 2:
        ok = False
    print(f'[{tag}{fn}] 计划购电费 = {tot_plan:,.0f} 元 (期望 {exp["plan"]:,}) | 紧急电量 = {em_kwh:,.0f} kWh '
          f'(期望 {exp["em"]:,}) | SOC越界 {bad_soc} | 功率越界 {bad_pow} | 链条不符 {bad_chain}')
    if exp['adj'] is not None:
        ad = read_plan(fn, '调整购电量')
        tot_adj = 0.0; bad_id = 0
        for k, dt in enumerate(ad['date'].tolist()):
            i = int(np.where(D.dates == dt)[0][0]); p = pm[i]
            P0 = np.array([float(pl.iloc[k][f'c{j}']) for j in range(1, 145)])
            P = np.array([float(ad.iloc[k][f'c{j}']) for j in range(1, 145)])
            S = np.maximum(P0 - P, 0.0); X = np.maximum(P - P0, 0.0)
            c = float(p @ (p * 0 + np.minimum(P0, P)) + 0.5 * p @ S + 1.5 * p @ X)
            tot_adj += c
            if abs(float(ad.iloc[k]['cost']) - c) > 0.6:
                bad_id += 1
        tag2 = 'OK ' if abs(tot_adj - exp['adj']) < 2 and bad_id == 0 else '!! '
        if abs(tot_adj - exp['adj']) >= 2 or bad_id:
            ok = False
        print(f'[{tag2}{fn} 三段计价] 计划$+$调整 $=$ {tot_adj:,.0f} 元 (期望 {exp["adj"]:,}) '
              f'| 逐日计价恒等式不符 {bad_id} 天')
print('\n结论:', '全部一致' if ok else '存在不一致')
