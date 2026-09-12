# -*- coding: utf-8 -*-
"""问题三/四 因果口径下的对照数字(统一实时平衡结算)
   输出: 04_建模求解/data/causal_variants.json
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, PR, run_q2, run_q3, clairvoyant, nostorage, ROOT

res = {}
t0 = time.time()

print('=== 问题二/三: 结算方式对照(固定电价) ===')
R_as = run_q2(mode='asplanned', q=0.7, reserve=1200.0)
print(f'  僵硬执行计划(无日内平衡): 总 {R_as["total"]:,.0f} 元')
R_ca = run_q2(mode='causal', q=0.7, reserve=1200.0)
print(f'  实时平衡控制(采用):       总 {R_ca["total"]:,.0f} 元')
res['q2_asplanned'] = R_as['total']; res['q2_causal'] = R_ca['total']
res['q2_causal_plan'] = R_ca['plan']; res['q2_causal_em'] = R_ca['em']
res['q2_em_kwh'] = float(R_ca['Em'][D.m2:].sum())

print('\n=== 问题三: 滚动调整价值(固定电价, 因果) ===')
q3v = {}
for at, name in [((), '仅用0:00预报'), ((6,), '+6:00'), ((12,), '+12:00'), ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动(采用)')]:
    R = run_q3(adjust=at)
    q3v[name] = dict(plan=R['plan'], dev=R['dev'], em=R['em'], tot=R['total'])
    print(f'  {name:12s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
res['q3'] = q3v
print(f'  [滚动价值] 全滚动相对仅0:00 = {q3v["仅用0:00预报"]["tot"]-q3v["全滚动(采用)"]["tot"]:,.0f} 元')

print('\n=== 问题四(附件4 实时电价): 因果口径 ===')
R42 = run_q2(p_matrix=PR, q=0.7, reserve=1200.0)
print(f'  Q4-2 同类日+报童q0.7(采用): 计划 {R42["plan"]:,.0f} + 紧急 {R42["em"]:,.0f} = {R42["total"]:,.0f} 元 | 紧急电量 {R42["Em"][D.m2:].sum():,.0f} kWh')
R42n = run_q2(p_matrix=PR, q=0.0, reserve=1200.0)
print(f'  Q4-2 同类日(无裕度):       计划 {R42n["plan"]:,.0f} + 紧急 {R42n["em"]:,.0f} = {R42n["total"]:,.0f} 元')
R42ns = nostorage(q=0.7, p_matrix=PR)
print(f'  Q4-2 无储能对照:           {R42ns:,.0f} 元')
CB4 = clairvoyant(p_matrix=PR)
print(f'  Q4 天眼链式下界:           {CB4:,.0f} 元')
res['q42'] = dict(plan=R42['plan'], em=R42['em'], tot=R42['total'], nomargin=R42n['total'],
                  nostorage=R42ns, bound=CB4, em_kwh=float(R42['Em'][D.m2:].sum()))
q43v = {}
for at, name in [((), '仅用0:00预报'), ((6, 12, 18), '全滚动(采用)')]:
    R = run_q3(p_matrix=PR, adjust=at)
    q43v[name] = dict(plan=R['plan'], dev=R['dev'], em=R['em'], tot=R['total'])
    print(f'  Q4-3 {name:12s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f} 元')
res['q43'] = q43v; res['q43']['bound'] = CB4
print(f'  [滚动价值] Q4-3 全滚动相对仅0:00 = {q43v["仅用0:00预报"]["tot"]-q43v["全滚动(采用)"]["tot"]:,.0f} 元')

CB2 = clairvoyant()
print(f'\n问题二/三 天眼链式下界(固定电价) = {CB2:,.0f} 元')
res['q23_bound'] = CB2
print(f'[总耗时] {time.time()-t0:.1f}s')
out = os.path.join(ROOT, r'04_建模求解\data\causal_variants.json')
json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved ->', out)
