# -*- coding: utf-8 -*-
"""汇总升级预测模型口径下论文所需的全部数字 -> 05_论文与结果/论文/paper_numbers.json"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import ROOT, D
import fc_upgraded as FU
import upgraded_core as UC

DATA = os.path.join(ROOT, '04_建模求解', 'data')
PAPER = os.path.join(ROOT, '05_论文与结果', '论文')
up = json.load(open(os.path.join(DATA, 'upgraded_all_numbers.json'), encoding='utf-8'))
more = json.load(open(os.path.join(DATA, 'upgraded_more_numbers.json'), encoding='utf-8'))
q4x = json.load(open(os.path.join(DATA, 'upgraded_q4_extra.json'), encoding='utf-8'))
cost = json.load(open(os.path.join(ROOT, r'07_预测模型实验\data\final_cost.json'), encoding='utf-8'))
exp = json.load(open(os.path.join(ROOT, r'07_预测模型实验\data\final_cost.json'), encoding='utf-8'))

out = dict(
    note='升级预测模型口径(正式采用): 见 04_建模求解/scripts/rebuild_upgraded.py',
    q1=json.load(open(os.path.join(DATA, 'q1_results.json'), encoding='utf-8'))
    if os.path.exists(os.path.join(DATA, 'q1_results.json')) else None,
    q2=up['q2'], q3=up['q3'], q42=up['q42'], q43=up['q43'],
    q_star=up['q_star'], q_tune_jan=up['q_tune'],
    fc=dict(load=up['mae_负荷'], pv=up['mae_光伏'], price=up['mae_电价'],
            load_class=up['load_class'], load_class_base=up['load_class_base'],
            load_weights=up['load_weights']),
    exec_modes=dict(asplanned=up['exec_asplanned'], causal=up['exec_causal'],
                    perfect=up['exec_perfect']),
    compare=dict(tianyan=up['cmp_天眼下界'], nostorage=up['cmp_无储能'],
                 typical=up['cmp_典型日曲线'], M7=up['cmp_M7_旧口径保留'],
                 M5=up['cmp_M5_旧口径保留'], M1=up['cmp_M1_旧口径保留'],
                 M2=up['cmp_M2_旧口径保留']),
    sensitivity=dict(eta_roundtrip=up['sens_eta_roundtrip'], feb_base=up['sens_feb_base'],
                     feb_v0=up['sens_feb_v0'], feb_socEq=up['sens_feb_socEq'],
                     q_year=up['sens_q_year'], settle_asplanned=up['sens_settle_asplanned'],
                     price_pred=up['sens_price_pred'], rule_A=up['sens_price_rule_A'],
                     buffer_forms={k.replace('buf_', ''): v for k, v in more.items() if k.startswith('buf_')}),
    oracle=dict(O_PV=up['oracle_pv'], O_L=up['oracle_load'], bound=up['cmp_天眼下界'],
                gap=up['q2']['total'] - up['cmp_天眼下界'],
                pv_value=up['q2']['total'] - up['oracle_pv'],
                load_value=up['q2']['total'] - up['oracle_load']),
    rolling={k[3:]: v for k, v in more.items() if k.startswith('q3_')},
    q4=dict(nostorage=q4x['q4_nostorage'], bound=q4x['q4_bound']),
    weekday=more.get('weekday_Q4-2(实时电价)'),
    experiment_cost=cost,
)
json.dump(out, open(os.path.join(PAPER, 'paper_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1, default=float)
print('written -> paper_numbers.json')
print(' q2 =', round(up['q2']['total']), ' q3 =', round(up['q3']['total']),
      ' q42 =', round(up['q42']['total']), ' q43 =', round(up['q43']['total']))
print(' 预测精度:', {k: round(v['mae'], 3) for k, v in
                    [('负荷', up['mae_负荷']), ('光伏', up['mae_光伏']), ('电价', up['mae_电价'])]})
