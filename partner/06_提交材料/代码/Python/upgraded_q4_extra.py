# -*- coding: utf-8 -*-
"""升级口径: 问题四的无储能对照与天眼下界(实时电价)"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from causal_core import ROOT, PR
import fc_upgraded as FU
import upgraded_core as UC

DATA = os.path.join(ROOT, '04_建模求解', 'data')
Q = json.load(open(os.path.join(DATA, 'upgraded_all_numbers.json'), encoding='utf-8'))['q_star']
L, _ = FU.build_load()
PV = FU.build_pv()
ns = UC.nostorage(L, PV, Q, pmat=PR)
cb = UC.clairvoyant(pmat=PR)
R = UC.run_q2(L, PV, q=Q, pmat=PR)
ns2 = UC.nostorage(L, PV, Q)
print(f'Q4-2 = {R["total"]:,.0f} | Q4 无储能 = {ns:,.0f} → 储能价值 {ns-R["total"]:,.0f} ({(ns-R["total"])/ns*100:.1f}%)')
print(f'Q4 天眼下界 = {cb:,.0f} → 距下界 {R["total"]-cb:,.0f}')
print(f'Q2 无储能 = {ns2:,.0f} → 储能价值 {ns2-13655835:,.0f} ({(ns2-13655835)/ns2*100:.1f}%)')
json.dump(dict(q4_nostorage=float(ns), q4_bound=float(cb), q2_nostorage=float(ns2),
               q42_total=float(R['total'])),
          open(os.path.join(DATA, 'upgraded_q4_extra.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
