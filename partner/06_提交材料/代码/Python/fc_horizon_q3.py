# -*- coding: utf-8 -*-
"""问题三: 各调整时刻"剩余时段"预报精度对照(解释为什么只有 12:00 调整有正收益)
   对比: 我们 0:00 依据(升级预测) vs 该时刻发布的附件3 预报, 指标 = 剩余时段 MAE(kW)
"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import A_core as AC
from causal_core import D

L, PV = AC.L, AC.PV
L_ACT, PV_ACT = AC.L_ACT, AC.PV_ACT
m2 = D.m2
out = {}
for h0, hh in [(36, 6), (72, 12), (108, 18)]:
    act_net = (L_ACT[m2:, h0:] - PV_ACT[m2:, h0:])
    our_net = (L[m2:, h0:] - PV[m2:, h0:])
    fc_net = (L[m2:, h0:] - D.fca_m[hh][m2:, h0:])
    d = dict(
        hours=f'{h0*10//60}:00',
        our_pv=float(np.abs(AC.PV[m2:, h0:] - PV_ACT[m2:, h0:]).mean() * 6),
        fc_pv=float(np.abs(D.fca_m[hh][m2:, h0:] - PV_ACT[m2:, h0:]).mean() * 6),
        our_net=float(np.abs(our_net - act_net).mean() * 6),
        fc_net_on_pv=float(np.abs(fc_net - act_net).mean() * 6))
    out[f'{hh}:00'] = d
    print(f'{hh:>2}:00 起剩余时段: 我们0:00依据 光伏MAE {d["our_pv"]:6.1f} kW / 净需求 {d["our_net"]:6.1f} kW | '
          f'该时刻附件3预报 光伏MAE {d["fc_pv"]:6.1f} kW / 净需求 {d["fc_net_on_pv"]:6.1f} kW')
json.dump(out, open(os.path.join(D.DATA if hasattr(D, 'DATA') else
                                 os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'data', 'fc_horizon_q3.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1, default=float)
print('-> data/fc_horizon_q3.json')
