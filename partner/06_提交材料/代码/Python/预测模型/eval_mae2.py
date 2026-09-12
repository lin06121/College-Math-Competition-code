# -*- coding: utf-8 -*-
"""阶段三: 误差反馈/稳健统计/堆叠融合 三类新模型的精度对比(不做 LP)
   输出: data/mae_table2.csv 与 最优组合的预测矩阵 data/best_preds.npz
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from fc_common import D, L_ACT, PV_ACT, FC0, RTP, metrics, DATA
import models as M
import models_ml as ML
import models_extra as MX

rows = []
best = {}


def add(tag, target, A_hat, A_act, keep=False):
    mt = metrics(A_hat, A_act)
    rows.append(dict(group=target, model=tag, mae_kw=mt['mae_kw'], rmse_kw=mt['rmse_kw'],
                     daily_mape=mt['daily_mape']))
    print(f'  [{target}] {tag:36s} MAE={mt["mae_kw"]:7.2f}  RMSE={mt["rmse_kw"]:7.2f}  '
          f'日总量MAPE={mt["daily_mape"]:5.2f}%')
    if keep:
        cand = best.get(target)
        if cand is None or mt['mae_kw'] < cand[1]:
            best[target] = (tag, mt['mae_kw'], A_hat)


t0 = time.time()
print('=== 负荷: 误差反馈 / 稳健统计 / 递推水平 ===')
Lb = D.Lb
cands_L = {'mean4cls': Lb,
           'median5': MX.robust_median(L_ACT, 5),
           'wrec': MX.recency_weighted(L_ACT, [0.4, 0.3, 0.2, 0.1]),
           'ef2': MX.err_feedback(Lb, L_ACT, k=2, damp=0.5),
           'ef3': MX.err_feedback(Lb, L_ACT, k=3, damp=0.35),
           'lvl': MX.level_smooth(Lb, L_ACT, alpha=0.35),
           'lgbm': ML.ml_forecast('L', 'lgbm')}
for k, v in cands_L.items():
    if k != 'mean4cls':
        add(k, 'L', v, L_ACT)
add('mean4cls(基线)', 'L', Lb, L_ACT)
st = MX.causal_stack([cands_L['mean4cls'], cands_L['ef2'], cands_L['lvl'], cands_L['lgbm']], L_ACT)
add('堆叠(岭回归融合4模型)', 'L', st, L_ACT, keep=True)
st2 = MX.causal_stack([cands_L['mean4cls'], cands_L['median5'], cands_L['wrec'], cands_L['ef3']], L_ACT)
add('堆叠(稳健4模型)', 'L', st2, L_ACT, keep=True)

print('=== 光伏: 误差反馈 / 稳健统计 / 堆叠 ===')
Pb = D.Pb
cands_PV = {'mean4': Pb,
            'median5': MX.robust_median(PV_ACT, 5),
            'wrec': MX.recency_weighted(PV_ACT, [0.4, 0.3, 0.2, 0.1]),
            'ef2': MX.err_feedback(Pb, PV_ACT, k=2, damp=0.5, clip=(0.7, 1.3)),
            'lvl': MX.level_smooth(Pb, PV_ACT, alpha=0.35),
            'fc0': FC0,
            'lgbm': ML.ml_forecast('PV', 'lgbm')}
for k, v in cands_PV.items():
    if k != 'mean4':
        add(k, 'PV', v, PV_ACT)
add('mean4(基线)', 'PV', Pb, PV_ACT)
stpv = MX.causal_stack([cands_PV['mean4'], cands_PV['ef2'], cands_PV['lgbm'], cands_PV['fc0']], PV_ACT,
                       clip=(0, None))
add('堆叠(mean4+ef2+lgbm+fc0)', 'PV', stpv, PV_ACT, keep=True)
stpv2 = MX.causal_stack([cands_PV['mean4'], cands_PV['median5'], cands_PV['wrec'], cands_PV['lgbm']], PV_ACT,
                        clip=(0, None))
add('堆叠(稳健4模型)', 'PV', stpv2, PV_ACT, keep=True)
for w in (0.6, 0.7, 0.8):
    add(f'融合: {w:.1f}×mean4+{1-w:.1f}×lgbm', 'PV', ML.blend(Pb, cands_PV['lgbm'], w), PV_ACT, keep=True)

print('=== 电价: 廓线+误差反馈 / 净负荷模型 / 堆叠 ===')
prof = M.price_models()['类别×时段廓线+自适应水平']
cands_P = {'prof': prof,
           'ef': MX.err_feedback(prof, RTP, k=2, damp=0.5, clip=(0.6, 1.6)),
           'lgbm': ML.ml_forecast('P', 'lgbm')}
for k, v in cands_P.items():
    add(k, 'P', v, RTP)
stp = MX.causal_stack([cands_P['prof'], cands_P['ef'], cands_P['lgbm']], RTP, clip=(0.005, 3.0))
add('堆叠(廓线+ef+lgbm)', 'P', stp, RTP, keep=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA, 'mae_table2.csv'), index=False, encoding='utf-8-sig')
print(f'\n[耗时] {time.time()-t0:.1f}s')
print('各目标最优:')
for t, (tag, mae, _) in best.items():
    print(f'  {t}: {tag}  MAE={mae:.2f} kW')
np.savez_compressed(os.path.join(DATA, 'best_preds.npz'),
                    **{f'best|{t}|{tag}': v for t, (tag, _, v) in best.items()},
                    **{f'L|{k}': v for k, v in cands_L.items()},
                    **{f'PV|{k}': v for k, v in cands_PV.items()},
                    **{f'P|{k}': v for k, v in cands_P.items()},
                    **{'STACK|L|岭融合4': st, 'STACK|L|稳健4': st2,
                       'STACK|PV|mean4+ef2+lgbm+fc0': stpv, 'STACK|PV|稳健4': stpv2,
                       'STACK|P|廓线+ef+lgbm': stp})
print('saved ->', os.path.join(DATA, 'mae_table2.csv'))
