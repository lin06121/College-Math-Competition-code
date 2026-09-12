# -*- coding: utf-8 -*-
"""阶段一: 只比精度(不做 LP) —— 各预测模型的 MAE/RMSE/日总量 MAPE
   输出: 07_预测模型实验/data/mae_table.csv 与 fig/fig_fc_mae.png
"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fc_common import (D, L_ACT, PV_ACT, RTP, PRICE, metrics, DATA, FIG, m2, n, DATES, IS_LOW)
import models as M
import models_ml as ML

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

t0 = time.time()
rows = []
pred = {}


def add(tag, target, A_hat, A_act):
    mt = metrics(A_hat, A_act)
    rows.append(dict(group=target, model=tag, mae_kw=mt['mae_kw'], rmse_kw=mt['rmse_kw'],
                     daily_mape=mt['daily_mape']))
    pred[(target, tag)] = A_hat
    print(f'  [{target}] {tag:34s} MAE={mt["mae_kw"]:7.2f} kW  RMSE={mt["rmse_kw"]:7.2f} kW  '
          f'日总量MAPE={mt["daily_mape"]:5.2f}%')


print('=== 负荷预测 ===')
Lm = M.load_models()
for k, v in Lm.items():
    add(k, 'L', v, L_ACT)
lr = ML.ml_forecast('L', 'ridge')
add('岭回归(逐时段特征)', 'L', lr, L_ACT)
lg = ML.ml_forecast('L', 'lgbm')
add('LightGBM(逐时段特征)', 'L', lg, L_ACT)
lh = ML.ml_forecast('L', 'lgbm', base=D.Lb)
add('基线+LightGBM残差修正', 'L', lh, L_ACT)
lh2 = ML.ml_forecast('L', 'lgbm', base=np.zeros_like(L_ACT))
print(f'  [检查] 纯 LightGBM(零基线) MAE={metrics(lh2, L_ACT)["mae_kw"]:.2f} kW')
for w in (0.3, 0.5, 0.7):
    add(f'融合: {w:.1f}×基线+{1-w:.1f}×GBDT残差', 'L', ML.blend(D.Lb, lh, w), L_ACT)

print('=== 光伏预测 ===')
for k, v in M.pv_models().items():
    add(k, 'PV', v, PV_ACT)
pvh = ML.ml_forecast('PV', 'lgbm', base=D.Pb)
add('基线+LightGBM残差修正', 'PV', pvh, PV_ACT)
pvl = ML.ml_forecast('PV', 'lgbm')
add('LightGBM(逐时段特征)', 'PV', pvl, PV_ACT)
add('融合: 0.5×晴空指数+0.5×基线残差GBDT', 'PV', 0.5 * pred[('PV', '晴空指数模型(近10日k×包络)')] + 0.5 * pvh, PV_ACT)
add('融合: 0.5×附件3+0.5×(基线残差GBDT)', 'PV', 0.5 * D.fc0m + 0.5 * pvh, PV_ACT)

print('=== 电价预测 ===')
for k, v in M.price_models().items():
    add(k, 'P', v, RTP)
pr = ML.ml_forecast('P', 'ridge')
add('岭回归(逐时段特征)', 'P', pr, RTP)
pl = ML.ml_forecast('P', 'lgbm')
add('LightGBM(逐时段特征)', 'P', pl, RTP)
plh = ML.ml_forecast('P', 'lgbm', base=M.price_models()['类别×时段廓线+自适应水平'])
add('廓线+LightGBM残差修正', 'P', plh, RTP)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA, 'mae_table.csv'), index=False, encoding='utf-8-sig')
print(f'\n[耗时] {time.time()-t0:.1f}s  -> {os.path.join(DATA, "mae_table.csv")}')

# 保存预测矩阵供阶段二使用
np.savez_compressed(os.path.join(DATA, 'preds.npz'),
                    **{f'{t}|{k}': v for (t, k), v in pred.items()})

# 图: 三类预测的精度对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, g, title in zip(axes, ['L', 'PV', 'P'], ['负荷 (kW)', '光伏 (kW)', '实时电价 (元/kWh)']):
    d = df[df.group == g].sort_values('mae_kw')
    ax.barh(range(len(d)), d['mae_kw'], color='#0072B2')
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d['model'], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel('MAE ' + title); ax.set_title(f'{g} 预测精度对比')
    ax.grid(alpha=0.3, axis='x')
    for i, v in enumerate(d['mae_kw']):
        ax.text(v, i, f' {v:.1f}', va='center', fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_fc_mae.png')); plt.close()
print('fig ->', os.path.join(FIG, 'fig_fc_mae.png'))
