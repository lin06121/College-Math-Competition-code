# -*- coding: utf-8 -*-
"""汇总论文所需全部数字 -> 05_论文与结果/论文/paper_numbers.json"""
import sys, io, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
from engine import solve_day
from io_results import load_all, ROOT

DATA = os.path.join(ROOT, r'04_建模求解\data')
PDIR = os.path.join(ROOT, r'05_论文与结果\论文')
os.makedirs(PDIR, exist_ok=True)

day, typ, fc0, fca, pers, tmpl, const = load_all()
day = day.copy(); day['dow'] = day['date'].dt.dayofweek
price = typ['price_typ'].to_numpy()
L_act = day.pivot(index='date', columns='slot', values='load_kw').sort_index().to_numpy() / 6.0
PV_act = day.pivot(index='date', columns='slot', values='pv_kw').sort_index().to_numpy() / 6.0
PR = day.pivot(index='date', columns='slot', values='price_q4').sort_index().to_numpy()
dates365 = pd.to_datetime(day.pivot(index='date', columns='slot', values='load_kw').sort_index().index)
dow = np.array([d.dayofweek for d in dates365])
n = len(dates365)
L_typ = typ['load_typ'].to_numpy() / 6.0
PV_typ = typ['pvfc_typ'].to_numpy() / 6.0
Lb = np.zeros_like(L_act)
for i in range(n):
    low_set = {4, 5}
    hist = [j for j in range(i) if (dow[j] in low_set) == (dow[i] in low_set)]
    Lb[i] = L_act[hist[-4:]].mean(axis=0) if hist else L_typ
Pb = np.zeros_like(PV_act)
for i in range(n):
    hist = list(range(max(0, i - 4), i))
    Pb[i] = PV_act[hist].mean(axis=0) if hist else PV_typ
eL = L_act - Lb; eP = PV_act - Pb
m = np.where(dates365 >= pd.Timestamp('2025-02-01'))[0][0]

# 无储能基准 (最终依据: 同类星期+dow; 报童q=0.7 on net residual)
buf = np.zeros((n, 144))
for i in range(n):
    lo = max(0, i - 30)
    if i - lo >= 7:
        buf[i] = np.quantile((eL - eP)[lo:i], 0.7, axis=0)
P0_ns = np.maximum(Lb + buf - Pb, 0)
E_ns = np.maximum(L_act - PV_act - P0_ns, 0)
cost_ns_fixed = float((price @ P0_ns.T + 5 * price @ E_ns.T)[m:].sum())
cost_ns_q4 = float(((PR * P0_ns).sum(axis=1) + (5 * PR * E_ns).sum(axis=1))[m:].sum())

# 周内规律
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
load_dow = (pd.Series(L_act.sum(axis=1), index=dates365).groupby(dow).mean() / 1000)
price_dow = pd.Series(PR.mean(axis=1), index=dates365).groupby(dow).mean()
low2 = [4, 5]
load_gap = (load_dow[~load_dow.index.isin(low2)].mean() - load_dow[low2].mean()) / load_dow[~load_dow.index.isin(low2)].mean()
price_gap = (price_dow[~price_dow.index.isin(low2)].mean() - price_dow[low2].mean()) / price_dow[~price_dow.index.isin(low2)].mean()
w1 = day[day.date.between('2025-01-01', '2025-01-07')].groupby(['date', 'dow'])['price_q4'].mean().reset_index()
w1_list = [(str(r.date.date()), names[int(r.dow)], round(float(r.price_q4), 4)) for r in w1.itertuples()]
d10 = day.groupby(['date', 'dow'])['price_q4'].mean().reset_index().nsmallest(10, 'price_q4')
d10_list = [(str(r.date.date()), names[int(r.dow)], round(float(r.price_q4), 4)) for r in d10.itertuples()]

# 光伏季节
pv_daily = PV_act.sum(axis=1)
pv_month = pd.Series(pv_daily, index=dates365).groupby(dates365.month).mean() / 1000

res = dict(
    q1=json.load(open(os.path.join(DATA, 'q1_results.json'), encoding='utf-8')),
    q2=json.load(open(os.path.join(DATA, 'q2_results.json'), encoding='utf-8')),
    q3=json.load(open(os.path.join(DATA, 'q3_results.json'), encoding='utf-8')),
    q4=json.load(open(os.path.join(DATA, 'q4_results.json'), encoding='utf-8')),
    refine=json.load(open(os.path.join(DATA, 'basis_refine.json'), encoding='utf-8')),
    pv_season=json.load(open(os.path.join(DATA, 'pv_seasonality.json'), encoding='utf-8')),
    noise=json.load(open(os.path.join(DATA, 'noise_decompose.json'), encoding='utf-8')),
    nostorage=dict(fixed=cost_ns_fixed, q4=cost_ns_q4),
    weekly=dict(load_MWh={names[k]: round(float(load_dow[k]), 1) for k in range(7)},
                price={names[k]: round(float(price_dow[k]), 4) for k in range(7)},
                load_gap_pct=round(float(load_gap) * 100, 1), price_gap_pct=round(float(price_gap) * 100, 1),
                jan_week1=w1_list, cheapest10=d10_list),
    pv_month_MWh={int(k): round(float(v), 1) for k, v in pv_month.items()},
    constants=const,
)
with open(os.path.join(PDIR, 'paper_numbers.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

print('无储能基准(固定电价, 新依据) =', f'{cost_ns_fixed:,.0f}', '元')
print('无储能基准(波动电价)        =', f'{cost_ns_q4:,.0f}', '元')
print('周内: 低日负荷/价格 =', names[4], names[5], '| 负荷低%.1f%% 电价低%.1f%%' % ((1 - load_dow[low2].mean() / load_dow[~load_dow.index.isin(low2)].mean()) * 100, (1 - price_dow[low2].mean() / price_dow[~price_dow.index.isin(low2)].mean()) * 100))
print('1月第一周电价:', w1_list)
print('全年最低10天:', d10_list)
print('月度光伏(MWh):', res['pv_month_MWh'])
print('paper_numbers.json written')
