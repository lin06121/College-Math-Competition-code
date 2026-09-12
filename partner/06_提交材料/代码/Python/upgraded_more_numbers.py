# -*- coding: utf-8 -*-
"""升级版口径的补充数字: 滚动方案对照 / 缓冲形式 / 问题四周内充放电"""
import sys, io, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from causal_core import D, price, PR, ROOT, balance
from engine import solve_day
import fc_upgraded as FU
import upgraded_core as UC

DATA = os.path.join(ROOT, '04_建模求解', 'data')
out = {}
t0 = time.time()
L_UP, INFO = FU.build_load()
PV_UP = FU.build_pv()
P_UP = FU.build_price()
Q = json.load(open(os.path.join(DATA, 'upgraded_all_numbers.json'), encoding='utf-8'))['q_star']

print('=== 问题三 滚动方案对照(固定电价) ===')
for at, nm in [((), '仅0:00'), ((6,), '+6:00'), ((12,), '+12:00'), ((6, 12), '+6,+12'), ((6, 12, 18), '全滚动')]:
    R = UC.run_q3(L_UP, PV_UP, q=Q, adjust=at)
    out[f'q3_{nm}'] = dict(plan=float(R['plan']), dev=float(R['dev']), em=float(R['em']), total=float(R['total']))
    print(f'  {nm:8s}: 计划 {R["plan"]:>12,.0f} + 调整 {R["dev"]:>9,.0f} + 紧急 {R["em"]:>10,.0f} = {R["total"]:>13,.0f}')

print('=== 缓冲形式对照(升级预测) ===')
e = (D.L_act - L_UP) - (D.PV_act - PV_UP)
m2 = D.m2


def buf(mode, q=Q, W=30, j=None):
    lo = max(0, j - W)
    if j - lo < 7:
        return np.zeros(144)
    if mode == 'net':
        return np.quantile(e[lo:j], q, axis=0)
    if mode == 'split':
        eL = D.L_act - L_UP; eP = D.PV_act - PV_UP
        return np.quantile(eL[lo:j], q, axis=0) + np.quantile(-eP[lo:j], q, axis=0)
    if mode == 'normal':
        from scipy import stats
        z = stats.norm.ppf(q); b = np.zeros(144); h = np.arange(144) // 6
        for k in range(24):
            b[h == k] = z * e[lo:j][:, h == k].ravel().std()
        return b
    raise ValueError(mode)


def run_with_buf(mode, q=Q, W=30):
    return UC.run_q2(L_UP, PV_UP, q=q, buffer_mat_in=np.array([buf(mode, q, W, j) for j in range(D.n)]))


base = UC.run_q2(L_UP, PV_UP, q=Q)['total']
out['buf_net30'] = float(base)
print(f'  经验分位(近30天, 采用): {base:,.0f}')
for tag, kw in [('正态参数化(按小时σ)', dict(mode='normal')), ('分噪声叠加', dict(mode='split')),
                ('经验分位 60 天窗', dict(mode='net', W=60)), ('经验分位 90 天窗', dict(mode='net', W=90))]:
    v = float(run_with_buf(**kw)['total'])
    out['buf_' + tag] = v
    print(f'  {tag:20s}: {v:,.0f}  (差 {v-base:+,.0f}, {(v-base)/base*100:+.2f}%)')

print('=== 问题四之二 周内净充电(升级预测) ===')
R42 = UC.run_q2(L_UP, PV_UP, q=Q, pmat=PR)
R2 = UC.run_q2(L_UP, PV_UP, q=Q)
names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
for tag, R in [('Q4-2(实时电价)', R42), ('Q2(基准电价)', R2)]:
    net = (R['C'] - R['D'])[m2:]
    dw = D.dow[m2:]
    row = [float(net[dw == k].sum()) / max(1, int((dw == k).sum())) for k in range(7)]
    out['weekday_' + tag] = row
    print(f'  {tag}: ' + '  '.join(f'{names[k]} {row[k]:+8.0f}' for k in range(7)))
json.dump(out, open(os.path.join(DATA, 'upgraded_more_numbers.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(f'[完成] {time.time()-t0:.0f}s')
