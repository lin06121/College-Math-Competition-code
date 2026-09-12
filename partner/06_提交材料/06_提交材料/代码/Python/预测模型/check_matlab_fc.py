# -*- coding: utf-8 -*-
"""解析 MATLAB 扩展预测模型运行日志(out_fc.txt), 与 Python 侧结果对照"""
import sys, io, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from fc_common import DATA, ROOT

LOGS = [os.path.join(ROOT, '04_建模求解', 'matlab', f)
        for f in ('out_fc.txt', 'out_fc42.txt', 'out_fc_full.txt')]
txt = ''
for f in LOGS:
    if os.path.exists(f):
        txt += open(f, 'rb').read().decode('utf-8', errors='replace') + '\n'
        txt += open(f, 'rb').read().decode('gbk', errors='replace') + '\n'
num = lambda s: float(s.replace(',', ''))

# ASCII 汇总行(RESULT ...)优先; 若缺失则回退到中文标签(可能因控制台编码失真)
def grab(key):
    m = re.search(rf'{key}=([\d.,]+)', txt)
    return float(m.group(1).replace(',', '')) if m else None


res = {}
if grab('q2_total') is not None:
    res['q2'] = dict(plan=grab('q2_plan'), em=grab('q2_em'), total=grab('q2_total'))
    res['q3'] = dict(total=grab('q3_total'), q3_zero=grab('q3_zero'))
    res['q42'] = dict(total=grab('q42_total'))
    res['q42_pred'] = dict(total=grab('q42pred_total'))
    res['q43'] = dict(total=grab('q43_total'))
    res['q_star'] = grab('q_star')
if grab('q2_total') is None:
    m = re.search(r'= (\d+), [^=]+= (\d+), [^=]+= (\d+) 元  \(Python', txt)
    if m:
        res['q2'] = dict(plan=num(m.group(1)), em=num(m.group(2)), total=num(m.group(3)))
    m = re.search(r'全滚动 = (\d+) 元, 仅0:00 = (\d+)', txt)
    if m:
        res['q3'] = dict(total=num(m.group(1)), q3_zero=num(m.group(2)))
    m = re.search(r'问题四之二\(电价已知\): 总费 = (\d+)', txt)
    if m:
        res['q42'] = dict(total=num(m.group(1)))
    m = re.search(r'问题四之二\(计划用预测电价, 结算按实际\): 总费 = (\d+)', txt)
    if m:
        res['q42_pred'] = dict(total=num(m.group(1)))
    m = re.search(r'问题四之三: 全滚动 = (\d+)', txt)
    if m:
        res['q43'] = dict(total=num(m.group(1)))
    m = re.search(r'q\*=(0\.\d)', txt)
    if m:
        res['q_star'] = float(m.group(1))

py = json.load(open(os.path.join(DATA, 'final_cost.json'), encoding='utf-8'))
rows = [('问题二(固定电价)', res.get('q2', {}).get('total'), py['q2_new']),
        ('问题三(滚动调整)', res.get('q3', {}).get('total'), py['q3_new']),
        ('问题四之二(电价已知)', res.get('q42', {}).get('total'), py['q42_new']),
        ('问题四之二(预测电价)', res.get('q42_pred', {}).get('total'), py['q42_predprice']),
        ('问题四之三(滚动)', res.get('q43', {}).get('total'), py['q43_new'])]
print(f'{"场景":22s} {"MATLAB":>14s} {"Python":>14s} {"差异":>10s} {"相对":>8s}')
ok = True
for nm, a, b in rows:
    if a is None:
        print(f'{nm:22s} {"(无)":>14s} {b:>14,.0f}'); ok = False; continue
    print(f'{nm:22s} {a:>14,.0f} {b:>14,.0f} {a-b:>+10,.0f} {(a-b)/b*100:>+7.3f}%')
print(f'\nMATLAB 选定的报童分位 q* = {res.get("q_star")} (Python 1 月标定同为 0.7)')
json.dump(dict(matlab=res, python=py), open(os.path.join(DATA, 'matlab_fc_compare.json'), 'w',
                                             encoding='utf-8'), ensure_ascii=False, indent=1)
print('saved -> data/matlab_fc_compare.json')
