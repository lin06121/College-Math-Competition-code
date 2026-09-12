# -*- coding: utf-8 -*-
"""同步 06_提交材料: 代码(纯 Python) + 文件(论文/结果/预测模型实验), 移除 MATLAB"""
import sys, io, os, shutil, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = r'D:\Study\国赛\国赛'
SUB = os.path.join(ROOT, '06_提交材料')


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def cptree(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ---------- 1) 代码: 纯 Python ----------
code = os.path.join(SUB, '代码', 'Python')
os.makedirs(code, exist_ok=True)
n = 0
for f in glob.glob(os.path.join(ROOT, '04_建模求解', 'scripts', '*.py')):
    if os.path.basename(f).startswith('_'):
        continue
    cp(f, os.path.join(code, os.path.basename(f))); n += 1
print('建模脚本', n)
cptree(os.path.join(ROOT, '03_数据工程', 'scripts'), os.path.join(code, '数据工程'))
cptree(os.path.join(ROOT, '07_预测模型实验', 'scripts'), os.path.join(code, '预测模型'))
EARLY = os.path.join(code, '早期口径存档')
os.makedirs(EARLY, exist_ok=True)
for f in ['q2.py', 'q3.py', 'q4.py', 'q2_forecast.py', 'q2_post.py', 'sensitiv' + 'ity.py',
          'basis_refine.py', 'causal_compare.py', 'causal_numbers.py', 'qtune_m4.py',
          'refine2.py', 'settle_three_way.py']:
    p = os.path.join(ROOT, '04_建模求解', 'scripts', f)
    if os.path.exists(p):
        cp(p, os.path.join(EARLY, f))

# ---------- 2) 文件: 结果文件 + 论文 + 预测模型实验 ----------
files = os.path.join(SUB, '文件')
os.makedirs(files, exist_ok=True)
for f in ['result1.xlsx', 'result2.xlsx', 'result3.xlsx', 'result4-2.xlsx', 'result4-3.xlsx']:
    cp(os.path.join(ROOT, '05_论文与结果', f), os.path.join(files, f))
print('结果文件 5 个')

PAPER = os.path.join(files, '论文')
cptree(os.path.join(ROOT, '05_论文与结果', '论文', 'sections'), os.path.join(PAPER, 'sections'))
cptree(os.path.join(ROOT, '05_论文与结果', '论文', 'figures'), os.path.join(PAPER, 'figures'))
# 字体目录(约 62 MB)不进入提交材料: 在线编译器用 08_Overleaf 单文件版即可, 本地编译见 05_论文与结果\论文\fonts
for f in ['main.tex', 'main.pdf', 'cumcm2026.sty', 'cumcmthesis.cls', '编译说明.md',
          'paper_numbers.json', 'paper_tables.json']:
    p = os.path.join(ROOT, '05_论文与结果', '论文', f)
    if os.path.exists(p):
        cp(p, os.path.join(PAPER, f))
# 旧的 code\ 副本(含 MATLAB)整体移除
if os.path.exists(os.path.join(PAPER, 'code')):
    shutil.rmtree(os.path.join(PAPER, 'code'))
print('论文工程已同步(含 11 张插图, 不含 fonts)')

EXP = os.path.join(files, '预测模型实验')
cptree(os.path.join(ROOT, '07_预测模型实验', 'fig'), os.path.join(EXP, 'fig'))
cptree(os.path.join(ROOT, '07_预测模型实验', 'data'), os.path.join(EXP, 'data'))
# 预测矩阵(npz, 约 20 MB 中间量)不进入提交材料, 保留精度/费用表
for f in glob.glob(os.path.join(EXP, 'data', '*.npz')):
    os.remove(f)
for f in ['预测模型实验报告.md']:
    p = os.path.join(ROOT, '07_预测模型实验', '报告', f)
    if os.path.exists(p):
        cp(p, os.path.join(EXP, f))
for f in ['mae_table.csv', 'mae_table2.csv', 'cost_final.csv', 'final_cost.json',
          'load_weights.json', 'q_sensitivity_new.json']:
    p = os.path.join(ROOT, '07_预测模型实验', 'data', f)
    if os.path.exists(p):
        cp(p, os.path.join(EXP, f))
print('预测模型实验已同步(不含 npz 中间量)')

# ---------- 3) 清理: 移除一切 MATLAB ----------
for d in glob.glob(os.path.join(SUB, '**', 'MATLAB'), recursive=True):
    shutil.rmtree(d); print('removed', d)
for f in glob.glob(os.path.join(SUB, '**', '*.m'), recursive=True):
    os.remove(f); print('removed', f)
print('完成 ->', SUB)
