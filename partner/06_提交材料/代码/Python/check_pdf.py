# -*- coding: utf-8 -*-
"""检查论文 PDF: 页数、是否有未解析引用(??)、附录 MATLAB 代码是否渲染"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import fitz

d = fitz.open(r'05_论文与结果\论文\main.pdf')
txt = ''.join(p.get_text() for p in d)
print('pages =', d.page_count)
print('unresolved "??" count =', txt.count('??'))
for key in ['MATLAB 复现代码', 'mysimplex', 'settle_balance', 'build_basis', 'tune_newsboy_q',
            'solve_day', 'lp_solve', 'quantile_linear', '代码文件清单', '06_提交材料']:
    print(f'  {key}: {txt.count(key)}')
