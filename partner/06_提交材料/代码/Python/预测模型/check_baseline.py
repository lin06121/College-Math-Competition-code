# -*- coding: utf-8 -*-
"""基线校验(注入式管道): 必须与论文/结果文件逐元一致
   Q2 13,848,183 | Q4-2 14,531,823 | Q3 14,047,116 | Q4-3 14,721,611
   说明: 问题三的 0:00 计划用附件3 的 0:00 预报(FC0), 6/12/18 用附件3 更新预报(FCADJ),
        报童缓冲沿用论文口径(基于 近4日均值 的残差)
"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fc_common import D, FC0, RTP, run_q2_fc, run_q3_fc, buffer_from_resid

ok = True
R2 = run_q2_fc(D.Lb, D.Pb, q=0.7)
print(f'Q2  基线注入: {R2["total"]:,.0f} 元 (期望 13,848,183)')
ok &= abs(R2['total'] - 13848183) < 2
R42 = run_q2_fc(D.Lb, D.Pb, q=0.7, price_plan=RTP)
print(f'Q4-2 基线注入: {R42["total"]:,.0f} 元 (期望 14,531,823)')
ok &= abs(R42['total'] - 14531823) < 2
BUF_BASE = buffer_from_resid(D.Lb, D.Pb, 0.7)
R3 = run_q3_fc(D.Lb, FC0, q=0.7, buf_mat=BUF_BASE)
print(f'Q3  基线注入: {R3["total"]:,.0f} 元 (期望 14,047,116)')
ok &= abs(R3['total'] - 14047116) < 2
R43 = run_q3_fc(D.Lb, FC0, q=0.7, price_plan=RTP, buf_mat=BUF_BASE)
print(f'Q4-3 基线注入: {R43["total"]:,.0f} 元 (期望 14,721,611)')
ok &= abs(R43['total'] - 14721611) < 2
print('\n校验结论:', '通过(4/4 与论文口径逐元一致)' if ok else '未通过')
