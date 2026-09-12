# -*- coding: utf-8 -*-
"""fc_shim: 把 causal_core 的数据对象适配成预测模型库所需的命名(与 07_预测模型实验 一致)
   仅用于让 fc_features.py / fc_ml.py 与实验脚本保持逐位一致。
"""
import os
import numpy as np
from causal_core import D, price as PRICE, PR as RTP, ROOT

n = D.n
H = 144
L_ACT = D.L_act
PV_ACT = D.PV_act
FC0 = D.fc0m
FCADJ = D.fca_m
L_TYP = D.L_typ
PV_TYP = D.PV_typ
DOW = D.dow
IS_LOW = np.array([d in {4, 5} for d in DOW])
DATES = D.dates
DATA = os.path.join(ROOT, '04_建模求解', 'data')
FIG = os.path.join(ROOT, '04_建模求解', 'fig')


def fourier(x, K, period=1.0):
    cols = []
    for k in range(1, K + 1):
        cols.append(np.sin(2 * np.pi * k * x / period))
        cols.append(np.cos(2 * np.pi * k * x / period))
    return np.column_stack(cols)
