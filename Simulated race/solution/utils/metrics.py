# -*- coding: utf-8 -*-
"""
utils/metrics.py — 通用指标归一化
==================================
供问题 2 / 问题 3 复用的 min-max 归一化工具。

题面关键约定：各“最大化/最小化”指标的归一化是**相对的**——
必须在“当前候选方案池 / 当前种群全体”上取最大值与最小值做 min-max，
而非使用某个固定上下界。当 max == min 时该项归一化为 0。
"""

from __future__ import annotations

import numpy as np


def minmax_normalize(raw: np.ndarray,
                     lower_is_better: bool = False) -> np.ndarray:
    """对一组原始指标值做 min-max 归一化到 [0,1]。

    参数
    ----
    raw : 一维数组，全体候选方案的某项原始指标值。
    lower_is_better : True 表示该指标越小越优（minimize），
                      归一化后仍返回 (raw - min)/(max - min)；
                      最终符号由 Z2 权重前的正负号统一处理。

    返回
    ----
    与 raw 等长的归一化数组；若 max == min 则全 0。
    """
    raw = np.asarray(raw, dtype=float)
    rmin, rmax = raw.min(), raw.max()
    if rmax - rmin < 1e-12:
        return np.zeros_like(raw)
    return (raw - rmin) / (rmax - rmin)


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """数值钳制到 [lo, hi]。"""
    return np.clip(np.asarray(x, dtype=float), lo, hi)