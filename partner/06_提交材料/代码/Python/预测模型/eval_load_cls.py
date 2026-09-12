# -*- coding: utf-8 -*-
"""负荷模型细化: 按"周五六 / 其他"分类评价, 并构造类别感知的误差反馈模型
   底线要求: 周五六与其他日两类都不能比基线更差
"""
import sys, io, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fc_common import D, L_ACT, IS_LOW, metrics, DATA, n, H
import models_extra as MX
import models_ml as ML

LOW = IS_LOW.copy()
m2 = D.m2


def mae_by_class(A_hat):
    e = np.abs(A_hat[m2:] - L_ACT[m2:]).mean(axis=1) * 6        # 每日 MAE(kW)
    lo = e[LOW[m2:]].mean(); hi = e[~LOW[m2:]].mean()
    return float(e.mean()), float(lo), float(hi)


def err_feedback_cls(base, A_act, k=3, damp=0.35, clip=(0.85, 1.15)):
    """只在同类别日内计算实际/预测比, 避免周内结构被跨类反馈污染"""
    out = np.zeros((n, H))
    for j in range(n):
        hist = [q for q in range(j) if LOW[q] == LOW[j]][-k:]
        if not hist:
            out[j] = base[j]; continue
        if len(hist) < k:                       # 同类历史不足时向 1 收缩
            d = damp * len(hist) / k
        else:
            d = damp
        r = np.array([A_act[q] / np.maximum(base[q], 1e-6) for q in hist]).mean(axis=0)
        r = np.clip(r, clip[0], clip[1])
        out[j] = base[j] * ((1 - d) + d * r)
    return out


if __name__ == '__main__':
    base = D.Lb
    cands = {
        '基线 mean4cls': base,
        'ef3(全局反馈)': MX.err_feedback(base, L_ACT, k=3, damp=0.35),
        'ef3cls(类别感知反馈)': err_feedback_cls(base, L_ACT, k=3, damp=0.35),
        'ef2cls(k=2,d=0.5)': err_feedback_cls(base, L_ACT, k=2, damp=0.5),
        'ef4cls(k=4,d=0.25)': err_feedback_cls(base, L_ACT, k=4, damp=0.25),
        'lvl(水平递推)': MX.level_smooth(base, L_ACT, alpha=0.35),
        'LGBM': ML.ml_forecast('L', 'lgbm'),
    }
    print(f'{"模型":26s} {"总MAE":>8s} {"周五六":>8s} {"其他":>8s}')
    res = {}
    for k, v in cands.items():
        a, b, c = mae_by_class(v)
        res[k] = (a, b, c)
        print(f'{k:26s} {a:8.2f} {b:8.2f} {c:8.2f}')

    # 类别感知融合: 基线 + 类别反馈(超参只在 1 月标定集上选, 2-12 月为外样本)
    best = None
    for k in (2, 3, 4, 5):
        for dmp in (0.15, 0.2, 0.25, 0.3, 0.4):
            v = err_feedback_cls(base, L_ACT, k=k, damp=dmp)
            e = np.abs(v - L_ACT).mean(axis=1) * 6
            val = e[7:31].mean()                 # 1 月标定段(与报童分位同口径)
            if best is None or val < best[0]:
                best = (val, k, dmp, v)
    print(f'\n[类别反馈选参(1月标定)] k={best[1]}, damp={best[2]}, 1月MAE={best[0]:.2f} kW')
    a, b, c = mae_by_class(best[3])
    print(f'[最终负荷模型(2-12月外样本)] 总 {a:.2f} / 周五六 {b:.2f} / 其他 {c:.2f} kW')
    ab, bb, cb = mae_by_class(base)
    print(f'[基线对照(2-12月)]            总 {ab:.2f} / 周五六 {bb:.2f} / 其他 {cb:.2f} kW')
    np.savez_compressed(os.path.join(DATA, 'load_best.npz'),
                        L_efcls=best[3], **{f'L|{k}': v for k, v in cands.items()})
    print('saved -> data/load_best.npz')
