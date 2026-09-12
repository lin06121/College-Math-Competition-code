# -*- coding: utf-8 -*-
"""负荷模型(三): 按"周五六/其他"两类分别做凸组合权重选择(1月标定), 2-12月外样本评价"""
import sys, io, os, itertools, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from fc_common import D, L_ACT, IS_LOW, DATA, n, H
import models_extra as MX
from eval_load2 import shape_by_class, panel_ridge

LOW = IS_LOW
m2 = D.m2


def mae_cls(v):
    e = np.abs(v[m2:] - L_ACT[m2:]).mean(axis=1) * 6
    return float(e.mean()), float(e[LOW[m2:]].mean()), float(e[~LOW[m2:]].mean())


def mae_idx(v, lo, hi):
    e = np.abs(v - L_ACT).mean(axis=1) * 6
    return float(e[lo:hi].mean())


def simplex_weights(m, step=0.1):
    """m 个模型的凸组合权重网格"""
    grid = np.arange(0, 1 + 1e-9, step)
    out = []
    for comb in itertools.product(grid, repeat=m - 1):
        s = sum(comb)
        if s <= 1 + 1e-9:
            out.append(list(comb) + [1 - s])
    return np.array(out)


if __name__ == '__main__':
    base = D.Lb

    def cls_level(base, A_act, K=4):
        """同类近 K 日总量均值; 同类历史为空时退回基线当日总量(避免 NaN)"""
        lv = np.zeros(n)
        for j in range(n):
            hist = [q for q in range(j) if LOW[q] == LOW[j]][-K:]
            lv[j] = A_act[hist].sum(axis=1).mean() if hist else base[j].sum()
        return lv

    sh6 = shape_by_class(base, L_ACT, K=6) * cls_level(base, L_ACT)[:, None]
    lvl = MX.level_smooth(base, L_ACT, alpha=0.35)
    ef3 = MX.err_feedback(base, L_ACT, k=3, damp=0.35)
    cands = [base, sh6, lvl, ef3]
    names = ['基线mean4cls', '水平×形状K6', '递推水平lvl', '全局反馈ef3']
    W = simplex_weights(len(cands), step=0.1)

    out = np.zeros((n, H))
    info = {}
    for cls, mask_rows in [('周五六', LOW), ('其他', ~LOW)]:
        rows_lo = [j for j in range(7, 31) if mask_rows[j]]      # 1月标定段
        X = np.stack([c[rows_lo] for c in cands])                # (m, days, H)
        Y = L_ACT[rows_lo]
        best = None
        for w in W:
            P = np.tensordot(w, X, axes=1)
            e = np.abs(P - Y).mean()
            if best is None or e < best[0]:
                best = (e, w)
        info[cls] = dict(w=best[1].tolist(), jan_mae=float(best[0] * 6))
        print(f'[{cls}] 1月最优权重 ' + ', '.join(f'{nm}:{wi:.1f}' for nm, wi in zip(names, best[1])))
        P = np.tensordot(best[1], np.stack(cands), axes=1)
        out[mask_rows] = P[mask_rows]

    a, b, c = mae_cls(out)
    ab, bb, cb = mae_cls(base)
    print(f'\n[组合模型(2-12月)] 总 {a:.2f} / 周五六 {b:.2f} / 其他 {c:.2f} kW')
    print(f'[基线   (2-12月)] 总 {ab:.2f} / 周五六 {bb:.2f} / 其他 {cb:.2f} kW')
    print(f'[改善] 总 {a-ab:+.2f} / 周五六 {b-bb:+.2f} / 其他 {c-cb:+.2f} kW')
    np.savez_compressed(os.path.join(DATA, 'load_combined.npz'), L_comb=out, L_base=base,
                        L_sh6=sh6, L_lvl=lvl, L_ef3=ef3)
    json.dump(info, open(os.path.join(DATA, 'load_weights.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('saved -> data/load_combined.npz')
