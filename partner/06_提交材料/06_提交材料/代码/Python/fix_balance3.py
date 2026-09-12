# -*- coding: utf-8 -*-
"""实时平衡控制的最终实现(严格物理可行):
   ① 按当前 SOC 把计划充放电裁剪为可行范围;
   ② 缺额: 先减充、再增放(受剩余功率与可用电量限制), 仍缺 -> 紧急购电;
   ③ 富余: 先减放、再增充(受剩余功率与容量限制), 仍富余 -> 弃光。
   因果性: 时段 t 只用 t 时刻实测与计划量, 不使用未来实测。
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

FUNC = '''def {NAME}(P0, Cpl, Dpl, soc0, L, PV, reserve=SOC_MIN):
    h = len(P0)
    C = np.zeros(h); D = np.zeros(h); Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        # ① 计划量裁剪为物理可行
        c_ok = min(max(float(Cpl[t]), 0.0), P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C))
        d_ok = min(max(float(Dpl[t]), 0.0), P_MAX, ETA_D * max(0.0, soc - reserve + ETA_C * c_ok))
        C[t], D[t] = c_ok, d_ok
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        # ② 缺额: 先减充, 再增放
        if gap > 1e-9:
            cut = min(C[t], gap); C[t] -= cut; gap -= cut
            if gap > 1e-9:
                avail = ETA_D * max(0.0, soc - reserve + ETA_C * C[t]) - D[t]
                extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, avail))
                D[t] += extra; gap -= extra
        # ③ 富余: 先减放, 再增充
        elif gap < -1e-9:
            cut = min(D[t], -gap); D[t] -= cut; gap += cut
            if gap < -1e-9:
                room = max(0.0, (SOC_MAX - soc + D[t] / ETA_D)) / ETA_C - C[t]
                extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, room))
                C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
        assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6, ('SOC越界', t, soc)
        assert C[t] <= P_MAX + 1e-9 and D[t] <= P_MAX + 1e-9, ('功率越界', t)
        assert min(C[t], D[t]) <= 1e-9, ('同时充放', t)
    return C, D, soc, Em
'''

for f, name in [(r'04_建模求解\scripts\rebuild_causal.py', 'balance'),
                (r'04_建模求解\scripts\settle_three_way.py', 'bal')]:
    s = open(f, encoding='utf-8').read()
    i = s.index('def %s(' % name)
    j2 = s.find('\ndef ', i + 1); j3 = s.find('\nprint(', i + 1); j4 = s.find('\n# ', i + 1)
    end = min([x for x in [j2, j3, j4, len(s)] if x > i])
    s = s[:i] + FUNC.replace('{NAME}', name) + s[end:]
    open(f, 'w', encoding='utf-8').write(s)
    print('replaced', name, 'in', f)
