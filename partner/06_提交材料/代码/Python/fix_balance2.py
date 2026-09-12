# -*- coding: utf-8 -*-
"""用正确的实时平衡规则替换旧实现(严格满足功率与 SOC 约束, 避免同时充放)"""
import io, sys, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

NEW_FUNC = '''def balance(P0, Cpl, Dpl, soc0, L, PV, reserve=SOC_MIN):
    """实时平衡控制(严格因果, 满足全部物理约束):
       缺额: 先减少充电, 再增加放电(受剩余功率与可用电量限制), 仍缺则紧急购电;
       富余: 先减少放电, 再增加充电(受剩余功率与容量限制), 仍富余则弃光。"""
    h = len(P0)
    C = Cpl.astype(float).copy(); D = Dpl.astype(float).copy()
    Em = np.zeros(h); soc = float(soc0)
    for t in range(h):
        gap = L[t] - (PV[t] + P0[t] + D[t] - C[t])
        if gap > 1e-9:                                    # 缺额
            cut = min(C[t], gap); C[t] -= cut; gap -= cut  # 先减充
            if gap > 1e-9:                                 # 再增放
                avail = max(0.0, (soc - reserve) * ETA_D) + C[t] * ETA_C
                extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, avail))
                D[t] += extra; gap -= extra
        elif gap < -1e-9:                                  # 富余
            cut = min(D[t], -gap); D[t] -= cut; gap += cut  # 先减放
            if gap < -1e-9:                                 # 再增充
                room = max(0.0, (SOC_MAX - soc) / ETA_C) + D[t] / ETA_D
                extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, room))
                C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
        assert SOC_MIN - 1e-6 <= soc <= SOC_MAX + 1e-6, ('SOC 越界', t, soc)
        assert C[t] <= P_MAX + 1e-9 and D[t] <= P_MAX + 1e-9, ('功率越界', t, C[t], D[t])
    return C, D, soc, Em
'''

for f, old_start in [(r'04_建模求解\scripts\rebuild_causal.py', 'def balance('),
                     (r'04_建模求解\scripts\settle_three_way.py', 'def bal(')]:
    s = open(f, encoding='utf-8').read()
    i = s.index(old_start)
    # 找到函数结束(下一个顶格 def 或注释块)
    j = s.index('\ndef ', i + 1) if '\ndef ' in s[i + 1:] else len(s)
    j2 = s.find('\ndef ', i + 1)
    j3 = s.find('\nprint(', i + 1)
    end = min([x for x in [j2, j3, len(s)] if x > i])
    new = NEW_FUNC if 'rebuild_causal' in f else NEW_FUNC.replace('def balance(', 'def bal(')
    s = s[:i] + new + s[end:]
    open(f, 'w', encoding='utf-8').write(s)
    print('replaced balance in', f)
