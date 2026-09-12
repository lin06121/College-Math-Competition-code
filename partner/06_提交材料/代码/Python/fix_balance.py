# -*- coding: utf-8 -*-
"""修正实时平衡规则: 严格限制总充放电功率 <= P_MAX, 不做破坏递推的 SOC 夹取"""
import io, sys, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OLD = '''        if gap > 1e-9:                                    # 缺额: 增放
            avail = max(0.0, (soc - max(SOC_MIN, reserve)) * ETA_D)
            extra = min(gap, P_MAX, avail)
            D[t] += extra; gap -= extra
        elif gap < -1e-9:                                 # 富余: 增充
            room = max(0.0, (SOC_MAX - soc) / ETA_C)
            extra = min(-gap, P_MAX, room)
            C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D
        soc = float(min(max(soc, SOC_MIN), SOC_MAX))      # 夹取, 防止浮点越界导致次日 LP 判不可行
    return C, D, soc, Em'''

NEW = '''        if gap > 1e-9:                                    # 缺额: 增放(受剩余功率与可用电量限制)
            avail_p = max(0.0, P_MAX - D[t])
            avail_e = max(0.0, (soc - max(SOC_MIN, reserve)) * ETA_D)
            extra = min(gap, avail_p, avail_e)
            D[t] += extra; gap -= extra
        elif gap < -1e-9:                                 # 富余: 增充(受剩余功率与容量限制)
            avail_p = max(0.0, P_MAX - C[t])
            room = max(0.0, (SOC_MAX - soc) / ETA_C)
            extra = min(-gap, avail_p, room)
            C[t] += extra; gap += extra
        Em[t] = max(0.0, gap)
        soc = soc + ETA_C * C[t] - D[t] / ETA_D           # 严格递推(不再夹取)
    return C, D, soc, Em'''

OLD_SMALL = '''        if gap > 1e-9:
            extra = min(gap, P_MAX, max(0.0, (soc - SOC_MIN) * ETA_D)); D[t] += extra; gap -= extra
        elif gap < -1e-9:
            extra = min(-gap, P_MAX, max(0.0, (SOC_MAX - soc) / ETA_C)); C[t] += extra; gap += extra
        Em[t] = max(0.0, gap); soc = float(min(max(soc + ETA_C*C[t] - D[t]/ETA_D, SOC_MIN), SOC_MAX))
    return C, D, soc, float(5*price @ Em)'''

NEW_SMALL = '''        if gap > 1e-9:
            extra = min(gap, max(0.0, P_MAX - D[t]), max(0.0, (soc - SOC_MIN) * ETA_D)); D[t] += extra; gap -= extra
        elif gap < -1e-9:
            extra = min(-gap, max(0.0, P_MAX - C[t]), max(0.0, (SOC_MAX - soc) / ETA_C)); C[t] += extra; gap += extra
        Em[t] = max(0.0, gap); soc = soc + ETA_C*C[t] - D[t]/ETA_D
    return C, D, soc, float(5*price @ Em)'''

for f in [r'04_建模求解\scripts\rebuild_causal.py', r'04_建模求解\scripts\settle_three_way.py']:
    s = open(f, encoding='utf-8').read()
    n = 0
    if OLD in s:
        s = s.replace(OLD, NEW); n += 1
    if OLD_SMALL in s:
        s = s.replace(OLD_SMALL, NEW_SMALL); n += 1
    open(f, 'w', encoding='utf-8').write(s)
    print(f, '-> 替换', n, '处')
