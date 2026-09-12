# -*- coding: utf-8 -*-
"""把两个脚本的路径更新为重组后的目录结构"""
import io
p = r'03_数据工程\scripts\build_dataset.py'
s = open(p, encoding='utf-8').read()
s = s.replace(r"BASE = r'C题\附件'", r"BASE = r'02_赛题数据\附件'")
s = s.replace(r"OUT = r'C题\数据工程'", r"OUT = r'03_数据工程'" + "\nDATA = os.path.join(OUT, 'data')\nos.makedirs(DATA, exist_ok=True)")
for name in ['typ_day', 'fc_hourly', 'day_slots', 'fc_slots_plan', 'fc_slots_adj',
             'persistence', 'fc_error_by_horizon', 'daily_totals', 'template_map',
             'stats_before_after']:
    s = s.replace("join(OUT, '%s" % name, "join(DATA, '%s" % name)
s = s.replace('输出: C题/数据工程/ 下 parquet/csv/json + fig 目录 + log.txt',
              '输出: 03_数据工程/data 下 parquet/csv/json + fig 目录 + log.txt')
open(p, 'w', encoding='utf-8').write(s)

p2 = r'03_数据工程\scripts\make_figs.py'
s2 = open(p2, encoding='utf-8').read()
s2 = s2.replace(r"OUT = r'C题\数据工程'", r"OUT = r'03_数据工程'")
for name in ['day_slots', 'typ_day', 'fc_slots_plan', 'fc_error_by_horizon', 'fc_hourly',
             'monthly_stats', 'constants.json']:
    s2 = s2.replace("os.path.join(OUT, '%s" % name, "os.path.join(OUT, 'data', '%s" % name)
open(p2, 'w', encoding='utf-8').write(s2)
print('patched ok')
