# -*- coding: utf-8 -*-
"""
第二遍：数据工程流水线
统一时间轴(365天×144个10min时段) + 多表join + 附件3清洗插值 + 衍生特征 + 质量核验
输出: 03_数据工程/data 下 parquet/csv/json + fig 目录 + log.txt
"""
import sys, io, os, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

BASE = r'02_赛题数据\附件'
OUT = r'03_数据工程'
DATA = os.path.join(OUT, 'data')
os.makedirs(DATA, exist_ok=True)
FIG = os.path.join(OUT, 'fig')
os.makedirs(FIG, exist_ok=True)
LOG = []
def log(msg):
    LOG.append(str(msg))

N_SLOT = 144
DAT = pd.date_range('2025-01-01', '2025-12-31', freq='D')  # 365 days
log(f'[axis] 规范时间轴: {DAT[0].date()} .. {DAT[-1].date()}, 天数={len(DAT)}, 每天 {N_SLOT} 个10min时段')

# ---------- 1. 附件1 典型日 ----------
a1 = pd.read_excel(os.path.join(BASE, '附件1.xlsx'))
a1.columns = ['tlabel', 'price', 'load', 'pvfc']
def tlabel2slot(s):
    s = str(s)
    if s.endswith('+1'):
        return 144
    parts = s.split(':')
    h, m = int(parts[0]), int(parts[1])
    return (h * 60 + m) // 10
a1['slot'] = a1['tlabel'].map(tlabel2slot)
log(f'[附件1] rows={len(a1)}, 时段标签映射 1..144 覆盖={sorted(a1.slot)==list(range(1,145))}, 缺失={int(a1.isna().sum().sum())}, 负值={int((a1[["price","load","pvfc"]]<0).sum().sum())}')
typ = a1.set_index('slot')[['price', 'load', 'pvfc']].rename(
    columns={'price': 'price_typ', 'load': 'load_typ', 'pvfc': 'pvfc_typ'}).sort_index()
typ.to_parquet(os.path.join(DATA, 'typ_day.parquet'))
log(f'[附件1] 典型日: 电价 {typ.price_typ.min():.4f}~{typ.price_typ.max():.4f} 元/kWh, 负载 {typ.load_typ.min():.0f}~{typ.load_typ.max():.0f} kW, PV预测 {typ.pvfc_typ.min():.1f}~{typ.pvfc_typ.max():.1f} kW')

# ---------- 2. 附件2 负载/PV + 附件4 电价 (wide 365x145 -> long) ----------
def wide2long(path, sheet, varname):
    d = pd.read_excel(os.path.join(BASE, path), sheet_name=sheet)
    datecol = d.columns[0]
    d = d.rename(columns={datecol: 'date'})
    assert d.shape[1] - 1 == N_SLOT, f'{path}/{sheet} 时段列数异常: {d.shape[1]-1}'
    d['date'] = pd.to_datetime(d['date'])
    # 列顺序即 slot 1..144 (第j列=当天第j个10min区间, 与"区间结束时刻"标注对齐)
    vals = d.iloc[:, 1:].to_numpy(dtype=float)
    long = pd.DataFrame({
        'date': np.repeat(d['date'].to_numpy(), N_SLOT),
        'slot': np.tile(np.arange(1, N_SLOT + 1), len(d)),
        varname: vals.ravel(),
    })
    return long, d

load_l, load_w = wide2long('附件2.xlsx', '小区负载', 'load_kw')
pv_l, pv_w = wide2long('附件2.xlsx', '光伏发电实际功率', 'pv_kw')
pr_l, pr_w = wide2long('附件4.xlsx', 'Sheet1', 'price_q4')
log(f'[附件2/附件4] wide->long: 各 {len(load_l)} 行 (=365x144={365*N_SLOT}); 日期重复={int(load_w.date.duplicated().sum())}; 缺失={int(load_w.isna().sum().sum())+int(pv_w.isna().sum().sum())+int(pr_w.isna().sum().sum())}')

# ---------- 3. 附件3 预报清洗 ----------
a3 = pd.read_excel(os.path.join(BASE, '附件3.xlsx'))
n_null_date_before = int(a3['日期'].isna().sum())
a3['日期'] = a3['日期'].ffill()
a3['日期'] = pd.to_datetime(a3['日期'])
ih_map = {'0:00': 0, '6:00': 6, '12:00': 12, '18:00': 18}
a3['issue_hour'] = a3['预报时刻'].map(ih_map)
assert a3['issue_hour'].isna().sum() == 0
# 每天4行且顺序 0,6,12,18
order_ok = (a3.groupby('日期')['issue_hour'].apply(list).apply(lambda x: x == [0, 6, 12, 18])).all()
log(f'[附件3] 日期空值前填 {n_null_date_before}->0 (结构性空: 每日后3行合并单元格); 每日4行且顺序[0,6,12,18]={order_ok}; 重复(日期,发布时刻)={int(a3.duplicated(["日期","issue_hour"]).sum())}; 数值缺失={int(a3.iloc[:,3:].isna().sum().sum())}, 负值={int((a3.iloc[:,3:]<0).sum().sum())}')
fc_cols = [f'预报{i}小时' for i in range(1, 25)]
rows = []
for _, r in a3.iterrows():
    d0, h0 = r['日期'], r['issue_hour']
    for k in range(1, 25):
        abs_dt = d0 + pd.Timedelta(hours=h0 + k)
        rows.append((d0, h0, k, abs_dt.date(), abs_dt.hour, float(r[f'预报{k}小时'])))
fc = pd.DataFrame(rows, columns=['date', 'issue_hour', 'horizon', 'fc_date', 'fc_hour', 'fc_kw'])
log(f'[附件3] long化: {len(fc)} 行 (=1460x24); fc_kw 范围 {fc.fc_kw.min():.1f}~{fc.fc_kw.max():.1f}; 负值={int((fc.fc_kw<0).sum())}')
fc.to_parquet(os.path.join(DATA, 'fc_hourly.parquet'))

# ---------- 4. 统一长表 join ----------
day = load_l.merge(pv_l, on=['date', 'slot'], how='outer', validate='one_to_one')
day = day.merge(pr_l, on=['date', 'slot'], how='outer', validate='one_to_one')
typ_s = typ.reset_index().rename(columns={'slot': 'slot'})
day = day.merge(typ_s, on='slot', how='left', validate='many_to_one')
assert len(day) == 365 * N_SLOT and day[['load_kw', 'pv_kw', 'price_q4']].isna().sum().sum() == 0
day['t_start_min'] = (day['slot'] - 1) * 10
day['hour'] = day['t_start_min'] // 60
day['load_kwh'] = day['load_kw'] / 6.0
day['pv_kwh'] = day['pv_kw'] / 6.0
day['doy'] = day['date'].dt.dayofyear
day['month'] = day['date'].dt.month
day['weekday'] = day['date'].dt.weekday
day = day.sort_values(['date', 'slot']).reset_index(drop=True)
day.to_parquet(os.path.join(DATA, 'day_slots.parquet'))
log(f'[join] 统一长表 day_slots: {len(day)} 行, 列={list(day.columns)}; 负载 {day.load_kw.min():.0f}~{day.load_kw.max():.0f} kW; PV {day.pv_kw.min():.2f}~{day.pv_kw.max():.1f} kW; 电价q4 {day.price_q4.min():.4f}~{day.price_q4.max():.4f}; 近零电价(<0.05)时段数={int((day.price_q4<0.05).sum())}')

# ---------- 5. 0:00 计划用 PV 预报 -> 144 时段线性插值 ----------
# 0:00发布: 点 1:00..24:00 = 预报1..24; 点 0:00 = 前一日18:00发布的预报6小时
fc_piv = None  # (仅占位, 各时点预报直接经 get_pt 索引)
def get_pt(d, h, iss_d, iss_h, fallback=0.0):
    k = (24 + h - iss_h) % 24 or 24
    sub = fc[(fc.date == pd.Timestamp(iss_d)) & (fc.issue_hour == iss_h) & (fc.horizon == k)]
    if len(sub) == 0:
        return fallback          # 链条首日(2025-01-01)无前一日18:00发布 -> 夜间PV=0
    return float(sub.fc_kw.iloc[0])
rows0 = []
for d in DAT:
    pts = np.zeros(25)
    pts[0] = get_pt(d, 0, d - pd.Timedelta(days=1), 18)   # 0:00点来自昨日18:00发布
    for hh in range(1, 25):
        pts[hh] = get_pt(d, hh, d, 0)
    xs = np.arange(0, 25, 1.0)                      # 小时点
    mids = (np.arange(1, N_SLOT + 1) - 1) * 10 + 5  # 各时段中点(分钟)
    vals = np.interp(mids / 60.0, xs, pts)
    rows0.append(pd.DataFrame({'date': d, 'slot': np.arange(1, N_SLOT + 1), 'fc0_kw': vals}))
fc0 = pd.concat(rows0, ignore_index=True)
assert len(fc0) == 365 * N_SLOT and fc0.fc0_kw.min() >= -1e-9
fc0.to_parquet(os.path.join(DATA, 'fc_slots_plan.parquet'))
log(f'[fc0] 0:00发布预报插值到144时段: 行数={len(fc0)}, 范围 {fc0.fc0_kw.min():.2f}~{fc0.fc0_kw.max():.1f} kW, 负值={int((fc0.fc0_kw<0).sum())}')

# ---------- 6. 6/12/18时点调整用预报 (覆盖发布时刻->当日24:00) ----------
rowsa = []
for d in DAT:
    for h0 in (6, 12, 18):
        nslots = N_SLOT - h0 * 6          # 从 h0:00 到 24:00 的时段数
        nh = 24 - h0 + 1                   # 小时点数 h0..24
        pts = np.zeros(nh)
        pts[0] = get_pt(d, h0, d, 0)       # 发布时刻点来自当日0:00发布(预报h0小时)
        for hh in range(h0 + 1, 25):
            pts[hh - h0] = get_pt(d, hh, d, h0)
        xs = np.arange(h0, 25, 1.0)
        mids = (np.arange(h0 * 6 + 1, N_SLOT + 1) - 1) * 10 + 5
        vals = np.interp(mids / 60.0, xs, pts)
        rowsa.append(pd.DataFrame({'date': d, 'issue_hour': h0,
                                   'slot': np.arange(h0 * 6 + 1, N_SLOT + 1), 'fc_kw': vals}))
fca = pd.concat(rowsa, ignore_index=True)
fca.to_parquet(os.path.join(DATA, 'fc_slots_adj.parquet'))
log(f'[fca] 6/12/18时点预报插值: 行数={len(fca)}, 覆盖 6:00起108时段+12:00起72时段+18:00起36时段, 负值={int((fca.fc_kw<0).sum())}')

# ---------- 7. 外推(持久)序列: day d 计划用 day d-1 实测 ----------
pers = day[['date', 'slot', 'load_kw', 'pv_kw']].copy()
pers['prev_date'] = pers['date'] - pd.Timedelta(days=1)
shifted = pers[['date', 'slot', 'load_kw', 'pv_kw']].rename(columns={'date': 'prev_date', 'load_kw': 'load_prev_kw', 'pv_kw': 'pv_prev_kw'})
pers = pers.merge(shifted, on=['prev_date', 'slot'], how='left', validate='one_to_one')
pers = pers[['date', 'slot', 'load_prev_kw', 'pv_prev_kw']].sort_values(['date', 'slot'])
# 2025-01-01 无前一日 -> 用附件1典型日(口径: 全年链首日以典型日曲线为计划依据)
typ_s2 = typ.reset_index()[['slot', 'load_typ', 'pvfc_typ']]
pers = pers.merge(typ_s2, on='slot', how='left')
mask = pers['date'] == pd.Timestamp('2025-01-01')
pers.loc[mask, 'load_prev_kw'] = pers.loc[mask, 'load_typ']
pers.loc[mask, 'pv_prev_kw'] = pers.loc[mask, 'pvfc_typ']
pers = pers.drop(columns=['load_typ', 'pvfc_typ'])
log(f'[pers] 外推序列: 行数={len(pers)}, 2025-01-01用典型日填充(链条首日), 其余=前一日实测; 缺失={int(pers.isna().sum().sum())}')
pers.to_parquet(os.path.join(DATA, 'persistence.parquet'))

# ---------- 8. 预报误差刻画 (vs 对齐时段值 / vs 小时均值) ----------
pv_slot = day.set_index(['date', 'slot'])['pv_kw']
recs = []
for _, r in fc.iterrows():
    ad = pd.Timestamp(r['fc_date']); ah = r['fc_hour']
    s_end = ah * 6                       # 结束于该整点的时段序号 (1..144)
    try:
        aligned = float(pv_slot.loc[(ad, s_end)]) if 1 <= s_end <= N_SLOT else np.nan
        s0, s1 = (ah - 1) * 6 + 1, ah * 6
        hourly_mean = float(pv_slot.loc[(ad, slice(s0, s1))].mean())
    except KeyError:
        aligned, hourly_mean = np.nan, np.nan   # 跨年(2026)目标小时无实测
    recs.append((r['date'], r['issue_hour'], r['horizon'], r['fc_kw'], aligned, hourly_mean))
fce = pd.DataFrame(recs, columns=['date', 'issue_hour', 'horizon', 'fc_kw', 'pv_aligned', 'pv_hourly'])
fce['err_aligned'] = fce['fc_kw'] - fce['pv_aligned']
fce['err_hourly'] = fce['fc_kw'] - fce['pv_hourly']
# 按发布时刻+提前期桶汇总
buckets = pd.cut(fce['horizon'], [0, 6, 12, 18, 24], labels=['1-6h', '7-12h', '13-18h', '19-24h'])
g = fce.groupby(['issue_hour', buckets]).agg(
    n=('horizon', 'size'), mae_a=('err_aligned', lambda s: s.abs().mean()),
    rmse_a=('err_aligned', lambda s: np.sqrt((s ** 2).mean())),
    me_a=('err_aligned', 'mean'),
    mae_h=('err_hourly', lambda s: s.abs().mean()),
    rmse_h=('err_hourly', lambda s: np.sqrt((s ** 2).mean())),
    me_h=('err_hourly', 'mean')).reset_index()
g.columns = ['issue_hour', 'horizon_bucket', 'n', 'mae_aligned', 'rmse_aligned', 'me_aligned', 'mae_hourly', 'rmse_hourly', 'me_hourly']
g.to_csv(os.path.join(DATA, 'fc_error_by_horizon.csv'), index=False, encoding='utf-8-sig')
log(f'[fc_error] 预报误差表: 行数={len(g)}; 0:00发布 MAE(对齐) {g[g.issue_hour==0].mae_aligned.mean():.1f} kW, 6:00发布 {g[g.issue_hour==6].mae_aligned.mean():.1f}, 12:00 {g[g.issue_hour==12].mae_aligned.mean():.1f}, 18:00 {g[g.issue_hour==18].mae_aligned.mean():.1f}')
# 相邻发布(6小时前)对同一绝对小时的预报更新幅度
fc2 = fc.copy()
fc2['abs_dt'] = pd.to_datetime(fc2['fc_date']) + pd.to_timedelta(fc2['fc_hour'], unit='h')
fc2['issue_dt'] = fc2['date'] + pd.to_timedelta(fc2['issue_hour'], unit='h')
m = fc2.merge(fc2[['issue_dt', 'abs_dt', 'fc_kw']].rename(columns={'fc_kw': 'fc_older'}),
              left_on=['abs_dt'], right_on=['abs_dt'], suffixes=('', '_old'))
m = m[m['issue_dt'] - m['issue_dt_old'] == pd.Timedelta(hours=6)]
log(f'[fc_update] 同绝对小时6小时后更新: 平均|更新量|={m.fc_kw.sub(m.fc_older).abs().mean():.1f} kW (n={len(m)})')

# ---------- 9. 每日汇总统计 ----------
dt = day.groupby('date').agg(
    load_energy_kwh=('load_kwh', 'sum'), pv_energy_kwh=('pv_kwh', 'sum'),
    load_peak_kw=('load_kw', 'max'), pv_peak_kw=('pv_kw', 'max'),
    price_q4_mean=('price_q4', 'mean'), price_q4_min=('price_q4', 'min'), price_q4_max=('price_q4', 'max'),
    price_typ_mean=('price_typ', 'mean'), price_typ_min=('price_typ', 'min'), price_typ_max=('price_typ', 'max')).reset_index()
dt.to_csv(os.path.join(DATA, 'daily_totals.csv'), index=False, encoding='utf-8-sig')
log(f'[daily] 日均负载电量 {dt.load_energy_kwh.mean():.0f} kWh, 日均光伏电量 {dt.pv_energy_kwh.mean():.0f} kWh, 日均电价(q4) {dt.price_q4_mean.mean():.4f} 元/kWh')

# ---------- 10. 结果模板映射 ----------
tmpl = {}
r1 = pd.read_excel(os.path.join(BASE, '附件5', 'result1.xlsx'), sheet_name='计划购电量', header=None)
tmpl['result1_rows'] = r1[0].tolist()
r2 = pd.read_excel(os.path.join(BASE, '附件5', 'result2.xlsx'), sheet_name='计划购电量', header=0)
tmpl['result2_cols'] = list(r2.columns)
tmpl['result2_dates'] = [str(x) for x in r2.iloc[:, 0].tolist()]
r2c = pd.read_excel(os.path.join(BASE, '附件5', 'result2.xlsx'), sheet_name='充放电量', header=0)
tmpl['cd_blocks'] = ['0:00-4:00', '4:00-8:00', '8:00-12:00', '12:00-16:00', '16:00-20:00', '20:00-24:00']
tmpl['cd_slot_blocks'] = [[1, 24], [25, 48], [49, 72], [73, 96], [97, 120], [121, 144]]
tmpl['emergency_format'] = '每日一行/连续时段合并为一个时间段; 日期仅写该日首行'
log(f"[tmpl] result1 计划购电量行标签 {len(tmpl['result1_rows'])}; result2 列数={len(tmpl['result2_cols'])}, 日期 {tmpl['result2_dates'][0]}..{tmpl['result2_dates'][-1]}, 共{len(tmpl['result2_dates'])}天")
log(f"[tmpl] result2 列标签: 首={tmpl['result2_cols'][1]}, 第144列={tmpl['result2_cols'][N_SLOT]}, 尾={tmpl['result2_cols'][-2:]}  => 位置对齐: 模板第j列=当天第j个10min区间")
with open(os.path.join(DATA, 'template_map.json'), 'w', encoding='utf-8') as f:
    json.dump(tmpl, f, ensure_ascii=False, indent=1, default=str)

# ---------- 11. 清洗前后统计对比表 ----------
stats = []
def add(tbl, aspect, before, after, note=''):
    stats.append({'表': tbl, '检查项': aspect, '清洗前': before, '清洗后': after, '说明': note})
add('附件1 典型日', '形态', '宽表 144行x4列(时间标签)', '长表 typ_day: 144x4(slot,电价,负载,PV预测)', '144个区间标签(结束时刻)映射为slot 1..144')
add('附件1 典型日', '缺失', '0', '0', '')
add('附件1 典型日', '负值', '0', '0', '')
add('附件1 典型日', '数值范围', f"电价0.3713~1.3952, 负载3309~5959kW, PV 0~7612kW", '同左(未截断)', '极值保留, 为套利与峰值负荷信息')
add('附件2 负载', '形态', '宽表 365x145(日期+144时段列)', '长表 52560行x3(date,slot,load_kw)', '第j列=当天第j个10min区间; 列头为区间结束时刻')
add('附件2 负载', '缺失', '0', '0', '')
add('附件2 负载', '负值', '0', '0', '')
add('附件2 负载', '重复日期', '0', '0(join后one_to_one校验通过)', '')
add('附件2 光伏', '形态', '宽表 365x145', '长表 52560行(同上)', '夜间存在0~0.03kW的测量噪声, 保留(物理非零, 非缺失)')
add('附件2 光伏', '缺失/负值', '0 / 0', '0 / 0', '')
add('附件2 光伏', '数值范围', '0~6566kW', '同左', '')
add('附件3 预报', '形态', '长表 1460行x26列(日期,发布时刻,预报1-24小时)', 'fc_hourly 35040行x6 + fc_slots_plan 52560行 + fc_slots_adj 78840行', '小时点预报线性插值到10min时段')
add('附件3 预报', '缺失', '1095(每日后3行日期空, 结构性合并单元格)', '0(向下填充)', '缺失机制=结构空, 非随机缺失')
add('附件3 预报', '负值', '0', '0(插值后亦非负)', '线性插值于非负点之间保持非负')
add('附件3 预报', '时间口径', '发布时刻0/6/12/18; 预报k小时=发布后第k个小时整点功率', '映射到绝对(日期,整点小时), 可跨天', '0:00发布预报24小时=当日1:00-24:00点')
add('附件4 电价', '形态', '宽表 365x145', '长表 52560行(price_q4)', '')
add('附件4 电价', '缺失/负值', '0 / 0', '0 / 0', '')
add('附件4 电价', '数值范围', '0.0076~1.7936 元/kWh', '同左; 近零(<0.05)时段 共{}个'.format(int((day.price_q4<0.05).sum())), '近零电价时段=储能在弃光/低价期充电的套利窗口, 保留')
add('统一长表', 'join', '4张表口径不一(结束时刻标签 vs 跨度标签 vs 小时点)', 'day_slots 52560行: date,slot,t_start_min,hour,load_kw,pv_kw,price_q4,price_typ,load_kwh,pv_kwh,...', '主键(date,slot); 结果模板按"位置对齐"复用同一slot轴')
add('统一长表', '覆盖', '附件2/4: 2025.1.1-12.31(365天); 结果窗口 2025.2.1-12.31(334天)', '全覆盖365天; 1月为SOC预热+外推历史', '结果文件仅需334天, 但链式SOC与0:00信息集需全年')
add('外推序列', '构造', '无', 'persistence 52560行: date d = 前一日d-1实测负载/PV; 2025-01-01用典型日曲线', 'Q2计划信息集口径: 0:00不可用当日实际数据(用户定案)')
add('模板映射', '结构', 'result1:144行; result2/3/4: 334天x147列(144时段+全天购电量+全天购电费)', 'template_map.json: 列j=当天第j个区间; 4h块=[1,24]...; 紧急购电=每日合并连续段', '模板跨度标签整体后移10min, 采用位置对齐')
stats_df = pd.DataFrame(stats)
stats_df.to_csv(os.path.join(DATA, 'stats_before_after.csv'), index=False, encoding='utf-8-sig')
log(f'[stats] 清洗前后对比表: {len(stats_df)} 行 -> stats_before_after.csv')

with open(os.path.join(OUT, 'log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG))
print('BUILD OK,', len(LOG), 'log lines')
