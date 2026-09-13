# 2026 高教社杯全国大学生数学建模竞赛 C 题

**微网与外部电网电力调控策略** —— 求解代码、结果文件与竞赛论文。

微网由**光伏发电 + 小区负载 + 储能设备**构成，并与外部电网交互购电。在满足小区负载的前提下，
通过制定**购电计划**与**储能充放电策略**，使购电费用尽可能低。储能可在电价低或光伏富余时充电、
电价高时放电。

## 四个子问题

| 问题 | 设定 | 模型类型 |
|---|---|---|
| **问题 1** | 电价、负载取附件 1 典型日，光伏给定；制定每日 0:00 计划，要求 `E(0) = E(144)` | 确定性 MILP |
| **问题 2** | 电价固定；负载与光伏逐时变化；计划不足触发**紧急购电（5 倍电价）** | 报童裕度日前计划 + 因果实时结算 |
| **问题 3** | 每天 0:00/6:00/12:00/18:00 获得未来 24 h 光伏预报；计划与调整量偏差按 **0.5× / 1.5×** 电价计违约 | 多阶段滚动优化 |
| **问题 4** | 电价实时波动（附件 4），重算问题 2、3 | 电价预报 + 因果结算 |

## 快速开始

> 默认 `python` 不可用（只有标准库），**所有脚本必须使用 Anaconda 解释器**。

```bash
PY="/d/devtool/anaconda/conda/python.exe"     # Python 3.13.9

# 中文输出在 Windows 控制台会乱码，务必带上 UTF-8 环境变量
PYTHONUTF8=1 "$PY" code/run_q1.py             # 问题 1（秒级）
PYTHONUTF8=1 "$PY" code/run_q2.py             # 问题 2（约 12 s）
PYTHONUTF8=1 "$PY" code/run_q3.py             # 问题 3（约 1 min，--ablate 另跑对照）
PYTHONUTF8=1 "$PY" code/run_q4.py             # 问题 4（约 4 min）
```

数据预处理（首次运行或改动附件后执行一次，产物在 `data/processed/`）：

```bash
PYTHONUTF8=1 "$PY" code/preprocess.py         # 清洗 → 对齐 → 插值 → 特征主表
PYTHONUTF8=1 "$PY" code/features.py           # 日/月/季统计与峰谷分析
PYTHONUTF8=1 "$PY" code/visualize.py          # EDA 图 → figures/
```

已安装：`numpy` `pandas` `scipy` `scikit-learn` `xgboost` `matplotlib` `seaborn` `openpyxl`。
**未安装任何专用优化求解器**，MILP 统一使用 `scipy.optimize.milp`（HiGHS 后端）。

## 目录结构

```
math_model/
├── code/                 求解与数据处理代码（见下节）
├── data/processed/       预处理产物：清洗后 CSV/Parquet、矩阵 NPZ、逐时段/逐日明细
├── figures/              EDA 图（visualize.py 产出）
├── passage/              竞赛论文（LaTeX，xelatex 编译）
│   ├── main.tex          正文
│   ├── ai_usage.tex      AI 使用声明（单独提交）
│   ├── cumcmthesis.cls   模板
│   └── figures/          论文插图
├── problem/C题/          题目 PDF 与附件 1–5
├── partner/06_提交材料/  队友提交材料
├── result1.xlsx …        ★ 交付结果文件（根目录）
└── table*_q*.csv         论文表 1/2/3 素材
```

## 代码模块

代码按「数据 → 预测 → 求解 → 输出 → 制图」五层组织，`run_q*.py` 为一键入口。

### 入口脚本

| 脚本 | 作用 |
|---|---|
| `run_q1.py` | 读附件 1 → 建 MILP → 求解 → 校验 → 出表 |
| `run_q2.py` | 报童裕度日前计划 + 因果实时结算 → 校验 → 出结果（`--compare` 出对照实验） |
| `run_q3.py` | 联合标定 (q, λ) → 全年滚动求解 → 校验 → 出结果（`--ablate` 出时刻价值对照） |
| `run_q4.py` | 波动电价下重算问题 2/3 + 电价信息价值对照 |

### 数据层

| 脚本 | 作用 |
|---|---|
| `config.py` | 路径、储能参数、时间粒度、季节映射、预报时刻等全局常量 |
| `io_utils.py` | 四个附件的底层读取、整点预报插值到 10 min、主表构建、矩阵导出 |
| `preprocess.py` | 清洗与对齐，产出 `full_timeseries.parquet` 与 `matrices.npz` |
| `features.py` | 日/月/季统计、峰谷与周期性分析、预测误差 |
| `visualize.py` | 典型日曲线、热力图、星期/季节效应等 EDA 图 |

### 预测层

| 脚本 | 作用 |
|---|---|
| `method_q2.py` | 问题 2/3 共用的日前计划与因果结算方法（报童分位、`causal_dispatch`） |
| `forecast_q2.py` | 负载/光伏因果预测（只用目标日之前的历史） |
| `forecast_q3.py` | 附件 3 整点预报对齐到 10 min 网格 |
| `forecast_q4.py` | 附件 4 实时电价读取与因果电价预报 |

### 求解层

| 脚本 | 作用 |
|---|---|
| `daily_solver.py` | 通用单日储能经济调度 MILP（可选紧急购电、终端储能价值、弃光） |
| `solver_q1.py` | 问题 1：固定 `E(0)=E(144)=6000` |
| `solver_q2.py` | 问题 2：日前计划 + 逐日 SOC 滚动 |
| `solver_q3.py` | 问题 3：四个预报时刻的分段滚动优化 |
| `stochastic_q2.py` | 两阶段随机规划（对照方法） |

### 输出层

| 脚本 | 作用 |
|---|---|
| `format_result_q1.py` | 论文表 1/2 与 `result1.xlsx` |
| `format_result_q2.py` | 论文表 1/2/3、紧急购电事件、`result2.xlsx` |
| `format_result_q3.py` | 论文表 1/2/3、紧急购电事件、`result3.xlsx` |

> 输出统一**复制附件 5 的模板再写入**，以保证表头与格式一致。

### 对照实验与制图

`compare_q2.py`（计划依据/方法/结算口径对齐实验）、`oracle_q2.py`（预测信息价值分解）、
`q2_counterfactual.py`（执行机制/依据/方法/储能四项反事实）、`exp_q2_pv.py` / `exp_q2_pv2.py`
（光伏预测是否需季节项）、`probe_rule6.py`（预报误差口径复现）、`fig_q2_pv_season.py`、
`gen_fig_q1.py`、`make_figs_q2.py`（论文插图）。

## 方法要点

**功率平衡**（核心约束）

```
P_grid(t) + PV(t) + P_dis(t) = Load(t) + P_ch(t)
```

**储能参数**

| 参数 | 数值 |
|---|---|
| 最大容量 / 允许范围 | 12000 / 1200–10800 kWh |
| 最大充放电功率 | 5000 kW |
| 充放电效率 | 0.9 |
| 初始储电量（2025-01-01 0:00） | 6000 kWh |

动态方程 `E(t+1) = E(t) + η·P_ch(t)·Δt − P_dis(t)/η·Δt`，充放电由 0-1 变量互斥。
时间粒度 10 min，每天 144 时段，`Δt = 1/6 h`，电量 `kWh = 功率(kW) × 1/6`。

**各问关键处理**

- **问题 2**：净需求残差取 **q 分位**作为报童安全裕度（`q* = 0.8`），实时用**因果**平衡控制
  （只用当前实测净负载与当前 SOC，不偷看未来）。
- **问题 3**：0:00 定全天计划，6:00/12:00/18:00 用新预报分段调整；目标含**终端储能价值**
  `−λ·E(144)`，联合标定得 `q* = 0.60`、`λ* = 0.45` 元/kWh。
- **问题 4**：假设 0:00 时**不知道**当天电价，用近 W 日逐时段滚动均值为计划电价，
  结算仍用附件 4 实际电价。

## 主要结果

| 问题 | 总费用 | 备注 |
|---|---|---|
| 问题 1 | 35 126.95 元 | 全天购电 59 482.7 kWh |
| 问题 2 | 13 678 365.38 元 | 计划费 1319.4 万 + 紧急费 48.5 万（2.1–12.31） |
| 问题 3 | 13 383 402.49 元 | 违约费 202 815.57 元（占 1.5%） |
| 问题 4-2 | 14 408 076.05 元 | 波动电价下重算问题 2 |
| 问题 4-3 | 14 150 821.70 元 | 较 4-2 省 257 254 元（1.79%） |

论文指定日期（表 3）：2025.3.20、2025.6.21、2025.9.23、2025.12.21。

## 交付物

- [x] `result1.xlsx`、`result2.xlsx`、`result3.xlsx`、`result4-2.xlsx`、`result4-3.xlsx`（仓库根目录）
- [x] `table{1,2,3}_q*.csv`、`emergency_events_q*.csv`、`ablation_q3.csv`、`price_info_q4_*.csv`
- [x] `code/` 下完整源代码
- [x] 竞赛论文 `passage/main.tex`（含 AI 使用声明 `passage/ai_usage.tex`）

## 编译论文

```bash
cd passage && xelatex main.tex && xelatex main.tex    # 基于 ctex，需编译两遍
```

## 注意事项

- 时间标签 `0:00+1` 表示**次日 0:00**（当天 24:00）；附件 2、4 的列标签是**时段结束时刻**。
- 附件 3 的 `日期` 列仅每天首行填写，读取时需 forward-fill；预报为整点值，需插值到 10 min。
- `~$附件*.xlsx` 是 Excel 锁文件，不要读取或提交。
- 仓库暂无 `.gitignore`，提交时注意排除 `__pycache__/`、`*.aux`、`*.log`、`*.out` 等中间产物。
- 详细建模推导与实验记录见 `code/问题{2,3,4}总结.md` 与 `code/eda_report.md`；
  仓库开发约定见 [CLAUDE.md](CLAUDE.md)。
