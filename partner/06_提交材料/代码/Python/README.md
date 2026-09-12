# 代码说明（纯 Python 实现）

本目录是提交材料的**代码部分**，用 Python 完整复现论文的全部结果（5 个结果文件、全部图表与数字）。

## 一、目录结构

```
代码/
├─ Python/                    主实现（论文全部脚本）
│   ├─ engine.py              通用单日 LP 引擎：计划模式与调整模式、SOC 递推、功率平衡（允许弃光）
│   ├─ io_results.py          结果写出：按附件5 模板逐单元格填充 result1/2/3/4-2/4-3.xlsx
│   ├─ fc_upgraded.py         ★ 升级预测模型：负荷“水平×形状 + 分类凸组合”、光伏/电价因果岭回归堆叠
│   ├─ fc_features.py / fc_ml.py   预测特征工程（时段/日历/滞后/滚动/附件3 预报）与 LightGBM 因果重训
│   ├─ upgraded_core.py       升级口径核心：报童缓冲、计划/滚动/结算管道、天眼下界、无储能对照
│   ├─ rebuild_upgraded.py    ★ 正式重算：构造升级预测并写出 5 个结果文件
│   ├─ upgraded_numbers.py / upgraded_more_numbers.py / upgraded_q4_extra.py
│   │                         论文全部数字：精度、主结果、执行方式、方法对照、灵敏度、预言机分解
│   ├─ causal_core.py / rebuild_causal.py   原依据口径的因果核心与对照重算
│   ├─ q1.py                  问题一（LP 精确求解、贪心/无储能对照、结果文件与图）
│   ├─ causal_basis_all.py / causal_variants.py / causal_saa.py / causal_unibasis.py
│   │                         方法对照：计划依据、报童分位、结算方式、滚动方案；SAA 失效验证
│   ├─ sensitivity_causal.py / eta_sens.py / buffer_forms_causal.py / q4_price_sens.py
│   │                         灵敏度与安全裕度形式对照实验
│   ├─ noise_decompose.py     残差季节偏置、重尾分布、预言机信息价值分解
│   ├─ q_tune_causal.py / q2_exec_modes.py / q4_weekday.py   1 月标定、执行方式、周内充放电
│   ├─ solver_robustness.py   数值稳健性：单纯形 vs 内点法（LP 多重最优解的漂移量级）
│   ├─ extract_tables.py / show_paper_tables.py   论文表 1/2/3 数据提取（与结果文件同源）
│   ├─ audit_results.py       结果文件终检（费用合计、购电量×电价、SOC 链条、限幅）
│   ├─ check_paper_tables.py  论文表格与结果文件逐数字一致性校验
│   ├─ make_figs_upgraded.py / make_figs_causal.py / make_pattern_figs.py / pv_seasonality.py  论文插图
│   ├─ crosscheck.py / verify_bound.py / bound_check.py   双算法一致性、策略与 LP 下界合法性校验
│   ├─ 数据工程/               原始附件清洗、统一时间轴、EDA（build_dataset.py、make_figs.py）
│   ├─ 预测模型/               预测模型升级实验全套脚本（精度对照、端到端费用、多样化图表）
│   └─ 早期口径存档/           旧结算口径（事后最优再调度）的对照实验，仅作存档
└─ （无其他语言实现）
```

## 二、运行方式

```powershell
$env:PYTHONPATH = "<工作区根>;<工作区根>\00_环境;<工作区根>\00_环境\.py_pkgs"
# 1) 正式重算：写出 5 个结果文件（升级预测模型口径）
python "04_建模求解\scripts\rebuild_upgraded.py"
# 2) 论文全部数字与对照/灵敏度
python "04_建模求解\scripts\upgraded_numbers.py"
python "04_建模求解\scripts\upgraded_more_numbers.py"
python "04_建模求解\scripts\upgraded_q4_extra.py"
# 3) 论文表 1/2/3 与插图
python "04_建模求解\scripts\extract_tables.py"
python "04_建模求解\scripts\make_figs_upgraded.py"
# 4) 自检
python "04_建模求解\scripts\audit_results.py"
python "04_建模求解\scripts\check_paper_tables.py"
# 5) 预测模型升级实验（可选，完整重跑）
python "07_预测模型实验\scripts\check_baseline.py"
python "07_预测模型实验\scripts\eval_mae.py" ... （见 预测模型/README.md）
```

依赖：`numpy / pandas / scipy / matplotlib / openpyxl / pyarrow / scikit-learn / lightgbm`
（已随工作区 `00_环境\.py_pkgs` 提供）。

> 路径约定：脚本用**工作区根目录**下的相对路径解析数据
> （`02_赛题数据`、`03_数据工程`、`04_建模求解`、`05_论文与结果`、`07_预测模型实验`）。
> 在提交目录内运行请保持同样的目录结构，或修改 `io_results.py` / `causal_core.py` 中的 `ROOT`。

## 三、结果与结论摘要

| 场景 | 采用方案 | 全年费用（2.1--12.31） |
|---|---|---|
| 问题一 | 单日 LP（端点储电量相等） | 购电量 59 482.67 kWh、购电费 35 126.95 元 |
| 问题二 | 升级预测（分类组合负荷 + 岭回归堆叠光伏）+ 报童 $q^*=0.7$ + 实时平衡 | **13 655 835 元** |
| 问题三 | 同上，**不追加滚动调整**（调整无正收益） | **13 655 835 元**（全滚动 13 814 638 元） |
| 问题四之二 | 实时电价下重算问题二 | **14 339 252 元**（预测电价 14 311 077 元） |
| 问题四之三 | 实时电价下重算问题三 | **14 507 043 元** |

预测精度（2.1--12.31 外样本）：负载 MAE 137.52 → **132.28 kW**（周五六 154.60 → 142.21、
其余五天 130.73 → 128.33）；光伏 151.95 → **136.85 kW**；实时电价 0.0581 → **0.0367 元/kWh**。

自检结果：5 个结果文件的计划购电费与论文逐元一致；计划购电量×电价、SOC 链条、限幅检查 **0 违规**；
论文表 1/2/3 与结果文件**逐数字一致**。
