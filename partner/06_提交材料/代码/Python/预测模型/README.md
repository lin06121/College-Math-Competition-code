# 预测模型（扩展模块）· Python 代码说明

本目录是"预测模型升级"实验的全部脚本：用**数学建模/机器学习方法**（而非简单平均）
预测**负荷（耗电）、光伏、实时电价**，并在与论文完全一致的因果口径下检验端到端费用。

## 一、结论（详见 `07_预测模型实验/报告/预测模型实验报告.md`）

| 场景 | 论文口径（基线） | 新预测模型 | 改善 |
|---|---|---|---|
| 问题二（固定电价） | 13 848 183 元 | **13 655 835 元** | −192 348 元（−1.39%） |
| 问题三（固定电价 + 滚动调整） | 14 047 116 元 | **13 890 433 元** | −156 684 元（−1.12%） |
| 问题四之二（实时电价） | 14 531 823 元 | **14 339 252 元** | −192 571 元（−1.33%） |
| 问题四之三（实时电价 + 滚动） | 14 721 611 元 | **14 581 798 元** | −139 813 元（−0.95%） |

精度：负荷 MAE 137.52 → **132.28 kW**（周五六 154.60 → 142.21、其他五天 130.73 → 128.33，
**两类同时改善**）；光伏 151.95 → **136.85 kW**；实时电价 0.0581 → **0.0367 元/kWh**。

## 二、脚本清单

| 脚本 | 功能 |
|---|---|
| `fc_common.py` | 数据装载 + **注入式因果管道**（`run_q2_fc` / `run_q3_fc`）+ 精度指标 |
| `check_baseline.py` | 管道基准复现校验（必须与论文四问逐元一致：13 848 183 / 14 047 116 / 14 531 823 / 14 721 611） |
| `models.py` | 特征工程 + 统计类模型（指数加权同类日、相似日 kNN、晴空指数、电价廓线） |
| `models_ml.py` | 岭回归 / LightGBM / 基线+GBDT 残差的**因果滚动重训**（每 14 天重训） |
| `models_extra.py` | 误差反馈、稳健中位数、递推水平、**因果堆叠**、条件分位缓冲 |
| `eval_mae.py` / `eval_mae2.py` | 预测精度对照（3 个目标 × 多模型） |
| `eval_load_cls.py` | 负荷按"周五六 / 其他五天"分类评价（含类别感知误差反馈） |
| `eval_load2.py` / `eval_load3.py` | 水平×形状分解、面板岭回归、**分类凸组合权重选择（1 月标定）** |
| `build_final_preds.py` | 汇总候选预测矩阵 → `data/final_preds.npz` |
| `eval_final.py` / `eval_final2.py` / **`eval_final3.py`** | 端到端费用评价（最终结论以 `eval_final3.py` 为准） |
| `eval_qsens.py` | 报童分位 q 的敏感性 |
| `export_for_matlab.py` | 导出 MATLAB 侧所需输入序列与基模型预测（`data/matlab_in/*.csv`） |
| `make_figs_fc.py` / `make_figs_fc2.py` | 实验图表（含哑铃图、雷达图、雨云图、ECDF、热力图、日历图、瀑布图、散点图、双轴图） |

## 三、运行方式

```powershell
$env:PYTHONPATH = "<工作区根>;<工作区根>\00_环境;<工作区根>\00_环境\.py_pkgs"
python check_baseline.py        # 1) 管道基准校验（必须"通过(4/4)"）
python eval_mae.py              # 2) 精度对照
python eval_mae2.py
python eval_load3.py            # 3) 最优负荷模型（分类凸组合）
python build_final_preds.py     # 4) 候选预测矩阵
python eval_final3.py           # 5) 端到端费用（结论）
python eval_qsens.py            # 6) q 敏感性
python export_for_matlab.py     # 7) 导出 MATLAB 输入
python make_figs_fc.py; python make_figs_fc2.py   # 8) 图表
```

依赖：`numpy / pandas / scipy / matplotlib / scikit-learn / lightgbm`（已随工作区 `00_环境\.py_pkgs` 提供）。

## 四、MATLAB 侧对应实现

`代码/MATLAB/build_fc_new.m` + `run_cumcm_c(false,'fc')` 用**纯基础 MATLAB**（无任何工具箱）
重建同一套预测模型：负荷分类组合、光伏/电价因果岭回归堆叠、报童缓冲、四问 LP 与实时平衡控制。
MATLAB 侧构造的预测矩阵与 Python 侧**逐元素一致**（比对输出：MAE 差 = 0.0000）。
因 MATLAB 无 LightGBM 等价物，两个 GBDT 基模型的预测以 CSV 序列输入（地位等同附件3 的预报）。
