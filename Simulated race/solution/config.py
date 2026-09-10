# -*- coding: utf-8 -*-
"""
config.py — 全局配置
=====================
集中管理路径、随机种子、以及跨问题共享的常量。
所有脚本从这里导入 RANDOM_SEED 与路径，确保可复现。

依赖：Python 3.9+、pandas、numpy、scipy、scikit-learn、xgboost、
      matplotlib、seaborn、openpyxl。

作者说明：输入数据位于附件压缩包解压目录，路径可据实际情况修改
INPUT_XLSX。输出统一写入 DATA_DIR。
"""

from pathlib import Path

# ------------------------------------------------------------------
# 目录结构
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # solution/ 目录
SRC_DIR = BASE_DIR / "src"                          # 各问题脚本
UTILS_DIR = BASE_DIR / "utils"                      # 通用工具
DATA_DIR = BASE_DIR / "data"                        # 输出 CSV / 图片
FIG_DIR = DATA_DIR / "figures"                      # 可视化图片
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 输入数据（参考附件路径，需按实际解压位置调整）
# ------------------------------------------------------------------
_INPUT_REL = (
    "../Simulated race/2026策联杯C题/C题/"
    "2026年度“策联杯”数学建模精英联赛-C题-附件/"
    "2026年度“策联杯”数学建模精英联赛-C题-附件/input_data/C题_数据附件.xlsx"
    
)
INPUT_XLSX = Path(_INPUT_REL)

# ------------------------------------------------------------------
# 复现性：全局随机种子（题面严格要求固定为 42）
# ------------------------------------------------------------------
RANDOM_SEED = 42

# ------------------------------------------------------------------
# 问题 1：转播观看人数预测
# ------------------------------------------------------------------
# 目标变量单位约定（详见 README.md 的“单位约定”小节）：
#   historical_matches.tv_viewers 单位为“人”，量级约 1e8。
#   模型以 (tv_viewers / 1e6) 为训练目标，即输出单位为“百万人”。
#   - result_1_test_prediction.csv  -> 百万人（浮点）
#   - result_1_match_prediction.csv -> 人（整数，= 百万人 * 1_000_000）
TV_VIEWER_SCALE = 1e6

# 历史表现滚动窗口长度（过去 N 场比赛）
ROLLING_WINDOW = 5

# 交叉验证折数 / GridSearchCV
CV_FOLDS = 5

# ------------------------------------------------------------------
# 问题 2：场馆与时段协同组合优化（遗传算法）
# ------------------------------------------------------------------
GA_POPULATION = 64           # 种群规模
GA_GENERATIONS = 150         # 迭代代数
GA_TOURNAMENT = 3            # 锦标赛选择规模
GA_CROSSOVER_PROB = 0.55     # 交叉概率
GA_MUTATION_PROB = 0.35      # 变异概率（单个基因）
GA_ELITISM = 2               # 保留精英个数

REST_MIN_HOURS = 60.0        # 相邻两轮最小休息时间（小时）

# 黄金时段阈值：global_prime_score 排名前该比例视为“黄金时段”
GOLDEN_PRIME_TOP_FRACTION = 0.25
# 大容量场馆阈值：capacity 排名前该比例视为“大容量场馆”
LARGE_CAPACITY_TOP_FRACTION = 0.25

# 每个场馆的总场次约束（已含于 venues 表，此处用作 fallback）
VENUE_MIN_TOTAL = 2
VENUE_MAX_TOTAL = 8

# 单位安保成本（10 万美元 = 0.1 百万美元）
SECURITY_UNIT_COST_MUSD = 0.1

# 三组赛各自允许的日期窗口（用于初始化与休息约束）。日期列为字符串，
# 为避免依赖具体日期，这里以第几天(1..20, 对应 06-11 ~ 06-30)表示区间。
ROUND_DAY_WINDOWS = {
    1: (1, 6),    # 第 1 轮：06-11 ~ 06-16
    2: (8, 13),   # 第 2 轮：06-18 ~ 06-23（第 7 天作缓冲）
    3: (15, 20),  # 第 3 轮：06-25 ~ 06-30（第 14 天作缓冲）
}

# 目标指标 Z2 权重
Z2_WEIGHTS = {
    "T": 0.25, "B": 0.25, "U": 0.15, "H": 0.10,
    "C": 0.08, "D": 0.07, "F": 0.06, "R": 0.04,
}

# ------------------------------------------------------------------
# 问题 3：动态资源优化
# ------------------------------------------------------------------
MC_N_SIMS = 20_000          # 蒙特卡洛模拟次数
POISSON_LAMBDA_MIN = 0.15
POISSON_LAMBDA_MAX = 4.50

FEEDBACK_N_MIN, FEEDBACK_N_MAX = 0.6, 1.5    # 现场反馈 f_N 钳制
FEEDBACK_V_MIN, FEEDBACK_V_MAX = 0.75, 1.25  # 转播反馈 f_V 钳制

# 票价调整比例 δ 的离散候选（受 max_ticket_discount/increase 0.15/0.20 约束）
TICKET_DELTA_CANDIDATES = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20)

# Z3 目标权重
Z3_WEIGHTS = {"TV": 0.35, "BV": 0.35, "A": 0.10, "C": 0.10, "R": 0.10}