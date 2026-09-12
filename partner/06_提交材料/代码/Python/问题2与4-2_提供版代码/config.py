"""
项目配置：路径、常量、储能参数、季节/月份映射
"""
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 附件路径
# 附件目录：依次尝试若干常见位置，避免交付后因目录结构不同而跑不起来
# （原实现写死 ROOT/"problem"/"C题"/"附件"；此处仅增强路径定位，不改变任何计算逻辑）
_ATT_CANDIDATES = [
    ROOT / "problem" / "C题" / "附件",
    ROOT / "附件",
    ROOT.parent / "附件",
    ROOT.parent.parent / "附件",
]
ATTACHMENT_DIR = next((p for p in _ATT_CANDIDATES if p.exists()), _ATT_CANDIDATES[0])
ATTACHMENT1 = ATTACHMENT_DIR / "附件1.xlsx"
ATTACHMENT2 = ATTACHMENT_DIR / "附件2.xlsx"
ATTACHMENT3 = ATTACHMENT_DIR / "附件3.xlsx"
ATTACHMENT4 = ATTACHMENT_DIR / "附件4.xlsx"
RESULT_TEMPLATE_DIR = ATTACHMENT_DIR / "附件5"

# 输出路径
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"

# 时间粒度
SLOTS_PER_DAY = 144
MINUTES_PER_SLOT = 10
HOURS_PER_SLOT = MINUTES_PER_SLOT / 60.0  # 1/6 h

# 储能参数
STORAGE_MAX_CAPACITY = 12000.0       # kWh
STORAGE_MAX_POWER = 5000.0           # kW
STORAGE_EFFICIENCY = 0.9
STORAGE_SOC_MIN = 1200.0             # kWh
STORAGE_SOC_MAX = 10800.0            # kWh
STORAGE_INITIAL_SOC = 6000.0         # 2025-01-01 0:00

# 季节映射（中文月份 -> 季节）
def get_season(month: int) -> str:
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    if month in (9, 10, 11):
        return "Autumn"
    return "Winter"

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
SEASON_LABELS = {"Spring": "春季", "Summer": "夏季", "Autumn": "秋季", "Winter": "冬季"}

# 关键日期
KEY_DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]

# 问题 3：附件 3 光伏预报的发布时刻及其对应时段下标
FORECAST_ISSUE_TIMES = ["0:00", "6:00", "12:00", "18:00"]
FORECAST_ISSUE_SLOTS = {"0:00": 0, "6:00": 36, "12:00": 72, "18:00": 108}

# 论文用表的时间段定义（用于后续结果输出）
PAPER_TABLE1_SLOTS = {
    "10:00-10:10": 60,
    "12:00-12:10": 72,
    "14:00-14:10": 84,
    "16:00-16:10": 96,
    "18:00-18:10": 108,
    "20:00-20:10": 120,
}

PAPER_TABLE2_RANGES = [
    ("0:00-4:00", 0, 23),
    ("4:00-8:00", 24, 47),
    ("8:00-12:00", 48, 71),
    ("12:00-16:00", 72, 95),
    ("16:00-20:00", 96, 119),
    ("20:00-24:00", 120, 143),
]
