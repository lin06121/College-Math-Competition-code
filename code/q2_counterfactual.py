"""
问题二反事实对照表：执行机制 / 计划依据 / 计划方法 / 储能。
全部在同一"0:00 计划 + 日内结算"框架下（2025-01-01 起预热储电量链条，2-01 起统计）。

输出 data/processed/q2_counterfactual.csv，供绘图与论文第 6 节使用。

用法：PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/q2_counterfactual.py
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED                      # noqa: E402
from compare_q2 import run_year                        # noqa: E402

# (分组, 图例标签, run_year 参数)
ROWS = [
    ("执行机制", "僵硬执行（无日内平衡）",
     dict(forecast_kind="mate", plan_kind="nv", q=0.8, settle="stiff")),
    ("执行机制", "因果实时平衡（采用）",
     dict(forecast_kind="mate", plan_kind="nv", q=0.8, settle="causal")),
    ("执行机制", "事后最优再调度（下界）",
     dict(forecast_kind="mate", plan_kind="nv", q=0.8, settle="recour")),

    ("计划依据", "同类日负荷+近4日光伏（采用）",
     dict(forecast_kind="mate", plan_kind="nv", q=0.8, settle="causal")),
    ("计划依据", "同星期几",
     dict(forecast_kind="ours", plan_kind="nv", q=0.8, settle="causal")),
    ("计划依据", "昨日外推",
     dict(forecast_kind="persist", plan_kind="nv", q=0.8, settle="causal")),
    ("计划依据", "附件1 典型日",
     dict(forecast_kind="typical", plan_kind="nv", q=0.8, settle="causal")),

    ("计划方法", "无裕度（点预测）",
     dict(forecast_kind="mate", plan_kind="point", settle="causal")),
    ("计划方法", "两阶段随机规划 SAA",
     dict(forecast_kind="mate", plan_kind="saa", settle="causal")),

    ("储能", "无储能",
     dict(forecast_kind="mate", plan_kind="nv", q=0.8, settle="stiff", storage=False)),
]


def main():
    rows = []
    for group, label, kw in ROWS:
        t0 = time.time()
        r = run_year(**kw)
        rows.append(dict(group=group, label=label,
                         plan_cost=r["plan_cost"], em_cost=r["em_cost"],
                         total_cost=r["total_cost"], em_kWh=r["em_kWh"],
                         plan_kWh=r["plan_kWh"], curt_kWh=r["curt_kWh"]))
        print(f"[{group}] {label:<24} 计划 {r['plan_cost']:>12,.0f}  "
              f"紧急 {r['em_cost']:>11,.0f}  总 {r['total_cost']:>13,.0f} 元  "
              f"紧急电量 {r['em_kWh']:>9,.0f} kWh   ({time.time() - t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(DATA_PROCESSED / "q2_counterfactual.csv", index=False, encoding="utf-8-sig")
    print("\n-> data/processed/q2_counterfactual.csv")


if __name__ == "__main__":
    main()
