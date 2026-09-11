"""
问题 1 求解器：调用通用单日求解器，固定 E(0)=E(144)=6000。
"""
import pandas as pd
from pathlib import Path

from config import DATA_PROCESSED, STORAGE_INITIAL_SOC
from daily_solver import build_and_solve, verify_solution


def solve_problem1(attachment1_path: Path = None, verbose: bool = False):
    """读取附件 1 并求解问题 1。"""
    if attachment1_path is None:
        attachment1_path = DATA_PROCESSED / "attachment1.csv"
    df = pd.read_csv(attachment1_path)
    df = df.sort_values("slot").reset_index(drop=True)

    price = df["price"].values
    load = df["load"].values
    pv = df["pv_forecast"].values
    time_str = df["time_str"].values

    sol = build_and_solve(
        price, load, pv,
        e0=STORAGE_INITIAL_SOC,
        cyclic=True,
        allow_emergency=False,
        verbose=verbose,
    )

    # 为了与问题 1 原有接口保持一致，做别名映射
    sol["P_grid_kW"] = sol["P_plan_kW"]
    sol["E_grid_kWh"] = sol["E_plan_kWh"]
    sol["total_grid_kWh"] = float(sol["E_plan_kWh"].sum())
    sol["total_cost"] = sol["total_plan_cost"]

    sol["time_str"] = time_str
    sol["slot"] = df["slot"].values
    sol["price"] = price
    sol["load"] = load
    sol["pv"] = pv
    return sol


if __name__ == "__main__":
    sol = solve_problem1(verbose=True)
    print("\n=== 问题 1 求解结果校验 ===")
    verify_solution(sol, sol["price"], sol["load"], sol["pv"],
                    e0=STORAGE_INITIAL_SOC, e_end_expected=STORAGE_INITIAL_SOC)
