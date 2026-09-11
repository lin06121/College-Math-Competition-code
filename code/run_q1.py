"""
问题 1 一键运行脚本：
    1) 读取附件 1 典型日数据
    2) 建立并求解 MILP
    3) 校验约束
    4) 生成论文表 1、表 2 与 result1.xlsx

使用方式：
    PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/run_q1.py
"""
from pathlib import Path

from solver_q1 import solve_problem1, verify_solution
from format_result_q1 import format_all, print_paper_tables
from visualize import plot_q1_solution


def main():
    print("=" * 60)
    print("问题 1：典型日确定性储能经济调度")
    print("=" * 60)

    print("\n[1/4] 读取附件 1 并建立模型 ...")
    sol = solve_problem1(verbose=True)

    print("\n[2/4] 校验最优解 ...")
    ok = verify_solution(
        sol,
        sol["price"],
        sol["load"],
        sol["pv"],
    )
    if not ok:
        raise RuntimeError("最优解校验未通过，请检查模型")

    print("\n[3/4] 生成论文表格与 result1.xlsx ...")
    res = format_all(sol)

    print("\n[4/4] 论文用表 ...")
    print_paper_tables(sol)

    print("\n[附加] 生成问题 1 结果曲线 ...")
    plot_q1_solution(sol)

    print("\n" + "=" * 60)
    print("完成。输出文件：")
    print(f"  result1.xlsx : {res['result_xlsx'].resolve()}")
    print(f"  table1_q1.csv: {res['result_xlsx'].parent / 'table1_q1.csv'}")
    print(f"  table2_q1.csv: {res['result_xlsx'].parent / 'table2_q1.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
