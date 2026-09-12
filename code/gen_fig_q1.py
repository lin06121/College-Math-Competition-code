# -*- coding: utf-8 -*-
"""生成论文图 6（问题一典型日最优调度，四面板）。

绘图函数取自同伴的 q1.py（plot_q1_schedule）；由于同伴数据根目录
(D:\\Study\\国赛\\国赛) 本机不存在、其 q1.py 无法直接运行，这里用我方
solver_q1 复算同一 LP（数值与论文一致：59 482.7 kWh / 35 126.95 元），
再调用同一绘图函数，输出到 passage/figures/fig_q1_schedule.png。

用法：
    PYTHONUTF8=1 D:/devtool/anaconda/conda/python.exe code/gen_fig_q1.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "partner" / "06_提交材料" / "06_提交材料" / "代码" / "Python"))

from solver_q1 import solve_problem1          # noqa: E402
from q1 import plot_q1_schedule               # noqa: E402


def main():
    sol = solve_problem1()
    out = ROOT / "passage" / "figures" / "fig_q1_schedule.png"
    png = plot_q1_schedule(
        price=sol["price"],
        load=sol["load"],          # kW
        pv=sol["pv"],              # kW
        P=sol["P_plan_kW"],        # kW
        C=sol["P_ch_kW"],          # kW
        D=sol["P_dis_kW"],         # kW
        E=sol["E_kWh"],            # kWh (145)
        out_path=out,
        merge_soc=True,            # 三面板：SOC 以右轴并入充放电面板
    )
    print(f"购电量 {sol['E_plan_kWh'].sum():.2f} kWh, 购电费 {sol['total_plan_cost']:.2f} 元")
    print("图已生成 ->", png)


if __name__ == "__main__":
    main()
