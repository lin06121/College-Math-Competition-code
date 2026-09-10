"""
数据可视化：典型日曲线、热力图、季节/星期效应、预测误差、净负载分布等
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

from config import FIGURES_DIR, KEY_DATES, SLOTS_PER_DAY, HOURS_PER_SLOT, get_season


def _setup_fonts():
    """尝试设置中文字体，失败则使用英文标签"""
    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _slot_to_hour(slot):
    return (slot + 0.5) * 10 / 60.0


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_typical_day(att1: pd.DataFrame, out_path: Path = None):
    """典型日：电价、负载、光伏"""
    _setup_fonts()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    hours = [_slot_to_hour(s) for s in att1["slot"]]

    ax1.plot(hours, att1["load"], label="Load", color="tab:blue")
    ax1.plot(hours, att1["pv_forecast"], label="PV forecast", color="tab:green")
    ax1.set_xlabel("Hour of day")
    ax1.set_ylabel("Power (kW)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(hours, att1["price"], label="Price", color="tab:red", linestyle="--")
    ax2.set_ylabel("Price (Yuan/kWh)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("Typical Day Profile (Attachment 1)")
    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "typical_day_profile.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, out_path: Path, cmap: str = "viridis"):
    """全年热力图：x=slot, y=date"""
    _setup_fonts()
    mat = df.pivot(index="date", columns="slot", values=value_col).values
    dates = df["date"].unique()
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Slot (10 min)")
    ax.set_ylabel("Date")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    # x 轴刻度：每 3 小时 = 18 slots
    xticks = np.arange(0, SLOTS_PER_DAY + 1, 18)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{int(s*10/60)}:00" for s in xticks])
    # y 轴每月一个刻度
    yticks = []
    ylabels = []
    for i, d in enumerate(dates):
        if i == 0 or d.day == 1:
            yticks.append(i)
            ylabels.append(d.strftime("%m-%d"))
    ax.set_yticks(yticks[::1])
    ax.set_yticklabels(ylabels[::1], fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_seasonal_profiles(df: pd.DataFrame, out_path: Path = None):
    """四季平均日曲线"""
    _setup_fonts()
    profiles = df.groupby(["season", "slot"], observed=False).agg(
        load=("load_kW", "mean"),
        pv=("pv_actual_kW", "mean"),
        price=("price_realtime", "mean"),
    ).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    labels = {"Spring": "春季", "Summer": "夏季", "Autumn": "秋季", "Winter": "冬季"}
    for ax, season in zip(axes.flat, seasons):
        sub = profiles[profiles["season"] == season]
        hours = [_slot_to_hour(s) for s in sub["slot"]]
        ax.plot(hours, sub["load"], label="Load", color="tab:blue")
        ax.plot(hours, sub["pv"], label="PV", color="tab:green")
        ax2 = ax.twinx()
        ax2.plot(hours, sub["price"], label="Price", color="tab:red", linestyle="--")
        ax.set_title(labels.get(season, season))
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Power (kW)")
        ax2.set_ylabel("Price (Yuan/kWh)", color="tab:red")
        ax.set_xlim(0, 24)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "seasonal_profiles.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_weekday_effect(df: pd.DataFrame, daily: pd.DataFrame, out_path: Path = None):
    """星期效应：平均负载曲线 + 日总电量"""
    _setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：日内平均曲线
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    prof = df.groupby(["dayofweek", "slot"])["load_kW"].mean().reset_index()
    for d in range(7):
        sub = prof[prof["dayofweek"] == d]
        hours = [_slot_to_hour(s) for s in sub["slot"]]
        axes[0].plot(hours, sub["load_kW"], label=weekday_labels[d])
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Load (kW)")
    axes[0].set_title("Average Load Curve by Weekday")
    axes[0].legend()
    axes[0].set_xlim(0, 24)

    # 右：日总电量
    daily_sum = daily.groupby("dayofweek").agg(load=("load_sum_kWh", "mean"), pv=("pv_sum_kWh", "mean")).reset_index()
    x = np.arange(7)
    width = 0.35
    axes[1].bar(x - width / 2, daily_sum["load"], width, label="Load")
    axes[1].bar(x + width / 2, daily_sum["pv"], width, label="PV")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(weekday_labels)
    axes[1].set_ylabel("Daily energy (kWh)")
    axes[1].set_title("Daily Energy by Weekday")
    axes[1].legend()

    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "weekday_effect.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_forecast_errors(errors: pd.DataFrame, out_path: Path = None):
    """预测误差分布与 lead time 关系"""
    _setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：误差直方图
    err_day = errors[(errors["pv_actual_kW"] > 0) & (errors["abs_error"] < errors["abs_error"].quantile(0.99))]
    axes[0].hist(err_day["error"], bins=100, edgecolor="k", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Forecast error (kW)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of PV Forecast Error")

    # 右：lead time vs MAE
    lead_err = errors[errors["pv_actual_kW"] > 0].groupby("lead_time_hours").agg(
        mae=("abs_error", "mean"),
        rmse=("abs_error", lambda x: np.sqrt((x ** 2).mean())),
    ).reset_index()
    axes[1].plot(lead_err["lead_time_hours"], lead_err["mae"], label="MAE", marker="o")
    axes[1].plot(lead_err["lead_time_hours"], lead_err["rmse"], label="RMSE", marker="s")
    axes[1].set_xlabel("Lead time (hours)")
    axes[1].set_ylabel("Error (kW)")
    axes[1].set_title("Forecast Error vs Lead Time")
    axes[1].legend()

    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "forecast_errors.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_net_load_distribution(df: pd.DataFrame, out_path: Path = None):
    """净负载分布与日内曲线"""
    _setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：净负载直方图
    axes[0].hist(df["net_load_actual_kW"], bins=200, color="tab:blue", edgecolor="k", alpha=0.6)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Net load = Load - PV (kW)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Net Load")

    # 右：分季节日内平均净负载
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        sub = df[df["season"] == season].groupby("slot")["net_load_actual_kW"].mean()
        hours = [_slot_to_hour(s) for s in sub.index]
        axes[1].plot(hours, sub.values, label=season)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Net load (kW)")
    axes[1].set_title("Average Net Load by Season")
    axes[1].legend()
    axes[1].set_xlim(0, 24)

    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "net_load_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_key_dates(df: pd.DataFrame, out_path: Path = None):
    """关键日期曲线"""
    _setup_fonts()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for ax, date_str in zip(axes.flat, KEY_DATES):
        sub = df[df["date"] == pd.Timestamp(date_str)]
        hours = [_slot_to_hour(s) for s in sub["slot"]]
        ax.plot(hours, sub["load_kW"], label="Load", color="tab:blue")
        ax.plot(hours, sub["pv_actual_kW"], label="PV actual", color="tab:green")
        ax.plot(hours, sub["pv_forecast_op_kW"], label="PV forecast", color="tab:green", linestyle="--")
        ax2 = ax.twinx()
        ax2.plot(hours, sub["price_realtime"], label="RT price", color="tab:red", linestyle=":")
        ax.set_title(date_str)
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Power (kW)")
        ax2.set_ylabel("Price (Yuan/kWh)", color="tab:red")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, 24)
    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "key_dates_profile.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_monthly_box(daily: pd.DataFrame, out_path: Path = None):
    """月度箱线图：日负载、日光伏、日电价"""
    _setup_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sns.boxplot(data=daily, x="month", y="load_sum_kWh", ax=axes[0])
    axes[0].set_title("Daily Load Energy by Month")
    axes[0].set_ylabel("kWh")
    sns.boxplot(data=daily, x="month", y="pv_sum_kWh", ax=axes[1])
    axes[1].set_title("Daily PV Energy by Month")
    axes[1].set_ylabel("kWh")
    sns.boxplot(data=daily, x="month", y="price_realtime_mean", ax=axes[2])
    axes[2].set_title("Daily Mean RT Price by Month")
    axes[2].set_ylabel("Yuan/kWh")
    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "monthly_boxplots.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_typical_vs_rt_price(df: pd.DataFrame, out_path: Path = None):
    """典型日电价与实时电价对比"""
    _setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：典型日曲线对比
    slot_prof = df.groupby("slot").agg(typical=("price_typical", "mean"), rt=("price_realtime", "mean")).reset_index()
    hours = [_slot_to_hour(s) for s in slot_prof["slot"]]
    axes[0].plot(hours, slot_prof["typical"], label="Attachment 1 (typical day)", marker="o", markersize=3)
    axes[0].plot(hours, slot_prof["rt"], label="Real-time mean", marker="s", markersize=3)
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Price (Yuan/kWh)")
    axes[0].set_title("Typical-day vs Real-time Price Curve")
    axes[0].legend()
    axes[0].set_xlim(0, 24)

    # 右：实时电价分布
    axes[1].hist(df["price_realtime"], bins=200, color="tab:orange", edgecolor="k", alpha=0.7)
    axes[1].axvline(df["price_typical"].mean(), color="blue", linestyle="--", label="Typical-day mean")
    axes[1].axvline(df["price_realtime"].mean(), color="red", linestyle="--", label="RT mean")
    axes[1].set_xlabel("Price (Yuan/kWh)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Real-time Price")
    axes[1].legend()

    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "typical_vs_rt_price.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_clear_cloudy_days(df: pd.DataFrame, daily: pd.DataFrame, out_path: Path = None):
    """晴天/阴天样本"""
    _setup_fonts()
    # 用日发电量/月均值比例和 CV 识别
    daily["mean_pv"] = df.groupby("date")["pv_actual_kW"].mean().values
    daily["std_pv"] = df.groupby("date")["pv_actual_kW"].std().values
    daily["cv"] = np.where(daily["mean_pv"] > 0, daily["std_pv"] / daily["mean_pv"], np.nan)
    daily["relative_pv"] = daily["pv_sum_kWh"] / daily.groupby("month")["pv_sum_kWh"].transform("mean")

    # 晴天：相对发电量在前 25% 且变异系数最小的几天
    clear_candidates = daily[daily["relative_pv"] >= daily["relative_pv"].quantile(0.75)]
    clear_days = clear_candidates.nsmallest(3, "cv")
    # 阴天：相对发电量在后 25% 且变异系数最大的几天
    cloudy_candidates = daily[daily["relative_pv"] <= daily["relative_pv"].quantile(0.25)]
    cloudy_days = cloudy_candidates.nlargest(3, "cv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for _, row in clear_days.iterrows():
        sub = df[df["date"] == row["date"]]
        hours = [_slot_to_hour(s) for s in sub["slot"]]
        axes[0].plot(hours, sub["pv_actual_kW"], label=row["date"].strftime("%m-%d"))
    axes[0].set_title("Clear Days (High PV, Low CV)")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("PV (kW)")
    if not clear_days.empty:
        axes[0].legend()
    axes[0].set_xlim(0, 24)

    for _, row in cloudy_days.iterrows():
        sub = df[df["date"] == row["date"]]
        hours = [_slot_to_hour(s) for s in sub["slot"]]
        axes[1].plot(hours, sub["pv_actual_kW"], label=row["date"].strftime("%m-%d"))
    axes[1].set_title("Cloudy Days (Low PV, High CV)")
    axes[1].set_xlabel("Hour of day")
    if not cloudy_days.empty:
        axes[1].legend()
    axes[1].set_xlim(0, 24)

    plt.tight_layout()
    if out_path is None:
        out_path = FIGURES_DIR / "clear_cloudy_days.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    _ensure_dir()
    _setup_fonts()
    print("加载数据 ...")
    df = pd.read_parquet(DATA_PROCESSED / "full_timeseries.parquet")
    daily = pd.read_csv(DATA_PROCESSED / "daily_features.csv", parse_dates=["date"])
    errors = pd.read_csv(DATA_PROCESSED / "forecast_errors.csv")
    att1 = pd.read_csv(DATA_PROCESSED / "attachment1.csv")

    print("绘制典型日曲线 ...")
    plot_typical_day(att1)

    print("绘制热力图 ...")
    plot_heatmap(df, "load_kW", "Load Heatmap (kW)", FIGURES_DIR / "load_heatmap.png", cmap="YlOrRd")
    plot_heatmap(df, "price_realtime", "Real-time Price Heatmap", FIGURES_DIR / "price_heatmap.png", cmap="RdYlGn_r")
    plot_heatmap(df, "pv_actual_kW", "PV Actual Heatmap (kW)", FIGURES_DIR / "pv_heatmap.png", cmap="YlGn")

    print("绘制季节与星期效应 ...")
    plot_seasonal_profiles(df)
    plot_weekday_effect(df, daily)

    print("绘制预测误差与净负载 ...")
    plot_forecast_errors(errors)
    plot_net_load_distribution(df)

    print("绘制关键日期与月度箱线图 ...")
    plot_key_dates(df)
    plot_monthly_box(daily)
    plot_typical_vs_rt_price(df)
    plot_clear_cloudy_days(df, daily)

    print("可视化完成，图片保存到:", FIGURES_DIR)


if __name__ == "__main__":
    from config import DATA_PROCESSED
    main()
