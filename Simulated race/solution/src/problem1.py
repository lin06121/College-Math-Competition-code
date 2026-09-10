# -*- coding: utf-8 -*-
"""
src/problem1.py — 任务一：转播观看人数预测模型
================================================
流程
----
1. 加载 historical_matches，按 dataset_split 切分训练 / 测试。
2. 赛前特征工程（严格防泄漏，仅用 teams / base_predictions 的先验 + 历史滚动）：
   - 球队静态属性（elo、身价、球迷基础、球星指数、实力分等）；
   - 交叉特征（实力差、球迷和/差、球星和、实力和等）；
   - 悬念/差距：历史赛由赔率隐含概率算熵；小组赛由 base_predictions 先验概率算熵；
   - 时间（月份、是否周末）与赛事（competition/stage 独热）；
   - 过去 5 场滚动战绩（shift，杜绝未来信息）。
3. XGBoost 回归 + GridSearchCV 调参，测试集评价 MSE。
4. 输出：
   - result_1_test_prediction.csv（百万人，浮点）
   - result_1_match_prediction.csv（人，整数 = 百万人 × 1_000_000）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 `python src/problem1.py` 运行：把 solution/ 加入模块搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import (DATA_DIR, RANDOM_SEED, TV_VIEWER_SCALE, CV_FOLDS,
                    ROLLING_WINDOW)
from utils.data_loader import load_all_sheets
from utils.features import (attach_team_features, add_cross_features,
                            entropy_uncertainty, implied_probabilities,
                            rolling_team_form, latest_team_form)

# ------------------------------------------------------------------
# 全局随机种子（复现性）
# ------------------------------------------------------------------
_rng = np.random.default_rng(RANDOM_SEED)

# 建模使用的数值特征列（最终进入模型的输入矩阵）
MODEL_FEATURES = [
    # 球队静态
    "a_elo_rating", "b_elo_rating", "a_market_value_musd", "b_market_value_musd",
    "a_fan_base_index", "b_fan_base_index", "a_star_index", "b_star_index",
    "a_strength_score", "b_strength_score", "a_strength_rank", "b_strength_rank",
    "a_avg_age", "b_avg_age", "a_style_attack", "b_style_attack",
    "a_style_defense", "b_style_defense", "a_host_flag", "b_host_flag",
    # 交叉
    "elo_diff", "market_sum", "market_diff", "fan_sum", "fan_abs_diff",
    "star_sum", "strength_diff", "strength_sum", "rank_diff", "same_confed",
    # 悬念 / 差距 / 时间
    "uncertainty", "lopsidedness", "month", "is_weekend", "neutral",
    # 滚动历史战绩
    "a_form_avg_goals_for", "a_form_avg_goals_against", "a_form_win_rate",
    "a_form_avg_points",
    "b_form_avg_goals_for", "b_form_avg_goals_against", "b_form_win_rate",
    "b_form_avg_points",
]


# ------------------------------------------------------------------
# 特征工程主流程
# ------------------------------------------------------------------
def build_match_features(historical: pd.DataFrame, teams: pd.DataFrame,
                         groups_matches: pd.DataFrame,
                         base_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造历史(含 train/test)与 72 场小组赛的特征表。

    返回
    ----
    hist_feat : DataFrame，含 match_id、dataset_split、tv_viewers(million) 及特征列
    grp_feat  : DataFrame，含 match_id、group_id、round、team_a/b 及特征列
    """
    # ---- 1. 历史赛特征 ----
    hist = historical.copy()
    hist = attach_team_features(hist, teams, "team_a")
    hist = attach_team_features(hist, teams, "team_b")
    hist = add_cross_features(hist, prefix="")

    # 赔率隐含概率 -> 悬念熵 + 强弱差距
    p_win, p_draw, p_loss = implied_probabilities(
        hist["odds_a"], hist["odds_draw"], hist["odds_b"])
    hist["uncertainty"] = entropy_uncertainty(p_win, p_draw, p_loss)
    hist["lopsidedness"] = np.abs(p_win - p_loss)

    # 时间与赛事
    hist["month"] = hist["date"].dt.month.astype(float)
    hist["weekday"] = hist["date"].dt.weekday
    hist["is_weekend"] = (hist["weekday"] >= 5).astype(float)
    hist["neutral"] = hist["neutral"].astype(float)
    hist["same_confed"] = (
        hist["a_confederation"] == hist["b_confederation"]).astype(float)

    # 滚动战绩（过去 5 场，严格 shift）
    hist = rolling_team_form(hist, window=ROLLING_WINDOW)

    hist["tv_viewers_million"] = hist["tv_viewers"].astype(float) / TV_VIEWER_SCALE
    hist_feat = hist.reset_index(drop=True)

    # ---- 2. 小组赛特征（72 场） ----
    grp = groups_matches.copy()
    grp = attach_team_features(grp, teams, "team_a")
    grp = attach_team_features(grp, teams, "team_b")

    # 合并 base_predictions 的先验概率与指数
    bp = base_predictions[[
        "match_id", "uncertainty_index", "attractiveness_index",
        "expected_attendance_base", "commercial_value_index",
        "p_a_win", "p_draw", "p_b_win",
    ]].copy()
    grp = grp.merge(bp, on="match_id", how="left")

    # 先验概率 -> 悬念熵 + 强弱差距（与历史侧同一公式体例）
    grp["uncertainty"] = entropy_uncertainty(
        grp["p_a_win"].astype(float), grp["p_draw"].astype(float),
        grp["p_b_win"].astype(float))
    grp["lopsidedness"] = np.abs(
        grp["p_a_win"].astype(float) - grp["p_b_win"].astype(float))

    grp = add_cross_features(grp, prefix="")

    # 时间与赛事（小组赛均为 2026-06，赛事 = 世界杯小组赛）
    grp["month"] = 6.0
    grp["is_weekend"] = 0.0
    grp["neutral"] = 1.0          # 世界杯在主办国承办，均为中立场地
    grp["same_confed"] = (
        grp["a_confederation"] == grp["b_confederation"]).astype(float)

    # 滚动战绩：取每队截至全部历史数据的最近 5 场
    latest = latest_team_form(historical, window=ROLLING_WINDOW)
    latest_a = latest.rename(columns={
        "team": "team_a", "avg_goals_for": "a_form_avg_goals_for",
        "avg_goals_against": "a_form_avg_goals_against",
        "win_rate": "a_form_win_rate"})
    latest_b = latest.rename(columns={
        "team": "team_b", "avg_goals_for": "b_form_avg_goals_for",
        "avg_goals_against": "b_form_avg_goals_against",
        "win_rate": "b_form_win_rate"})
    grp = grp.merge(latest_a, on="team_a", how="left")
    grp = grp.merge(latest_b, on="team_b", how="left")
    grp["a_form_avg_points"] = grp["a_form_win_rate"] * 3.0
    grp["b_form_avg_points"] = grp["b_form_win_rate"] * 3.0

    grp_feat = grp.reset_index(drop=True)
    return hist_feat, grp_feat


# ------------------------------------------------------------------
# 类别独热编码
# ------------------------------------------------------------------
def one_hot_categorical(hist_feat: pd.DataFrame, grp_feat: pd.DataFrame,
                        competitions, stages) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对 competition / stage 做联合独热编码（类别集合取自历史+小组赛并集）。"""
    hist_feat = hist_feat.copy()
    grp_feat = grp_feat.copy()
    grp_feat["competition"] = "World Cup"
    grp_feat["stage"] = "Group"

    for col, cats in (("competition", competitions), ("stage", stages)):
        for cat in cats:
            hist_feat[f"{col}_{cat}"] = (hist_feat[col] == cat).astype(float)
            grp_feat[f"{col}_{cat}"] = (grp_feat[col] == cat).astype(float)

    comp_cols = [f"competition_{c}" for c in competitions]
    stage_cols = [f"stage_{s}" for s in stages]
    return hist_feat, grp_feat, comp_cols + stage_cols


# ------------------------------------------------------------------
# 建模入口
# ------------------------------------------------------------------
def run(verbose: bool = True) -> dict:
    sheets = load_all_sheets()
    historical = sheets["historical_matches"]
    teams = sheets["teams"]
    groups_matches = sheets["groups_matches"]
    base_predictions = sheets["base_predictions"]

    # 类别并集
    competitions = sorted(historical["competition"].dropna().unique()) \
        + ["World Cup"]
    competitions = sorted(set(competitions))
    stages = sorted(historical["stage"].dropna().unique()) + ["Group"]
    stages = sorted(set(stages))

    # 特征
    hist_feat, grp_feat = build_match_features(
        historical, teams, groups_matches, base_predictions)
    hist_feat, grp_feat, oe_cols = one_hot_categorical(
        hist_feat, grp_feat, competitions, stages)

    feat_cols = MODEL_FEATURES + oe_cols
    if verbose:
        print(f"[P1] 特征数 = {len(feat_cols)}")

    # 训练 / 测试切分（依据 dataset_split）
    train = hist_feat[hist_feat["dataset_split"] == "train"].reset_index(drop=True)
    test = hist_feat[hist_feat["dataset_split"] == "test"].reset_index(drop=True)
    X_train = train[feat_cols].fillna(0.0)
    y_train = train["tv_viewers_million"].values
    X_test = test[feat_cols].fillna(0.0)
    X_grp = grp_feat[feat_cols].fillna(0.0)

    # ---- XGBoost 回归 + GridSearchCV ----
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.10],
    }
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_SEED,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
    )
    gs = GridSearchCV(base_model, param_grid, scoring="neg_mean_squared_error",
                      cv=CV_FOLDS, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    if verbose:
        print(f"[P1] 最优参数 = {gs.best_params_}")

    # 测试集预测；测试集标签被隐藏（tv_viewers 为空），
    # 故用 GridSearchCV 训练折的交叉验证 MSE 作为模型精度评价。
    pred_test_million = model.predict(X_test)
    cv_mse = float(-gs.best_score_)          # scoring = neg_mean_squared_error
    rmse = float(np.sqrt(cv_mse))
    if verbose:
        print(f"[P1] 交叉验证 MSE = {cv_mse:.4f} (百万^2)  "
              f"RMSE = {rmse:.4f} (百万)，即 {rmse * 1e6:.0f} 人")

    # 72 场小组赛预测
    pred_grp_million = model.predict(X_grp)

    # ---- 输出 ----
    # 1) 测试集：百万人（浮点）
    out_test = pd.DataFrame({
        "match_id_test": test["match_id"],
        "predicted_test_tv_viewers": np.round(pred_test_million, 6),
    })
    out_test.to_csv(DATA_DIR / "result_1_test_prediction.csv", index=False)

    # 2) 小组赛：人（整数，= 百万人 × 1_000_000）
    out_match = pd.DataFrame({
        "match_id": grp_feat["match_id"],
        "team_a": grp_feat["team_a"],
        "team_b": grp_feat["team_b"],
        "predicted_tv_viewers": np.round(pred_grp_million * TV_VIEWER_SCALE).astype(int),
    })
    out_match.to_csv(DATA_DIR / "result_1_match_prediction.csv", index=False)

    if verbose:
        print("[P1] 已写出 result_1_test_prediction.csv / "
              "result_1_match_prediction.csv")

    return {
        "mse": cv_mse, "rmse": rmse,
        "best_params": gs.best_params_,
        "pred_grp_million": pred_grp_million,   # 供问题2直接复用
        "grp_feat": grp_feat,
        "model": model, "feat_cols": feat_cols,
    }


if __name__ == "__main__":
    run(verbose=True)