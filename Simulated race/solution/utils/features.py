# -*- coding: utf-8 -*-
"""
utils/features.py — 问题 1 特征工程（赛前特征，严格防泄漏）
============================================================
只使用赛前可用信息：teams 表静态属性、base_predictions 的先验概率、
历史比赛的日期 / 赔率，以及“过去 N 场”的滚动战绩（严格 shift，杜绝未来信息）。

单位约定：historical_matches.tv_viewers 单位为“人”，量级约 1e8。
本模块输出的目标 tv_viewers_million = tv_viewers / 1e6。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ROLLING_WINDOW


# ------------------------------------------------------------------
# 信息熵型“悬念指数”：给定三路概率 -> 归一化熵 [0,1]
# ------------------------------------------------------------------
def entropy_uncertainty(p_win: np.ndarray, p_draw: np.ndarray,
                        p_loss: np.ndarray) -> np.ndarray:
    """由 (p 胜, p 平, p 负) 计算归一化熵，作为悬念/不确定性代理。

    熵最大（三路等概率）时取 1，最小（某一路确定）时取 0。
    """
    p = np.vstack([p_win, p_draw, p_loss]).T            # (n, 3)
    p = np.clip(p, 1e-9, 1.0)                           # 避免 log(0)
    p = p / p.sum(axis=1, keepdims=True)                # 归一化
    ent = -(p * np.log(p)).sum(axis=1)
    return ent / np.log(3.0)                            # 按三路等概率熵归一


def implied_probabilities(odds_a: pd.Series, odds_draw: pd.Series,
                          odds_b: pd.Series) -> tuple[np.ndarray, ...]:
    """由博彩赔率计算隐含概率（1/赔率 归一化）。返回 (p_win, p_draw, p_loss)。"""
    inv_a = 1.0 / odds_a.values.astype(float)
    inv_d = 1.0 / odds_draw.values.astype(float)
    inv_b = 1.0 / odds_b.values.astype(float)
    s = inv_a + inv_d + inv_b
    return inv_a / s, inv_d / s, inv_b / s


# ------------------------------------------------------------------
# 球队静态属性关联
# ------------------------------------------------------------------
_TEAM_FEATURES = [
    "elo_rating", "market_value_musd", "fan_base_index", "star_index",
    "strength_score", "strength_rank", "avg_age", "style_attack",
    "style_defense", "host_flag", "confederation",
]


def attach_team_features(df: pd.DataFrame, teams: pd.DataFrame,
                         team_col: str = "team_a") -> pd.DataFrame:
    """按球队名关联 teams 表静态属性，特征加前缀 a_/b_ 或单项合并。

    返回在原 df 基础上新增 team 侧特征（前缀为 team_col -> a_/b_）。
    具体以调用方传参决定：车前缀固定为 a_ 或 b_。
    """
    t = teams[["team_name"] + _TEAM_FEATURES].rename(
        columns={c: f"{'a' if team_col == 'team_a' else 'b'}_{c}"
                 for c in _TEAM_FEATURES}
    )
    out = df.merge(t, left_on=team_col, right_on="team_name", how="left")
    out = out.drop(columns=["team_name"])
    return out


# ------------------------------------------------------------------
# 滚动历史战绩（过去 N 场，严格 shift 防泄漏）
# ------------------------------------------------------------------
def rolling_team_form(historical: pd.DataFrame,
                      window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """为 historical_matches 逐场计算“截至赛前”过去 window 场的球队形态。

    返回与 historical 等长的 DataFrame，新增 a/b 两侧特征：
      {side}_form_avg_goals_for / avg_goals_against / win_rate / points
    采用 groupby(team)+shift(1) 保证不含当前场次，从根本上避免未来信息泄露。
    """
    hist = historical.copy()
    hist = hist.sort_values("date").reset_index(drop=True)

    # 长表：每场比赛拆成两行（本队视角），逐队做滚动统计
    cols_a = {"team_a": "team", "team_b": "opp", "goals_a": "gf",
              "goals_b": "ga", "date": "date", "match_id": "match_id"}
    cols_b = {"team_b": "team", "team_a": "opp", "goals_b": "gf",
              "goals_a": "ga", "date": "date", "match_id": "match_id"}
    long_df = pd.concat([
        hist[list(cols_a)].rename(columns=cols_a),
        hist[list(cols_b)].rename(columns=cols_b),
    ], ignore_index=True)

    # 胜平负积分（本队视角）
    long_df["points"] = np.select(
        [long_df["gf"] > long_df["ga"], long_df["gf"] == long_df["ga"]],
        [3, 1], default=0,
    )
    long_df["win"] = (long_df["gf"] > long_df["ga"]).astype(int)

    # 逐队 shift 后滚动，避免使用当前场
    feats = []
    for team, g in long_df.groupby("team"):
        g = g.sort_values("date").reset_index(drop=True)
        g["gf_roll"] = g["gf"].shift(1).rolling(window, min_periods=1).mean()
        g["ga_roll"] = g["ga"].shift(1).rolling(window, min_periods=1).mean()
        g["win_roll"] = g["win"].shift(1).rolling(window, min_periods=1).mean()
        g["pts_roll"] = g["points"].shift(1).rolling(window, min_periods=1).mean()
        feats.append(g[["match_id", "team", "gf_roll", "ga_roll",
                        "win_roll", "pts_roll"]])
    long_feats = pd.concat(feats, ignore_index=True)

    # 合并回 a / b 两侧
    def merge_side(side):
        prefix = "a" if side == "team_a" else "b"
        s = long_feats.rename(columns={
            "team": side,  # 球队名作为合并键
            "gf_roll": f"{prefix}_form_avg_goals_for",
            "ga_roll": f"{prefix}_form_avg_goals_against",
            "win_roll": f"{prefix}_form_win_rate",
            "pts_roll": f"{prefix}_form_avg_points",
        })
        return s

    out = hist.merge(merge_side("team_a"), on=["match_id", "team_a"], how="left")
    out = out.merge(merge_side("team_b"), on=["match_id", "team_b"], how="left")
    return out


def latest_team_form(historical: pd.DataFrame,
                     window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """返回每支球队“截至全部历史数据”最近 window 场的形态（用于赛程预测）。

    供 problem1 预测 72 场小组赛时使用——小组赛发生在 2026 年，
    全部历史比赛均视为“过去”，故直接取每队最近 window 场均值即可。
    """
    hist = historical.sort_values("date").reset_index(drop=True)
    cols_a = {"team_a": "team", "team_b": "opp", "goals_a": "gf",
              "goals_b": "ga", "date": "date"}
    cols_b = {"team_b": "team", "team_a": "opp", "goals_b": "gf",
              "goals_a": "ga", "date": "date"}
    long_df = pd.concat([
        hist[list(cols_a)].rename(columns=cols_a),
        hist[list(cols_b)].rename(columns=cols_b),
    ], ignore_index=True)
    long_df["win"] = (long_df["gf"] > long_df["ga"]).astype(int)

    latest = []
    for team, g in long_df.groupby("team"):
        g = g.sort_values("date")
        latest.append({
            "team": team,
            "avg_goals_for": g["gf"].tail(window).mean(),
            "avg_goals_against": g["ga"].tail(window).mean(),
            "win_rate": g["win"].tail(window).mean(),
        })
    return pd.DataFrame(latest)


# ------------------------------------------------------------------
# 交叉特征
# ------------------------------------------------------------------
def add_cross_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """由 a_/b_ 两侧属性构造交叉特征。

    prefix 可选，用于区分历史(有赔率) 与小组赛(有先验概率) 两套特征命名。
    """
    out = df.copy()
    for side in ("a", "b"):
        other = "b" if side == "a" else "a"
        out[f"{prefix}elo_diff"] = out[f"a_elo_rating"] - out[f"b_elo_rating"]
        out[f"{prefix}market_sum"] = (out[f"a_market_value_musd"]
                                      + out[f"b_market_value_musd"])
        out[f"{prefix}market_diff"] = (out[f"a_market_value_musd"]
                                       - out[f"b_market_value_musd"])
        out[f"{prefix}fan_sum"] = (out[f"a_fan_base_index"]
                                   + out[f"b_fan_base_index"])
        out[f"{prefix}fan_abs_diff"] = (
            out[f"a_fan_base_index"] - out[f"b_fan_base_index"]).abs()
        out[f"{prefix}star_sum"] = out[f"a_star_index"] + out[f"b_star_index"]
        out[f"{prefix}strength_diff"] = (out[f"a_strength_score"]
                                         - out[f"b_strength_score"])
        out[f"{prefix}strength_sum"] = (out[f"a_strength_score"]
                                        + out[f"b_strength_score"])
        out[f"{prefix}rank_diff"] = (out[f"a_strength_rank"]
                                     - out[f"b_strength_rank"])
        break  # 交叉特征只需算一次
    return out