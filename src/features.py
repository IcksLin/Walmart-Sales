"""特征工程：时间特征、门店内滞后/滚动特征、设计矩阵（含门店固定效应）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """派生年/月/周号/季度等时间特征。"""
    out = df.copy()
    date = out[date_col]
    out["year"] = date.dt.year
    out["month"] = date.dt.month
    out["quarter"] = date.dt.quarter
    out["week"] = date.dt.isocalendar().week.astype(int)
    out["dayofyear"] = date.dt.dayofyear
    return out


def history_feature_names(lags: list[int], windows: list[int]) -> list[str]:
    names = [f"lag{lag}" for lag in lags]
    for w in windows:
        names += [f"roll_mean{w}", f"roll_std{w}"]
    return names


def add_history_features(
    df: pd.DataFrame,
    target: str,
    store_col: str,
    date_col: str,
    lags: list[int],
    windows: list[int],
    log_target: bool = True,
) -> pd.DataFrame:
    """按门店构造仅使用过去信息的目标滞后/滚动特征，避免未来泄漏。

    滞后与滚动均基于 shift，当前期永远不进入自身特征。
    """
    out = df.sort_values([store_col, date_col]).reset_index(drop=True).copy()
    y = out[target].to_numpy(dtype=float)
    if log_target:
        y = np.log1p(y)

    for name in history_feature_names(lags, windows):
        out[name] = np.nan

    for _, idx in out.groupby(store_col).indices.items():
        idx = np.sort(idx)
        series = pd.Series(y[idx])
        for lag in lags:
            values = np.full(len(idx), np.nan)
            if lag < len(idx):
                values[lag:] = y[idx[:-lag]]
            out.loc[idx, f"lag{lag}"] = values
        for w in windows:
            shifted = series.shift(1)
            out.loc[idx, f"roll_mean{w}"] = shifted.rolling(w).mean().to_numpy()
            out.loc[idx, f"roll_std{w}"] = shifted.rolling(w).std().to_numpy()
    return out


def build_design(
    df: pd.DataFrame,
    exogenous: list[str],
    time_features: list[str],
    history_features: list[str],
    store_col: str,
    include_store: bool,
) -> pd.DataFrame:
    """组装特征矩阵。

    include_store=True 时对门店做 one-hot（门店固定效应）；
    单店建模时设 False（门店恒定，无信息）。
    """
    blocks = [df[exogenous].astype(float)]
    if time_features:
        blocks.append(df[time_features].astype(float))
    if history_features:
        blocks.append(df[history_features].astype(float))
    design = pd.concat(blocks, axis=1)
    if include_store:
        dummies = pd.get_dummies(df[store_col], prefix=store_col).astype(float)
        design = pd.concat([design, dummies], axis=1)
    return design.reset_index(drop=True)
