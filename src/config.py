"""配置加载、缺省合并与快照落盘。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "threads": 1,
    "runs_dir": "runs",
    "data": {
        "path": "practice_1.csv",
        "date_col": "Date",
        "date_format": "%d-%m-%Y",
        "target": "Weekly_Sales",
        "store_col": "Store",
        "holiday_col": "Holiday_Flag",
    },
    "features": {
        "exogenous": ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"],
        "time_features": ["month", "week", "quarter"],
        "lag_features": [1, 2, 4, 52],
        "rolling_windows": [4, 8, 13],
        "log_target": True,
        "store_fixed_effects": True,
    },
    "validation": {
        "n_splits": 5,
        "test_size": 13,
        "min_train": 52,
        "gap": 0,
        "expanding": True,
    },
    "final_holdout": {"test_size": 13, "gap": 0},
    "models": ["seasonal_naive", "store_mean", "ridge_global", "ridge_store"],
    "ridge": {"alpha": 1.0},
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    "comparison": {"reference": "seasonal_naive", "alpha": 0.05},
    "dtw": {"normalize": False, "band": None, "k_min": 2, "k_max": 10, "min_cluster_size": 5},
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 并合并缺省值。"""
    with open(path, "r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULTS, user_cfg)


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """将配置快照写入 JSON，保证实验可追溯。"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, ensure_ascii=False, indent=2)
