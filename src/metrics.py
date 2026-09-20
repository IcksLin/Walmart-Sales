"""评价指标与目标反变换。

所有指标在原始金额尺度计算；log1p 建模后必须用 Duan smearing 做偏差校正。
"""

from __future__ import annotations

import numpy as np


def smearing_factor(residuals_log: np.ndarray) -> float:
    """Duan smearing 因子：mean(exp(residual))。"""
    return float(np.mean(np.exp(residuals_log)))


def back_transform(pred_log: np.ndarray, factor: float) -> np.ndarray:
    """将 log1p 空间预测还原到原尺度：expm1(pred + log(factor))。"""
    return np.expm1(np.asarray(pred_log) + np.log(factor))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def wmae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """Walmart 官方加权 MAE，节假日周权重默认为 5。"""
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    ratio = np.where(denom == 0, 0.0, 2.0 * np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom))
    return float(np.mean(ratio))


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """计算一组完整指标。weights 用于 WMAE（节假日 ×5）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if weights is None:
        weights = np.ones_like(y_true)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "wmae": wmae(y_true, y_pred, weights),
        "wape": wape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
