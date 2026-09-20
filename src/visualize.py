"""可视化工作流：按分析线输出图像与日志。

line_1 数据探索 : 变量分布图、变量间热力相关图
line_2 训练过程 : 训练/验证 LOSS 变化曲线
line_3 测试评估 : 测试集预测值 vs 真实值、残差分布
line_4 模型解释 : 线性模型特征权重

所有产物写入 output/line_X/{logs,pictures}。

运行: python src/visualize.py [--line all|1|2|3|4]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from config import load_config
from features import build_design, history_feature_names
from logging_utils import get_line_logger
from output_paths import line_dirs
from plot_style import open_axes
from seed import env_snapshot, set_global_seed
from train import date_masks, fill_design, load_data
from validation import final_holdout

MAIN_VARS = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag"]


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def line_1_data_profile(df: pd.DataFrame, logger) -> None:
    logs_dir, pics_dir = line_dirs(1)
    with open(logs_dir / "data_profile_env.json", "w", encoding="utf-8") as handle:
        json.dump(env_snapshot(), handle, ensure_ascii=False, indent=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), MAIN_VARS):
        data = np.log1p(df[col]) if col == "Weekly_Sales" else df[col]
        label = "log1p(Weekly_Sales)" if col == "Weekly_Sales" else col
        ax.hist(data, bins=40, color="#1f77b4", alpha=0.8)
        ax.set_title(f"Distribution: {label}")
        ax.set_ylabel("count")
        open_axes(ax, "y")
    fig.suptitle("Main variable distributions", fontsize=14)
    _save(fig, pics_dir / "variable_distributions.png")
    logger.info("line_1 变量分布图: %s", pics_dir / "variable_distributions.png")

    corr = df[MAIN_VARS].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(MAIN_VARS)))
    ax.set_yticks(range(len(MAIN_VARS)))
    ax.set_xticklabels(MAIN_VARS, rotation=45, ha="right")
    ax.set_yticklabels(MAIN_VARS)
    for i in range(len(MAIN_VARS)):
        for j in range(len(MAIN_VARS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlation heatmap")
    _save(fig, pics_dir / "correlation_heatmap.png")
    logger.info("line_1 相关热力图: %s", pics_dir / "correlation_heatmap.png")

    corr.to_csv(logs_dir / "correlation_matrix.csv")
    logger.info("相关系数矩阵已保存")


def _time_split_masks(df, cfg):
    date_col = cfg["data"]["date_col"]
    dates = np.sort(df[date_col].unique())
    cutoff = dates[int(len(dates) * 0.8)]
    return (df[date_col] < cutoff).to_numpy(), (df[date_col] >= cutoff).to_numpy()


def line_2_training(df: pd.DataFrame, cfg: dict, logger) -> None:
    logs_dir, pics_dir = line_dirs(2)
    store_col = cfg["data"]["store_col"]
    target = cfg["data"]["target"]
    history_cols = history_feature_names(cfg["features"]["lag_features"], cfg["features"]["rolling_windows"])
    design = build_design(
        df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, True
    )
    tr_mask, va_mask = _time_split_masks(df, cfg)
    design = fill_design(design, np.flatnonzero(tr_mask))
    Xtr, Xva = design.to_numpy(float)[tr_mask], design.to_numpy(float)[va_mask]
    y = np.log1p(df[target].to_numpy(float))
    ytr, yva = y[tr_mask], y[va_mask]

    xgb = XGBRegressor(random_state=cfg["seed"], n_jobs=1, verbosity=0, eval_metric="rmse", **cfg["xgboost"])
    xgb.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    hist = xgb.evals_result()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["validation_0"]["rmse"], label="train")
    ax.plot(hist["validation_1"]["rmse"], label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMSE (log1p space)")
    ax.set_title("XGBoost loss curve")
    ax.legend()
    open_axes(ax, "y")
    _save(fig, pics_dir / "loss_curve_xgboost.png")

    lgb_hist: dict = {}
    lgbm = LGBMRegressor(random_state=cfg["seed"], n_jobs=1, verbose=-1, **cfg["lightgbm"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm.fit(
            Xtr, ytr,
            eval_set=[(Xtr, ytr), (Xva, yva)],
            eval_metric="rmse",
            callbacks=[lgb.record_evaluation(lgb_hist)],
        )
    keys = list(lgb_hist.keys())
    train_key = "training" if "training" in keys else keys[0]
    val_key = "valid_0" if "valid_0" in keys else keys[1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lgb_hist[train_key]["rmse"], label="train")
    ax.plot(lgb_hist[val_key]["rmse"], label="validation")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMSE (log1p space)")
    ax.set_title("LightGBM loss curve")
    ax.legend()
    open_axes(ax, "y")
    _save(fig, pics_dir / "loss_curve_lightgbm.png")

    with open(logs_dir / "loss_curves_env.json", "w", encoding="utf-8") as handle:
        json.dump(env_snapshot(cfg["seed"]), handle, ensure_ascii=False, indent=2)
    logger.info("line_2 loss 曲线: %s", pics_dir)


def line_3_evaluation(df: pd.DataFrame, cfg: dict, logger) -> None:
    logs_dir, pics_dir = line_dirs(3)
    store_col = cfg["data"]["store_col"]
    date_col = cfg["data"]["date_col"]
    target = cfg["data"]["target"]
    history_cols = history_feature_names(cfg["features"]["lag_features"], cfg["features"]["rolling_windows"])
    design = build_design(
        df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, True
    )
    dates = np.sort(df[date_col].unique())
    tr_pos, te_pos = final_holdout(len(dates), cfg["final_holdout"]["test_size"], cfg["final_holdout"]["gap"])
    tr_idx, te_idx = date_masks(df, dates, tr_pos, te_pos, cfg)
    X = fill_design(design, tr_idx).to_numpy(float)
    y = np.log1p(df[target].to_numpy(float))
    f = X.shape[1]

    xgb = XGBRegressor(random_state=cfg["seed"], n_jobs=1, verbosity=0, **cfg["xgboost"])
    xgb.fit(X[tr_idx], y[tr_idx])
    resid_tr = y[tr_idx] - xgb.predict(X[tr_idx])
    factor = float(np.mean(np.exp(resid_tr)))
    pred = np.expm1(xgb.predict(X[te_idx]) + np.log(factor))
    true = df[target].to_numpy(float)[te_idx]
    te_dates = df[date_col].to_numpy()[te_idx]

    order = np.argsort(te_dates)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].scatter(true, pred, s=8, alpha=0.3)
    lim = [0, max(true.max(), pred.max()) * 1.05]
    axes[0].plot(lim, lim, "r--", linewidth=1)
    axes[0].set_xlabel("actual")
    axes[0].set_ylabel("predicted")
    axes[0].set_title("Predicted vs actual (test set)")
    open_axes(axes[0], "both")

    n = min(200, len(order))
    sel = order[-n:]
    axes[1].plot(range(n), true[sel], label="actual", marker="o", markersize=3)
    axes[1].plot(range(n), pred[sel], label="predicted", marker="x", markersize=3)
    axes[1].set_xlabel("test week index (last 200 obs)")
    axes[1].set_ylabel("Weekly_Sales")
    axes[1].set_title("Test series: predicted vs actual")
    axes[1].legend()
    open_axes(axes[1], "y")
    _save(fig, pics_dir / "pred_vs_actual.png")

    resid = true - pred
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(resid, bins=50, color="#d62728", alpha=0.8)
    axes[0].set_title("Residual distribution (test)")
    axes[0].set_xlabel("actual - predicted")
    axes[1].scatter(pred, resid, s=8, alpha=0.3)
    axes[1].axhline(0, color="k", linewidth=1)
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("residual")
    axes[1].set_title("Residual vs predicted")
    open_axes(axes[0], "y")
    open_axes(axes[1], "both")
    _save(fig, pics_dir / "residual_distribution.png")

    pd.DataFrame({"Date": te_dates, "Store": df[store_col].to_numpy()[te_idx], "actual": true, "predicted": pred}).to_csv(
        logs_dir / "test_predictions.csv", index=False
    )
    logger.info("line_3 测试评估图: %s | 特征维度 %d", pics_dir, f)


def line_4_interpretation(df: pd.DataFrame, cfg: dict, logger) -> None:
    logs_dir, pics_dir = line_dirs(4)
    store_col = cfg["data"]["store_col"]
    target = cfg["data"]["target"]
    history_cols = history_feature_names(cfg["features"]["lag_features"], cfg["features"]["rolling_windows"])
    design = build_design(
        df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, include_store=False
    )
    design = design.fillna(design.median(numeric_only=True))
    X = design.to_numpy(float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    y = np.log1p(df[target].to_numpy(float))

    model = Ridge(alpha=cfg["ridge"]["alpha"])
    model.fit(X, y)
    coefs = pd.Series(model.coef_, index=design.columns).sort_values(key=np.abs, ascending=False)
    top = coefs.head(20).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in top.values]
    ax.barh(range(len(top)), top.values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("standardized coefficient (log1p target)")
    ax.set_title("Linear model feature weights (Ridge, top 20)")
    open_axes(ax, "x")
    _save(fig, pics_dir / "linear_feature_weights.png")

    coefs.to_csv(logs_dir / "linear_feature_weights.csv", header=["coef"])
    logger.info("line_4 线性特征权重图: %s", pics_dir / "linear_feature_weights.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--line", default="all", choices=["all", "1", "2", "3", "4"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["seed"], cfg["threads"])
    df = load_data(cfg)

    if args.line in ("all", "1"):
        logger = get_line_logger("visualize_line1", 1)
        logger.info("line_1 数据探索开始")
        line_1_data_profile(df, logger)
    if args.line in ("all", "2"):
        logger = get_line_logger("visualize_line2", 2)
        logger.info("line_2 训练曲线开始")
        line_2_training(df, cfg, logger)
    if args.line in ("all", "3"):
        logger = get_line_logger("visualize_line3", 3)
        logger.info("line_3 测试评估开始")
        line_3_evaluation(df, cfg, logger)
    if args.line in ("all", "4"):
        logger = get_line_logger("visualize_line4", 4)
        logger.info("line_4 模型解释开始")
        line_4_interpretation(df, cfg, logger)


if __name__ == "__main__":
    main()
