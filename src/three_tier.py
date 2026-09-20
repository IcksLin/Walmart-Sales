"""三层建模：A 全局池化 / B DTW 簇级 / C 单店，滚动 CV 对比与门店级评估。

运行: python src/three_tier.py --config configs/default.yaml --tag three_tier
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config, save_config
from dtw import cluster_stores, compute_dtw_matrix, select_n_clusters
from features import build_design, history_feature_names
from logging_utils import get_line_logger
from metrics import back_transform, evaluate, smearing_factor
from output_paths import line_dirs
from seed import env_snapshot, set_global_seed
from train import (
    build_model,
    date_masks,
    fill_design,
    load_data,
    seasonal_naive_predict,
)
from validation import rolling_origin_splits


def store_series_matrix(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """门店 × 周的 log1p 销售矩阵。

    默认保留规模（不做 z-score），使 DTW 聚类得到规模相近的门店群；
    可用 cfg['dtw']['normalize'] 切换为仅按形状聚类。
    """
    store_col = cfg["data"]["store_col"]
    date_col = cfg["data"]["date_col"]
    target = cfg["data"]["target"]
    pivot = df.pivot(index=store_col, columns=date_col, values=target).sort_index()
    stores = pivot.index.to_numpy()
    values = np.log1p(pivot.to_numpy(dtype=float))
    if cfg.get("dtw", {}).get("normalize", False):
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        values = (values - mean) / np.where(std == 0, 1, std)
    return stores, values


def grouped_predict(model_name, design, tr_idx, va_idx, y_log, cfg, group_values) -> np.ndarray:
    """按分组（簇或门店）分别训练模型。"""
    X = fill_design(design, tr_idx)
    out = np.full(len(va_idx), np.nan)
    tr_groups = group_values[tr_idx]
    va_groups = group_values[va_idx]
    for group in np.unique(va_groups):
        tr_g = tr_idx[tr_groups == group]
        va_g = np.flatnonzero(va_groups == group)
        if len(tr_g) < 10:
            out[va_g] = np.expm1(np.mean(y_log[tr_g])) if len(tr_g) else np.nan
            continue
        model = build_model(model_name, cfg)
        model.fit(X.iloc[tr_g], y_log[tr_g])
        factor = smearing_factor(y_log[tr_g] - model.predict(X.iloc[tr_g]))
        out[va_g] = back_transform(model.predict(X.iloc[va_idx[va_g]]), factor)
    return out


def per_store_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tier, store), g in pred_df.groupby(["tier", "Store"]):
        err = np.abs(g["y_true"] - g["y_pred"])
        rows.append(
            {
                "tier": tier,
                "Store": store,
                "wape": err.sum() / np.abs(g["y_true"]).sum(),
                "mae": err.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default="three_tier")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logs_dir, figs_dir = line_dirs(2)
    logger = get_line_logger("three_tier", 2)
    set_global_seed(cfg["seed"], cfg["threads"])
    save_config(cfg, logs_dir / "three_tier_config.json")
    with open(logs_dir / "three_tier_env.json", "w", encoding="utf-8") as handle:
        json.dump(env_snapshot(cfg["seed"]), handle, ensure_ascii=False, indent=2)

    df = load_data(cfg)
    store_col = cfg["data"]["store_col"]
    date_col = cfg["data"]["date_col"]
    target = cfg["data"]["target"]
    holiday_col = cfg["data"]["holiday_col"]
    y_true = df[target].to_numpy(dtype=float)
    y_log = np.log1p(y_true)
    weights = np.where(df[holiday_col].to_numpy() == 1, 5.0, 1.0)
    history_cols = history_feature_names(cfg["features"]["lag_features"], cfg["features"]["rolling_windows"])
    designs = {
        "with_store": build_design(
            df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, True
        ),
        "no_store": build_design(
            df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, False
        ),
    }

    # --- DTW 门店聚类 ---
    stores, series = store_series_matrix(df, cfg)
    dtw_cfg = cfg.get("dtw", {})
    logger.info("计算 DTW 距离矩阵: %d 门店 × %d 周", *series.shape)
    t0 = time.time()
    dist = compute_dtw_matrix(series, band=dtw_cfg.get("band"))
    logger.info("DTW 完成，用时 %.1fs", time.time() - t0)
    best_k, scores, elbow_k = select_n_clusters(
        dist,
        k_min=dtw_cfg.get("k_min", 2),
        k_max=dtw_cfg.get("k_max", 10),
        min_cluster_size=dtw_cfg.get("min_cluster_size", 5),
    )
    labels = cluster_stores(dist, best_k)
    store_to_cluster = dict(zip(stores.tolist(), labels.tolist()))
    logger.info("自动定簇: 轮廓最优 K=%d, 肘部 K=%d, 各簇规模=%s", best_k, elbow_k, np.bincount(labels).tolist())
    logger.info("轮廓系数: %s", {k: round(v["silhouette"], 3) for k, v in scores.items()})

    cluster_values = df[store_col].map(store_to_cluster).to_numpy()
    store_values = df[store_col].to_numpy()

    pd.DataFrame({"Store": stores, "cluster": labels}).to_csv(logs_dir / "dtw_clusters.csv", index=False)

    plt.figure()
    ks = sorted(scores)
    plt.plot(ks, [scores[k]["silhouette"] for k in ks], marker="o")
    plt.axvline(best_k, color="red", linestyle="--", label=f"silhouette best K={best_k}")
    plt.axvline(elbow_k, color="green", linestyle=":", label=f"elbow K={elbow_k}")
    plt.xlabel("n_clusters")
    plt.ylabel("silhouette")
    plt.title("DTW clustering silhouette")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "dtw_silhouette.png", dpi=150)
    plt.close()

    order = np.argsort(labels, kind="stable")
    plt.figure(figsize=(8, 7))
    plt.imshow(dist[np.ix_(order, order)], cmap="viridis")
    plt.colorbar(label="DTW distance")
    plt.title(f"Store DTW distance matrix (ordered by cluster, K={best_k})")
    plt.tight_layout()
    plt.savefig(figs_dir / "store_dtw_heatmap.png", dpi=150)
    plt.close()

    # --- 三层滚动 CV ---
    dates = np.sort(df[date_col].unique())
    vcfg = cfg["validation"]
    splits = rolling_origin_splits(
        len(dates), vcfg["n_splits"], vcfg["test_size"], vcfg["min_train"], vcfg["gap"], vcfg["expanding"]
    )
    engine = "xgboost"
    tiers = ["seasonal_naive", "A_global", "B_cluster", "C_store"]

    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    for tier in tiers:
        logger.info("=== 层级: %s ===", tier)
        all_true, all_pred, all_w = [], [], []
        for fold, (train_pos, val_pos) in enumerate(splits):
            tr_idx, va_idx = date_masks(df, dates, train_pos, val_pos, cfg)
            if tier == "seasonal_naive":
                pred = seasonal_naive_predict(df, tr_idx, va_idx, cfg)
            elif tier == "A_global":
                from train import predict

                pred = predict(engine, df, designs, tr_idx, va_idx, y_log, cfg)
            elif tier == "B_cluster":
                pred = grouped_predict(engine, designs["no_store"], tr_idx, va_idx, y_log, cfg, cluster_values)
            else:
                pred = grouped_predict(engine, designs["no_store"], tr_idx, va_idx, y_log, cfg, store_values)
            y_val = y_true[va_idx]
            w_val = weights[va_idx]
            metrics = evaluate(y_val, pred, w_val)
            metric_rows.append({"tier": tier, "fold": fold, **metrics})
            all_true.append(y_val)
            all_pred.append(pred)
            all_w.append(w_val)
            pred_frames.append(
                pd.DataFrame(
                    {
                        "tier": tier,
                        "fold": fold,
                        "Store": df.iloc[va_idx][store_col].to_numpy(),
                        "Date": df.iloc[va_idx][date_col].to_numpy(),
                        "y_true": y_val,
                        "y_pred": pred,
                    }
                )
            )
        overall = evaluate(np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_w))
        metric_rows.append({"tier": tier, "fold": -1, **overall})
        logger.info("  总体 | WMAE=%.1f WAPE=%.4f R2=%.4f", overall["wmae"], overall["wape"], overall["r2"])

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(logs_dir / "three_tier_metrics.csv", index=False)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    per_store = per_store_metrics(pred_df)
    per_store.to_csv(logs_dir / "three_tier_per_store.csv", index=False)

    summary = (
        metrics_df[metrics_df["fold"] == -1][["tier", "wmae", "wape", "rmse", "r2"]]
        .sort_values("wmae")
        .to_string(index=False)
    )
    logger.info("三层总体对比(按 WMAE):\n%s", summary)

    dist_summary = (
        per_store.groupby("tier")["wape"]
        .agg(["median", lambda s: s.quantile(0.9), "max"])
        .rename(columns={"<lambda_0>": "p90", "max": "worst"})
        .sort_values("median")
    )
    logger.info("门店级 WAPE 分布:\n%s", dist_summary.to_string())

    logger.info("完成。产物输出到: %s", logs_dir.parent)


if __name__ == "__main__":
    main()
