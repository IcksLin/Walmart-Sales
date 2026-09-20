"""分层评估与极端门店诊断。

按年份 / 节假日 / 门店 / 温度分段评估三层模型，并对最优层级做残差与极端门店诊断。

运行: python src/stratified_eval.py --config configs/default.yaml --tag stratified
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan

from config import load_config, save_config
from dtw import cluster_stores, compute_dtw_matrix, select_n_clusters
from features import build_design, history_feature_names
from logging_utils import get_line_logger
from metrics import evaluate
from output_paths import line_dirs
from seed import env_snapshot, set_global_seed
from three_tier import grouped_predict, store_series_matrix
from train import date_masks, load_data, predict, seasonal_naive_predict
from validation import rolling_origin_splits

PRIMARY_TIER = "B_cluster"


def build_predictions(df, cfg, designs, cluster_values, store_values):
    """滚动 CV 收集各层逐样本预测（含分层标签）。"""
    store_col = cfg["data"]["store_col"]
    date_col = cfg["data"]["date_col"]
    target = cfg["data"]["target"]
    holiday_col = cfg["data"]["holiday_col"]
    y_true = df[target].to_numpy(dtype=float)
    y_log = np.log1p(y_true)
    weights = np.where(df[holiday_col].to_numpy() == 1, 5.0, 1.0)

    dates = np.sort(df[date_col].unique())
    v = cfg["validation"]
    splits = rolling_origin_splits(len(dates), v["n_splits"], v["test_size"], v["min_train"], v["gap"], v["expanding"])
    engine = "xgboost"

    frames = []
    for tier in ["seasonal_naive", "A_global", "B_cluster", "C_store"]:
        for fold, (tr_pos, va_pos) in enumerate(splits):
            tr_idx, va_idx = date_masks(df, dates, tr_pos, va_pos, cfg)
            if tier == "seasonal_naive":
                pred = seasonal_naive_predict(df, tr_idx, va_idx, cfg)
            elif tier == "A_global":
                pred = predict(engine, df, designs, tr_idx, va_idx, y_log, cfg)
            elif tier == "B_cluster":
                pred = grouped_predict(engine, designs["no_store"], tr_idx, va_idx, y_log, cfg, cluster_values)
            else:
                pred = grouped_predict(engine, designs["no_store"], tr_idx, va_idx, y_log, cfg, store_values)
            frames.append(
                pd.DataFrame(
                    {
                        "tier": tier,
                        "fold": fold,
                        "Store": df.iloc[va_idx][store_col].to_numpy(),
                        "Date": df.iloc[va_idx][date_col].to_numpy(),
                        "year": df.iloc[va_idx]["year"].to_numpy(),
                        "is_holiday": df.iloc[va_idx][holiday_col].to_numpy(),
                        "Temperature": df.iloc[va_idx]["Temperature"].to_numpy(),
                        "y_true": y_true[va_idx],
                        "y_pred": pred,
                        "weight": weights[va_idx],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _metrics(group: pd.DataFrame) -> dict:
    return evaluate(group["y_true"].to_numpy(), group["y_pred"].to_numpy(), group["weight"].to_numpy())


def stratified_tables(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    for key, cols in {
        "year": ["year"],
        "holiday": ["is_holiday"],
        "store": ["Store"],
    }.items():
        rows = []
        for group_keys, g in pred.groupby(["tier"] + cols):
            tier = group_keys[0]
            label = group_keys[1:]
            rows.append({"tier": tier, cols[0]: label[0] if len(label) == 1 else label, **_metrics(g)})
        tables[key] = pd.DataFrame(rows)

    temp_bins = [-100, 32, 50, 70, 90, 200]
    labels = ["<32F", "32-50F", "50-70F", "70-90F", ">90F"]
    pred = pred.copy()
    pred["temp_bin"] = pd.cut(pred["Temperature"], bins=temp_bins, labels=labels, right=False).astype(str)
    rows = []
    for (tier, tb), g in pred.groupby(["tier", "temp_bin"]):
        rows.append({"tier": tier, "temp_bin": tb, **_metrics(g)})
    tables["temperature"] = pd.DataFrame(rows)
    return tables


def residual_diagnostics(pred: pd.DataFrame, logger) -> pd.DataFrame:
    rows = []
    for tier, g in pred.groupby("tier"):
        resid = g["y_true"].to_numpy() - g["y_pred"].to_numpy()
        weekly = g.assign(resid=resid).groupby("Date")["resid"].mean().sort_index()
        lb = acorr_ljungbox(weekly, lags=[8], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
        bp = het_breuschpagan(resid, np.column_stack([np.ones(len(g)), g["y_pred"].to_numpy()]))
        rows.append(
            {
                "tier": tier,
                "resid_mean": float(resid.mean()),
                "resid_std": float(resid.std()),
                "ljung_box_p": lb_p,
                "breusch_pagan_p": float(bp[1]),
                "autocorr_ok": lb_p > 0.05,
                "homoskedastic_ok": float(bp[1]) > 0.05,
            }
        )
    table = pd.DataFrame(rows)
    logger.info("残差诊断:\n%s", table.to_string(index=False))
    return table


def extreme_store_diagnosis(pred: pd.DataFrame, df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    store_col = cfg["data"]["store_col"]
    target = cfg["data"]["target"]
    stats = df.groupby(store_col)[target].agg(["mean", "std"]).rename(columns={"mean": "mean_sales", "std": "std_sales"})
    lags = df.sort_values([store_col, cfg["data"]["date_col"]]).groupby(store_col)[target].apply(
        lambda s: np.log1p(s).autocorr(lag=1)
    ).rename("lag1_autocorr")
    rows = []
    for tier, g in pred.groupby("tier"):
        per_store = []
        for store, sg in g.groupby("Store"):
            err = np.abs(sg["y_true"] - sg["y_pred"])
            per_store.append({"Store": store, "wape": err.sum() / np.abs(sg["y_true"]).sum(), "mae": err.mean()})
        ps = pd.DataFrame(per_store)
        ps = ps.join(stats, on="Store").join(lags, on="Store")
        ps["tier"] = tier
        rows.append(ps)
    table = pd.concat(rows, ignore_index=True)
    primary = table[table["tier"] == PRIMARY_TIER].sort_values("wape", ascending=False)
    logger.info("极端门店(最差 10, %s):\n%s", PRIMARY_TIER, primary.head(10).to_string(index=False))
    return table


def make_figures(tables: dict[str, pd.DataFrame], extreme: pd.DataFrame, pred: pd.DataFrame, figs_dir: Path) -> None:
    fig, ax = plt.subplots()
    for tier, g in tables["year"].groupby("tier"):
        g = g.sort_values("year")
        ax.plot(g["year"], g["wape"], marker="o", label=tier)
    ax.set_xlabel("year")
    ax.set_ylabel("WAPE")
    ax.set_title("WAPE by year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "stratified_by_year.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    hol = tables["holiday"].copy()
    hol["is_holiday"] = hol["is_holiday"].map({0: "normal", 1: "holiday"})
    pivot = hol.pivot(index="tier", columns="is_holiday", values="wape")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("WAPE")
    ax.set_title("WAPE by holiday flag")
    fig.tight_layout()
    fig.savefig(figs_dir / "stratified_by_holiday.png", dpi=150)
    plt.close(fig)

    primary_pred = pred[pred["tier"] == PRIMARY_TIER]
    per_store = []
    for store, sg in primary_pred.groupby("Store"):
        err = np.abs(sg["y_true"] - sg["y_pred"])
        per_store.append({"Store": store, "wape": err.sum() / np.abs(sg["y_true"]).sum()})
    ps = pd.DataFrame(per_store).sort_values("wape", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ps["Store"].astype(str), ps["wape"])
    ax.set_xlabel("Store")
    ax.set_ylabel("WAPE")
    ax.set_title(f"Per-store WAPE ({PRIMARY_TIER})")
    fig.tight_layout()
    fig.savefig(figs_dir / "store_wape_sorted.png", dpi=150)
    plt.close(fig)

    resid = primary_pred["y_true"].to_numpy() - primary_pred["y_pred"].to_numpy()
    fig, ax = plt.subplots()
    ax.hist(resid, bins=50)
    ax.set_title(f"Residual distribution ({PRIMARY_TIER})")
    fig.tight_layout()
    fig.savefig(figs_dir / "residual_hist.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default="stratified")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logs_dir, figs_dir = line_dirs(3)
    logger = get_line_logger("stratified", 3)
    set_global_seed(cfg["seed"], cfg["threads"])
    save_config(cfg, logs_dir / "stratified_config.json")
    with open(logs_dir / "stratified_env.json", "w", encoding="utf-8") as handle:
        json.dump(env_snapshot(cfg["seed"]), handle, ensure_ascii=False, indent=2)

    df = load_data(cfg)
    store_col = cfg["data"]["store_col"]
    history_cols = history_feature_names(cfg["features"]["lag_features"], cfg["features"]["rolling_windows"])
    designs = {
        "with_store": build_design(
            df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, True
        ),
        "no_store": build_design(
            df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, False
        ),
    }

    stores, series = store_series_matrix(df, cfg)
    dtw_cfg = cfg.get("dtw", {})
    dist = compute_dtw_matrix(series, band=dtw_cfg.get("band"))
    best_k, _, _ = select_n_clusters(
        dist, dtw_cfg.get("k_min", 2), dtw_cfg.get("k_max", 10), dtw_cfg.get("min_cluster_size", 5)
    )
    labels = cluster_stores(dist, best_k)
    store_to_cluster = dict(zip(stores.tolist(), labels.tolist()))
    cluster_values = df[store_col].map(store_to_cluster).to_numpy()
    store_values = df[store_col].to_numpy()
    logger.info("DTW 聚类 K=%d", best_k)

    pred = build_predictions(df, cfg, designs, cluster_values, store_values)
    pred.to_csv(logs_dir / "stratified_predictions.csv", index=False)

    tables = stratified_tables(pred)
    for name, table in tables.items():
        table.to_csv(logs_dir / f"stratified_by_{name}.csv", index=False)

    logger.info("按年份 WAPE:\n%s", tables["year"].pivot(index="tier", columns="year", values="wape").to_string())
    logger.info("按节假日 WAPE:\n%s", tables["holiday"].pivot(index="tier", columns="is_holiday", values="wape").to_string())
    logger.info("按温度 WAPE:\n%s", tables["temperature"].pivot(index="tier", columns="temp_bin", values="wape").to_string())

    resid = residual_diagnostics(pred, logger)
    resid.to_csv(logs_dir / "residual_diagnostics.csv", index=False)

    extreme = extreme_store_diagnosis(pred, df, cfg, logger)
    extreme.sort_values(["tier", "wape"], ascending=[True, False]).to_csv(
        logs_dir / "store_diagnostics.csv", index=False
    )

    make_figures(tables, extreme, pred, figs_dir)
    logger.info("完成。产物输出到: %s", logs_dir.parent)


if __name__ == "__main__":
    main()
