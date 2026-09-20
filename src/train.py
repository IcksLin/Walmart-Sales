"""基线流水线入口：加载 -> 特征 -> 滚动 CV -> 评估 -> 显著性检验 -> 日志落盘。

运行示例:
    python src/train.py --config configs/default.yaml --tag baseline --holdout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import wilcoxon
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from config import load_config, save_config
from features import add_history_features, add_time_features, build_design, history_feature_names
from logging_utils import get_line_logger
from metrics import back_transform, evaluate, mae, smearing_factor
from output_paths import line_dirs
from seed import env_snapshot, set_global_seed
from validation import final_holdout, rolling_origin_splits

SUPERVISED = {"ridge_global", "ridge_store", "lightgbm", "xgboost"}


def load_data(cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["path"])
    df[data_cfg["date_col"]] = pd.to_datetime(df[data_cfg["date_col"]], format=data_cfg["date_format"])
    df = add_time_features(df, data_cfg["date_col"])
    feat_cfg = cfg["features"]
    df = add_history_features(
        df,
        data_cfg["target"],
        data_cfg["store_col"],
        data_cfg["date_col"],
        feat_cfg["lag_features"],
        feat_cfg["rolling_windows"],
        feat_cfg["log_target"],
    )
    df = df.sort_values([data_cfg["store_col"], data_cfg["date_col"]]).reset_index(drop=True)
    return df


def seasonal_naive_predict(df, tr_idx, va_idx, cfg) -> np.ndarray:
    target = cfg["data"]["target"]
    store_col = cfg["data"]["store_col"]
    date_col = cfg["data"]["date_col"]
    tr = df.iloc[tr_idx]
    lookup = dict(
        zip(zip(tr[store_col].astype(int), tr[date_col]), tr[target].astype(float))
    )
    store_mean = tr.groupby(store_col)[target].mean()
    val = df.iloc[va_idx]
    out = np.empty(len(val), dtype=float)
    for pos, (_, row) in enumerate(val.iterrows()):
        key = (int(row[store_col]), row[date_col] - pd.Timedelta(weeks=52))
        out[pos] = lookup[key] if key in lookup else store_mean.get(row[store_col], np.nan)
    return out


def store_mean_predict(df, tr_idx, va_idx, cfg) -> np.ndarray:
    target = cfg["data"]["target"]
    store_col = cfg["data"]["store_col"]
    store_mean = df.iloc[tr_idx].groupby(store_col)[target].mean()
    return df.iloc[va_idx][store_col].map(store_mean).to_numpy(dtype=float)


def fill_design(design: pd.DataFrame, tr_idx: np.ndarray) -> pd.DataFrame:
    """用训练行中位数填补缺失（早期 lag52 为 NaN），仅用训练统计，无泄漏。"""
    medians = design.iloc[tr_idx].median(numeric_only=True)
    return design.fillna(medians)


def build_model(model_name: str, cfg: dict):
    seed = cfg["seed"]
    if model_name == "ridge_global" or model_name == "ridge_store":
        return Ridge(alpha=cfg["ridge"]["alpha"])
    if model_name == "lightgbm":
        return LGBMRegressor(random_state=seed, verbose=-1, n_jobs=1, **cfg["lightgbm"])
    if model_name == "xgboost":
        return XGBRegressor(random_state=seed, n_jobs=1, verbosity=0, **cfg["xgboost"])
    raise ValueError(f"未知模型: {model_name}")


def supervised_predict(model_name, design, tr_idx, va_idx, y_log, cfg) -> np.ndarray:
    X = fill_design(design, tr_idx)
    model = build_model(model_name, cfg)
    model.fit(X.iloc[tr_idx], y_log[tr_idx])
    factor = smearing_factor(y_log[tr_idx] - model.predict(X.iloc[tr_idx]))
    return back_transform(model.predict(X.iloc[va_idx]), factor)


def store_supervised_predict(model_name, design, df, tr_idx, va_idx, y_log, cfg) -> np.ndarray:
    store_col = cfg["data"]["store_col"]
    X = fill_design(design, tr_idx)
    out = np.empty(len(va_idx), dtype=float)
    tr_stores = df.iloc[tr_idx][store_col].to_numpy()
    va_stores = df.iloc[va_idx][store_col].to_numpy()
    for store in np.unique(va_stores):
        tr_s = tr_idx[tr_stores == store]
        va_s = np.flatnonzero(va_stores == store)
        if len(tr_s) < 10:
            out[va_s] = np.expm1(np.mean(y_log[tr_s])) if len(tr_s) else np.nan
            continue
        model = build_model(model_name, cfg)
        model.fit(X.iloc[tr_s], y_log[tr_s])
        factor = smearing_factor(y_log[tr_s] - model.predict(X.iloc[tr_s]))
        out[va_s] = back_transform(model.predict(X.iloc[va_idx[va_s]]), factor)
    return out


def predict(model_name, df, designs, tr_idx, va_idx, y_log, cfg) -> np.ndarray:
    if model_name == "seasonal_naive":
        return seasonal_naive_predict(df, tr_idx, va_idx, cfg)
    if model_name == "store_mean":
        return store_mean_predict(df, tr_idx, va_idx, cfg)
    if model_name == "ridge_store":
        return store_supervised_predict(model_name, designs["no_store"], df, tr_idx, va_idx, y_log, cfg)
    return supervised_predict(model_name, designs["with_store"], tr_idx, va_idx, y_log, cfg)


def date_masks(df, dates, train_pos, val_pos, cfg):
    date_col = cfg["data"]["date_col"]
    tr_idx = np.flatnonzero(df[date_col].isin(set(dates[train_pos])).to_numpy())
    va_idx = np.flatnonzero(df[date_col].isin(set(dates[val_pos])).to_numpy())
    return tr_idx, va_idx


def significance_table(abs_errors: dict[str, np.ndarray], wmae_by_model: dict[str, float], cfg) -> pd.DataFrame:
    ref = cfg["comparison"]["reference"]
    alpha = cfg["comparison"]["alpha"]
    ref_err = abs_errors[ref]
    rows = []
    for name, err in abs_errors.items():
        if name == ref:
            rows.append({"model": name, "wmae": wmae_by_model[name], "wmae_vs_ref": 0.0, "p_value": np.nan, "significant": False})
            continue
        diff = err - ref_err
        try:
            _, p_value = wilcoxon(err, ref_err, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p_value = np.nan
        rows.append(
            {
                "model": name,
                "wmae": wmae_by_model[name],
                "wmae_vs_ref": wmae_by_model[name] - wmae_by_model[ref],
                "p_value": float(p_value),
                "significant": bool(p_value < alpha) if not np.isnan(p_value) else False,
            }
        )
    return pd.DataFrame(rows).sort_values("wmae").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--holdout", action="store_true", help="额外跑末段留出终验")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logs_dir, _ = line_dirs(2)
    logger = get_line_logger("train", 2)
    set_global_seed(cfg["seed"], cfg["threads"])
    save_config(cfg, logs_dir / "train_config.json")
    with open(logs_dir / "train_env.json", "w", encoding="utf-8") as handle:
        json.dump(env_snapshot(cfg["seed"]), handle, ensure_ascii=False, indent=2)

    logger.info("输出目录: %s", logs_dir.parent)
    logger.info("随机种子固定为 %s，线程数 %s", cfg["seed"], cfg["threads"])

    df = load_data(cfg)
    target = cfg["data"]["target"]
    store_col = cfg["data"]["store_col"]
    date_col = cfg["data"]["date_col"]
    holiday_col = cfg["data"]["holiday_col"]
    history_cols = history_feature_names(cfg["features"]["lag_features"], cfg["features"]["rolling_windows"])
    y_true = df[target].to_numpy(dtype=float)
    y_log = np.log1p(y_true)
    weights = np.where(df[holiday_col].to_numpy() == 1, 5.0, 1.0)
    logger.info("数据规模: %d 行, %d 门店, %d 周", len(df), df[store_col].nunique(), df[date_col].nunique())
    logger.info("特征: 外生%d + 时间%d + 滞后滚动%d + 门店哑变量", len(cfg["features"]["exogenous"]), len(cfg["features"]["time_features"]), len(history_cols))

    designs = {
        "with_store": build_design(
            df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, True
        ),
        "no_store": build_design(
            df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, False
        ),
    }

    dates = np.sort(df[date_col].unique())
    vcfg = cfg["validation"]
    splits = rolling_origin_splits(
        len(dates), vcfg["n_splits"], vcfg["test_size"], vcfg["min_train"], vcfg["gap"], vcfg["expanding"]
    )
    logger.info("滚动 CV: %d 折, 每折验证 %d 周", len(splits), vcfg["test_size"])

    metric_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    abs_errors: dict[str, np.ndarray] = {}
    all_true_ref: np.ndarray | None = None
    all_w_ref: np.ndarray | None = None

    for model_name in cfg["models"]:
        logger.info("=== 模型: %s ===", model_name)
        all_true, all_pred, all_w = [], [], []
        for fold, (train_pos, val_pos) in enumerate(splits):
            tr_idx, va_idx = date_masks(df, dates, train_pos, val_pos, cfg)
            pred = predict(model_name, df, designs, tr_idx, va_idx, y_log, cfg)
            y_val = y_true[va_idx]
            w_val = weights[va_idx]
            naive = seasonal_naive_predict(df, tr_idx, va_idx, cfg)
            naive_mae = mae(y_val, naive)
            metrics = evaluate(y_val, pred, w_val)
            metrics["mase"] = metrics["mae"] / naive_mae if naive_mae > 0 else float("nan")
            metric_rows.append({"model": model_name, "fold": fold, **metrics})
            logger.info(
                "  fold %d | WMAE=%.1f WAPE=%.4f RMSE=%.1f R2=%.4f MASE=%.3f",
                fold, metrics["wmae"], metrics["wape"], metrics["rmse"], metrics["r2"], metrics["mase"],
            )
            all_true.append(y_val)
            all_pred.append(pred)
            all_w.append(w_val)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "fold": fold,
                        "Store": df.iloc[va_idx][store_col].to_numpy(),
                        "Date": df.iloc[va_idx][date_col].to_numpy(),
                        "y_true": y_val,
                        "y_pred": pred,
                    }
                )
            )
        y_concat = np.concatenate(all_true)
        p_concat = np.concatenate(all_pred)
        w_concat = np.concatenate(all_w)
        overall = evaluate(y_concat, p_concat, w_concat)
        metric_rows.append({"model": model_name, "fold": -1, **overall})
        abs_errors[model_name] = np.abs(y_concat - p_concat)
        if model_name == cfg["comparison"]["reference"]:
            all_true_ref, all_w_ref = y_concat, w_concat
        logger.info(
            "  总体 | WMAE=%.1f WAPE=%.4f RMSE=%.1f R2=%.4f",
            overall["wmae"], overall["wape"], overall["rmse"], overall["r2"],
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(logs_dir / "train_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(logs_dir / "train_predictions.csv", index=False)

    wmae_by_model = {
        row["model"]: row["wmae"] for _, row in metrics_df[metrics_df["fold"] == -1].iterrows()
    }
    comparison = significance_table(abs_errors, wmae_by_model, cfg)
    comparison.to_csv(logs_dir / "train_comparison.csv", index=False)

    logger.info("模型总体对比(按 WMAE 升序, 参考=%s):\n%s", cfg["comparison"]["reference"], comparison.to_string(index=False))

    if args.holdout:
        h = cfg["final_holdout"]
        train_pos, test_pos = final_holdout(len(dates), h["test_size"], h["gap"])
        tr_idx, te_idx = date_masks(df, dates, train_pos, test_pos, cfg)
        logger.info("=== 末段留出终验 (%d 周) ===", h["test_size"])
        for model_name in cfg["models"]:
            pred = predict(model_name, df, designs, tr_idx, te_idx, y_log, cfg)
            res = evaluate(y_true[te_idx], pred, weights[te_idx])
            logger.info("  %-15s WMAE=%.1f WAPE=%.4f RMSE=%.1f", model_name, res["wmae"], res["wape"], res["rmse"])

    logger.info("完成。产物输出到: %s", logs_dir.parent)


if __name__ == "__main__":
    main()
