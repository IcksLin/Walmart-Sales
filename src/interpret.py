"""解释型分析：固定效应回归（系数/显著性/VIF）+ SHAP 特征重要性。

回答“哪些因素影响销售额、方向与强弱”，并交叉验证两种证据。

运行: python src/interpret.py --config configs/default.yaml --tag interpret
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
import shap
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor

from config import load_config, save_config
from features import add_history_features, add_time_features, build_design, history_feature_names
from logging_utils import get_line_logger
from output_paths import line_dirs
from seed import env_snapshot, set_global_seed


def prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    data_cfg = cfg["data"]
    df = add_time_features(df, data_cfg["date_col"])
    feat = cfg["features"]
    df = add_history_features(
        df,
        data_cfg["target"],
        data_cfg["store_col"],
        data_cfg["date_col"],
        feat["lag_features"],
        feat["rolling_windows"],
        feat["log_target"],
    )
    return df


def fixed_effects_regression(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    """log1p(sales) ~ 外生因素 + 门店固定效应 + 年份固定效应，HC1 稳健标准误。"""
    data_cfg = cfg["data"]
    feat = cfg["features"]
    target = data_cfg["target"]
    store_col = data_cfg["store_col"]

    y = np.log1p(df[target].to_numpy(dtype=float))
    exog = feat["exogenous"]
    core = df[exog].astype(float)
    store_dummies = pd.get_dummies(df[store_col], prefix="Store", drop_first=True).astype(float)
    year_dummies = pd.get_dummies(df["year"], prefix="Year", drop_first=True).astype(float)

    X = pd.concat([core, store_dummies, year_dummies], axis=1)
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit(cov_type="HC1")

    # 仅核心外生因素系数
    rows = []
    for name in exog:
        rows.append(
            {
                "feature": name,
                "coef": model.params[name],
                "std_err": model.bse[name],
                "t": model.tvalues[name],
                "p_value": model.pvalues[name],
                "ci_low": model.conf_int().loc[name, 0],
                "ci_high": model.conf_int().loc[name, 1],
            }
        )
    coef_table = pd.DataFrame(rows)
    logger.info("固定效应回归 R²=%.4f, 调整R²=%.4f", model.rsquared, model.rsquared_adj)
    logger.info("核心因素系数(log空间):\n%s", coef_table.to_string(index=False))
    return coef_table


def vif_table(df: pd.DataFrame, cfg: dict, logger) -> pd.DataFrame:
    exog = cfg["features"]["exogenous"]
    X = sm.add_constant(df[exog].astype(float))
    rows = []
    for i, name in enumerate(X.columns):
        if name == "const":
            continue
        rows.append({"feature": name, "vif": variance_inflation_factor(X.to_numpy(), i)})
    table = pd.DataFrame(rows).sort_values("vif", ascending=False)
    logger.info("VIF(共线性诊断):\n%s", table.to_string(index=False))
    return table.reset_index(drop=True)


def shap_analysis(df: pd.DataFrame, cfg: dict, figs_dir: Path, logs_dir: Path, logger):
    data_cfg = cfg["data"]
    feat = cfg["features"]
    store_col = data_cfg["store_col"]
    target = data_cfg["target"]
    history_cols = history_feature_names(feat["lag_features"], feat["rolling_windows"])

    # 与最优模型 B_cluster 保持同一特征集：不含门店哑变量
    design = build_design(
        df, feat["exogenous"], feat["time_features"], history_cols, store_col, include_store=False
    )
    design = design.fillna(design.median(numeric_only=True))
    y = np.log1p(df[target].to_numpy(dtype=float))

    model = XGBRegressor(random_state=cfg["seed"], n_jobs=1, verbosity=0, **cfg["xgboost"])
    model.fit(design, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(design)

    importance = pd.DataFrame(
        {
            "feature": design.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    non_store = importance[~importance["feature"].str.startswith("Store_")].reset_index(drop=True)
    logger.info("SHAP 全局重要性 Top15(全部):\n%s", importance.head(15).to_string(index=False))
    logger.info("SHAP 全局重要性 Top15(剔除门店哑变量):\n%s", non_store.head(15).to_string(index=False))

    top_features = non_store["feature"].head(15).tolist()
    pd.DataFrame(shap_values, columns=design.columns)[top_features].to_csv(
        logs_dir / "shap_values_top.csv", index=False
    )
    non_store.to_csv(logs_dir / "shap_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values, design, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(figs_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, design, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(figs_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    for col in [c for c in ["Temperature", "CPI", "Unemployment", "Fuel_Price"] if c in design.columns]:
        plt.figure()
        shap.dependence_plot(col, shap_values, design, show=False)
        plt.tight_layout()
        plt.savefig(figs_dir / f"shap_dependence_{col}.png", dpi=150, bbox_inches="tight")
        plt.close()

    return importance, non_store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tag", default="interpret")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logs_dir, figs_dir = line_dirs(4)
    logger = get_line_logger("interpret", 4)
    set_global_seed(cfg["seed"], cfg["threads"])
    save_config(cfg, logs_dir / "interpret_config.json")
    with open(logs_dir / "interpret_env.json", "w", encoding="utf-8") as handle:
        json.dump(env_snapshot(cfg["seed"]), handle, ensure_ascii=False, indent=2)

    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["path"])
    df[data_cfg["date_col"]] = pd.to_datetime(df[data_cfg["date_col"]], format=data_cfg["date_format"])
    df = prepare(df, cfg)

    logger.info("输出目录: %s", logs_dir.parent)

    coef_table = fixed_effects_regression(df, cfg, logger)
    coef_table.to_csv(logs_dir / "regression_coefficients.csv", index=False)

    vif = vif_table(df, cfg, logger)
    vif.to_csv(logs_dir / "vif.csv", index=False)

    shap_analysis(df, cfg, figs_dir, logs_dir, logger)

    logger.info("完成。产物输出到: %s", logs_dir.parent)


if __name__ == "__main__":
    main()
