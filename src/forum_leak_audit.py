"""复现并审计论坛 notebook（walmart_sales_prediction.ipynb）的方法。

目的：量化其两处数据泄漏对 R² 的贡献。
- 泄漏1: store_sales_encoded 在**全量数据**上做目标编码（含测试集目标）
- 泄漏2: 用 random train_test_split 切分时间序列

输出 2x2 对照表（切分方式 × 编码方式）。

运行: python src/forum_leak_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from seed import set_global_seed
from output_paths import line_dirs

DATA = "practice_1.csv"
TARGET = "Weekly_Sales"


def build_features(df: pd.DataFrame, train_mask: np.ndarray | None, full_encoding: bool) -> pd.DataFrame:
    """构造论坛同款特征；encoding 可控制是否使用全量数据（泄漏）。"""
    out = df.copy()
    out["Weekly_Sales_log"] = np.log1p(out[TARGET])
    out["is_peak_season"] = out["Date"].dt.month.isin([10, 11, 12]).astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["Date"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["Date"].dt.month / 12)
    out["sales_lag_1"] = out.groupby("Store")["Weekly_Sales_log"].shift(1)
    out["sales_lag_2"] = out.groupby("Store")["Weekly_Sales_log"].shift(2)
    med = out["Weekly_Sales_log"].median()
    out[["sales_lag_1", "sales_lag_2"]] = out[["sales_lag_1", "sales_lag_2"]].fillna(med)

    if full_encoding or train_mask is None:
        means = out.groupby("Store")["Weekly_Sales_log"].mean()
    else:
        means = out.loc[train_mask].groupby("Store")["Weekly_Sales_log"].mean()
    out["store_sales_encoded"] = out["Store"].map(means)

    cols = [
        "Store",
        "Holiday_Flag",
        "Temperature",
        "Fuel_Price",
        "CPI",
        "Unemployment",
        "is_peak_season",
        "month_sin",
        "month_cos",
        "sales_lag_1",
        "sales_lag_2",
        "store_sales_encoded",
    ]
    return out[cols]


def fit_eval(X_tr, y_tr, X_te, y_te) -> dict:
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0, n_jobs=1, verbosity=0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "r2": r2_score(y_te, pred),
        "rmse_log": root_mean_squared_error(y_te, pred),
        "mae_log": mean_absolute_error(y_te, pred),
    }


def main() -> None:
    set_global_seed(0, 1)
    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    y = np.log1p(df[TARGET]).to_numpy()

    n = len(df)
    rand_tr = np.zeros(n, dtype=bool)
    idx = np.arange(n)
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=0)
    rand_tr[tr_idx] = True

    dates = np.sort(df["Date"].unique())
    cutoff = dates[int(len(dates) * 0.8)]
    time_tr = (df["Date"] < cutoff).to_numpy()

    rows = []
    for split_name, tr_mask in [("随机切分(泄漏)", rand_tr), ("时间切分(正确)", time_tr)]:
        for enc_name, full_enc in [("全量目标编码(泄漏)", True), ("训练集目标编码(正确)", False)]:
            feats = build_features(df, None if full_enc else tr_mask, full_encoding=full_enc)
            X = feats.to_numpy(dtype=float)
            res = fit_eval(X[tr_mask], y[tr_mask], X[~tr_mask], y[~tr_mask])
            rows.append({"切分": split_name, "目标编码": enc_name, **res})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    logs_dir, _ = line_dirs(3)
    table.to_csv(logs_dir / "forum_leak_audit.csv", index=False)
    print(f"\n已保存 {logs_dir / 'forum_leak_audit.csv'}")


if __name__ == "__main__":
    main()
