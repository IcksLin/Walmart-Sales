"""C++ 加速模块基准：DTW 距离矩阵 + Ridge 超参搜索，与 Python 对照验证。

运行: python src/bench_cpp.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))

from config import load_config  # noqa: E402
from dtw import compute_dtw_matrix  # noqa: E402
from features import build_design, history_feature_names  # noqa: E402
from output_paths import root  # noqa: E402
from train import date_masks, load_data  # noqa: E402
from validation import rolling_origin_splits  # noqa: E402

import walmart_cpp  # noqa: E402


def python_ridge_search(X, y, fold_idx, weights, alphas):
    wmae, wape = [], []
    for alpha in alphas:
        werr = wsum = err = ysum = 0.0
        for tr, va in fold_idx:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[va])
            e = np.abs(y[va] - pred)
            werr += float((weights[va] * e).sum())
            wsum += float(weights[va].sum())
            err += float(e.sum())
            ysum += float(np.abs(y[va]).sum())
        wmae.append(werr / wsum)
        wape.append(err / ysum)
    return np.array(wmae), np.array(wape)


def main() -> None:
    cfg = load_config("configs/default.yaml")
    df = load_data(cfg)
    target = cfg["data"]["target"]
    store_col = cfg["data"]["store_col"]
    holiday_col = cfg["data"]["holiday_col"]
    date_col = cfg["data"]["date_col"]

    # DTW 对照
    stores, series = df.pivot(index=store_col, columns=date_col, values=target).pipe(
        lambda p: (p.index.to_numpy(), np.log1p(p.to_numpy(float)))
    )
    t0 = time.time()
    dist_py = compute_dtw_matrix(series)
    t_py = time.time() - t0
    t0 = time.time()
    dist_cpp = walmart_cpp.dtw_distance_matrix(np.ascontiguousarray(series), -1, 0)
    t_cpp = time.time() - t0

    # Ridge 超参搜索对照
    history_cols = history_feature_names(cfg["features"]["lag_features"], cfg["features"]["rolling_windows"])
    design = build_design(
        df, cfg["features"]["exogenous"], cfg["features"]["time_features"], history_cols, store_col, True
    )
    design = design.fillna(design.median(numeric_only=True))
    design.insert(0, "const", 1.0)
    X = np.ascontiguousarray(design.to_numpy(float))
    y = np.log1p(df[target].to_numpy(float))
    weights = np.where(df[holiday_col].to_numpy() == 1, 5.0, 1.0)

    dates = np.sort(df[date_col].unique())
    v = cfg["validation"]
    splits = rolling_origin_splits(len(dates), v["n_splits"], v["test_size"], v["min_train"], v["gap"], v["expanding"])
    tr_list, va_list = [], []
    for tr_pos, va_pos in splits:
        tr, va = date_masks(df, dates, tr_pos, va_pos, cfg)
        tr_list.append(tr.astype(np.int64))
        va_list.append(va.astype(np.int64))

    alphas = np.logspace(-3, 3, 25)
    t0 = time.time()
    py_wmae, py_wape = python_ridge_search(X, y, list(zip(tr_list, va_list)), weights, alphas)
    t_py2 = time.time() - t0
    t0 = time.time()
    res = walmart_cpp.ridge_grid_search(X, y, tr_list, va_list, weights, alphas, 0)
    t_cpp2 = time.time() - t0

    cpp_wmae = np.array(res["wmae"])
    cpp_wape = np.array(res["wape"])
    best_py = alphas[int(np.argmin(py_wmae))]

    print("=== DTW 距离矩阵 ===")
    print(f"Python {t_py:.3f}s | C++ {t_cpp:.4f}s | 加速 {t_py/t_cpp:.0f}x | 最大差异 {np.abs(dist_py-dist_cpp).max():.2e}")
    print("\n=== Ridge alpha 搜索 (25 alpha × 5 折) ===")
    print(f"Python {t_py2:.3f}s | C++ {t_cpp2:.4f}s | 加速 {t_py2/t_cpp2:.0f}x")
    print(f"WMAE 最大差异 {np.abs(py_wmae-cpp_wmae).max():.3e} | WAPE 最大差异 {np.abs(py_wape-cpp_wape).max():.3e}")
    print(f"最优 alpha: Python {best_py:.5f} | C++ {res['best_alpha']:.5f}")

    out = pd.DataFrame(
        {
            "alpha": res["alphas"],
            "cpp_wmae": cpp_wmae,
            "py_wmae": py_wmae,
            "cpp_wape": cpp_wape,
            "py_wape": py_wape,
        }
    )
    out_path = root() / "cpp_benchmark.csv"
    out.to_csv(out_path, index=False)
    print(f"\n已保存 {out_path}")


if __name__ == "__main__":
    main()
