"""模型预测准确率排名横向柱状图。

汇总所有模型/层级的滚动 CV 总体指标，按 WMAE 排名绘制横向柱状图：
横坐标为指标值，纵坐标为方法。

运行: python src/ranking_chart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from output_paths import root
from plot_style import open_axes, tidy_ticks

METRICS = [("wmae", "WMAE (lower better)"), ("wape", "WAPE (lower better)"), ("rmse", "RMSE (lower better)"), ("r2", "R² (higher better)")]
COLORS = {
    "B_cluster (DTW+XGBoost)": "#2ca02c",
    "A_global (XGBoost)": "#1f77b4",
    "C_store (XGBoost)": "#d62728",
    "lightgbm": "#8c564b",
    "ridge_global": "#17becf",
    "ridge_store": "#bcbd22",
    "store_mean": "#7f7f7f",
    "seasonal_naive": "#ff7f0e",
}


def collect_ranking() -> pd.DataFrame:
    frames = []

    train_metrics = Path("output/line_2/logs/train_metrics.csv")
    if train_metrics.exists():
        df = pd.read_csv(train_metrics)
        frames.append(df[df["fold"] == -1][["model", "wmae", "wape", "rmse", "r2"]].rename(columns={"model": "method"}))

    tier_metrics = Path("output/line_2/logs/three_tier_metrics.csv")
    if tier_metrics.exists():
        df = pd.read_csv(tier_metrics)
        df = df[df["fold"] == -1][["tier", "wmae", "wape", "rmse", "r2"]].rename(columns={"tier": "method"})
        frames.append(df)

    ranking = pd.concat(frames, ignore_index=True)
    ranking = ranking.drop_duplicates(subset="method", keep="first")
    ranking = ranking[ranking["method"] != "xgboost"]
    ranking["method"] = ranking["method"].replace(
        {
            "A_global": "A_global (XGBoost)",
            "B_cluster": "B_cluster (DTW+XGBoost)",
            "C_store": "C_store (XGBoost)",
        }
    )
    return ranking.sort_values("wmae").reset_index(drop=True)


def plot(ranking: pd.DataFrame, out_path: Path) -> None:
    methods = ranking["method"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    for ax, (col, title) in zip(axes.ravel(), METRICS):
        values = ranking[col].to_numpy()
        order = ranking[col].argsort()
        if col == "r2":
            order = order[::-1]
        y = np.arange(len(methods))
        colors = [COLORS.get(m, "#1f77b4") for m in ranking["method"]]
        ax.barh(y, values[order], color=[colors[i] for i in order], height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([methods[i] for i in order])
        ax.invert_yaxis()

        vmax = float(values.max())
        pad = vmax * 0.22
        ax.set_xlim(0, vmax + pad)
        open_axes(ax, grid_axis="x")
        tidy_ticks(ax, vmax + pad)

        ax.set_xlabel(title)
        ax.set_title(title)
        for pos, idx in enumerate(order):
            ax.text(
                values[idx] + pad * 0.04,
                pos,
                f"{values[idx]:,.4f}",
                va="center",
                ha="left",
                fontsize=8.5,
            )
    fig.suptitle("Model prediction accuracy ranking (rolling CV, sorted by WMAE)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_wmae(ranking: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    values = ranking["wmae"].to_numpy()
    colors = [COLORS.get(m, "#1f77b4") for m in ranking["method"]]
    ax.barh(np.arange(len(ranking)), values, color=colors, height=0.72)
    ax.set_yticks(np.arange(len(ranking)))
    ax.set_yticklabels(ranking["method"])
    ax.invert_yaxis()

    vmax = float(values.max())
    pad = vmax * 0.22
    ax.set_xlim(0, vmax + pad)
    open_axes(ax, grid_axis="x")
    tidy_ticks(ax, vmax + pad)

    ax.set_xlabel("WMAE (lower is better)")
    ax.set_title("Model ranking by WMAE (rolling CV)")
    for pos, v in enumerate(values):
        ax.text(v + pad * 0.04, pos, f"{v:,.0f}", va="center", ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    out = root()
    ranking = collect_ranking()
    ranking.to_csv(out / "model_ranking.csv", index=False)
    print(ranking.to_string(index=False))
    plot(ranking, out / "model_ranking.png")
    plot_single_wmae(ranking, out / "model_ranking_wmae.png")
    print(f"\n已保存 {out}/model_ranking.csv, model_ranking.png, model_ranking_wmae.png")


if __name__ == "__main__":
    main()
