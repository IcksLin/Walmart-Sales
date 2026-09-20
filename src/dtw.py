"""DTW 动态时间规整距离矩阵与门店聚类。

纯 Python/NumPy 参考实现；P3 将用 C++ + OpenMP 替换本模块的 compute_dtw_matrix。
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int | None = None) -> float:
    """两条序列的 DTW 距离（Sakoe-Chiba band 可选）。"""
    n, m = len(a), len(b)
    if band is None:
        band = max(n, m)
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf)
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return float(prev[m])


def compute_dtw_matrix(series: np.ndarray, band: int | None = None) -> np.ndarray:
    """计算 N 条序列的两两 DTW 距离矩阵（对称，对角 0）。

    Args:
        series: 形状 (N, T)。
    """
    n = series.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(series[i], series[j], band=band)
            dist[i, j] = dist[j, i] = d
    return dist


def _elbow_k(ks: list[int], wcss: list[float]) -> int:
    """WCSS 曲线到首末连线的最大距离点作为肘点。"""
    x = np.asarray(ks, dtype=float)
    y = np.asarray(wcss, dtype=float)
    x = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    dist = np.abs((y[-1] - y[0]) * x - (x[-1] - x[0]) * y + x[-1] * y[0] - y[-1] * x[0])
    return int(ks[int(np.argmax(dist))])


def select_n_clusters(
    dist: np.ndarray, k_min: int = 2, k_max: int = 10, min_cluster_size: int = 5
) -> tuple[int, dict, int]:
    """肘部法(WCSS) + 轮廓系数双判据自动定簇数。

    min_cluster_size 用于排除把离群店单独成簇的退化解。
    """
    scores: dict[int, dict[str, float]] = {}
    for k in range(k_min, k_max + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="precomputed", linkage="average"
        ).fit_predict(dist)
        if np.bincount(labels).min() < min_cluster_size:
            continue
        sil = silhouette_score(dist, labels, metric="precomputed")
        wcss = 0.0
        for c in np.unique(labels):
            members = np.flatnonzero(labels == c)
            if len(members) > 1:
                sub = dist[np.ix_(members, members)]
                wcss += float(sub[np.triu_indices(len(members), k=1)].mean() * len(members))
        scores[k] = {"silhouette": float(sil), "wcss": wcss}
    if not scores:
        raise ValueError("无满足 min_cluster_size 的簇数，请放宽约束或调整距离")
    best_k = max(scores, key=lambda k: scores[k]["silhouette"])
    elbow = _elbow_k(sorted(scores), [scores[k]["wcss"] for k in sorted(scores)])
    return best_k, scores, elbow


def cluster_stores(dist: np.ndarray, n_clusters: int) -> np.ndarray:
    return AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    ).fit_predict(dist)
