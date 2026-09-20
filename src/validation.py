"""无泄漏的时间序列切分：滚动原点（扩展/滑动窗口）与最终留出。"""

from __future__ import annotations

import numpy as np


def rolling_origin_splits(
    n: int,
    n_splits: int = 5,
    test_size: int = 13,
    min_train: int = 52,
    gap: int = 0,
    expanding: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """按时间顺序生成滚动原点切分。

    Args:
        n: 样本（或时间点）总数。
        n_splits: 折数。
        test_size: 每折验证段长度。
        min_train: 训练段最小长度。
        gap: 训练与验证之间的间隔（避免边界泄漏）。
        expanding: True 为扩展窗口，False 为固定长度滑动窗口。

    Returns:
        (train_idx, val_idx) 列表，均为位置索引且 train 严格早于 val。
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        val_end = n - (n_splits - 1 - i) * test_size
        val_start = val_end - test_size
        train_end = val_start - gap
        if train_end < min_train:
            continue
        train_start = 0 if expanding else max(0, train_end - min_train)
        splits.append((np.arange(train_start, train_end), np.arange(val_start, val_end)))
    return splits


def final_holdout(n: int, test_size: int = 13, gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """末段留出：最后 test_size 个时间点作永久测试集。"""
    test_start = n - test_size
    train = np.arange(0, test_start - gap)
    test = np.arange(test_start, n)
    return train, test
