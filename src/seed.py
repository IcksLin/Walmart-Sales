"""全局随机种子与确定性设置，保证实验可复现。"""

from __future__ import annotations

import os
import platform
import random
import sys
from importlib import metadata


def set_global_seed(seed: int = 42, threads: int = 1) -> None:
    """固定 Python / NumPy 随机性，并锁定线程数以获得确定结果。

    Args:
        seed: 全局种子。
        threads: BLAS/OpenMP 线程数；设为 1 可最大化确定性。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if threads:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[var] = str(threads)

    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def env_snapshot(seed: int | None = None) -> dict:
    """采集运行环境信息，供 runs/<id>/env.json 落盘。"""
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "lightgbm",
        "shap",
    ]
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "seed": seed,
        "packages": versions,
    }
