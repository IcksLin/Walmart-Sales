"""统一输出目录结构。

output/
├── <排行图等汇总文件平铺>          # 与各 line 同级
├── line_1/{logs,pictures}          # 数据分布与相关性
├── line_2/{logs,pictures}          # 训练过程（loss）
├── line_3/{logs,pictures}          # 测试集评估（预测/残差）
└── line_4/{logs,pictures}          # 模型解释（线性权重/SHAP）
"""

from __future__ import annotations

from pathlib import Path

OUTPUT_ROOT = Path("output")
LINES = {1: "data_profile", 2: "training", 3: "evaluation", 4: "interpretation"}


def root() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


def line_dirs(line: int) -> tuple[Path, Path]:
    logs = OUTPUT_ROOT / f"line_{line}" / "logs"
    pictures = OUTPUT_ROOT / f"line_{line}" / "pictures"
    logs.mkdir(parents=True, exist_ok=True)
    pictures.mkdir(parents=True, exist_ok=True)
    return logs, pictures
