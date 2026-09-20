"""日志与输出目录管理。

所有产物写入 output/：
- output/line_X/logs/     日志与表格
- output/line_X/pictures/ 图片
- output/                 排行图等汇总文件（与各 line 同级平铺）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from output_paths import line_dirs

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _build_logger(name: str, log_file: Path, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def get_line_logger(name: str, line: int, level: int = logging.INFO) -> logging.Logger:
    """日志写入 output/line_<line>/logs/<name>.log，同时输出到控制台。"""
    logs_dir, _ = line_dirs(line)
    return _build_logger(name, logs_dir / f"{name}.log", level)


def get_logger(name: str, run_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    """兼容旧接口：日志写入指定目录。"""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return _build_logger(name, Path(run_dir) / "log.txt", level)
