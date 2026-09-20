"""统一的图表风格：开放式笛卡尔坐标轴（无四边框）。"""

from __future__ import annotations

from matplotlib.ticker import FuncFormatter, MaxNLocator


def open_axes(ax, grid_axis: str | None = "x") -> None:
    """隐藏上/右边框，保留左/下轴，形成开放坐标系。grid_axis=None 关闭网格。"""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#888888")
        ax.spines[side].set_linewidth(0.8)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)


def auto_formatter(max_value: float) -> FuncFormatter:
    """按数量级自动选择刻度格式（大数加千分位，小数保留 2~3 位）。"""
    if max_value >= 1000:
        return FuncFormatter(lambda x, _: f"{x:,.0f}")
    if max_value >= 1:
        return FuncFormatter(lambda x, _: f"{x:.2f}")
    return FuncFormatter(lambda x, _: f"{x:.2f}")


def tidy_ticks(ax, max_value: float, nbins: int = 6) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))
    ax.xaxis.set_major_formatter(auto_formatter(max_value))
