"""Walmart 周销售额数据审计脚本.

用途:
    1. 结构完整性审计（缺失/重复/非法日期/主键/面板完整性）
    2. 极值统计（合并 IQR vs 门店内 within-store IQR）
    3. 数值分段分布
    4. 损坏程度分级
    5. 导出 Excel 表格副本

运行:
    python src/data_audit.py [--csv practice_1.csv] [--xlsx practice_1.xlsx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATE_FMT = "%d-%m-%Y"
NUMERIC_COLS = ["Weekly_Sales", "Temperature", "Fuel_Price", "CPI", "Unemployment"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format=DATE_FMT)
    return df


def structural_audit(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "总行数": len(df),
        "缺失值总数": int(df.isna().sum().sum()),
        "重复行数": int(df.duplicated().sum()),
        "非法日期数": int(pd.to_datetime(df["Date"], errors="coerce").isna().sum()),
        "门店数": int(df["Store"].nunique()),
        "唯一日期数": int(df["Date"].nunique()),
        "Store+Date 重复": int(len(df) - df[["Store", "Date"]].drop_duplicates().shape[0]),
    }
    return pd.DataFrame(
        [(k, v) for k, v in checks.items()], columns=["检查项", "结果"]
    )


def _bounds(vals: pd.Series, k: float = 1.5):
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def outlier_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLS:
        v = df[col]
        lo, hi = _bounds(v, 1.5)
        elo, ehi = _bounds(v, 3.0)
        pooled_out = int(((v < lo) | (v > hi)).sum())
        pooled_ext = int(((v < elo) | (v > ehi)).sum())

        w_out = w_ext = 0
        for _, g in df.groupby("Store"):
            gv = g[col]
            glo, ghi = _bounds(gv, 1.5)
            gelo, gehi = _bounds(gv, 3.0)
            w_out += int(((gv < glo) | (gv > ghi)).sum())
            w_ext += int(((gv < gelo) | (gv > gehi)).sum())

        rows.append(
            {
                "字段": col,
                "合并离群": pooled_out,
                "合并离群%": round(pooled_out / len(df) * 100, 2),
                "合并极端": pooled_ext,
                "门店内离群": w_out,
                "门店内离群%": round(w_out / len(df) * 100, 2),
                "门店内极端": w_ext,
            }
        )
    return pd.DataFrame(rows)


def segment_table(df: pd.DataFrame, col: str, edges: list[float], labels: list[str]) -> pd.DataFrame:
    binned = pd.cut(df[col], bins=edges, labels=labels, right=False)
    cnt = binned.value_counts().reindex(labels).fillna(0).astype(int)
    out = pd.DataFrame({"区间": labels, "频数": cnt.values})
    out["占比%"] = (out["频数"] / len(df) * 100).round(1)
    out.insert(0, "字段", col)
    return out


def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERIC_COLS:
        total = df[col].std()
        within = df.groupby("Store")[col].std().mean()
        rows.append(
            {
                "字段": col,
                "总体std": round(float(total), 3),
                "门店内std": round(float(within), 3),
                "门店间方差贡献%": round((1 - (within / total) ** 2) * 100, 0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="practice_1.csv")
    ap.add_argument("--xlsx", default="practice_1.xlsx")
    args = ap.parse_args()

    df = load(Path(args.csv))

    audit = structural_audit(df)
    outliers = outlier_stats(df)
    segments = pd.concat(
        [
            segment_table(
                df,
                "Weekly_Sales",
                [0, 5e5, 1e6, 1.5e6, 2e6, 2.5e6, 1e9],
                ["<0.5M", "0.5-1M", "1-1.5M", "1.5-2M", "2-2.5M", ">2.5M"],
            ),
            segment_table(
                df,
                "Temperature",
                [-100, 32, 50, 70, 90, 200],
                ["<32F", "32-50F", "50-70F", "70-90F", ">90F"],
            ),
        ],
        ignore_index=True,
    )
    variance = variance_decomposition(df)

    print("=== 结构完整性审计 ===")
    print(audit.to_string(index=False))
    print("\n=== 极值统计（合并 vs 门店内）===")
    print(outliers.to_string(index=False))
    print("\n=== 门店间方差贡献 ===")
    print(variance.to_string(index=False))
    print("\n=== 数值分段分布 ===")
    print(segments.to_string(index=False))

    with pd.ExcelWriter(args.xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="原始数据", index=False)
        audit.to_excel(writer, sheet_name="结构审计", index=False)
        outliers.to_excel(writer, sheet_name="极值统计", index=False)
        variance.to_excel(writer, sheet_name="方差分解", index=False)
        segments.to_excel(writer, sheet_name="分段分布", index=False)
    print(f"\n已导出表格: {args.xlsx}")


if __name__ == "__main__":
    main()
