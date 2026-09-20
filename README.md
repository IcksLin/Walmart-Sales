# Walmart 周销售额影响因素分析与预测

基于 Walmart 45 家门店、143 周（2010-02-05 ~ 2012-10-26）的周销售额数据，回答两个问题：
**哪些因素影响销售额**，以及**如何对未来周销售额做出稳健预测**。
分析流程无未来信息泄漏，模型以扩展窗口滚动交叉验证评估，主指标为 Walmart 官方加权绝对误差 WMAE（节假日周权重为 5）。

## 数据来源

- Kaggle 数据集：<https://www.kaggle.com/datasets/mikhail1681/walmart-sales>
- 本地文件：`practice_1.csv`（45 店 × 143 周，共 6435 行，无缺失），`practice_1.xlsx` 为其表格副本
- 字段：`Store`、`Date`、`Weekly_Sales`（目标）、`Holiday_Flag`、`Temperature`、`Fuel_Price`、`CPI`、`Unemployment`

## 主要结果

最优模型为 **B_cluster = DTW + 层次聚类 + XGBoost**：对门店销售序列计算 DTW 距离矩阵，
聚类为 5 组（轮廓系数自动定簇），每组训练一个浅层 XGBoost。

| 场景 | WMAE | WAPE | R² | MASE |
|---|---|---|---|---|
| 滚动 CV（汇总） | 58,500 | 0.0472 | 0.9812 | — |
| 末段留出（最后 13 周） | 35,977 | 0.0329 | 0.9913 | 0.6417 |

门店级 WAPE 中位数 0.0317、P90 0.0476、最差门店 0.0692。
影响因素方面，销售额首要由门店自身历史动量与规模决定（`lag1` 主导），
宏观变量统计显著但效应量小（节假日 +5.1%，失业率 −3.9%/pp）。

## 项目结构

```
practice_0/
├── configs/default.yaml     # 单一配置源（数据/特征/验证/模型/种子）
├── src/                     # 分析与建模代码
├── cpp/                     # pybind11 + OpenMP 加速模块（DTW、Ridge 超参搜索）
├── notebooks/               # Kaggle 自包含 Notebook
├── report/                  # LaTeX 技术报告与 PDF
├── docs/                    # 工程规划、术语表
├── output/                  # 全部交付产物（图/表/日志）
└── practice_1.csv           # 原始数据
```

## 快速开始

```bash
pip install -r requirements.txt

# 数据审计
python src/data_audit.py

# 基线流水线（滚动 CV + 末段留出）
python src/train.py --holdout

# 三层建模（DTW 簇 / 全局 / 单店）
python src/three_tier.py

# 解释型分析（固定效应回归 + SHAP）
python src/interpret.py

# 分层评估与极端门店诊断
python src/stratified_eval.py

# 可视化（分布/相关/loss/预测对比/残差/线性权重）
python src/visualize.py

# 论坛方法泄漏审计、模型排名图
python src/forum_leak_audit.py
python src/ranking_chart.py
```

随机种子统一在 `configs/default.yaml` 的 `seed`（默认 42）。

## 方法概览

1. **数据清洗**：结构审计确认无缺失与重复；按门店内判据识别异常，确认清洗的本质是去混淆（门店效应 + 时间趋势）而非纠错。
2. **特征工程**：目标 `log1p`；门店内滞后（1/2/4/52 周）与滚动均值/标准差（4/8/13 周）仅用过去值；日期派生月/周号/季度；门店固定效应。
3. **三层建模**：全局池化（1 个模型）、DTW 簇级（5 个模型）、单店（45 个模型）对比，簇级最优。
4. **解释分析**：固定效应回归（门店 + 年份固定效应、HC1 稳健标准误）、VIF 诊断、SHAP 与线性权重相互印证。
5. **评估**：扩展窗口滚动原点 CV + 末段留出，主指标 WMAE，辅以 WAPE/RMSE/R²/MASE，并做分年份/节假日/温度/门店的分层评估。

## 输出结构

所有产物写入 `output/`，汇总文件与各 line 同级平铺：

```
output/
├── model_ranking.png / model_ranking_wmae.png / model_ranking.csv   # 汇总（平铺）
├── cpp_benchmark.csv
├── line_1/{logs,pictures}   # 数据探索
├── line_2/{logs,pictures}   # 训练与聚类
├── line_3/{logs,pictures}   # 测试评估
└── line_4/{logs,pictures}   # 模型解释
```

`output/README.md` 说明了各 line 使用的技术手段与最优模型结果。

## 技术报告

`report/report.pdf` 为 LaTeX 技术报告（XeLaTeX + ctex），包含数据清洗、特征选择、模型选取、训练过程、
评价指标、可视化分析、最优结果与优缺点，并附术语表。

```bash
cd report && make      # 重新编译 report.pdf
```

## Kaggle Notebook

`notebooks/walmart_best_model_kaggle.ipynb` 是自包含的最优模型 Notebook，数据集定位按
`/kaggle/input/**` → `kagglehub.dataset_download("mikhail1681/walmart-sales")` → 本地文件逐级回退。
在 Kaggle 上先 **Add Data** 挂载 `mikhail1681/walmart-sales`，再 **Save & Run All** 即可复现。

## C++ 加速模块

```bash
cd cpp
cmake -B build -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd ..
python src/bench_cpp.py 
```

## 模块说明

| 模块 | 职责 |
|---|---|
| `src/config.py` | 配置加载与快照 |
| `src/seed.py` | 随机种子与环境快照 |
| `src/output_paths.py` | 统一 `output/line_X` 目录 |
| `src/logging_utils.py` | 分级双通道日志 |
| `src/metrics.py` | 原尺度指标 + Duan smearing |
| `src/validation.py` | 滚动原点切分 / 末段留出 |
| `src/features.py` | 时间/滞后滚动特征与设计矩阵 |
| `src/train.py` | 基线模型流水线 |
| `src/dtw.py` | DTW 距离矩阵与聚类 |
| `src/three_tier.py` | 三层建模与对比 |
| `src/interpret.py` | 固定效应回归 + SHAP |
| `src/stratified_eval.py` | 分层评估与极端门店诊断 |
| `src/visualize.py` | 可视化工作流 |
| `src/ranking_chart.py` | 模型排名图 |
| `src/plot_style.py` | 开放式坐标轴样式 |
| `src/forum_leak_audit.py` | 论坛方法泄漏审计 |
| `src/bench_cpp.py` | C++ 加速基准 |
| `src/data_audit.py` | 数据审计 |

## 文档

- 工程规划：[`docs/engineering_plan.md`](docs/engineering_plan.md)
- 术语与指标：[`docs/glossary.md`](docs/glossary.md)
- 技术报告：[`report/report.pdf`](report/report.pdf)

## 可复现性

- 种子单一来源：`configs/default.yaml` 的 `seed`
- 线程锁定：`threads: 1` 保证 BLAS 确定性
- 每个 line 的 `logs/` 保存配置快照、环境版本、日志、指标与预测
