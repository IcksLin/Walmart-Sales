# Walmart 周销售额分析与预测

工程规划见 [`docs/engineering_plan.md`](docs/engineering_plan.md)。
术语/指标/缩写语义表见 [`docs/glossary.md`](docs/glossary.md)。

## 环境

```bash
pip install -r requirements.txt
```

## 数据

- `practice_1.csv`：原始数据（45 店 × 143 周）
- `practice_1.xlsx`：表格副本（含审计结果多 sheet）

## 运行

```bash
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

# 论坛方法泄漏审计
python src/forum_leak_audit.py

# 模型准确率排名横向柱状图
python src/ranking_chart.py
```

种子统一在 `configs/default.yaml` 的 `seed`（默认 42）。

## 技术报告

`report/` 为 LaTeX 技术报告（XeLaTeX + ctex），已生成 `report/report.pdf`：

```bash
cd report && make      # 重新编译 report.pdf
```

## Kaggle Notebook

`notebooks/walmart_best_model_kaggle.ipynb` 是**最优模型 B_cluster（DTW + 层次聚类 + XGBoost）**的自包含 Notebook：

- 数据集定位：Kaggle 挂载 `/kaggle/input/**` → `kagglehub.dataset_download("mikhail1681/walmart-sales")` → 本地 `practice_1.csv` 逐级回退；
- 流程：特征工程 → DTW 距离矩阵 + 自动定簇 → 簇内 XGBoost → 滚动 CV + 末段留出 → 图表 + 特征重要性 → 可选 `submission.csv`；
- 上传：Kaggle → **Create → Notebook** → 粘贴内容 / 上传该 `.ipynb` → **Add Data** 挂载 `mikhail1681/walmart-sales` → **Save & Run All (Commit)**。

## C++ 加速模块

```bash
cd cpp
cmake -B build -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd ..
python src/bench_cpp.py   # DTW ≈1000x+, Ridge 超参搜索 ≈10x+
```

## 输出结构

所有交付产物写入 `output/`，排行图等汇总文件与各 line 同级平铺：

```
output/
├── model_ranking.png / model_ranking_wmae.png / model_ranking.csv   # 汇总（平铺）
├── cpp_benchmark.csv
├── line_1/                         # 数据探索
│   ├── logs/                       #   log、相关系数矩阵
│   └── pictures/                   #   变量分布图、相关热力图
├── line_2/                         # 训练
│   ├── logs/                       #   train/three_tier 日志与指标
│   └── pictures/                   #   loss 曲线、DTW 聚类图
├── line_3/                         # 测试评估
│   ├── logs/                       #   分层指标、残差诊断、泄漏审计
│   └── pictures/                   #   预测对比、残差分布、门店 WAPE
└── line_4/                         # 模型解释
    ├── logs/                       #   回归系数、VIF、SHAP、线性权重
    └── pictures/                   #   线性特征权重、SHAP 图
```

## 架构

```
configs/default.yaml
   │
load_data → features.build → validation.splits → model.fit/predict
   │                                              │
   └──────── output/line_X ◄──── metrics.evaluate
```

| 模块 | 职责 |
|---|---|
| `src/config.py` | 配置加载与快照 |
| `src/seed.py` | 随机种子与环境快照 |
| `src/output_paths.py` | 统一 output/line_X 目录 |
| `src/logging_utils.py` | 分级双通道日志 |
| `src/metrics.py` | 原尺度指标 + Duan smearing |
| `src/validation.py` | 滚动原点切分 / 末段留出 |
| `src/features.py` | 时间/滞后滚动特征与设计矩阵 |
| `src/train.py` | 基线流水线 |
| `src/three_tier.py` | DTW 三层建模 |
| `src/interpret.py` | 固定效应回归 + SHAP |
| `src/stratified_eval.py` | 分层评估与诊断 |
| `src/visualize.py` | 可视化工作流 |
| `src/ranking_chart.py` | 模型排名图 |
| `src/dtw.py` | DTW 距离与聚类 |
| `src/plot_style.py` | 开放式坐标轴样式 |
| `src/data_audit.py` | 数据审计 |

## 可复现性

- 种子单一来源：`configs/default.yaml` 的 `seed`
- 线程锁定：`threads: 1` 保证 BLAS 确定性
- 每个 line 的 `logs/` 保存配置快照、环境版本、日志、指标与预测
