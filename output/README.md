# output 产物说明

本目录汇总全部交付产物。`output/` 根为汇总文件，各 `line_X/` 为该分析线的图片与日志。

## 各线技术手段

| 线 | 主题 | 使用到的技术 / 算法 | 主要产物 |
|---|---|---|---|
| `line_1` | 数据探索 | 直方图分布、`log1p` 变换、Pearson 相关热力图 | `pictures/variable_distributions.png`、`pictures/correlation_heatmap.png`、`logs/correlation_matrix.csv` |
| `line_2` | 训练 | 滚动原点 CV、`log1p` 目标、Duan smearing 还原、WMAE/WAPE/MASE、Wilcoxon 显著性检验；模型：季节朴素、门店均值、Ridge（全局/单店）、LightGBM、XGBoost；**DTW + 层次聚类（轮廓系数+肘部法）+ XGBoost 三层建模**；损失曲线 | `logs/train_metrics.csv`、`logs/three_tier_metrics.csv`、`logs/dtw_clusters.csv`、`pictures/loss_curve_*.png`、`pictures/store_dtw_heatmap.png` |
| `line_3` | 测试评估 | 末段留出、分层评估（年份/节假日/温度/门店）、残差诊断（Ljung-Box、Breusch-Pagan）、数据泄漏审计、预测-真实对比、残差分布；主模型 **B_cluster: DTW+XGBoost** | `logs/stratified_*.csv`、`logs/residual_diagnostics.csv`、`logs/store_diagnostics.csv`、`logs/forum_leak_audit.csv`、`pictures/pred_vs_actual.png`、`pictures/residual_distribution.png` |
| `line_4` | 模型解释 | 固定效应 OLS + HC1 稳健标准误、VIF 共线性诊断、SHAP（TreeExplainer）、标准化 Ridge 特征权重 | `logs/regression_coefficients.csv`、`logs/vif.csv`、`logs/shap_importance.csv`、`pictures/shap_*.png`、`pictures/linear_feature_weights.png` |

## 关键模型对照

| 名称 | 技术手段 |
|---|---|
| `seasonal_naive` | 季节朴素（去年同周） |
| `store_mean` | 门店均值 |
| `ridge_global` / `ridge_store` | Ridge 回归（全局门店固定效应 / 单店） |
| `lightgbm` / `xgboost` | 梯度提升树（全局） |
| `A_global` | 全局池化 + 门店固定效应 + XGBoost |
| `B_cluster` | **DTW + 层次聚类 + XGBoost（最优）** |
| `C_store` | 单店 XGBoost |

## 根目录汇总文件

| 文件 | 技术手段 |
|---|---|
| `model_ranking.png` / `model_ranking_wmae.png` / `model_ranking.csv` | 滚动 CV 下 WMAE/WAPE/RMSE/R² 排名（开放式坐标轴横向柱状图） |
| `cpp_benchmark.csv` | C++ 加速对照：DTW 距离矩阵（OpenMP）、Ridge 闭式解超参搜索 |

## 复现

```bash
python src/visualize.py        # line_1 ~ line_4 图表
python src/train.py --holdout  # line_2 基线
python src/three_tier.py       # line_2 三层建模
python src/interpret.py        # line_4 解释
python src/stratified_eval.py  # line_3 分层评估
python src/ranking_chart.py    # 排名图
```

固定随机种子：`configs/default.yaml` 的 `seed: 42`。

## 最优模型与测试表现

**最优模型：`B_cluster` = DTW + 层次聚类（K=5）+ XGBoost**（簇内各训一个浅层 XGBoost）。

### 滚动 CV（5 折扩展窗口，总体）

| 指标 | 值 | 偏向 |
|---|---|---|
| WMAE | **58,500.2** | 越小越好 |
| WAPE | **0.0472** | 越小越好 |
| RMSE | 77,775.2 | 越小越好 |
| R² | **0.9812** | 越大越好 |

对比：A_global 61,087 / LightGBM 62,580 / 季节朴素 62,770 / C_store 68,138，B_cluster 全面最优。

### 末段留出终验（最后 13 周，无泄漏）

| 指标 | 值 |
|---|---|
| WMAE | **35,977.4** |
| WAPE | **0.0329（≈3.3%）** |
| MAE | 33,842.4 |
| RMSE | 49,143.6 |
| R² | **0.9913** |
| MASE | 0.6417（<1，优于季节朴素） |

门店级 WAPE：中位数 **0.0317**、P90 0.0476、最差门店 **0.0692**——即使最差门店误差也不足 7%。

### 结论

- B_cluster 在滚动 CV 与末段留出上均为最优，且**最差门店表现最好**；
- 相对强基线季节朴素，WMAE 降低约 6.8%（CV）/ 35%（留出）；
- 单店建模（C_store）因每店仅 143 样本而过拟合，整体最差；
- 主要短板为**节假日周**（WAPE 偏高，见 `line_3/logs/stratified_by_holiday.csv`）。

## Kaggle Notebook

最优模型已导出为自包含 Notebook：`../notebooks/walmart_best_model_kaggle.ipynb`。
在 Kaggle 上先 **Add Data → `mikhail1681/walmart-sales`** 挂载数据集，再 **Save & Run All** 即可复现上述结果。
