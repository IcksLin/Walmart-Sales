# 术语、指标与缩写语义表

本表汇总项目中出现的一切指标、参数与首字母缩写。`P^2` 按上下文理解为 **R²（决定系数）**，另外也列出 **p 值** 以区分。

## 1. 评价指标

> **值偏向**列说明指标“越大越好 / 越小越好 / 无好坏”。

| 缩写/符号 | 全称 | 中文 | 公式 / 含义 | 值偏向 | 本项目用途 |
|---|---|---|---|---|---|
| **MAE** | Mean Absolute Error | 平均绝对误差 | `mean(|y − ŷ|)` | **越小越好**（下限 0） | 原尺度绝对误差（金额） |
| **MSE** | Mean Squared Error | 均方误差 | `mean((y − ŷ)²)` | **越小越好**（下限 0） | 惩罚大误差 |
| **RMSE** | Root Mean Squared Error | 均方根误差 | `sqrt(MSE)` | **越小越好**（下限 0） | 主报告指标之一，与原值同量纲 |
| **MAPE** | Mean Absolute Percentage Error | 平均绝对百分比误差 | `mean(|y−ŷ|/|y|)` | **越小越好**（≥0%） | 对接近 0 的值不稳定，本项目少用 |
| **sMAPE** | symmetric MAPE | 对称百分比误差 | `mean(2|y−ŷ|/(|y|+|ŷ|))` | **越小越好**（0~200%） | 抗除零，辅助 |
| **WAPE / WMAPE** | Weighted Absolute Percentage Error | 加权/整体绝对百分比误差 | `Σ|y−ŷ| / Σ|y|` | **越小越好**（≥0%） | 跨店可比，避免大店主导 |
| **WMAE** | Weighted MAE | 加权平均绝对误差 | `Σwᵢ|y−ŷ| / Σwᵢ`，节假日 `w=5` | **越小越好** | **主指标**，Walmart 业务口径 |
| **MASE** | Mean Absolute Scaled Error | 平均绝对缩放误差 | `MAE / MAE_naive` | **越小越好**；`<1` 优于朴素，`=1` 持平 | 相对季节朴素 |
| **R²**（即 P²） | Coefficient of Determination | 决定系数 | `1 − SS_res/SS_tot` | **越大越好**（可负，上限 1） | 解释力；本数据天然很高 |
| **Adj-R²** | Adjusted R² | 调整决定系数 | 对特征数惩罚后的 R² | **越大越好** | 比较不同特征数模型 |
| **p 值** | p-value | 显著性概率 | 原假设成立时观测更极端的概率 | **越小越好**（`<0.05` 显著） | 系数/检验显著性 |
| **t 值 / z 值** | t-statistic | 检验统计量 | `系数/标准误` | **绝对值越大越好** | 系数显著性 |
| **CI** | Confidence Interval | 置信区间 | 通常 95% | **不含 0 越好**（无大小概念） | 系数的区间估计 |

## 2. 统计与检验

| 缩写/符号 | 全称 | 中文 | 含义 | 值偏向 | 本项目用途 |
|---|---|---|---|---|---|
| **VIF** | Variance Inflation Factor | 方差膨胀因子 | `1/(1−R²ⱼ)` | **越小越好**（`<5` 良好，`<10` 可接受） | 共线性诊断 |
| **OLS** | Ordinary Least Squares | 普通最小二乘 | 最小化残差平方和 | 方法（无偏向） | 回归基线 |
| **HC1** | Heteroskedasticity-Consistent 1 | 异方差稳健标准误 | 稳健协方差估计 | 方法（无偏向） | 回归推断 |
| **ACF** | Autocorrelation Function | 自相关函数 | 序列与自身滞后的相关 | 诊断（`|值|`大即强自相关） | 时序诊断 |
| **PACF** | Partial ACF | 偏自相关函数 | 剔除中间滞后后的相关 | 诊断 | 定滞后阶数 |
| **Ljung-Box** | Ljung-Box test | 自相关检验 | 残差是否白噪声 | **p 越大越好**（`>0.05` 不拒绝白噪声） | 模型充分性 |
| **Breusch-Pagan** | Breusch-Pagan test | 异方差检验 | 残差方差是否恒定 | **p 越大越好**（不拒绝同方差） | 回归诊断 |
| **ADF** | Augmented Dickey-Fuller | 单位根检验 | 序列平稳性 | **统计量越负 / p 越小越平稳** | 非平稳判别 |
| **相关系数 r** | Pearson/Spearman | 相关 | `cov/(σσ)` | **`|r|` 越大越强**（符号表方向） | 单因素关系 |
| **FE** | Fixed Effects | 固定效应 | 个体专属截距 | 方法（无偏向） | 吸收门店固有差异 |
| **Pooling** | Pooled model | 池化模型 | 多主体混合建模 | 方法（无偏向） | 全局基线 |
| **DTW** | Dynamic Time Warping | 动态时间规整 | 允许时间轴伸缩的序列距离 | **距离越小越相似** | 门店聚类 |
| **Sakoe-Chiba band** | — | 约束带 | DTW 对齐的带状约束 | 参数（越大越慢/越准） | 加速 DTW |
| **WCSS** | Within-Cluster Sum of Squares | 簇内平方和 | 聚类紧致度 | **越小越好** | 肘部法定簇数 |
| **Silhouette** | Silhouette Coefficient | 轮廓系数 | `(b−a)/max(a,b) ∈ [−1,1]` | **越大越好** | 聚类质量/定簇数 |
| **Elbow** | Elbow Method | 肘部法 | WCSS 曲线拐点 | 方法（无偏向） | 定簇数 |
| **PDP** | Partial Dependence Plot | 部分依赖图 | 单特征对预测的平均边际效应 | 诊断（看形状/方向） | 可解释性 |
| **ICE** | Individual Conditional Expectation | 个体条件期望 | 每个样本的 PDP | 诊断 | 交互/异质性 |
| **SHAP** | SHapley Additive exPlanations | 沙普利加性解释 | 博弈论贡献分解 | **`|SHAP|` 越大越重要**（符号表方向） | 特征重要性/方向 |

## 3. 模型与算法

| 缩写 | 全称 | 中文 | 要点 |
|---|---|---|---|
| **Ridge** | Ridge Regression | 岭回归 | L2 惩罚，抗共线，单店首选 |
| **Lasso** | Least Absolute Shrinkage and Selection Operator | 套索回归 | L1 惩罚，可置零特征 |
| **ElasticNet** | Elastic Net | 弹性网 | L1+L2 混合 |
| **RF** | Random Forest | 随机森林 | Bagging 多树取平均 |
| **GBDT** | Gradient Boosting Decision Tree | 梯度提升决策树 | 串行拟合残差 |
| **LightGBM** | Light Gradient Boosting Machine | — | 直方图分箱 + Leaf-wise + GOSS + EFB |
| **XGBoost** | Extreme Gradient Boosting | — | 正则化 GBDT，本项目最优引擎 |
| **GOSS** | Gradient-based One-Side Sampling | 单边梯度采样 | 保留大梯度样本 |
| **EFB** | Exclusive Feature Bundling | 互斥特征捆绑 | 稀疏降维 |
| **Leaf-wise** | Leaf-wise tree growth | 叶子优先生长 | 同叶子数损失更低，需限深 |
| **Histogram** | Histogram binning | 直方图分箱 | 分裂搜索 O(样本)→O(桶) |
| **SMAPE** | — | 见指标表 | — |

## 4. 验证与工程

| 缩写 | 全称 | 中文 | 含义 |
|---|---|---|---|
| **EDA** | Exploratory Data Analysis | 探索性数据分析 | 建模前数据探索 |
| **CV** | Cross-Validation | 交叉验证 | 轮换训练/验证 |
| **K-fold** | K-fold CV | K 折交叉验证 | 分 K 份轮流验证 |
| **TimeSeriesSplit** | — | 时序切分 | 训练只含过去的滚动切分 |
| **Rolling origin** | Rolling-origin CV | 滚动原点验证 | 验证段随时间前推 |
| **Expanding window** | — | 扩展窗口 | 训练集递增 |
| **Sliding window** | — | 滑动窗口 | 训练窗口定长滑动 |
| **Holdout** | Hold-out set | 留出集 | 全程只碰一次的测试集 |
| **Nested CV** | Nested CV | 嵌套交叉验证 | 外层评估+内层调参 |
| **Leakage** | Data leakage | 数据泄漏 | 未来/测试信息进入训练 |
| **OOD** | Out-of-Distribution | 分布外 | 测试分布偏离训练 |
| **IQR** | Interquartile Range | 四分位距 | `Q3−Q1`，异常判定 |
| **Winsorize** | Winsorization | 缩尾处理 | 截断极端值 |
| **Target encoding** | — | 目标编码 | 用目标均值编码类别（易泄漏） |
| **One-hot** | One-hot encoding | 独热编码 | 类别转 0/1 哑变量 |
| **log1p / expm1** | log(1+x) / exp(x)−1 | — | 处理右偏与还原 |
| **Duan smearing** | Duan smearing estimator | 杜安涂抹 | 对数模型还原的偏差校正 |
| **seed** | Random seed | 随机种子 | 复现实验 |
| **CLI** | Command-Line Interface | 命令行接口 | 脚本入口 |
| **API** | Application Programming Interface | 应用编程接口 | 模块接口 |

## 5. C++ 加速工具链（P3）

| 缩写 | 全称 | 中文 | 含义 |
|---|---|---|---|
| **C++** | C++17 | — | 系统级高性能语言 |
| **pybind11** | — | — | 把 C++ 暴露给 Python 的绑定库 |
| **OpenMP** | Open Multi-Processing | 共享内存并行 | `#pragma omp parallel for` |
| **CMake** | — | — | 跨平台构建工具 |
| **MDS** | Multidimensional Scaling | 多维缩放 | 距离矩阵降维可视化 |
| **O(N²)** | — | 平方复杂度 | DTW 全矩阵的复杂度 |

## 6. 易混澄清

- **R² ≠ 准确率**：R² 是回归解释力（可负），分类才有“准确率”。论坛所谓“98% 准确率”实为 R²。
- **WAPE vs MAPE**：WAPE 是整体加权（分母 Σ|y|），MAPE 是逐点平均，前者更稳健。
- **WMAE vs WAPE**：WMAE 带节假日权重（×5），WAPE 不带。
- **RMSE vs MAE**：RMSE 对大误差更敏感；MAE 更稳健。
- **MASE < 1** 才表示优于季节朴素；`=1` 表示与朴素持平。
- **池化 vs 固定效应**：池化共享全部参数；固定效应额外给每个个体一个截距。

## 7. 值偏向速查

| 越大越好 | 越小越好 | 视情况（符号/阈值） |
|---|---|---|
| R²、Adj-R²、Silhouette、SHAP 绝对值、t 值绝对值、`|r|` | MAE、MSE、RMSE、MAPE、sMAPE、WAPE、WMAE、MASE、WCSS、VIF | p 值（越小越显著）、CI（不含 0）、Ljung-Box/Breusch-Pagan 的 p（越大越“正常”）、DTW（越小越相似） |

> 记忆法：**误差类全越小越好**；**解释力/聚类质量类越大越好**；**检验类看 p 与阈值**。
