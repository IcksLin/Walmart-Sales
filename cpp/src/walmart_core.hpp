#pragma once

#include <cstddef>
#include <vector>

namespace wc {

// 计算 N 条长度为 T 的序列的两两 DTW 距离矩阵（行主序 N*N，对称，对角 0）。
// band < 0 表示无约束（全带宽）。
std::vector<double> dtw_distance_matrix(const std::vector<double>& series, int n, int t, int band);

// Ridge 超参搜索：对每个 alpha 在给定折上拟合闭式解，返回各 alpha 的 WMAE/WAPE。
struct RidgeSearchResult {
    std::vector<double> alphas;
    std::vector<double> wmae;
    std::vector<double> wape;
    int best_index = -1;
};

// X: n*p 行主序；y: n；weights: n；folds: 每折的训练/验证下标。
RidgeSearchResult ridge_grid_search(
    const std::vector<double>& X,
    const std::vector<double>& y,
    int n,
    int p,
    const std::vector<std::vector<int>>& train_idx,
    const std::vector<std::vector<int>>& val_idx,
    const std::vector<double>& weights,
    const std::vector<double>& alphas);

}  // namespace wc
