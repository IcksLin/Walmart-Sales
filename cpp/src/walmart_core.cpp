#include "walmart_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace wc {

namespace {

double dtw_distance(const double* a, const double* b, int n, int m, int band) {
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<double> prev(m + 1, INF);
    std::vector<double> curr(m + 1, INF);
    prev[0] = 0.0;
    for (int i = 1; i <= n; ++i) {
        std::fill(curr.begin(), curr.end(), INF);
        int lo = 1;
        int hi = m;
        if (band >= 0) {
            lo = std::max(1, i - band);
            hi = std::min(m, i + band);
        }
        for (int j = lo; j <= hi; ++j) {
            double cost = std::fabs(a[i - 1] - b[j - 1]);
            double best = std::min({prev[j], curr[j - 1], prev[j - 1]});
            curr[j] = cost + best;
        }
        std::swap(prev, curr);
    }
    return prev[m];
}

// 高斯消元求解 A x = rhs（A 为 p*p 行主序，原地修改副本）。
void solve_linear(std::vector<double> A, std::vector<double> rhs, int p, std::vector<double>& x) {
    for (int col = 0; col < p; ++col) {
        int pivot = col;
        double best = std::fabs(A[col * p + col]);
        for (int r = col + 1; r < p; ++r) {
            double v = std::fabs(A[r * p + col]);
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (pivot != col) {
            for (int c = 0; c < p; ++c) std::swap(A[col * p + c], A[pivot * p + c]);
            std::swap(rhs[col], rhs[pivot]);
        }
        double diag = A[col * p + col];
        if (std::fabs(diag) < 1e-12) diag = (diag < 0 ? -1e-12 : 1e-12);
        for (int r = col + 1; r < p; ++r) {
            double factor = A[r * p + col] / diag;
            if (factor == 0.0) continue;
            for (int c = col; c < p; ++c) A[r * p + c] -= factor * A[col * p + c];
            rhs[r] -= factor * rhs[col];
        }
    }
    x.assign(p, 0.0);
    for (int r = p - 1; r >= 0; --r) {
        double sum = rhs[r];
        for (int c = r + 1; c < p; ++c) sum -= A[r * p + c] * x[c];
        x[r] = sum / A[r * p + r];
    }
}

}  // namespace

std::vector<double> dtw_distance_matrix(const std::vector<double>& series, int n, int t, int band) {
    std::vector<double> dist(static_cast<size_t>(n) * n, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < n; ++i) {
        const double* a = series.data() + static_cast<size_t>(i) * t;
        for (int j = i + 1; j < n; ++j) {
            const double* b = series.data() + static_cast<size_t>(j) * t;
            double d = dtw_distance(a, b, t, t, band);
            dist[static_cast<size_t>(i) * n + j] = d;
            dist[static_cast<size_t>(j) * n + i] = d;
        }
    }
    return dist;
}

RidgeSearchResult ridge_grid_search(
    const std::vector<double>& X,
    const std::vector<double>& y,
    int n,
    int p,
    const std::vector<std::vector<int>>& train_idx,
    const std::vector<std::vector<int>>& val_idx,
    const std::vector<double>& weights,
    const std::vector<double>& alphas) {
    (void)n;
    const int n_folds = static_cast<int>(train_idx.size());

    // 预计算每折的 Gram 矩阵与 X'y（与 alpha 无关）。
    std::vector<std::vector<double>> grams(n_folds, std::vector<double>(static_cast<size_t>(p) * p, 0.0));
    std::vector<std::vector<double>> xtys(n_folds, std::vector<double>(p, 0.0));
    for (int f = 0; f < n_folds; ++f) {
        auto& G = grams[f];
        auto& b = xtys[f];
        for (int idx : train_idx[f]) {
            const double* row = X.data() + static_cast<size_t>(idx) * p;
            for (int a = 0; a < p; ++a) {
                b[a] += row[a] * y[idx];
                for (int c = a; c < p; ++c) G[static_cast<size_t>(a) * p + c] += row[a] * row[c];
            }
        }
        for (int a = 0; a < p; ++a)
            for (int c = 0; c < a; ++c) G[static_cast<size_t>(a) * p + c] = G[static_cast<size_t>(c) * p + a];
    }

    RidgeSearchResult result;
    result.alphas = alphas;
    result.wmae.assign(alphas.size(), 0.0);
    result.wape.assign(alphas.size(), 0.0);
    const int n_alpha = static_cast<int>(alphas.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int k = 0; k < n_alpha; ++k) {
        double alpha = alphas[k];
        double total_werr = 0.0, total_w = 0.0, total_err = 0.0, total_y = 0.0;
        for (int f = 0; f < n_folds; ++f) {
            std::vector<double> A = grams[f];
            for (int d = 0; d < p; ++d) A[static_cast<size_t>(d) * p + d] += alpha;
            std::vector<double> beta;
            solve_linear(A, xtys[f], p, beta);
            for (int idx : val_idx[f]) {
                const double* row = X.data() + static_cast<size_t>(idx) * p;
                double pred = 0.0;
                for (int a = 0; a < p; ++a) pred += beta[a] * row[a];
                double err = std::fabs(y[idx] - pred);
                total_werr += weights[idx] * err;
                total_w += weights[idx];
                total_err += err;
                total_y += std::fabs(y[idx]);
            }
        }
        result.wmae[k] = total_werr / total_w;
        result.wape[k] = total_err / total_y;
    }

    int best = 0;
    for (int k = 1; k < n_alpha; ++k)
        if (result.wmae[k] < result.wmae[best]) best = k;
    result.best_index = best;
    return result;
}

}  // namespace wc
