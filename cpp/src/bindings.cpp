#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "walmart_core.hpp"

namespace py = pybind11;

py::array_t<double> py_dtw_distance_matrix(py::array_t<double, py::array::c_style | py::array::forcecast> series,
                                           int band, int threads) {
    auto buf = series.request();
    if (buf.ndim != 2) throw std::runtime_error("series 必须是二维 (N, T)");
    int n = static_cast<int>(buf.shape[0]);
    int t = static_cast<int>(buf.shape[1]);

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    std::vector<double> data(static_cast<size_t>(n) * t);
    const double* src = static_cast<const double*>(buf.ptr);
    std::copy(src, src + static_cast<size_t>(n) * t, data.begin());

    std::vector<double> dist = wc::dtw_distance_matrix(data, n, t, band);
    py::array_t<double> out({n, n});
    std::copy(dist.begin(), dist.end(), static_cast<double*>(out.request().ptr));
    return out;
}

py::dict py_ridge_grid_search(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                              py::array_t<double, py::array::c_style | py::array::forcecast> y,
                              py::list train_idx, py::list val_idx,
                              py::array_t<double, py::array::c_style | py::array::forcecast> weights,
                              py::array_t<double, py::array::c_style | py::array::forcecast> alphas,
                              int threads) {
    auto xb = X.request();
    auto yb = y.request();
    auto wb = weights.request();
    auto ab = alphas.request();
    int n = static_cast<int>(xb.shape[0]);
    int p = static_cast<int>(xb.shape[1]);

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    std::vector<double> Xv(static_cast<size_t>(n) * p);
    std::vector<double> yv(n);
    std::vector<double> wv(n);
    std::vector<double> av(ab.shape[0]);
    std::copy(static_cast<const double*>(xb.ptr), static_cast<const double*>(xb.ptr) + static_cast<size_t>(n) * p, Xv.begin());
    std::copy(static_cast<const double*>(yb.ptr), static_cast<const double*>(yb.ptr) + n, yv.begin());
    std::copy(static_cast<const double*>(wb.ptr), static_cast<const double*>(wb.ptr) + n, wv.begin());
    std::copy(static_cast<const double*>(ab.ptr), static_cast<const double*>(ab.ptr) + av.size(), av.begin());

    auto to_indices = [](py::list lst) {
        std::vector<std::vector<int>> out;
        out.reserve(lst.size());
        for (auto item : lst) {
            auto arr = item.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
            auto b = arr.request();
            const int64_t* ptr = static_cast<const int64_t*>(b.ptr);
            out.emplace_back(ptr, ptr + b.size);
        }
        return out;
    };
    auto tr = to_indices(train_idx);
    auto va = to_indices(val_idx);

    wc::RidgeSearchResult res = wc::ridge_grid_search(Xv, yv, n, p, tr, va, wv, av);

    py::dict out;
    out["alphas"] = res.alphas;
    out["wmae"] = res.wmae;
    out["wape"] = res.wape;
    out["best_index"] = res.best_index;
    out["best_alpha"] = res.alphas[res.best_index];
    return out;
}

PYBIND11_MODULE(walmart_cpp, m) {
    m.doc() = "Walmart 分析 C++ 加速模块（DTW 距离矩阵 + Ridge 超参搜索）";
    m.def("dtw_distance_matrix", &py_dtw_distance_matrix, py::arg("series"), py::arg("band") = -1,
          py::arg("threads") = 0, "OpenMP 并行的 DTW 距离矩阵");
    m.def("ridge_grid_search", &py_ridge_grid_search, py::arg("X"), py::arg("y"), py::arg("train_idx"),
          py::arg("val_idx"), py::arg("weights"), py::arg("alphas"), py::arg("threads") = 0,
          "OpenMP 并行的 Ridge alpha 搜索（闭式解）");
}
