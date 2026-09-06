#include "amak/dct.hpp"
#include "amak/amak.hpp"
#include "binary.hpp"
#include "dct_kernels.hpp"

#include <cmath>

namespace amak {

DctPlan::DctPlan(std::size_t size, DctBackend backend) : size_(size), kernels_(nullptr) {
    detail::require(size > 0 && size <= kMaxTileDimension, "DCT size must be in 1..256");
    kernels_ = &detail::dct_kernels(backend);
    basis_.resize(size * size);
    const double n = static_cast<double>(size);
    const double pi = std::acos(-1.0);
    for (std::size_t k = 0; k < size; ++k) {
        const double alpha = std::sqrt((k == 0 ? 1.0 : 2.0) / n);
        for (std::size_t j = 0; j < size; ++j)
            basis_[k * size + j] = alpha * std::cos(
                pi * (static_cast<double>(j) + 0.5) * static_cast<double>(k) / n);
    }
}

DctBackend DctPlan::backend() const noexcept { return kernels_->backend; }

void DctPlan::forward(const double* input, double* output) const {
    kernels_->forward_f64(basis_.data(), size_, input, output);
}

void DctPlan::forward(const float* input, double* output) const {
    kernels_->forward_f32(basis_.data(), size_, input, output);
}

void DctPlan::inverse(const double* input, double* output) const {
    kernels_->inverse_f64(basis_.data(), size_, input, output);
}

std::vector<double> forward_2d(const std::vector<double>& spatial,
                               std::size_t rows, std::size_t cols) {
    const DctPlan dr(rows, DctBackend::scalar), dc(cols, DctBackend::scalar);
    detail::require(spatial.size() == rows * cols, "DCT tile shape mismatch");
    std::vector<double> tmp(rows * cols), result(rows * cols);
    // Right multiply by D_cols^T, then left multiply by D_rows.
    for (std::size_t r = 0; r < rows; ++r)
        dc.forward(spatial.data() + r * cols, tmp.data() + r * cols);
    for (std::size_t u = 0; u < rows; ++u) {
        for (std::size_t v = 0; v < cols; ++v) {
            double sum = 0;
            for (std::size_t r = 0; r < rows; ++r) sum += dr.basis(u, r) * tmp[r * cols + v];
            result[u * cols + v] = sum;
        }
    }
    return result;
}

std::vector<double> inverse_2d(const std::vector<double>& frequency,
                               std::size_t rows, std::size_t cols) {
    const DctPlan dr(rows, DctBackend::scalar), dc(cols, DctBackend::scalar);
    detail::require(frequency.size() == rows * cols, "inverse DCT tile shape mismatch");
    std::vector<double> tmp(rows * cols), result(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t v = 0; v < cols; ++v) {
            double sum = 0;
            for (std::size_t u = 0; u < rows; ++u)
                sum += dr.basis(u, r) * frequency[u * cols + v];
            tmp[r * cols + v] = sum;
        }
        dc.inverse(tmp.data() + r * cols, result.data() + r * cols);
    }
    return result;
}

} // namespace amak
