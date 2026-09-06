#pragma once

#include "amak/backend.hpp"

#include <cstddef>
#include <vector>

namespace amak {
namespace detail { struct DctKernels; }

// Orthonormal DCT-II; inverse is its transpose (orthonormal DCT-III).
// O(n^2) basis multiplication with scalar or optional runtime-dispatched AVX2
// kernels, not an FFT. The default stays scalar for reproducible offline packing.
class DctPlan {
public:
    explicit DctPlan(std::size_t size, DctBackend backend = DctBackend::scalar);
    std::size_t size() const noexcept { return size_; }
    DctBackend backend() const noexcept;
    std::size_t storage_bytes() const noexcept { return basis_.capacity() * sizeof(double); }
    double basis(std::size_t frequency, std::size_t spatial) const noexcept {
        return basis_[frequency * size_ + spatial];
    }
    // Input and output must each hold size() elements and must not overlap.
    // No alignment or padding beyond those elements is required. Both forward
    // overloads accumulate in double; the float overload avoids a staging copy.
    void forward(const double* input, double* output) const;
    void forward(const float* input, double* output) const;
    void inverse(const double* input, double* output) const;

private:
    std::size_t size_;
    std::vector<double> basis_;
    const detail::DctKernels* kernels_;
};

// Offline/test helpers, deliberately pinned to scalar arithmetic. Inference
// does not call inverse_2d or reconstruct spatial weight tiles.
std::vector<double> forward_2d(const std::vector<double>& spatial,
                               std::size_t rows, std::size_t cols);
std::vector<double> inverse_2d(const std::vector<double>& frequency,
                               std::size_t rows, std::size_t cols);

} // namespace amak
