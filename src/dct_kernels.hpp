#pragma once

#include "amak/backend.hpp"

#include <cstddef>

namespace amak::detail {

// All basis arrays are row-major D[k, j], and buffers may be unaligned.
// Inputs and outputs must be disjoint, with n live elements each.
struct DctKernels {
    DctBackend backend;
    void (*forward_f64)(const double* basis, std::size_t n, const double* input, double* output);
    void (*forward_f32)(const double* basis, std::size_t n, const float* input, double* output);
    void (*inverse_f64)(const double* basis, std::size_t n, const double* input, double* output);
};

// Returns a process-lifetime table; feature dispatch is outside the hot loops.
const DctKernels& dct_kernels(DctBackend requested);

} // namespace amak::detail
