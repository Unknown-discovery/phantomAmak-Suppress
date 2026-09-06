#include "benchmark.hpp"
#include "amak/dct.hpp"
#include "binary.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <vector>

namespace amak::cli {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kTrials = 5;

struct Timing {
    double ns_per_call;
    double checksum;
};

template<class Transform>
Timing measure(const Transform& transform, const std::vector<double>& output,
                std::uint64_t iterations, std::uint64_t warmup) {
    for (std::uint64_t i = 0; i < warmup; ++i) transform();
    // Compiler fences and a per-iteration output checksum keep the calls from
    // being hoisted/eliminated by the supported optimizing toolchains. The small
    // checksum/loop overhead is included in both measurements.
    volatile double checksum = 0;
    const auto start = Clock::now();
    for (std::uint64_t i = 0; i < iterations; ++i) {
        std::atomic_signal_fence(std::memory_order_seq_cst);
        transform();
        std::atomic_signal_fence(std::memory_order_seq_cst);
        checksum += output[static_cast<std::size_t>(i % output.size())];
    }
    const double ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
    return {ns / static_cast<double>(iterations), checksum};
}

template<class Reference, class Selected>
void compare(std::ostream& out, const char* label, const Reference& reference,
             const Selected& selected, const std::vector<double>& a,
             const std::vector<double>& b, std::uint64_t iterations,
             std::uint64_t warmup, double& checksum) {
    std::array<double, kTrials> scalar_times{}, selected_times{};
    for (std::size_t trial = 0; trial < kTrials; ++trial) {
        Timing scalar{}, chosen{};
        // Alternate ordering to reduce systematic cache/frequency-order bias.
        if (trial % 2 == 0) {
            scalar = measure(reference, a, iterations, warmup);
            chosen = measure(selected, b, iterations, warmup);
        } else {
            chosen = measure(selected, b, iterations, warmup);
            scalar = measure(reference, a, iterations, warmup);
        }
        scalar_times[trial] = scalar.ns_per_call;
        selected_times[trial] = chosen.ns_per_call;
        checksum += scalar.checksum + chosen.checksum;
    }
    std::sort(scalar_times.begin(), scalar_times.end());
    std::sort(selected_times.begin(), selected_times.end());
    const auto scalar_median = scalar_times[kTrials / 2];
    const auto selected_median = selected_times[kTrials / 2];
    double max_error = 0;
    for (std::size_t i = 0; i < a.size(); ++i) max_error = std::max(max_error, std::abs(a[i] - b[i]));
    out << std::left << std::setw(14) << label << std::right << std::fixed << std::setprecision(2)
        << std::setw(15) << scalar_median << std::setw(16) << selected_median;
    if (selected_median > 0) out << std::setw(12) << scalar_median / selected_median;
    else out << std::setw(12) << "n/a";
    out << std::scientific << std::setprecision(3) << std::setw(16) << max_error << '\n';
}

} // namespace

void benchmark_dct(std::ostream& out, std::size_t size, std::uint64_t iterations,
                   std::uint64_t warmup, DctBackend requested) {
    detail::require(size > 0 && size <= 256, "benchmark size must be in 1..256");
    detail::require(iterations > 0 && iterations <= 1000000, "benchmark iterations must be in 1..1000000");
    detail::require(warmup <= 1000000, "benchmark warmup must be in 0..1000000");
    const auto backend = resolve_dct_backend(requested);
    const DctPlan scalar(size, DctBackend::scalar), selected(size, backend);
    std::vector<double> input(size), scalar_output(size), selected_output(size);
    std::vector<float> input_f32(size);
    for (std::size_t i = 0; i < size; ++i) {
        input[i] = std::sin(static_cast<double>(i) * 0.31) + std::cos(static_cast<double>(i) * 0.17) * 0.2;
        input_f32[i] = static_cast<float>(input[i]);
    }
    out << "DCT kernel benchmark | size=" << size << " | iterations=" << iterations
        << " | warmup=" << warmup << " | trials=" << kTrials << '\n'
        << "Selected DCT backend: " << dct_backend_name(backend) << '\n'
        << "Median ns/call; ratio = scalar / selected (not an end-to-end speedup).\n"
        << std::left << std::setw(14) << "operation" << std::right << std::setw(15) << "scalar ns"
        << std::setw(16) << "selected ns" << std::setw(12) << "ratio" << std::setw(16) << "max_abs_error" << '\n';
    double checksum = 0;
    compare(out, "f64 forward", [&] { scalar.forward(input.data(), scalar_output.data()); },
            [&] { selected.forward(input.data(), selected_output.data()); },
            scalar_output, selected_output, iterations, warmup, checksum);
    compare(out, "f32 forward", [&] { scalar.forward(input_f32.data(), scalar_output.data()); },
            [&] { selected.forward(input_f32.data(), selected_output.data()); },
            scalar_output, selected_output, iterations, warmup, checksum);
    compare(out, "f64 inverse", [&] { scalar.inverse(input.data(), scalar_output.data()); },
            [&] { selected.inverse(input.data(), selected_output.data()); },
            scalar_output, selected_output, iterations, warmup, checksum);
    out << "Checksum: " << std::scientific << std::setprecision(9) << checksum << '\n'
        << "Warm buffers/plans; excludes plan setup, entropy decoding, I/O and coefficient multiplication.\n"
        << "Includes dispatch/loop/checksum overhead; small sizes and noisy hosts may not benefit.\n";
}

} // namespace amak::cli
