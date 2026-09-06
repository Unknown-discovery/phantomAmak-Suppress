#pragma once

#include "amak/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace amak::cli {

// Warm kernel-only comparison. Allocations/plan construction/I/O are excluded.
void benchmark_dct(std::ostream& out, std::size_t size, std::uint64_t iterations,
                   std::uint64_t warmup, DctBackend requested);

} // namespace amak::cli
