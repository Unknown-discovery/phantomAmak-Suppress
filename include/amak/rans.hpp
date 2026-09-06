#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace amak::rans {

constexpr std::uint32_t kProbabilityBits = 12;
constexpr std::uint32_t kTotalFrequency = 1U << kProbabilityBits;
constexpr std::uint32_t kLowerBound = 1U << 23;
constexpr std::size_t kMaxSymbols = 256 * 256;
using Frequencies = std::array<std::uint16_t, 256>;
// Frequencies, cumulative frequencies, and the per-slot symbol lookup table.
constexpr std::size_t kDecodeTableBytes = 2 * sizeof(Frequencies) + kTotalFrequency;

struct Encoded {
    Frequencies frequencies{};
    std::vector<std::uint8_t> payload;
};

// The byte alphabet maps q to q + 128, including q == -128 on decode.
// Blocks must be nonempty and contain at most kMaxSymbols coefficients.
Encoded encode(const std::vector<std::int8_t>& coefficients);
void decode(const Frequencies& frequencies, const std::uint8_t* payload,
            std::size_t payload_bytes, std::int8_t* output,
            std::size_t symbol_count);

} // namespace amak::rans
