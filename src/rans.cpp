#include "amak/rans.hpp"
#include "binary.hpp"

#include <algorithm>
#include <numeric>

namespace amak::rans {
namespace {

std::size_t symbol(std::int8_t q) {
    return static_cast<std::size_t>(static_cast<int>(q) + 128);
}

Frequencies normalize(const std::vector<std::int8_t>& coefficients) {
    std::array<std::uint64_t, 256> counts{};
    for (const auto q : coefficients) ++counts[symbol(q)];
    std::vector<std::size_t> observed;
    for (std::size_t s = 0; s < counts.size(); ++s)
        if (counts[s] != 0) observed.push_back(s);
    const auto budget = kTotalFrequency - static_cast<std::uint32_t>(observed.size());
    const auto n = coefficients.size();
    Frequencies frequencies{};
    std::array<std::uint64_t, 256> remainder{};
    std::uint32_t used = 0;
    for (const auto s : observed) {
        const auto scaled = budget * counts[s];
        frequencies[s] = static_cast<std::uint16_t>(1 + scaled / n);
        remainder[s] = scaled % n;
        used += frequencies[s];
    }
    std::sort(observed.begin(), observed.end(), [&](std::size_t a, std::size_t b) {
        return remainder[a] != remainder[b] ? remainder[a] > remainder[b] : a < b;
    });
    for (std::size_t i = 0; i < kTotalFrequency - used; ++i) ++frequencies[observed[i]];
    return frequencies;
}

Frequencies cumulative(const Frequencies& frequencies) {
    Frequencies c{};
    std::uint32_t sum = 0;
    for (std::size_t s = 0; s < frequencies.size(); ++s) {
        detail::require(frequencies[s] <= kTotalFrequency, "rANS frequency exceeds 4096");
        c[s] = static_cast<std::uint16_t>(sum);
        sum += frequencies[s];
        detail::require(sum <= kTotalFrequency, "rANS frequency sum exceeds 4096");
    }
    detail::require(sum == kTotalFrequency, "rANS frequencies must sum to 4096");
    return c;
}

} // namespace

Encoded encode(const std::vector<std::int8_t>& coefficients) {
    detail::require(!coefficients.empty() && coefficients.size() <= kMaxSymbols,
                    "rANS block must contain 1..65536 symbols");
    Encoded encoded;
    encoded.frequencies = normalize(coefficients);
    const auto c = cumulative(encoded.frequencies);
    std::vector<std::uint8_t> emitted;
    emitted.reserve(2 * coefficients.size());
    std::uint32_t state = kLowerBound;
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
        const auto s = symbol(*it);
        const std::uint32_t f = encoded.frequencies[s];
        const std::uint32_t threshold = ((kLowerBound >> kProbabilityBits) << 8) * f;
        while (state >= threshold) {
            emitted.push_back(static_cast<std::uint8_t>(state & 255U));
            state >>= 8;
        }
        state = ((state / f) << kProbabilityBits) + state % f + c[s];
    }
    encoded.payload.resize(4 + emitted.size());
    detail::put32(encoded.payload.data(), state);
    std::reverse_copy(emitted.begin(), emitted.end(), encoded.payload.begin() + 4);
    return encoded;
}

void decode(const Frequencies& frequencies, const std::uint8_t* payload,
            std::size_t payload_bytes, std::int8_t* output, std::size_t symbol_count) {
    detail::require(symbol_count > 0 && symbol_count <= kMaxSymbols,
                    "rANS block must contain 1..65536 symbols");
    detail::require(payload != nullptr && output != nullptr, "null rANS buffer");
    detail::require(payload_bytes >= 4 && payload_bytes <= 2 * symbol_count + 4,
                    "invalid rANS payload length");
    const auto c = cumulative(frequencies);
    std::array<std::uint8_t, kTotalFrequency> lookup{};
    for (std::size_t s = 0; s < frequencies.size(); ++s)
        for (std::uint32_t j = c[s]; j < std::uint32_t(c[s]) + frequencies[s]; ++j)
            lookup[j] = static_cast<std::uint8_t>(s);

    std::uint32_t state = detail::u32(payload);
    detail::require(state >= kLowerBound && state < (kLowerBound << 8),
                    "rANS initial state out of range");
    std::size_t cursor = 4;
    for (std::size_t i = 0; i < symbol_count; ++i) {
        const auto slot = state & (kTotalFrequency - 1);
        const auto s = lookup[slot];
        output[i] = static_cast<std::int8_t>(static_cast<int>(s) - 128);
        state = std::uint32_t(frequencies[s]) * (state >> kProbabilityBits) + slot - c[s];
        while (state < kLowerBound) {
            detail::require(cursor < payload_bytes, "truncated rANS renormalization bytes");
            state = (state << 8) | payload[cursor++];
        }
    }
    detail::require(state == kLowerBound, "rANS terminal state mismatch");
    detail::require(cursor == payload_bytes, "unused bytes at end of rANS payload");
}

} // namespace amak::rans
