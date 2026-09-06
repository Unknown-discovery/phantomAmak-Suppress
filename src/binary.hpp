#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

namespace amak::detail {

constexpr std::size_t kHeaderBytes = 64;
constexpr std::size_t kDirectoryEntryBytes = 128;
constexpr std::size_t kChunkHeaderBytes = 64;
constexpr std::array<std::uint8_t, 8> kMagic{{'A', 'M', 'A', 'K', '\r', '\n', 0x1a, '\n'}};
constexpr std::array<std::uint8_t, 4> kChunkMagic{{'A', 'C', 'N', 'K'}};
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "AMAK needs IEEE-754 binary32 floats");

inline void require(bool condition, const char* message) {
    // Successful checks in decode/dot loops must not allocate a std::string.
    if (!condition) throw std::runtime_error(std::string("amak: ") + message);
}
inline void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("amak: " + message);
}
inline std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    require(b <= std::numeric_limits<std::uint64_t>::max() - a, "size addition overflow");
    return a + b;
}
inline std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    require(a == 0 || b <= std::numeric_limits<std::uint64_t>::max() / a,
            "size multiplication overflow");
    return a * b;
}
inline std::uint64_t ceil_div(std::uint64_t a, std::uint64_t b) {
    require(b != 0, "division by zero");
    return a / b + (a % b != 0 ? 1 : 0);
}
inline std::size_t as_size(std::uint64_t value) {
    require(value <= std::numeric_limits<std::size_t>::max(), "size exceeds address space");
    return static_cast<std::size_t>(value);
}
inline std::streamoff as_offset(std::uint64_t value) {
    require(value <= static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()),
            "file offset is not representable");
    return static_cast<std::streamoff>(value);
}
inline std::uint16_t u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(std::uint16_t(p[0]) | (std::uint16_t(p[1]) << 8));
}
inline std::uint32_t u32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
           (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}
inline std::uint64_t u64(const std::uint8_t* p) {
    return std::uint64_t(u32(p)) | (std::uint64_t(u32(p + 4)) << 32);
}
inline void put16(std::uint8_t* p, std::uint16_t v) {
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
}
inline void put32(std::uint8_t* p, std::uint32_t v) {
    for (unsigned i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
inline void put64(std::uint8_t* p, std::uint64_t v) {
    for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}
inline float f32(const std::uint8_t* p) {
    const auto bits = u32(p);
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}
inline void put_float(std::uint8_t* p, float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    put32(p, bits);
}
inline bool all_zero(const std::uint8_t* p, std::size_t count) {
    return std::all_of(p, p + count, [](std::uint8_t b) { return b == 0; });
}

// Unfinalized update allows CRC over a header and body without concatenating.
inline std::uint32_t crc_update(std::uint32_t crc, const std::uint8_t* data,
                                std::size_t size) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            auto x = i;
            for (int bit = 0; bit < 8; ++bit)
                x = (x >> 1) ^ ((x & 1U) ? 0xedb88320U : 0U);
            t[i] = x;
        }
        return t;
    }();
    for (std::size_t i = 0; i < size; ++i)
        crc = table[(crc ^ data[i]) & 255U] ^ (crc >> 8);
    return crc;
}
inline std::uint32_t crc32(const std::uint8_t* data, std::size_t size) {
    return crc_update(0xffffffffU, data, size) ^ 0xffffffffU;
}
inline void read_exact(std::istream& in, std::uint8_t* data, std::size_t size) {
    require(size <= static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()),
            "read size is not representable");
    if (size == 0) return;
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size));
    require(static_cast<bool>(in), "truncated file or read error");
}
inline void write_exact(std::ostream& out, const std::uint8_t* data, std::size_t size) {
    require(size <= static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()),
            "write size is not representable");
    if (size == 0) return;
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    require(static_cast<bool>(out), "file write failed");
}
inline std::uint64_t stream_size(std::istream& in) {
    in.seekg(0, std::ios::end);
    const auto position = in.tellg();
    require(position >= 0, "cannot determine file length (a seekable file is required)");
    in.seekg(0, std::ios::beg);
    require(static_cast<bool>(in), "cannot seek input file");
    return static_cast<std::uint64_t>(position);
}

} // namespace amak::detail
