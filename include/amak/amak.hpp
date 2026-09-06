#pragma once

#include "amak/backend.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace amak {

constexpr std::uint32_t kMaxTileDimension = 256;
constexpr std::size_t kMaxPrefetchSlots = 8;

struct Limits {
    std::uint32_t max_tensors = 4096;
    std::uint64_t max_dimension = 1ULL << 20;
    std::uint64_t max_chunks_per_tensor = 1ULL << 24;
    std::size_t max_activation_elements = 1ULL << 24;
};

struct TensorSource {
    std::string name;
    std::uint64_t rows = 0; // output features
    std::uint64_t cols = 0; // input features
    std::uint32_t tile_rows = 32;
    std::uint32_t tile_cols = 32;
    std::vector<float> weights; // offline source, row-major [rows, cols]
    float quantization_step = 0; // zero = automatic per-tile step; otherwise > 0
};

struct TensorInfo {
    std::string name;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint32_t tile_rows = 0;
    std::uint32_t tile_cols = 0;
    std::uint64_t chunk_count = 0;
    std::uint64_t data_offset = 0;
    std::uint64_t data_bytes = 0;
};

// This is an in-memory object, NOT an on-disk ABI struct.
struct DecodedTile {
    std::uint64_t index = 0;
    std::uint64_t row_start = 0;
    std::uint64_t col_start = 0;
    std::uint16_t rows = 0;
    std::uint16_t cols = 0;
    float delta = 0;
    std::vector<std::int8_t> coefficients; // Q, output-frequency-major
};

struct StreamStats {
    std::uint64_t chunks_decoded = 0;
    std::uint64_t encoded_bytes_read = 0; // includes chunk headers, not directory
    std::uint64_t coefficients_decoded = 0;
    std::size_t slot_count = 0;
    std::size_t coefficient_capacity_bytes = 0; // allocated across all slots
    std::size_t decoder_scratch_bytes = 0; // encoded buffer + header + rANS tables
};

// Offline encoder. Publishes a finished file by rename from a temporary sibling.
// It never stores the source float weights in the resulting container.
void write_container(const std::string& path,
                     const std::vector<TensorSource>& tensors,
                     const Limits& limits = {});

class Container {
public:
    // Only header/directory validation is eager. Keep the file immutable while used.
    explicit Container(std::string path, Limits limits = {});

    const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
    const Limits& limits() const noexcept { return limits_; }
    const std::string& path() const noexcept { return path_; }
    std::uint64_t file_bytes() const noexcept { return file_bytes_; }
    std::size_t find_tensor(const std::string& name) const;

    // Callback executes in the caller's thread. The tile reference and its data
    // expire when the callback returns. Each visit opens an independent stream.
    // 0 = synchronous; 1..8 = bounded producer/consumer slots, including the slot
    // currently being consumed. Exceptions cancel and join the producer.
    StreamStats visit_tiles(
        std::size_t tensor_index,
        const std::function<void(const DecodedTile&)>& callback,
        std::size_t prefetch_slots = 2) const;

private:
    std::string path_;
    Limits limits_;
    std::uint64_t file_bytes_ = 0;
    std::vector<TensorInfo> tensors_;
};

struct RunOptions {
    std::size_t batch = 1;
    std::size_t prefetch_slots = 2;
    DctBackend dct_backend = DctBackend::automatic;
};

struct RunResult {
    std::vector<float> output; // spatial [batch, output_features]
    StreamStats stream;
    DctBackend dct_backend = DctBackend::scalar; // resolved execution backend
    std::size_t activation_scratch_bytes = 0; // frequency buffers + one spatial row tile
    std::size_t transform_plan_bytes = 0; // small double-precision DCT bases
};

// Computes Y = X W_quantized^T + bias, without reconstructing spatial W.
// Input/output are row-major. An optional spatial bias has length W.rows.
RunResult linear(const Container& container, std::size_t tensor_index,
                 const std::vector<float>& input,
                 const RunOptions& options = {},
                 const std::vector<float>& bias = {});

} // namespace amak
