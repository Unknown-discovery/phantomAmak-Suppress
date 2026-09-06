#include "amak/amak.hpp"
#include "amak/dct.hpp"
#include "amak/rans.hpp"
#include "binary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <utility>

namespace amak {
namespace {
using namespace detail;
namespace fs = std::filesystem;

void validate_name(const std::string& name) {
    require(!name.empty() && name.size() < 48, "tensor names must have 1..47 bytes");
    for (const unsigned char c : name)
        require(c >= 0x21 && c <= 0x7e, "tensor names must be printable ASCII without spaces");
}

void validate_shape(const TensorInfo& t, const Limits& limits) {
    require(t.rows > 0 && t.rows <= limits.max_dimension &&
            t.cols > 0 && t.cols <= limits.max_dimension, "tensor dimension exceeds limits or is zero");
    require(t.tile_rows > 0 && t.tile_rows <= kMaxTileDimension &&
            t.tile_cols > 0 && t.tile_cols <= kMaxTileDimension, "tile dimensions must be in 1..256");
    (void)multiply(t.rows, t.cols);
    const auto expected = multiply(ceil_div(t.rows, t.tile_rows), ceil_div(t.cols, t.tile_cols));
    require(t.chunk_count == expected && expected <= limits.max_chunks_per_tensor,
            "invalid or excessive chunk count");
}

struct TileShape {
    std::uint64_t row, col;
    std::uint16_t rows, cols;
};

TileShape tile_shape(const TensorInfo& t, std::uint64_t index) {
    const auto column_tiles = ceil_div(t.cols, t.tile_cols);
    const auto row = multiply(index / column_tiles, t.tile_rows);
    const auto col = multiply(index % column_tiles, t.tile_cols);
    require(index < t.chunk_count && row < t.rows && col < t.cols, "invalid chunk ordinal");
    return {row, col,
            static_cast<std::uint16_t>(std::min<std::uint64_t>(t.tile_rows, t.rows - row)),
            static_cast<std::uint16_t>(std::min<std::uint64_t>(t.tile_cols, t.cols - col))};
}

// A freshly created sibling directory gives exclusive ownership of the staging
// path without platform-specific file APIs. Rename publishes only a complete file.
class StagedFile {
public:
    explicit StagedFile(const std::string& target) : target_(target) {
        require(!target_.filename().empty(), "output must name a file");
        const auto parent = target_.has_parent_path() ? target_.parent_path() : fs::path(".");
        std::random_device random;
        for (unsigned attempt = 0; attempt < 32; ++attempt) {
            const auto token = (std::uint64_t(random()) << 32) | random();
            const auto candidate = parent / (".amak-tmp-" + std::to_string(token));
            std::error_code error;
            if (fs::create_directory(candidate, error)) {
                directory_ = candidate;
                file_ = directory_ / "payload";
                return;
            }
            require(!error, "cannot create output staging directory: " + error.message());
        }
        throw std::runtime_error("amak: cannot obtain a unique output staging directory");
    }
    StagedFile(const StagedFile&) = delete;
    StagedFile& operator=(const StagedFile&) = delete;
    ~StagedFile() {
        std::error_code ignored;
        fs::remove(file_, ignored);
        fs::remove(directory_, ignored);
    }
    const fs::path& path() const noexcept { return file_; }
    void publish() {
        std::error_code error;
        fs::rename(file_, target_, error);
        require(!error, "cannot publish output file: " + error.message());
    }

private:
    fs::path target_, directory_, file_;
};

std::vector<std::uint8_t> encode_chunk(const TensorSource& source,
                                      const TensorInfo& t, std::uint64_t index) {
    const auto shape = tile_shape(t, index);
    const std::size_t count = std::size_t(shape.rows) * shape.cols;
    std::vector<double> spatial(count);
    for (std::size_t r = 0; r < shape.rows; ++r)
        for (std::size_t c = 0; c < shape.cols; ++c)
            spatial[r * shape.cols + c] = source.weights[as_size(
                (shape.row + r) * t.cols + shape.col + c)];
    const auto frequency = forward_2d(spatial, shape.rows, shape.cols);
    float delta = source.quantization_step;
    if (delta == 0) {
        double maximum = 0;
        for (const auto value : frequency) maximum = std::max(maximum, std::abs(value));
        const double step = maximum == 0 ? 1.0 : maximum / 127.0;
        require(std::isfinite(step) && step <= std::numeric_limits<float>::max(),
                "automatic quantization step cannot fit in float32");
        delta = static_cast<float>(step);
        if (delta == 0) delta = std::numeric_limits<float>::denorm_min();
    }
    require(std::isfinite(delta) && delta > 0, "invalid quantization step");
    std::vector<std::int8_t> q(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double rounded = std::round(frequency[i] / static_cast<double>(delta));
        require(std::isfinite(rounded) && rounded >= -127 && rounded <= 127,
                "quantization step would clip coefficients (increase the step)");
        q[i] = static_cast<std::int8_t>(rounded);
    }
    const auto encoded = rans::encode(q);
    std::uint16_t alphabet = 0;
    for (const auto f : encoded.frequencies) if (f != 0) ++alphabet;
    const std::size_t bytes = kChunkHeaderBytes + 4 * std::size_t(alphabet) + encoded.payload.size();
    std::vector<std::uint8_t> chunk(bytes, 0);
    std::copy(kChunkMagic.begin(), kChunkMagic.end(), chunk.begin());
    put16(chunk.data() + 4, kChunkHeaderBytes);
    put64(chunk.data() + 8, index);
    put64(chunk.data() + 16, shape.row);
    put64(chunk.data() + 24, shape.col);
    put16(chunk.data() + 32, shape.rows);
    put16(chunk.data() + 34, shape.cols);
    put_float(chunk.data() + 36, delta);
    put32(chunk.data() + 40, static_cast<std::uint32_t>(count));
    put16(chunk.data() + 44, alphabet);
    chunk[46] = rans::kProbabilityBits;
    chunk[47] = 1;
    put32(chunk.data() + 48, static_cast<std::uint32_t>(encoded.payload.size()));
    put32(chunk.data() + 52, static_cast<std::uint32_t>(bytes));
    std::size_t cursor = kChunkHeaderBytes;
    for (std::size_t s = 0; s < encoded.frequencies.size(); ++s) {
        if (encoded.frequencies[s] == 0) continue;
        chunk[cursor] = static_cast<std::uint8_t>(s);
        put16(chunk.data() + cursor + 2, encoded.frequencies[s]);
        cursor += 4;
    }
    std::copy(encoded.payload.begin(), encoded.payload.end(), chunk.begin() + static_cast<std::ptrdiff_t>(cursor));
    put32(chunk.data() + 56, crc32(chunk.data(), chunk.size()));
    return chunk;
}

class TileDecoder {
public:
    TileDecoder(const std::string& path, const TensorInfo& tensor, std::uint64_t file_bytes)
        : in_(path, std::ios::binary), t_(tensor), cursor_(tensor.data_offset),
          end_(add(tensor.data_offset, tensor.data_bytes)) {
        require(in_.is_open(), "cannot open container: " + path);
        require(stream_size(in_) == file_bytes, "container length changed after opening");
        in_.seekg(as_offset(cursor_));
        require(static_cast<bool>(in_), "cannot seek to tensor data");
        const auto capacity = std::size_t(t_.tile_rows) * t_.tile_cols;
        body_.reserve(4 * 256 + 2 * capacity + 4);
        stats_.decoder_scratch_bytes = body_.capacity() + kChunkHeaderBytes + rans::kDecodeTableBytes;
    }

    void next(DecodedTile& tile) {
        require(next_ < t_.chunk_count, "read beyond final tensor chunk");
        require(add(cursor_, kChunkHeaderBytes) <= end_, "chunk header exceeds tensor extent");
        std::array<std::uint8_t, kChunkHeaderBytes> header{};
        read_exact(in_, header.data(), header.size());
        const auto* h = header.data();
        require(std::equal(kChunkMagic.begin(), kChunkMagic.end(), h), "invalid chunk magic");
        require(u16(h + 4) == kChunkHeaderBytes && u16(h + 6) == 0 && u32(h + 60) == 0,
                "unsupported chunk header size, flags or reserved fields");
        const auto expected = tile_shape(t_, next_);
        require(u64(h + 8) == next_ && u64(h + 16) == expected.row && u64(h + 24) == expected.col &&
                u16(h + 32) == expected.rows && u16(h + 34) == expected.cols,
                "chunk order, coordinates or edge dimensions mismatch");
        const float delta = f32(h + 36);
        require(std::isfinite(delta) && delta > 0, "nonfinite or nonpositive chunk step");
        const auto count = u32(h + 40);
        require(count == std::uint32_t(expected.rows) * expected.cols, "chunk symbol count mismatch");
        const auto alphabet = u16(h + 44);
        require(alphabet > 0 && alphabet <= 256, "invalid chunk alphabet count");
        require(h[46] == rans::kProbabilityBits && h[47] == 1, "unsupported entropy codec");
        const auto payload_bytes = u32(h + 48);
        require(payload_bytes >= 4 && payload_bytes <= 2 * count + 4, "invalid chunk payload size");
        const auto total = u32(h + 52);
        require(total == kChunkHeaderBytes + 4 * std::uint32_t(alphabet) + payload_bytes,
                "chunk framing length mismatch");
        require(add(cursor_, total) <= end_, "chunk data exceeds tensor extent");
        body_.resize(total - kChunkHeaderBytes);
        read_exact(in_, body_.data(), body_.size());
        const auto stored_crc = u32(h + 56);
        put32(header.data() + 56, 0);
        auto crc = crc_update(0xffffffffU, header.data(), header.size());
        crc = crc_update(crc, body_.data(), body_.size()) ^ 0xffffffffU;
        require(crc == stored_crc, "chunk CRC-32 mismatch");

        rans::Frequencies frequencies{};
        int previous = -1;
        for (std::size_t i = 0; i < alphabet; ++i) {
            const auto* entry = body_.data() + 4 * i;
            require(entry[0] > previous && entry[1] == 0 && u16(entry + 2) > 0,
                    "invalid or noncanonical frequency table entry");
            frequencies[entry[0]] = u16(entry + 2);
            previous = entry[0];
        }
        tile.index = next_;
        tile.row_start = expected.row;
        tile.col_start = expected.col;
        tile.rows = expected.rows;
        tile.cols = expected.cols;
        tile.delta = delta;
        tile.coefficients.resize(count);
        rans::decode(frequencies, body_.data() + 4 * std::size_t(alphabet), payload_bytes,
                     tile.coefficients.data(), count);
        cursor_ += total; // checked against end_ above
        ++next_;
        ++stats_.chunks_decoded;
        stats_.encoded_bytes_read += total;
        stats_.coefficients_decoded += count;
    }

    void finish() const {
        require(next_ == t_.chunk_count && cursor_ == end_, "unused bytes or missing chunks in tensor extent");
    }
    StreamStats stats() const noexcept { return stats_; }

private:
    std::ifstream in_;
    TensorInfo t_;
    std::uint64_t cursor_, end_, next_ = 0;
    std::vector<std::uint8_t> body_; // producer-private encoded table + payload
    StreamStats stats_;
};

} // namespace

void write_container(const std::string& path, const std::vector<TensorSource>& tensors,
                     const Limits& limits) {
    require(!tensors.empty() && tensors.size() <= limits.max_tensors,
            "tensor count is zero or exceeds limits");
    std::set<std::string> names;
    std::vector<TensorInfo> info;
    info.reserve(tensors.size());
    for (const auto& source : tensors) {
        validate_name(source.name);
        require(names.insert(source.name).second, "duplicate tensor name");
        TensorInfo t;
        t.name = source.name;
        t.rows = source.rows;
        t.cols = source.cols;
        t.tile_rows = source.tile_rows;
        t.tile_cols = source.tile_cols;
        require(t.tile_rows != 0 && t.tile_cols != 0, "zero tile dimension");
        t.chunk_count = multiply(ceil_div(t.rows, t.tile_rows), ceil_div(t.cols, t.tile_cols));
        validate_shape(t, limits);
        require(multiply(t.rows, t.cols) == source.weights.size(), "source weight shape mismatch");
        require(std::isfinite(source.quantization_step) && source.quantization_step >= 0,
                "quantization step must be zero (automatic) or finite and positive");
        for (const auto value : source.weights)
            require(std::isfinite(value), "source weights must be finite");
        info.push_back(std::move(t));
    }

    const auto directory_bytes = multiply(tensors.size(), kDirectoryEntryBytes);
    std::vector<std::uint8_t> directory(as_size(directory_bytes), 0);
    std::array<std::uint8_t, kHeaderBytes> header{};
    StagedFile staged(path);
    std::ofstream out(staged.path(), std::ios::binary | std::ios::trunc);
    require(out.is_open(), "cannot create output file");
    write_exact(out, header.data(), header.size());
    write_exact(out, directory.data(), directory.size());
    std::uint64_t cursor = add(kHeaderBytes, directory_bytes);
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto& t = info[i];
        t.data_offset = cursor;
        for (std::uint64_t k = 0; k < t.chunk_count; ++k) {
            const auto chunk = encode_chunk(tensors[i], t, k);
            cursor = add(cursor, chunk.size());
            (void)as_offset(cursor);
            write_exact(out, chunk.data(), chunk.size());
        }
        t.data_bytes = cursor - t.data_offset;
        auto* entry = directory.data() + i * kDirectoryEntryBytes;
        std::copy(t.name.begin(), t.name.end(), entry);
        put16(entry + 48, 1);
        entry[50] = 2;
        entry[51] = 1;
        entry[52] = 1;
        entry[53] = 1;
        put64(entry + 56, t.rows);
        put64(entry + 64, t.cols);
        put32(entry + 72, t.tile_rows);
        put32(entry + 76, t.tile_cols);
        put64(entry + 80, t.chunk_count);
        put64(entry + 88, t.data_offset);
        put64(entry + 96, t.data_bytes);
    }
    std::copy(kMagic.begin(), kMagic.end(), header.begin());
    put16(header.data() + 8, 1);
    put32(header.data() + 12, kHeaderBytes);
    put32(header.data() + 20, static_cast<std::uint32_t>(tensors.size()));
    put64(header.data() + 24, kHeaderBytes);
    put64(header.data() + 32, directory_bytes);
    put64(header.data() + 40, cursor);
    put32(header.data() + 48, crc32(directory.data(), directory.size()));
    put32(header.data() + 52, crc32(header.data(), header.size()));
    out.seekp(0);
    require(static_cast<bool>(out), "cannot seek output header");
    write_exact(out, header.data(), header.size());
    write_exact(out, directory.data(), directory.size());
    out.flush();
    require(static_cast<bool>(out), "cannot flush output file");
    out.close();
    require(!out.fail(), "cannot close output file");
    staged.publish();
}

Container::Container(std::string path, Limits limits)
    : path_(std::move(path)), limits_(limits) {
    std::ifstream in(path_, std::ios::binary);
    require(in.is_open(), "cannot open container: " + path_);
    file_bytes_ = stream_size(in);
    require(file_bytes_ >= kHeaderBytes, "truncated file header");
    std::array<std::uint8_t, kHeaderBytes> header{};
    read_exact(in, header.data(), header.size());
    auto* h = header.data();
    require(std::equal(kMagic.begin(), kMagic.end(), h), "invalid AMAK magic");
    const auto header_crc = u32(h + 52);
    put32(h + 52, 0);
    require(crc32(h, header.size()) == header_crc, "header CRC-32 mismatch");
    require(u16(h + 8) == 1 && u16(h + 10) == 0, "unsupported AMAK version (expected 1.0)");
    require(u32(h + 12) == kHeaderBytes && u32(h + 16) == 0 && u64(h + 56) == 0,
            "unsupported file header size, flags or reserved fields");
    const auto count = u32(h + 20);
    require(count > 0 && count <= limits_.max_tensors, "tensor count is zero or exceeds limits");
    require(u64(h + 24) == kHeaderBytes, "invalid directory offset");
    const auto directory_bytes = multiply(count, kDirectoryEntryBytes);
    require(u64(h + 32) == directory_bytes, "directory size mismatch");
    require(u64(h + 40) == file_bytes_, "physical file length does not match header");
    const auto directory_end = add(kHeaderBytes, directory_bytes);
    require(directory_end <= file_bytes_, "directory exceeds file length");
    std::vector<std::uint8_t> directory(as_size(directory_bytes));
    read_exact(in, directory.data(), directory.size());
    require(crc32(directory.data(), directory.size()) == u32(h + 48), "directory CRC-32 mismatch");

    std::set<std::string> names;
    std::uint64_t cursor = directory_end;
    tensors_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto* e = directory.data() + i * kDirectoryEntryBytes;
        const auto* terminator = std::find(e, e + 48, std::uint8_t(0));
        require(terminator != e + 48 && all_zero(terminator, static_cast<std::size_t>(e + 48 - terminator)),
                "tensor name lacks a terminator or has nonzero padding");
        TensorInfo t;
        t.name.assign(reinterpret_cast<const char*>(e), static_cast<std::size_t>(terminator - e));
        validate_name(t.name);
        require(names.insert(t.name).second, "duplicate tensor name");
        require(u16(e + 48) == 1 && e[50] == 2 && e[51] == 1 && e[52] == 1 && e[53] == 1,
                "unsupported tensor kind, rank, transform, quantizer or codec");
        require(u16(e + 54) == 0 && all_zero(e + 104, 24), "unsupported tensor flags or reserved fields");
        t.rows = u64(e + 56);
        t.cols = u64(e + 64);
        t.tile_rows = u32(e + 72);
        t.tile_cols = u32(e + 76);
        t.chunk_count = u64(e + 80);
        t.data_offset = u64(e + 88);
        t.data_bytes = u64(e + 96);
        validate_shape(t, limits_);
        require(t.data_offset == cursor, "tensor extents must be contiguous and ordered");
        require(t.data_bytes >= multiply(t.chunk_count, kChunkHeaderBytes + 4 + 4),
                "tensor extent is too small for its chunk count");
        cursor = add(cursor, t.data_bytes);
        require(cursor <= file_bytes_, "tensor extent exceeds file length");
        tensors_.push_back(std::move(t));
    }
    require(cursor == file_bytes_, "unused bytes outside tensor extents");
}

std::size_t Container::find_tensor(const std::string& name) const {
    for (std::size_t i = 0; i < tensors_.size(); ++i)
        if (tensors_[i].name == name) return i;
    throw std::runtime_error("amak: unknown tensor: " + name);
}

StreamStats Container::visit_tiles(std::size_t tensor_index,
                                  const std::function<void(const DecodedTile&)>& callback,
                                  std::size_t prefetch_slots) const {
    require(tensor_index < tensors_.size(), "tensor index out of range");
    require(static_cast<bool>(callback), "tile callback is empty");
    require(prefetch_slots <= kMaxPrefetchSlots, "prefetch must be in 0..8");
    const auto& t = tensors_[tensor_index];
    const std::size_t tile_capacity = std::size_t(t.tile_rows) * t.tile_cols;
    TileDecoder decoder(path_, t, file_bytes_);
    if (prefetch_slots == 0) {
        DecodedTile tile;
        tile.coefficients.reserve(tile_capacity);
        for (std::uint64_t i = 0; i < t.chunk_count; ++i) {
            decoder.next(tile);
            callback(tile);
        }
        decoder.finish();
        auto stats = decoder.stats();
        stats.slot_count = 1;
        stats.coefficient_capacity_bytes = tile.coefficients.capacity();
        return stats;
    }

    enum class State { free, filling, ready, reading };
    struct Slot {
        State state = State::free;
        DecodedTile tile;
    };
    const auto slot_count = static_cast<std::size_t>(std::min<std::uint64_t>(prefetch_slots, t.chunk_count));
    std::vector<Slot> slots(slot_count);
    std::size_t coefficient_bytes = 0;
    for (auto& slot : slots) {
        slot.tile.coefficients.reserve(tile_capacity);
        coefficient_bytes += slot.tile.coefficients.capacity();
    }
    std::mutex mutex;
    std::condition_variable changed;
    bool cancelled = false, finished = false;
    std::exception_ptr failure;
    std::thread producer([&] {
        try {
            for (std::uint64_t i = 0; i < t.chunk_count; ++i) {
                auto& slot = slots[static_cast<std::size_t>(i % slot_count)];
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return cancelled || slot.state == State::free; });
                    if (cancelled) return;
                    slot.state = State::filling;
                }
                decoder.next(slot.tile); // file, encoded buffer and rANS tables are producer-private
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (cancelled) return;
                    slot.state = State::ready;
                }
                changed.notify_all();
            }
            decoder.finish();
            {
                std::lock_guard<std::mutex> lock(mutex);
                finished = true;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            failure = std::current_exception();
            finished = true;
        }
        changed.notify_all();
    });

    try {
        for (std::uint64_t i = 0; i < t.chunk_count; ++i) {
            auto& slot = slots[static_cast<std::size_t>(i % slot_count)];
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return failure || finished || slot.state == State::ready; });
                if (failure) std::rethrow_exception(failure);
                require(slot.state == State::ready, "prefetch ended before all chunks were ready");
                slot.state = State::reading;
            }
            callback(slot.tile); // no mutex held; producer cannot reuse a reading slot
            {
                std::lock_guard<std::mutex> lock(mutex);
                slot.state = State::free;
            }
            changed.notify_all();
        }
        producer.join();
        if (failure) std::rethrow_exception(failure);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            cancelled = true;
        }
        changed.notify_all();
        if (producer.joinable()) producer.join();
        throw;
    }
    auto stats = decoder.stats();
    stats.slot_count = slot_count;
    stats.coefficient_capacity_bytes = coefficient_bytes;
    return stats;
}

} // namespace amak
