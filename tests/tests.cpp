#include "amak/amak.hpp"
#include "amak/dct.hpp"
#include "amak/rans.hpp"
#include "binary.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs = std::filesystem;
namespace bin = amak::detail;

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error( \
    std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #condition); } while (false)

void near(double actual, double expected, double tolerance = 1e-10) {
    if (!(std::abs(actual - expected) <= tolerance))
        throw std::runtime_error("not close: " + std::to_string(actual) + " vs " + std::to_string(expected));
}

template<class Function>
void rejects(Function&& fn, const std::string& message = "") {
    bool threw = false;
    try { fn(); }
    catch (const std::exception& error) {
        threw = true;
        if (!message.empty()) CHECK(std::string(error.what()).find(message) != std::string::npos);
    }
    CHECK(threw);
}

class TempDirectory {
public:
    TempDirectory() {
        std::random_device random;
        for (int i = 0; i < 32; ++i) {
            path_ = fs::temp_directory_path() / ("amak-tests-" + std::to_string(random()) + "-" + std::to_string(random()));
            if (fs::create_directory(path_)) return;
        }
        throw std::runtime_error("cannot make test directory");
    }
    ~TempDirectory() { std::error_code ignored; fs::remove_all(path_, ignored); }
    std::string file(const std::string& name) const { return (path_ / name).string(); }
    const fs::path& path() const { return path_; }
private:
    fs::path path_;
};

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    CHECK(in.is_open());
    std::vector<std::uint8_t> bytes(bin::as_size(bin::stream_size(in)));
    bin::read_exact(in, bytes.data(), bytes.size());
    return bytes;
}

void write_bytes(const std::string& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    CHECK(out.is_open());
    bin::write_exact(out, bytes.data(), bytes.size());
}

void fix_header_crc(std::vector<std::uint8_t>& bytes) {
    bin::put32(bytes.data() + 52, 0);
    bin::put32(bytes.data() + 52, bin::crc32(bytes.data(), 64));
}
void fix_directory_crc(std::vector<std::uint8_t>& bytes) {
    bin::put32(bytes.data() + 48, bin::crc32(bytes.data() + 64, bin::as_size(bin::u64(bytes.data() + 32))));
    fix_header_crc(bytes);
}
void fix_chunk_crc(std::vector<std::uint8_t>& bytes, std::size_t start) {
    const auto size = bin::u32(bytes.data() + start + 52);
    CHECK(start + size <= bytes.size());
    bin::put32(bytes.data() + start + 56, 0);
    bin::put32(bytes.data() + start + 56, bin::crc32(bytes.data() + start, size));
}

amak::TensorSource source(std::string name, std::size_t rows, std::size_t cols,
                          std::uint32_t tile_rows, std::uint32_t tile_cols) {
    amak::TensorSource t;
    t.name = std::move(name);
    t.rows = rows;
    t.cols = cols;
    t.tile_rows = tile_rows;
    t.tile_cols = tile_cols;
    std::mt19937 random(static_cast<std::uint32_t>(rows * 31 + cols));
    std::uniform_real_distribution<float> distribution(-0.25F, 0.25F);
    t.weights.resize(rows * cols);
    for (auto& value : t.weights) value = distribution(random);
    return t;
}

void test_binary_and_dct() {
    const std::string check = "123456789";
    CHECK(bin::crc32(reinterpret_cast<const std::uint8_t*>(check.data()), check.size()) == 0xcbf43926U);
    std::array<std::uint8_t, 24> bytes{};
    bin::put64(bytes.data(), 0x0123456789abcdefULL);
    CHECK(bytes[0] == 0xef && bytes[7] == 0x01);
    CHECK(bin::u64(bytes.data()) == 0x0123456789abcdefULL);
    bin::put_float(bytes.data() + 8, -1.25F);
    CHECK(bin::u32(bytes.data() + 8) == 0xbfa00000U);
    CHECK(bin::f32(bytes.data() + 8) == -1.25F);
    rejects([] { (void)bin::add(UINT64_MAX, 1); }, "overflow");
    rejects([] { (void)bin::multiply(UINT64_MAX, 2); }, "overflow");
    CHECK(bin::ceil_div(UINT64_MAX, 2) == UINT64_MAX / 2 + 1);
    rejects([] { amak::DctPlan invalid(0); });
    rejects([] { amak::DctPlan invalid(257); });
    rejects([] { amak::forward_2d({1, 2}, 2, 2); }, "shape");
    rejects([] { amak::inverse_2d({1, 2}, 2, 2); }, "shape");

    for (const std::size_t n : {1U, 2U, 3U, 7U, 16U, 31U, 256U}) {
        amak::DctPlan dct(n);
        std::vector<double> x(n), f(n), y(n), ones(n, 1.0);
        for (std::size_t i = 0; i < n; ++i) x[i] = std::sin(static_cast<double>(i) * 0.31) + 0.2;
        dct.forward(x.data(), f.data());
        dct.inverse(f.data(), y.data());
        for (std::size_t i = 0; i < n; ++i) near(x[i], y[i], 1e-12);
        const auto energy = [](const std::vector<double>& v) { return std::inner_product(v.begin(), v.end(), v.begin(), 0.0); };
        near(energy(x), energy(f), 1e-10);
        dct.forward(ones.data(), f.data());
        near(f[0], std::sqrt(static_cast<double>(n)), 1e-12);
        for (std::size_t k = 1; k < n; ++k) near(f[k], 0, 1e-12);
    }
}

void test_dct_backends() {
    using amak::DctBackend;
    const bool available = amak::avx2_dct_available();
    CHECK(!available || amak::avx2_dct_compiled());
    const auto selected = available ? DctBackend::avx2 : DctBackend::scalar;
    CHECK(amak::resolve_dct_backend(DctBackend::automatic) == selected);
    CHECK(amak::resolve_dct_backend(DctBackend::scalar) == DctBackend::scalar);
    CHECK(std::string(amak::dct_backend_name(DctBackend::automatic)) == "auto");
    CHECK(amak::DctPlan(8).backend() == DctBackend::scalar); // encoder default must not drift
    CHECK(amak::DctPlan(8, DctBackend::automatic).backend() == selected);
    const auto invalid = static_cast<DctBackend>(99);
    rejects([&] { amak::resolve_dct_backend(invalid); }, "invalid DCT backend");
    rejects([&] { amak::DctPlan plan(8, invalid); }, "invalid DCT backend");
    if (!available) {
        rejects([] { amak::resolve_dct_backend(DctBackend::avx2); }, "unavailable");
        rejects([] { amak::DctPlan plan(8, DctBackend::avx2); }, "unavailable");
    } else {
        CHECK(amak::DctPlan(8, DctBackend::avx2).backend() == DctBackend::avx2);
    }

    // Every legal dimension: 4/8-lane boundaries, all odd tails, and maximum n.
    // Buffers end exactly after n live elements. Prefix offsets exercise unaligned
    // loads/stores and ASan detects any attempt to read or write a padded tail.
    constexpr double sentinel = 123456789.0;
    for (std::size_t n = 1; n <= 256; ++n) {
        const amak::DctPlan reference(n, DctBackend::scalar), accelerated(n, selected);
        CHECK(reference.storage_bytes() == accelerated.storage_bytes());
        std::vector<double> x_storage(n + 1, sentinel), a_storage(n + 1, sentinel),
                            b_storage(n + 1, sentinel), roundtrip_storage(n + 1, sentinel);
        std::vector<float> f_storage(n + 1, static_cast<float>(sentinel));
        auto* x = x_storage.data() + 1;
        auto* a = a_storage.data() + 1;
        auto* b = b_storage.data() + 1;
        auto* roundtrip = roundtrip_storage.data() + 1;
        auto* f = f_storage.data() + 1;
        if (available) {
            CHECK(reinterpret_cast<std::uintptr_t>(x) % 32 != 0);
            CHECK(reinterpret_cast<std::uintptr_t>(f) % 16 != 0);
        }
        double energy = 0;
        for (std::size_t j = 0; j < n; ++j) {
            x[j] = std::sin(static_cast<double>(j) * 0.37) +
                (j % 2 == 0 ? 0.3 : -0.3) * std::cos(static_cast<double>(n + j) * 0.19);
            energy += x[j] * x[j];
        }
        const double tolerance = 64 * std::numeric_limits<double>::epsilon() *
            static_cast<double>(n) * std::sqrt(energy);
        reference.forward(x, a);
        accelerated.forward(x, b);
        for (std::size_t j = 0; j < n; ++j) near(b[j], a[j], tolerance);
        accelerated.inverse(b, roundtrip);
        for (std::size_t j = 0; j < n; ++j) near(roundtrip[j], x[j], tolerance);
        reference.inverse(x, a);
        accelerated.inverse(x, b);
        for (std::size_t j = 0; j < n; ++j) near(b[j], a[j], tolerance);

        // Mixed-scale float activations, then subnormal float activations.
        for (const bool subnormal : {false, true}) {
            std::vector<double> converted(n), converted_reference(n);
            double float_energy = 0;
            for (std::size_t j = 0; j < n; ++j) {
                f[j] = subnormal ? std::numeric_limits<float>::denorm_min() * static_cast<float>(static_cast<int>(j % 7) - 3)
                                 : static_cast<float>(x[j] * (j % 3 == 0 ? 1e10 : 1e-8));
                converted[j] = f[j];
                float_energy += converted[j] * converted[j];
            }
            reference.forward(f, a);
            reference.forward(converted.data(), converted_reference.data());
            accelerated.forward(f, b);
            const double float_tolerance = 64 * std::numeric_limits<double>::epsilon() *
                static_cast<double>(n) * std::sqrt(float_energy);
            for (std::size_t j = 0; j < n; ++j) {
                CHECK(a[j] == converted_reference[j]);
                near(b[j], a[j], float_tolerance);
            }
        }
        CHECK(x_storage.front() == sentinel && a_storage.front() == sentinel && b_storage.front() == sentinel);
        CHECK(roundtrip_storage.front() == sentinel && f_storage.front() == static_cast<float>(sentinel));
    }
    // Large/cancelling doubles still use double accumulators, never a float narrowing.
    for (const std::size_t n : {7U, 32U, 65U, 256U}) {
        const amak::DctPlan reference(n, DctBackend::scalar), accelerated(n, selected);
        std::vector<double> x(n), a(n), b(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = i % 3 == 0 ? 1e100 : (i % 3 == 1 ? -1e100 : 1e-100);
        const double tolerance = 64 * std::numeric_limits<double>::epsilon() * static_cast<double>(n) *
            std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
        reference.forward(x.data(), a.data());
        accelerated.forward(x.data(), b.data());
        for (std::size_t i = 0; i < n; ++i) near(b[i], a[i], tolerance);
        reference.inverse(x.data(), a.data());
        accelerated.inverse(x.data(), b.data());
        for (std::size_t i = 0; i < n; ++i) near(b[i], a[i], tolerance);
    }
    std::cout << "  exercised DCT backend: " << amak::dct_backend_name(selected)
              << "; AVX2 compiled=" << amak::avx2_dct_compiled() << ", available=" << available << '\n';
}

void test_rectangular_identity() {
    constexpr std::size_t m = 5, n = 7;
    const auto t = source("w", m, n, m, n);
    const std::vector<double> w(t.weights.begin(), t.weights.end());
    const auto frequency = amak::forward_2d(w, m, n);
    const auto restored = amak::inverse_2d(frequency, m, n);
    for (std::size_t i = 0; i < w.size(); ++i) near(w[i], restored[i], 1e-12);
    const amak::DctPlan dm(m), dn(n);
    for (std::size_t b = 0; b < 3; ++b) {
        std::vector<double> x(n), x_hat(n), y_hat(m), y(m);
        for (std::size_t j = 0; j < n; ++j) x[j] = std::cos(static_cast<double>(b + 2 * j) * 0.3);
        dn.forward(x.data(), x_hat.data());
        for (std::size_t u = 0; u < m; ++u)
            for (std::size_t v = 0; v < n; ++v) y_hat[u] += frequency[u * n + v] * x_hat[v];
        dm.inverse(y_hat.data(), y.data());
        for (std::size_t r = 0; r < m; ++r) {
            double dense = 0;
            for (std::size_t c = 0; c < n; ++c) dense += w[r * n + c] * x[c];
            near(y[r], dense, 1e-12);
        }
    }
}

void test_rans() {
    auto round_trip = [](const std::vector<std::int8_t>& q) {
        const auto encoded = amak::rans::encode(q);
        CHECK(std::accumulate(encoded.frequencies.begin(), encoded.frequencies.end(), 0U) == 4096);
        CHECK(encoded.payload.size() >= 4 && encoded.payload.size() <= 2 * q.size() + 4);
        std::vector<std::int8_t> decoded(q.size());
        amak::rans::decode(encoded.frequencies, encoded.payload.data(), encoded.payload.size(), decoded.data(), decoded.size());
        CHECK(q == decoded);
        const auto again = amak::rans::encode(q);
        CHECK(again.frequencies == encoded.frequencies && again.payload == encoded.payload);
    };
    for (const auto q : {-128, -1, 0, 1, 127}) {
        const std::vector<std::int8_t> values(65536, static_cast<std::int8_t>(q));
        round_trip(values);
        const auto encoded = amak::rans::encode(values);
        CHECK(encoded.payload == std::vector<std::uint8_t>({0x00, 0x00, 0x80, 0x00}));
        CHECK(encoded.frequencies[static_cast<std::size_t>(q + 128)] == 4096);
    }
    // Independently derivable two-symbol fixture locks state byte order/CDF.
    const auto golden = amak::rans::encode({0, 1});
    CHECK(golden.frequencies[128] == 2048 && golden.frequencies[129] == 2048);
    CHECK(golden.payload == std::vector<std::uint8_t>({0x00, 0x10, 0x00, 0x02}));
    std::vector<std::int8_t> full_alphabet;
    for (int i = -128; i <= 127; ++i) full_alphabet.push_back(static_cast<std::int8_t>(i));
    round_trip(full_alphabet);
    std::mt19937 random(42);
    for (const std::size_t n : {1U, 2U, 7U, 255U, 1024U, 65536U}) {
        std::vector<std::int8_t> q(n);
        for (auto& value : q) value = static_cast<std::int8_t>(static_cast<int>(random() % 256) - 128);
        round_trip(q);
        for (auto& value : q) if ((random() % 10) != 0) value = 0;
        round_trip(q);
    }
    rejects([] { amak::rans::encode({}); });
    rejects([] { amak::rans::encode(std::vector<std::int8_t>(65537)); });

    auto encoded = amak::rans::encode(full_alphabet);
    std::vector<std::int8_t> output(full_alphabet.size());
    auto decode = [&](const amak::rans::Frequencies& f, const std::vector<std::uint8_t>& p) {
        amak::rans::decode(f, p.data(), p.size(), output.data(), output.size());
    };
    auto frequencies = encoded.frequencies;
    frequencies[0] = 0;
    rejects([&] { decode(frequencies, encoded.payload); }, "sum");
    frequencies = encoded.frequencies;
    frequencies[0] = 5000;
    rejects([&] { decode(frequencies, encoded.payload); }, "frequency");
    for (std::size_t n = 0; n < encoded.payload.size(); ++n) {
        const std::vector<std::uint8_t> truncated(encoded.payload.begin(), encoded.payload.begin() + static_cast<std::ptrdiff_t>(n));
        rejects([&] { decode(encoded.frequencies, truncated); });
    }
    auto payload = encoded.payload;
    payload.push_back(0);
    rejects([&] { decode(encoded.frequencies, payload); }, "unused");
    for (const auto invalid : {0U, amak::rans::kLowerBound - 1, amak::rans::kLowerBound << 8, UINT32_MAX}) {
        payload = encoded.payload;
        bin::put32(payload.data(), invalid);
        rejects([&] { decode(encoded.frequencies, payload); }, "initial state");
    }
    encoded = amak::rans::encode({0});
    payload = encoded.payload;
    bin::put32(payload.data(), amak::rans::kLowerBound + 1);
    rejects([&] { amak::rans::decode(encoded.frequencies, payload.data(), payload.size(), output.data(), 1); }, "terminal state");
    rejects([&] { amak::rans::decode(encoded.frequencies, nullptr, 4, output.data(), 1); }, "null");
    rejects([&] { amak::rans::decode(encoded.frequencies, payload.data(), 4, nullptr, 1); }, "null");
    rejects([&] { amak::rans::decode(encoded.frequencies, payload.data(), 4, output.data(), 0); });
    rejects([&] { amak::rans::decode(encoded.frequencies, payload.data(), 4, output.data(), 65537); });

    // Malformed payloads without a CRC: exercise the decoder itself under sanitizers.
    encoded = amak::rans::encode(full_alphabet);
    for (unsigned attempt = 0; attempt < 300; ++attempt) {
        payload = encoded.payload;
        payload[random() % payload.size()] ^= static_cast<std::uint8_t>(1U << (random() % 8));
        try { decode(encoded.frequencies, payload); } catch (const std::runtime_error&) {}
    }
}

std::vector<double> dense_quantized(const amak::Container& container, std::size_t index,
                                    const amak::TensorSource& original, double& squared_bound) {
    const auto& t = container.tensors()[index];
    std::vector<double> dense(static_cast<std::size_t>(t.rows * t.cols));
    double frequency_error = 0;
    squared_bound = 0;
    container.visit_tiles(index, [&](const amak::DecodedTile& tile) {
        std::vector<double> frequency(tile.coefficients.size()), spatial_source(tile.coefficients.size());
        for (std::size_t r = 0; r < tile.rows; ++r)
            for (std::size_t c = 0; c < tile.cols; ++c)
                spatial_source[r * tile.cols + c] = original.weights[static_cast<std::size_t>(
                    (tile.row_start + r) * t.cols + tile.col_start + c)];
        const auto source_frequency = amak::forward_2d(spatial_source, tile.rows, tile.cols);
        for (std::size_t k = 0; k < frequency.size(); ++k) {
            CHECK(tile.coefficients[k] >= -127);
            frequency[k] = static_cast<double>(tile.delta) * tile.coefficients[k];
            const auto error = frequency[k] - source_frequency[k];
            CHECK(std::abs(error) <= static_cast<double>(tile.delta) * 0.500001 + 1e-12);
            frequency_error += error * error;
        }
        squared_bound += static_cast<double>(frequency.size()) * tile.delta * tile.delta / 4;
        const auto spatial = amak::inverse_2d(frequency, tile.rows, tile.cols);
        for (std::size_t r = 0; r < tile.rows; ++r)
            for (std::size_t c = 0; c < tile.cols; ++c)
                dense[static_cast<std::size_t>((tile.row_start + r) * t.cols + tile.col_start + c)] = spatial[r * tile.cols + c];
    }, 0);
    double spatial_error = 0;
    for (std::size_t k = 0; k < dense.size(); ++k) {
        const auto error = dense[k] - original.weights[k];
        spatial_error += error * error;
    }
    near(spatial_error, frequency_error, 1e-11);
    CHECK(spatial_error <= squared_bound + 1e-11);
    return dense;
}

void test_container_and_forward() {
    TempDirectory tmp;
    const auto path = tmp.file("multi.amak");
    std::vector<amak::TensorSource> sources{
        source("rect", 5, 7, 3, 4), source("row", 1, 9, 4, 3),
        source("col", 9, 1, 4, 3), source("divisible", 4, 6, 2, 3),
        source("scalar", 1, 1, 1, 1), source("oversize_tile", 3, 2, 5, 8),
        source("edges", 11, 13, 4, 5), source("max_dimension", 256, 2, 256, 256),
        source("zero", 3, 4, 2, 3), source("fixed_step", 7, 5, 3, 2),
        source("max_area_edges", 257, 259, 256, 256)};
    std::fill(sources[8].weights.begin(), sources[8].weights.end(), 0.0F);
    sources[9].quantization_step = 0.025F;
    amak::write_container(path, sources);
    amak::write_container(tmp.file("repeat.amak"), sources);
    CHECK(read_bytes(path) == read_bytes(tmp.file("repeat.amak")));
    const amak::Container container(path);
    CHECK(container.tensors().size() == sources.size());
    CHECK(container.file_bytes() == fs::file_size(path));
    rejects([&] { container.find_tensor("missing"); }, "unknown tensor");
    for (std::size_t index = 0; index < sources.size(); ++index) {
        const auto& original = sources[index];
        const auto& info = container.tensors()[index];
        CHECK(container.find_tensor(original.name) == index);
        CHECK(info.rows == original.rows && info.cols == original.cols);
        double bound = 0;
        const auto dense = dense_quantized(container, index, original, bound);
        const auto m = static_cast<std::size_t>(info.rows), n = static_cast<std::size_t>(info.cols);
        constexpr std::size_t batch = 3;
        std::vector<float> x(batch * n), bias(m);
        for (std::size_t k = 0; k < x.size(); ++k) x[k] = static_cast<float>(std::sin(static_cast<double>(k) * 0.41));
        for (std::size_t k = 0; k < bias.size(); ++k) bias[k] = static_cast<float>(0.03 * static_cast<double>(k));
        const auto sync = amak::linear(container, index, x, {batch, 0}, bias);
        const auto scalar = amak::linear(container, index, x, {batch, 0, amak::DctBackend::scalar}, bias);
        CHECK(sync.dct_backend == amak::resolve_dct_backend(amak::DctBackend::automatic));
        CHECK(scalar.dct_backend == amak::DctBackend::scalar);
        CHECK(sync.transform_plan_bytes == scalar.transform_plan_bytes);
        CHECK(sync.activation_scratch_bytes == sizeof(double) * (
            batch * (std::min<std::size_t>(m, info.tile_rows) + std::min<std::size_t>(n, info.tile_cols)) +
            std::min<std::size_t>(m, info.tile_rows)));
        for (std::size_t i = 0; i < sync.output.size(); ++i)
            near(sync.output[i], scalar.output[i], 2e-6 * std::max(1.0, std::abs(static_cast<double>(scalar.output[i]))));
        for (const std::size_t depth : {1U, 2U, 4U, 8U}) {
            const auto async = amak::linear(container, index, x, {batch, depth}, bias);
            CHECK(sync.output == async.output);
            CHECK(async.stream.slot_count == std::min<std::uint64_t>(depth, info.chunk_count));
            CHECK(async.stream.chunks_decoded == info.chunk_count);
            CHECK(async.stream.coefficients_decoded == info.rows * info.cols);
            CHECK(async.stream.encoded_bytes_read == info.data_bytes);
            CHECK(async.stream.coefficient_capacity_bytes == async.stream.slot_count * info.tile_rows * info.tile_cols);
        }
        for (std::size_t b = 0; b < batch; ++b) {
            double output_error = 0, input_energy = 0;
            for (std::size_t c = 0; c < n; ++c) input_energy += static_cast<double>(x[b * n + c]) * x[b * n + c];
            for (std::size_t r = 0; r < m; ++r) {
                double expected = bias[r], unquantized = bias[r];
                for (std::size_t c = 0; c < n; ++c) {
                    expected += dense[r * n + c] * x[b * n + c];
                    unquantized += static_cast<double>(original.weights[r * n + c]) * x[b * n + c];
                }
                near(sync.output[b * m + r], expected, 1e-6 * std::max(1.0, std::abs(expected)));
                const auto error = static_cast<double>(sync.output[b * m + r]) - unquantized;
                output_error += error * error;
            }
            CHECK(std::sqrt(output_error) <= std::sqrt(bound * input_energy) + 1e-5);
        }
    }
    container.visit_tiles(8, [](const amak::DecodedTile& tile) {
        CHECK(tile.delta == 1.0F);
        for (const auto q : tile.coefficients) CHECK(q == 0);
    });
    container.visit_tiles(9, [](const amak::DecodedTile& tile) { CHECK(tile.delta == 0.025F); });
}

void test_stream_lifecycle_and_bounds() {
    TempDirectory tmp;
    const auto path = tmp.file("stream.amak");
    amak::write_container(path, {source("w", 33, 35, 4, 4)});
    amak::Container container(path);
    struct CallbackFailure : std::runtime_error { CallbackFailure() : std::runtime_error("callback failed") {} };
    for (const std::size_t depth : {0U, 1U, 2U, 8U}) {
        std::uint64_t ordinal = 0;
        std::set<const void*> buffers;
        const auto stats = container.visit_tiles(0, [&](const amak::DecodedTile& tile) {
            CHECK(tile.index == ordinal++);
            CHECK(tile.coefficients.size() == std::size_t(tile.rows) * tile.cols);
            buffers.insert(tile.coefficients.data());
        }, depth);
        CHECK(ordinal == container.tensors()[0].chunk_count);
        CHECK(buffers.size() == stats.slot_count);
        CHECK(stats.coefficient_capacity_bytes == (depth == 0 ? 1 : depth) * 16);
        CHECK(stats.decoder_scratch_bytes == 64 + 4 * 256 + 2 * 16 + 4 + amak::rans::kDecodeTableBytes);
        for (const auto at : {0ULL, 3ULL, 80ULL}) {
            bool propagated = false;
            try {
                container.visit_tiles(0, [&](const amak::DecodedTile& tile) {
                    if (tile.index == at) throw CallbackFailure();
                }, depth);
            } catch (const CallbackFailure&) { propagated = true; }
            CHECK(propagated);
        }
    }
    const std::vector<float> input(35, 0.25F);
    const auto expected = amak::linear(container, 0, input).output;
    for (int repeat = 0; repeat < 10; ++repeat) {
        auto first = std::async(std::launch::async, [&] { return amak::linear(container, 0, input, {1, 2}).output; });
        auto second = std::async(std::launch::async, [&] { return amak::linear(container, 0, input, {1, 1}).output; });
        CHECK(first.get() == expected && second.get() == expected);
    }
    // Larger parameter count does not grow coefficient or decoder buffers.
    const auto small = tmp.file("small.amak");
    amak::write_container(small, {source("w", 8, 8, 4, 4)});
    const auto a = amak::Container(small).visit_tiles(0, [](const amak::DecodedTile&) {}, 2);
    const auto b = container.visit_tiles(0, [](const amak::DecodedTile&) {}, 2);
    CHECK(a.coefficient_capacity_bytes == b.coefficient_capacity_bytes);
    CHECK(a.decoder_scratch_bytes == b.decoder_scratch_bytes);
}

void test_invalid_files() {
    TempDirectory tmp;
    const auto path = tmp.file("valid.amak"), bad = tmp.file("bad.amak");
    amak::write_container(path, {source("w", 5, 7, 3, 4)});
    const auto valid = read_bytes(path);
    const auto first_chunk = bin::as_size(bin::u64(valid.data() + 64 + 88));
    CHECK(first_chunk == 192);
    auto bad_open = [&](std::vector<std::uint8_t> bytes) {
        write_bytes(bad, bytes);
        rejects([&] { amak::Container reader(bad); });
    };
    auto bad_stream = [&](std::vector<std::uint8_t> bytes) {
        write_bytes(bad, bytes);
        const amak::Container reader(bad); // lazy opening deliberately still succeeds
        for (const std::size_t depth : {0U, 1U, 2U, 8U})
            rejects([&] { reader.visit_tiles(0, [](const amak::DecodedTile&) {}, depth); });
    };
    for (std::size_t n = 0; n < valid.size(); ++n)
        bad_open(std::vector<std::uint8_t>(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(n)));
    auto bytes = valid;
    bytes.push_back(0);
    bad_open(bytes);
    for (const std::size_t offset : {0U, 8U, 16U, 20U, 32U, 48U, 52U, 56U, 64U, 130U}) {
        bytes = valid; bytes[offset] ^= 1; bad_open(bytes);
    }
    for (const std::size_t offset : {8U, 10U, 12U, 16U, 56U}) {
        bytes = valid; bytes[offset] ^= 1; fix_header_crc(bytes); bad_open(bytes);
    }
    for (const std::size_t offset : {20U, 24U, 32U, 40U}) {
        bytes = valid; bin::put32(bytes.data() + offset, UINT32_MAX); fix_header_crc(bytes); bad_open(bytes);
    }
    bytes = valid; bin::put32(bytes.data() + 20, 0); fix_header_crc(bytes); bad_open(bytes);
    for (const std::size_t offset : {48U, 50U, 51U, 52U, 53U, 54U, 104U, 127U}) {
        bytes = valid; bytes[64 + offset] ^= 1; fix_directory_crc(bytes); bad_open(bytes);
    }
    bytes = valid; bytes[64] = 0; fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; bytes[64] = ' '; fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; bytes[64 + 2] = 'x'; fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; std::fill(bytes.begin() + 64, bytes.begin() + 112, 'x'); fix_directory_crc(bytes); bad_open(bytes);
    for (const std::size_t offset : {56U, 64U, 72U, 76U, 80U}) {
        bytes = valid; bin::put64(bytes.data() + 64 + offset, 0); fix_directory_crc(bytes); bad_open(bytes);
    }
    bytes = valid; bin::put64(bytes.data() + 64 + 56, UINT64_MAX); fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; bin::put32(bytes.data() + 64 + 72, 257); fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; bin::put64(bytes.data() + 64 + 88, 64); fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; bin::put64(bytes.data() + 64 + 96, UINT64_MAX); fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; bin::put64(bytes.data() + 64 + 96, 1); fix_directory_crc(bytes); bad_open(bytes);
    bytes = valid; bin::put64(bytes.data() + 64 + 96, valid.size() - first_chunk - 1); fix_directory_crc(bytes); bad_open(bytes);

    for (const std::size_t offset : {0U, 4U, 6U, 8U, 16U, 24U, 32U, 34U, 40U, 46U, 47U, 60U}) {
        bytes = valid; bytes[first_chunk + offset] ^= 1;
        fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    }
    for (const float step : {0.0F, -1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        bytes = valid; bin::put_float(bytes.data() + first_chunk + 36, step);
        fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    }
    for (const auto alphabet : {0U, 257U}) {
        bytes = valid; bin::put16(bytes.data() + first_chunk + 44, static_cast<std::uint16_t>(alphabet));
        fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    }
    for (const auto payload : {0U, 3U, UINT32_MAX}) {
        bytes = valid; bin::put32(bytes.data() + first_chunk + 48, payload);
        fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    }
    bytes = valid; bin::put32(bytes.data() + first_chunk + 52, UINT32_MAX); bad_stream(bytes);
    const auto alphabet = bin::u16(valid.data() + first_chunk + 44);
    CHECK(alphabet >= 2);
    bytes = valid; bytes[first_chunk + 64 + 1] = 1; fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    bytes = valid; bytes[first_chunk + 64 + 4] = bytes[first_chunk + 64]; fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    bytes = valid; bin::put16(bytes.data() + first_chunk + 64 + 2, 0); fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    bytes = valid; bin::put16(bytes.data() + first_chunk + 64 + 2, 5000); fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    bytes = valid;
    const auto f = bin::u16(bytes.data() + first_chunk + 64 + 2);
    bin::put16(bytes.data() + first_chunk + 64 + 2, static_cast<std::uint16_t>(f + 1));
    fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    bytes = valid; bin::put32(bytes.data() + first_chunk + 64 + 4 * alphabet, 0);
    fix_chunk_crc(bytes, first_chunk); bad_stream(bytes);
    bytes = valid; bytes.back() ^= 1; bad_stream(bytes); // producer error near final chunk
    bytes = valid; bytes[first_chunk + 56] ^= 1; bad_stream(bytes);

    // A producer can fail after publishing the last tile: finish() must propagate.
    bytes = valid; bytes.push_back(0);
    bin::put64(bytes.data() + 40, bytes.size());
    bin::put64(bytes.data() + 64 + 96, bytes.size() - first_chunk);
    fix_directory_crc(bytes); bad_stream(bytes);

    // Arithmetic overflow is rejected even when caller increases dimension limits.
    bytes = valid;
    bin::put64(bytes.data() + 64 + 56, UINT64_MAX);
    bin::put64(bytes.data() + 64 + 64, 2);
    fix_directory_crc(bytes); write_bytes(bad, bytes);
    amak::Limits huge;
    huge.max_dimension = UINT64_MAX;
    rejects([&] { amak::Container reader(bad, huge); }, "overflow");

    // Directory names/extent relationships across more than one tensor.
    amak::write_container(path, {source("a", 2, 2, 2, 2), source("b", 2, 2, 2, 2)});
    bytes = read_bytes(path); bytes[64 + 128] = 'a'; fix_directory_crc(bytes); bad_open(bytes);
    bytes = read_bytes(path); bin::put64(bytes.data() + 64 + 128 + 88, 320); fix_directory_crc(bytes); bad_open(bytes);
    amak::Container before_mutation(path);
    fs::resize_file(path, fs::file_size(path) - 1);
    rejects([&] { before_mutation.visit_tiles(0, [](const amak::DecodedTile&) {}); }, "length changed");
}

void test_validation_and_atomic_writer() {
    TempDirectory tmp;
    const auto path = tmp.file("w.amak");
    const auto good = source("w", 2, 3, 2, 2);
    amak::write_container(path, {good});
    const auto bytes = read_bytes(path);
    const amak::Container container(path);
    rejects([&] { amak::write_container(path, {}); });
    rejects([&] { amak::write_container(path, {good, good}); }, "duplicate");
    for (const auto name : {"", "with space", "newline\n", "012345678901234567890123456789012345678901234567890"}) {
        auto bad = good; bad.name = name; rejects([&] { amak::write_container(path, {bad}); });
    }
    auto bad = good; bad.rows = 0; rejects([&] { amak::write_container(path, {bad}); });
    bad = good; bad.tile_rows = 0; rejects([&] { amak::write_container(path, {bad}); });
    bad = good; bad.tile_cols = 257; rejects([&] { amak::write_container(path, {bad}); });
    bad = good; bad.weights.pop_back(); rejects([&] { amak::write_container(path, {bad}); }, "shape");
    for (const auto invalid : {-1.0F, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        bad = good; bad.quantization_step = invalid; rejects([&] { amak::write_container(path, {bad}); }, "step");
    }
    for (const auto invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        bad = good; bad.weights[0] = invalid; rejects([&] { amak::write_container(path, {bad}); }, "finite");
    }
    bad = good; bad.quantization_step = 1e-10F;
    rejects([&] { amak::write_container(path, {bad}); }, "clip"); // fails during staged encoding
    bad = source("huge", 256, 256, 256, 256);
    std::fill(bad.weights.begin(), bad.weights.end(), std::numeric_limits<float>::max());
    rejects([&] { amak::write_container(path, {bad}); }, "step cannot fit");
    CHECK(read_bytes(path) == bytes); // existing destination must survive every failure
    for (const auto& item : fs::directory_iterator(tmp.path())) CHECK(item.path().filename() == "w.amak");

    const std::vector<float> input(3, 1.0F);
    rejects([&] { container.visit_tiles(9, [](const amak::DecodedTile&) {}); }, "index");
    rejects([&] { container.visit_tiles(0, {}); }, "empty");
    rejects([&] { container.visit_tiles(0, [](const amak::DecodedTile&) {}, 9); }, "prefetch");
    rejects([&] { amak::linear(container, 9, input); }, "index");
    rejects([&] { amak::linear(container, 0, input, {0, 2}); }, "batch");
    rejects([&] { amak::linear(container, 0, input, {1, 9}); }, "prefetch");
    rejects([&] { amak::linear(container, 0, input, {1, 0, static_cast<amak::DctBackend>(99)}); }, "invalid DCT backend");
    if (!amak::avx2_dct_available())
        rejects([&] { amak::linear(container, 0, input, {1, 0, amak::DctBackend::avx2}); }, "unavailable");
    rejects([&] { amak::linear(container, 0, {}, {1, 0}); }, "shape");
    rejects([&] { amak::linear(container, 0, input, {}, {1}); }, "bias");
    rejects([&] { amak::linear(container, 0, input, {std::numeric_limits<std::size_t>::max(), 0}); }, "overflow");
    auto nonfinite = input; nonfinite[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { amak::linear(container, 0, nonfinite); }, "finite");
    rejects([&] { amak::linear(container, 0, input, {}, {0, std::numeric_limits<float>::infinity()}); }, "finite");
    amak::Limits limits;
    limits.max_activation_elements = 2;
    rejects([&] { amak::linear(amak::Container(path, limits), 0, input); }, "limit");
    limits = {}; limits.max_dimension = 2;
    rejects([&] { amak::Container reader(path, limits); }, "dimension");
    limits = {}; limits.max_tensors = 0;
    rejects([&] { amak::Container reader(path, limits); }, "count");
    limits = {}; limits.max_chunks_per_tensor = 1;
    rejects([&] { amak::Container reader(path, limits); }, "chunk count");

    auto scalar = source("scalar", 1, 1, 1, 1);
    for (const auto value : {-0.5F, 0.5F}) {
        scalar.weights[0] = value; scalar.quantization_step = 1;
        amak::write_container(path, {scalar});
        amak::Container(path).visit_tiles(0, [&](const amak::DecodedTile& tile) {
            CHECK(tile.coefficients[0] == (value < 0 ? -1 : 1));
        });
    }
    scalar.quantization_step = 0;
    scalar.weights[0] = std::numeric_limits<float>::denorm_min();
    amak::write_container(path, {scalar});
    CHECK(amak::linear(amak::Container(path), 0, {1}).output[0] == scalar.weights[0]);
    scalar.weights[0] = 2;
    amak::write_container(path, {scalar});
    rejects([&] { amak::linear(amak::Container(path), 0, {std::numeric_limits<float>::max()}); }, "overflows");
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"little-endian, CRC, DCT roundtrip and Parseval", test_binary_and_dct},
        {"DCT backend dispatch, all 1..256 sizes, unaligned tails and precision", test_dct_backends},
        {"rectangular DCT-domain multiplication identity", test_rectangular_identity},
        {"rANS roundtrips, golden states and malformed payloads", test_rans},
        {"multi-tensor streaming vs quantized dense and quantization bounds", test_container_and_forward},
        {"bounded ring reuse, cancellation and concurrent visits", test_stream_lifecycle_and_bounds},
        {"truncation, checksums, adversarial framing and producer failures", test_invalid_files},
        {"API limits, quantizer edges and atomic output publication", test_validation_and_atomic_writer},
    };
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "[pass] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }
    std::cout << tests.size() << " test groups passed\n";
    return 0;
}
