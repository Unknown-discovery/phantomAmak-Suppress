#include "amak/amak.hpp"
#include "binary.hpp"
#include "benchmark.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
using amak::detail::require;
using Clock = std::chrono::steady_clock;
constexpr std::size_t kCliMaxElements = 1ULL << 24;

void usage() {
    std::cout <<
        "AMAK 1.0 experimental CPU linear runtime\n\n"
        "  amak backends                   (CPU/build dispatch capabilities)\n"
        "  amak bench-dct [--size N --iterations I --warmup W]\n"
        "    --dct-backend auto|scalar|avx2 (default: auto; size=64, I=2000, W=100)\n"
        "  amak pack WEIGHTS.f32 MODEL.amak --rows M --cols N [options]\n"
        "    --tile-rows R --tile-cols C  (defaults: 32, 32; maximum: 256)\n"
        "    --name NAME                  (default: linear.weight)\n"
        "    --step DELTA                 (default: automatic per tile)\n"
        "  amak inspect MODEL.amak        (header/directory only)\n"
        "  amak verify MODEL.amak         (decode and validate every chunk)\n"
        "  amak run MODEL.amak INPUT.f32 OUTPUT.f32 [options]\n"
        "    --batch B --prefetch P        (defaults: 1, 2; P=0 is synchronous)\n"
        "    --tensor NAME                 (default: first tensor)\n"
        "    --dct-backend auto|scalar|avx2 (default: auto; coefficient multiply stays scalar)\n"
        "  amak demo MODEL.amak [options]\n"
        "    --rows M --cols N --tile-rows R --tile-cols C --batch B\n"
        "    --iterations I               (defaults: 48, 64, 16, 16, 2, 3)\n"
        "    --dct-backend auto|scalar|avx2 (default: auto)\n\n"
        "Raw .f32 files are little-endian, row-major IEEE-754 float32, without\n"
        "headers. W=[M,N], X=[B,N], Y=[B,M]. This is not a transformer runner.\n";
}

class Options {
public:
    Options(int argc, char** argv, int first, std::set<std::string> allowed) {
        for (int i = first; i < argc; i += 2) {
            const std::string key = argv[i];
            require(allowed.count(key) != 0, "unknown option: " + key);
            require(i + 1 < argc, "missing value for " + key);
            require(values_.emplace(key, argv[i + 1]).second, "duplicate option: " + key);
        }
    }
    std::string text(const std::string& key, const std::string& fallback = "") const {
        const auto found = values_.find(key);
        return found == values_.end() ? fallback : found->second;
    }
    std::uint64_t number(const std::string& key, std::uint64_t fallback) const {
        const auto value = text(key);
        if (value.empty()) {
            require(values_.count(key) == 0, "empty numeric option: " + key);
            return fallback;
        }
        std::uint64_t parsed = 0;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        require(result.ec == std::errc() && result.ptr == value.data() + value.size(),
                "invalid unsigned integer for " + key + ": " + value);
        return parsed;
    }
    float step() const {
        const auto value = text("--step");
        if (values_.count("--step") == 0) return 0;
        char* end = nullptr;
        const double parsed = std::strtod(value.c_str(), &end);
        require(end != value.c_str() && end == value.c_str() + value.size() &&
                std::isfinite(parsed) && parsed > 0 && parsed <= std::numeric_limits<float>::max(),
                "--step must be finite, positive and representable in float32");
        const auto result = static_cast<float>(parsed);
        require(result > 0, "--step underflows float32");
        return result;
    }

    amak::DctBackend backend() const {
        const auto name = text("--dct-backend", "auto");
        if (name == "auto") return amak::DctBackend::automatic;
        if (name == "scalar") return amak::DctBackend::scalar;
        if (name == "avx2") return amak::DctBackend::avx2;
        throw std::runtime_error("amak: --dct-backend must be auto, scalar or avx2");
    }

private:
    std::map<std::string, std::string> values_;
};

std::size_t bounded_count(std::uint64_t a, std::uint64_t b) {
    const auto count = amak::detail::multiply(a, b);
    require(a > 0 && b > 0 && count <= kCliMaxElements, "CLI array limit exceeded (1..16777216 elements)");
    return amak::detail::as_size(count);
}

std::vector<float> read_f32(const std::string& path, std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "cannot open float input: " + path);
    require(amak::detail::stream_size(in) == amak::detail::multiply(count, 4),
            "raw float file length does not match its shape: " + path);
    std::vector<float> values(count);
    std::array<std::uint8_t, 4096> bytes{};
    for (std::size_t begin = 0; begin < count;) {
        const auto n = std::min<std::size_t>(bytes.size() / 4, count - begin);
        amak::detail::read_exact(in, bytes.data(), n * 4);
        for (std::size_t j = 0; j < n; ++j) values[begin + j] = amak::detail::f32(bytes.data() + 4 * j);
        begin += n;
    }
    return values;
}

void write_f32(const std::string& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(out.is_open(), "cannot open float output: " + path);
    std::array<std::uint8_t, 4096> bytes{};
    for (std::size_t begin = 0; begin < values.size();) {
        const auto n = std::min<std::size_t>(bytes.size() / 4, values.size() - begin);
        for (std::size_t j = 0; j < n; ++j) amak::detail::put_float(bytes.data() + 4 * j, values[begin + j]);
        amak::detail::write_exact(out, bytes.data(), n * 4);
        begin += n;
    }
    out.close();
    require(!out.fail(), "cannot finish float output");
}

void require_distinct(const std::string& input, const std::string& output) {
    namespace fs = std::filesystem;
    std::error_code error;
    const bool equivalent = fs::equivalent(input, output, error);
    require(!equivalent, "output must not overwrite an input file");
    require(fs::weakly_canonical(input) != fs::weakly_canonical(output),
            "output must not overwrite an input path");
}

amak::TensorSource source_options(const Options& options, bool demo) {
    amak::TensorSource source;
    source.name = options.text("--name", "linear.weight");
    source.rows = options.number("--rows", demo ? 48 : 0);
    source.cols = options.number("--cols", demo ? 64 : 0);
    const auto r = options.number("--tile-rows", demo ? 16 : 32);
    const auto c = options.number("--tile-cols", demo ? 16 : 32);
    require(r > 0 && r <= 256 && c > 0 && c <= 256, "tile dimensions must be in 1..256");
    source.tile_rows = static_cast<std::uint32_t>(r);
    source.tile_cols = static_cast<std::uint32_t>(c);
    source.quantization_step = options.step();
    (void)bounded_count(source.rows, source.cols);
    require(source.rows <= amak::Limits{}.max_dimension && source.cols <= amak::Limits{}.max_dimension,
            "source dimensions exceed default reader limits");
    return source;
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void print_memory(const amak::RunResult& result) {
    std::cout << "  DCT backend: " << amak::dct_backend_name(result.dct_backend)
              << " (coefficient multiply: scalar)\n"
              << "  coefficient slots: " << result.stream.slot_count
              << " (" << result.stream.coefficient_capacity_bytes << " bytes total)\n"
              << "  decoder buffer/tables: " << result.stream.decoder_scratch_bytes << " bytes\n"
              << "  DCT plans: " << result.transform_plan_bytes << " bytes\n"
              << "  activation scratch (frequency + spatial tile): " << result.activation_scratch_bytes << " bytes\n"
              << "  spatial output: " << result.output.size() * sizeof(float) << " bytes\n";
}

void demo(const std::string& path, const Options& options) {
    auto source = source_options(options, true);
    const auto backend = amak::resolve_dct_backend(options.backend());
    const auto batch = amak::detail::as_size(options.number("--batch", 2));
    const auto iterations = options.number("--iterations", 3);
    require(iterations > 0 && iterations <= 10000, "iterations must be in 1..10000");
    const auto input_count = bounded_count(batch, source.cols);
    (void)bounded_count(batch, source.rows);
    source.weights.resize(bounded_count(source.rows, source.cols));
    std::mt19937 random(7);
    std::uniform_real_distribution<float> weight(-0.125F, 0.125F), activation(-1.0F, 1.0F);
    for (auto& value : source.weights) value = weight(random);
    std::vector<float> input(input_count);
    for (auto& value : input) value = activation(random);
    const std::vector<amak::TensorSource> tensors{std::move(source)};
    const auto& w = tensors.front();
    auto start = Clock::now();
    amak::write_container(path, tensors); // scalar offline DCT, independent of execution backend
    const auto pack_ms = elapsed_ms(start);
    start = Clock::now();
    amak::Container container(path);
    const auto open_ms = elapsed_ms(start);
    auto timed_forward = [&](amak::DctBackend which, std::size_t prefetch) {
        const amak::RunOptions run_options{batch, prefetch, which};
        auto result = amak::linear(container, 0, input, run_options); // untimed warmup
        const auto begin = Clock::now();
        for (std::uint64_t i = 0; i < iterations; ++i)
            result = amak::linear(container, 0, input, run_options);
        const auto ms = elapsed_ms(begin) / static_cast<double>(iterations);
        return std::make_pair(std::move(result), ms);
    };
    const auto reference = timed_forward(amak::DctBackend::scalar, 0);
    const auto synchronous = backend == amak::DctBackend::scalar ? reference : timed_forward(backend, 0);
    const auto prefetched = timed_forward(backend, 2);
    require(synchronous.first.output == prefetched.first.output, "synchronous/prefetch output mismatch");
    double backend_error = 0;
    for (std::size_t i = 0; i < reference.first.output.size(); ++i) {
        const double a = reference.first.output[i], b = synchronous.first.output[i];
        backend_error = std::max(backend_error, std::abs(a - b));
        require(std::abs(a - b) <= 2e-6 * std::max(1.0, std::abs(a)), "DCT backend numerical mismatch");
    }
    double square_error = 0, maximum_error = 0;
    const auto m = static_cast<std::size_t>(w.rows), n = static_cast<std::size_t>(w.cols);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t r = 0; r < m; ++r) {
            double dense_source = 0;
            for (std::size_t c = 0; c < n; ++c)
                dense_source += static_cast<double>(w.weights[r * n + c]) * input[b * n + c];
            const auto error = static_cast<double>(prefetched.first.output[b * m + r]) - dense_source;
            square_error += error * error;
            maximum_error = std::max(maximum_error, std::abs(error));
        }
    }
    std::cout << std::fixed << std::setprecision(6)
              << "Synthetic linear demo: W=[" << m << ',' << n << "], batch=" << batch << '\n'
              << "Selected DCT backend: " << amak::dct_backend_name(backend) << '\n'
              << "Container: " << path << " (" << container.file_bytes() << " bytes)\n"
              << "Source f32 bytes: " << w.weights.size() * 4
              << "; coefficient int8 bytes before entropy coding: " << w.weights.size() << '\n'
              << "Offline pack: " << pack_ms << " ms; metadata open: " << open_ms << " ms\n"
              << "Mean scalar-DCT synchronous forward: " << reference.second << " ms\n"
              << "Mean selected-DCT synchronous forward: " << synchronous.second << " ms\n"
              << "Mean selected-DCT 2-slot prefetched forward: " << prefetched.second << " ms\n"
              << "Synchronous/prefetch outputs: identical\n"
              << "Scalar/selected max absolute difference: " << std::scientific << backend_error << '\n'
              << std::fixed << "Against original weights (includes quantization error):\n"
              << "  RMSE: " << std::sqrt(square_error / static_cast<double>(prefetched.first.output.size()))
              << "; max absolute error: " << maximum_error << '\n';
    print_memory(prefetched.first);
    std::cout << "Memory numbers describe explicit runtime buffers, not process RSS.\n"
              << "Demo retains the dense source for validation; run does not.\n"
              << "Warm forwards include per-call plans/I/O/thread setup; no guaranteed speedup.\n"
              << "Use bench-dct to isolate warmed transform kernels from full-forward overhead.\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") {
            usage();
            return 0;
        }
        const std::string command = argv[1];
        if (command == "backends") {
            require(argc == 2, "usage: amak backends");
            std::cout << "AVX2 DCT compiled: " << (amak::avx2_dct_compiled() ? "yes" : "no") << '\n'
                      << "AVX2 DCT available: " << (amak::avx2_dct_available() ? "yes" : "no") << '\n'
                      << "Automatic DCT backend: " << amak::dct_backend_name(
                          amak::resolve_dct_backend(amak::DctBackend::automatic)) << '\n'
                      << "Coefficient multiply: scalar; file format: AMAK 1.0\n";
        } else if (command == "bench-dct") {
            const Options options(argc, argv, 2, {"--size", "--iterations", "--warmup", "--dct-backend"});
            amak::cli::benchmark_dct(std::cout, amak::detail::as_size(options.number("--size", 64)),
                                     options.number("--iterations", 2000), options.number("--warmup", 100),
                                     options.backend());
        } else if (command == "inspect" || command == "verify") {
            require(argc == 3, "usage: amak " + command + " MODEL.amak");
            amak::Container container(argv[2]);
            std::cout << "AMAK 1.0 | " << container.file_bytes() << " bytes | "
                      << container.tensors().size() << " tensor(s)\n";
            for (std::size_t i = 0; i < container.tensors().size(); ++i) {
                const auto& t = container.tensors()[i];
                std::cout << "[" << i << "] " << t.name << " W=[" << t.rows << ',' << t.cols
                          << "] tile=[" << t.tile_rows << ',' << t.tile_cols << "] chunks=" << t.chunk_count
                          << " offset=" << t.data_offset << " bytes=" << t.data_bytes << '\n';
                if (command == "verify") container.visit_tiles(i, [](const amak::DecodedTile&) {}, 0);
            }
            std::cout << (command == "verify" ? "All chunk checksums and rANS streams validated.\n"
                                             : "Only header/directory validated; use verify to scan coefficients.\n");
        } else if (command == "pack") {
            require(argc >= 4, "usage: amak pack WEIGHTS.f32 MODEL.amak --rows M --cols N");
            const Options options(argc, argv, 4, {"--rows", "--cols", "--tile-rows", "--tile-cols", "--name", "--step"});
            require_distinct(argv[2], argv[3]);
            auto source = source_options(options, false);
            source.weights = read_f32(argv[2], bounded_count(source.rows, source.cols));
            const std::vector<amak::TensorSource> tensors{std::move(source)};
            amak::write_container(argv[3], tensors);
            std::cout << "Wrote " << argv[3] << " (" << amak::Container(argv[3]).file_bytes() << " bytes)\n";
        } else if (command == "run") {
            require(argc >= 5, "usage: amak run MODEL.amak INPUT.f32 OUTPUT.f32 [--batch B] [--prefetch P]");
            const Options options(argc, argv, 5, {"--batch", "--prefetch", "--tensor", "--dct-backend"});
            require_distinct(argv[2], argv[4]);
            require_distinct(argv[3], argv[4]);
            amak::Container container(argv[2]);
            const auto name = options.text("--tensor", container.tensors().front().name);
            const auto index = container.find_tensor(name);
            const auto batch = amak::detail::as_size(options.number("--batch", 1));
            const auto prefetch = amak::detail::as_size(options.number("--prefetch", 2));
            require(prefetch <= amak::kMaxPrefetchSlots, "prefetch must be in 0..8");
            (void)bounded_count(batch, container.tensors()[index].rows);
            const auto input = read_f32(argv[3], bounded_count(batch, container.tensors()[index].cols));
            const auto start = Clock::now();
            const auto result = amak::linear(container, index, input, {batch, prefetch, options.backend()});
            const auto ms = elapsed_ms(start);
            write_f32(argv[4], result.output);
            std::cout << "Wrote spatial output [" << batch << ',' << container.tensors()[index].rows
                      << "] to " << argv[4] << " in " << ms << " ms (forward only)\n";
            print_memory(result);
        } else if (command == "demo") {
            require(argc >= 3, "usage: amak demo MODEL.amak [--rows M --cols N ...]");
            const Options options(argc, argv, 3, {"--rows", "--cols", "--tile-rows", "--tile-cols", "--batch", "--iterations", "--dct-backend"});
            demo(argv[2], options);
        } else {
            throw std::runtime_error("amak: unknown command: " + command + " (use --help)");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
