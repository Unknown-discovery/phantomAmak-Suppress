#include "amak/amak.hpp"
#include "amak/dct.hpp"
#include "binary.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace amak {

RunResult linear(const Container& container, std::size_t tensor_index,
                 const std::vector<float>& input, const RunOptions& options,
                 const std::vector<float>& bias) {
    using namespace detail;
    require(tensor_index < container.tensors().size(), "tensor index out of range");
    require(options.batch > 0, "batch must be positive");
    require(options.prefetch_slots <= kMaxPrefetchSlots, "prefetch must be in 0..8");
    const auto backend = resolve_dct_backend(options.dct_backend);
    const auto& t = container.tensors()[tensor_index];
    const auto input_count = multiply(options.batch, t.cols);
    const auto output_count = multiply(options.batch, t.rows);
    const auto limit = container.limits().max_activation_elements;
    require(input_count <= limit && output_count <= limit, "activation element limit exceeded");
    require(input.size() == input_count, "input shape must be [batch, input_features]");
    require(bias.empty() || bias.size() == t.rows, "bias must have output_features elements");
    for (const auto value : input) require(std::isfinite(value), "input activations must be finite");
    for (const auto value : bias) require(std::isfinite(value), "bias values must be finite");

    const auto rows = as_size(t.rows), cols = as_size(t.cols);
    const auto max_rows = as_size(std::min<std::uint64_t>(t.rows, t.tile_rows));
    const auto max_cols = as_size(std::min<std::uint64_t>(t.cols, t.tile_cols));
    std::map<std::size_t, DctPlan> plans;
    auto add_plan = [&](std::size_t n) { if (n != 0) plans.try_emplace(n, n, backend); };
    add_plan(max_rows);
    add_plan(max_cols);
    add_plan(as_size(t.rows % t.tile_rows));
    add_plan(as_size(t.cols % t.tile_cols));

    RunResult result;
    result.dct_backend = backend;
    result.output.resize(as_size(output_count));
    std::vector<double> input_frequency(as_size(multiply(options.batch, max_cols)));
    std::vector<double> output_frequency(as_size(multiply(options.batch, max_rows)));
    std::vector<double> spatial_tile(max_rows); // reused across batch rows; never a weight tile
    result.activation_scratch_bytes = (input_frequency.capacity() + output_frequency.capacity() +
                                       spatial_tile.capacity()) * sizeof(double);
    for (const auto& entry : plans) result.transform_plan_bytes += entry.second.storage_bytes();

    result.stream = container.visit_tiles(tensor_index, [&](const DecodedTile& tile) {
        const auto& dc = plans.at(tile.cols);
        const auto& dr = plans.at(tile.rows);
        if (tile.col_start == 0) std::fill(output_frequency.begin(), output_frequency.end(), 0.0);
        const auto row_start = as_size(tile.row_start), col_start = as_size(tile.col_start);
        for (std::size_t b = 0; b < options.batch; ++b) {
            auto* x_hat = input_frequency.data() + b * max_cols;
            auto* y_hat = output_frequency.data() + b * max_rows;
            dc.forward(input.data() + b * cols + col_start, x_hat);
            for (std::size_t u = 0; u < tile.rows; ++u) {
                double dot = 0;
                for (std::size_t v = 0; v < tile.cols; ++v)
                    dot += static_cast<double>(tile.coefficients[u * tile.cols + v]) * x_hat[v];
                // Factor the scalar outside the dot product. No delta*Q array.
                // This coefficient multiply remains the scalar reference kernel.
                y_hat[u] += static_cast<double>(tile.delta) * dot;
            }
            if (tile.col_start + tile.cols == t.cols) {
                dr.inverse(y_hat, spatial_tile.data());
                for (std::size_t r = 0; r < tile.rows; ++r) {
                    const double spatial = spatial_tile[r] +
                        (bias.empty() ? 0.0 : static_cast<double>(bias[row_start + r]));
                    require(std::isfinite(spatial) && std::abs(spatial) <= std::numeric_limits<float>::max(),
                            "linear output overflows float32");
                    result.output[b * rows + row_start + r] = static_cast<float>(spatial);
                }
            }
        }
    }, options.prefetch_slots);
    return result;
}

} // namespace amak
