#include "Normalization.hpp"

#include <utility>

namespace splash::ops {

void Normalization::addRms(metal::CommandGraph &graph,
                           metal::MetalBuffer input,
                           metal::MetalBuffer weight,
                           metal::MetalBuffer output, uint32_t width,
                           uint32_t rows) {
  graph.add("norm_rms",
            {std::move(input), std::move(weight), std::move(output)}, width,
            {rows, 1, 1});
}

void Normalization::addRmsWithQ4Sums(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    metal::MetalBuffer weight, metal::MetalBuffer output,
    metal::MetalBuffer sums, uint32_t width, uint32_t rows) {
  graph.add("prefill_norm_rms_sums32",
            {std::move(input), std::move(weight), std::move(output),
             std::move(sums)},
            width, {rows, 1, 1});
}

} // namespace splash::ops
