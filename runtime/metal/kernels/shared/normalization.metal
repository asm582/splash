#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"

kernel void norm_rms(device const bfloat *input [[buffer(0)]],
                        device const bfloat *weight [[buffer(1)]],
                        device bfloat *output [[buffer(2)]],
                        constant uint &width [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  float inverse_rms = rms_inverse(input + row * width, width, reductions,
                                  thread_index, lane, simd_group);
  for (uint column = thread_index; column < width; column += 256) {
    output[row * width + column] =
        bfloat(float(input[row * width + column]) * inverse_rms *
               float(weight[column]));
  }
}
