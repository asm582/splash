#include "metal/kernels/common/q8_store_row.h"

template <uint KVHeads>
inline void splash_q8_store_verify_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device char *q8_keys, device float *q8_key_scales,
    device char *q8_values, device float *q8_value_scales,
    device const uint *page_table0, device const uint *page_table1,
    device const uint *page_table2, device const uint *page_table3,
    constant SplashChunkedPrefillParams *params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  constexpr uint Rows = SPLASH_TARGET_VERIFY_ROWS;
  constexpr uint GroupsPerLane = 2 * Rows * KVHeads;
  uint batch = group / GroupsPerLane;
  uint local_group = group % GroupsPerLane;
  constant SplashChunkedPrefillParams &lane_params = params[batch];
  if (!splash_chunk_contract_valid(lane_params) ||
      lane_params.chunk_tokens != Rows ||
      thread_index >= SplashQ8HeadDimension)
    return;
  device const uint *page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  ulong lane_tensor_stride =
      ulong(KVHeads) * lane_params.chunk_stride * SplashQ8HeadDimension;
  chunk_keys += batch * lane_tensor_stride;
  chunk_values += batch * lane_tensor_stride;

  uint rows = Rows * KVHeads;
  bool value_tensor = local_group >= rows;
  uint row = value_tensor ? local_group - rows : local_group;
  uint chunk_token = row % Rows;
  uint head = row / Rows;
  splash_q8_store_row<KVHeads>(chunk_keys, chunk_values, q8_keys,
                                 q8_key_scales, q8_values, q8_value_scales,
                                 page_table, lane_params, maxima, value_tensor,
                                 head, chunk_token, thread_index, simd_lane,
                                 simd_group);
}

kernel void verify_attention_q8_store(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device char *q8_keys [[buffer(2)]],
    device float *q8_key_scales [[buffer(3)]],
    device char *q8_values [[buffer(4)]],
    device float *q8_value_scales [[buffer(5)]],
    device const uint *page_table0 [[buffer(6)]],
    device const uint *page_table1 [[buffer(7)]],
    device const uint *page_table2 [[buffer(8)]],
    device const uint *page_table3 [[buffer(9)]],
    constant SplashChunkedPrefillParams *params [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_q8_store_verify_phase<4>(
      chunk_keys, chunk_values, q8_keys, q8_key_scales, q8_values,
      q8_value_scales, page_table0, page_table1, page_table2, page_table3,
      params, maxima, group, thread_index, simd_lane, simd_group);
}

kernel void verify_attention_q8_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device char *q8_keys [[buffer(2)]],
    device float *q8_key_scales [[buffer(3)]],
    device char *q8_values [[buffer(4)]],
    device float *q8_value_scales [[buffer(5)]],
    device const uint *page_table0 [[buffer(6)]],
    device const uint *page_table1 [[buffer(7)]],
    device const uint *page_table2 [[buffer(8)]],
    device const uint *page_table3 [[buffer(9)]],
    constant SplashChunkedPrefillParams *params [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_q8_store_verify_phase<2>(
      chunk_keys, chunk_values, q8_keys, q8_key_scales, q8_values,
      q8_value_scales, page_table0, page_table1, page_table2, page_table3,
      params, maxima, group, thread_index, simd_lane, simd_group);
}
