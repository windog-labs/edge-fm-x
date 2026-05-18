#pragma once

#include <edge-fm/core.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace edge_fm {

void launch_increment_uint32_scalar(uint32_t* value, cudaStream_t stream);
void launch_increment_int32_triplet(int32_t* values, cudaStream_t stream);
void launch_finalize_decode_token(const int32_t* sampled_token,
                                  int32_t* response_token,
                                  uint32_t* d_kv_len,
                                  cudaStream_t stream);
void launch_copy_decode_cache_slot(const void* src,
                                   void* cache_base,
                                   int elems_per_token,
                                   DType dtype,
                                   const uint32_t* d_kv_len,
                                   cudaStream_t stream);
void launch_copy_decode_kv_cache_slots(const void* k_src,
                                       const void* v_src,
                                       void* k_cache_base,
                                       void* v_cache_base,
                                       int k_elems_per_token,
                                       int v_elems_per_token,
                                       DType dtype,
                                       const uint32_t* d_kv_len,
                                       cudaStream_t stream);
void launch_decode_mrope_apply_q_write_kv(const void* q_src,
                                          const void* k_src,
                                          const void* v_src,
                                          void* q_dst,
                                          void* k_cache_base,
                                          void* v_cache_base,
                                          int num_qo_heads,
                                          int num_kv_heads,
                                          int head_dim,
                                          const int32_t* position_ids,
                                          const int32_t* mrope_section_cumsum,
                                          float rope_theta,
                                          float rope_scale,
                                          DType dtype,
                                          const uint32_t* d_kv_len,
                                          cudaStream_t stream);
void launch_prefill_copy_k_and_prerotate(const void* k_src,
                                         void* k_cache_dst,
                                         void* k_rot_dst,
                                         int seq_len,
                                         int num_kv_heads,
                                         int head_dim,
                                         int qkv_stride_n,
                                         float rope_theta,
                                         float rope_scale,
                                         DType dtype,
                                         cudaStream_t stream);

} // namespace edge_fm
