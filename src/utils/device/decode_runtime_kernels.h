#pragma once

#include <edge-fm/core.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace edge_fm {

void launch_increment_uint32_scalar(uint32_t* value, cudaStream_t stream);
void launch_increment_int32_triplet(int32_t* values, cudaStream_t stream);
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

} // namespace edge_fm
