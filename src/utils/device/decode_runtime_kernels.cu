#include "utils/device/decode_runtime_kernels.h"

#include "utils/check.h"

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace edge_fm {

namespace {

__global__ void increment_uint32_scalar_kernel(uint32_t* value) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        value[0] += 1;
    }
}

__global__ void increment_int32_triplet_kernel(int32_t* values) {
    int idx = threadIdx.x;
    if (idx < 3) {
        values[idx] += 1;
    }
}

template <typename T>
__global__ void copy_decode_cache_slot_kernel(const T* src,
                                              T* cache_base,
                                              int elems_per_token,
                                              const uint32_t* d_kv_len)
{
    const uint32_t kv_len = d_kv_len[0];
    if (kv_len == 0) {
        return;
    }
    const uint32_t slot = kv_len - 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < elems_per_token) {
        cache_base[static_cast<size_t>(slot) * static_cast<size_t>(elems_per_token) + static_cast<size_t>(idx)] = src[idx];
    }
}

template <typename T>
__global__ void copy_decode_kv_cache_slots_kernel(const T* k_src,
                                                  const T* v_src,
                                                  T* k_cache_base,
                                                  T* v_cache_base,
                                                  int k_elems_per_token,
                                                  int v_elems_per_token,
                                                  const uint32_t* d_kv_len)
{
    const uint32_t kv_len = d_kv_len[0];
    if (kv_len == 0) {
        return;
    }
    const uint32_t slot = kv_len - 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k_elems_per_token) {
        k_cache_base[static_cast<size_t>(slot) * static_cast<size_t>(k_elems_per_token) +
                     static_cast<size_t>(idx)] = k_src[idx];
    }
    if (idx < v_elems_per_token) {
        v_cache_base[static_cast<size_t>(slot) * static_cast<size_t>(v_elems_per_token) +
                     static_cast<size_t>(idx)] = v_src[idx];
    }
}

} // namespace

void launch_increment_uint32_scalar(uint32_t* value, cudaStream_t stream) {
    increment_uint32_scalar_kernel<<<1, 1, 0, stream>>>(value);
}

void launch_increment_int32_triplet(int32_t* values, cudaStream_t stream) {
    increment_int32_triplet_kernel<<<1, 3, 0, stream>>>(values);
}

void launch_copy_decode_cache_slot(const void* src,
                                   void* cache_base,
                                   int elems_per_token,
                                   DType dtype,
                                   const uint32_t* d_kv_len,
                                   cudaStream_t stream)
{
    constexpr int kBlock = 256;
    const int grid = (elems_per_token + kBlock - 1) / kBlock;
    if (dtype == DType::BFloat16) {
        copy_decode_cache_slot_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(src),
            static_cast<__nv_bfloat16*>(cache_base),
            elems_per_token,
            d_kv_len);
        return;
    }
    if (dtype == DType::Float16) {
        copy_decode_cache_slot_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const half*>(src),
            static_cast<half*>(cache_base),
            elems_per_token,
            d_kv_len);
        return;
    }

    throw ConfigurationError("launch_copy_decode_cache_slot only supports Float16 / BFloat16");
}

void launch_copy_decode_kv_cache_slots(const void* k_src,
                                       const void* v_src,
                                       void* k_cache_base,
                                       void* v_cache_base,
                                       int k_elems_per_token,
                                       int v_elems_per_token,
                                       DType dtype,
                                       const uint32_t* d_kv_len,
                                       cudaStream_t stream)
{
    constexpr int kBlock = 256;
    const int max_elems = std::max(k_elems_per_token, v_elems_per_token);
    const int grid = (max_elems + kBlock - 1) / kBlock;
    if (dtype == DType::BFloat16) {
        copy_decode_kv_cache_slots_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k_src),
            static_cast<const __nv_bfloat16*>(v_src),
            static_cast<__nv_bfloat16*>(k_cache_base),
            static_cast<__nv_bfloat16*>(v_cache_base),
            k_elems_per_token,
            v_elems_per_token,
            d_kv_len);
        return;
    }
    if (dtype == DType::Float16) {
        copy_decode_kv_cache_slots_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const half*>(k_src),
            static_cast<const half*>(v_src),
            static_cast<half*>(k_cache_base),
            static_cast<half*>(v_cache_base),
            k_elems_per_token,
            v_elems_per_token,
            d_kv_len);
        return;
    }

    throw ConfigurationError("launch_copy_decode_kv_cache_slots only supports Float16 / BFloat16");
}

} // namespace edge_fm
