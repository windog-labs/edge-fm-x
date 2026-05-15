#include "engine/tasks/token_generation/cuda/kernels/decode_runtime_kernels.h"

#include "utils/check.h"

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace edge_fm {

namespace {

template <typename T>
__device__ __forceinline__ float dtype_to_float(T value);

template <>
__device__ __forceinline__ float dtype_to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float dtype_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T float_to_dtype(float value);

template <>
__device__ __forceinline__ half float_to_dtype<half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_dtype<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

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

__global__ void finalize_decode_token_kernel(const int32_t* sampled_token,
                                             int32_t* response_token,
                                             uint32_t* d_kv_len)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        if (response_token != nullptr && sampled_token != nullptr) {
            response_token[0] = sampled_token[0];
        }
        if (d_kv_len != nullptr) {
            d_kv_len[0] += 1;
        }
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

__global__ void copy_decode_cache_slot_vec16_kernel(const uint4* src,
                                                    uint4* cache_base,
                                                    int vecs_per_token,
                                                    const uint32_t* d_kv_len)
{
    const uint32_t kv_len = d_kv_len[0];
    if (kv_len == 0) {
        return;
    }
    const uint32_t slot = kv_len - 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vecs_per_token) {
        cache_base[static_cast<size_t>(slot) * static_cast<size_t>(vecs_per_token) +
                   static_cast<size_t>(idx)] = src[idx];
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

__global__ void copy_decode_kv_cache_slots_vec16_kernel(const uint4* k_src,
                                                        const uint4* v_src,
                                                        uint4* k_cache_base,
                                                        uint4* v_cache_base,
                                                        int k_vecs_per_token,
                                                        int v_vecs_per_token,
                                                        const uint32_t* d_kv_len)
{
    const uint32_t kv_len = d_kv_len[0];
    if (kv_len == 0) {
        return;
    }
    const uint32_t slot = kv_len - 1;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < k_vecs_per_token) {
        k_cache_base[static_cast<size_t>(slot) * static_cast<size_t>(k_vecs_per_token) +
                     static_cast<size_t>(idx)] = k_src[idx];
    }
    if (idx < v_vecs_per_token) {
        v_cache_base[static_cast<size_t>(slot) * static_cast<size_t>(v_vecs_per_token) +
                     static_cast<size_t>(idx)] = v_src[idx];
    }
}

template <typename T>
__global__ void decode_mrope_apply_q_write_kv_kernel(
    const T* q_src,
    const T* k_src,
    const T* v_src,
    T* q_dst,
    T* k_cache_base,
    T* v_cache_base,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    const int32_t* position_ids,
    const int32_t* mrope_section_cumsum,
    float rope_rcp_scale,
    float rope_theta,
    const uint32_t* d_kv_len)
{
    const uint32_t kv_len = d_kv_len[0];
    if (kv_len == 0) {
        return;
    }

    const int half_dim = head_dim / 2;
    const int d = static_cast<int>(threadIdx.x);
    if (d >= half_dim) {
        return;
    }

    const int head_idx = static_cast<int>(blockIdx.y);
    const int cum0 = mrope_section_cumsum[0];
    const int cum1 = mrope_section_cumsum[1];
    const int d_lo = d;
    const int d_hi = d + half_dim;
    const int section_lo = (d_lo < cum0) ? 0 : ((d_lo < cum1) ? 1 : 2);
    const int section_hi = section_lo;
    const int pos_lo = position_ids[section_lo];
    const int pos_hi = position_ids[section_hi];
    const float inv_freq_d = 1.0f / powf(rope_theta, static_cast<float>(2 * d) / static_cast<float>(head_dim));
    const float angle_lo = static_cast<float>(pos_lo) * inv_freq_d * rope_rcp_scale;
    const float angle_hi = static_cast<float>(pos_hi) * inv_freq_d * rope_rcp_scale;

    float sin_lo, cos_lo, sin_hi, cos_hi;
    __sincosf(angle_lo, &sin_lo, &cos_lo);
    __sincosf(angle_hi, &sin_hi, &cos_hi);

    if (head_idx < num_qo_heads) {
        const size_t base = static_cast<size_t>(head_idx) * static_cast<size_t>(head_dim);
        const float val_lo = dtype_to_float(q_src[base + static_cast<size_t>(d_lo)]);
        const float val_hi = dtype_to_float(q_src[base + static_cast<size_t>(d_hi)]);
        q_dst[base + static_cast<size_t>(d_lo)] = float_to_dtype<T>(val_lo * cos_lo - val_hi * sin_lo);
        q_dst[base + static_cast<size_t>(d_hi)] = float_to_dtype<T>(val_hi * cos_hi + val_lo * sin_hi);
        return;
    }

    const int kv_head = head_idx - num_qo_heads;
    if (kv_head >= num_kv_heads) {
        return;
    }

    const size_t kv_head_base = static_cast<size_t>(kv_head) * static_cast<size_t>(head_dim);
    const float k_val_lo = dtype_to_float(k_src[kv_head_base + static_cast<size_t>(d_lo)]);
    const float k_val_hi = dtype_to_float(k_src[kv_head_base + static_cast<size_t>(d_hi)]);
    const T k_rot_lo = float_to_dtype<T>(k_val_lo * cos_lo - k_val_hi * sin_lo);
    const T k_rot_hi = float_to_dtype<T>(k_val_hi * cos_hi + k_val_lo * sin_hi);

    const size_t kv_elems_per_token = static_cast<size_t>(num_kv_heads) * static_cast<size_t>(head_dim);
    const size_t slot_base = static_cast<size_t>(kv_len - 1U) * kv_elems_per_token;
    const size_t cache_base = slot_base + kv_head_base;

    k_cache_base[cache_base + static_cast<size_t>(d_lo)] = k_rot_lo;
    k_cache_base[cache_base + static_cast<size_t>(d_hi)] = k_rot_hi;
    v_cache_base[cache_base + static_cast<size_t>(d_lo)] = v_src[kv_head_base + static_cast<size_t>(d_lo)];
    v_cache_base[cache_base + static_cast<size_t>(d_hi)] = v_src[kv_head_base + static_cast<size_t>(d_hi)];
}

inline bool can_vectorize_vec16(const void* ptr, int bytes) {
    constexpr int kVecBytes = static_cast<int>(sizeof(uint4));
    return ptr != nullptr &&
        bytes > 0 &&
        (bytes % kVecBytes) == 0 &&
        (reinterpret_cast<uintptr_t>(ptr) % alignof(uint4)) == 0;
}

} // namespace

void launch_increment_uint32_scalar(uint32_t* value, cudaStream_t stream) {
    increment_uint32_scalar_kernel<<<1, 1, 0, stream>>>(value);
}

void launch_increment_int32_triplet(int32_t* values, cudaStream_t stream) {
    increment_int32_triplet_kernel<<<1, 3, 0, stream>>>(values);
}

void launch_finalize_decode_token(const int32_t* sampled_token,
                                  int32_t* response_token,
                                  uint32_t* d_kv_len,
                                  cudaStream_t stream)
{
    finalize_decode_token_kernel<<<1, 1, 0, stream>>>(sampled_token, response_token, d_kv_len);
}

void launch_copy_decode_cache_slot(const void* src,
                                   void* cache_base,
                                   int elems_per_token,
                                   DType dtype,
                                   const uint32_t* d_kv_len,
                                   cudaStream_t stream)
{
    constexpr int kVecBytes = static_cast<int>(sizeof(uint4));
    constexpr int kBlock = 256;
    const int bytes = elems_per_token * static_cast<int>(get_dtype_size(dtype));
    if (can_vectorize_vec16(src, bytes) && can_vectorize_vec16(cache_base, bytes)) {
        const int vecs = bytes / kVecBytes;
        const int grid = (vecs + kBlock - 1) / kBlock;
        copy_decode_cache_slot_vec16_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const uint4*>(src),
            static_cast<uint4*>(cache_base),
            vecs,
            d_kv_len);
        return;
    }

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
    constexpr int kVecBytes = static_cast<int>(sizeof(uint4));
    constexpr int kBlock = 256;
    const int k_bytes = k_elems_per_token * static_cast<int>(get_dtype_size(dtype));
    const int v_bytes = v_elems_per_token * static_cast<int>(get_dtype_size(dtype));
    if (can_vectorize_vec16(k_src, k_bytes) &&
        can_vectorize_vec16(v_src, v_bytes) &&
        can_vectorize_vec16(k_cache_base, k_bytes) &&
        can_vectorize_vec16(v_cache_base, v_bytes))
    {
        const int k_vecs = k_bytes / kVecBytes;
        const int v_vecs = v_bytes / kVecBytes;
        const int grid = (std::max(k_vecs, v_vecs) + kBlock - 1) / kBlock;
        copy_decode_kv_cache_slots_vec16_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const uint4*>(k_src),
            static_cast<const uint4*>(v_src),
            static_cast<uint4*>(k_cache_base),
            static_cast<uint4*>(v_cache_base),
            k_vecs,
            v_vecs,
            d_kv_len);
        return;
    }

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
                                          cudaStream_t stream)
{
    check<ConfigurationError>(head_dim > 0 && (head_dim % 2) == 0,
                              "launch_decode_mrope_apply_q_write_kv expects even head_dim");
    check<ConfigurationError>(num_qo_heads > 0 && num_kv_heads > 0,
                              "launch_decode_mrope_apply_q_write_kv expects positive head counts");
    check<ConfigurationError>(position_ids != nullptr && mrope_section_cumsum != nullptr,
                              "launch_decode_mrope_apply_q_write_kv expects valid M-RoPE metadata");

    const dim3 grid(1U, static_cast<unsigned>(num_qo_heads + num_kv_heads));
    const dim3 block(static_cast<unsigned>(head_dim / 2));
    const float rope_rcp_scale = 1.0f / rope_scale;

    if (dtype == DType::BFloat16) {
        decode_mrope_apply_q_write_kv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q_src),
            static_cast<const __nv_bfloat16*>(k_src),
            static_cast<const __nv_bfloat16*>(v_src),
            static_cast<__nv_bfloat16*>(q_dst),
            static_cast<__nv_bfloat16*>(k_cache_base),
            static_cast<__nv_bfloat16*>(v_cache_base),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            position_ids,
            mrope_section_cumsum,
            rope_rcp_scale,
            rope_theta,
            d_kv_len);
        return;
    }
    if (dtype == DType::Float16) {
        decode_mrope_apply_q_write_kv_kernel<<<grid, block, 0, stream>>>(
            static_cast<const half*>(q_src),
            static_cast<const half*>(k_src),
            static_cast<const half*>(v_src),
            static_cast<half*>(q_dst),
            static_cast<half*>(k_cache_base),
            static_cast<half*>(v_cache_base),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            position_ids,
            mrope_section_cumsum,
            rope_rcp_scale,
            rope_theta,
            d_kv_len);
        return;
    }

    throw ConfigurationError("launch_decode_mrope_apply_q_write_kv only supports Float16 / BFloat16");
}

} // namespace edge_fm
