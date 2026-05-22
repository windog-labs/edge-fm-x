#include "operators/qwen3_5/qwen3_5_ops.h"

#include "utils/check.h"
#include "utils/device/cuda_utils.h"
#include "utils/device/memory.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <string>

namespace edge_fm {
namespace {

constexpr int32_t kGatedDeltaSequenceValueTile = 32;
constexpr int32_t kGatedDeltaPrecomputedValueTile = 16;
constexpr int32_t kGatedDeltaSequenceThreads = 256;
constexpr int32_t kGatedDeltaPrecomputeThreads = 128;

template <typename T>
__device__ float load_as_float(const T* ptr, int64_t idx) {
    return static_cast<float>(ptr[idx]);
}

template <typename T>
__device__ void store_from_float(T* ptr, int64_t idx, float value) {
    ptr[idx] = static_cast<T>(value);
}

template <>
__device__ void store_from_float<half>(half* ptr, int64_t idx, float value) {
    ptr[idx] = __float2half(value);
}

template <>
__device__ void store_from_float<__nv_bfloat16>(__nv_bfloat16* ptr, int64_t idx, float value) {
    ptr[idx] = __float2bfloat16(value);
}

__device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    return log1pf(expf(x));
}

template <typename T>
__device__ float round_to_dtype(float x) {
    return x;
}

template <>
__device__ float round_to_dtype<half>(float x) {
    return __half2float(__float2half(x));
}

template <>
__device__ float round_to_dtype<__nv_bfloat16>(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

template <typename T>
__device__ float l2norm_sum_for_transformers_dtype(const T* x, int32_t dim) {
    float sum = 0.0f;
    for (int32_t i = 0; i < dim; ++i) {
        const float v = load_as_float(x, i);
        sum += round_to_dtype<T>(v * v);
    }
    return round_to_dtype<T>(sum);
}

template <typename T>
__device__ float l2norm_inv_for_transformers_dtype(float sum, float eps) {
    return round_to_dtype<T>(rsqrtf(round_to_dtype<T>(sum + eps)));
}

template <typename T>
__device__ float l2norm_value_for_transformers_dtype(const T* x, int32_t idx, float inv_norm) {
    return round_to_dtype<T>(load_as_float(x, idx) * inv_norm);
}

template <typename T, typename WeightT, bool kGated>
__global__ void qwen3_5_rmsnorm_kernel(const T* __restrict__ input,
                                       const T* __restrict__ gate,
                                       const WeightT* __restrict__ weight,
                                       T* __restrict__ output,
                                       int64_t rows,
                                       int64_t hidden,
                                       float eps) {
    extern __shared__ float smem[];
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float sum_sq = 0.0f;
    for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float x = load_as_float(input, row * hidden + col);
        sum_sq += x * x;
    }

    smem[threadIdx.x] = sum_sq;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);
    for (int64_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const int64_t idx = row * hidden + col;
        float y = load_as_float(input, idx) * inv_rms;
        if constexpr (kGated) {
            y *= load_as_float(weight, col);
            y *= silu(load_as_float(gate, idx));
        } else {
            y *= 1.0f + load_as_float(weight, col);
        }
        store_from_float(output, idx, y);
    }
}

template <typename T, typename WeightT, bool kGated, int32_t kHidden, int32_t kThreads>
__global__ void qwen3_5_rmsnorm_fixed_hidden_kernel(const T* __restrict__ input,
                                                    const T* __restrict__ gate,
                                                    const WeightT* __restrict__ weight,
                                                    T* __restrict__ output,
                                                    int64_t rows,
                                                    float eps) {
    extern __shared__ float smem[];
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    float sum_sq = 0.0f;
    const int64_t row_base = row * static_cast<int64_t>(kHidden);
    for (int32_t col = static_cast<int32_t>(threadIdx.x); col < kHidden; col += kThreads) {
        const float x = load_as_float(input, row_base + col);
        sum_sq += x * x;
    }

    smem[threadIdx.x] = sum_sq;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(smem[0] / static_cast<float>(kHidden) + eps);
    for (int32_t col = static_cast<int32_t>(threadIdx.x); col < kHidden; col += kThreads) {
        const int64_t idx = row_base + col;
        float y = load_as_float(input, idx) * inv_rms;
        if constexpr (kGated) {
            y *= load_as_float(weight, col);
            y *= silu(load_as_float(gate, idx));
        } else {
            y *= 1.0f + load_as_float(weight, col);
        }
        store_from_float(output, idx, y);
    }
}

template <typename T, typename WeightT, bool kGated>
__global__ void qwen3_5_rmsnorm_one_warp_kernel(const T* __restrict__ input,
                                                const T* __restrict__ gate,
                                                const WeightT* __restrict__ weight,
                                                T* __restrict__ output,
                                                int64_t rows,
                                                int64_t hidden,
                                                float eps) {
    const int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    float sum_sq = 0.0f;
    for (int64_t col = lane; col < hidden; col += 32) {
        const float x = load_as_float(input, row * hidden + col);
        sum_sq += x * x;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
    }

    const float total = __shfl_sync(0xffffffffu, sum_sq, 0);
    const float inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);
    for (int64_t col = lane; col < hidden; col += 32) {
        const int64_t idx = row * hidden + col;
        float y = load_as_float(input, idx) * inv_rms;
        if constexpr (kGated) {
            y *= load_as_float(weight, col);
            y *= silu(load_as_float(gate, idx));
        } else {
            y *= 1.0f + load_as_float(weight, col);
        }
        store_from_float(output, idx, y);
    }
}

template <typename T>
__global__ void add_kernel(const T* __restrict__ lhs,
                           const T* __restrict__ rhs,
                           T* __restrict__ output,
                           int64_t elements) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    store_from_float(output, idx, load_as_float(lhs, idx) + load_as_float(rhs, idx));
}

template <typename T>
__global__ void mul_sigmoid_kernel(const T* __restrict__ input,
                                   const T* __restrict__ gate,
                                   T* __restrict__ output,
                                   int64_t elements) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }
    const float y = load_as_float(input, idx) * sigmoid(load_as_float(gate, idx));
    store_from_float(output, idx, y);
}

template <typename T>
__global__ void split_q_gate_kernel(const T* __restrict__ q_proj,
                                    T* __restrict__ query,
                                    T* __restrict__ gate,
                                    int32_t seq_len,
                                    int32_t num_heads,
                                    int32_t head_dim) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(seq_len) * num_heads * head_dim;
    if (idx >= elements) {
        return;
    }
    const int32_t d = static_cast<int32_t>(idx % head_dim);
    const int64_t head_linear = idx / head_dim;
    const int32_t head = static_cast<int32_t>(head_linear % num_heads);
    const int32_t token = static_cast<int32_t>(head_linear / num_heads);
    const int64_t proj_base = (static_cast<int64_t>(token) * num_heads + head) * (2LL * head_dim);
    query[idx] = q_proj[proj_base + d];
    gate[static_cast<int64_t>(token) * num_heads * head_dim + static_cast<int64_t>(head) * head_dim + d] =
        q_proj[proj_base + head_dim + d];
}

template <typename T>
__global__ void depthwise_causal_conv1d_kernel(const T* __restrict__ input,
                                               const T* __restrict__ weight,
                                               const T* __restrict__ conv_state,
                                               T* __restrict__ output,
                                               int32_t seq_len,
                                               int32_t channels,
                                               int32_t kernel_size) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(seq_len) * channels;
    if (idx >= elements) {
        return;
    }
    const int32_t channel = static_cast<int32_t>(idx % channels);
    const int32_t token = static_cast<int32_t>(idx / channels);

    float acc = 0.0f;
    for (int32_t j = 0; j < kernel_size; ++j) {
        const int32_t source_token = token + j - (kernel_size - 1);
        float x = 0.0f;
        if (source_token >= 0) {
            x = load_as_float(input, static_cast<int64_t>(source_token) * channels + channel);
        } else {
            const int32_t state_idx = token + 1 + j;
            if (state_idx >= 0 && state_idx < kernel_size) {
                x = load_as_float(conv_state, static_cast<int64_t>(channel) * kernel_size + state_idx);
            }
        }
        acc += load_as_float(weight, (static_cast<int64_t>(channel) * kernel_size) + j) * x;
    }
    store_from_float(output, idx, silu(acc));
}

template <typename T>
__global__ void depthwise_causal_conv1d_kernel4(const T* __restrict__ input,
                                                const T* __restrict__ weight,
                                                const T* __restrict__ conv_state,
                                                T* __restrict__ output,
                                                int32_t seq_len,
                                                int32_t channels) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(seq_len) * channels;
    if (idx >= elements) {
        return;
    }
    const int32_t channel = static_cast<int32_t>(idx % channels);
    const int32_t token = static_cast<int32_t>(idx / channels);
    const int64_t weight_base = static_cast<int64_t>(channel) * 4;

    float x0 = 0.0f;
    float x1 = 0.0f;
    float x2 = 0.0f;
    float x3 = load_as_float(input, static_cast<int64_t>(token) * channels + channel);
    if (token >= 3) {
        x0 = load_as_float(input, static_cast<int64_t>(token - 3) * channels + channel);
        x1 = load_as_float(input, static_cast<int64_t>(token - 2) * channels + channel);
        x2 = load_as_float(input, static_cast<int64_t>(token - 1) * channels + channel);
    } else {
        const int64_t state_base = static_cast<int64_t>(channel) * 4;
        if (token == 0) {
            x0 = load_as_float(conv_state, state_base + 1);
            x1 = load_as_float(conv_state, state_base + 2);
            x2 = load_as_float(conv_state, state_base + 3);
        } else if (token == 1) {
            x0 = load_as_float(conv_state, state_base + 2);
            x1 = load_as_float(conv_state, state_base + 3);
            x2 = load_as_float(input, channel);
        } else {
            x0 = load_as_float(conv_state, state_base + 3);
            x1 = load_as_float(input, channel);
            x2 = load_as_float(input, static_cast<int64_t>(1) * channels + channel);
        }
    }

    float acc = 0.0f;
    acc += load_as_float(weight, weight_base + 0) * x0;
    acc += load_as_float(weight, weight_base + 1) * x1;
    acc += load_as_float(weight, weight_base + 2) * x2;
    acc += load_as_float(weight, weight_base + 3) * x3;
    store_from_float(output, idx, silu(acc));
}

template <typename T>
__global__ void depthwise_causal_conv1d_kernel4_single_token_update(const T* __restrict__ input,
                                                                    const T* __restrict__ weight,
                                                                    T* __restrict__ conv_state,
                                                                    T* __restrict__ output,
                                                                    int32_t channels) {
    const int32_t channel = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) {
        return;
    }
    T* state = conv_state + static_cast<int64_t>(channel) * 4;
    const int64_t weight_base = static_cast<int64_t>(channel) * 4;

    const T s1 = state[1];
    const T s2 = state[2];
    const T s3 = state[3];
    const T x = input[channel];
    float acc = 0.0f;
    acc += load_as_float(weight, weight_base + 0) * static_cast<float>(s1);
    acc += load_as_float(weight, weight_base + 1) * static_cast<float>(s2);
    acc += load_as_float(weight, weight_base + 2) * static_cast<float>(s3);
    acc += load_as_float(weight, weight_base + 3) * static_cast<float>(x);
    store_from_float(output, channel, silu(acc));

    state[0] = s1;
    state[1] = s2;
    state[2] = s3;
    state[3] = x;
}

template <typename T>
__global__ void update_conv_state_kernel(const T* __restrict__ input,
                                         const T* __restrict__ conv_state,
                                         T* __restrict__ next_conv_state,
                                         int32_t seq_len,
                                         int32_t channels,
                                         int32_t kernel_size) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = static_cast<int64_t>(channels) * kernel_size;
    if (idx >= elements) {
        return;
    }
    const int32_t state_idx = static_cast<int32_t>(idx % kernel_size);
    const int32_t channel = static_cast<int32_t>(idx / kernel_size);
    const int32_t concat_pos = seq_len + state_idx;
    if (concat_pos < kernel_size) {
        next_conv_state[idx] = conv_state[static_cast<int64_t>(channel) * kernel_size + concat_pos];
    } else {
        const int32_t input_token = concat_pos - kernel_size;
        next_conv_state[idx] = input[static_cast<int64_t>(input_token) * channels + channel];
    }
}

template <typename T, typename BiasT>
__global__ void compute_g_beta_kernel(const T* __restrict__ a,
                                      const T* __restrict__ b,
                                      const float* __restrict__ a_log,
                                      const BiasT* __restrict__ dt_bias,
                                      float* __restrict__ g,
                                      float* __restrict__ beta,
                                      int64_t rows,
                                      int64_t heads) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t elements = rows * heads;
    if (idx >= elements) {
        return;
    }
    const int64_t head = idx % heads;
    const float a_val = load_as_float(a, idx);
    const float b_val = load_as_float(b, idx);
    const float dt = load_as_float(dt_bias, head);
    g[idx] = -expf(a_log[head]) * softplus(a_val + dt);
    beta[idx] = sigmoid(b_val);
}

template <typename T, typename BiasT>
__global__ void compute_decay_beta_kernel(const T* __restrict__ a,
                                          const T* __restrict__ b,
                                          const float* __restrict__ a_log,
                                          const BiasT* __restrict__ dt_bias,
                                          float* __restrict__ decay,
                                          float* __restrict__ beta,
                                          int64_t elements,
                                          int64_t heads) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }

    const int64_t head = idx % heads;
    const float a_val = load_as_float(a, idx);
    const float b_val = load_as_float(b, idx);
    const float dt = load_as_float(dt_bias, head);
    const float g = -expf(a_log[head]) * softplus(a_val + dt);
    decay[idx] = expf(g);
    beta[idx] = sigmoid(b_val);
}

__global__ void gated_delta_recurrent_step_kernel(const float* __restrict__ query,
                                                  const float* __restrict__ key,
                                                  const float* __restrict__ value,
                                                  const float* __restrict__ g,
                                                  const float* __restrict__ beta,
                                                  float* __restrict__ recurrent_state,
                                                  float* __restrict__ output,
                                                  int32_t num_heads,
                                                  int32_t key_dim,
                                                  int32_t value_dim) {
    const int32_t head = blockIdx.x;
    if (head >= num_heads || threadIdx.x != 0) {
        return;
    }

    const float* query_h = query + static_cast<int64_t>(head) * key_dim;
    const float* key_h = key + static_cast<int64_t>(head) * key_dim;
    const float* value_h = value + static_cast<int64_t>(head) * value_dim;
    float* state_h = recurrent_state + static_cast<int64_t>(head) * key_dim * value_dim;
    float* output_h = output + static_cast<int64_t>(head) * value_dim;

    float q_norm_sq = 0.0f;
    float k_norm_sq = 0.0f;
    for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
        q_norm_sq += query_h[k_idx] * query_h[k_idx];
        k_norm_sq += key_h[k_idx] * key_h[k_idx];
    }
    const float q_scale = rsqrtf(q_norm_sq + 1e-6f) * rsqrtf(static_cast<float>(key_dim));
    const float k_scale = rsqrtf(k_norm_sq + 1e-6f);
    const float decay = expf(g[head]);
    const float beta_h = beta[head];

    for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
        for (int32_t v_idx = 0; v_idx < value_dim; ++v_idx) {
            const int64_t state_idx = static_cast<int64_t>(k_idx) * value_dim + v_idx;
            state_h[state_idx] *= decay;
        }
    }

    for (int32_t v_idx = 0; v_idx < value_dim; ++v_idx) {
        float kv_mem = 0.0f;
        for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
            kv_mem += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] * key_h[k_idx] * k_scale;
        }
        const float delta = (value_h[v_idx] - kv_mem) * beta_h;
        for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
            state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] += key_h[k_idx] * k_scale * delta;
        }
    }

    for (int32_t v_idx = 0; v_idx < value_dim; ++v_idx) {
        float acc = 0.0f;
        for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
            acc += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] * query_h[k_idx] * q_scale;
        }
        output_h[v_idx] = acc;
    }
}

template <typename T>
__global__ void gated_delta_sequence_kernel(const T* __restrict__ mixed_qkv,
                                            const float* __restrict__ g,
                                            const float* __restrict__ beta,
                                            float* __restrict__ recurrent_state,
                                            T* __restrict__ output,
                                            int32_t seq_len,
                                            int32_t num_heads,
                                            int32_t key_dim,
                                            int32_t value_dim) {
    const int32_t head = blockIdx.x;
    const int32_t value_tile_begin = blockIdx.y * kGatedDeltaSequenceValueTile;
    const int32_t value_tile_size = min(kGatedDeltaSequenceValueTile, value_dim - value_tile_begin);
    if (head >= num_heads) {
        return;
    }
    if (value_tile_size <= 0) {
        return;
    }

    const int32_t key_total = num_heads * key_dim;
    const int32_t value_total = num_heads * value_dim;
    const int32_t conv_dim = 2 * key_total + value_total;
    float* state_h = recurrent_state + static_cast<int64_t>(head) * key_dim * value_dim;
    extern __shared__ float shared[];
    float* delta_s = shared;
    float* scalar_s = shared + kGatedDeltaSequenceValueTile;
    constexpr int32_t kQInvNorm = 0;
    constexpr int32_t kKInvNorm = 1;
    constexpr int32_t kDecay = 2;
    constexpr int32_t kBeta = 3;
    const int64_t state_elements = static_cast<int64_t>(key_dim) * value_tile_size;
    const float q_scale = rsqrtf(static_cast<float>(key_dim));

    for (int32_t token = 0; token < seq_len; ++token) {
        const T* row = mixed_qkv + static_cast<int64_t>(token) * conv_dim;
        const T* query_h = row + static_cast<int64_t>(head) * key_dim;
        const T* key_h = row + static_cast<int64_t>(key_total) + static_cast<int64_t>(head) * key_dim;
        const T* value_h = row + static_cast<int64_t>(2 * key_total) + static_cast<int64_t>(head) * value_dim;
        T* output_h = output + static_cast<int64_t>(token) * value_total + static_cast<int64_t>(head) * value_dim;

        if (threadIdx.x == 0) {
            scalar_s[kQInvNorm] =
                l2norm_inv_for_transformers_dtype<T>(
                    l2norm_sum_for_transformers_dtype(query_h, key_dim),
                    1e-6f);
            scalar_s[kKInvNorm] =
                l2norm_inv_for_transformers_dtype<T>(
                    l2norm_sum_for_transformers_dtype(key_h, key_dim),
                    1e-6f);
            scalar_s[kDecay] = expf(g[static_cast<int64_t>(token) * num_heads + head]);
            scalar_s[kBeta] = beta[static_cast<int64_t>(token) * num_heads + head];
        }
        __syncthreads();
        const float q_inv_norm = scalar_s[kQInvNorm];
        const float k_inv_norm = scalar_s[kKInvNorm];
        const float decay = scalar_s[kDecay];
        const float beta_h = scalar_s[kBeta];

        for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
            const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
            const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
            state_h[static_cast<int64_t>(k_idx) * value_dim + value_tile_begin + local_v] *= decay;
        }
        __syncthreads();

        for (int32_t local_v = threadIdx.x; local_v < value_tile_size; local_v += blockDim.x) {
            const int32_t v_idx = value_tile_begin + local_v;
            float kv_mem = 0.0f;
            for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
                const float k_norm = l2norm_value_for_transformers_dtype(key_h, k_idx, k_inv_norm);
                kv_mem += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] *
                          k_norm;
            }
            delta_s[local_v] = (load_as_float(value_h, v_idx) - kv_mem) * beta_h;
        }
        __syncthreads();

        for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
            const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
            const int32_t v_idx = value_tile_begin + local_v;
            const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
            const float k_norm = l2norm_value_for_transformers_dtype(key_h, k_idx, k_inv_norm);
            state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] += k_norm * delta_s[local_v];
        }
        __syncthreads();

        for (int32_t local_v = threadIdx.x; local_v < value_tile_size; local_v += blockDim.x) {
            const int32_t v_idx = value_tile_begin + local_v;
            float acc = 0.0f;
            for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
                const float q_norm = l2norm_value_for_transformers_dtype(query_h, k_idx, q_inv_norm) * q_scale;
                acc += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] *
                       q_norm;
            }
            store_from_float(output_h, v_idx, acc);
        }
        if (token + 1 < seq_len) {
            __syncthreads();
        }
    }
}

template <typename T, typename BiasT>
__global__ void gated_delta_sequence_from_ab_kernel(const T* __restrict__ mixed_qkv,
                                                    const T* __restrict__ a,
                                                    const T* __restrict__ b,
                                                    const float* __restrict__ a_log,
                                                    const BiasT* __restrict__ dt_bias,
                                                    float* __restrict__ recurrent_state,
                                                    T* __restrict__ output,
                                                    int32_t seq_len,
                                                    int32_t num_heads,
                                                    int32_t key_dim,
                                                    int32_t value_dim) {
    const int32_t head = blockIdx.x;
    const int32_t value_tile_begin = blockIdx.y * kGatedDeltaSequenceValueTile;
    const int32_t value_tile_size = min(kGatedDeltaSequenceValueTile, value_dim - value_tile_begin);
    if (head >= num_heads) {
        return;
    }
    if (value_tile_size <= 0) {
        return;
    }

    const int32_t key_total = num_heads * key_dim;
    const int32_t value_total = num_heads * value_dim;
    const int32_t conv_dim = 2 * key_total + value_total;
    float* state_h = recurrent_state + static_cast<int64_t>(head) * key_dim * value_dim;
    extern __shared__ float shared[];
    float* delta_s = shared;
    float* scalar_s = shared + kGatedDeltaSequenceValueTile;
    constexpr int32_t kQInvNorm = 0;
    constexpr int32_t kKInvNorm = 1;
    constexpr int32_t kDecay = 2;
    constexpr int32_t kBeta = 3;
    const int64_t state_elements = static_cast<int64_t>(key_dim) * value_tile_size;
    const float q_scale = rsqrtf(static_cast<float>(key_dim));

    for (int32_t token = 0; token < seq_len; ++token) {
        const T* row = mixed_qkv + static_cast<int64_t>(token) * conv_dim;
        const T* query_h = row + static_cast<int64_t>(head) * key_dim;
        const T* key_h = row + static_cast<int64_t>(key_total) + static_cast<int64_t>(head) * key_dim;
        const T* value_h = row + static_cast<int64_t>(2 * key_total) + static_cast<int64_t>(head) * value_dim;
        T* output_h = output + static_cast<int64_t>(token) * value_total + static_cast<int64_t>(head) * value_dim;

        if (threadIdx.x == 0) {
            scalar_s[kQInvNorm] =
                l2norm_inv_for_transformers_dtype<T>(
                    l2norm_sum_for_transformers_dtype(query_h, key_dim),
                    1e-6f);
            scalar_s[kKInvNorm] =
                l2norm_inv_for_transformers_dtype<T>(
                    l2norm_sum_for_transformers_dtype(key_h, key_dim),
                    1e-6f);
            const int64_t gate_idx = static_cast<int64_t>(token) * num_heads + head;
            const float a_val = load_as_float(a, gate_idx);
            const float b_val = load_as_float(b, gate_idx);
            const float dt = load_as_float(dt_bias, head);
            const float g_value = -expf(a_log[head]) * softplus(a_val + dt);
            scalar_s[kDecay] = expf(g_value);
            scalar_s[kBeta] = sigmoid(b_val);
        }
        __syncthreads();
        const float q_inv_norm = scalar_s[kQInvNorm];
        const float k_inv_norm = scalar_s[kKInvNorm];
        const float decay = scalar_s[kDecay];
        const float beta_h = scalar_s[kBeta];

        for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
            const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
            const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
            state_h[static_cast<int64_t>(k_idx) * value_dim + value_tile_begin + local_v] *= decay;
        }
        __syncthreads();

        for (int32_t local_v = threadIdx.x; local_v < value_tile_size; local_v += blockDim.x) {
            const int32_t v_idx = value_tile_begin + local_v;
            float kv_mem = 0.0f;
            for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
                const float k_norm = l2norm_value_for_transformers_dtype(key_h, k_idx, k_inv_norm);
                kv_mem += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] *
                          k_norm;
            }
            delta_s[local_v] = (load_as_float(value_h, v_idx) - kv_mem) * beta_h;
        }
        __syncthreads();

        for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
            const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
            const int32_t v_idx = value_tile_begin + local_v;
            const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
            const float k_norm = l2norm_value_for_transformers_dtype(key_h, k_idx, k_inv_norm);
            state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] += k_norm * delta_s[local_v];
        }
        __syncthreads();

        for (int32_t local_v = threadIdx.x; local_v < value_tile_size; local_v += blockDim.x) {
            const int32_t v_idx = value_tile_begin + local_v;
            float acc = 0.0f;
            for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
                const float q_norm = l2norm_value_for_transformers_dtype(query_h, k_idx, q_inv_norm) * q_scale;
                acc += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] *
                       q_norm;
            }
            store_from_float(output_h, v_idx, acc);
        }
        if (token + 1 < seq_len) {
            __syncthreads();
        }
    }
}

template <typename T, typename BiasT>
__global__ void gated_delta_sequence_from_ab_full_tile_128_kernel(const T* __restrict__ mixed_qkv,
                                                                  const T* __restrict__ a,
                                                                  const T* __restrict__ b,
                                                                  const float* __restrict__ a_log,
                                                                  const BiasT* __restrict__ dt_bias,
                                                                  float* __restrict__ recurrent_state,
                                                                  T* __restrict__ output,
                                                                  int32_t seq_len,
                                                                  int32_t num_heads) {
    constexpr int32_t kKeyDim = 128;
    constexpr int32_t kValueDim = 128;
    constexpr int32_t kValueTile = kGatedDeltaSequenceValueTile;
    constexpr int32_t kStateElements = kKeyDim * kValueTile;
    const int32_t head = blockIdx.x;
    const int32_t value_tile_begin = blockIdx.y * kValueTile;
    if (head >= num_heads) {
        return;
    }

    const int32_t key_total = num_heads * kKeyDim;
    const int32_t value_total = num_heads * kValueDim;
    const int32_t conv_dim = 2 * key_total + value_total;
    float* state_h = recurrent_state + static_cast<int64_t>(head) * kKeyDim * kValueDim;
    extern __shared__ float shared[];
    float* delta_s = shared;
    float* scalar_s = shared + kValueTile;
    constexpr int32_t kQInvNorm = 0;
    constexpr int32_t kKInvNorm = 1;
    constexpr int32_t kDecay = 2;
    constexpr int32_t kBeta = 3;
    const float q_scale = rsqrtf(static_cast<float>(kKeyDim));

    for (int32_t token = 0; token < seq_len; ++token) {
        const T* row = mixed_qkv + static_cast<int64_t>(token) * conv_dim;
        const T* query_h = row + static_cast<int64_t>(head) * kKeyDim;
        const T* key_h = row + static_cast<int64_t>(key_total) + static_cast<int64_t>(head) * kKeyDim;
        const T* value_h = row + static_cast<int64_t>(2 * key_total) + static_cast<int64_t>(head) * kValueDim;
        T* output_h = output + static_cast<int64_t>(token) * value_total + static_cast<int64_t>(head) * kValueDim;

        if (threadIdx.x == 0) {
            scalar_s[kQInvNorm] =
                l2norm_inv_for_transformers_dtype<T>(
                    l2norm_sum_for_transformers_dtype(query_h, kKeyDim),
                    1e-6f);
            scalar_s[kKInvNorm] =
                l2norm_inv_for_transformers_dtype<T>(
                    l2norm_sum_for_transformers_dtype(key_h, kKeyDim),
                    1e-6f);
            const int64_t gate_idx = static_cast<int64_t>(token) * num_heads + head;
            const float a_val = load_as_float(a, gate_idx);
            const float b_val = load_as_float(b, gate_idx);
            const float dt = load_as_float(dt_bias, head);
            const float g_value = -expf(a_log[head]) * softplus(a_val + dt);
            scalar_s[kDecay] = expf(g_value);
            scalar_s[kBeta] = sigmoid(b_val);
        }
        __syncthreads();
        const float q_inv_norm = scalar_s[kQInvNorm];
        const float k_inv_norm = scalar_s[kKInvNorm];
        const float decay = scalar_s[kDecay];
        const float beta_h = scalar_s[kBeta];

        for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < kStateElements; idx += kGatedDeltaSequenceThreads) {
            const int32_t local_v = idx % kValueTile;
            const int32_t k_idx = idx / kValueTile;
            state_h[static_cast<int64_t>(k_idx) * kValueDim + value_tile_begin + local_v] *= decay;
        }
        __syncthreads();

        for (int32_t local_v = static_cast<int32_t>(threadIdx.x); local_v < kValueTile; local_v += kGatedDeltaSequenceThreads) {
            const int32_t v_idx = value_tile_begin + local_v;
            float kv_mem = 0.0f;
            for (int32_t k_idx = 0; k_idx < kKeyDim; ++k_idx) {
                const float k_norm = l2norm_value_for_transformers_dtype(key_h, k_idx, k_inv_norm);
                kv_mem += state_h[static_cast<int64_t>(k_idx) * kValueDim + v_idx] *
                          k_norm;
            }
            delta_s[local_v] = (load_as_float(value_h, v_idx) - kv_mem) * beta_h;
        }
        __syncthreads();

        for (int32_t idx = static_cast<int32_t>(threadIdx.x); idx < kStateElements; idx += kGatedDeltaSequenceThreads) {
            const int32_t local_v = idx % kValueTile;
            const int32_t v_idx = value_tile_begin + local_v;
            const int32_t k_idx = idx / kValueTile;
            const float k_norm = l2norm_value_for_transformers_dtype(key_h, k_idx, k_inv_norm);
            state_h[static_cast<int64_t>(k_idx) * kValueDim + v_idx] += k_norm * delta_s[local_v];
        }
        __syncthreads();

        for (int32_t local_v = static_cast<int32_t>(threadIdx.x); local_v < kValueTile; local_v += kGatedDeltaSequenceThreads) {
            const int32_t v_idx = value_tile_begin + local_v;
            float acc = 0.0f;
            for (int32_t k_idx = 0; k_idx < kKeyDim; ++k_idx) {
                const float q_norm = l2norm_value_for_transformers_dtype(query_h, k_idx, q_inv_norm) * q_scale;
                acc += state_h[static_cast<int64_t>(k_idx) * kValueDim + v_idx] *
                       q_norm;
            }
            store_from_float(output_h, v_idx, acc);
        }
        if (token + 1 < seq_len) {
            __syncthreads();
        }
    }
}

template <typename T, typename BiasT>
__global__ void gated_delta_sequence_from_ab_single_token_128_kernel(const T* __restrict__ mixed_qkv,
                                                                     const T* __restrict__ a,
                                                                     const T* __restrict__ b,
                                                                     const float* __restrict__ a_log,
                                                                     const BiasT* __restrict__ dt_bias,
                                                                     float* __restrict__ recurrent_state,
                                                                     T* __restrict__ output,
                                                                     int32_t num_heads) {
    constexpr int32_t kKeyDim = 128;
    constexpr int32_t kValueDim = 128;
    constexpr int32_t kValueTile = kGatedDeltaSequenceValueTile;
    constexpr int32_t kBlockThreads = 128;
    const int32_t head = blockIdx.x;
    const int32_t value_tile_begin = blockIdx.y * kValueTile;
    if (head >= num_heads) {
        return;
    }

    const int32_t key_total = num_heads * kKeyDim;
    float* state_h = recurrent_state + static_cast<int64_t>(head) * kKeyDim * kValueDim;
    extern __shared__ float shared[];
    float* q_norm_s = shared;
    float* k_norm_s = q_norm_s + kBlockThreads;
    float* scalar_s = k_norm_s + kBlockThreads;
    constexpr int32_t kQInvNorm = 0;
    constexpr int32_t kKInvNorm = 1;
    constexpr int32_t kDecay = 2;
    constexpr int32_t kBeta = 3;
    const float q_scale = rsqrtf(static_cast<float>(kKeyDim));

    const T* query_h = mixed_qkv + static_cast<int64_t>(head) * kKeyDim;
    const T* key_h = mixed_qkv + static_cast<int64_t>(key_total) + static_cast<int64_t>(head) * kKeyDim;
    const T* value_h = mixed_qkv + static_cast<int64_t>(2 * key_total) + static_cast<int64_t>(head) * kValueDim;
    T* output_h = output + static_cast<int64_t>(head) * kValueDim;

    if (threadIdx.x < kKeyDim) {
        const float q = load_as_float(query_h, static_cast<int32_t>(threadIdx.x));
        const float k = load_as_float(key_h, static_cast<int32_t>(threadIdx.x));
        q_norm_s[threadIdx.x] = round_to_dtype<T>(q * q);
        k_norm_s[threadIdx.x] = round_to_dtype<T>(k * k);
    } else {
        q_norm_s[threadIdx.x] = 0.0f;
        k_norm_s[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (uint32_t stride = kBlockThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            q_norm_s[threadIdx.x] += q_norm_s[threadIdx.x + stride];
            k_norm_s[threadIdx.x] += k_norm_s[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scalar_s[kQInvNorm] =
            l2norm_inv_for_transformers_dtype<T>(
                round_to_dtype<T>(q_norm_s[0]),
                1e-6f);
        scalar_s[kKInvNorm] =
            l2norm_inv_for_transformers_dtype<T>(
                round_to_dtype<T>(k_norm_s[0]),
                1e-6f);
        const int64_t gate_idx = head;
        const float a_val = load_as_float(a, gate_idx);
        const float b_val = load_as_float(b, gate_idx);
        const float dt = load_as_float(dt_bias, head);
        const float g_value = -expf(a_log[head]) * softplus(a_val + dt);
        scalar_s[kDecay] = expf(g_value);
        scalar_s[kBeta] = sigmoid(b_val);
    }
    __syncthreads();
    const float q_inv_norm = scalar_s[kQInvNorm];
    const float k_inv_norm = scalar_s[kKInvNorm];
    const float decay = scalar_s[kDecay];
    const float beta_h = scalar_s[kBeta];

    if (threadIdx.x < kKeyDim) {
        q_norm_s[threadIdx.x] =
            l2norm_value_for_transformers_dtype(query_h, static_cast<int32_t>(threadIdx.x), q_inv_norm) *
            q_scale;
        k_norm_s[threadIdx.x] =
            l2norm_value_for_transformers_dtype(key_h, static_cast<int32_t>(threadIdx.x), k_inv_norm);
    }
    __syncthreads();

    const int32_t local_v = static_cast<int32_t>(threadIdx.x);
    if (local_v < kValueTile) {
        const int32_t v_idx = value_tile_begin + local_v;
        float kv_mem = 0.0f;
        for (int32_t k_idx = 0; k_idx < kKeyDim; ++k_idx) {
            float* state_ptr = state_h + static_cast<int64_t>(k_idx) * kValueDim + v_idx;
            const float state_value = *state_ptr * decay;
            *state_ptr = state_value;
            kv_mem += state_value * k_norm_s[k_idx];
        }
        const float delta = (load_as_float(value_h, v_idx) - kv_mem) * beta_h;
        float acc = 0.0f;
        for (int32_t k_idx = 0; k_idx < kKeyDim; ++k_idx) {
            float* state_ptr = state_h + static_cast<int64_t>(k_idx) * kValueDim + v_idx;
            const float state_value = *state_ptr + k_norm_s[k_idx] * delta;
            *state_ptr = state_value;
            acc += state_value * q_norm_s[k_idx];
        }
        store_from_float(output_h, v_idx, acc);
    }
}

template <typename T>
__global__ void precompute_gated_delta_qk_kernel(const T* __restrict__ mixed_qkv,
                                                 float* __restrict__ query_norm,
                                                 float* __restrict__ key_norm,
                                                 int32_t seq_len,
                                                 int32_t num_heads,
                                                 int32_t key_dim,
                                                 int32_t value_dim) {
    const int32_t token = blockIdx.x;
    const int32_t head = blockIdx.y;
    const bool is_query = blockIdx.z == 0;
    if (token >= seq_len || head >= num_heads) {
        return;
    }

    const int32_t key_total = num_heads * key_dim;
    const int32_t value_total = num_heads * value_dim;
    const int32_t conv_dim = 2 * key_total + value_total;
    const T* row = mixed_qkv + static_cast<int64_t>(token) * conv_dim;
    const T* x_h = is_query
        ? row + static_cast<int64_t>(head) * key_dim
        : row + static_cast<int64_t>(key_total) + static_cast<int64_t>(head) * key_dim;

    extern __shared__ float smem[];
    float sum = 0.0f;
    for (int32_t k_idx = threadIdx.x; k_idx < key_dim; k_idx += blockDim.x) {
        const float v = load_as_float(x_h, k_idx);
        sum += round_to_dtype<T>(v * v);
    }

    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_norm = l2norm_inv_for_transformers_dtype<T>(smem[0], 1e-6f);
    const float q_scale = is_query ? rsqrtf(static_cast<float>(key_dim)) : 1.0f;
    float* out = is_query ? query_norm : key_norm;
    const int64_t base = (static_cast<int64_t>(token) * num_heads + head) * key_dim;
    for (int32_t k_idx = threadIdx.x; k_idx < key_dim; k_idx += blockDim.x) {
        out[base + k_idx] = l2norm_value_for_transformers_dtype(x_h, k_idx, inv_norm) * q_scale;
    }
}

template <typename T, bool kInputIsDecay, int32_t kValueTile>
__global__ void gated_delta_sequence_precomputed_kernel(const T* __restrict__ mixed_qkv,
                                                        const float* __restrict__ g_or_decay,
                                                        const float* __restrict__ beta,
                                                        const float* __restrict__ query_norm,
                                                        const float* __restrict__ key_norm,
                                                        float* __restrict__ recurrent_state,
                                                        T* __restrict__ output,
                                                        int32_t seq_len,
                                                        int32_t num_heads,
                                                        int32_t key_dim,
                                                        int32_t value_dim) {
    const int32_t head = blockIdx.x;
    const int32_t value_tile_begin = blockIdx.y * kValueTile;
    const int32_t value_tile_size = min(kValueTile, value_dim - value_tile_begin);
    if (head >= num_heads) {
        return;
    }
    if (value_tile_size <= 0) {
        return;
    }

    const int32_t key_total = num_heads * key_dim;
    const int32_t value_total = num_heads * value_dim;
    const int32_t conv_dim = 2 * key_total + value_total;
    float* state_h = recurrent_state + static_cast<int64_t>(head) * key_dim * value_dim;
    extern __shared__ float delta_s[];
    const int64_t state_elements = static_cast<int64_t>(key_dim) * value_tile_size;

    for (int32_t token = 0; token < seq_len; ++token) {
        const T* row = mixed_qkv + static_cast<int64_t>(token) * conv_dim;
        const T* value_h = row + static_cast<int64_t>(2 * key_total) + static_cast<int64_t>(head) * value_dim;
        T* output_h = output + static_cast<int64_t>(token) * value_total + static_cast<int64_t>(head) * value_dim;
        const float* query_h = query_norm + (static_cast<int64_t>(token) * num_heads + head) * key_dim;
        const float* key_h = key_norm + (static_cast<int64_t>(token) * num_heads + head) * key_dim;
        const float g_value = g_or_decay[static_cast<int64_t>(token) * num_heads + head];
        const float decay = kInputIsDecay ? g_value : expf(g_value);
        const float beta_h = beta[static_cast<int64_t>(token) * num_heads + head];

        if constexpr ((kValueTile % 4) == 0) {
            if (value_tile_size == kValueTile && (value_tile_begin % 4) == 0) {
                constexpr int32_t kVecWidth = 4;
                constexpr int32_t kVecsPerTile = kValueTile / kVecWidth;
                const int64_t state_vecs = static_cast<int64_t>(key_dim) * kVecsPerTile;
                for (int64_t idx = threadIdx.x; idx < state_vecs; idx += blockDim.x) {
                    const int32_t local_vec = static_cast<int32_t>(idx % kVecsPerTile);
                    const int32_t local_v = local_vec * kVecWidth;
                    const int32_t k_idx = static_cast<int32_t>(idx / kVecsPerTile);
                    float4* state_vec = reinterpret_cast<float4*>(
                        state_h + static_cast<int64_t>(k_idx) * value_dim + value_tile_begin + local_v);
                    float4 value = *state_vec;
                    value.x *= decay;
                    value.y *= decay;
                    value.z *= decay;
                    value.w *= decay;
                    *state_vec = value;
                }
            } else {
                for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
                    const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
                    const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
                    state_h[static_cast<int64_t>(k_idx) * value_dim + value_tile_begin + local_v] *= decay;
                }
            }
        } else {
            for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
                const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
                const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
                state_h[static_cast<int64_t>(k_idx) * value_dim + value_tile_begin + local_v] *= decay;
            }
        }
        __syncthreads();

        for (int32_t local_v = threadIdx.x; local_v < value_tile_size; local_v += blockDim.x) {
            const int32_t v_idx = value_tile_begin + local_v;
            float kv_mem = 0.0f;
            for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
                kv_mem += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] * key_h[k_idx];
            }
            delta_s[local_v] = (load_as_float(value_h, v_idx) - kv_mem) * beta_h;
        }
        __syncthreads();

        if constexpr ((kValueTile % 4) == 0) {
            if (value_tile_size == kValueTile && (value_tile_begin % 4) == 0) {
                constexpr int32_t kVecWidth = 4;
                constexpr int32_t kVecsPerTile = kValueTile / kVecWidth;
                const int64_t state_vecs = static_cast<int64_t>(key_dim) * kVecsPerTile;
                for (int64_t idx = threadIdx.x; idx < state_vecs; idx += blockDim.x) {
                    const int32_t local_vec = static_cast<int32_t>(idx % kVecsPerTile);
                    const int32_t local_v = local_vec * kVecWidth;
                    const int32_t k_idx = static_cast<int32_t>(idx / kVecsPerTile);
                    const float k_value = key_h[k_idx];
                    float4* state_vec = reinterpret_cast<float4*>(
                        state_h + static_cast<int64_t>(k_idx) * value_dim + value_tile_begin + local_v);
                    float4 value = *state_vec;
                    value.x += k_value * delta_s[local_v + 0];
                    value.y += k_value * delta_s[local_v + 1];
                    value.z += k_value * delta_s[local_v + 2];
                    value.w += k_value * delta_s[local_v + 3];
                    *state_vec = value;
                }
            } else {
                for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
                    const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
                    const int32_t v_idx = value_tile_begin + local_v;
                    const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
                    state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] += key_h[k_idx] * delta_s[local_v];
                }
            }
        } else {
            for (int64_t idx = threadIdx.x; idx < state_elements; idx += blockDim.x) {
                const int32_t local_v = static_cast<int32_t>(idx % value_tile_size);
                const int32_t v_idx = value_tile_begin + local_v;
                const int32_t k_idx = static_cast<int32_t>(idx / value_tile_size);
                state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] += key_h[k_idx] * delta_s[local_v];
            }
        }
        __syncthreads();

        for (int32_t local_v = threadIdx.x; local_v < value_tile_size; local_v += blockDim.x) {
            const int32_t v_idx = value_tile_begin + local_v;
            float acc = 0.0f;
            for (int32_t k_idx = 0; k_idx < key_dim; ++k_idx) {
                acc += state_h[static_cast<int64_t>(k_idx) * value_dim + v_idx] * query_h[k_idx];
            }
            store_from_float(output_h, v_idx, acc);
        }
        if (token + 1 < seq_len) {
            __syncthreads();
        }
    }
}

template <typename T>
__global__ void partial_interleaved_mrope_kernel(T* __restrict__ data,
                                                const int32_t* __restrict__ position_ids,
                                                int32_t mrope_section_h,
                                                int32_t mrope_section_w,
                                                int32_t seq_len,
                                                int32_t num_heads,
                                                int32_t head_dim,
                                                int32_t rotary_dim,
                                                float rope_theta) {
    const int32_t token = blockIdx.x;
    const int32_t head = blockIdx.y;
    const int32_t d = threadIdx.x;
    const int32_t freq_dim = rotary_dim / 2;
    if (token >= seq_len || head >= num_heads || d >= freq_dim) {
        return;
    }

    int32_t axis = 0;
    const int32_t h_limit = mrope_section_h * 3;
    const int32_t w_limit = mrope_section_w * 3;
    if (d >= 1 && d < h_limit && ((d - 1) % 3) == 0) {
        axis = 1;
    }
    if (d >= 2 && d < w_limit && ((d - 2) % 3) == 0) {
        axis = 2;
    }

    const float exponent = static_cast<float>(2 * d) / static_cast<float>(rotary_dim);
    const float inv_freq = 1.0f / powf(rope_theta, exponent);
    const float angle = static_cast<float>(position_ids[axis * seq_len + token]) * inv_freq;
    float s, c;
    __sincosf(angle, &s, &c);

    const int64_t base = (static_cast<int64_t>(token) * num_heads + head) * head_dim;
    const float lo = load_as_float(data, base + d);
    const float hi = load_as_float(data, base + d + freq_dim);
    store_from_float(data, base + d, lo * c - hi * s);
    store_from_float(data, base + d + freq_dim, hi * c + lo * s);
}

void validate_2d_same_device(const Tensor& tensor,
                             const std::string& name,
                             DType dtype,
                             Device device,
                             int32_t device_id) {
    auto [tensor_device, tensor_device_id] = tensor.device();
    check<DeviceError>(tensor_device == device && tensor_device_id == device_id,
                       name + " tensor must be on the same device");
    check<ConfigurationError>(tensor.dtype() == dtype, name + " tensor dtype mismatch");
    check<InvalidRequestError>(tensor.shape().size() == 2, name + " tensor must be 2D");
}

void validate_3d_same_device(const Tensor& tensor,
                             const std::string& name,
                             DType dtype,
                             Device device,
                             int32_t device_id) {
    auto [tensor_device, tensor_device_id] = tensor.device();
    check<DeviceError>(tensor_device == device && tensor_device_id == device_id,
                       name + " tensor must be on the same device");
    check<ConfigurationError>(tensor.dtype() == dtype, name + " tensor dtype mismatch");
    check<InvalidRequestError>(tensor.shape().size() == 3, name + " tensor must be 3D");
}

void validate_last_dim_weight(const Tensor& weight, int64_t hidden, DType dtype, Device device, int32_t device_id) {
    auto [weight_device, weight_device_id] = weight.device();
    check<DeviceError>(weight_device == device && weight_device_id == device_id,
                       "weight tensor must be on the same device");
    check<ConfigurationError>(weight.dtype() == dtype, "weight tensor dtype mismatch");
    check<InvalidRequestError>(
        weight.shape().size() == 1 && weight.shape()[0] == hidden,
        "weight tensor shape must match hidden dimension");
}

void validate_last_dim_weight_dtype_any(const Tensor& weight,
                                        int64_t hidden,
                                        Device device,
                                        int32_t device_id) {
    auto [weight_device, weight_device_id] = weight.device();
    check<DeviceError>(weight_device == device && weight_device_id == device_id,
                       "weight tensor must be on the same device");
    check<ConfigurationError>(
        weight.dtype() == DType::Float32 || weight.dtype() == DType::Float16 || weight.dtype() == DType::BFloat16,
        "weight tensor dtype must be Float32/Float16/BFloat16");
    check<InvalidRequestError>(
        weight.shape().size() == 1 && weight.shape()[0] == hidden,
        "weight tensor shape must match hidden dimension");
}

int64_t element_count(const Tensor& tensor) {
    int64_t elements = 1;
    for (int64_t dim : tensor.shape()) {
        elements *= dim;
    }
    return elements;
}

template <typename T, typename WeightT, bool kGated>
void launch_rmsnorm(const Tensor& input,
                    const Tensor* gate,
                    const Tensor& weight,
                    Tensor& output,
                    float eps,
                    cudaStream_t stream) {
    const auto& shape = input.shape();
    const int64_t rows = shape[0];
    const int64_t hidden = shape[1];
    if (hidden <= 128) {
        constexpr uint32_t threads = 32;
        qwen3_5_rmsnorm_one_warp_kernel<T, WeightT, kGated><<<rows, threads, 0, stream>>>(
            static_cast<const T*>(input.data_ptr()),
            gate == nullptr ? nullptr : static_cast<const T*>(gate->data_ptr()),
            static_cast<const WeightT*>(weight.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            rows,
            hidden,
            eps);
        CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5_rmsnorm small-hidden kernel launch failed");
        return;
    }
    const uint32_t threads = 256;
    const size_t smem = threads * sizeof(float);
    if constexpr (!kGated) {
        if (hidden == 1024) {
            qwen3_5_rmsnorm_fixed_hidden_kernel<T, WeightT, kGated, 1024, 256><<<rows, threads, smem, stream>>>(
                static_cast<const T*>(input.data_ptr()),
                gate == nullptr ? nullptr : static_cast<const T*>(gate->data_ptr()),
                static_cast<const WeightT*>(weight.data_ptr()),
                static_cast<T*>(output.data_ptr()),
                rows,
                eps);
            CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5_rmsnorm fixed-hidden 1024 kernel launch failed");
            return;
        }
        if (hidden == 2048) {
            qwen3_5_rmsnorm_fixed_hidden_kernel<T, WeightT, kGated, 2048, 256><<<rows, threads, smem, stream>>>(
                static_cast<const T*>(input.data_ptr()),
                gate == nullptr ? nullptr : static_cast<const T*>(gate->data_ptr()),
                static_cast<const WeightT*>(weight.data_ptr()),
                static_cast<T*>(output.data_ptr()),
                rows,
                eps);
            CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5_rmsnorm fixed-hidden 2048 kernel launch failed");
            return;
        }
    }
    qwen3_5_rmsnorm_kernel<T, WeightT, kGated><<<rows, threads, smem, stream>>>(
        static_cast<const T*>(input.data_ptr()),
        gate == nullptr ? nullptr : static_cast<const T*>(gate->data_ptr()),
        static_cast<const WeightT*>(weight.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        rows,
        hidden,
        eps);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5_rmsnorm kernel launch failed");
}

template <typename T>
void launch_add(const Tensor& lhs, const Tensor& rhs, Tensor& output, cudaStream_t stream) {
    const int64_t elements = element_count(output);
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    add_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(lhs.data_ptr()),
        static_cast<const T*>(rhs.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        elements);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 add kernel launch failed");
}

template <typename T>
void launch_mul_sigmoid(const Tensor& input, const Tensor& gate, Tensor& output, cudaStream_t stream) {
    const int64_t elements = element_count(output);
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    mul_sigmoid_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(input.data_ptr()),
        static_cast<const T*>(gate.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        elements);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 mul sigmoid kernel launch failed");
}

template <typename T>
void launch_split_q_gate(const Tensor& q_proj,
                         Tensor& query,
                         Tensor& gate,
                         int32_t num_heads,
                         int32_t head_dim,
                         cudaStream_t stream) {
    const int32_t seq_len = static_cast<int32_t>(q_proj.shape()[0]);
    const int64_t elements = static_cast<int64_t>(seq_len) * num_heads * head_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    split_q_gate_kernel<T><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(q_proj.data_ptr()),
        static_cast<T*>(query.data_ptr()),
        static_cast<T*>(gate.data_ptr()),
        seq_len,
        num_heads,
        head_dim);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 split q/gate kernel launch failed");
}

template <typename T>
void launch_depthwise_conv(const Tensor& input,
                           const Tensor& weight,
                           Tensor& conv_state,
                           Tensor& output,
                           bool update_state,
                           cudaStream_t stream) {
    const int32_t seq_len = static_cast<int32_t>(input.shape()[0]);
    const int32_t channels = static_cast<int32_t>(input.shape()[1]);
    const int32_t kernel_size = static_cast<int32_t>(weight.shape()[2]);
    const int64_t elements = static_cast<int64_t>(seq_len) * channels;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    if (kernel_size == 4 && seq_len == 1 && update_state) {
        depthwise_causal_conv1d_kernel4_single_token_update<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(input.data_ptr()),
            static_cast<const T*>(weight.data_ptr()),
            static_cast<T*>(conv_state.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            channels);
        CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 depthwise conv kernel4 single-token update launch failed");
        return;
    }
    if (kernel_size == 4) {
        depthwise_causal_conv1d_kernel4<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(input.data_ptr()),
            static_cast<const T*>(weight.data_ptr()),
            static_cast<const T*>(conv_state.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            seq_len,
            channels);
    } else {
        depthwise_causal_conv1d_kernel<T><<<blocks, threads, 0, stream>>>(
            static_cast<const T*>(input.data_ptr()),
            static_cast<const T*>(weight.data_ptr()),
            static_cast<const T*>(conv_state.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            seq_len,
            channels,
            kernel_size);
    }
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 depthwise conv kernel launch failed");

    if (!update_state) {
        return;
    }
    auto [device, device_id] = input.device();
    (void)device;
    const size_t state_bytes = static_cast<size_t>(channels) * static_cast<size_t>(kernel_size) * sizeof(T);
    void* tmp_ptr = StaticBufferManager::get_cache_buf("qwen3_5_conv_state_update_tmp", state_bytes, device_id);
    const int64_t state_elements = static_cast<int64_t>(channels) * kernel_size;
    const int state_blocks = static_cast<int>((state_elements + threads - 1) / threads);
    update_conv_state_kernel<T><<<state_blocks, threads, 0, stream>>>(
        static_cast<const T*>(input.data_ptr()),
        static_cast<const T*>(conv_state.data_ptr()),
        static_cast<T*>(tmp_ptr),
        seq_len,
        channels,
        kernel_size);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 update conv state kernel launch failed");
    CUDA_CHECK_THROW(cudaMemcpyAsync(
                         conv_state.data_ptr(),
                         tmp_ptr,
                         state_bytes,
                         cudaMemcpyDeviceToDevice,
                         stream),
                     "qwen3_5 copy next conv state");
}

template <typename T, typename BiasT>
void launch_compute_g_beta(const Tensor& a,
                           const Tensor& b,
                           const Tensor& a_log,
                           const Tensor& dt_bias,
                           Tensor& g,
                           Tensor& beta,
                           cudaStream_t stream) {
    const int64_t rows = a.shape()[0];
    const int64_t heads = a.shape()[1];
    const int64_t elements = rows * heads;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    compute_g_beta_kernel<T, BiasT><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(a.data_ptr()),
        static_cast<const T*>(b.data_ptr()),
        static_cast<const float*>(a_log.data_ptr()),
        static_cast<const BiasT*>(dt_bias.data_ptr()),
        static_cast<float*>(g.data_ptr()),
        static_cast<float*>(beta.data_ptr()),
        rows,
        heads);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 compute g/beta kernel launch failed");
}

template <typename T, typename BiasT>
void launch_compute_decay_beta(const Tensor& a,
                               const Tensor& b,
                               const Tensor& a_log,
                               const Tensor& dt_bias,
                               Tensor& decay,
                               Tensor& beta,
                               cudaStream_t stream) {
    const int64_t rows = a.shape()[0];
    const int64_t heads = a.shape()[1];
    const int64_t elements = rows * heads;
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    compute_decay_beta_kernel<T, BiasT><<<blocks, threads, 0, stream>>>(
        static_cast<const T*>(a.data_ptr()),
        static_cast<const T*>(b.data_ptr()),
        static_cast<const float*>(a_log.data_ptr()),
        static_cast<const BiasT*>(dt_bias.data_ptr()),
        static_cast<float*>(decay.data_ptr()),
        static_cast<float*>(beta.data_ptr()),
        elements,
        heads);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 compute decay/beta kernel launch failed");
}

template <typename T>
void launch_gated_delta_sequence(const Tensor& mixed_qkv,
                                 const Tensor& g,
                                 const Tensor& beta,
                                 Tensor& recurrent_state,
                                 Tensor& output,
                                 cudaStream_t stream) {
    const int32_t seq_len = static_cast<int32_t>(mixed_qkv.shape()[0]);
    const int32_t num_heads = static_cast<int32_t>(recurrent_state.shape()[0]);
    const int32_t key_dim = static_cast<int32_t>(recurrent_state.shape()[1]);
    const int32_t value_dim = static_cast<int32_t>(recurrent_state.shape()[2]);
    dim3 grid(
        num_heads,
        (value_dim + kGatedDeltaSequenceValueTile - 1) / kGatedDeltaSequenceValueTile);
    const size_t smem = static_cast<size_t>(kGatedDeltaSequenceValueTile + 4) * sizeof(float);
    gated_delta_sequence_kernel<T><<<grid, kGatedDeltaSequenceThreads, smem, stream>>>(
        static_cast<const T*>(mixed_qkv.data_ptr()),
        static_cast<const float*>(g.data_ptr()),
        static_cast<const float*>(beta.data_ptr()),
        static_cast<float*>(recurrent_state.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        seq_len,
        num_heads,
        key_dim,
        value_dim);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 gated delta sequence kernel launch failed");
}

template <typename T, typename BiasT>
void launch_gated_delta_sequence_from_ab(const Tensor& mixed_qkv,
                                         const Tensor& a,
                                         const Tensor& b,
                                         const Tensor& a_log,
                                         const Tensor& dt_bias,
                                         Tensor& recurrent_state,
                                         Tensor& output,
                                         cudaStream_t stream) {
    const int32_t seq_len = static_cast<int32_t>(mixed_qkv.shape()[0]);
    const int32_t num_heads = static_cast<int32_t>(recurrent_state.shape()[0]);
    const int32_t key_dim = static_cast<int32_t>(recurrent_state.shape()[1]);
    const int32_t value_dim = static_cast<int32_t>(recurrent_state.shape()[2]);
    if (key_dim == 128 && value_dim == 128) {
        dim3 grid(num_heads, 128 / kGatedDeltaSequenceValueTile);
        const size_t smem = static_cast<size_t>(kGatedDeltaSequenceValueTile + 4) * sizeof(float);
        if (seq_len == 1) {
            constexpr int32_t kSingleTokenThreads = 128;
            const size_t single_token_smem =
                static_cast<size_t>(2 * kSingleTokenThreads + 4) * sizeof(float);
            gated_delta_sequence_from_ab_single_token_128_kernel<T, BiasT><<<grid, kSingleTokenThreads, single_token_smem, stream>>>(
                static_cast<const T*>(mixed_qkv.data_ptr()),
                static_cast<const T*>(a.data_ptr()),
                static_cast<const T*>(b.data_ptr()),
                static_cast<const float*>(a_log.data_ptr()),
                static_cast<const BiasT*>(dt_bias.data_ptr()),
                static_cast<float*>(recurrent_state.data_ptr()),
                static_cast<T*>(output.data_ptr()),
                num_heads);
            CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 gated delta sequence from a/b single-token 128 kernel launch failed");
            return;
        }
        gated_delta_sequence_from_ab_full_tile_128_kernel<T, BiasT><<<grid, kGatedDeltaSequenceThreads, smem, stream>>>(
            static_cast<const T*>(mixed_qkv.data_ptr()),
            static_cast<const T*>(a.data_ptr()),
            static_cast<const T*>(b.data_ptr()),
            static_cast<const float*>(a_log.data_ptr()),
            static_cast<const BiasT*>(dt_bias.data_ptr()),
            static_cast<float*>(recurrent_state.data_ptr()),
            static_cast<T*>(output.data_ptr()),
            seq_len,
            num_heads);
        CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 gated delta sequence from a/b full-tile 128 kernel launch failed");
        return;
    }
    dim3 grid(
        num_heads,
        (value_dim + kGatedDeltaSequenceValueTile - 1) / kGatedDeltaSequenceValueTile);
    const size_t smem = static_cast<size_t>(kGatedDeltaSequenceValueTile + 4) * sizeof(float);
    gated_delta_sequence_from_ab_kernel<T, BiasT><<<grid, kGatedDeltaSequenceThreads, smem, stream>>>(
        static_cast<const T*>(mixed_qkv.data_ptr()),
        static_cast<const T*>(a.data_ptr()),
        static_cast<const T*>(b.data_ptr()),
        static_cast<const float*>(a_log.data_ptr()),
        static_cast<const BiasT*>(dt_bias.data_ptr()),
        static_cast<float*>(recurrent_state.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        seq_len,
        num_heads,
        key_dim,
        value_dim);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 gated delta sequence from a/b kernel launch failed");
}

template <typename T>
void launch_precompute_gated_delta_qk(const Tensor& mixed_qkv,
                                      Tensor& query_norm,
                                      Tensor& key_norm,
                                      cudaStream_t stream) {
    const int32_t seq_len = static_cast<int32_t>(mixed_qkv.shape()[0]);
    const int32_t num_heads = static_cast<int32_t>(query_norm.shape()[1]);
    const int32_t key_dim = static_cast<int32_t>(query_norm.shape()[2]);
    const int32_t value_dim =
        static_cast<int32_t>(mixed_qkv.shape()[1] / num_heads - 2 * key_dim);
    dim3 grid(seq_len, num_heads, 2);
    const size_t smem = static_cast<size_t>(kGatedDeltaPrecomputeThreads) * sizeof(float);
    precompute_gated_delta_qk_kernel<T><<<grid, kGatedDeltaPrecomputeThreads, smem, stream>>>(
        static_cast<const T*>(mixed_qkv.data_ptr()),
        static_cast<float*>(query_norm.data_ptr()),
        static_cast<float*>(key_norm.data_ptr()),
        seq_len,
        num_heads,
        key_dim,
        value_dim);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 precompute gated delta q/k kernel launch failed");
}

template <typename T>
void launch_gated_delta_sequence_precomputed(const Tensor& mixed_qkv,
                                             const Tensor& g,
                                             const Tensor& beta,
                                             const Tensor& query_norm,
                                             const Tensor& key_norm,
                                             Tensor& recurrent_state,
                                             Tensor& output,
                                             cudaStream_t stream) {
    const int32_t seq_len = static_cast<int32_t>(mixed_qkv.shape()[0]);
    const int32_t num_heads = static_cast<int32_t>(recurrent_state.shape()[0]);
    const int32_t key_dim = static_cast<int32_t>(recurrent_state.shape()[1]);
    const int32_t value_dim = static_cast<int32_t>(recurrent_state.shape()[2]);
    dim3 grid(
        num_heads,
        (value_dim + kGatedDeltaPrecomputedValueTile - 1) / kGatedDeltaPrecomputedValueTile);
    const size_t smem = static_cast<size_t>(kGatedDeltaPrecomputedValueTile) * sizeof(float);
    gated_delta_sequence_precomputed_kernel<T, false, kGatedDeltaPrecomputedValueTile><<<grid, kGatedDeltaSequenceThreads, smem, stream>>>(
        static_cast<const T*>(mixed_qkv.data_ptr()),
        static_cast<const float*>(g.data_ptr()),
        static_cast<const float*>(beta.data_ptr()),
        static_cast<const float*>(query_norm.data_ptr()),
        static_cast<const float*>(key_norm.data_ptr()),
        static_cast<float*>(recurrent_state.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        seq_len,
        num_heads,
        key_dim,
        value_dim);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 precomputed gated delta sequence kernel launch failed");
}

template <typename T>
void launch_gated_delta_sequence_precomputed_decay(const Tensor& mixed_qkv,
                                                   const Tensor& decay,
                                                   const Tensor& beta,
                                                   const Tensor& query_norm,
                                                   const Tensor& key_norm,
                                                   Tensor& recurrent_state,
                                                   Tensor& output,
                                                   cudaStream_t stream) {
    const int32_t seq_len = static_cast<int32_t>(mixed_qkv.shape()[0]);
    const int32_t num_heads = static_cast<int32_t>(recurrent_state.shape()[0]);
    const int32_t key_dim = static_cast<int32_t>(recurrent_state.shape()[1]);
    const int32_t value_dim = static_cast<int32_t>(recurrent_state.shape()[2]);
    dim3 grid(
        num_heads,
        (value_dim + kGatedDeltaPrecomputedValueTile - 1) / kGatedDeltaPrecomputedValueTile);
    const size_t smem = static_cast<size_t>(kGatedDeltaPrecomputedValueTile) * sizeof(float);
    gated_delta_sequence_precomputed_kernel<T, true, kGatedDeltaPrecomputedValueTile><<<grid, kGatedDeltaSequenceThreads, smem, stream>>>(
        static_cast<const T*>(mixed_qkv.data_ptr()),
        static_cast<const float*>(decay.data_ptr()),
        static_cast<const float*>(beta.data_ptr()),
        static_cast<const float*>(query_norm.data_ptr()),
        static_cast<const float*>(key_norm.data_ptr()),
        static_cast<float*>(recurrent_state.data_ptr()),
        static_cast<T*>(output.data_ptr()),
        seq_len,
        num_heads,
        key_dim,
        value_dim);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 precomputed decay gated delta sequence kernel launch failed");
}

template <typename T>
void launch_partial_mrope(Tensor& tensor,
                          const Tensor& position_ids,
                          const std::vector<int32_t>& mrope_section,
                          float rope_theta,
                          int32_t rotary_dim,
                          cudaStream_t stream) {
    const auto& shape = tensor.shape();
    const int32_t seq_len = static_cast<int32_t>(shape[0]);
    const int32_t heads = static_cast<int32_t>(shape[1]);
    const int32_t head_dim = static_cast<int32_t>(shape[2]);
    dim3 grid(seq_len, heads);
    dim3 block(rotary_dim / 2);
    partial_interleaved_mrope_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<T*>(tensor.data_ptr()),
        static_cast<const int32_t*>(position_ids.data_ptr()),
        mrope_section[1],
        mrope_section[2],
        seq_len,
        heads,
        head_dim,
        rotary_dim,
        rope_theta);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 partial M-RoPE kernel launch failed");
}

} // namespace

void qwen3_5_rmsnorm_forward(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps,
    cudaStream_t stream) {
    auto [device, device_id] = input.device();
    validate_2d_same_device(output, "output", input.dtype(), device, device_id);
    validate_last_dim_weight(weight, input.shape().at(1), input.dtype(), device, device_id);
    check<InvalidRequestError>(output.shape() == input.shape(), "output shape must match input shape");

    if (input.dtype() == DType::BFloat16) {
        launch_rmsnorm<__nv_bfloat16, __nv_bfloat16, false>(input, nullptr, weight, output, eps, stream);
    } else if (input.dtype() == DType::Float16) {
        launch_rmsnorm<half, half, false>(input, nullptr, weight, output, eps, stream);
    } else if (input.dtype() == DType::Float32) {
        launch_rmsnorm<float, float, false>(input, nullptr, weight, output, eps, stream);
    } else {
        throw ConfigurationError("qwen3_5_rmsnorm supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_gated_rmsnorm_forward(
    const Tensor& input,
    const Tensor& gate,
    const Tensor& weight,
    Tensor& output,
    float eps,
    cudaStream_t stream) {
    auto [device, device_id] = input.device();
    validate_2d_same_device(gate, "gate", input.dtype(), device, device_id);
    validate_2d_same_device(output, "output", input.dtype(), device, device_id);
    validate_last_dim_weight_dtype_any(weight, input.shape().at(1), device, device_id);
    check<InvalidRequestError>(gate.shape() == input.shape(), "gate shape must match input shape");
    check<InvalidRequestError>(output.shape() == input.shape(), "output shape must match input shape");

    if (input.dtype() == DType::BFloat16) {
        if (weight.dtype() == DType::Float32) {
            launch_rmsnorm<__nv_bfloat16, float, true>(input, &gate, weight, output, eps, stream);
        } else if (weight.dtype() == DType::Float16) {
            launch_rmsnorm<__nv_bfloat16, half, true>(input, &gate, weight, output, eps, stream);
        } else {
            launch_rmsnorm<__nv_bfloat16, __nv_bfloat16, true>(input, &gate, weight, output, eps, stream);
        }
    } else if (input.dtype() == DType::Float16) {
        if (weight.dtype() == DType::Float32) {
            launch_rmsnorm<half, float, true>(input, &gate, weight, output, eps, stream);
        } else if (weight.dtype() == DType::BFloat16) {
            launch_rmsnorm<half, __nv_bfloat16, true>(input, &gate, weight, output, eps, stream);
        } else {
            launch_rmsnorm<half, half, true>(input, &gate, weight, output, eps, stream);
        }
    } else if (input.dtype() == DType::Float32) {
        if (weight.dtype() == DType::Float16) {
            launch_rmsnorm<float, half, true>(input, &gate, weight, output, eps, stream);
        } else if (weight.dtype() == DType::BFloat16) {
            launch_rmsnorm<float, __nv_bfloat16, true>(input, &gate, weight, output, eps, stream);
        } else {
            launch_rmsnorm<float, float, true>(input, &gate, weight, output, eps, stream);
        }
    } else {
        throw ConfigurationError("qwen3_5_gated_rmsnorm supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_add_forward(
    const Tensor& lhs,
    const Tensor& rhs,
    Tensor& output,
    cudaStream_t stream) {
    auto [device, device_id] = lhs.device();
    auto [rhs_device, rhs_device_id] = rhs.device();
    auto [out_device, out_device_id] = output.device();
    check<DeviceError>(
        rhs_device == device && rhs_device_id == device_id &&
            out_device == device && out_device_id == device_id,
        "qwen3_5_add tensors must be on the same device");
    check<ConfigurationError>(
        lhs.dtype() == rhs.dtype() && lhs.dtype() == output.dtype(),
        "qwen3_5_add dtype mismatch");
    check<InvalidRequestError>(
        lhs.shape() == rhs.shape() && lhs.shape() == output.shape(),
        "qwen3_5_add shape mismatch");

    if (lhs.dtype() == DType::BFloat16) {
        launch_add<__nv_bfloat16>(lhs, rhs, output, stream);
    } else if (lhs.dtype() == DType::Float16) {
        launch_add<half>(lhs, rhs, output, stream);
    } else if (lhs.dtype() == DType::Float32) {
        launch_add<float>(lhs, rhs, output, stream);
    } else {
        throw ConfigurationError("qwen3_5_add supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_mul_sigmoid_forward(
    const Tensor& input,
    const Tensor& gate,
    Tensor& output,
    cudaStream_t stream) {
    auto [device, device_id] = input.device();
    validate_2d_same_device(gate, "gate", input.dtype(), device, device_id);
    validate_2d_same_device(output, "output", input.dtype(), device, device_id);
    check<InvalidRequestError>(
        gate.shape() == input.shape() && output.shape() == input.shape(),
        "qwen3_5_mul_sigmoid shape mismatch");

    if (input.dtype() == DType::BFloat16) {
        launch_mul_sigmoid<__nv_bfloat16>(input, gate, output, stream);
    } else if (input.dtype() == DType::Float16) {
        launch_mul_sigmoid<half>(input, gate, output, stream);
    } else if (input.dtype() == DType::Float32) {
        launch_mul_sigmoid<float>(input, gate, output, stream);
    } else {
        throw ConfigurationError("qwen3_5_mul_sigmoid supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_split_q_gate_forward(
    const Tensor& q_proj,
    Tensor& query,
    Tensor& gate,
    int32_t num_heads,
    int32_t head_dim,
    cudaStream_t stream) {
    auto [device, device_id] = q_proj.device();
    validate_2d_same_device(gate, "gate", q_proj.dtype(), device, device_id);
    auto [query_device, query_device_id] = query.device();
    check<DeviceError>(
        query_device == device && query_device_id == device_id,
        "query tensor must be on the same device");
    check<ConfigurationError>(query.dtype() == q_proj.dtype(), "query tensor dtype mismatch");
    check<InvalidRequestError>(
        q_proj.shape().size() == 2 &&
            q_proj.shape()[1] == static_cast<int64_t>(num_heads) * head_dim * 2,
        "q_proj must be [seq_len, num_heads * head_dim * 2]");
    check<InvalidRequestError>(
        query.shape().size() == 3 && query.shape()[0] == q_proj.shape()[0] &&
            query.shape()[1] == num_heads && query.shape()[2] == head_dim,
        "query must be [seq_len, num_heads, head_dim]");
    check<InvalidRequestError>(
        gate.shape() == std::vector<int64_t>{q_proj.shape()[0], static_cast<int64_t>(num_heads) * head_dim},
        "gate must be [seq_len, num_heads * head_dim]");

    if (q_proj.dtype() == DType::BFloat16) {
        launch_split_q_gate<__nv_bfloat16>(q_proj, query, gate, num_heads, head_dim, stream);
    } else if (q_proj.dtype() == DType::Float16) {
        launch_split_q_gate<half>(q_proj, query, gate, num_heads, head_dim, stream);
    } else if (q_proj.dtype() == DType::Float32) {
        launch_split_q_gate<float>(q_proj, query, gate, num_heads, head_dim, stream);
    } else {
        throw ConfigurationError("qwen3_5_split_q_gate supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_depthwise_causal_conv1d_forward(
    const Tensor& input,
    const Tensor& weight,
    Tensor& conv_state,
    Tensor& output,
    bool update_state,
    cudaStream_t stream) {
    auto [device, device_id] = input.device();
    validate_2d_same_device(output, "output", input.dtype(), device, device_id);
    auto [weight_device, weight_device_id] = weight.device();
    auto [state_device, state_device_id] = conv_state.device();
    check<DeviceError>(
        weight_device == device && weight_device_id == device_id &&
            state_device == device && state_device_id == device_id,
        "conv tensors must be on the same device");
    check<ConfigurationError>(
        weight.dtype() == input.dtype() && conv_state.dtype() == input.dtype(),
        "conv tensors dtype mismatch");
    check<InvalidRequestError>(
        input.shape().size() == 2 && weight.shape().size() == 3 && conv_state.shape().size() == 2,
        "conv input/weight/state rank mismatch");
    const int64_t channels = input.shape()[1];
    const int64_t kernel_size = weight.shape()[2];
    check<InvalidRequestError>(
        weight.shape()[0] == channels && weight.shape()[1] == 1 &&
            conv_state.shape()[0] == channels && conv_state.shape()[1] == kernel_size &&
            output.shape() == input.shape(),
        "conv input/weight/state/output shape mismatch");

    if (input.dtype() == DType::BFloat16) {
        launch_depthwise_conv<__nv_bfloat16>(input, weight, conv_state, output, update_state, stream);
    } else if (input.dtype() == DType::Float16) {
        launch_depthwise_conv<half>(input, weight, conv_state, output, update_state, stream);
    } else if (input.dtype() == DType::Float32) {
        launch_depthwise_conv<float>(input, weight, conv_state, output, update_state, stream);
    } else {
        throw ConfigurationError("qwen3_5_depthwise_causal_conv1d supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_compute_g_beta_forward(
    const Tensor& a,
    const Tensor& b,
    const Tensor& a_log,
    const Tensor& dt_bias,
    Tensor& g,
    Tensor& beta,
    cudaStream_t stream) {
    auto [device, device_id] = a.device();
    validate_2d_same_device(b, "b", a.dtype(), device, device_id);
    validate_2d_same_device(g, "g", DType::Float32, device, device_id);
    validate_2d_same_device(beta, "beta", DType::Float32, device, device_id);
    auto [alog_device, alog_device_id] = a_log.device();
    auto [dt_device, dt_device_id] = dt_bias.device();
    check<DeviceError>(
        alog_device == device && alog_device_id == device_id &&
            dt_device == device && dt_device_id == device_id,
        "a_log/dt_bias must be on the same device");
    check<ConfigurationError>(a_log.dtype() == DType::Float32, "a_log must be Float32");
    check<ConfigurationError>(
        dt_bias.dtype() == DType::Float32 || dt_bias.dtype() == DType::Float16 ||
            dt_bias.dtype() == DType::BFloat16,
        "dt_bias must be Float32/Float16/BFloat16");
    check<InvalidRequestError>(
        a.shape() == b.shape() && a.shape() == g.shape() && a.shape() == beta.shape(),
        "a/b/g/beta shape mismatch");
    check<InvalidRequestError>(
        a_log.shape().size() == 1 && dt_bias.shape().size() == 1 &&
            a_log.shape()[0] == a.shape()[1] && dt_bias.shape()[0] == a.shape()[1],
        "a_log and dt_bias must be [num_heads]");

    if (a.dtype() == DType::BFloat16) {
        if (dt_bias.dtype() == DType::Float32) {
            launch_compute_g_beta<__nv_bfloat16, float>(a, b, a_log, dt_bias, g, beta, stream);
        } else if (dt_bias.dtype() == DType::Float16) {
            launch_compute_g_beta<__nv_bfloat16, half>(a, b, a_log, dt_bias, g, beta, stream);
        } else {
            launch_compute_g_beta<__nv_bfloat16, __nv_bfloat16>(a, b, a_log, dt_bias, g, beta, stream);
        }
    } else if (a.dtype() == DType::Float16) {
        if (dt_bias.dtype() == DType::Float32) {
            launch_compute_g_beta<half, float>(a, b, a_log, dt_bias, g, beta, stream);
        } else if (dt_bias.dtype() == DType::BFloat16) {
            launch_compute_g_beta<half, __nv_bfloat16>(a, b, a_log, dt_bias, g, beta, stream);
        } else {
            launch_compute_g_beta<half, half>(a, b, a_log, dt_bias, g, beta, stream);
        }
    } else if (a.dtype() == DType::Float32) {
        if (dt_bias.dtype() == DType::Float16) {
            launch_compute_g_beta<float, half>(a, b, a_log, dt_bias, g, beta, stream);
        } else if (dt_bias.dtype() == DType::BFloat16) {
            launch_compute_g_beta<float, __nv_bfloat16>(a, b, a_log, dt_bias, g, beta, stream);
        } else {
            launch_compute_g_beta<float, float>(a, b, a_log, dt_bias, g, beta, stream);
        }
    } else {
        throw ConfigurationError("qwen3_5_compute_g_beta supports Float32/Float16/BFloat16 a/b");
    }
}

void qwen3_5_compute_decay_beta_forward(
    const Tensor& a,
    const Tensor& b,
    const Tensor& a_log,
    const Tensor& dt_bias,
    Tensor& decay,
    Tensor& beta,
    cudaStream_t stream) {
    auto [device, device_id] = a.device();
    validate_2d_same_device(b, "b", a.dtype(), device, device_id);
    validate_2d_same_device(decay, "decay", DType::Float32, device, device_id);
    validate_2d_same_device(beta, "beta", DType::Float32, device, device_id);
    auto [alog_device, alog_device_id] = a_log.device();
    auto [dt_device, dt_device_id] = dt_bias.device();
    check<DeviceError>(
        alog_device == device && alog_device_id == device_id &&
            dt_device == device && dt_device_id == device_id,
        "a_log/dt_bias must be on the same device");
    check<ConfigurationError>(a_log.dtype() == DType::Float32, "a_log must be Float32");
    check<ConfigurationError>(
        dt_bias.dtype() == DType::Float32 || dt_bias.dtype() == DType::Float16 ||
            dt_bias.dtype() == DType::BFloat16,
        "dt_bias must be Float32/Float16/BFloat16");
    check<InvalidRequestError>(
        a.shape() == b.shape() && a.shape() == decay.shape() && a.shape() == beta.shape(),
        "a/b/decay/beta shape mismatch");
    check<InvalidRequestError>(
        a_log.shape().size() == 1 && dt_bias.shape().size() == 1 &&
            a_log.shape()[0] == a.shape()[1] && dt_bias.shape()[0] == a.shape()[1],
        "a_log and dt_bias must be [num_heads]");

    if (a.dtype() == DType::BFloat16) {
        if (dt_bias.dtype() == DType::Float32) {
            launch_compute_decay_beta<__nv_bfloat16, float>(a, b, a_log, dt_bias, decay, beta, stream);
        } else if (dt_bias.dtype() == DType::Float16) {
            launch_compute_decay_beta<__nv_bfloat16, half>(a, b, a_log, dt_bias, decay, beta, stream);
        } else {
            launch_compute_decay_beta<__nv_bfloat16, __nv_bfloat16>(a, b, a_log, dt_bias, decay, beta, stream);
        }
    } else if (a.dtype() == DType::Float16) {
        if (dt_bias.dtype() == DType::Float32) {
            launch_compute_decay_beta<half, float>(a, b, a_log, dt_bias, decay, beta, stream);
        } else if (dt_bias.dtype() == DType::BFloat16) {
            launch_compute_decay_beta<half, __nv_bfloat16>(a, b, a_log, dt_bias, decay, beta, stream);
        } else {
            launch_compute_decay_beta<half, half>(a, b, a_log, dt_bias, decay, beta, stream);
        }
    } else if (a.dtype() == DType::Float32) {
        if (dt_bias.dtype() == DType::Float16) {
            launch_compute_decay_beta<float, half>(a, b, a_log, dt_bias, decay, beta, stream);
        } else if (dt_bias.dtype() == DType::BFloat16) {
            launch_compute_decay_beta<float, __nv_bfloat16>(a, b, a_log, dt_bias, decay, beta, stream);
        } else {
            launch_compute_decay_beta<float, float>(a, b, a_log, dt_bias, decay, beta, stream);
        }
    } else {
        throw ConfigurationError("qwen3_5_compute_decay_beta supports Float32/Float16/BFloat16 a/b");
    }
}

void qwen3_5_gated_delta_recurrent_step(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& g,
    const Tensor& beta,
    Tensor& recurrent_state,
    Tensor& output,
    cudaStream_t stream) {
    auto [device, device_id] = query.device();
    check<ConfigurationError>(query.dtype() == DType::Float32, "gated_delta_step v1 expects Float32 query");
    validate_2d_same_device(key, "key", DType::Float32, device, device_id);
    validate_2d_same_device(value, "value", DType::Float32, device, device_id);
    validate_2d_same_device(output, "output", DType::Float32, device, device_id);
    check<InvalidRequestError>(g.shape().size() == 1 && beta.shape().size() == 1,
                               "g and beta must be 1D [num_heads]");
    auto [g_device, g_device_id] = g.device();
    auto [beta_device, beta_device_id] = beta.device();
    check<DeviceError>(g_device == device && g_device_id == device_id &&
                           beta_device == device && beta_device_id == device_id,
                       "g and beta tensors must be on the same device");
    check<ConfigurationError>(g.dtype() == DType::Float32 && beta.dtype() == DType::Float32,
                              "g and beta must be Float32");
    check<ConfigurationError>(recurrent_state.dtype() == DType::Float32,
                              "recurrent_state must be Float32");
    auto [state_device, state_device_id] = recurrent_state.device();
    check<DeviceError>(state_device == device && state_device_id == device_id,
                       "recurrent_state tensor must be on the same device");

    const auto& q_shape = query.shape();
    const auto& v_shape = value.shape();
    const int32_t num_heads = static_cast<int32_t>(q_shape[0]);
    const int32_t key_dim = static_cast<int32_t>(q_shape[1]);
    const int32_t value_dim = static_cast<int32_t>(v_shape[1]);
    check<InvalidRequestError>(key.shape() == q_shape, "key shape must match query shape");
    check<InvalidRequestError>(value.shape().size() == 2 && value.shape()[0] == num_heads,
                               "value must be [num_heads, value_dim]");
    check<InvalidRequestError>(g.shape()[0] == num_heads && beta.shape()[0] == num_heads,
                               "g/beta head count mismatch");
    check<InvalidRequestError>(
        recurrent_state.shape().size() == 3 && recurrent_state.shape()[0] == num_heads &&
            recurrent_state.shape()[1] == key_dim && recurrent_state.shape()[2] == value_dim,
        "recurrent_state must be [num_heads, key_dim, value_dim]");
    check<InvalidRequestError>(output.shape() == v_shape, "output shape must match value shape");

    gated_delta_recurrent_step_kernel<<<num_heads, 1, 0, stream>>>(
        static_cast<const float*>(query.data_ptr()),
        static_cast<const float*>(key.data_ptr()),
        static_cast<const float*>(value.data_ptr()),
        static_cast<const float*>(g.data_ptr()),
        static_cast<const float*>(beta.data_ptr()),
        static_cast<float*>(recurrent_state.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        num_heads,
        key_dim,
        value_dim);
    CUDA_CHECK_THROW(cudaGetLastError(), "qwen3_5 gated delta recurrent step kernel launch failed");
}

void qwen3_5_gated_delta_sequence_forward(
    const Tensor& mixed_qkv,
    const Tensor& g,
    const Tensor& beta,
    Tensor& recurrent_state,
    Tensor& output,
    cudaStream_t stream) {
    auto [device, device_id] = mixed_qkv.device();
    validate_2d_same_device(g, "g", DType::Float32, device, device_id);
    validate_2d_same_device(beta, "beta", DType::Float32, device, device_id);
    validate_2d_same_device(output, "output", mixed_qkv.dtype(), device, device_id);
    auto [state_device, state_device_id] = recurrent_state.device();
    check<DeviceError>(
        state_device == device && state_device_id == device_id,
        "recurrent_state must be on the same device");
    check<ConfigurationError>(recurrent_state.dtype() == DType::Float32, "recurrent_state must be Float32");
    check<InvalidRequestError>(
        recurrent_state.shape().size() == 3,
        "recurrent_state must be [num_heads, key_dim, value_dim]");
    const int64_t num_heads = recurrent_state.shape()[0];
    const int64_t key_dim = recurrent_state.shape()[1];
    const int64_t value_dim = recurrent_state.shape()[2];
    const int64_t conv_dim = num_heads * (2 * key_dim + value_dim);
    check<InvalidRequestError>(
        mixed_qkv.shape().size() == 2 && mixed_qkv.shape()[1] == conv_dim,
        "mixed_qkv must be [seq_len, num_heads * (2 * key_dim + value_dim)]");
    check<InvalidRequestError>(
        g.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads} &&
            beta.shape() == g.shape(),
        "g/beta must be [seq_len, num_heads]");
    check<InvalidRequestError>(
        output.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads * value_dim},
        "output must be [seq_len, num_heads * value_dim]");

    if (mixed_qkv.dtype() == DType::BFloat16) {
        launch_gated_delta_sequence<__nv_bfloat16>(mixed_qkv, g, beta, recurrent_state, output, stream);
    } else if (mixed_qkv.dtype() == DType::Float16) {
        launch_gated_delta_sequence<half>(mixed_qkv, g, beta, recurrent_state, output, stream);
    } else if (mixed_qkv.dtype() == DType::Float32) {
        launch_gated_delta_sequence<float>(mixed_qkv, g, beta, recurrent_state, output, stream);
    } else {
        throw ConfigurationError("qwen3_5_gated_delta_sequence supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_gated_delta_sequence_from_ab_forward(
    const Tensor& mixed_qkv,
    const Tensor& a,
    const Tensor& b,
    const Tensor& a_log,
    const Tensor& dt_bias,
    Tensor& recurrent_state,
    Tensor& output,
    cudaStream_t stream) {
    auto [device, device_id] = mixed_qkv.device();
    validate_2d_same_device(a, "a", mixed_qkv.dtype(), device, device_id);
    validate_2d_same_device(b, "b", mixed_qkv.dtype(), device, device_id);
    validate_2d_same_device(output, "output", mixed_qkv.dtype(), device, device_id);
    auto [alog_device, alog_device_id] = a_log.device();
    auto [dt_device, dt_device_id] = dt_bias.device();
    auto [state_device, state_device_id] = recurrent_state.device();
    check<DeviceError>(
        alog_device == device && alog_device_id == device_id &&
            dt_device == device && dt_device_id == device_id &&
            state_device == device && state_device_id == device_id,
        "a_log/dt_bias/recurrent_state must be on the same device");
    check<ConfigurationError>(a_log.dtype() == DType::Float32, "a_log must be Float32");
    check<ConfigurationError>(
        dt_bias.dtype() == DType::Float32 || dt_bias.dtype() == DType::Float16 ||
            dt_bias.dtype() == DType::BFloat16,
        "dt_bias must be Float32/Float16/BFloat16");
    check<ConfigurationError>(recurrent_state.dtype() == DType::Float32, "recurrent_state must be Float32");
    check<InvalidRequestError>(
        recurrent_state.shape().size() == 3,
        "recurrent_state must be [num_heads, key_dim, value_dim]");
    const int64_t num_heads = recurrent_state.shape()[0];
    const int64_t key_dim = recurrent_state.shape()[1];
    const int64_t value_dim = recurrent_state.shape()[2];
    const int64_t conv_dim = num_heads * (2 * key_dim + value_dim);
    check<InvalidRequestError>(
        mixed_qkv.shape().size() == 2 && mixed_qkv.shape()[1] == conv_dim,
        "mixed_qkv must be [seq_len, num_heads * (2 * key_dim + value_dim)]");
    check<InvalidRequestError>(
        a.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads} &&
            b.shape() == a.shape(),
        "a/b must be [seq_len, num_heads]");
    check<InvalidRequestError>(
        a_log.shape().size() == 1 && dt_bias.shape().size() == 1 &&
            a_log.shape()[0] == num_heads && dt_bias.shape()[0] == num_heads,
        "a_log and dt_bias must be [num_heads]");
    check<InvalidRequestError>(
        output.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads * value_dim},
        "output must be [seq_len, num_heads * value_dim]");

    if (mixed_qkv.dtype() == DType::BFloat16) {
        if (dt_bias.dtype() == DType::Float32) {
            launch_gated_delta_sequence_from_ab<__nv_bfloat16, float>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        } else if (dt_bias.dtype() == DType::Float16) {
            launch_gated_delta_sequence_from_ab<__nv_bfloat16, half>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        } else {
            launch_gated_delta_sequence_from_ab<__nv_bfloat16, __nv_bfloat16>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        }
    } else if (mixed_qkv.dtype() == DType::Float16) {
        if (dt_bias.dtype() == DType::Float32) {
            launch_gated_delta_sequence_from_ab<half, float>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        } else if (dt_bias.dtype() == DType::BFloat16) {
            launch_gated_delta_sequence_from_ab<half, __nv_bfloat16>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        } else {
            launch_gated_delta_sequence_from_ab<half, half>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        }
    } else if (mixed_qkv.dtype() == DType::Float32) {
        if (dt_bias.dtype() == DType::Float16) {
            launch_gated_delta_sequence_from_ab<float, half>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        } else if (dt_bias.dtype() == DType::BFloat16) {
            launch_gated_delta_sequence_from_ab<float, __nv_bfloat16>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        } else {
            launch_gated_delta_sequence_from_ab<float, float>(
                mixed_qkv, a, b, a_log, dt_bias, recurrent_state, output, stream);
        }
    } else {
        throw ConfigurationError("qwen3_5_gated_delta_sequence_from_ab supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_precompute_gated_delta_qk_forward(
    const Tensor& mixed_qkv,
    Tensor& query_norm,
    Tensor& key_norm,
    cudaStream_t stream) {
    auto [device, device_id] = mixed_qkv.device();
    validate_3d_same_device(query_norm, "query_norm", DType::Float32, device, device_id);
    validate_3d_same_device(key_norm, "key_norm", DType::Float32, device, device_id);
    check<InvalidRequestError>(
        mixed_qkv.shape().size() == 2,
        "mixed_qkv must be [seq_len, num_heads * (2 * key_dim + value_dim)]");
    check<InvalidRequestError>(
        query_norm.shape() == key_norm.shape(),
        "query_norm/key_norm shape mismatch");
    check<InvalidRequestError>(
        query_norm.shape()[0] == mixed_qkv.shape()[0],
        "query_norm/key_norm seq_len must match mixed_qkv");
    const int64_t num_heads = query_norm.shape()[1];
    const int64_t key_dim = query_norm.shape()[2];
    check<InvalidRequestError>(
        num_heads > 0 && key_dim > 0 && mixed_qkv.shape()[1] > num_heads * 2 * key_dim &&
            mixed_qkv.shape()[1] % num_heads == 0,
        "mixed_qkv last dim must contain q/k/value for query_norm shape");

    if (mixed_qkv.dtype() == DType::BFloat16) {
        launch_precompute_gated_delta_qk<__nv_bfloat16>(mixed_qkv, query_norm, key_norm, stream);
    } else if (mixed_qkv.dtype() == DType::Float16) {
        launch_precompute_gated_delta_qk<half>(mixed_qkv, query_norm, key_norm, stream);
    } else if (mixed_qkv.dtype() == DType::Float32) {
        launch_precompute_gated_delta_qk<float>(mixed_qkv, query_norm, key_norm, stream);
    } else {
        throw ConfigurationError("qwen3_5_precompute_gated_delta_qk supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_gated_delta_sequence_precomputed_forward(
    const Tensor& mixed_qkv,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& query_norm,
    const Tensor& key_norm,
    Tensor& recurrent_state,
    Tensor& output,
    cudaStream_t stream) {
    auto [device, device_id] = mixed_qkv.device();
    validate_2d_same_device(g, "g", DType::Float32, device, device_id);
    validate_2d_same_device(beta, "beta", DType::Float32, device, device_id);
    validate_3d_same_device(query_norm, "query_norm", DType::Float32, device, device_id);
    validate_3d_same_device(key_norm, "key_norm", DType::Float32, device, device_id);
    validate_2d_same_device(output, "output", mixed_qkv.dtype(), device, device_id);
    auto [state_device, state_device_id] = recurrent_state.device();
    check<DeviceError>(
        state_device == device && state_device_id == device_id,
        "recurrent_state must be on the same device");
    check<ConfigurationError>(recurrent_state.dtype() == DType::Float32, "recurrent_state must be Float32");
    check<InvalidRequestError>(
        recurrent_state.shape().size() == 3,
        "recurrent_state must be [num_heads, key_dim, value_dim]");
    const int64_t num_heads = recurrent_state.shape()[0];
    const int64_t key_dim = recurrent_state.shape()[1];
    const int64_t value_dim = recurrent_state.shape()[2];
    const int64_t conv_dim = num_heads * (2 * key_dim + value_dim);
    check<InvalidRequestError>(
        mixed_qkv.shape().size() == 2 && mixed_qkv.shape()[1] == conv_dim,
        "mixed_qkv must be [seq_len, num_heads * (2 * key_dim + value_dim)]");
    check<InvalidRequestError>(
        g.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads} &&
            beta.shape() == g.shape(),
        "g/beta must be [seq_len, num_heads]");
    check<InvalidRequestError>(
        query_norm.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads, key_dim} &&
            key_norm.shape() == query_norm.shape(),
        "query_norm/key_norm must be [seq_len, num_heads, key_dim]");
    check<InvalidRequestError>(
        output.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads * value_dim},
        "output must be [seq_len, num_heads * value_dim]");

    if (mixed_qkv.dtype() == DType::BFloat16) {
        launch_gated_delta_sequence_precomputed<__nv_bfloat16>(
            mixed_qkv, g, beta, query_norm, key_norm, recurrent_state, output, stream);
    } else if (mixed_qkv.dtype() == DType::Float16) {
        launch_gated_delta_sequence_precomputed<half>(
            mixed_qkv, g, beta, query_norm, key_norm, recurrent_state, output, stream);
    } else if (mixed_qkv.dtype() == DType::Float32) {
        launch_gated_delta_sequence_precomputed<float>(
            mixed_qkv, g, beta, query_norm, key_norm, recurrent_state, output, stream);
    } else {
        throw ConfigurationError("qwen3_5_gated_delta_sequence_precomputed supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_gated_delta_sequence_precomputed_decay_forward(
    const Tensor& mixed_qkv,
    const Tensor& decay,
    const Tensor& beta,
    const Tensor& query_norm,
    const Tensor& key_norm,
    Tensor& recurrent_state,
    Tensor& output,
    cudaStream_t stream) {
    auto [device, device_id] = mixed_qkv.device();
    validate_2d_same_device(decay, "decay", DType::Float32, device, device_id);
    validate_2d_same_device(beta, "beta", DType::Float32, device, device_id);
    validate_3d_same_device(query_norm, "query_norm", DType::Float32, device, device_id);
    validate_3d_same_device(key_norm, "key_norm", DType::Float32, device, device_id);
    validate_2d_same_device(output, "output", mixed_qkv.dtype(), device, device_id);
    auto [state_device, state_device_id] = recurrent_state.device();
    check<DeviceError>(
        state_device == device && state_device_id == device_id,
        "recurrent_state must be on the same device");
    check<ConfigurationError>(recurrent_state.dtype() == DType::Float32, "recurrent_state must be Float32");
    check<InvalidRequestError>(
        recurrent_state.shape().size() == 3,
        "recurrent_state must be [num_heads, key_dim, value_dim]");
    const int64_t num_heads = recurrent_state.shape()[0];
    const int64_t key_dim = recurrent_state.shape()[1];
    const int64_t value_dim = recurrent_state.shape()[2];
    const int64_t conv_dim = num_heads * (2 * key_dim + value_dim);
    check<InvalidRequestError>(
        mixed_qkv.shape().size() == 2 && mixed_qkv.shape()[1] == conv_dim,
        "mixed_qkv must be [seq_len, num_heads * (2 * key_dim + value_dim)]");
    check<InvalidRequestError>(
        decay.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads} &&
            beta.shape() == decay.shape(),
        "decay/beta must be [seq_len, num_heads]");
    check<InvalidRequestError>(
        query_norm.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads, key_dim} &&
            key_norm.shape() == query_norm.shape(),
        "query_norm/key_norm must be [seq_len, num_heads, key_dim]");
    check<InvalidRequestError>(
        output.shape() == std::vector<int64_t>{mixed_qkv.shape()[0], num_heads * value_dim},
        "output must be [seq_len, num_heads * value_dim]");

    if (mixed_qkv.dtype() == DType::BFloat16) {
        launch_gated_delta_sequence_precomputed_decay<__nv_bfloat16>(
            mixed_qkv, decay, beta, query_norm, key_norm, recurrent_state, output, stream);
    } else if (mixed_qkv.dtype() == DType::Float16) {
        launch_gated_delta_sequence_precomputed_decay<half>(
            mixed_qkv, decay, beta, query_norm, key_norm, recurrent_state, output, stream);
    } else if (mixed_qkv.dtype() == DType::Float32) {
        launch_gated_delta_sequence_precomputed_decay<float>(
            mixed_qkv, decay, beta, query_norm, key_norm, recurrent_state, output, stream);
    } else {
        throw ConfigurationError("qwen3_5_gated_delta_sequence_precomputed_decay supports Float32/Float16/BFloat16");
    }
}

void qwen3_5_apply_partial_interleaved_mrope(
    Tensor& query,
    Tensor& key,
    const Tensor& position_ids,
    const std::vector<int32_t>& mrope_section,
    float rope_theta,
    int32_t rotary_dim,
    cudaStream_t stream) {
    check<InvalidRequestError>(mrope_section.size() == 3, "mrope_section must have 3 entries");
    check<InvalidRequestError>(rotary_dim > 0 && (rotary_dim % 2) == 0,
                               "rotary_dim must be positive and even");
    auto [device, device_id] = query.device();
    auto [key_device, key_device_id] = key.device();
    auto [pos_device, pos_device_id] = position_ids.device();
    check<DeviceError>(key_device == device && key_device_id == device_id &&
                           pos_device == device && pos_device_id == device_id,
                       "query/key/position_ids must be on the same device");
    check<ConfigurationError>(query.dtype() == key.dtype(), "query and key dtype must match");
    check<ConfigurationError>(position_ids.dtype() == DType::Int32, "position_ids must be Int32");
    check<InvalidRequestError>(query.shape().size() == 3 && key.shape().size() == 3,
                               "query and key must be [seq_len, heads, head_dim]");
    check<InvalidRequestError>(query.shape()[0] == key.shape()[0] && query.shape()[2] == key.shape()[2],
                               "query/key seq_len and head_dim must match");
    check<InvalidRequestError>(rotary_dim <= query.shape()[2],
                               "rotary_dim must be <= head_dim");
    check<InvalidRequestError>(
        position_ids.shape().size() == 2 && position_ids.shape()[0] == 3 &&
            position_ids.shape()[1] == query.shape()[0],
        "position_ids must be [3, seq_len]");

    if (query.dtype() == DType::BFloat16) {
        launch_partial_mrope<__nv_bfloat16>(query, position_ids, mrope_section, rope_theta, rotary_dim, stream);
        launch_partial_mrope<__nv_bfloat16>(key, position_ids, mrope_section, rope_theta, rotary_dim, stream);
    } else if (query.dtype() == DType::Float16) {
        launch_partial_mrope<half>(query, position_ids, mrope_section, rope_theta, rotary_dim, stream);
        launch_partial_mrope<half>(key, position_ids, mrope_section, rope_theta, rotary_dim, stream);
    } else if (query.dtype() == DType::Float32) {
        launch_partial_mrope<float>(query, position_ids, mrope_section, rope_theta, rotary_dim, stream);
        launch_partial_mrope<float>(key, position_ids, mrope_section, rope_theta, rotary_dim, stream);
    } else {
        throw ConfigurationError("partial M-RoPE supports Float32/Float16/BFloat16");
    }
}

} // namespace edge_fm
