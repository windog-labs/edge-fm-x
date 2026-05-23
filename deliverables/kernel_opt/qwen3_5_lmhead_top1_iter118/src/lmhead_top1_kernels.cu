#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstdint>

namespace {

constexpr int kWarpSize = 32;

struct Candidate {
    float value;
    int32_t index;
};

__device__ __forceinline__ Candidate empty_candidate() {
    return Candidate{-FLT_MAX, INT_MAX};
}

__device__ __forceinline__ Candidate better(Candidate a, Candidate b) {
    if (b.value > a.value || (b.value == a.value && b.index < a.index)) {
        return b;
    }
    return a;
}

__device__ __forceinline__ Candidate warp_reduce_best(Candidate local) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        Candidate other{
            __shfl_down_sync(0xffffffffu, local.value, offset),
            __shfl_down_sync(0xffffffffu, local.index, offset),
        };
        local = better(local, other);
    }
    return local;
}

template <int kWarpsPerBlock>
__global__ void stage1_scalar_kernel(const __nv_bfloat16* __restrict__ hidden,
                                     const __nv_bfloat16* __restrict__ weight,
                                     int32_t hidden_size,
                                     int32_t vocab_size,
                                     float* __restrict__ tmp_values,
                                     int32_t* __restrict__ tmp_indices) {
    __shared__ float warp_values[kWarpsPerBlock];
    __shared__ int32_t warp_indices[kWarpsPerBlock];

    const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    const int32_t vocab_id = static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + warp;

    float sum = 0.0f;
    if (vocab_id < vocab_size) {
        const __nv_bfloat16* row =
            weight + static_cast<size_t>(vocab_id) * static_cast<size_t>(hidden_size);
        for (int32_t k = lane; k < hidden_size; k += 2 * kWarpSize) {
            sum += __bfloat162float(hidden[k]) * __bfloat162float(row[k]);
            const int32_t next_k = k + kWarpSize;
            if (next_k < hidden_size) {
                sum += __bfloat162float(hidden[next_k]) * __bfloat162float(row[next_k]);
            }
        }
    }

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        warp_values[warp] = (vocab_id < vocab_size) ? sum : -FLT_MAX;
        warp_indices[warp] = (vocab_id < vocab_size) ? vocab_id : INT_MAX;
    }
    __syncthreads();

    if (warp == 0) {
        Candidate local = empty_candidate();
        if (lane < kWarpsPerBlock) {
            local = Candidate{warp_values[lane], warp_indices[lane]};
        }
        Candidate best = warp_reduce_best(local);
        if (lane == 0) {
            tmp_values[blockIdx.x] = best.value;
            tmp_indices[blockIdx.x] = best.index;
        }
    }
}

template <int kWarpsPerBlock>
__global__ void stage1_bf162_kernel(const __nv_bfloat16* __restrict__ hidden,
                                    const __nv_bfloat16* __restrict__ weight,
                                    int32_t hidden_size,
                                    int32_t vocab_size,
                                    float* __restrict__ tmp_values,
                                    int32_t* __restrict__ tmp_indices) {
    __shared__ float warp_values[kWarpsPerBlock];
    __shared__ int32_t warp_indices[kWarpsPerBlock];

    const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    const int32_t vocab_id = static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const int32_t pair_count = hidden_size / 2;
    const auto* hidden2 = reinterpret_cast<const __nv_bfloat162*>(hidden);

    float sum = 0.0f;
    if (vocab_id < vocab_size) {
        const __nv_bfloat16* row =
            weight + static_cast<size_t>(vocab_id) * static_cast<size_t>(hidden_size);
        const auto* row2 = reinterpret_cast<const __nv_bfloat162*>(row);
        for (int32_t p = lane; p < pair_count; p += kWarpSize) {
            const float2 h = __bfloat1622float2(hidden2[p]);
            const float2 w = __bfloat1622float2(row2[p]);
            sum += h.x * w.x + h.y * w.y;
        }
    }

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }

    if (lane == 0) {
        warp_values[warp] = (vocab_id < vocab_size) ? sum : -FLT_MAX;
        warp_indices[warp] = (vocab_id < vocab_size) ? vocab_id : INT_MAX;
    }
    __syncthreads();

    if (warp == 0) {
        Candidate local = empty_candidate();
        if (lane < kWarpsPerBlock) {
            local = Candidate{warp_values[lane], warp_indices[lane]};
        }
        Candidate best = warp_reduce_best(local);
        if (lane == 0) {
            tmp_values[blockIdx.x] = best.value;
            tmp_indices[blockIdx.x] = best.index;
        }
    }
}

__global__ void stage2_kernel(const float* __restrict__ tmp_values,
                              const int32_t* __restrict__ tmp_indices,
                              int32_t candidate_count,
                              int32_t* __restrict__ token_out) {
    __shared__ float shared_values[256];
    __shared__ int32_t shared_indices[256];

    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    Candidate local = empty_candidate();
    for (int32_t i = tid; i < candidate_count; i += static_cast<int32_t>(blockDim.x)) {
        local = better(local, Candidate{tmp_values[i], tmp_indices[i]});
    }

    shared_values[tid] = local.value;
    shared_indices[tid] = local.index;
    __syncthreads();

    for (int stride = static_cast<int32_t>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            Candidate best = better(
                Candidate{shared_values[tid], shared_indices[tid]},
                Candidate{shared_values[tid + stride], shared_indices[tid + stride]});
            shared_values[tid] = best.value;
            shared_indices[tid] = best.index;
        }
        __syncthreads();
    }

    if (tid == 0) {
        token_out[0] = shared_indices[0] == INT_MAX ? 0 : shared_indices[0];
    }
}

template <int kWarpsPerBlock>
void launch_stage1(const __nv_bfloat16* hidden,
                   const __nv_bfloat16* weight,
                   int32_t hidden_size,
                   int32_t vocab_size,
                   float* tmp_values,
                   int32_t* tmp_indices,
                   bool vec2,
                   cudaStream_t stream) {
    const int32_t blocks = (vocab_size + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 grid(blocks);
    const dim3 block(kWarpSize * kWarpsPerBlock);
    if (vec2) {
        stage1_bf162_kernel<kWarpsPerBlock><<<grid, block, 0, stream>>>(
            hidden, weight, hidden_size, vocab_size, tmp_values, tmp_indices);
    } else {
        stage1_scalar_kernel<kWarpsPerBlock><<<grid, block, 0, stream>>>(
            hidden, weight, hidden_size, vocab_size, tmp_values, tmp_indices);
    }
}

} // namespace

torch::Tensor top1_bf16(torch::Tensor hidden,
                        torch::Tensor weight,
                        int64_t warps_per_block,
                        bool vec2) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(hidden.dim() == 1, "hidden must have shape [hidden_size]");
    TORCH_CHECK(weight.dim() == 2, "weight must have shape [vocab_size, hidden_size]");
    TORCH_CHECK(weight.size(1) == hidden.numel(), "weight hidden dimension mismatch");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(!vec2 || (hidden.numel() % 2 == 0), "vec2 path requires an even hidden size");

    const int32_t hidden_size = static_cast<int32_t>(hidden.numel());
    const int32_t vocab_size = static_cast<int32_t>(weight.size(0));
    const int32_t candidate_count =
        static_cast<int32_t>((static_cast<int64_t>(vocab_size) + warps_per_block - 1) / warps_per_block);

    auto float_opts = torch::TensorOptions().device(hidden.device()).dtype(torch::kFloat32);
    auto int_opts = torch::TensorOptions().device(hidden.device()).dtype(torch::kInt32);
    torch::Tensor tmp_values = torch::empty({candidate_count}, float_opts);
    torch::Tensor tmp_indices = torch::empty({candidate_count}, int_opts);
    torch::Tensor token = torch::empty({1}, int_opts);

    const auto* hidden_ptr =
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>());
    const auto* weight_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
    auto* values_ptr = tmp_values.data_ptr<float>();
    auto* indices_ptr = tmp_indices.data_ptr<int32_t>();
    auto* token_ptr = token.data_ptr<int32_t>();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (warps_per_block) {
        case 16:
            launch_stage1<16>(hidden_ptr, weight_ptr, hidden_size, vocab_size,
                              values_ptr, indices_ptr, vec2, stream);
            break;
        case 24:
            launch_stage1<24>(hidden_ptr, weight_ptr, hidden_size, vocab_size,
                              values_ptr, indices_ptr, vec2, stream);
            break;
        case 32:
            launch_stage1<32>(hidden_ptr, weight_ptr, hidden_size, vocab_size,
                              values_ptr, indices_ptr, vec2, stream);
            break;
        default:
            TORCH_CHECK(false, "warps_per_block must be one of 16, 24, or 32");
    }
    stage2_kernel<<<1, 256, 0, stream>>>(values_ptr, indices_ptr, candidate_count, token_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return token;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("top1_bf16", &top1_bf16, "BF16 LMHead top1 standalone variants");
}
