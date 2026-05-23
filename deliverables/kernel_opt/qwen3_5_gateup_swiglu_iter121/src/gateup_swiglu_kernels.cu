#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

template <int kWarpsPerBlock>
__global__ void gateup_swiglu_scalar_kernel(const __nv_bfloat16* __restrict__ hidden,
                                            const __nv_bfloat16* __restrict__ weight,
                                            __nv_bfloat16* __restrict__ output,
                                            int32_t hidden_size,
                                            int32_t intermediate_size) {
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    const int32_t out_id = static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (out_id >= intermediate_size) {
        return;
    }

    const __nv_bfloat16* up_row =
        weight + static_cast<size_t>(out_id) * static_cast<size_t>(hidden_size);
    const __nv_bfloat16* gate_row =
        weight + static_cast<size_t>(out_id + intermediate_size) * static_cast<size_t>(hidden_size);

    float up = 0.0f;
    float gate = 0.0f;
    for (int32_t k = lane; k < hidden_size; k += 2 * kWarpSize) {
        const float h0 = __bfloat162float(hidden[k]);
        up += h0 * __bfloat162float(up_row[k]);
        gate += h0 * __bfloat162float(gate_row[k]);
        const int32_t k_next = k + kWarpSize;
        if (k_next < hidden_size) {
            const float h1 = __bfloat162float(hidden[k_next]);
            up += h1 * __bfloat162float(up_row[k_next]);
            gate += h1 * __bfloat162float(gate_row[k_next]);
        }
    }

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        up += __shfl_down_sync(0xffffffffu, up, offset);
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
    }
    if (lane == 0) {
        output[out_id] = __float2bfloat16(silu(gate) * up);
    }
}

template <int kWarpsPerBlock>
__global__ void gateup_swiglu_bf162_kernel(const __nv_bfloat16* __restrict__ hidden,
                                           const __nv_bfloat16* __restrict__ weight,
                                           __nv_bfloat16* __restrict__ output,
                                           int32_t hidden_size,
                                           int32_t intermediate_size) {
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
    const int32_t warp = static_cast<int32_t>(threadIdx.x) / kWarpSize;
    const int32_t out_id = static_cast<int32_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (out_id >= intermediate_size) {
        return;
    }

    const int32_t pair_count = hidden_size / 2;
    const auto* hidden2 = reinterpret_cast<const __nv_bfloat162*>(hidden);
    const __nv_bfloat16* up_row =
        weight + static_cast<size_t>(out_id) * static_cast<size_t>(hidden_size);
    const __nv_bfloat16* gate_row =
        weight + static_cast<size_t>(out_id + intermediate_size) * static_cast<size_t>(hidden_size);
    const auto* up2 = reinterpret_cast<const __nv_bfloat162*>(up_row);
    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate_row);

    float up = 0.0f;
    float gate = 0.0f;
    for (int32_t p = lane; p < pair_count; p += kWarpSize) {
        const float2 h = __bfloat1622float2(hidden2[p]);
        const float2 u = __bfloat1622float2(up2[p]);
        const float2 g = __bfloat1622float2(gate2[p]);
        up += h.x * u.x + h.y * u.y;
        gate += h.x * g.x + h.y * g.y;
    }

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        up += __shfl_down_sync(0xffffffffu, up, offset);
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
    }
    if (lane == 0) {
        output[out_id] = __float2bfloat16(silu(gate) * up);
    }
}

template <int kWarpsPerBlock>
void launch_gateup(const __nv_bfloat16* hidden,
                   const __nv_bfloat16* weight,
                   __nv_bfloat16* output,
                   int32_t hidden_size,
                   int32_t intermediate_size,
                   bool vec2,
                   cudaStream_t stream) {
    const int32_t blocks = (intermediate_size + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 grid(blocks);
    const dim3 block(kWarpSize * kWarpsPerBlock);
    if (vec2) {
        gateup_swiglu_bf162_kernel<kWarpsPerBlock><<<grid, block, 0, stream>>>(
            hidden, weight, output, hidden_size, intermediate_size);
    } else {
        gateup_swiglu_scalar_kernel<kWarpsPerBlock><<<grid, block, 0, stream>>>(
            hidden, weight, output, hidden_size, intermediate_size);
    }
}

} // namespace

torch::Tensor gateup_swiglu_bf16(torch::Tensor hidden,
                                 torch::Tensor weight,
                                 int64_t warps_per_block,
                                 bool vec2) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be BF16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be BF16");
    TORCH_CHECK(hidden.dim() == 1, "hidden shape must be [hidden_size]");
    TORCH_CHECK(weight.dim() == 2, "weight shape must be [2 * intermediate, hidden_size]");
    TORCH_CHECK(weight.size(1) == hidden.numel(), "hidden dimension mismatch");
    TORCH_CHECK(weight.size(0) % 2 == 0, "weight rows must be [up, gate]");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(!vec2 || (hidden.numel() % 2 == 0), "vec2 path requires an even hidden size");

    const int32_t hidden_size = static_cast<int32_t>(hidden.numel());
    const int32_t intermediate_size = static_cast<int32_t>(weight.size(0) / 2);
    auto output = torch::empty({intermediate_size}, hidden.options());

    const auto* hidden_ptr =
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>());
    const auto* weight_ptr =
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>());
    auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    switch (warps_per_block) {
        case 8:
            launch_gateup<8>(hidden_ptr, weight_ptr, output_ptr, hidden_size, intermediate_size, vec2, stream);
            break;
        case 16:
            launch_gateup<16>(hidden_ptr, weight_ptr, output_ptr, hidden_size, intermediate_size, vec2, stream);
            break;
        case 24:
            launch_gateup<24>(hidden_ptr, weight_ptr, output_ptr, hidden_size, intermediate_size, vec2, stream);
            break;
        case 32:
            launch_gateup<32>(hidden_ptr, weight_ptr, output_ptr, hidden_size, intermediate_size, vec2, stream);
            break;
        default:
            TORCH_CHECK(false, "warps_per_block must be 8, 16, 24, or 32");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gateup_swiglu_bf16", &gateup_swiglu_bf16, "Qwen3.5 GateUp+SwiGLU BF16 standalone");
}
