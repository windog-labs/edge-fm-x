# CUDA-Agent Full SKILL Reference

You are a PyTorch and CUDA expert. Accelerate the given PyTorch Model by creating a high-performance CUDA C++ extension, targeting the best possible performance with a minimum requirement of 5% faster than torch.compile baseline.

## 1. CRITICAL RESTRICTIONS

### ⚠️ STRICTLY FORBIDDEN
- **NO torch operators in C++**: NEVER use `torch::*` or `torch::nn::functional::*` in binding.cpp or .cu files
- **NO torch operations in model_new.py**: Only tensor creation and your custom ops allowed
- **NO third-party libraries**: Except cuBLAS (GEMM only) and cuDNN (Conv only)
- **NO modifications to utils/ directory**
- **NO modifications to binding.cpp or binding_registry.h**: These are fixed infrastructure

### ✅ ALLOWED ONLY
- **C++**: Raw CUDA kernels (for custom ops), cuBLAS (for GEMM), cuDNN (MANDATORY for Conv/ConvTranspose)
- **Python**: torch.tensor creation, custom extension ops, tensor properties (.shape, .device)
- **Memory**: torch::empty_like for allocation only
- **Focus**: Implement kernels in `kernels/` directory only

## 2. WORKSPACE STRUCTURE

```
.
├── binding_registry.h    # Do NOT modify - registration system
├── binding.cpp           # Do NOT modify - main module binding
├── kernels/              # YOUR WORK: Implement all kernels here
├── utils/                # DO NOT modify - Compilation, verification and profiling tools 
├── model.py              # Do NOT modify - Original PyTorch model
└── model_new.py          # YOUR WORK: Your optimized model using custom ops.
```

### File Types and Usage
- **`.cu` files**: CUDA kernels with `__global__` functions (custom implementations)
- **`.cpp` files**: cuDNN/cuBLAS API calls (NO custom kernels)
- **`_binding.cpp` files**: PyTorch tensor handling and Python bindings

## 3. UNIFIED WORKFLOW

### Step 1: Implementation

Create paired files in `kernels/`:

**kernels/my_kernel.cu** (Pure CUDA implementation):
```cuda
#include <cuda_runtime.h>

template<int BLOCK_SIZE, int TILE_SIZE>
__global__ void my_kernel_impl(float* output, const float* input, int size) {
    extern __shared__ float smem[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < size; i += stride) {
        output[i] = /* computation */;
    }
}

extern "C" void my_kernel_launcher(
    float* output, const float* input, int size,
    int config, cudaStream_t stream
) {
    int blocks = (size + 255) / 256;
    int shared_mem_size = 0;
    switch(config) {
        case 0: shared_mem_size = 256 * sizeof(float);
            my_kernel_impl<256, 16><<<blocks, 256, shared_mem_size, stream>>>(output, input, size); break;
        case 1: shared_mem_size = 128 * sizeof(float);
            my_kernel_impl<128, 32><<<blocks, 128, shared_mem_size, stream>>>(output, input, size); break;
        default: my_kernel_impl<256, 16><<<blocks, 256, 0, stream>>>(output, input, size);
    }
}
```

**kernels/my_kernel_binding.cpp** (PyTorch binding):
```cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void my_kernel_launcher(float* output, const float* input, int size, int config, cudaStream_t stream);

torch::Tensor my_kernel_forward(torch::Tensor input, int config = 0) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dtype() == torch::kFloat32);
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    my_kernel_launcher(output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), config, stream);
    return output;
}

void register_my_kernel(pybind11::module& m) {
    m.def("my_kernel_forward", &my_kernel_forward, py::arg("input"), py::arg("config") = 0);
}
REGISTER_BINDING(my_kernel, register_my_kernel);
```

**model_new.py**:
```python
import torch, torch.nn as nn, cuda_extension

class ModelNew(nn.Module):
    def __init__(self, ...):  # MUST match Model signature
        super().__init__()
        self.weight = nn.Parameter(torch.randn(...))
    def forward(self, x):
        x = cuda_extension.my_kernel_forward(x, config=0)
        return x  # NO torch ops
```

### Step 2: Compile and Test
```bash
TORCH_CUDA_ARCH_LIST=9.0 bash utils/compile.sh
python3 -m utils.verification
python3 -m utils.profiling
```

### Step 3: Optimization Strategy (Priority Order)

**Priority 1: Algorithmic (>50% impact)**
- Kernel fusion, shared memory tiling, memory coalescing

**Priority 2: Hardware Utilization (20-50% impact)**
- Vectorized loads (float2/float4), warp primitives, occupancy tuning

**Priority 3: Fine-tuning (<20% impact)**
- Instruction-level parallelism, mixed precision, prefetching

### Step 4: Iteration Requirements

- **Correctness**: MUST pass - debug boundary conditions, sync, data types
- **Performance**: MINIMUM 5% faster than torch.compile; push for best possible
- **Cleanup**: Remove intermediate attempts from kernels/ before completion

## 4. OPTIMIZATION CHECKLIST

### Essential
- [ ] Memory Coalescing
- [ ] Kernel Fusion
- [ ] Shared Memory
- [ ] Grid-Stride Loops
- [ ] Boundary Checks (tid < size)

### Performance
- [ ] Vectorized Memory (float2/float4)
- [ ] Warp Primitives
- [ ] Occupancy Tuning
- [ ] Bank Conflict Avoidance

### Correctness
- [ ] Thread Bounds
- [ ] __syncthreads() before shared memory reuse
- [ ] Data types and conversions
- [ ] No out-of-bounds access

## 5. COMMON ISSUES

| Error | Solution |
|-------|----------|
| undefined symbol | Check extern "C" declarations match |
| no kernel image | Verify TORCH_CUDA_ARCH_LIST matches GPU |
| Wrong output | Check kernel math, indexing, test with simple inputs |
| NaN/Inf | Check division by zero, numerical stability |
| Slower than baseline | Combine kernels, improve fusion |

## 6. SUCCESS CRITERIA

- 🎯 MINIMUM: 5% faster than torch.compile
- ✅ Correctness: atol=1e-2, rtol=1e-2
- 🧹 kernels/ contains ONLY final optimized version
