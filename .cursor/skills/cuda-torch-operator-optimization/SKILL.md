---
name: cuda-torch-operator-optimization
description: Optimize a single PyTorch operator with custom CUDA kernels using the CUDA-Agent workflow. Use when the user wants to accelerate a torch operator with CUDA, implement a CUDA extension for PyTorch, or optimize element-wise/GEMM/conv operations with custom kernels.
---

# CUDA Torch Operator Optimization

Accelerate a single PyTorch operator by implementing a custom CUDA C++ extension, following the CUDA-Agent agent_workdir workflow.

## Prerequisites

- Use `agent_workdir` from [CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent) as workspace template
- Workspace must have: `binding.cpp`, `binding_registry.h`, `utils/`, `kernels/`

## Quick Workflow

### 1. Setup model.py (baseline, do not modify after setup)

```python
# model.py - Original PyTorch implementation
class Model(nn.Module):
    def forward(self, ...):
        return ...  # Pure PyTorch

def get_inputs():
    return [torch.randn(...).cuda(), ...]

def get_init_inputs():
    return [...]  # Model.__init__ args
```

### 2. Implement kernel in kernels/

- `kernels/my_op.cu`: Pure CUDA kernel (NO torch::* in .cu)
- `kernels/my_op_binding.cpp`: PyTorch binding with REGISTER_BINDING

### 3. Create model_new.py

Use only `cuda_extension.my_op_forward(...)` and tensor creation. **NO torch ops.**

### 4. Iterate

```bash
bash utils/compile.sh
python3 -m utils.verification   # Must pass
python3 -m utils.profiling     # Target: ≥5% faster than torch.compile
```

## Critical Constraints

| Forbidden | Allowed |
|-----------|---------|
| torch::* in .cu or _binding.cpp | Raw CUDA kernels, cuBLAS (GEMM), cuDNN (Conv) |
| torch ops in model_new.py | cuda_extension.*, torch.tensor creation |
| Modify utils/, binding.cpp, binding_registry.h | Modify kernels/ and model_new.py only |

## Success Criteria

- ✅ Correctness passes (atol=1e-2, rtol=1e-2)
- 🎯 ≥5% faster than torch.compile baseline
- 🧹 kernels/ contains only final optimized version

## Full Reference

For optimization checklist, code templates, and troubleshooting, see [reference.md](reference.md).
