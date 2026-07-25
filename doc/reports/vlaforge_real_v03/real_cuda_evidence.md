# VLAForge real-model host-CUDA evidence

This report covers an RTX 3060 (sm_86), CUDA 12.8, PyTorch 2.10.0+cu128 host. It makes no Orin claim.

## Full-compute path comparison

| Model | Eager mean | Direct AOTI mean | Generated C++ mean | C++ vs eager | C++ overhead vs direct |
|---|---:|---:|---:|---:|---:|
| diffusiondrive | 19.361 ms | 16.168 ms | 16.304 ms | 1.187x | +0.84% |
| smolvla | 112.912 ms | 45.131 ms | 45.194 ms | 2.498x | +0.14% |

Generated Session and direct AOTI use the same compiled model artifacts. Their near-equal full-compute latency bounds framework overhead; eager speedups are not attributed to new CUDA kernels.

## Stateful invocation and exact-reuse ablations

| Model | Full mean | Same revision mean | New revision mean | Missing revision mean | Same/full speedup |
|---|---:|---:|---:|---:|---:|
| diffusiondrive | 16.304 ms | 2.947 ms | 16.188 ms | 16.210 ms | 5.533x |
| smolvla | 45.194 ms | 0.689 ms | 1.041 ms | 1.044 ms | 65.563x |

DiffusionDrive same-revision Runs reuse only the exact condition cache. SmolVLA also exercises Adapter-owned action queue/cursor state; this queue is not a core-IR assumption.

## 10,000-Run soak

| Model | Commits | Cache hit/miss | State commits | CUDA drift | RSS drift | Aborts |
|---|---:|---:|---:|---:|---:|---:|
| diffusiondrive | 10000 | 10000/0 | 0 | 0 B | 4 KiB | 0 |
| smolvla | 10000 | 200/0 | 20000 | 0 B | 52 KiB | 0 |

## Profiling interpretation

- NSYS and NCU ran the generated no-Python C++ binaries, not Python wrappers.
- Kernel time remains in upstream AOTI/cuDNN/CUTLASS/Triton kernels. VLAForge does not claim those kernels as contributions.
- Scalar Region storage is now 16-byte aligned. The pre-fix SmolVLA bundle emitted two warning sites per Run; the current bundle emits none.
