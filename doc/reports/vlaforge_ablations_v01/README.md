# VLAForge paper ablations

Status: **passed**.

## 1. InputRevision exact reuse

Five fresh processes per cell; confidence intervals use process-cluster bootstrap. DiffusionDrive is the cache-only control. SmolVLA non-full modes also exercise the Adapter-owned action queue and therefore are not presented as cache-only.

| Model | Mode | Mean ms [95% CI] | p50 | p90 | p99 | Full/mode | Hit/miss |
|---|---|---:|---:|---:|---:|---:|---:|
| diffusiondrive | full | 16.403 [16.383, 16.424] | 16.390 | 16.958 | 17.333 | 1.000x | 0/500 |
| diffusiondrive | missing | 16.391 [16.369, 16.417] | 16.369 | 16.960 | 17.334 | 1.001x | 0/500 |
| diffusiondrive | new | 16.400 [16.383, 16.413] | 16.393 | 16.967 | 17.339 | 1.000x | 0/500 |
| diffusiondrive | same | 3.064 [3.058, 3.070] | 3.048 | 3.059 | 3.362 | 5.353x | 500/0 |
| smolvla | full | 45.505 [45.442, 45.569] | 45.378 | 45.776 | 47.769 | 1.000x | 0/500 |
| smolvla | missing | 1.047 [1.044, 1.050] | 0.083 | 0.085 | 48.166 | 43.479x | 0/10 |
| smolvla | new | 1.047 [1.046, 1.049] | 0.083 | 0.085 | 48.207 | 43.450x | 0/10 |
| smolvla | same | 0.694 [0.692, 0.696] | 0.083 | 0.085 | 30.507 | 65.606x | 10/0 |

## 2. Static arena lifetime packing

| Model | Baseline | Packed | Saved | 10k CUDA drift | RSS drift |
|---|---:|---:|---:|---:|---:|
| smolvla | 2,331,712 B | 2,329,792 B | 1,920 B (0.082%) | 0 B | 52 KiB |
| diffusiondrive | 5,158,016 B | 5,155,392 B | 2,624 B (0.051%) | 0 B | 4 KiB |

The control is the compiler certificate's unpacked logical-lifetime memory plan. This is a plan-level packing ablation, not a dynamic allocator microbenchmark.

## 3. Transaction failure and retry

| Model | Abort | Retry commit | Retry state commit | Output preserved | State version |
|---|---:|---:|---:|---|---|
| smolvla | 1 | 1 | 2 | true | passed |
| diffusiondrive | 1 | 1 | 0 | true | not_applicable_stateless_model |

## 4. Deployment boundary

| Model | Generated C++ overhead vs direct AOTI | Exact direct/C++ output |
|---|---:|---|
| smolvla | mean +0.508% [+0.430%, +0.628%] | true |
| diffusiondrive | mean +0.509% [+0.151%, +0.711%] | true |

The clean installed-wheel evaluation passed both session/invocation residency modes, ran under an invalid Python environment without libpython, and rejected all 8 negative schema/ABI/artifact/input cases per variant.

## Claim boundary

All timing is RTX 3060 (sm_86)/CUDA 12.8 Host-CUDA. AOTI/cuDNN/CUTLASS/Triton model kernels are upstream; VLAForge claims the state/cache/transaction/memory and verified artifact orchestration semantics. Orin, real-vehicle loops, sensor synchronization, middleware and legacy EdgeFM kernel optimization are out of scope.
