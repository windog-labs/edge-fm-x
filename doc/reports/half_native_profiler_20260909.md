# Half Native CUDA Profiler Evidence

2026-09-09 01:20 UTC. Nsight Systems profiled one complete measured native
Session call per lane on RTX 3060 (one warmup plus one measured call, CUDA and
NVTX tracing). CPU backtrace/context tracing was not supported and disabled;
these reports are CUDA-only and not timing or energy evidence.

## Artifacts

- `profiler-001/baseline.nsys-rep`, `fp16.nsys-rep`, `bf16.nsys-rep`
- Per-lane `*_kern_cuda_gpu_kern_sum.csv` from `nsys stats`

Total sampled GPU kernel time is 167.510 ms baseline, 167.362 ms FP16 and
167.804 ms BF16 across 26300-26440 kernel launches (two Session calls under
profiler overhead). Top kernels are the model's existing BF16 attention/GEMM
path in all three lanes; the half action-head change is not visible as a top
kernel because this SmolVLA trace is dominated by the VLM/vision and diffusion
network.

## Limits

The profiler output includes warmup, profiler overhead and two calls; do not
use it as formal latency. It supports the conclusion that the action head
precision change is not the dominant native CUDA cost in this model and does
not produce an observable kernel-level acceleration on RTX.

NSYS raw files and stats CSVs are under
`artifacts/edgefm-vla-goal/20260906-044509/half-native-benchmark-001/profiler-001/`.
