# G5 FP16/BF16 Evidence Summary

2026-09-09 02:30 UTC. This page indexes the current SmolVLA half-precision
evidence used by the ICRA paper goal. It is a CUDA/RTX working-set summary,
not Orin/BPU or full-paper acceptance.

## What Is Established

- Public half replacement: `deployment/half_linear.py` and
  `linear_precision.precision_kind`.
- Python free-running quality: FP16 and BF16 both hold-out 8/8 under
  `MSE <= 1e-5`, `cosine >= 0.9999`.
- Native no-Python full Session: both byte-exact to their Python candidates,
  16/16 pass, held-out 8/8, process maps contain no Python.
- Formal native off benchmark (5x1024): mean 93.351683 baseline,
  94.182185 FP16, 93.792310 BF16 ms. No native half speedup.
- Formal native required replay (5x1024): mean 77.170051 baseline,
  77.217964 FP16, 77.271447 BF16 ms; counter
  `REPLAY_FINAL,11,1,10,1152,0`.
- Stepwise error budget: step0 head inputs byte exact; step9 held-out head MSE
  is ~1.15e-4 FP16 / ~1.26e-4 BF16 while final action MSE remains below 1e-5.
- NSYS CUDA profile: action head precision is not the dominant native kernel.

## Evidence Chain

- Python quality/audit: `half_free_running_20260908.md`
- FP16 native: `fp16_native_full_20260908.md`
- BF16 native: `bf16_native_full_20260908.md`
- Formal off benchmark: `half_native_benchmark_formal_20260909.md`
- Error budget: `half_stepwise_error_20260909.md`
- Profiler: `half_native_profiler_20260909.md`
- Replay pilot: `half_native_replay_pilot_20260909.md`
- Formal replay: `half_native_replay_formal_20260909.md`

## Preserved Negative

INT8 global/site/step/step-group held-out quality remains 0/8. That failure is
not erased by the FP16/BF16 passes. There is no claim that half precision is
lossless, faster in ordinary execution, or equivalent on all models.
