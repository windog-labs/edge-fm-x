# Qwen3.5 Long-Loop Closure

- Date: `2026-05-23`
- Scope: Qwen3.5 text-only greedy generation on RTX 3060 / SM86.
- Contract: exact token alignment with Transformers for 0.8B and 2B; no approximation, quantization, or large runtime/scheduler redesign in this phase.

## Summary

The local kernel phase is closed. The remaining material hotspots are either at an operator-specific ceiling or require changing the algorithm/precision contract:

| Area | Latest evidence | Status |
|---|---|---|
| GateUp+SwiGLU decode GEMV | Iter121 accepted; Iter122 NCU shows `95.34%` DRAM and `93.18%` achieved occupancy | At current exact BF16 memory ceiling |
| LMHead top1 | Iter110 NCU shows `97.60%` DRAM; Iter117 exact row-norm pruning prunes `0 / 248320`; Iter118 BF16x2 sweep is only `~0.2%` | At current exact search ceiling |
| P0 GatedDeltaNet prefill | Iter105 best token-safe current-layout path; remaining raw SOL gap is recurrence/barrier/small-grid bound | Plateaued without new state/layout algorithm |
| Dense linear table tactics | Iter107/114/116 accepted useful shape-family records; weak single-shape records rejected | Closed for low-risk cublasLt table tuning |
| Decode conv/GatedDelta/RMSNorm, RMSNorm/add/attention | Accepted or rejected with token/full-generate gates | Closed for local fusion/retune |

## Knowledge-Pass Result

The KernelPilot knowledge pass was scoped to GEMM, activation fusion, and quantization:

- `matmul-gemm`: skinny batch decode GEMV is memory-bound; material gains usually need fewer bytes, quantization, or a different fused projection.
- `activation-fusion`: standalone activation kernels should be fused into neighboring compute; Iter121 already moved GateUp+SwiGLU into one GEMV+activation launch.
- `quantization-fp8`: production serving stacks get large GEMV gains from FP8/FP4/INT4/W8A8 paths, but those require weight format, scale layout, and tolerance contracts.
- TensorRT-LLM / vLLM references point to FP8/FP4/AWQ/GPTQ and fused-MoE/GEMV work as the next high-value family, not a simple BF16 launch tweak.

## Why Not Continue The Current Exact-BF16 Loop

For GateUp+SwiGLU, the accepted kernel reads the required BF16 weights for two dot products and already reaches `95.34%` DRAM throughput. A pure launch-shape rewrite can at most recover a small fraction of the remaining `~4.66%` memory headroom. Since this kernel accounts for `29.3%` of the graph-off decode GPU time, the upper bound from perfect memory saturation is roughly `1.4%` decode-GPU-side, before graph/runtime noise and correctness risk.

For LMHead top1, the current exact algorithm already reaches DRAM roofline. The only meaningful improvement is reading fewer vocab rows/bytes. The tested exact row-norm bound has no pruning headroom on the sampled Qwen3.5 states.

For P0 GatedDeltaNet, the remaining gap is not a raw hardware-peak failure. The recurrence and token-boundary barriers limit available parallelism. Prior standalone wins failed to transfer to full generate unless they preserved the exact production dataflow.

## Deferred High-Cost Work

These are valid future projects, but they should be separate specs with explicit acceptance/tolerance choices:

1. Compressed or quantized Qwen3.5 GEMV/LMHead path: define FP8/INT8/INT4 format, scales, calibration, and token-alignment or tolerance gate.
2. Exact LMHead indexing/search: build a new data structure that can provably skip rows for this model distribution; current Cauchy row-norm pruning is blocked.
3. Larger fused projection design: fuse multiple decode projections or layer-local chains in a standalone repo, then migrate only if it beats the current graph-on matrix.
4. Runtime/scheduler redesign: batch tiny recurrent kernels or multi-layer scheduling. This violates the current "no large runtime refactor" constraint and needs user confirmation.

## Closure Decision

Converge this phase. The best current state is worth committing and pushing. Continuing with the same local-BF16 kernel loop is low ROI; the next gains require a new algorithmic contract rather than more micro-tuning.
