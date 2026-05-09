# 3060 Fused MLP Review Gate

## Why This Needs Review

The current RTX 3060 gap is no longer explained by a missed small table record.
After disabling default prefill SwiGLU fusion, `3B / 2048x32` still has an
official CUDA graph prefill gap of roughly `+200 ms` versus TRT-Edge-LLM.

The post-default-off graph-off profile shows EdgeFM prefill time is dominated by
existing MLP linear work:

- GateUp: `242.41 ms` (`50.48%` of graph-off prefill GPU time)
- DownProj: `125.26 ms` (`26.09%`)
- activation: `14.64 ms` (`3.05%`)
- attention and QKV/OProj combined are secondary for this case

The `3B / 2048x64` graph-off confirmation trace shows the same prefill shape:
GateUp `243.82 ms` and DownProj `124.74 ms`. Decode length changes total
latency but does not change the long-prefill bottleneck.

TRT-Edge-LLM `3B / 2048x1` nsys reverse attribution gives a useful reference:

- total prefill kernel time: `285.46 ms`
- GEMM category: `240.70 ms` (`84.32%`)
- Gate+Up: `136.50 ms` from `36` TensorRT XMMA GEMM launches
- DownProj: `71.36 ms` from `36` TensorRT XMMA GEMM launches
- QKV: `16.96 ms` from `36` TensorRT Myelin `FcCast` GEMM launches
- OProj: `14.06 ms` from `36` TensorRT XMMA GEMM launches
- AttentionPlugin: `22.90 ms`, including `fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm86_kernel_nl`
- SwiGLU: `14.53 ms` from `36` TensorRT Myelin `SiluMul` launches

The strongest inference is that TRT-Edge-LLM is not removing the MLP structure.
It still runs a Gate+Up GEMM, a separate SwiGLU, and a DownProj GEMM per layer,
but its TensorRT XMMA/Myelin GEMM choices are materially faster on 3060 than
EdgeFM's current long-prefill linear paths.

Small existing-table attempts did not close the gap:

- `fused_gate_up algo_index=3` plus `mlp_down algo_index=2` improved the
  official `3B / 2048x32` target slice by only `2.12 ms` mean, below the
  `>=1%` acceptance rule.
- cublasLt explicit configs did not beat the best heuristic candidates.
- FlashInfer prefill attention `cta_tile_q=128` is only about `0.46 ms` expected
  full-prefill movement for this case.
- The existing TensorRT-LLM/CUTLASS `fused_moe` reuse path for GateUp+SwiGLU was
  already rejected as default-on because it regressed 3060 prefill.

## Proposed Direction

Do not write a new kernel from scratch.

If approved, investigate a larger prefill MLP path that reuses or extends
existing third-party kernels first:

- TensorRT-Edge-LLM/TensorRT XMMA GEMM behavior as measured by
  `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping_kernel_summary.md`
- TensorRT-LLM / CUTLASS fused MoE or GEMM epilogue paths already vendored under
  `third_party/` and currently included by `src/operators/fused_gate_up_activation_op.cu`
- FlashInfer fused MoE / GEMM building blocks under `third_party/flashinfer`
- CUTLASS GEMM + custom epilogue patterns under `third_party/cutlass`
- TRT-Edge-LLM plugin/runtime behavior as the performance reference, without
  copying opaque engine internals into EdgeFM

The first design target should be prefill MLP only:

- input: `[M, hidden]`
- GateUp weights: `[2 * intermediate, hidden]` in EdgeFM's current fused layout
- DownProj weights: `[hidden, intermediate]`
- output: `[M, hidden]`
- dtype: BF16 first, FP16 only after BF16 behavior is stable
- target shapes: `3B m=2048`, then `3B m=1024`, then `1.5B m=2048`

## Required Boundaries

- Keep the current `linear + activation + linear` fallback untouched.
- Gate the prototype behind an env var or operator-table impl id.
- Do not change engine/layer/operator ownership broadly until a prototype shows
  end-to-end improvement.
- Keep allocation and workspace ownership explicit; no persistent hidden global
  buffers without cleanup.
- Preserve CUDA graph compatibility before promoting the path.
- Do not promote graph-off results to official conclusions.

## Acceptance Gate

Before default-enabling any larger path:

- Operator correctness must compare against the existing two-stage path for the
  representative BF16 shapes.
- Generation correctness must pass:
  `tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or test_generate_token_alignment_cuda_graph"`.
- `scripts/operator_table/validate_operator_tables.py` must pass if a table
  record is added.
- Official target CUDA graph benchmark must improve by at least `>=1%`.
- Regression slice must include at least `0.5B / 2048x32`,
  `1.5B / 2048x32`, and `3B / 1024x32`.

## Current Recommendation

Proceed only after review.

The evidence says another blind cublasLt/FlashInfer table sweep is unlikely to
close the `3B / 2048` gap. The next low-risk step is to inspect whether EdgeFM
can reuse or wrap an existing third-party GEMM path closer to TRT's XMMA/Myelin
behavior for Gate+Up and DownProj. If that cannot be done within the current
operator boundary, the next meaningful implementation is a reviewed prefill MLP
prototype based on existing third-party kernels, with a clean fallback and a
strict CUDA graph end-to-end acceptance gate.
