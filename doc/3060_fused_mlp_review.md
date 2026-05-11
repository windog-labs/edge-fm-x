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

## Direct Myelin / XMMA Reuse Check

Checked on `2026-05-10`.

The profiled TRT kernels are real performance references, but they are not
currently exposed as standalone EdgeFM-callable operators:

- `sm80_xmma_gemm_*_trt` comes from TensorRT engine execution, not from a
  source-visible TRT-Edge-LLM GEMM launcher.
- `__myl_FcCast_*` and `__myl_SiluMul_*` are TensorRT Myelin generated kernels
  embedded in the serialized engine.
- `third_party/TensorRT-Edge-LLM/cpp` exposes runtime, attention plugin,
  embedding, KV-cache, sampling, and INT4 groupwise GEMM code, but no public
  BF16/FP16 dense Myelin/XMMA launcher that can be registered as
  `LinearLayer::LinearImpl`.
- The current `src/operators/linear_impl.cu` registry already has the right
  local extension point for source-visible implementations (`cublasLt`,
  `cutlass`, `cutile`, `agent`), but adding a fake `myelin` or `xmma` impl id
  would be misleading unless it calls a buildable API in this repo.

Conclusion: direct Myelin/XMMA operator replacement is not a near-term
production path. The only ways to use those TensorRT tactics are either:

1. build a TensorRT-backed subgraph or engine bridge and call it as a coarse
   backend boundary; or
2. approximate the same behavior using public/source-visible third-party
   kernels such as cublasLt, CUTLASS, TensorRT-LLM CUTLASS kernels, FlashInfer,
   or cuTile.

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

## Reviewed Bridge Design

This is the only reviewed direction that could use TensorRT tactics without
writing a kernel from scratch. It is intentionally not approved for
implementation yet.

### Option A: Isolated TensorRT MLP subengine feasibility

Build a temporary `.tmp_codex` TensorRT engine for a representative MLP subgraph:

- input: `[M, hidden]`
- graph: `GateUp MatMul -> SwiGLU -> DownProj MatMul`
- dtype: BF16 first when available, otherwise FP16 for feasibility only
- shapes: start with `3B m=2048`, then `3B m=1024`
- weights: synthetic or one checkpoint layer only; this is an attribution
  probe, not an accepted production path

Accept the feasibility result only if nsys shows TensorRT selecting the same
`sm80_xmma_gemm_*_trt` / `__myl_*` families and the isolated MLP latency is
materially below EdgeFM's current GateUp + SwiGLU + DownProj path.

#### Probe Result

Ran on `2026-05-10 10:45 +0800`.

The isolated subengine reproduced TensorRT's internal tactic families, but only
the FP16 path reproduced TRT-Edge-LLM's speed:

- BF16 subengine, `3B m=2048/h=2048/i=11008`:
  - engine inspector selected `sm80_xmma_gemm_bf16bf16_*` for GateUp and
    DownProj plus `__myl_SiluMul_*`
  - nsys timed-region median: `11.33 ms` per layer
  - kernel split per layer: GateUp `7.207 ms`, DownProj `3.701 ms`,
    SwiGLU `0.401 ms`
  - this does not beat the current EdgeFM graph-off MLP estimate of
    `10.62 ms` per layer and is far from the TRT-Edge-LLM inferred
    `6.18 ms` per layer
  - conclusion: reject a BF16 TensorRT MLP subengine as the next production
    bridge
- FP16 subengine, same shape:
  - engine inspector selected the same family visible in the TRT-Edge-LLM
    prefill trace: `sm80_xmma_gemm_f16f16_*` plus `__myl_SiluMul_*`
  - nsys timed-region median: `6.26 ms` per layer
  - kernel split per layer: GateUp `3.856 ms`, DownProj `1.984 ms`,
    SwiGLU `0.402 ms`
  - this matches the TRT-Edge-LLM inferred `6.18 ms` per layer and explains
    why TRT prefill MLP is much faster

Artifacts:

- `.tmp_codex/bench/trt_mlp_subengine_3b_bf16_nsys_run.json`
- `.tmp_codex/bench/trt_mlp_subengine_3b_bf16_inspector.json`
- `.tmp_codex/nsys/trt_mlp_subengine_3b_bf16_kernel_summary.md`
- `.tmp_codex/bench/trt_mlp_subengine_3b_fp16_nsys_run.json`
- `.tmp_codex/bench/trt_mlp_subengine_3b_fp16_inspector.json`
- `.tmp_codex/nsys/trt_mlp_subengine_3b_fp16_kernel_summary.md`

This is an attribution result, not production approval. Any production path
based on FP16 TensorRT MLP requires a separate review covering precision,
generation correctness, weight ownership, memory pressure, and CUDA graph
compatibility.

#### Runtime-Weight FP16 TensorRT MLP Probe

Ran on `2026-05-11 09:41 +0800`.

The first isolated FP16 subengine used TensorRT constants and serialized about
`130 MB` of weights for one 3B MLP layer. That proves the tactic but is not
memory-safe as a direct 36-layer production design. A second probe therefore
built the same MLP graph with `gateup_weight` and `down_weight` as TensorRT
runtime inputs.

Results with actual Qwen2.5-3B checkpoint weights, explicitly cast from BF16 to
FP16 for the probe:

- engine size: `67 KB` because it no longer embeds the MLP weights
- layer 0, same-shape runtime weights: `6.01 ms` median; fresh rerun
  `2026-05-11` measured `6.06 ms` median
- layer 35, reusing the same engine with different runtime weight pointers:
  `6.23 ms` median; fresh rerun measured `6.26 ms` median
- inspector tactics: `sm80_xmma_gemm_f16f16_*` for GateUp and DownProj plus a
  Myelin-generated activation kernel
- validation: torch FP16 reference mean relative error is about `0.59-0.60%`

This removes the biggest memory objection to a prototype: a single shape engine
can bind already-loaded per-layer GPU weights instead of duplicating all MLP
weights inside serialized TensorRT engines.

Artifacts:

- `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_fp16_runtime_weights_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_fp16_runtime_weights_verify_inspector.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_fp16_runtime_weights_reuse_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_fp16_runtime_weights_reuse_verify_inspector.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_fp16_runtime_weights_rerun_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_fp16_runtime_weights_reuse_rerun_verify.json`

This still is not a default-on production approval. The remaining gates are:

- generation correctness after using FP16 MLP inside a BF16 checkpoint flow
- CUDA graph capture/replay compatibility for TensorRT enqueue
- binding EdgeFM's existing fused GateUp and DownProj GPU weight buffers without
  layout copies
- a `3B / 2048x32` official CUDA graph target-slice improvement of at least 1%
  with no meaningful regression

#### Source-Visible CUTLASS Layout Check

Ran on `2026-05-10 10:58 +0800`.

To test whether the TRT FP16 gap is mainly a weight-layout issue, a temporary
`third_party/cutlass` probe compared EdgeFM's current `RowMajor x ColumnMajor`
GEMM layout with a prepacked `RowMajor x RowMajor` B layout for the same 3B
prefill shapes.

Best FP16 results:

- GateUp `M=2048,N=22016,K=2048`: best prepacked CUTLASS result
  `6.666 ms`, using `64x128x32_s3_w32x64`
- DownProj `M=2048,N=2048,K=11008`: best prepacked CUTLASS result
  `3.440 ms`, using `64x128x32_s3_w32x64`

Compare the isolated TensorRT FP16 subengine:

- GateUp `3.856 ms`
- DownProj `1.984 ms`

Conclusion: simple weight prepacking plus source-visible classic CUTLASS
`device::Gemm` does not reproduce TRT's FP16 XMMA performance. Do not add a
production prepacked-weight path from this result alone.

Artifacts:

- `.tmp_codex/bench/3060_20260510_cutlass_layout_gateup_fp16_sweep.jsonl`
- `.tmp_codex/bench/3060_20260510_cutlass_layout_down_fp16_sweep.jsonl`
- `.tmp_codex/bench/3060_20260510_cutlass_layout_fp16_sweep_summary.json`

#### Source-Visible cuTile MatMul Probe

Ran on `2026-05-10 11:28 +0800`.

To test whether the existing cuTile dense MatMul sample could serve as a
source-visible FP16/BF16 runner, a temporary probe ran
`third_party/cutile-python/samples/MatMul.py` against the same 3B GateUp and
DownProj shapes after enabling the pip-provided TileIR compiler
(`nvidia-cuda-tileiras`).

Best persistent `64x128x64` results:

- FP16 GateUp `9.87 ms`
- FP16 DownProj `4.80 ms`
- BF16 GateUp `9.75 ms`
- BF16 DownProj `4.79 ms`

Compare current references:

- current BF16 cublasLt GateUp `~6.76-6.85 ms`
- current BF16 cublasLt DownProj `~3.49-3.55 ms`
- TRT FP16 subengine GateUp `3.856 ms`
- TRT FP16 subengine DownProj `1.984 ms`

Conclusion: the existing cuTile dense MatMul sample is not a viable
replacement for the 3060 MLP path. Keep cuTile only as a diagnostic exploration
path unless a materially different fused or sparse runner appears.

Artifacts:

- `.tmp_codex/bench/3060_20260510_cutile_matmul_fp16_probe.json`
- `.tmp_codex/bench/3060_20260510_cutile_matmul_bf16_probe.json`
- `.tmp_codex/bench/3060_20260510_cutile_matmul_summary.json`

#### Source-Visible Third-Party Search

Checked on `2026-05-10 11:06 +0800`.

The remaining vendored paths do not provide a direct RTX 3060 dense FP16
XMMA-equivalent runner:

- `third_party/flashinfer/csrc/trtllm_gemm_runner.cu` only accepts FP8/E2M1
  inputs and writes BF16 output. It is not the dense FP16 MLP path observed in
  the TRT-Edge-LLM trace.
- `third_party/flashinfer/csrc/tgv_gemm.cu` dispatches through SM100/UMMA/TMA
  code and the config file is explicitly SM100-oriented, so it is not usable on
  RTX 3060 `sm86`.
- TensorRT-LLM CUTLASS FP16/BF16 code under
  `third_party/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels`
  is MoE/grouped, fused-MoE, or quantized GEMM. The only current build-integrated
  fused-MoE helper is the `cutlass_prefill_swiglu` diagnostic path, already
  rejected as default-on for 3060.
- `src/operators/CMakeLists.txt` does not compile a standalone
  `MoeGemmRunner<half, half, half>` dense replacement today; wiring one would be
  a new grouped/MoE-based production path and would need operator evidence before
  touching mainline code.

Conclusion: do not add a production `myelin`, `xmma`, `tgv`, or dense FP16
third-party impl id from the current vendor tree. The next implementation-grade
path is either a separately reviewed FP16 TensorRT-backed MLP bridge or a new
source-visible third-party runner that beats the current operator path in an
isolated probe first.

### Option B: Production TensorRT-backed MLP operator

Potential benefit: use TensorRT tactics for GateUp, SwiGLU, and DownProj while
keeping the rest of EdgeFM unchanged.

Known blockers:

- TensorRT engines usually own and store weights, so a per-layer MLP engine
  would duplicate MLP weights and can reintroduce 3060 12GB OOM.
- One engine per decoder layer would add large memory, build, and lifecycle
  complexity.
- A single shared engine cannot simply switch among 36 different layer weights
  without extra weight binding or refit machinery.
- CUDA graph compatibility must be revalidated because TensorRT enqueue and
  EdgeFM graph capture/replay are different ownership models.

Do not implement this until a separate FP16 bridge review proves that the
precision change is acceptable and a memory plan shows that 3B still fits on
RTX 3060 12GB. The BF16 subengine result is not enough.

### Option C: TensorRT-backed full prefill stage bridge

This is even larger and should not be implemented in this tuning pass.

It would ask TensorRT to run full prefill and then hand off KV cache or hidden
state to EdgeFM decode. Current TRT-Edge-LLM runtime does not expose a clean
API for exporting KV cache in EdgeFM's layout, and EdgeFM decode is already not
the main gap. Treat this as a separate project if it is ever pursued.

### Recommendation

Run Option A as a temporary attribution experiment only. Do not add production
TRT-backed operators until the isolated subengine proves the expected kernel
selection and a memory-safe production design is reviewed.

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
- Do not expose a public `myelin` or `xmma` impl id unless it launches a real,
  buildable implementation in this repo.
- Do not duplicate full 3B MLP weights in TensorRT subengines unless a measured
  memory plan proves the model still fits on RTX 3060 12GB.
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
close the `3B / 2048` gap. The current vendored source-visible search did not
find a direct RTX 3060 dense FP16 runner close to TRT's XMMA/Myelin behavior.
The next meaningful implementation is therefore a reviewed FP16 prefill MLP
bridge or a newly identified source-visible third-party runner, with a clean
fallback and a strict CUDA graph end-to-end acceptance gate.

For the Myelin/XMMA request specifically, the isolated probe has now shown that
BF16 TensorRT tactics do not beat EdgeFM, while FP16 TensorRT tactics match
TRT-Edge-LLM. The source-visible CUTLASS prepacked-B check did not close the
gap, `trtllm_gemm_runner` is FP8/E2M1-only, and `tgv_gemm` is SM100-only. A
production TensorRT-backed FP16 MLP bridge remains review-gated and should not
be implemented directly from this document.
