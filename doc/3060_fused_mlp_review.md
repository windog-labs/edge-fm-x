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

#### EdgeFM-Layout BF16 Runtime Weights With FP16 Compute

Ran on `2026-05-11 10:00 +0800`.

The previous runtime-weight FP16 probe still assumed preconverted FP16 runtime
weights in TensorRT's `[H, 2I]` and `[I, H]` matrix layout. That is not the
resident EdgeFM layout. A follow-up probe therefore used the production-shaped
inputs:

- activation input: BF16 `[M, H]`
- GateUp runtime weight: BF16 `[2I, H]` in EdgeFM's current fused `[up, gate]`
  order
- DownProj runtime weight: BF16 `[H, I]`
- TensorRT graph: cast activation and weights to FP16 inside the engine, run
  MatMul with `MatrixOperation::kTRANSPOSE` for both weights, run Myelin
  activation, and cast output back to BF16

The inspector confirms `sm80_xmma_gemm_f16f16_*` for both GEMMs, plus Myelin
cast/activation layers. It does not select the slower BF16 XMMA path.

Results:

- 3B `m=2048`, layer 0: `7.02 ms` median; layer 35 with the same engine:
  `7.18 ms` median
- 3B `m=1024`, layer 0: `4.12 ms` median
- 3B `m=512`, layer 0: `2.47 ms` median
- 1.5B `m=2048`, layer 0: `4.29 ms` median
- 0.5B `m=2048`, layer 0: `1.57 ms` median
- engine sizes: `80-93 KB`; weights remain runtime inputs, not serialized
- validation: torch FP16-compute/BF16-output reference mean relative error is
  about `0.4-0.6%`

Artifacts:

- `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_bf16_edgefm_layout_fp16_compute_reuse_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_m1024_bf16_edgefm_layout_fp16_compute_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_m512_bf16_edgefm_layout_fp16_compute_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_1p5b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`
- `.tmp_codex/bench/20260511_trt_mlp_0p5b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`

This is now the preferred prototype shape. It directly addresses the earlier
layout and persistent FP16 weight-copy concern. It still requires production C++
verification for TensorRT enqueue under CUDA graph capture, pointer binding to
EdgeFM-owned tensors, generation correctness, and official CUDA graph latency.

#### EdgeFM Bridge Matrix

Ran on `2026-05-11 11:25-11:43 +0800`.

An optional C++ bridge was prototyped behind both compile and runtime gates:

- compile gate: `BUILD_TRT_MLP_BRIDGE=ON`
- runtime gate: `EDGE_FM_PREFILL_TRT_MLP=1`
- engine directory: `EDGE_FM_TRT_MLP_ENGINE_DIR`
- default build/path: native MLP, no TensorRT dependency

The bridge loads a runtime-weight TensorRT engine per MLP shape, binds the
existing EdgeFM BF16 activation, GateUp weight, DownProj weight, and output
tensors, and falls back to native `linear + activation + linear` on any missing
engine, IO mismatch, disabled env var, or enqueue failure.

During this validation, an independent prefill CUDA graph bug was found:
first-time prefill graph capture instantiated the graph but did not replay it
before `generate()` advanced. That could leave the first sampled token
uninitialized. The fix is to immediately launch the captured graph after
capture; the new regression test is
`test_generate_token_alignment_prefill_cuda_graph_first_request`.

Correctness after the graph fix:

- `3B / 2048x32` bridge+CUDA graph matches
  `transformers.generate(do_sample=False, eos_token_id=None)` for all 32
  generated tokens.
- raw:
  `.tmp_codex/bench/3060_20260511_1117_3b_2048x32_trt_mlp_bridge_transformers_generate_compare_after_prefill_graph_fix.json`
- obsolete temporary references:
  `.tmp_codex/bench/3060_20260511_1102_3b_2048x32_trt_mlp_bridge_transformers_token_compare.json`
  and
  `.tmp_codex/bench/3060_20260511_1111_3b_2048x32_trt_mlp_bridge_transformers_token_compare_after_prefill_graph_fix.json`
  used an inconsistent ad hoc reference sequence and should not be cited.

Engine coverage after the graph fix:

- `3B`: `m=512/1024/2048`, `H=2048`, `I=11008`
- `1.5B`: `m=512/1024/2048`, `H=1536`, `I=8960`
- `0.5B`: `m=512/1024/2048`, `H=896`, `I=4864`

Experimental EdgeFM-only performance matrix after the graph fix:

| Case | Native EdgeFM | Bridge EdgeFM | Latest TRT | Bridge gain | Remaining gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3B / 2048x32` | `1125.25 ms` | `986.44 ms` | `927.69 ms` | `+138.81 ms` | `+58.75 ms` |
| `3B / 2048x64` | `1778.55 ms` | `1638.02 ms` | `1584.36 ms` | `+140.53 ms` | `+53.65 ms` |
| `3B / 1024x32` | `858.97 ms` | `810.40 ms` | `764.35 ms` | `+48.57 ms` | `+46.05 ms` |
| `3B / 1024x64` | `1493.97 ms` | `1446.06 ms` | `1405.06 ms` | `+47.91 ms` | `+41.00 ms` |
| `1.5B / 2048x32` | `566.06 ms` | `501.22 ms` | `474.23 ms` | `+64.84 ms` | `+26.99 ms` |
| `0.5B / 512x32` | `132.39 ms` | `130.57 ms` | `134.66 ms` | `+1.82 ms` | `-4.09 ms` |

The full experimental matrix improved all 18 cases versus native EdgeFM. It
beats the latest official TRT baseline in `2/18` mean cases and `2/18` median
cases, both `0.5B / 512`. This is still not a same-run TRT comparison.

Residual `3B / 2048x1` bridge profile:

- bridge prefill kernel total: `350.41 ms`
- TRT prefill kernel total: `285.46 ms`
- bridge MLP core without casts: GateUp `134.11 ms`, SwiGLU `14.47 ms`,
  DownProj `68.33 ms`, total `216.92 ms`
- TRT inferred MLP core: GateUp `136.50 ms`, SwiGLU `14.53 ms`,
  DownProj `71.36 ms`, total `222.39 ms`
- bridge-specific casts: BF16 weight casts `29.86 ms`,
  activation/output casts `3.77 ms`
- remaining non-MLP deltas versus TRT: QKV `+16.08 ms`, OProj `+10.04 ms`,
  prefill attention `+10.40 ms`

This means the bridge has already matched TRT's MLP GEMM/activation core for
the 3B long-prefill case. The next high-value experiment is not another GEMM
tile search; it is a review-gated memory/performance check for persistent FP16
MLP weight copies that could remove the runtime BF16 weight-cast cost. That
must remain optional until the 3B 12GB memory plan is proven.

FP16 runtime-weight input follow-up, `2026-05-11 12:03 +0800`:

- BF16 EdgeFM-layout runtime weights cast inside TensorRT: `7.02 ms/layer`
- GateUp weight bound as FP16, DownProj still BF16: `6.60 ms/layer`
- DownProj weight bound as FP16, GateUp still BF16: `6.90 ms/layer`
- GateUp and DownProj both bound as FP16: `6.25 ms/layer`
- pure FP16 runtime weights in the older probe layout: `6.06 ms/layer`

The `both` nsys probe shows two `sm80_xmma_gemm_f16f16_*` kernels plus the
activation/input/output cast kernels. The large BF16 weight-cast kernels seen
in the bridge residual trace are absent from the timed region. This validates
the direction but also sharpens the memory tradeoff: copying both GateUp and
DownProj MLP weights for 3B costs about `4.57 GiB` and previous warmed memory
simulation left only about `123 MiB` free after one generate. GateUp-only costs
about `3.02 GiB`, captured most of the single-layer gain, and previously left
about `1.7 GiB` warmed free headroom. That makes GateUp-only the first
implementation candidate; both-weights copy remains review-gated.

GateUp-FP16 bridge implementation slice, `2026-05-11 12:36 +0800`:

- compile/runtime gates remain unchanged: `BUILD_TRT_MLP_BRIDGE=ON`,
  `EDGE_FM_PREFILL_TRT_MLP=1`, and explicit
  `EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup`
- no kernel was written from scratch; the persistent BF16->FP16 GateUp copies
  are created by a small TensorRT cast subengine
- actual bridge logs for `3B / m=2048` show the `fp16weights-gateup` engine
  loaded, one cast engine built for `r22016_c2048`, and `36` persistent GateUp
  copies of `90,177,536` bytes each
- correctness: `3B / 2048x32` CUDA graph matched
  `transformers.generate(do_sample=False, eos_token_id=None)` for all `32`
  generated tokens
- warmed performance: `3B / 2048x32` EdgeFM mean `964.20 ms`, median
  `964.34 ms`, prefill avg `334.83 ms`, decode avg `629.12 ms`
- comparison: `-22.24 ms` versus the BF16-weight bridge and `-161.06 ms`
  versus native EdgeFM, but still `+36.51 ms` slower than the latest TRT
  baseline `927.69 ms`
- implementation cleanup: Qwen's operator cache reset now also clears bridge
  runtime/cast/persistent-copy caches, and the build-path helper preserves
  explicit `EDGE_FM_BUILD_DIR` priority so bridge gates cannot silently import
  the default build

Artifacts:

- target performance:
  `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_runs3_clean.json`
- correctness:
  `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_transformers_compare_clean.json`
- post-cleanup smoke:
  `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_post_cache_reset_smoke_clean.json`

Conclusion: GateUp-FP16 is a real incremental win and should remain the next
bridge sub-candidate, but it is not enough to reach TRT and is not ready for
default-on. The immediate next step is engine generation and regression slices
for the remaining official/high-value shapes; after that, residual profiling
should focus on DownProj cast, QKV, OProj, and attention rather than another
small GEMM tile search.

GateUp-FP16 full official-shape matrix, `2026-05-11 12:54 +0800`:

All official MLP shapes now have GateUp-FP16 TensorRT engines:
`0.5B/1.5B/3B x m=512/1024/2048`. The 18-case EdgeFM-only CUDA graph matrix
completed without missing-engine fallback. TRT numbers below reuse the latest
official baseline because the Python 3.13 bridge build and Python 3.10
`edge_fm_trt` binding are still ABI-split.

Summary:

- `18/18` cases improved versus the BF16 bridge
- total mean improvement versus BF16 bridge: `191.57 ms`
- mean per-case improvement versus BF16 bridge: `10.64 ms`
- `3/18` mean cases and `3/18` median cases reach or beat the latest TRT
  baseline
- BF16 bridge had `2/18`, so GateUp-FP16 is better but still far from the final
  18-case target

Top remaining gaps:

| Case | GateUp-FP16 EdgeFM | Latest TRT | Gap | Gain vs BF16 bridge | Prefill avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3b 2048x32` | `970.55 ms` | `927.69 ms` | `+42.86 ms` | `+15.89 ms` | `339.82 ms` |
| `3b 2048x64` | `1621.74 ms` | `1584.36 ms` | `+37.37 ms` | `+16.28 ms` | `340.18 ms` |
| `3b 1024x32` | `787.77 ms` | `764.35 ms` | `+23.43 ms` | `+22.63 ms` | `172.62 ms` |
| `0.5b 2048x64` | `311.50 ms` | `293.41 ms` | `+18.09 ms` | `+2.64 ms` | `54.02 ms` |
| `3b 1024x64` | `1422.92 ms` | `1405.06 ms` | `+17.86 ms` | `+23.14 ms` | `173.11 ms` |
| `1.5b 2048x64` | `829.03 ms` | `811.69 ms` | `+17.34 ms` | `+9.30 ms` | `165.18 ms` |
| `1.5b 2048x32` | `491.52 ms` | `474.23 ms` | `+17.29 ms` | `+9.70 ms` | `164.88 ms` |
| `1.5b 1024x64` | `726.91 ms` | `713.25 ms` | `+13.65 ms` | `+9.13 ms` | `83.08 ms` |

Artifacts:

- full matrix:
  `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_full_llm_matrix_runs3_clean.json`
- summary:
  `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_full_llm_matrix_runs3_summary.json`
- engine probe raws:
  `.tmp_codex/bench/20260511_trt_mlp_0p5b_m512_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_0p5b_m1024_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_0p5b_m2048_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_1p5b_m512_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_1p5b_m1024_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_1p5b_m2048_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_3b_m512_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_3b_m1024_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_gateup_verify.json`

Conclusion: keep GateUp-FP16 as the best current default-off bridge candidate.
The next useful work is a fresh residual profile under this mode. Another
small GEMM tile search is unlikely to close the remaining `3B / 2048` gap; the
remaining candidates are DownProj weight-cast removal, QKV/OProj, and prefill
attention.

GateUp-FP16 residual profile, `2026-05-11 13:18 +0800`:

The follow-up `3B / 2048x1` graph-off profile confirms that GateUp-FP16 changes
the shape of the problem:

- stage kernel total: EdgeFM bridge `336.08 ms`, TRT reference `285.46 ms`,
  delta `+50.62 ms`
- MLP core without casts: EdgeFM bridge `222.54 ms`, TRT reference
  `222.39 ms`, delta `+0.14 ms`
- GateUp XMMA delta: `+0.11 ms`
- DownProj XMMA delta: `+0.03 ms`
- SwiGLU delta: `-0.001 ms`
- remaining bridge casts: `11.88 ms`, dominated by DownProj BF16->FP16 weight
  cast `9.93 ms`
- non-MLP deltas versus TRT: QKV `+16.39 ms`, OProj `+10.54 ms`, prefill
  attention `+11.29 ms`, norm `+0.31 ms`

This rules out another small MLP GEMM tile search as the next high-value path.
DownProj-FP16 is still worth one bounded target-slice test because it may remove
about `10 ms`, but it cannot close the full `3B / 2048x32` gap by itself. The
next major review-gated candidates are QKV/OProj TensorRT-style subpaths or an
attention-plugin/FlashInfer prefill route backed by a fresh profile.

GateUp+DownProj-FP16 target slice, `2026-05-11 13:24 +0800`:

The bounded DownProj follow-up used `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both`, which
keeps GateUp in persistent FP16 and additionally binds persistent FP16 DownProj
weights. The single `3B / 2048x32` target slice did not OOM and did not fall
back to native MLP:

- loaded `fp16weights-both_m2048_h2048_i11008.engine`
- created `36` GateUp FP16 copies of `90,177,536` bytes each
- created `36` DownProj FP16 copies of `45,088,768` bytes each
- correctness passed versus `transformers.generate(do_sample=False,
  eos_token_id=None)` for all `32` generated tokens
- EdgeFM mean `952.10 ms`, median `952.18 ms`, prefill avg `322.12 ms`
- improvement versus GateUp-FP16 `3B / 2048x32`: `18.45 ms` mean,
  `17.70 ms` prefill
- remaining gap versus latest TRT baseline: `+24.41 ms`

This is a real improvement, but it is still not a default-on decision. It uses
the highest-memory MLP mode and has only one official target slice plus
correctness coverage. The memory probe confirms the risk: after engine init the
process still had `5,343.6 MiB` free, but after first generate created all
persistent FP16 MLP copies and CUDA graph state it had only `161.6 MiB` free;
the same low headroom remained after timed runs. Therefore `both` is
performance-valid but memory-unsafe for 3B expansion/default-on in the current
policy. Keep GateUp-FP16 as the safer bridge candidate and move
attention/QKV/OProj ahead unless a selective/lazy/evictable FP16-weight memory
design is reviewed.

Artifacts:

- target slice:
  `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_runs3_clean.json`
- summary:
  `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_summary.json`
- correctness:
  `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_transformers_compare_clean.json`
- memory:
  `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_memory_probe_clean.json`

#### QKV / OProj TensorRT Linear Probe

Ran on `2026-05-11 13:35 +0800`.

After GateUp-FP16, the residual profile shows QKV `+16.39 ms` and OProj
`+10.54 ms` versus TRT on `3B / m=2048`. A temporary TensorRT linear subengine
therefore tested the same kind of runtime-weight, EdgeFM-layout bridge for
attention projections:

- input/output: BF16
- compute: FP16
- runtime weight layout: EdgeFM resident `[out_features, in_features]`
- shapes: QKV `[M=2048, K=2048, N=2560]`, OProj
  `[M=2048, K=2048, N=2048]`
- weights: synthetic; this is tactic feasibility, not correctness against
  checkpoint output

Results, estimated across 36 layers:

| Path | QKV | OProj | Combined |
| --- | ---: | ---: | ---: |
| current EdgeFM profile | `33.35 ms` | `24.60 ms` | `57.95 ms` |
| TRT reference profile | `16.96 ms` | `14.06 ms` | `31.01 ms` |
| TensorRT BF16 weight inputs | `21.34 ms` | `18.03 ms` | `39.37 ms` |
| TensorRT FP16 weight inputs | `19.06 ms` | `16.18 ms` | `35.24 ms` |

The inspector reports `sm80_xmma_gemm_f16f16_*` plus Myelin cast/FcCast
tactics. The FP16-weight-input estimate is still `4.23 ms` slower than the TRT
reference, but it is `22.71 ms` faster than current EdgeFM QKV+OProj. That is
large enough to be the next serious candidate, especially because `both` MLP
mode leaves a `+24.41 ms` end-to-end gap but is memory-unsafe for 3B default-on.

This should not be implemented as an ad hoc code change. It needs a separate
review covering:

- whether to bridge QKV only, OProj only, or both
- how to handle Q/K/V split, RoPE/KV copy, and attention input layout without
  adding back the saved time as memory copies
- whether FP16 persistent projection weights fit in 3B when combined with the
  safer GateUp-FP16 mode
- how to keep native cublasLt fallback and CUDA graph behavior intact

Artifacts:

- summary:
  `.tmp_codex/bench/20260511_trt_linear_qkv_oproj_3b_m2048_summary.json`
- QKV BF16/FP16 weight probes:
  `.tmp_codex/bench/20260511_trt_linear_qkv_3b_m2048_bf16weight_verify.json`,
  `.tmp_codex/bench/20260511_trt_linear_qkv_3b_m2048_fp16weight_verify.json`
- OProj BF16/FP16 weight probes:
  `.tmp_codex/bench/20260511_trt_linear_oproj_3b_m2048_bf16weight_verify.json`,
  `.tmp_codex/bench/20260511_trt_linear_oproj_3b_m2048_fp16weight_verify.json`

Artifacts:

- raw:
  `.tmp_codex/nsys/3060_3b_2048x1_gateup_fp16_bridge_mapping.nsys-rep`
- triage:
  `.tmp_codex/nsys/3060_3b_2048x1_gateup_fp16_bridge_mapping_triage.md`
- structured summary:
  `.tmp_codex/nsys/3060_3b_2048x1_gateup_fp16_bridge_residual_summary.json`

Artifacts:

- clean 18-case EdgeFM-only bridge matrix:
  `.tmp_codex/bench/3060_20260511_1230_full_llm_matrix_trt_mlp_bridge_edgefm_only_clean.json`
- 18-case summary:
  `.tmp_codex/bench/3060_20260511_1230_full_llm_matrix_trt_mlp_bridge_edgefm_only_summary.json`
- residual profile summary:
  `.tmp_codex/nsys/3060_3b_2048x1_trt_mlp_bridge_residual_summary.json`
- residual profile triage:
  `.tmp_codex/nsys/3060_3b_2048x1_trt_mlp_bridge_mapping_triage.md`
- FP16 weight-input probes:
  `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_both_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_gateup_verify.json`,
  `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_down_verify.json`
- FP16 weight-input nsys:
  `.tmp_codex/nsys/trt_mlp_3b_m2048_fp16weights_both_probe.nsys-rep`
- summary:
  `.tmp_codex/bench/3060_20260511_1125_trt_mlp_bridge_slice_summary_after_prefill_graph_fix.json`
- target slice:
  `.tmp_codex/bench/3060_20260511_1118_3b_2048x32_trt_mlp_bridge_edgefm_only_after_prefill_graph_fix.json`
- 3B regression:
  `.tmp_codex/bench/3060_20260511_1120_3b_512_1024x32_trt_mlp_bridge_regression_after_prefill_graph_fix.json`
- 0.5B/1.5B regression:
  `.tmp_codex/bench/3060_20260511_1122_0p5_1p5_2048x32_trt_mlp_bridge_regression_after_prefill_graph_fix.json`

Conclusion: the bridge is a valid, high-impact experimental direction, but it
is not approved as default-on. It now covers all official MLP shapes and
improves all 18 experimental EdgeFM-only cases versus native EdgeFM. The
remaining blockers are the `3B` long-prefill residual gap, same-run in-process
TRT numbers, and full optional-path correctness/packaging. The current Python
3.13 bridge build and Python 3.10 `edge_fm_trt` binding are still ABI-split in
this workspace.

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

Known blockers and current status:

- The runtime-weight prototype avoids serialized per-layer MLP weight
  duplication by binding EdgeFM-owned weights at enqueue time.
- One shared engine per shape is feasible for same-shape layers; layer-specific
  weights are rebound per layer.
- CUDA graph capture/replay is compatible enough for the tested slices after
  the prefill first-capture replay fix.
- Remaining blockers: a compatible same-run TRT benchmark path, full
  optional-path packaging/correctness, and a residual `~59 ms` gap on
  `3B / 2048x32`.

Do not default-enable this path until those blockers are closed. The prototype
may remain compile/runtime gated for continued measurement.

### Option C: TensorRT-backed full prefill stage bridge

This is even larger and should not be implemented in this tuning pass.

It would ask TensorRT to run full prefill and then hand off KV cache or hidden
state to EdgeFM decode. Current TRT-Edge-LLM runtime does not expose a clean
API for exporting KV cache in EdgeFM's layout, and EdgeFM decode is already not
the main gap. Treat this as a separate project if it is ever pursued.

### Recommendation

Continue Option B only as a gated prototype. Do not add a public `myelin` or
`xmma` operator id, and do not make TensorRT engines part of the default runtime
until the full correctness and 18-case performance gates are met.

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
