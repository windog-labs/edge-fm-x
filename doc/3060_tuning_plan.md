# 3060 LLM Runtime and Kernel Tuning Plan

## Goal

Close and then exceed TRT-Edge-LLM on the RTX 3060 LLM matrix without large
EdgeFM design changes:

- models: `Qwen2.5-{0.5B,1.5B,3B}`
- prefill lengths: `512`, `1024`, `2048`
- decode lengths: `32`, `64`

The official comparison remains `EdgeFM(cuda graph)` versus `TRT-Edge-LLM`.
`graph-off`, `nsys`, `ncu`, operator microbenchmarks, and isolated subengines are
attribution tools only. The standing safety rules live in
[3060_tuning_rules.md](./3060_tuning_rules.md).

## Current Acceptance Ladder

The 2026-05-17 working target is now staged:

1. Tactical target: remove all `10 ms+` gaps on the official long-context LLM
   matrix. A remaining gap within `5 ms` is acceptable as a Stage-2 checkpoint
   when the remaining work clearly requires long-loop kernel research.
2. Full target: continue source-visible/plugin-op work until EdgeFM catches and
   then exceeds external `TRT-Edge-LLM`.
3. Acceptance rule: keep correctness first, then absorb stable localized wins
   even below `1%` when they move a target slice toward the `<=5 ms` tactical
   target without adjacent regressions.

Current pressure point: after the table-driven TRT-FMHA plugin-op attention,
3B SwiGLU launch retune, guarded QKV persistent-weight checkpoint, the
0.5B/1.5B plugin-op attention extension, the default-on strided-QKV prefill
runtime path, the token-dim Q/KV pack refinement, source-op table-selection log
demotion, source-op runtime config caching, the 3B `m=1024` MLP down-tile split,
the 1.5B decode chunk refinement, the 0.5B q14 decode chunk refinement, and
the 0.5B `m=2048` MLP/QKV source-op refinements, and the 3B `m=512` MLP
down-tile split, the full checked 3060 LLM matrix now has no positive residual
above `+3 ms`. The latest targeted checks move the former 3B short-context
positive slice clearly faster than TRT-Edge-LLM.

Latest full-matrix gate check, CUDA graph on, warmup `2`, runs `5`:

- `0.5B`: `512x32 -8.712 ms`, `512x64 -5.772 ms`, `1024x32 -0.640 ms`,
  `1024x64 -0.356 ms`, `2048x32 +1.191 ms`, `2048x64 -1.195 ms`.
- `1.5B`: `512x32 +0.163 ms`, `512x64 +0.826 ms`, `1024x32 +0.063 ms`,
  `1024x64 -2.793 ms`, `2048x32 -2.801 ms`, `2048x64 -8.611 ms`.
- `3B`: latest full-matrix pass before the 512 MLP split was `512x32
  +0.812 ms`, `512x64 -6.605 ms`, `1024x32 -0.100 ms`, `1024x64 -6.857 ms`,
  `2048x32 -1.565 ms`, `2048x64 -13.949 ms`; targeted post-split checks move
  `512x32` to `-5.161 ms` and `512x64` to `-11.822 ms`.

The next accepted change does not need to exceed `1%`; it only needs to be
correctness-clean, repeatable, and move one of the remaining positive slices
toward or below zero without creating a new `+3 ms` residual.

Current quick-probe status under this gate:

- Rejected: runtime strided-QKV/fused-K combo, strided-QKV-only,
  BF16-direct QKV/OProj weights, OProj tile variants, MLP down tile variants,
  current 3B `lm_head_top1`, post-token-dim-pack 0.5B decode-attention
  `no_split_kv_threshold` variants including the fresh `192` probe, unstable
  0.5B MLP `gateup_tile=128x256x32_s4`, real-path 0.5B decode `lm_head`
  cublasLt retune, and standalone cuTile/FlashInfer CTA source-op replacements
  that fail correctness or lose to current FlashInfer.
- Accepted checkpoints:
  - 3B long-prefill table-driven
  `trt_context_fmha_plugin_attention` with pair-wise contiguous Q/KV packing,
  BF16-to-FP16 cast, and `contiguous_q_kv_min_seq_len=2048`. This is a
  plugin-op path, not a serialized TensorRT engine bridge.
  - 3B long-prefill shape-scoped SwiGLU launch retune:
    `hidden=2048|intermediate=11008|m=2048` uses the table-controlled
    2D-grid SwiGLU path with `swiglu_threads=64`, while other model sizes keep
    their prior launch policy.
  - 3B `m=2048` QKV source-op persistent FP16 weights with a
    `persistent_min_free_mb=64` guard. OProj persistence remains rejected/OOM
    prone.
  - Qwen2.5 0.5B/1.5B/3B long-prefill plugin-op attention records now enable
    `contiguous_q_kv_token_dim_pack`, which computes RoPE once per
    `(token, dim)` and loops over heads to shave sub-ms packing overhead.
  - Source-op table-selection messages are debug-only by default; true
    experimental env gates and fallback warnings still log at warning level.
  - 3B `m=1024` MLP uses `down_tile=128x128x32_warp32x64`, while 0.5B/1.5B
    `m=1024` MLP stays unchanged.
  - 0.5B and 1.5B decode attention use `chunk_alignment=32` and
    `chunk_candidates=[64,96,128,192]`; 3B decode attention stays on the
    previous `[64,128,192,256]` policy.
  - 0.5B `m=2048` MLP uses `activation_mode=mixed_bf16`, and 0.5B
    `m=2048` QKV source-op uses `input_mode=mixed_bf16`.
  - 3B `m=512` MLP uses `down_tile=128x128x32_warp32x64`.
- Active path: keep the all-slice `<=3 ms` checkpoint intact and continue
  targeting the few remaining positive slices, especially `0.5B 2048x32`,
  `1.5B 512x64`, and `1.5B 512x32`. Do not re-sweep current
  `lm_head_top1`, 0.5B prefill plugin CTA, or FlashInfer fallback knobs unless
  a fresh profile points back to them.

This plan supersedes the older 2026-05-09/2026-05-12 active work queue. Historical
results stay in [3060_tuning_log.md](./3060_tuning_log.md), but the active queue
below is the current source of truth for the next optimization rounds.

## Latest Active Decision

- 2026-05-18 tactical update:
  - Fresh graph-off mapping after the plugin-op attention checkpoint shows the
    largest runtime inside the 3B `S=2048` prefill path is now MLP/linear and
    conversion work, not the FMHA body alone:
    GateUp `~141.4 ms`, Down `~72.7 ms`, FMHA plugin kernel `~19.2 ms`,
    `swiglu_half2_kernel` `~14.5 ms`, `bf16_to_half2_kernel` `~6.1 ms`,
    and pair-wise Q/KV RoPE pack `~3.5 ms` across the profiled request.
  - Rejected follow-ups:
    RoPE inv-freq precompute regressed `3B 2048x32` to `926.137 ms`;
    MLP `activation_mode=mixed_bf16` was neutral/slower at `925.608 ms`;
    the initial global 1D SwiGLU block-size sweep was not stable enough for
    non-3B shapes; a SwiGLU `__expf` fast-math probe regressed to
    `924.893 ms`; QKV+OProj persistent weights OOMed.
  - Accepted follow-up:
    2D-grid SwiGLU plus table-controlled `swiglu_threads` made `64` threads the
    best same-run 3B/S2048 choice, and guarded QKV persistent weights moved
    `3B 2048x32` to `921.612 ms` (`prefill 295.720 ms`) with core generate and
    prefill-linear regression passing. Because QKV persistence leaves about
    `74.8 MB` free on the tested 3060, it is guarded and limited to the 3B
    `m=2048` QKV shape; OProj persistence stays off.
  - The current Stage-2 tactical gap is therefore `+5.101 ms` on
    `3B 2048x32`. This clears the "no 10+ ms gap" bar and is effectively at
    the relaxed `<=5 ms` checkpoint, but still worth one more low-risk pass.

- 2026-05-18 secondary-size update:
  - Extended table-driven `trt_context_fmha_plugin_attention` to the 0.5B and
    1.5B prefill attention shapes:
    `num_qo_heads=14|num_kv_heads=2|head_dim=64` and
    `num_qo_heads=12|num_kv_heads=2|head_dim=128`. This remains a plugin-op
    path, not a serialized TensorRT engine bridge.
  - Fixed the plugin-op fallback path so short sequences first reuse the
    BF16 FlashInfer pre-rotate implementation when RoPE is Llama-style and the
    head dimension is even. This keeps adjacent `512/1024` paths on the
    previously accepted fast path instead of silently falling back to the
    slower generic FlashInfer route.
  - Accepted paired results:
    - `0.5B 2048x32`: `174.166 -> 172.723 ms`, gap to TRT now `+4.873 ms`.
    - `0.5B 2048x64`: `300.653 -> 298.682 ms`, gap to TRT now `+7.030 ms`.
    - `1.5B 2048x32`: `474.041 -> 471.802 ms`, gap to TRT now `+2.993 ms`.
    - `1.5B 2048x64`: `809.992 -> 807.196 ms`, gap to TRT now `+4.006 ms`.
  - Rejected same-round candidates: 0.5B/1.5B decode-attention retune, 0.5B
    and 1.5B MLP tile sweeps, `prefill_max_mma_kv_cap=2` for those shapes,
    cublasLt LMHead algorithm records for fp32 logits, and default promotion
    of `lm_head_top1`.
  - Follow-up 0.5B decode recheck also rejects SM86 decode fused-SwiGLU for
    this round. It could be a large theoretical activation/GEMV fusion win, but
    the existing source records RTX 3060 numerical-equivalence risk, and simply
    opening the device gate triggers a very expensive TensorRT-LLM fused-MoE
    template rebuild before correctness can even be proven.
  - Validation after the source/table change:
    `cmake --build build-3060 -j$(nproc)`,
    `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`,
    profile/tuning script `py_compile`,
    `python3 -m pytest -q tests/operators/test_attention_prefill.py tests/operators/test_prefill_linear.py`,
    and the core Qwen generate regression slices.
  - Raw artifacts:
    `.tmp_codex/bench/stage2_20260518_official_plugin_attn_v2/`,
    `.tmp_codex/bench/stage2_20260518_0p5_adjacent_recheck/`,
    `.tmp_codex/nsys/stage2_20260518_other_sizes/`,
    `.tmp_codex/tune/stage2_20260518_0p5_decode_linear/`,
    `.tmp_codex/tune/stage2_20260518_lm_head/`.

- 2026-05-18 strided-QKV default update:
  - Default-enabled the correctness-clean BF16 no-prefix prefill strided-QKV
    attention runtime path and its side-stream KV copy. Both remain reversible
    through `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=0` and
    `EDGE_FM_PREFILL_STRIDED_QKV_SIDE_STREAM=0`.
  - Latest checked gaps versus external TRT-Edge-LLM:
    `0.5B 2048x32 +4.571 ms`, `0.5B 2048x64 +6.231 ms`,
    `1.5B 2048x32 +2.183 ms`, `1.5B 2048x64 +2.370 ms`.
  - 1.5B is now inside the requested `<=3 ms` band for both decode lengths.
    0.5B remains the active queue because `2048x64` still has a `+3.534 ms`
    decode gap on top of the `+2.642 ms` prefill gap.
  - Rejected and removed/kept default-off in the same pass: CUDA-graph
    response memcpy node (`+0.480/+0.983 ms` regression on 0.5B), BF16 decode
    logits (`+0.135/+0.465 ms` regression on 0.5B), SM86 decode fused-SwiGLU
    env gate (token alignment failure), 0.5B decode GateUp cublasLt explicit
    configs (current heuristic fastest), FlashInfer fallback instead of the
    0.5B plugin-op prefill attention (`+2.073/+2.246 ms` regression),
    decode-attention `no_split_kv_threshold=1024` (microbench win did not
    transfer to full generate), FP16 decode logits (1.5B cuBLASLt status `15`),
    and fused-K-RoPE copy under the current plugin-op attention contract.
  - Validation after cleanup: `build-3060`, operator table validation, profile
    script `py_compile`, and the core Qwen generate/sampler subset
    (`10 passed`).

- 2026-05-18 token-dim Q/KV pack update:
  - Added a table-gated contiguous Q/KV packing variant for the direct
    TRT-FMHA plugin-op attention path. It reduces repeated RoPE `powf/sincos`
    work by computing per `(token, dim)` and looping over heads.
  - Accepted on the 0.5B/1.5B/3B `S=2048` plugin-op records after correctness
    and end-to-end A/B checks. Largest checked wins are `3B 2048x32 -0.836 ms`,
    `1.5B 2048x32 -0.635 ms`, and `0.5B 2048x64 -0.458 ms` in reverse-order
    confirmation.
  - This is a small source-visible runtime improvement, not a new TensorRT
    engine bridge dependency. Continue with 0.5B decode/runtime and only
    reopen larger FMHA/GEMM Humanize loops when fresh profiling points to a
    concrete remaining operator gap.

- Stage 2 fresh profiling now overrides the older guesswork about the remaining
  TRT-Edge-LLM gap:
  - Paired graph-off mapping for `3B 2048x1` shows EdgeFM source-op prefill at
    `320.681 ms` versus external `TRT-Edge-LLM` at `289.819 ms`.
  - The largest fresh residuals are prefill attention (`+12.016 ms`) and
    BF16/FP16 boundary or pack overhead around QKV/OProj. Pure GEMM residuals
    still exist, but they are no longer the first Stage-2 target.
  - The trace shows `bf16_to_half2_kernel` at `9.930 ms` and
    `half2_to_bf16_kernel` at `2.389 ms` for this slice; reducing conversion
    boundaries is now an explicit optimization axis.
  - A 3B `S=2048` persistent-linear table probe improved graph-off `2048x1` by
    `3.215 ms`, but graph-on `2048x32` failed with persistent allocation
    pressure and cuBLASLt status `15`. It is rejected and must not be promoted
    to the 3060 table.
  - Default-off `plugin-op` attention with BF16-to-FP16 cast was rechecked on
    `3B 2048x32`: it reduced graph-on total by only `1.822 ms` versus paired
    native (`0.19%` end-to-end). The plugin body itself is faster, but
    BF16-pack/RoPE/output-cast overhead consumes most of the win.
  - This TRT-Edge-LLM checkout has SM86 FP16 context-FMHA cubins but no BF16
    context-FMHA cubins; do not promote a fake BF16 type toggle.
  - `activation_mode=mixed_bf16` for 3B MLP was also rechecked and rejected
    again (`935.835 ms` versus paired native `935.548 ms` on `2048x32`).
  - CUTLASS mixed-input source-op probes were expanded to QKV/OProj linear and
    MLP on the long-prefill matrix. After the hard `1%` gate was relaxed for
    localized, correctness-clean wins, `cutlass_prefill_linear_source_op`
    `input_mode=mixed_bf16` was promoted into the 3060 LLM table for the
    checked prefill linear shapes at `m=512/1024/2048`:
    - `2048x32` deltas: `0.5B -0.488 ms`, `1.5B -2.405 ms`,
      `3B -1.762 ms` total.
    - `512/1024x32` paired checks are also positive across the three model
      sizes, with no measured adjacent-shape regression.
    - A follow-up biased-QKV mixed path removes the QKV input cast when selected
      by table. It is accepted for 3B (`2048x32` total `-1.018 ms`, prefill
      `-1.263 ms`) and positive across 1.5B/3B adjacent checks:
      `1.5B 512x32 -1.026 ms`, `1.5B 1024x32 -0.539 ms`,
      `1.5B 2048x64 -0.179 ms`, `3B 512x32 -0.477 ms`,
      `3B 1024x32 -0.360 ms`, `3B 2048x64 -0.589 ms` total. 0.5B QKV stays
      `fp16_cast` because the same check regressed total by `+0.647 ms`.
    - Current table policy: 0.5B OProj stays `mixed_bf16`, 0.5B QKV stays
      `fp16_cast`, and 1.5B/3B QKV/OProj stay `mixed_bf16`.
    - MLP mixed BF16 input remains rejected/default-off: `0.5B/2048x32`
      improved by `-1.601 ms` (`~0.90%`), while `1.5B` and `3B` are tied or
      slower.
  - Post-acceptance `2048x32` comparison against external `TRT-Edge-LLM`:
    - `0.5B`: EdgeFM `173.6 ms`, TRT `166.5 ms`, total gap `+7.0 ms`;
      prefill gap `+5.2 ms`, decode gap `+1.8 ms`.
    - `1.5B`: EdgeFM `477.2 ms`, TRT `469.9 ms`, total gap `+7.4 ms`;
      prefill gap `+7.5 ms`, decode is tied.
    - `3B`: EdgeFM `934.7 ms`, TRT `917.2 ms`, total gap `+17.5 ms`;
      prefill gap `+21.3 ms`, decode is `3.8 ms` faster than TRT.
  - Low-risk positive changes no longer require a hard `1%` gate before being
    absorbed. The working rule is now: correctness first, then accept stable
    sub-1% wins when they are localized, shape-specific, and useful as a
    stepping stone for the next optimization round.
  - The first such absorbed Stage-2 step is pre-rotating BF16 Llama RoPE before
    FlashInfer prefill attention and running FlashInfer with `pos_encoding=None`.
    It is selected in the 3060 LLM table for the three concrete Qwen2.5 prefill
    attention shapes:
    - `0.5B 2048x32`: `177.974 ms -> 176.085 ms`, delta `-1.889 ms`.
    - `1.5B 2048x32`: `479.023 ms -> 476.601 ms`, delta `-2.421 ms`.
    - `3B 2048x32`: `938.262 ms -> 933.502 ms`, delta `-4.761 ms`.
  - Selective persistent FP16 weights for one 3B `S=2048` linear role at a time
    were also below the default gate:
    - QKV-only: `932.108 ms` total, `306.496 ms` prefill, but only `44.8 MB`
      free after runs.
    - OProj-only: `932.772 ms` total, `306.995 ms` prefill, `114.8 MB` free.
  - Production NCU for the active 3B FlashInfer prefill attention kernel
    confirms the next bottleneck is latency/occupancy/resource pressure rather
    than DRAM bandwidth:
    - duration `1.359520 ms` for the sampled layer
    - compute throughput `38.49%`, DRAM throughput `4.87%`, L2 throughput
      `32.38%`
    - issue slots busy `21.59%`, no eligible `78.41%`
    - theoretical occupancy `16.67%`, achieved occupancy `15.92%`
    - `168` registers/thread and `49.15 KB` dynamic shared memory per block
    - dominant production stall ratio: `math_pipe_throttle=4.336`
  - 0.5B decode follow-up did not produce a defaultable low-risk win:
    - decode-attention table sweep moved only from `0.028960 ms` to
      `0.028875 ms` average, effectively noise.
    - `lm_head_top1` sequential recheck improved `0.5B 2048x64` by
      `1.329 ms` total (`0.44%`), below the default promotion gate.
  - A later FlashInfer `NUM_MMA_KV` cap probe found a small, stable
      shape-specific prefill attention win for 3B. `prefill_max_mma_kv_cap=2`
      is now enabled only for the 3B prefill-prerotate attention shape:
      `3B 2048x32` moves by `-0.757 ms` total / `-0.648 ms` prefill on the
      10-run confirmation, with adjacent `512/1024` shapes still slightly
      positive on total time. The 0.5B 2048-only win (`-0.199 ms` total) is
      rejected for default use because `512x32` regresses by `+0.428 ms`; the
      1.5B shape also stays no-cap because the same probe was neutral/slightly
      slower (`+0.079 ms` total).
  - A default prefix-prefill correctness regression was found and fixed while
    retesting runtime dataflow candidates. Normal prefill attention again reads
    full `prefix + suffix` K/V, and the FlashInfer pre-rotate path receives a
    Q RoPE position offset for suffix tokens. Prefix/no-prefix CUDA graph token
    alignment is restored.
  - The strided-QKV runtime dataflow probe is now correctness-clean under CUDA
    graph after that fix. On `3B 2048x32` graph-on it moves from
    `934.046 ms` to `932.921 ms` total and from `307.249 ms` to `306.328 ms`
    prefill. Adjacent checks reject default promotion: `512x32 +0.543 ms`,
    `1024x32 +0.520 ms`, `2048x64 +1.328 ms` total. Keep it as an env-gated
    diagnostic only.
  - A fused K-copy + K-RoPE pre-rotate runtime probe
    (`EDGE_FM_PREFILL_FUSED_K_ROPE_COPY=1`) is correctness-clean and improves
    3B `2048x32`, but not enough to default:
    - first confirmation: `934.046 -> 930.596 ms`;
    - reversed-order confirmation: `934.502 -> 933.120 ms`;
    - adjacent checks regress slightly: `512x32 +0.447 ms`,
      `1024x32 +0.156 ms`, `2048x64 +0.159 ms`.
    Keep default-off; revisit only if a shape-specific runtime policy is added.
  - `lm_head_top1` was rechecked with the relaxed sub-1% rule. It remains
    default-off/full-logits-by-default, but is now a useful 0.5B decode tuning
    mode:
    - `0.5B 2048x64`: `301.058 -> 299.448 ms`, reversed
      `300.773 -> 299.922 ms`;
    - `0.5B 512x64`: `-0.329 ms`;
    - `0.5B 1024x64`: `-0.335 ms`;
    - `0.5B 2048x32`: `-0.654 ms`.
    1.5B/3B remain rejected for default because total time is neutral/slower.
  - Current default `2048x32` gap versus external TRT-Edge-LLM after the direct
    BF16/FP16 conversion kernel cleanup:
    - `0.5B`: EdgeFM `173.965 ms`, TRT `167.613 ms`, gap `+6.352 ms`
      (`+3.79%`);
    - `1.5B`: EdgeFM `474.902 ms`, TRT `468.809 ms`, gap `+6.093 ms`
      (`+1.30%`);
    - `3B`: EdgeFM `928.890 ms`, TRT `917.630 ms`, gap `+11.260 ms`
      (`+1.23%`).
    The 3B gap remains the main Stage-2 queue, but the residual is now closer
    to `~11 ms` rather than `~13 ms` on the official long-prefill target.
  - A narrow 3B attention table resweep found `cta128_cap0` as a short 3-run
    false positive (`929.941 ms`) but 10-run confirmation rejected it
    (`935.229 ms` versus current `932.168 ms`). Keep the current attention
    table policy; do not continue this CTA/cap parameter surface without new
    NCU evidence.
  - RoPE cache follow-ups are now exhausted as standalone production changes:
    - full sin/cos table cache regressed `3B 2048x32`
      (`935.281 ms` versus paired `934.441 ms`);
    - inv-freq cache passed correctness and showed small early movement, but a
      stronger no-inv paired table was faster (`932.718 ms` versus table-default
      inv-freq `933.660 ms` on `3B 2048x32`).
    - Both cache variants were removed from the production path. Future RoPE
      work must remove a real boundary, such as QKV split/write plus RoPE or
      RoPE inside a source-visible FMHA prelude.
  - MLP boundary follow-ups are now also exhausted for the current source-op
    shape:
    - prefill SwiGLU fusion is slower than the current 3B two-stage path
      (`2048`: `8.431 ms` fused versus `7.681 ms` current);
    - a narrower `mixed_gateup` diagnostic that only removed the MLP input cast
      before GateUp regressed `3B 2048x32` from `931.984 ms` total /
      `305.338 ms` prefill to `933.848 ms` total / `306.860 ms` prefill, so it
      was removed rather than kept as a default-off implementation.
    - The 1.5B `m=2048` MLP tile split does not transfer to the 3B
      `hidden=2048|intermediate=11008` shape. A `gateup_tile=128x256x32`
      probe showed only noise (`930.338 ms` versus paired current
      `930.205 ms` on `3B 2048x32`), while down-tile split variants regressed
      prefill by roughly `3.4-4.2 ms`.
  - Runtime weight prefetch for nonpersistent 3B QKV/OProj source-op records is
    rejected and removed from the production path. It moved `3B 2048x32` from
    `932.192 ms` to `932.366 ms` and `3B 2048x1` from `306.664 ms` to
    `307.080 ms`.
  - Direct BF16/FP16 conversion kernels are accepted. Replacing BF16->float->FP16
    and FP16->float->BF16 conversion bodies with direct CUDA BF16/FP16
    constructors keeps token alignment green and moves the 3B official target
    to `928.890 ms` total / `303.043 ms` prefill. Adjacent `512/1024x32`
    checks were non-regressing in the current artifact set.
  - A table-driven TRT-FMHA plugin-op attention checkpoint is accepted for the
    3B long-prefill shape. Implementation details:
    `trt_context_fmha_plugin_attention` now reads `impl_params` during
    `supports()`, and the 3060 table enables `allow_bf16_fp16_cast`,
    `contiguous_q_kv`, and `contiguous_q_kv_min_seq_len=2048` only for
    `num_qo_heads=16|num_kv_heads=2|head_dim=128` prefill. The path calls
    the TRT-Edge-LLM `ContextFMHARunner` as an operator/plugin runner and does
    not use serialized TensorRT engines or execution contexts. Confirmed
    graph-on results:
    - `3B 2048x32`: `925.432 ms` total / `299.124 ms` prefill versus TRT
      `916.511 ms`, gap `+8.921 ms`.
    - `3B 2048x64`: `1572.824 ms` total / `300.061 ms` prefill versus TRT
      `1568.463 ms`, gap `+4.362 ms`.
    - Adjacent short shapes stay within the tactical band:
      `512x32 686.910 ms` versus TRT `686.297 ms`, and `1024x32
      761.862 ms` versus TRT `757.664 ms`.
    This closes the `10 ms+` residual for the checked 3B long-context matrix;
    the next tactical queue is to push `2048x32` from `+8.92 ms` to `<=5 ms`.
  - Fresh post-direct-conversion graph-off attribution still shows conversion
    and residency boundaries, but not enough to justify risky memory knobs:
    `3B 2048x1` prefill model is `312.322 ms`, with CUTLASS MLP/linear GEMMs
    at `248.360 ms`, FlashInfer attention at `28.293 ms`, SwiGLU at
    `14.431 ms`, RoPE at `2.444 ms`, and visible BF16/FP16 conversion kernels
    at `8.465 ms`.
  - Single-role persistent 3B linear weights are rejected for default use after
    the current-table recheck. OProj-only is a tiny `0.466 ms` total win on
    `3B 2048x32`, but leaves only `122.8 MB` free; QKV-only regresses total
    and leaves only `52.8 MB` free.
  - A 3B FlashInfer attention cap resweep is also rejected for default use.
    The layer microbench favored `prefill_max_mma_kv_cap={0,4,8}`, but full
    graph-on generate regressed: `cap=8` gives `929.625 ms`, `cap=0` gives
    `930.722 ms`, versus the current `cap=2` baseline `928.890 ms`.
  - Immediate queue: stop promoting large memory-residency/cast knobs for 3B
    default. Small table-local conversion-boundary wins may still be absorbed
    when paired checks are consistently positive, as with linear
    `mixed_bf16`. The bigger remaining gap should be attacked through
    source-visible boundary work and BF16 prefill attention replacement in the
    Humanize / KernelPilot long loop, not more scalar attention seeds or
    standalone RoPE caches. Keep 0.5B decode as a secondary queue item.

- The active optimization plan is now split into three parallel tracks. Execute
  them in this order unless fresh profile evidence changes the queue:

  ### Track 1: Runtime Scheduling And Dataflow

  Goal: reduce critical-path dependencies without changing model math.

  - R1: Strided-K/V attention input plus delayed KV-cache write.
    - Current path: QKV GEMM writes packed `[Q,K,V]`; prefill immediately copies
      K/V into KV cache, then attention reads from the cache.
    - Status: correctness-clean after the prefix KV/RoPE prefill fix. The
      current no-prefix runtime gate improves `3B 2048x32` graph-on by about
      `-1.1 ms` total / `-0.9 ms` prefill on a 10-run paired check.
    - Conclusion: keep default-off while collecting adjacent-shape evidence.
      Promote only if `512/1024/2048` and `decode=32/64` checks are stable and
      the benefit survives reversed-order measurement.
  - R2: Side-stream RoPE/pre-rotate scheduling.
    - Current absorbed path pre-rotates Q/K before no-RoPE FlashInfer attention
      and already moves `0.5B/1.5B/3B 2048x32` by `-1.9/-2.4/-4.8 ms`.
    - Candidate: run Q and K pre-rotate on independent streams or fuse the
      pre-rotate with the QKV split/write path.
    - Status: simple Q/K pre-rotate kernel coalescing was measured and
      rejected. It reduced one launch but made `3B 2048x32` slower
      (`934.093 ms` total, `307.894 ms` prefill) versus the paired unfused
      baseline (`932.764 ms` total, `306.816 ms` prefill).
    - Status: side-stream K pre-rotate was also measured and rejected. It was
      tied with the same-batch default (`934.105 ms` versus `934.034 ms`) and
      adds event/stream complexity without a measurable gain.
    - Status: cached RoPE sin/cos tables for the existing pre-rotate kernels
      were measured and rejected. The gated path passed operator and generate
      smoke tests, but `3B 2048x32` slowed from the paired default
      `934.441 ms` total / `308.200 ms` prefill to `935.281 ms` total /
      `308.774 ms` prefill. The rejected code was removed after validation.
    - Status: cached RoPE inv-freq tables were also measured and rejected. A
      small env-gated gain did not survive a stronger no-inv paired table check;
      the no-inv table ran `3B 2048x32` at `932.718 ms` total / `306.762 ms`
      prefill versus `933.660 ms` total / `307.644 ms` prefill with table
      default inv-freq enabled. The rejected code and table params were removed.
    - Status: combining the production Q and K pre-rotate launches into one
      standalone kernel was measured across 3B/1.5B/0.5B and rejected. It only
      helped 3B by `0.281 ms` total in the paired run, was noise on 0.5B, and
      regressed 1.5B `2048x32` by `+1.466 ms` total. The rejected branch was
      removed after correctness checks.
    - Status: fusing K-cache copy with K RoPE pre-rotation was measured behind
      `EDGE_FM_PREFILL_FUSED_K_ROPE_COPY=1` and kept default-off. It passed the
      core generate subset and moved 3B/1.5B by only `-0.168/-0.347 ms` total,
      with 0.5B at noise (`+0.056 ms` total).
    - Status: coarse fused QKV split plus Q/K RoPE was measured across
      3B/1.5B/0.5B and rejected after rollback validation. It was only a
      noise-level win on 3B (`-0.199 ms` total) and regressed 1.5B/0.5B
      (`+0.586/+0.313 ms` total). This closes the current scalar
      split/rotate/copy fusion line.
    - Estimate: stream-only overlap is limited because attention must wait for
      both Q and K; expected standalone gain is likely `0.5-2 ms` for 3B.
      Plain Q/K kernel coalescing and scalar QKV/RoPE boundary fusion are not
      enough; the next fusion candidate must move into source-visible FMHA or a
      FlashInfer internal prelude where it can reduce resource stalls and
      scratch traffic together.
  - R3: Host/runtime overhead trimming.
    - Current prefill prepare host time is already tiny in the profile JSON, and
      graph-off D2D copies are below `1 ms`.
    - Estimate: low ceiling unless a fresh NSYS trace shows launch or sync
      bubbles outside the CUDA graph. Keep this as a verification check rather
      than the first implementation target.

  ### Track 2: Operator Fusion, Early/Late Compute, And Boundary Removal

  Goal: remove boundary kernels and data-format churn around hot operators.

  - F1: Fuse Q/K Llama RoPE with packed-QKV split or with a source-visible FMHA
    prelude.
    - This is the stronger version of the accepted pre-rotate attention step.
    - Expected ceiling: the accepted unfused pre-rotate table already gives
      `-4.6 ms` prefill on 3B `2048x32`; fusion should aim for another
      `1-4 ms` if it removes extra launches/scratch traffic.
    - Current reject note: merging the standalone Q and K RoPE kernels into one
      standalone kernel was not stable across model sizes, so future F1 work
      should avoid launch-only fusion and instead eliminate scratch traffic or
      conversion boundaries.
    - Current reject note: table-caching RoPE angles was also slower, so the
      next RoPE attempt should not be a standalone lookup-table variant. It
      needs to be fused into QKV split/write or FMHA prelude work to reduce a
      real boundary.
    - Current diagnostic note: K-copy + K-RoPE fusion is correct and slightly
      positive on larger models, but too small to default by itself. Treat it as
      evidence for a larger QKV split/write + Q/K RoPE fusion, not as the final
      production shape.
    - Current reject note: a coarse fused QKV split plus Q/K RoPE kernel did
      not become that larger win; it barely helped 3B and regressed smaller
      models. Future F1 work should be scoped inside FMHA/source-op prelude
      work rather than as another standalone scalar boundary kernel.
  - F2: Fuse or delay residual copies only if profile evidence shows real copy
    cost.
    - Current D2D memcpy total is `0.630 ms` in `3B 2048x1`, so broad copy-fusion
      work is lower priority than attention/format boundaries.
  - F3: Recheck `lm_head_top1` and sampler/finalize only after prefill work.
    - Decode is tied or faster than TRT for 1.5B/3B long-prefill shapes.
    - Current `lm_head_top1` recheck remains inconsistent: it helps 0.5B
      `2048x32` by `-0.675 ms`, but regresses 0.5B `2048x64` by `+0.632 ms`
      and 3B `2048x64` by `+0.732 ms`. Keep it default-off.
    - Keep decode ideas recorded, but do not let them steal the next prefill
      iteration unless a new matrix shows decode becoming the blocker.

  ### Track 3: Operator Tuning With `edge-fm-cuda-kernel-optimizer`

  Goal: use NSYS/NCU evidence to tune or replace the remaining slow operators.

  - O1: Continue attention work through the Humanize / KernelPilot loop.
    - New CTA matrix check is closed: `prefill_cta_tile_q=64` remains fastest
      for Qwen2.5 `0.5B/1.5B/3B` at `512/1024/2048`; `16` and `128` both
      regress. Do not spend more queue time on plain FlashInfer CTA table
      sweeps for the current pre-rotate path.
    - FlashInfer `trtllm-gen` FMHA is not a viable 3060 plugin-op route in this
      checkout. A Qwen-like BF16 context probe can reach `TllmGenFmhaRunner`,
      but runtime fails on RTX 3060 / `sm_86` with `Unsupported architecture`.
    - A localized FlashInfer `NUM_MMA_KV` cap knob is accepted only as a
      shape-specific table tweak, not as a general FMHA solution. Keep
      `prefill_max_mma_kv_cap=2` for 3B prefill-prerotate attention; keep
      0.5B and 1.5B no-cap because 0.5B adjacent `512x32` regresses and 1.5B
      is neutral/slightly slower.
    - 0.5B MLP `activation_mode=mixed_bf16` was rechecked against the current
      accepted linear-mixed table and remains rejected: `512x32` is noise,
      `1024x32` only moves total by `-0.027 ms`, and `2048x32` regresses by
      `+0.780 ms`.
    - Current production NCU shows low occupancy and `math_pipe_throttle` on
      FlashInfer prefill attention, not DRAM saturation.
    - Candidate outputs: source-visible BF16 FMHA, fused RoPE+FMHA, or a
      production-safe plugin-op that does not require TensorRT engine runtime.
    - Immediate next action: use the existing
      `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516` workspace to
      turn the FlashInfer-vs-TRT attention gap into a source-visible FMHA
      experiment, with NCU evidence before any production code change.
  - O2: Only revisit GEMM when a fresh operator gap table points to it.
    - Ordinary cublasLt/CUTLASS 2.x table churn has repeatedly plateaued.
    - Accept only measured end-to-end wins, including sub-1% wins when they are
      stable and localized.
  - O3: Keep the 3060 table shape-specific.
    - Different model sizes and stages may choose different source-op configs.
    - Every table change needs an artifact path, paired benchmark, and closest
      correctness test before commit.

- Stage 1 is now implemented as actual source cleanup, not just a planning
  decision:
  - EdgeFM Qwen2.5 no longer builds or calls the internal TensorRT engine
    prefill bridges `trt_mlp_bridge` and `trt_linear_bridge`.
  - The source-op CUTLASS/CUDA path is the maintained EdgeFM path for Qwen2.5
    prefill linear/MLP work.
  - The remaining CUTLASS source-op code is no longer model-level bridge code:
    QKV/OProj is selected through `op_kind=linear` with
    `impl_id=cutlass_prefill_linear_source_op`, and MLP is selected through
    `op_kind=mlp` with `impl_id=cutlass_prefill_mlp_source_op`.
  - `qwen2_5.cpp` should not call source-op kernels directly. Future custom
    Qwen prefill operators should enter through the layer/operator registry and
    the 3060 operator table, not through ad hoc model members.
  - External `TRT-Edge-LLM` remains the Stage-2 benchmark reference. Direct
    source-visible/plugin-op assets can still be evaluated when they do not
    require serialized TensorRT engines or TensorRT execution contexts.

- Stage 2 fresh long-prefill baseline versus external `TRT-Edge-LLM` is now
  available for `S=2048`:
  - `0.5B 2048x32`: EdgeFM `177.878 ms`, TRT-Edge-LLM `167.850 ms`,
    gap `+10.028 ms` / `+5.97%`.
  - `0.5B 2048x64`: EdgeFM `303.503 ms`, TRT-Edge-LLM `292.060 ms`,
    gap `+11.443 ms` / `+3.92%`.
  - `1.5B 2048x32`: EdgeFM `478.075 ms`, TRT-Edge-LLM `467.580 ms`,
    gap `+10.495 ms` / `+2.24%`.
  - `1.5B 2048x64`: EdgeFM `813.252 ms`, TRT-Edge-LLM `803.190 ms`,
    gap `+10.062 ms` / `+1.25%`.
  - `3B 2048x32`: EdgeFM `936.562 ms`, TRT-Edge-LLM `916.511 ms`,
    gap `+20.051 ms` / `+2.19%`.
  - `3B 2048x64`: EdgeFM `1584.191 ms`, TRT-Edge-LLM `1568.463 ms`,
    gap `+15.729 ms` / `+1.00%`.
  - Active decision: start Stage 2 with prefill attribution, especially
    `3B 2048x32`; decode is not the primary blocker for 1.5B/3B because EdgeFM
    decode is tied or faster there. Keep 0.5B decode in the follow-up queue.

- Stage 1 bridge-removal work is now closed by decision:
  - Source-op is already close enough to true `trt_bridge` for the tested 3060
    matrix. The only remaining material gap is 1.5B long prefill:
    `+3.213 ms` on `2048x32` and `+3.396 ms` on `2048x64` versus the valid
    true-bridge baseline.
  - Round AQ attributed the residual to scattered GEMM/linear tactics
    (GateUp/Down/QKV/OProj), not attention or decode.
  - Round AR/AS/AT rejected the remaining low-risk source-op table/code probes:
    linear warp-shape variants, MLP `down_output=fp16_cast`, and linear base
    tile rechecks.
  - Product decision: do not spend more Stage-1 time chasing the last few
    milliseconds. Remove the internal TensorRT engine bridge from EdgeFM
    Qwen2.5 and move active work to Stage 2.

- Stage 2 active queue:
  - Re-profile EdgeFM source-op/current-best versus external `trt-edge-llm` on
    0.5B/1.5B/3B, prefill `512/1024/2048`, decode `32/64`, CUDA graph on.
  - Prioritize end-to-end wins against `trt-edge-llm`, not just true bridge.
    Decode can be optimized first if fresh profiling shows a larger or easier
    gap than prefill.
  - Keep the active vocabulary narrow: `native/source-op`, optional
    `plugin-op`, and external `trt-edge-llm` reference.
  - Escalate to Humanize/KernelPilot only for kernels whose fresh Stage-2 gap
    offers at least `>=1%` end-to-end potential and has a reliable standalone
    correctness/benchmark path.

- Round AR tested a low-risk linear source-op extension before escalating to a
  heavier custom GEMM loop:
  - Temporarily added MLP-style warp-shape CUTLASS tile values to
    `CutlassPrefillLinearBridge` and swept QKV/OProj on `1.5B S=2048`.
  - Graph-off `2048x32` prefilter found no win. The least-bad candidate,
    OProj `128x128x32_warp32x64`, still regressed prefill by `+0.120 ms`; all
    QKV variants regressed by at least `+0.482 ms`.
  - CUDA-graph confirmation of that least-bad OProj candidate also regressed:
    `2048x32` `478.448 -> 478.822 ms` and `2048x64`
    `814.005 -> 814.101 ms`.
  - Decision: reject and revert the temporary linear warp-shape production
    code. Do not add these tile modes to the official table. QKV/OProj are not
    solved by the same CUTLASS 2.x warp-shape route that helped MLP down.

- Round AQ replaced the ambiguous bridge-removal baseline with a valid
  true-bridge rebaseline and serial NSYS role attribution:
  - Valid true bridge requires `build-3060-trt-mlp-release`, the temporary
    bridge-only operator table
    `.tmp_codex/bench/3060_20260517_roundaq_bridge_only_operator_table.json`,
    TensorRT `LD_LIBRARY_PATH`, and `EDGE_FM_PREFILL_TRT_MLP=1` /
    `EDGE_FM_PREFILL_TRT_LINEAR=1`.
  - Current source-op still trails true bridge on the only material Stage-1
    blocker: `1.5B 2048x32` by `+3.213 ms` total / `+2.985 ms` prefill and
    `1.5B 2048x64` by `+3.396 ms` total / `+3.163 ms` prefill.
  - Plugin-op attention narrows the gap but does not close it:
    `2048x32` remains `+2.306 ms` total behind true bridge, and `2048x64`
    remains `+1.352 ms` behind.
  - Serial graph-off mapping attributes the remaining source-op residual to
    GEMM/linear roles: GateUp `+1.243 ms`, Down `+0.686 ms`, QKV `+0.447 ms`,
    and OProj `+0.300 ms`. Prefill attention is effectively tied and SwiGLU is
    slightly faster than bridge.
  - Active decision: stop treating attention/decode as the Stage-1 blocker.
    Continue with source-visible GEMM/linear candidates for 1.5B long prefill.

- Round AP tested a tiny plugin-op attention optimization that cached the
  TRT-Edge-LLM `ContextFMHARunner` RoPE table instead of rebuilding it inside
  the QKV pack kernel on each prefill layer:
  - Build passed, but serial 1.5B `S=2048` CUDA-graph benchmarks did not improve
    the Round AO plugin-op baseline.
  - `2048x32`: Round AO plugin-op `477.311 ms` total / `152.485 ms` prefill;
    RoPE-cache candidate `477.365 ms` total / `152.377 ms` prefill.
  - `2048x64`: Round AO plugin-op `811.566 ms` total / `151.750 ms` prefill;
    RoPE-cache candidate `812.460 ms` total / `152.539 ms` prefill.
  - Decision: reject and revert the `attention_op.cu` cache code. The useful
    plugin-op evidence remains Round AO; it is still default-off and diagnostic.

- Round AO rechecked the source-visible TRT-Edge-LLM attention plugin-op on the
  only remaining Stage-1 blocker, 1.5B `S=2048`:
  - The path uses `ContextFMHARunner` directly as an EdgeFM operator, not a
    serialized TensorRT engine. Because this checkout ships only FP16 FMHA
    cubins in `fmha_cubin.h`, BF16 Q/K/V still need the diagnostic
    BF16-to-FP16 pack/cast path.
  - `2048x32`: native source-op `478.218 ms` total / `153.491 ms` prefill;
    plugin-op `477.311 ms` total / `152.485 ms` prefill.
  - `2048x64`: native source-op `813.611 ms` total / `153.932 ms` prefill;
    plugin-op `811.566 ms` total / `151.750 ms` prefill.
  - Decision: reject as an official default/table change. It helps, especially
    on `2048x64`, but remains below the `>=1%` gate and still does not clearly
    beat true `trt_bridge` for 1.5B long-prefill. A third-party patch that merely
    flips BF16 support in `ContextFMHARunner::canImplement` is not sufficient;
    BF16 cubin generation would be a separate source-kernel project, and prior
    BF16 FMHA generator evidence was slower than current FlashInfer.

- Round AN refined the third-party/CUTLASS mixed-input route:
  - Round AM was correct that SM86 has no native floating-point
    `BF16 x FP16` tensor-core MMA instruction exposed as a direct arch-MMA
    primitive. However, CUTLASS already provides
    `arch::OpMultiplyAddMixedInputUpcast`, which can compile a mixed
    `BF16 activation x FP16 weight` path by upcasting the narrower operand
    before issuing a supported MMA.
  - Standalone MLP harness results for `cutlass_mixed_bf16out_candidate`
    are positive: `0.5B/S2048` `1.66730 -> 1.61782 ms`, `1.5B/S2048`
    `4.27256 -> 4.18618 ms`, and `3B/S2048` `6.62480 -> 6.53717 ms`.
  - Production end-to-end generate rejects the route despite the standalone
    win. With `activation_mode=mixed_bf16`, CUDA-graph native `2048x32`
    regressed on all tested sizes: `0.5B` `177.176 -> 177.773 ms`,
    `1.5B` `479.309 -> 480.043 ms`, and `3B` `937.131 -> 937.802 ms`.
  - Decision: keep the production support default-off as a diagnostic knob, but
    do not add `activation_mode=mixed_bf16` to the official 3060 operator table.
    The official table has no accepted mixed-activation record. This route does
    not close the remaining 1.5B `S=2048` Stage-1 bridge-removal blocker.

- Round AM checked whether third-party CUTLASS source modifications can unlock
  the remaining 1.5B `S=2048` source-op gap:
  - A narrow no-reference CUTLASS profiler for SM86 `h16816gemm` TN kernels now
    builds and enumerates 45 kernels.
  - Profiler-only GateUp and Down candidates did not translate into
    backportable generate wins. `down_tile=default` was revalidated on real
    generate and regressed both `2048x32` and `2048x64`; GateUp
    `256x128x32` had already been rejected by Round AJ real-generate data.
  - A temporary CUTLASS mixed-input probe for
    `BF16 activation x FP16 weight -> FP16 output` failed at the arch-MMA layer.
    PTX evidence shows this is not just a missing CUTLASS wrapper:
    SM80/SM86 tensor-core MMA supports FP16 x FP16 and BF16 x BF16, not a
    general BF16 x FP16 floating-point MMA path.
  - Active decision: third-party source modifications are allowed, but only
    when they expose or adapt a real supported primitive. For the current
    1.5B/S2048 residual, ordinary CUTLASS 2.x profiler/table work is closed.
    The next branch is either a shape-local bridge fallback removal plan, or a
    deeper custom CUDA/CuTe GEMM effort with a clear cost/benefit gate.

- Round AL swept the remaining smaller 1.5B `S=2048` source-op residuals and
  did not find an acceptable table backport:
  - MLP down standalone harness found `128x128x64_warp64x32` only
    `~0.003 ms/layer` faster than the current `128x128x32_warp32x64`, and
    prior end-to-end Round AI already beat the older down tile. Reject.
  - QKV `128x256x32` improved `2048x32` by `0.34 ms` total but regressed
    `2048x64` by `0.21 ms`; below the acceptance gate. Reject.
  - OProj default/current remains best for the target shape. Reject changes.
  - KernelPilot/source scan of local `third_party/TensorRT-Edge-LLM` found
    attention FMHA/XQA and INT4 groupwise GEMM plugin assets, but no standalone
    dense FP16/BF16 GEMM plugin/kernel for the current unquantized Qwen MLP or
    QKV/OProj path. The remaining bridge advantage is still a TensorRT engine
    Myelin/XMMA tactic, not a directly reusable plugin-op in this checkout.
  - CUTLASS CuTe DSL quick check found the local Python DSL package lacks the
    built `_mlir` module. A CuTe DSL attempt is possible, but it is a separate
    toolchain/standalone-kernel project rather than a same-day table tweak.
  - Active decision: ordinary source-op table/CUTLASS 2.x tuning is now at
    plateau for Stage-1. Either start a deeper custom CuTe/CUDA GEMM effort for
    the final few milliseconds, or plan staged `trt_bridge` removal with a
    shape-local fallback for 1.5B `S=2048`.

- Round AK closed the first 1.5B `S=2048` gate-up Humanize pass without a
  backportable win:
  - Standalone workspace:
    `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/`.
  - The harness compared current CUTLASS `128x256x32_s3` against bridge-like
    `256x128x32`, default `128x128x64`, `128x128x32` warp-shape variants,
    stage4, `64x256x32`, K64 candidates, threadblock swizzle variants, and
    `torch.mm` / cuBLAS.
  - Current remains the fastest stable candidate found here:
    `~2.46 ms/layer` standalone median. `GemmIdentityThreadblockSwizzle<4>`
    was only noise-level (`0.00064 ms/layer` median faster in round-robin, mean
    slightly slower), so it is rejected.
  - Full NCU shows the source-op gate-up is already tensor-pipe/math-pipe heavy:
    compute throughput `90.35%`, SM busy `90.77%`, DRAM throughput `36.21%`,
    and sampled stalls dominated by math-pipe throttle.
  - Decision: stop ordinary CUTLASS 2.x table/stage/swizzle probing for this
    gate-up shape. Do not backport a gate-up change from this round.
  - Active next work for Stage-1 bridge removal: target the remaining
    non-gate-up residuals in order, MLP down (`+0.780 ms`), QKV (`+0.471 ms`),
    and OProj (`+0.315 ms`), or evaluate a source-visible TRT-Edge-LLM
    plugin/kernel asset that does not require TensorRT engine/runtime.

- Round AJ fresh profiling narrowed the remaining 1.5B `S=2048` bridge-removal
  gap:
  - Current Round AI source-op versus true bridge on graph-off `2048x1` mapping:
    gate-up GEMM is `+1.348 ms`, MLP down is `+0.780 ms`, QKV is `+0.471 ms`,
    OProj is `+0.315 ms`, and prefill attention is essentially tied
    (`+0.029 ms`).
  - A targeted gate-up table recheck rejected `gateup_tile=256x128x32`
    (`+0.791 ms` total on `2048x32`) and `gateup_tile=default`
    (`+4.956 ms` total).
  - Superseded by Round AK: the first gate-up Humanize pass did not find a
    stable backportable CUTLASS 2.x win.
  - Humanize workspace is seeded at
    `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/`, with a
    standalone harness, baseline NCU digest, and rejected-attempt ledgers for
    `M=2048,K=1536,N=17920`.

- Round AI accepted a marginal 1.5B long-prefill table tie-breaker:
  - The official 1.5B `m=2048|hidden=1536|intermediate=8960` MLP source-op
    record now uses `down_tile=128x128x32_warp32x64` with
    `tile=256x128x32`, `gateup_tile=128x256x32`, and
    `persistent_weights=true`.
  - Fresh 5-run results: `2048x32` is `478.372 ms` total
    (`153.820 ms` prefill) and `2048x64` is `813.229 ms` total
    (`153.757 ms` prefill).
  - This is only a small improvement over Round AH, but it wins both long
    prefill slices and is shape-local.
  - True `trt_bridge` remains faster: the residual is about `+2.8-3.3 ms`
    total/prefill. This is still the only material Stage-1 bridge-removal
    blocker.
  - Active next work: stop ad-hoc table-mode expansion for 1.5B `S=2048`.
    Move to NCU-backed source-op/Humanize or plugin-op investigation for the
    exact remaining GEMM tactic gap.

- Round AH accepted a second 1.5B long-prefill source-op table refinement:
  - `CutlassPrefillMlpBridge` now exposes two diagnostic `128x128x64` warp-shape
    variants for the MLP down projection. The accepted 1.5B `S=2048` table uses
    `down_tile=128x128x64_warp64x32` with the existing
    `tile=256x128x32`, `gateup_tile=128x256x32`, and
    `persistent_weights=true`.
  - Fresh 5-run source-op results: `2048x32` is `478.478 ms` total
    (`153.868 ms` prefill) and `2048x64` is `813.675 ms` total
    (`153.944 ms` prefill).
  - Compared with the immediate official source-op rerun, this removes about
    `1.9-2.4 ms` total and about `2.1-2.3 ms` prefill on the two long-prefill
    slices.
  - Superseded by Round AI, which found a marginally faster `128x128x32`
    stage4 warp-shape down tile.

- Round AG accepted a corrected `records`-based 0.5B short-prefill source-op
  table update:
  - Temporary benchmark tables must edit the current `records` field; older
    `entries`-based ad-hoc sweeps are invalid diagnostics and should not be
    used for decisions.
  - For 0.5B `m=512|hidden=896|intermediate=4864`, the official table now uses
    `tile=256x128x32` with `down_tile=128x128x32`.
  - Fresh source-op results are now `12.77 ms` prefill on `512x32` and
    `12.59 ms` on `512x64`, versus true `trt_bridge` `12.85 ms` and
    `12.73 ms`. End-to-end throughput is effectively tied with bridge.
  - 0.5B `S=512` is no longer a Stage-1 bridge-removal blocker.
  - True bridge references must run with `--edgefm-mode as-is`; `native`
    explicitly clears bridge env vars and is not a bridge baseline.
  - Superseded by Round AH for 1.5B `S=2048`: a later warp-shape candidate
    improved the source-op table but did not close the true-bridge gap.

- Round AF accepted a narrow 1.5B long-prefill MLP source-op table refinement:
  - `CutlassPrefillMlpBridge` now supports optional `gateup_tile` and
    `down_tile` params, falling back to the existing `tile` value for all older
    records.
  - For 1.5B `m=2048|hidden=1536|intermediate=8960`, the Round AF accepted
    table config was `tile=256x128x32`, `gateup_tile=128x256x32`,
    `down_tile=default`, `persistent_weights=true`. Round AH later refined only
    the down tile.
  - Official table-driven rerun improved 1.5B `2048x32` from `482.383 ms`
    to `478.282 ms`, and `2048x64` from `817.800 ms` to `813.844 ms`.
  - The valid `trt_bridge` reference is still ahead: `475.482 ms` for
    `2048x32` and `810.713 ms` for `2048x64`. Remaining residual is now about
    `+2.8-3.1 ms`, down from `+6.9-7.1 ms`.
  - Active next work: profile the remaining 1.5B `S=2048` source-op residual at
    role/kernel level. Avoid broad repeated CUTLASS tile sweeps unless the
    profile points to a specific GEMM shape or cast/residency bottleneck.

- Round AE changed the immediate bridge-removal gate from "beat TRT-Edge-LLM"
  to the two-stage target requested on 2026-05-16:
  - Stage 1: remove EdgeFM's TensorRT engine `trt_bridge` only when the
    no-engine source-op path catches or beats the current EdgeFM `trt_bridge`
    on the same model/shape.
  - Stage 2: after that, keep optimizing against TRT-Edge-LLM as the external
    reference.
  - CUTLASS prefill MLP and QKV/OProj source-op bridges now read exact
    `model + hw_profile + stage + shape_sig` records from the 3060 operator
    table via the bridge-only op kinds `qwen_prefill_mlp_bridge` and
    `qwen_prefill_linear_bridge`. These records do not replace normal
    `linear` table selection.
  - The table-driven source-op matrix improved from the previous `5/18` to
    `11/18` shapes at or faster than `trt_bridge`. 3B is now ahead on all
    tested shapes, including `2048x32` (`936.768 ms` versus `943.426 ms`) and
    `2048x64` (`1583.752 ms` versus `1590.451 ms`).
  - `trt_bridge` is still not globally removable. Remaining blockers are
    1.5B long-prefill (`2048x32` is `+6.901 ms`, `2048x64` is `+7.087 ms`)
    and smaller 0.5B residuals, especially `512x32` (`+1.725 ms`).
  - The fastest stable 3B `S=2048` table requires MLP persistent FP16 weights
    but scratch QKV/OProj weights; enabling all three persistent paths OOMs on
    RTX 3060. Keep this shape-specific memory boundary in the table and do not
    generalize persistent weights blindly.
  - Active next work: target 1.5B `S=2048` prefill first. If table/source-op
    cannot close the `~7 ms` residual, keep bridge only for that shape while
    removing it for accepted 3B shapes in a later staged rollout.

- Round AC re-ran the existing scalar query-block FMHA seed after NCU access was
  restored:
  - 3B `S=32` correctness still passes (`max_abs=0.015625`), but long-sequence
    performance remains noncompetitive: `S=512 5.620 ms`, `S=2048 76.095 ms`.
  - This confirms the scalar query-loop branch is closed. Future FMHA work must
    start from a tensor-core/source-visible schedule, a fused
    QKV-pack/RoPE/FMHA design, or a correctness-preserving plugin/source asset.

- Round AB rejected native BF16 CUTLASS MLP as a cast-removal route:
  - Standalone correctness is clean: 0.5B `S=128` smoke passes with
    `max_abs_error=2.86e-6`, and 3B `S=2048` candidates pass BF16 reference
    comparison.
  - Performance is not viable on RTX 3060: current FP16-weight source-op seed is
    `6.611 ms` on 3B `S=2048`, while native BF16 CUTLASS candidates are
    `11.831-12.239 ms`.
  - This rules out replacing FP16 source-op GEMMs with BF16 GEMMs just to remove
    activation/weight casts. The active cast direction should instead focus on
    reducing duplicate conversions or improving activation/weight residency
    while keeping the fast FP16 Tensor Core path.

- Round AA rejected the generated FlashInfer/TRT-LLM FMHA v2 SM86 BF16 route:
  - The generator can emit Qwen-like SM86 BF16 contiguous-Q/KV causal kernels,
    and the temporary probe passes 3B `S=32` no-RoPE correctness.
  - Performance is not competitive: 3B `S=2048` non-tiled is
    `1.567664 ms/layer` and tiled is `1.723360 ms/layer`, slower than the
    current FlashInfer no-RoPE core (`0.798592-0.851712 ms/layer`) and fused
    RoPE path.
  - Generated source headers are not source-op migration friendly, so this
    path is rejected for both performance and provenance.
  - Active prefill attention direction should skip generated FMHA v2 and move
    to either fused QKV-pack/RoPE/FMHA or the larger BF16/FP16 source-op
    activation residency/dtype problem.

- Round Z closed the small FlashInfer-internal CTA diagnostic:
  - For fused-RoPE FlashInfer, explicit `CTA_TILE_Q=64` regresses and
    `CTA_TILE_Q=128` is roughly the current/default behavior.
  - For no-RoPE FlashInfer core, `CTA_TILE_Q=64` is materially faster:
    3B `S=2048` core `0.798592 ms` versus default `0.851712 ms`.
  - Best standalone combo is now pair-wise RoPE pre-rotation plus no-RoPE
    FlashInfer `CTA64`: 3B `S=2048` `0.863744 ms`, 1.5B `S=2048`
    `0.666448 ms`, 0.5B `S=2048` `0.417664 ms`.
  - This is still below the end-to-end acceptance gate once converted to full
    model latency, and it adds extra launches/workspace. Do not migrate it as
    a standalone production operator.
  - Active attention direction changes from FlashInfer table/CTA tweaking to
    a different FMHA core schedule, fused QKV-pack/RoPE/FMHA, or a
    correctness-preserving plugin/source-op route.

- Round Y tested a standalone FlashInfer RoPE placement diagnostic:
  - `RoPE pre-rotate + FlashInfer(PosEncoding=None)` is correct and faster
    than current fused-RoPE FlashInfer on the attention microbench:
    3B `S=2048` `0.917648 ms` versus `0.971968 ms`, 1.5B `S=2048`
    `0.748544 ms` versus `0.770048 ms`, 0.5B `S=2048` `0.446464 ms`
    versus `0.466784 ms`.
  - This is useful evidence but still below the migration gate. The 3B
    `S=2048` movement is only about `1.96 ms` over 36 layers, below `>=1%`
    end-to-end on the current `~935 ms` source-op slice.
  - The no-RoPE FlashInfer core remains `~0.852 ms/layer`, while TRT's
    FMHA+RoPE reference is about `~0.65 ms/layer`. Next attention work should
    attack the FMHA core schedule or a fused QKV-pack/RoPE/FMHA layout, not
    merely migrate a separate pre-rotation operator.

- Round X refreshed the source-op versus TRT gap with warm graph-off mapping:
  - Current no-TRT source-op 3B `2048x1` prefill is `318.044 ms`; TRT
    reference is `289.951 ms`, leaving `+28.093 ms`.
  - Most production roles are now close to TRT: MLP GateUp `+2.953 ms`,
    DownProj `+0.188 ms`, QKV `+0.668 ms`, OProj `+0.008 ms`, SwiGLU
    `-0.030 ms`, norm `+0.192 ms`, lm_head `+0.010 ms`.
  - The real residual is prefill attention including RoPE/KV
    (`35.030 ms` versus `23.399 ms`, `+11.631 ms`) plus source-op
    BF16/FP16 casts (`+12.255 ms`).
  - Active priority: continue prefill attention Humanize/source-op work first.
    Cast removal likely needs a larger activation-residency or BF16-native
    source-op design and should stay behind an explicit design gate.

- Round W rejected using the existing fused GateUp/SwiGLU operator inside the
  accepted MLP source-op bridge:
  - The temporary default-off branch passed token alignment (`3/3`) with
    `EDGE_FM_PREFILL_CUTLASS_MLP=1` plus
    `EDGE_FM_PREFILL_SWIGLU_FUSION=1`.
  - It regressed 3B `2048x32` to `1074.953 / 449.389 ms` total/prefill versus
    the current source-op lane around `~935 / ~309 ms`.
  - The temporary code was reverted and `build-3060 --target install` passed.
    Do not wire the current `cutlass_prefill_swiglu` implementation into the
    MLP bridge; keep it as a standalone diagnostic only.

- Round V rejected the small MLP SwiGLU fast-math probe:
  - Temporarily replacing source-op SwiGLU `expf` with `__expf` passed token
    alignment (`3/3`) but measured 3B `2048x32` at
    `938.400 / 312.065 ms` total/prefill, which does not beat the current
    source-op lane and misses the `>=1%` acceptance gate.
  - The temporary code was reverted and `build-3060 --target install` passed
    after restore.
  - This rules out another tiny scalar fast-math pass on SwiGLU for now. Keep
    focus on RoPE/layout-aware FMHA fusion or a different GEMM/XMMA-like
    source-op route.

- Round U narrowed the cuTile/FMHA blocker:
  - Moving RoPE out of the FMHA inner loop is correct and much faster than the
    in-loop RoPE-cache path. Pair-wise pre-rotation improves 3B `S=2048` from
    `1.626 ms` to `1.036 ms`.
  - This still misses current EdgeFM FlashInfer with RoPE (`0.985 ms` at
    3B `S=2048`, `0.096 ms` at `S=512`), so it is rejected for migration.
  - `64x64` remains the best cuTile tile after RoPE pre-rotation; larger tile
    variants regress. The next attention step must reduce the extra rotation
    launches/workspace traffic or move to a C++/CUTLASS source-op that fuses
    RoPE/layout with the core schedule.

- Round T rechecked the QKV/OProj source-op lane after the FMHA RoPE blocker:
  - With current MLP source-op enabled on 3B `2048x32`, enabling CUTLASS linear
    for both QKV and OProj still helps materially versus the fallback linear
    path: `938.655 / 312.310 ms` total/prefill versus
    `952.936 / 326.664 ms`.
  - QKV-only and OProj-only each account for roughly half of that prefill
    movement, so this lane is real.
  - A temporary diagnostic tile sweep did not find a better replacement:
    current `128x256x32` measured `934.582 / 309.128 ms`, while
    `128x256x32_s4`, `64x256x32`, and `256x64x32` all regressed.
  - The diagnostic tile code was removed and `build-3060 --target install`
    passed after restore. Do not spend more time on classic CUTLASS linear tile
    variants without a new source/tactic.

- Round S found a useful but not-yet-migratable source-visible FMHA signal:
  cuTile's Apache-2.0 BF16 causal GQA core schedule with
  `TILE_M=64,TILE_N=64` beats the current EdgeFM FlashInfer attention microbench
  on core math for the tested shapes, including 3B `S=2048`
  (`0.933 ms` versus `0.985 ms`).
  - This is **not** an accepted replacement because the no-RoPE core path is
    not the real EdgeFM contract.
  - A first realistic RoPE-cache variant is correct but slower, with 3B
    `S=2048` at `1.626 ms`, so the direct migration is rejected.
  - The next attention task should focus on RoPE/layout fusion around the
    64x64 core idea, not on another table-only attention sweep.

- Round R rejected two near-term attention replacement routes:
  - CUTLASS41 BF16 fixed-seqlen FMHA builds and runs on `sm_86`, but the
    Qwen2.5-3B-like shape is `~0.092 ms` at `S=512` and `~1.069 ms` at
    `S=2048`; this does not beat current EdgeFM FlashInfer at long prefill
    (`~0.982 ms`).
  - The direct TRT `ContextFMHARunner` `plugin-op` can improve the current
    source-op slice from `936.410 / 310.621 ms` to
    `933.052 / 307.102 ms` on 3B `2048x32`, but the gain is only `~0.36%`
    end-to-end, relies on BF16->FP16 packing/cast, and fails the token
    alignment gate (`3/20` aligned on the 1.5B fixture). A temporary
    contiguous-Q/KV wrapper reached `931.301 / 305.265 ms`, but failed the same
    correctness gate and was removed.
- `trt_bridge` still cannot be removed. The route that still looks technically
  plausible is either a BF16-native source-op FMHA or a fused
  QKV-pack/RoPE/FMHA path that avoids round-tripping every layer through FP16.
  That should be treated as a larger focused kernel task, not a small table
  edit.

## Current Reference State

- Fresh 2026-05-16 serial rerun on `3B / 2048x32` is the current active
  profiling seed for the "remove TRT bridge" effort:
  - native: `1104.44 ms` total, `478.97 ms` prefill, `625.39 ms` decode
  - bridge diagnostic (`MLP auto + QKV/OProj bridge`): `941.83 ms` total,
    `315.06 ms` prefill, `626.60 ms` decode
  - TRT reference: `920.13 ms` total, about `284.10 ms` prefill,
    `635.80 ms` decode
  - artifacts: `.tmp_codex/bench/fresh_3060_20260516/*3b_2048x32*.json`
- Fresh 2026-05-16 graph-off stage-role attribution separates two targets:
  - if keeping bridge as the diagnostic baseline, the remaining gap is mostly
    `bridge_cast` (`20.28 ms`) plus attention/layout
    (`35.43 ms` EdgeFM bridge attention versus `23.45 ms` TRT plugin)
  - if removing bridge, the largest source-visible replacement target is native
    prefill MLP: GateUp `253.24 ms` plus DownProj `129.78 ms`, compared with
    `~210-212 ms` for the TRT/bridge-equivalent roles
- NCU access was repaired on 2026-05-16:
  `/proc/driver/nvidia/params` now reports `RmProfilingAdminOnly: 0`, and the
  optimizer env check reports `ncu_can_read_counters: true`.
- The active Humanize source-op handoff for bridge removal is
  `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/`.
  Round 0 created a finite BF16/FP16-weight MLP harness and baseline profile
  digest. The first FP32-acc CUTLASS seed was flat on 3B, but the follow-up
  `128x128x64_s3_f16acc` mutation is now the best source-visible seed:
  3B/S2048 `6.743 ms`, 1.5B/S2048 `4.245 ms`, 0.5B/S2048 `1.539 ms`.
- The default-off EdgeFM integration spike is complete:
  - gate: `EDGE_FM_PREFILL_CUTLASS_MLP=1`
  - fast mode: `EDGE_FM_CUTLASS_MLP_ACCUM=fp16`
  - optional residency: `EDGE_FM_CUTLASS_MLP_PERSISTENT_WEIGHTS=1`
  - real-checkpoint CUDA graph token alignment passed on the Qwen regression
    fixture.
  - target slice movement without a TensorRT engine:
    - 3B `2048x32`: native `1111.018 ms` -> source-op persistent
      `965.386 ms`; TRT reference `916.206 ms`
    - 1.5B `2048x32`: native `557.607 ms` -> source-op persistent
      `492.819 ms`; TRT reference `468.957 ms`
    - 0.5B `2048x64`: native `324.569 ms` -> source-op persistent
      `307.219 ms`; TRT reference `292.075 ms`
  - decision: keep default-off. It is a real no-bridge milestone, but not yet a
    TRT replacement; 3B persistent leaves only about `430 MB` free.
- Round K extended the source-visible path with QKV/OProj CUTLASS linear and a
  direct BF16-output trim:
  - gates: `EDGE_FM_PREFILL_CUTLASS_LINEAR=1`,
    `EDGE_FM_CUTLASS_LINEAR_ROLES=qkv|oproj|both|all`, and
    `EDGE_FM_CUTLASS_LINEAR_PERSISTENT_WEIGHTS=1`
  - correctness: QKV-only, OProj-only, both-enabled token alignment passed; MLP
    + linear source-op also passed the core generate regression
  - best no-engine measured slices:
    - 3B `2048x32`: `950.190 ms` total, `324.168 ms` prefill, still
      `+33.983 ms` versus TRT `916.206 ms`
    - 1.5B `2048x32`: `485.315 ms` total, `160.529 ms` prefill, still
      `+16.358 ms` versus TRT
    - 0.5B `2048x64`: best total remains the old MLP+linear source-op at
      `305.840 ms`; direct BF16 output improved prefill but not total
  - decision: keep default-off and diagnostic. This is progress toward removing
    `trt_bridge`, but not enough to remove it from reference/fallback use.
- Round L added a prefill-only MLP tile selector for the source-op path:
  - gate: `EDGE_FM_CUTLASS_MLP_TILE=default|auto|128x256x32|256x128x32`
  - `auto` uses `256x128x32` only on the verified 3B and 0.5B shapes; 1.5B
    keeps the previous `128x128x64` tile because the isolated sweep regressed.
  - current best no-engine measured slices:
    - 3B `2048x32`: `943.566 ms` total, `317.028 ms` prefill, still
      about `+27.360 ms` versus the earlier TRT `916.206 ms` reference
    - 3B `2048x64`: `1587.863 ms` total, `315.522 ms` prefill, still
      about `+17.4 ms` versus the Round B TRT reference
    - 0.5B `2048x32`: `178.429 ms` total, `54.719 ms` prefill
    - 0.5B `2048x64`: `304.975 ms` total, `54.773 ms` prefill, still
      about `+12.900 ms` versus TRT `292.075 ms`
  - fresh graph-off `3B / 2048x1` attribution after auto tile:
    source-op MLP GEMMs `215.277 ms`, linear GEMMs `32.468 ms`,
    FlashInfer attention `35.272 ms`, SwiGLU `16.667 ms`, casts `16.164 ms`.
  - decision: accepted as a default-off prefill improvement, not a default path
    change and not sufficient to remove `trt_bridge`.
- Round M vectorized the default-off source-op BF16/FP16 cast kernels:
  - affected files:
    `src/models/qwen2_5/cutlass_mlp_bridge.cu` and
    `src/models/qwen2_5/cutlass_linear_bridge.cu`
  - aligned even-sized conversions use `__nv_bfloat162` / `__half2`; scalar
    kernels remain as fallback.
  - current best no-engine measured slices:
    - 3B `2048x32`: `938.357 ms` total, `312.644 ms` prefill, still
      about `+22.151 ms` versus the earlier TRT `916.206 ms` reference
    - 3B `2048x64`: `1584.073 ms` total, `312.201 ms` prefill, still
      about `+13.6 ms` versus the Round B TRT reference
    - 1.5B `2048x32`: `485.854 ms` total, `161.084 ms` prefill
    - 0.5B `2048x64`: `303.308 ms` total, `53.333 ms` prefill, still
      about `+11.233 ms` versus TRT `292.075 ms`
  - fresh graph-off `3B / 2048x1` attribution after vector cast:
    source-op MLP GEMMs `213.462 ms`, linear GEMMs `32.220 ms`,
    FlashInfer attention `34.930 ms`, SwiGLU `16.597 ms`, casts `12.309 ms`.
  - decision: accepted as a default-off prefill improvement. The next active
    bottlenecks are GEMM/XMMA and long-prefill FMHA; decode stays parked.
- Round N added another small default-off prefill source-op polish:
  - `src/models/qwen2_5/cutlass_mlp_bridge.cu` now uses an aligned/even
    `__half2` SwiGLU kernel with scalar fallback.
  - `src/models/qwen2_5/cutlass_linear_bridge.cu` exposes a minimal diagnostic
    tile knob,
    `EDGE_FM_CUTLASS_LINEAR_TILE=default|auto|128x256x32`; `auto` maps only the
    verified 3B QKV/OProj prefill shapes to `128x256x32`.
  - current source-op matrix versus Round M:
    - 3B `2048x32`: `937.273 ms` total, `311.230 ms` prefill
      (`-1.084 / -1.414 ms`)
    - 3B `2048x64`: `1584.529 ms` total, `311.676 ms` prefill
      (`+0.456 / -0.525 ms`)
    - 1.5B `2048x64`: `819.430 ms` total, `159.713 ms` prefill
      (`-2.232 / -1.805 ms`)
    - 0.5B `2048x64`: `302.659 ms` total, `52.933 ms` prefill
      (`-0.649 / -0.400 ms`)
  - a focused five-run `3B / 2048x32` check measured `935.994 ms` total and
    `310.172 ms` prefill with `linear=auto`; after pruning unaccepted tile
    variants from the code, a final quick check measured `934.917 ms` total and
    `309.263 ms` prefill.
  - final graph-off grouping:
    MLP CUTLASS GEMMs `213.621 ms`, attention `35.030 ms`, linear QKV/OProj
    `31.935 ms`, SwiGLU half2 `14.433 ms`, casts `12.308 ms`.
  - an x4-per-thread cast-kernel follow-up was rejected: token alignment passed
    but `3B / 2048x32` regressed to `936.946 / 311.085 ms`; code reverted.
  - a legal FlashInfer CTA follow-up kept `cta128` as the best microbench value
    but only moved source-op `3B / 2048x32` to `934.283 / 308.723 ms`; below
    gate, no table change.
  - decision: keep as default-off source-op polish, not as a default path or
    bridge-removal claim. End-to-end movement is still below the `>=1%` gate.
- Round O rejected another MLP tile follow-up:
  - standalone `128x256x32_s4_f16acc` improved the 3B/S2048 MLP harness by only
    about `0.7%` versus the current 3B auto seed.
  - production attempts either exceeded the 3B persistent-weight memory budget
    when linear persistent was enabled, or regressed the practical 3B source-op
    mode with linear scratch: `938.254 / 311.863 ms` versus restored
    `935.052 / 309.435 ms` and Round N final `934.917 / 309.263 ms`.
  - decision: do not add more small classic CUTLASS MLP tile variants without a
    new source/tactic; the active prefill queue moves to BF16-correct FMHA or a
    genuinely different source-visible GEMM/XMMA-equivalent route.
- Round P collected a focused FMHA stall digest:
  - current FlashInfer BF16 3B/S2048 attention is about `1.059 ms/layer` in the
    standalone harness.
  - valid Ampere stall metrics show `short_scoreboard=5.10%`,
    `long_scoreboard=1.82%`, active warps `15.92%`, and DRAM only `4.89%`.
  - decision: skip more `cta_tile_q` table-only work. The next FMHA candidate
    must be source-visible and tiled, with lower shared-memory/register pressure
    and a bank-conflict-aware K/V tile.
- Round Q rejected the scalar query-block FMHA source-op branch:
  - pre-rotation plus query-block4 reduced the standalone seed from
    `S=2048 179.641 ms` to `83.959 ms`; adding K/V reuse reached about
    `79.637 ms`.
  - query-block8 was not a breakthrough: `S=512 5.811 ms`,
    `S=2048 77.507 ms`.
  - current EdgeFM FlashInfer is still `S=512 0.096 ms`,
    `S=2048 0.982 ms`, so the scalar route misses the `within 2x` gate by a
    very large margin.
  - next attention work should start from a tensor-core schedule, with
    `third_party/cutlass/examples/41_fused_multi_head_attention/` as the next
    local source scan candidate; do not spend more time on scalar query-loop
    mutations.
- A quick 3B/S2048 prefill attention cta128 probe is recorded but rejected for
  the official table:
  - single-layer microbench improved `0.973 -> 0.955 ms`;
  - temporary end-to-end `3B 2048x32` improved `938.357 -> 936.272 ms`;
  - total gain is only about `0.22%`, below the `>=1%` table-update gate.
- Active priority is now explicitly prefill-first. Decode opportunities should
  be recorded in the parking lot unless fresh profiling shows a `>=1%`
  end-to-end gain:
  - `lm_head_top1` stays rejected;
  - prior decode-attention table fix remains accepted historical work;
  - no new decode work should preempt MLP/linear/FMHA/cast prefill work.
- Fresh source-op versus TRT `3B / 2048x1` NSYS shows the remaining no-bridge
  gap is mixed:
  - source-op CUTLASS GEMM family: `253.785 ms`
  - source-op FlashInfer prefill attention: `35.364 ms`
  - source-op BF16/FP16 cast kernels before the BF16-output trim: `20.987 ms`
  - TRT reference uses XMMA GEMM groups (`138.692 ms` GateUp plus
    `86.089 ms` for the QKV/Down-style group) and FMHA `19.681 ms`
  - next optimization should be Humanize/source-visible GEMM/FMHA work, not
    another broad cublasLt table sweep.
- Round B refreshed the full 18-case LLM matrix on 2026-05-15. Effective
  results are in
  `.tmp_codex/bench/3060_20260515_roundb_effective_summary.json`.
- The first Round B bridge matrix accidentally loaded the non-bridge
  `libedge_fm.so` because `LD_LIBRARY_PATH` put `build-3060/install/lib` before
  the bridge build. Treat
  `.tmp_codex/bench/3060_20260515_roundb_matrix/*_bridge.json` as invalid
  bridge data; use `.tmp_codex/bench/3060_20260515_roundb_bridge_fixed/`.
- Effective Round B CUDA graph reference slices:
  - `3B / 2048x32`: native `1112.91 ms`, bridge `943.68 ms`, TRT `918.75 ms`
  - `3B / 2048x64`: native `1760.43 ms`, bridge `1590.87 ms`, TRT `1570.43 ms`
  - `1.5B / 2048x32`: native `559.78 ms`, bridge `480.66 ms`, TRT `469.61 ms`
  - `0.5B / 2048x64`: native `323.65 ms`, bridge `302.24 ms`, TRT `292.36 ms`
- The safe bridge remains useful but not enough: largest remaining gaps are
  mostly prefill, with `3B / 2048` about `+20-25 ms` end-to-end and
  `+29 ms` prefill versus TRT.
- Fresh `3B / 2048x1` graph-off attribution:
  - native kernel time `505.53 ms`; GateUp + DownProj account for `388.05 ms`
  - bridge kernel time `322.97 ms`
  - TRT kernel time `289.01 ms`
  - bridge residual versus TRT is mainly bridge casts (`20.34 ms`) and attention
    (`35.34 ms` versus TRT attention plugin `23.46 ms`)
- Round C steady `3B / 2048x1` attribution confirms the same shape after rerun:
  - bridge `gateup` runtime prefill `323.227 ms`, kernel total `322.829 ms`
  - bridge `both` runtime prefill `313.328 ms`, kernel total `312.682 ms`
  - TRT runtime prefill `288.955 ms`, kernel total `287.961 ms`
  - EdgeFM attention stays about `35.4 ms` while TRT plugin+rope is
    `23.3 ms`; the extra `~12 ms` is now the cleanest attention target
  - `both` removes about `10 ms` of bridge casts, but 3B still resolves to
    `gateup` in the new default-off `auto` mode because of memory headroom
- Fresh low-risk bridge probes on `3B / 2048x32`:
  - MLP `gateup` FP16 weights + attention tile128: `937.53 ms`, `1892 MB` free
  - MLP `gateup` + QKV/OProj FP16 linear weights + tile128: `934.41 ms`,
    `1264 MB` free
  - MLP `both` FP16 weights + tile128: `923.16 ms`, `358 MB` free
  - TRT reference: `914.49 ms`
- `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` is the fastest measured EdgeFM diagnostic
  slice, but it is still memory-unsafe for default or full-matrix acceptance.
  Round B `3B / 2048x32` improved `940.21 -> 927.69 ms`, but free memory dropped
  from about `1895 MB` to `361 MB`.
- Round C added `EDGE_FM_TRT_MLP_FP16_WEIGHTS=auto` as a default-off safety gate:
  the bridge estimates extra DownProj FP16 copy memory and uses `both` only when
  the estimate is below `EDGE_FM_TRT_MLP_AUTO_DOWN_MAX_MB` (default `1024` MB).
  Current Qwen2.5 shapes resolve 0.5B/1.5B to `both` and 3B to `gateup`, with a
  missing-`both` engine fallback to `gateup`.
- Round C long-prefill paired results make `auto` worth carrying as an
  experimental option: `1.5B / 2048x32` improves `480.445 -> 475.265 ms`
  (`-1.078%`) while keeping about `5713 MB` free. The `1.5B / 2048x64` and
  0.5B slices improve but remain below the 1% acceptance gate, so this is not a
  default production path yet.
- New Round B attention prefill split candidates for 0.5B/1.5B/3B improved
  microbenchmarks, but all failed the end-to-end acceptance gate. Do not update
  prefill attention table entries from these split candidates.
- Fresh `0.5B / 2048x64` decode probes reject `lm_head_top1` again: graph-off
  traces show regular `LMHead` work is replaced by `lm_head_top1::stage1_kernel`
  with roughly the same `~50 ms` total cost, and graph-on end-to-end is slightly
  slower (`309.18 ms` full logits versus `309.56 ms` top1).
- Fresh accepted `0.5B / 2048x64` decode-attention table fix:
  - root cause: duplicate 3060 records for `num_qo_heads=14|num_kv_heads=2|head_dim=64`
    left a later stale decode-attention entry overriding the faster entry
  - fix: update the later duplicate to `short_seq_bdz=4`, `long_seq_bdz=4`,
    `long_seq_threshold=1024`, `no_split_kv_threshold=256`,
    `min_chunk_size=64`, `chunk_alignment=64`,
    `chunk_candidates=[64,128,256,512]`
  - end-to-end result: `308.89 ms -> 302.38 ms` on same-run CUDA graph `runs=5`
    (`-2.10%`), with decode `256.02 ms -> 249.55 ms`
  - TRT remains ahead on that slice: `292.35 ms`, so the residual is about
    `+10.03 ms`
  - `0.5B / 2048x32` after the same fix measures EdgeFM `177.47 ms` versus TRT
    `167.50 ms`, residual `+9.97 ms`
- Fresh 0.5B decode-linear retune found no extra table win: existing decode
  QKV/OProj/DownProj records already match the best microbench candidates, and
  decode GateUp/LMHead prefer baseline heuristics.
- Round J added the first default-off `plugin-op` attention scaffold that calls
  TRT-Edge-LLM `ContextFMHARunner` directly from an EdgeFM attention operator,
  without a TensorRT engine or execution context. It is a diagnostic integration
  point, not an accepted BF16 fast path:
  - compile gate: `BUILD_TRT_PLUGIN_OPS=ON`
  - runtime gate: `EDGE_FM_PREFILL_TRT_FMHA_PLUGIN=1`
  - BF16 cast diagnostic gate:
    `EDGE_FM_PREFILL_TRT_FMHA_PLUGIN_ALLOW_BF16_FP16_CAST=1`
  - the BF16 cast path failed real `1.5B` CUDA graph token alignment (`17/20`
    mismatched generated steps), so it is rejected for production claims
  - safe `plugin-op` mode leaves BF16 Qwen on the existing FlashInfer fallback
    unless the diagnostic cast gate is explicitly enabled
- Humanize source-visible follow-up has started in
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/`. Round 0 only
  initialized the standalone repo, goal tracker, round contract, and ledgers.
  The first standalone BF16 FMHA harness is now in place. It benchmarks the
  current EdgeFM `AttentionLayer.forward_prefill` path and validates `S=32`
  against an FP32 causal GQA reference. Fresh `S=2048` baselines are:
  - 0.5B: `0.463906 ms/layer` mean
  - 1.5B: `0.755977 ms/layer` mean
  - 3B: `0.985320 ms/layer` mean
  NCU counter collection is now available after the 2026-05-16 driver
  permission fix, so future FMHA candidates should include real `.ncu-rep`
  artifacts plus profile digests. The next executable step is a source-visible
  BF16-correct CUDA candidate in the standalone repo, not an EdgeFM `src/`
  migration.
  A first scalar two-pass CUDA seed has been added and passes small-shape
  correctness, but it is far slower than EdgeFM (`3B/S512` seed `23.75 ms`
  versus EdgeFM `0.0958 ms`). Treat it only as a mutation starting point.
The old 2026-05-11 full-matrix bridge artifact remains historical context only.
For full-matrix comparisons, the current source of truth is the Round B
effective summary above. For bridge-removal target selection, use the fresh
2026-05-16 profiling seed and Humanize MLP workspace listed at the top of this
section.

## Fresh Artifact Layout

- Environment snapshots:
  `.tmp_codex/bench/3060_20260515_*_env.json`
- Benchmark slices and matrices:
  `.tmp_codex/bench/3060_20260515_*`
- Fresh 2026-05-16 bridge-removal seed:
  `.tmp_codex/bench/fresh_3060_20260516/*`
- NSYS traces:
  `.tmp_codex/nsys/3060_20260515_*`
- Fresh 2026-05-16 NSYS traces:
  `.tmp_codex/nsys/fresh_3060_20260516/*`
- Long-loop standalone optimization repos, only when a specific kernel deserves
  Humanize + KernelPilot:
  `deliverables/kernel_opt/3060_<kernel_or_role>_<date>/`
- Current bridge-removal Humanize repo:
  `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/`

Do not commit generated `.engine`, `.nsys-rep`, `.ncu-rep`, sqlite, or large raw
benchmark payloads unless a later review explicitly asks for them. The docs
should point to those artifacts and summarize the result.

## Execution Plan

### 0. Re-establish the Baseline

1. Snapshot toolchain, GPU, driver, build, Nsight, and CUDA-counter access with
   the optimizer skill's `check_env.py`. Treat `ncu` availability and counter
   readability separately. Current host state is fixed:
   `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 0`, so non-root
   NCU counter collection is available.
2. Verify model paths for `0.5B`, `1.5B`, and `3B`, and record which TRT engines
   are locally available.
3. Full Round B native/bridge/TRT matrix is complete. Re-run only when the table,
   bridge mode, build, or engine set changes.
4. When measuring bridge cases, ensure the bridge build `lib/` and `install/lib/`
   precede `build-3060/install/lib` in `LD_LIBRARY_PATH`; otherwise the Python
   module can load the non-bridge shared library and silently fall back.

Current benchmark commands use the pytest benchmark entry points or
`scripts/profile/profile_edgefm_generate_case.py`. The old
`report_qwen_benchmark_suite.py` command in earlier notes is obsolete because
that helper is no longer present in this checkout.

### 1. Re-profile Before Editing

For every Tier-0 miss, capture two traces:

- graph-off mapping trace for kernel and role attribution
- graph-on formal trace for official CUDA graph behavior

Use the optimizer skill's known-path tables to map the top kernels to one of:

- linear / QKV / OProj
- MLP GateUp / SwiGLU / DownProj
- FlashInfer prefill attention
- norm / sampler / finalize
- TensorRT bridge cast or IO overhead

Do not start a new kernel implementation until the target kernel, shape, call
path, correctness reference, and acceptance threshold are written in the log.

### 2. Low-risk Existing-path Tuning

This lane stays inside the current operator/table/runtime boundaries:

- treat optional bridge cast reduction as diagnostic unless it clears 1%:
  fresh MLP/linear persistent FP16 probes improved `3B / 2048x32` but did not
  create a safe accepted default path
- keep FlashInfer prefill attention table split candidates rejected unless a new
  paired end-to-end run clears 1%; Round B split candidates for 0.5B/1.5B/3B did
  not clear that bar
- keep small-model decode GEMV/attention out of the active queue unless a new
  profile changes the gap: the fresh 2026-05-16 `2048x64` matrix shows 3B
  decode already faster than TRT, 1.5B decode tied, and only 0.5B decode behind
  by about `4.15 ms`; `lm_head_top1`, real FP32-logits LMHead cublasLt tuning,
  and a 384-candidate 0.5B decode attention sweep all failed the `>=1%` gate
- keep the repaired `retune_qwen_operator_tables.py` dtype plumbing available
  for later BF16/FP16 table sweeps
- validate 3060 operator table shape signatures against the fresh profile
- check cublasLt tactic drift only for shapes proven hot by the trace
- keep changes only when the target CUDA graph slice improves by at least `1%`
  and no regression slice moves meaningfully backward

Rejected routes stay rejected unless fresh evidence changes their trigger:
classic CUTLASS dense GEMM tile sweeps, cuTile dense MatMul, standalone
`prefill_cta_tile_q=128`, temporary `prefill_num_mma_kv_cap`,
`lm_head_top1`, 0.5B decode attention table-only sweeps, temporary FP16
checkpoint conversion, cublasLt native row-major descriptors, and the old
default-on prefill SwiGLU fusion attempt.

### 3. Bridge Cleanup and Measurement

TensorRT bridge work remains optional, compile/runtime-gated, and default-off.
The useful bridge questions are now narrow:

1. Can the QKV/OProj bridge be packaged with reproducible engine generation and
   clean fallback behavior?
2. Can bridge cast overhead be reduced without persistent 3B memory risk? The
   current answer is "partially": MLP `both` is fast but has only about
   `358-361 MB` 3B headroom. The explicit `auto` gate keeps 3B on `gateup` while
   allowing small/mid models to use `both` when engine coverage and the memory
   cap allow it.
3. Can same-host or same-process TRT comparisons be made reproducible enough for
   the official matrix?
4. Does the bridge still win after the fresh baseline and current branch layout?

No generated TensorRT engine should be treated as source code. A bridge change is
accepted only after correctness, fallback, missing-engine, and full-matrix gates.
The 2026-05-16 decode matrix means bridge removal should not be blocked on
decode. It is blocked on proving source-op prefill can fully replace the useful
MLP/QKV/OProj bridge diagnostics and close the remaining prefill gap versus TRT
reference across 0.5B/1.5B/3B.

### 4. Source-visible Kernel Optimization

Use Humanize + KernelPilot only after profiling identifies a high-value
source-visible kernel or operator role. The standalone repo must include:

- correctness reference and shape matrix
- baseline benchmark and NCU profile digest
- attempt, optimization, source-idea, and lineage ledgers
- rollback and migration rules before touching `src/`

Likely escalation candidates if fresh profiles justify them:

- continue prefill MLP source-op refinement only if a new source-visible GEMM
  tactic can beat the current CUTLASS `128x128x64_s3_f16acc` seed. The
  production spike already removed most of the native MLP gap, so do not rerun
  small classic CUTLASS table sweeps without new evidence.
- QKV/OProj source-visible linear path. The current `3B / 2048x1` source-op
  trace still shows native QKV around `0.964 ms/layer` and OProj around
  `0.711 ms/layer`; replacing those without a TRT engine is the next plausible
  no-bridge full-model gap closer.
- prefill attention or KV write/read path
- fused GateUp/SwiGLU/DownProj elementwise/cast overhead

The long-loop goal is a measured in-repo win, not an exploratory rewrite.

### 5. Memory-policy Investigation

`EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` remains blocked as a 3B default
optimization. It completes `3B / 2048` and reduces the TRT gap to single digits,
but current headroom is only about `358-361 MB`.

The accepted low-risk direction is now the explicit/default-off
`EDGE_FM_TRT_MLP_FP16_WEIGHTS=auto` gate:

- estimate extra DownProj FP16 copy memory from model config
- compare it with `EDGE_FM_TRT_MLP_AUTO_DOWN_MAX_MB` (default `1024` MB)
- resolve small/mid models to `both` when safe
- resolve 3B to `gateup` unless a future memory policy changes the cap
- fall back from missing `both` engines to matching `gateup` engines

Do not make `auto` the default until full-matrix correctness, missing-engine,
memory, and paired TRT comparison gates are complete.

Round E full-matrix refresh:

- Missing `both` engines for 0.5B/1.5B `m=512/1024` were generated so the
  `auto` policy could be measured without missing-engine fallback.
- Current optional bridge matrix (`MLP auto + QKV/OProj bridge`, CUDA graph,
  `warmup=1`, `runs=3`) confirms the intended policy:
  - 0.5B/1.5B resolve to `both`
  - 3B resolves to `gateup`
- Compared with Round B bridge, `auto` improves 1.5B by `-25.28 ms` summed
  across six cases and 0.5B by `-5.65 ms`; 3B is effectively unchanged by
  design.
- Remaining top gaps versus TRT are now:
  - 3B `2048x32`: `+24.68 ms` total, `+29.05 ms` prefill
  - 3B `2048x64`: `+20.02 ms` total, `+29.00 ms` prefill
  - 3B `1024x32`: `+13.38 ms` total, `+19.50 ms` prefill
  - 0.5B `2048x64`: `+10.33 ms` total, `+5.99 ms` prefill, `+4.25 ms` decode
- A current-auto `0.5B / 2048x64` graph-off mapping reaffirms that the decode
  part of this gap is still the already-rejected pair: full LMHead and
  FlashInfer decode attention. Do not reopen `lm_head_top1` or decode attention
  retune unless a new profile shows a credible `>=1%` graph-on end-to-end win.
- Active conclusion: `auto` remains a useful default-off bridge policy, but the
  hard target has moved back to 3B long-prefill attention/layout evidence.

For the existing bridge diagnostic path, the next high-value route is still
attention/layout evidence rather than another MLP table tweak. Round C shows
EdgeFM prefill attention at about `35.4 ms` versus TRT plugin+rope at about
`23.3 ms` for `3B / 2048x1`. Round F captured NCU counters for the current
FlashInfer prefill kernel and confirmed a low-occupancy, low-eligible-warp
Ampere shape: about `38%` SM/memory throughput, `16.7%` theoretical occupancy,
`49 KB` dynamic shared memory per CTA, and `math_pipe_throttle` as the dominant
stall.

For the no-bridge objective, the fresh 2026-05-16 `3B / 2048x32` profile
promotes prefill MLP source-op work ahead of attention: the native GateUp +
DownProj prefill roles are about `170 ms` slower than the TRT/bridge-equivalent
roles over the full 36-layer prefill. Do not treat this as a request for another
cublasLt table sweep; the active route is a source-visible Humanize loop under
`deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/`.

Round J production spike result: MLP source-op is now in `src/` and default-off.
It should remain a benchmarkable no-bridge mode while the next source-op target
is selected. The top two options are QKV/OProj source-visible linear and
source-visible BF16 prefill attention; prioritize by fresh profile evidence, not
by the older bridge-residual order alone.

Round D narrowed the attention route:

- `prefill_cta_tile_q=128` remains rejected for 3B. It wins the isolated
  `2048` attention microbench but only improves current bridge/auto
  `3B / 2048x32` generate by `-0.22%`, below the 1% gate.
- A standalone TensorRT-Edge-LLM `AttentionPlugin` engine for the same 3B shape
  initially reported `~1.39 ms/layer` via Python event timing, but NSYS showed
  the actual plugin GPU work at about `0.65 ms/layer` for FMHA+RoPE. This
  matches the TRT network and is a real candidate.
- The bridge-layout repro is now complete:
  - `BF16 hidden -> FP16 QKV -> AttentionPlugin -> FP16 OProj -> BF16 output`
    is only about `7-8 ms` better than the current Round C QKV + attention +
    OProj kernel sum for 3B/2048.
  - `BF16 packed QKV -> FP16 cast -> AttentionPlugin -> BF16 output` graph
    replay improves isolated attention by an estimated `6.17 ms` for 3B,
    `3.13 ms` for 1.5B, and `2.61 ms` for 0.5B at `S=2048`.
  - These gains are real but below the low-risk `>=1%` official end-to-end gate.
- Do not add an attention plugin bridge as a default or production path from
  this evidence alone. Reaching TRT's full attention advantage likely requires
  a broader FP16 packed-QKV/cache or activation-residency route, which is outside
  the current no-large-design-change lane.
- The simple source-visible BF16/FP16 cast probe is tied with TensorRT internal
  casts (`~53 us` for `2048x2048`, `~66 us` for `2048x2560`), so bridge cast
  reduction should also be considered a larger FP16 residency problem, not a
  quick custom cast-kernel fix.

Round F resolved the NCU evidence gap:

- Current FlashInfer `SinglePrefillWithKVCacheKernel` for 3B/2048 is not DRAM
  bandwidth-bound. It is limited by low occupancy/eligible warps and math-pipe
  pressure under the current CTA shape.
- TRT's standalone FMHA kernel for the same shape runs in a similar low
  occupancy envelope but is much more efficient: `740 us`, `63%` SM throughput,
  `55%` memory throughput, `32 KB` dynamic shared memory, and `math_pipe_throttle`
  `1.69` versus EdgeFM's `1.345 ms`, `38%`, `49 KB`, and `4.34`.
- A temporary `prefill_num_mma_kv_cap` knob looked good in microbench
  (`cta64,cap2` total `1.2664 ms` versus baseline `1.3808 ms`, `-8.29%`) but
  failed paired CUDA graph generate: `942.981 -> 942.672 ms` on `3B/2048x32`,
  only `-0.033%`. The source experiment was reverted.
- Round G closed the remaining small FP16 residency shortcuts:
  - pure FP16 attention inputs did not beat BF16 on the 3B attention shape, and
    FlashInfer `use_fp16_qk_reduction=True` is not available in the current
    package build
  - `3B / 2048x32` with MLP `both` plus QKV/OProj persistent FP16 weights OOMs
    around layer 34
  - 0.5B QKV/OProj FP16-weight bridge regresses slightly, while 1.5B improves
    only `0.31-0.57%` end-to-end, below the `>=1%` gate
- Round H added a direct
  `trt_edgellm::ContextFMHARunner` probe under
  `deliverables/kernel_opt/3060_prefill_attention_20260515/`. It reproduces the
  3B `S=2048,Hq=16,Hkv=2,D=128` TRT FMHA+RoPE timing at `0.650708 ms` mean /
  `0.651264 ms` median and NCU `739 us` kernel duration. This confirms a real
  runner-level candidate, but it is still only a standalone attribution tool.
- Round J wired a default-off direct TRT FMHA runner wrapper into EdgeFM as
  `trt_context_fmha_plugin_attention`, but the current BF16 execution path
  requires BF16->FP16 casts and failed token alignment. Round R also rejected a
  temporary contiguous-Q/KV variant for the same reason. Keep the packed path
  behind `--plugin-op-allow-bf16-fp16-cast` for diagnostics only, not as a
  performance candidate.
- The remaining attention work should now be either a real source-visible
  BF16-correct prefill attention replacement, an explicitly reviewed FP16
  packed-QKV/cache/activation-residency experiment, or a plugin-op route that
  passes token alignment without a lossy BF16 cast. Do not repeat
  operator-table-only sweeps, the `num_mma_kv_cap` idea, the BF16 wrapper probe,
  or QKV/OProj FP16-weight bridge sweeps without a new acceptance argument.

Immediate Round I route:

1. Done: build correctness coverage around the direct FMHA runner for normal
   RoPE and causal output reference on a small sequence, plus
   `S=512/1024/2048` normal-RoPE smoke coverage.
2. Done: measure the actual EdgeFM-compatible wrapper cost for packed BF16 QKV
   to packed FP16 QKV plus FP16 output back to BF16. The 3B `S=2048` total is
   `0.787520 ms/layer`, versus current EdgeFM FlashInfer BF16
   `0.971232 ms/layer`, giving only a `~6.6 ms` full-model attention upper
   bound across 36 layers.
3. Done: run a default-off operator-level integration spike. It builds and is
   safe by default, but the BF16 cast diagnostic path is rejected because real
   generate token alignment fails. A follow-up contiguous-Q/KV diagnostic was
   faster but also failed alignment and was removed.
4. Done: a mixed-cublasLt cast-removal probe found no heuristic candidate for
   `A=BF16 activation, B=FP16 weight` with either FP16 or BF16 output, including
   3B GateUp/QKV shapes. Do not route production source-op work through mixed
   cublasLt descriptors.
5. Done: a production-like qkv-strided Q diagnostic shows the AttentionLayer
   standalone baseline is not hiding a large Q-layout penalty. 3B `S=2048`
   contiguous Q is `0.984016 ms`, qkv-strided Q is `0.985792 ms`; do not add a
   Q-contiguous copy/fusion path for this reason alone.
6. Next: do not benchmark plugin-op BF16 as a performance candidate until the
   alignment problem is fixed. If the no-bridge effort continues, move to a
   source-visible BF16-correct FMHA or a reviewed FP16 residency design instead
   of another table-only attention tweak.
7. Active long-loop handoff: continue in
   `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/` with a standalone
   BF16 FMHA harness covering normal RoPE, causal GQA correctness, benchmark
   metadata, and first NCU digest.

## Current Active Queue

1. Done: Stage-1 internal EdgeFM `trt_bridge` removal is accepted and the
   Qwen2.5 TensorRT engine prefill bridge code is removed.
2. Stage 2 target remains external `TRT-Edge-LLM`, but the latest full
   18-slice matrix is now parity-or-better for all 0.5B and 3B shapes. A
   higher-run 1.5B recheck reduces `512x32` to practical parity
   (`+0.197 ms` avg, `+0.135 ms` median); the only stable positive residual is
   `1.5B 512x64`, about `+0.9 ms`.
3. `1.5B 512x64` is the only active Stage-3 row. Only accept changes that
   improve this row or keep it neutral while preserving 1.5B `512x32/1024/2048`
   and all 0.5B/3B parity rows.
4. `0.5B` and `3B` should stay in checkpoint-preservation mode. Do not spend
   more table churn on their historical gaps unless a fresh profile shows a
   repeatable regression or a new source-visible opportunity.
5. Stop ordinary gate-up/MLP-down/QKV/OProj CUTLASS 2.x table churn unless a
   fresh Stage-2 profile gives a measurable, repeatable end-to-end opportunity.
   The fused-bias QKV boundary probe is rejected on 3B correctness, so do not
   expose a `fuse_bias` table knob without a new same-shape reference test. The
   `1.5B` GateUp Humanize workspace also confirms the current CUTLASS
   `128x256x32_s3` candidate remains the fastest stable source-op variant found
   so far; further progress likely needs a nontrivial source-visible GEMM route,
   not more small table permutations.
6. Rejected: full BF16<->FP16 prefill attention wrapper. It regressed 3B
   `2048x32` from `931.702 ms` to `937.198 ms` and increased
   `attention_op.cu` compile cost. Only revisit FP16 attention residency if the
   upstream QKV source-op can produce resident FP16 Q/K/V without extra
   per-layer full-tensor cast kernels.
7. Rejected: FlashInfer forced-CTA half QK accumulation. It can be made to
   compile in the standalone diagnostic branch, but 3B `S=32`
   pre-rotate+no-RoPE fails correctness (`max_abs=3.023071`,
   `mean_abs=0.329942`) while the float-accumulation branch passes. Do not use
   half QK accumulation for EdgeFM BF16 prefill.
8. Rejected: 3B `m=2048` MLP source-op tile split/mixed-BF16 probes. The only
   apparent win (`gateup_tile=128x256x32`) failed paired 10-run confirmation;
   down-only, gate+down split, and mixed-BF16 all regress. Keep the 3B MLP
   table record unchanged.
9. Rejected: 3B fused-QKV CUTLASS fused-bias epilogue. Same-process layer
   microbench was faster (`0.644352 -> 0.585728 ms`) but failed correctness
   badly (`max_abs=91.75`, `mean_abs=1.02755`), and `StoreT=false` did not fix
   it. Keep the safe no-fuse mixed-BF16 path.
10. Accepted for the current 0.5B/1.5B long-context default: prefill
   strided-QKV attention plus side-stream KV copy. It remains opt-out via
   `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=0` and
   `EDGE_FM_PREFILL_STRIDED_QKV_SIDE_STREAM=0`; keep watching adjacent shapes
   when future runtime changes touch this path.
11. Rejected for default: fused K cache copy + K RoPE pre-rotate. It passes
   graph token alignment, but the latest same-day paired current-table recheck
   regressed 3B `2048x32` from `928.014 ms` to `930.134 ms`; treat the earlier
   positive signal as order/noise until a larger fusion changes the cost model.
12. Decode tuning note: `lm_head_top1` remains explicit/default-off. Historical
   0.5B decode-heavy checks were positive, but the latest current-table
   `0.5B 2048x64` fresh run was neutral/slower (`300.235 -> 300.269 ms`).
   Do not enable for default 0.5B/1.5B/3B paths.
13. Rejected: `1.5B` prefill attention short/long split64 table. Microbench
   improved the isolated attention layer, but full `2048x32` generate regressed
   from `474.645 ms` to `474.783 ms`; keep the current attention table.
14. Local TRT-Edge-LLM plugin-op scan did not find a dense FP16/BF16 GEMM plugin
   asset for this path. Do not plan a plugin-op GEMM integration unless a new
   source-visible asset is identified.
15. Review [doc/3060_fused_mlp_review.md](./3060_fused_mlp_review.md) before
   changing engine/layer/operator boundaries for a larger prefill MLP path.
16. Rejected again: standalone prefill GateUp+SwiGLU fusion using
   `scripts/tune/profile_prefill_swiglu_kernels.py`. Current-tree BF16
   recheck on `2026-05-17` shows the fused path is slower than the two-stage
   path: `1.5B S=2048` `5.125 ms` vs `4.778 ms`, and `3B S=2048`
   `8.214 ms` vs `7.304 ms`. Do not route the default MLP path through this
   fused-MoE helper.
17. Rejected: 3B QKV/OProj source-op `weight_mode=bf16_direct`. It avoids
   nonpersistent FP16 weight casts but slows the actual GEMM path badly:
   `3B 2048x32` moved from `928.890 ms` / prefill `303.043 ms` to
   `951.409 ms` / prefill `325.187 ms`.
18. Rejected: 3B QKV/OProj source-op `fp16_cast + overlap_casts`. Side-stream
   weight conversion cannot pay for the extra activation cast; `3B 2048x32`
   moved to `932.897 ms` / prefill `306.735 ms`.
19. Rejected: 1.5B GateUp `128x128x32_s3` from a CUTLASS profiler col-output
   lead. The actual EdgeFM row-output standalone harness measured
   `2.960320 ms` median versus current `2.502432 ms`.
20. Accepted: 3B/S2048 GateUp source-op `128x256x32_s4`, scoped only to
    `m=2048|hidden=2048|intermediate=11008`. The current production matrix
    recheck showed small but stable wins: `3B 2048x32`
    `928.890 -> 927.804 ms`, `3B 1024x32` `761.629 -> 759.645 ms`, and
    `3B 2048x64` `1577.463 -> 1575.828 ms`. This supersedes the older
    standalone-only s4 rejection for this one 3B production shape, but it does
    not reopen broad CUTLASS table churn.
21. Rejected: BF16/FP16 conversion kernel two-pair unroll. The conversion
    kernels are visible in the fresh profile, but the unrolled version regressed
    `3B 2048x32` from `927.804 ms` to `929.335 ms`, so the direct one-pair
    vector conversion remains default.
22. Rejected for default, retained as diagnostic: strided-QKV attention runtime
    gate. The path has small local wins on `1.5B 2048x32`
    (`474.902 -> 474.105 ms`) and one `3B` run (`927.804 -> 927.437 ms`), but
    the rebuilt `3B` forced-on/default checks were noisy (`929.994 ms` and
    `931.387 ms`) and `0.5B` regressed when forced on (`173.965 -> 174.225 ms`).
    Keep `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=1` as an explicit profiling
    route only.
23. Accepted: 3B/S2048 SwiGLU launch retune. The production path now uses a
    2D-grid SwiGLU kernel plus table-controlled `swiglu_threads`. The current
    3B `m=2048|hidden=2048|intermediate=11008` table sets `swiglu_threads=64`;
    all other model sizes keep their previous launch policy unless explicitly
    tuned.
24. Accepted with memory guard: 3B/S2048 QKV source-op persistent FP16 weights.
    The active QKV record sets `persistent_weights=true` and
    `persistent_min_free_mb=64`; OProj persistence remains off because
    QKV+OProj OOMs. Current confirmation: `3B 2048x32` `921.612 ms`, prefill
    `295.720 ms`, leaving `+5.101 ms` versus external TRT; `3B 2048x64`
    `1570.681 ms` leaves `+2.218 ms`. Validation passed for operator table
    schema, profile script py_compile, token alignment, generate
    max-new/deferred-stop/metrics/compact-vocab smoke group, and
    `tests/operators/test_prefill_linear.py`.
25. Accepted: source-op runtime cleanup for the official 3060 table path.
    Default-level source-op selection logs are now debug-only, and prefill
    linear/MLP source-op runtime configs are cached by shape. This is not a
    major CUDA win, but it removes repeated host logging/table-resolution work
    and keeps the 0.5B long-context checkpoint at `297.489 ms` for `2048x64`
    and `171.068 ms` for `2048x32`. Validation passed for operator-table
    schema, profile/tuning script py_compile, and the 21-test focused
    operator/generate regression subset.
26. Fresh post-cache long-context matrix versus same-machine TRT reference:
    `0.5B` remains the active default-path gap (`+4.686 ms` at `2048x32`,
    `+6.153 ms` at `2048x64`), `1.5B` is inside the requested `<=3 ms`
    stretch target (`+2.282/+2.400 ms`), `3B 2048x64` is effectively tied
    (`+0.751 ms`), and `3B 2048x32` is just over the tactical line
    (`+5.015 ms`). Use
    `.tmp_codex/bench/stage2_20260518_post_cache_current_matrix/` as the next
    optimization baseline.
27. Accepted: 0.5B decode attention `chunk_candidates=[64,128,192,256]` for
    both duplicate `num_qo_heads=14|num_kv_heads=2|head_dim=64` decode table
    records. Updating only one duplicate record did not hit the resolved path;
    updating both gave a real endpoint win. Reverse confirmation versus an old
    overlay showed `2048x64` `297.764 -> 289.190 ms` and `2048x32`
    `171.443 -> 167.308 ms`. Current matrix now has 0.5B at `+0.480 ms`
    (`2048x32`) and `-1.550 ms` (`2048x64`) versus TRT. Active gap shifts to
    `3B 2048x32` at about `+5.5 ms`; keep 1.5B under watch because
    `2048x64` is `+2.9 ms`.
28. Accepted: extend the same decode attention chunk policy to q12 and q16
    records. The long-context matrix is now effectively parity-or-better
    against the fresh same-machine TRT reference:
    `0.5B` `+0.870/-1.531 ms`, `1.5B` `-2.363/-8.104 ms`, and
    `3B` `-1.191/-12.599 ms` for `2048x32/64`. Five of six slices beat TRT;
    the only remaining positive residual is below `1 ms`. Raw artifacts:
    `.tmp_codex/bench/stage2_20260518_after_decode_attention_chunks192_all_sizes_matrix/`.
29. Full 18-case paired matrix after the decode-attention chunk update shows
    the long-context target is solved, but a smaller mid-context queue remains:
    `1.5B 1024x32` `+3.544 ms`, `1.5B 1024x64` `+4.518 ms`, and
    `3B 1024x32` `+3.679 ms`. `long_seq_threshold=1536` was rejected because
    it regressed `1.5B 1024x32` despite minor wins elsewhere. Next action is
    higher-run confirmation of these 1024 residuals before another table/kernel
    change.

## Command Cheatsheet

Environment:

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/check_env.py \
  --out .tmp_codex/bench/3060_20260515_env.json
```

Single EdgeFM profile slice:

```bash
EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 \
EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 \
python3 scripts/profile/profile_edgefm_generate_case.py \
  --model-path examples/qwen2.5-3b-instruct/qwen2.5-3b-instruct \
  --prefill-len 2048 --decode-len 32 --runs 3 --warmup 1 \
  --use-cuda-graph --json
```

Pytest EdgeFM matrix slice:

```bash
EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 \
EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 \
EDGE_FM_BENCH_LLM_MODELS=3b \
EDGE_FM_BENCH_PREFILL_LIST=2048 \
EDGE_FM_BENCH_DECODE_LIST=32 \
python3 -m pytest -s tests/engine/test_qwen2_generate.py -k benchmark_llm -q
```

Pytest TRT comparison slice:

```bash
EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 \
EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 \
EDGE_FM_BENCH_LLM_MODELS=3b \
EDGE_FM_BENCH_PREFILL_LIST=2048 \
EDGE_FM_BENCH_DECODE_LIST=32 \
python3 -m pytest -s tests/engine/test_qwen2_generate.py -k benchmark_trt_edgellm -q
```

Graph-off mapping trace:

```bash
nsys profile -o .tmp_codex/nsys/3060_20260515_3b_2048x32_mapping \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path examples/qwen2.5-3b-instruct/qwen2.5-3b-instruct \
    --prefill-len 2048 --decode-len 32 --profile-range
```

## Acceptance Gates

- Correctness passes before and after an optimization:
  `test_generate_token_alignment`, CUDA graph alignment, and the closest
  operator/layer tests.
- `scripts/operator_table/validate_operator_tables.py` passes after table edits.
- A target-slice win no longer has a hard `1%` cutoff. Stable sub-1% wins may be
  absorbed when the change is localized, correctness-clean, and does not regress
  adjacent slices in the available paired checks.
- The full 18-case matrix must be rerun before a performance change is called
  globally accepted/default-final. Smaller table steps can be committed as
  measured local improvements while the matrix rerun is queued.
- Failed experiments are reverted quickly and recorded under rejected/obsolete
  notes instead of staying in the active queue.
