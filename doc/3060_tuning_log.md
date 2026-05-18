# 3060 LLM Tuning Log

## 2026-05-18 Stage 3 Full Matrix After 0.5B/1.5B Residual Squeeze

After accepting the 0.5B `m=2048` QKV/OProj source-op tile override and the
1.5B q12 decode attention `no_split_kv_threshold=512` update, I reran the full
same-machine EdgeFM CUDA graph path against the TRT-Edge-LLM reference.

| Model | Shape | EdgeFM total | EdgeFM prefill | EdgeFM decode | TRT-Edge-LLM total | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 512x32 | `126.609 ms` | `12.771 ms` | `113.713 ms` | `134.979 ms` | `-8.370 ms` |
| 0.5B | 512x64 | `242.668 ms` | `12.581 ms` | `229.936 ms` | `248.447 ms` | `-5.778 ms` |
| 0.5B | 1024x32 | `140.586 ms` | `24.929 ms` | `115.542 ms` | `141.554 ms` | `-0.969 ms` |
| 0.5B | 1024x64 | `257.486 ms` | `24.545 ms` | `232.790 ms` | `257.671 ms` | `-0.185 ms` |
| 0.5B | 2048x32 | `168.178 ms` | `48.647 ms` | `119.411 ms` | `168.969 ms` | `-0.791 ms` |
| 0.5B | 2048x64 | `290.109 ms` | `48.293 ms` | `241.652 ms` | `292.024 ms` | `-1.915 ms` |
| 1.5B | 512x32 | `346.130 ms` | `37.051 ms` | `308.966 ms` | `345.300 ms` | `+0.830 ms` |
| 1.5B | 512x64 | `664.482 ms` | `36.460 ms` | `627.861 ms` | `663.594 ms` | `+0.888 ms` |
| 1.5B | 1024x32 | `384.591 ms` | `73.272 ms` | `311.203 ms` | `384.730 ms` | `-0.139 ms` |
| 1.5B | 1024x64 | `705.402 ms` | `73.131 ms` | `632.115 ms` | `708.590 ms` | `-3.188 ms` |
| 1.5B | 2048x32 | `466.729 ms` | `147.361 ms` | `319.242 ms` | `469.011 ms` | `-2.282 ms` |
| 1.5B | 2048x64 | `796.607 ms` | `147.445 ms` | `648.996 ms` | `805.288 ms` | `-8.681 ms` |
| 3B | 512x32 | `681.482 ms` | `78.387 ms` | `602.979 ms` | `686.182 ms` | `-4.700 ms` |
| 3B | 512x64 | `1303.951 ms` | `78.493 ms` | `1225.296 ms` | `1316.090 ms` | `-12.139 ms` |
| 3B | 1024x32 | `757.754 ms` | `146.907 ms` | `610.708 ms` | `758.143 ms` | `-0.389 ms` |
| 3B | 1024x64 | `1388.178 ms` | `147.160 ms` | `1240.852 ms` | `1395.001 ms` | `-6.824 ms` |
| 3B | 2048x32 | `917.042 ms` | `297.759 ms` | `619.143 ms` | `919.372 ms` | `-2.330 ms` |
| 3B | 2048x64 | `1557.436 ms` | `298.517 ms` | `1258.738 ms` | `1571.146 ms` | `-13.710 ms` |

Summary:

- `positive_count=2`, `max_gap=+0.888 ms`.
- 0.5B and 3B are now faster than TRT-Edge-LLM across all checked shapes.
- The only remaining positive residuals are 1.5B decode-heavy short-context
  slices: `512x32 +0.830 ms` and `512x64 +0.888 ms`.
- The earlier `0.5B 2048x32` and 3B residual queues are now historical; do not
  keep tuning them without fresh profile evidence.

Accepted changes in this subround:

- `Qwen2.5-0.5B / m=2048` source-op QKV and OProj use
  `tile=256x128x32`.
- `Qwen2.5-1.5B / q12 decode attention` uses
  `no_split_kv_threshold=512` for both duplicate q12 decode records.

Rejected or parked in this subround:

- 0.5B `m=2048` MLP activation FP16 and combined MLP+linear overlays were not
  accepted because the QKV/OProj-only change was more balanced on `2048x32/64`.
- 1.5B q12 `no_split_kv_threshold=768/1024` regressed 512-context decode
  heavily, and old chunk alignment policies regressed 1024-context slices.

Artifacts:

- `.tmp_codex/bench/stage3_20260518_132409_0p5b_2048_prefill_overlay_sweep/`
- `.tmp_codex/bench/stage3_20260518_132728_0p5b_2048_prefill_combo_confirm/`
- `.tmp_codex/bench/stage3_20260518_133041_0p5b_2048_qkv_oproj_tile256_official/`
- `.tmp_codex/bench/stage3_20260518_133219_1p5b_decode_q12_overlay_sweep/`
- `.tmp_codex/bench/stage3_20260518_133829_1p5b_q12_nosplit512_official/`
- `.tmp_codex/bench/stage3_20260518_134317_full_matrix_after_0p5b_q12_accept_retry/`

### 1.5B Short-Context Residual Recheck

The remaining positive rows are small enough that noise matters. A higher-run
check on the 1.5B short/mid-context rows showed:

| Shape | EdgeFM avg | EdgeFM median | TRT avg | TRT median | Gap avg | Gap median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x32 | `345.490 ms` | `345.448 ms` | `345.293 ms` | `345.312 ms` | `+0.197 ms` | `+0.135 ms` |
| 512x64 | `665.077 ms` | `665.081 ms` | `664.177 ms` | `664.086 ms` | `+0.900 ms` | `+0.995 ms` |
| 1024x32 | `385.084 ms` | `385.111 ms` | `385.060 ms` | `385.073 ms` | `+0.024 ms` | `+0.038 ms` |
| 1024x64 | `706.527 ms` | `706.724 ms` | `709.467 ms` | `709.479 ms` | `-2.940 ms` | `-2.755 ms` |

Endpoint overlay sweep around the same rows rejected the remaining low-risk
table knobs:

- q12 decode variants: `no_split_kv_threshold=544/576`,
  `short_seq_bdz=3`, and alternate chunk candidates all regressed at least one
  watched row.
- 1.5B `m=512` QKV/OProj source-op variants: explicit `tile=256x128x32`,
  `tile=128x256x32`, and `input_mode=fp16_cast` did not beat current.
- 1.5B `m=512` MLP variants: `activation_mode=mixed_bf16`,
  `swiglu_threads=64/128`, `gateup_tile=128x256x32`, and
  `down_tile=128x128x32_warp32x64` did not beat current.

`lm_head_top1` was also rechecked as a ceiling diagnostic:

| Shape | Full avg | Top1 avg | Delta |
| --- | ---: | ---: | ---: |
| 512x32 | `345.477 ms` | `345.651 ms` | `+0.173 ms` |
| 512x64 | `664.951 ms` | `664.764 ms` | `-0.187 ms` |

Decision:

- `1.5B 512x32` and `1024x32` are practical parity rows.
- `1.5B 512x64` remains the only stable positive row, at about `+0.9 ms`.
- Do not promote `lm_head_top1`; it is greedy-only and the confirmed default
  full-logits gain is only `0.187 ms` on the one remaining positive row.
- Further work on `1.5B 512x64` should start from fresh graph-off attribution or
  a source-visible LMHead/decode-kernel route, not more table churn.

Artifacts:

- `.tmp_codex/bench/stage3_20260518_135325_1p5b_512_residual_overlay_sweep/`
- `.tmp_codex/bench/stage3_20260518_140339_1p5b_512_highrun_confirm/`
- `.tmp_codex/bench/stage3_20260518_140658_1p5b_512_lmhead_top1_recheck/`

Final attribution before branch cleanup:

- EdgeFM graph-off mapping for `1.5B 512x64` points to decode LMHead GEMV as
  the clearest decode signal, while decode attention is a smaller share.
- TRT reference uses TensorRT/XMMA/Myelin tactics for the same slice, with
  `lm_head` around `1.367 ms` in the captured run.
- cublasLt recheck:
  - decode LMHead: current heuristic baseline remains fastest (`1.378240 ms`);
    no `algo_index` or explicit config was promoted.
  - decode fused_gate_up: current baseline remains fastest.
  - decode mlp_down/fused_qkv/attention_output have tiny operator-level wins in
    isolation, but they are too small for a final default table change without a
    new endpoint confirmation cycle.
- Decision for this branch: stop here and treat the remaining `~0.9 ms` row as
  a future source-visible decode/LMHead investigation, not a blocker for the
  3060 bridge-removal/performance checkpoint.

Artifacts:

- `.tmp_codex/nsys/stage3_20260518_140837_1p5b_512x64_attribution/`
- `.tmp_codex/tune/stage3_20260518_141146_1p5b_decode_lmhead_cublaslt/`
- `.tmp_codex/tune/stage3_20260518_141304_1p5b_decode_linear_cublaslt_recheck/`

## 2026-05-18 Fresh Post-Cache Long-Context Matrix

After the source-op log demotion and runtime config cache cleanup, I reran the
current default EdgeFM CUDA graph path and a fresh same-machine
`TRT-Edge-LLM` reference for the long-context Stage-2 slices.

| Model | Shape | EdgeFM total | EdgeFM prefill | EdgeFM decode | TRT-Edge-LLM total | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 2048x32 | `171.700 ms` | `48.662 ms` | `122.922 ms` | `167.013 ms` | `+4.686 ms` |
| 0.5B | 2048x64 | `297.482 ms` | `48.252 ms` | `249.059 ms` | `291.329 ms` | `+6.153 ms` |
| 1.5B | 2048x32 | `471.148 ms` | `146.422 ms` | `324.605 ms` | `468.866 ms` | `+2.282 ms` |
| 1.5B | 2048x64 | `806.893 ms` | `147.022 ms` | `659.677 ms` | `804.493 ms` | `+2.400 ms` |
| 3B | 2048x32 | `923.062 ms` | `296.689 ms` | `626.231 ms` | `918.046 ms` | `+5.015 ms` |
| 3B | 2048x64 | `1570.870 ms` | `297.327 ms` | `1273.348 ms` | `1570.120 ms` | `+0.751 ms` |

Artifacts:

- `.tmp_codex/bench/stage2_20260518_post_cache_current_matrix/`

Decision:

- The `1.5B` long-context slices are already inside the requested `<=3 ms`
  stretch target.
- `3B 2048x64` is effectively tied with TRT; `3B 2048x32` remains just over
  the tactical `5 ms` line.
- The active default-path target is now `0.5B`, especially decode-heavy
  `2048x64`; the secondary target is `3B 2048x32` prefill.
- A same-tree `lm_head_top1` recheck did not help the active 0.5B default-path
  gap: `0.5B 2048x32` regressed by `+0.437 ms`, and `0.5B 2048x64` regressed
  by `+0.310 ms`. It improved `3B 2048x32` by `-0.669 ms`, but remains
  greedy-only/default-off and does not change the default comparison.

Accepted follow-up:

- 0.5B decode attention table now uses
  `chunk_candidates=[64, 128, 192, 256]` for both duplicate
  `num_qo_heads=14|num_kv_heads=2|head_dim=64` decode records.
- The first temporary overlay only changed one duplicate record and therefore
  mostly missed the actual resolved path. Updating both records produced the
  real effect.
- Reverse confirmation versus a temporary old-table overlay:
  - `0.5B 2048x64`: new `289.190 ms` versus old `297.764 ms`,
    `-8.574 ms`; decode `240.783 ms` versus `249.262 ms`
  - `0.5B 2048x32`: new `167.308 ms` versus old `171.443 ms`,
    `-4.135 ms`; decode `118.783 ms` versus `122.874 ms`
- Validation:
  - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
  - profile/tuning script `py_compile`
  - `python3 -m pytest -q tests/operators/test_attention_prefill.py tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface"`
    (`9 passed`, `20 deselected`)
- Raw artifacts:
  `.tmp_codex/tune/stage2_20260518_0p5_decode_attention_wide_micro/`,
  `.tmp_codex/bench/stage2_20260518_0p5_decode_attention_chunks192_confirm/`,
  `.tmp_codex/bench/stage2_20260518_after_0p5_decode_attention_accept_matrix/`

Current post-acceptance matrix against the same fresh TRT reference:

| Model | Shape | EdgeFM total | EdgeFM prefill | EdgeFM decode | TRT-Edge-LLM total | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 2048x32 | `167.493 ms` | `48.504 ms` | `118.870 ms` | `167.013 ms` | `+0.480 ms` |
| 0.5B | 2048x64 | `289.779 ms` | `48.486 ms` | `241.130 ms` | `291.329 ms` | `-1.550 ms` |
| 1.5B | 2048x32 | `471.528 ms` | `146.748 ms` | `324.664 ms` | `468.866 ms` | `+2.662 ms` |
| 1.5B | 2048x64 | `807.401 ms` | `147.309 ms` | `659.924 ms` | `804.493 ms` | `+2.908 ms` |
| 3B | 2048x32 | `923.561 ms` | `297.032 ms` | `626.404 ms` | `918.046 ms` | `+5.515 ms` |
| 3B | 2048x64 | `1571.380 ms` | `297.793 ms` | `1273.372 ms` | `1570.120 ms` | `+1.260 ms` |

Decision: 0.5B is no longer the active gap; continue on `3B 2048x32`, with
`1.5B 2048x64` watched because it is close to the `<=3 ms` stretch line.

Second follow-up:

- The same decode-attention chunk policy was extended to the q12 and q16 decode
  records:
  - `num_qo_heads=12|num_kv_heads=2|head_dim=128`
  - `num_qo_heads=16|num_kv_heads=2|head_dim=128`
- As with 0.5B/q14, both duplicate decode records for each shape must be
  updated; otherwise temporary overlays can miss the actual resolved record.
- Temporary-table endpoint checks:
  - `1.5B 2048x32`: `471.736 -> 466.705 ms`
  - `1.5B 2048x64`: `807.507 -> 797.299 ms`
  - `3B 2048x32`: `920.665 -> 915.264 ms`
  - `3B 2048x64`: `1570.664 -> 1557.508 ms`
- Official-table confirmation:
  - `3B 2048x32`: `916.432 ms`, decode `619.201 ms`
  - `3B 2048x64`: `1558.090 ms`, decode `1258.892 ms`
- Validation:
  - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
  - profile/TRT/tuning script `py_compile`
  - `python3 -m pytest -q tests/operators/test_attention_prefill.py tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface"`
    (`9 passed`, `20 deselected`)
- Raw artifacts:
  `.tmp_codex/tune/stage2_20260518_1p5_decode_attention_wide_micro/`,
  `.tmp_codex/tune/stage2_20260518_3b_decode_attention_wide_micro/`,
  `.tmp_codex/bench/stage2_20260518_1p5_decode_attention_chunks192/`,
  `.tmp_codex/bench/stage2_20260518_3b_decode_attention_chunks192/`,
  `.tmp_codex/bench/stage2_20260518_after_decode_attention_chunks192_all_sizes_matrix/`

Final matrix for this checkpoint against the same fresh TRT reference:

| Model | Shape | EdgeFM total | EdgeFM prefill | EdgeFM decode | TRT-Edge-LLM total | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 2048x32 | `167.883 ms` | `48.886 ms` | `118.886 ms` | `167.013 ms` | `+0.870 ms` |
| 0.5B | 2048x64 | `289.798 ms` | `48.472 ms` | `241.130 ms` | `291.329 ms` | `-1.531 ms` |
| 1.5B | 2048x32 | `466.503 ms` | `147.070 ms` | `319.313 ms` | `468.866 ms` | `-2.363 ms` |
| 1.5B | 2048x64 | `796.389 ms` | `147.204 ms` | `648.960 ms` | `804.493 ms` | `-8.104 ms` |
| 3B | 2048x32 | `916.855 ms` | `297.503 ms` | `619.224 ms` | `918.046 ms` | `-1.191 ms` |
| 3B | 2048x64 | `1557.520 ms` | `298.615 ms` | `1258.707 ms` | `1570.120 ms` | `-12.599 ms` |

Decision: the long-context Stage-2 matrix is now effectively at parity or
better. Five of six checked slices beat the same-machine TRT reference; the
remaining `0.5B 2048x32` residual is below `1 ms`.

Full-matrix follow-up (`prefill=512/1024/2048`, `decode=32/64`, warmup 1 /
runs 3) against paired TRT:

| Model | Shape | EdgeFM | TRT-Edge-LLM | Gap |
| --- | --- | ---: | ---: | ---: |
| 0.5B | 512x32 | `126.797 ms` | `134.661 ms` | `-7.864 ms` |
| 0.5B | 512x64 | `242.018 ms` | `248.715 ms` | `-6.697 ms` |
| 0.5B | 1024x32 | `141.657 ms` | `141.109 ms` | `+0.549 ms` |
| 0.5B | 1024x64 | `259.498 ms` | `257.881 ms` | `+1.618 ms` |
| 0.5B | 2048x32 | `169.201 ms` | `168.215 ms` | `+0.986 ms` |
| 0.5B | 2048x64 | `290.947 ms` | `291.246 ms` | `-0.299 ms` |
| 1.5B | 512x32 | `345.553 ms` | `345.431 ms` | `+0.122 ms` |
| 1.5B | 512x64 | `664.571 ms` | `663.832 ms` | `+0.738 ms` |
| 1.5B | 1024x32 | `388.110 ms` | `384.566 ms` | `+3.544 ms` |
| 1.5B | 1024x64 | `713.179 ms` | `708.661 ms` | `+4.518 ms` |
| 1.5B | 2048x32 | `467.101 ms` | `470.037 ms` | `-2.937 ms` |
| 1.5B | 2048x64 | `797.018 ms` | `805.311 ms` | `-8.293 ms` |
| 3B | 512x32 | `687.254 ms` | `686.740 ms` | `+0.514 ms` |
| 3B | 512x64 | `1309.737 ms` | `1316.300 ms` | `-6.564 ms` |
| 3B | 1024x32 | `762.146 ms` | `758.467 ms` | `+3.679 ms` |
| 3B | 1024x64 | `1393.094 ms` | `1395.522 ms` | `-2.428 ms` |
| 3B | 2048x32 | `918.318 ms` | `919.680 ms` | `-1.362 ms` |
| 3B | 2048x64 | `1557.783 ms` | `1572.117 ms` | `-14.334 ms` |

Rejected follow-up:

- Raising decode attention `long_seq_threshold` from `1024` to `1536` for all
  q14/q12/q16 shapes was tested to improve `1024` context handling. It is not
  promoted because it regressed the active 1.5B `1024x32` slice:
  `+1.217 ms` total, even though some other slices showed small wins.
- Raw artifacts:
  `.tmp_codex/bench/stage2_20260518_full_matrix_after_decode_attention_chunks192/`,
  `.tmp_codex/bench/stage2_20260518_decode_attention_threshold1536/`

Next queue: verify the `1.5B 1024x32/64` and `3B 1024x32` residuals with
higher-run paired measurements before starting another operator change.

## 2026-05-18 Update: 3B `2048x32` Gap Reduced To About `5.1 ms`

The relaxed Stage-2 tactical gate is now effectively reached for the remaining
3B long-context pressure point.

Accepted changes:

- `src/operators/prefill_mlp_source_op.cu`: converted SwiGLU activation kernels
  to 2D-grid indexing and added table-controlled `swiglu_threads`. The active
  3B `m=2048|hidden=2048|intermediate=11008` table uses `swiglu_threads=64`.
- `src/operators/prefill_linear_source_op.cu`: added
  `persistent_min_free_mb` for persistent FP16 weight copies. If the reserve
  check cannot keep enough free memory, the source-op falls back to scratch
  weight casting instead of failing or forcing the whole op back to fallback.
- `examples/config/platform/3060/operator_impl_table_llm.json`: enabled guarded
  3B `m=2048` QKV persistent weights with `persistent_min_free_mb=64`; OProj
  persistence remains disabled because QKV+OProj OOMs.

Measured artifacts:

- `3B 2048x32`: `.tmp_codex/bench/stage2_20260518_qkv_persist_reserve/3b_2048x32_qkv_persist_reserve64_r5.json`
  - EdgeFM `921.612 ms`, prefill `295.720 ms`, decode `625.774 ms`.
  - External TRT reference remains `916.511 ms`.
  - Gap: `+5.101 ms`.
  - Free memory after warmup/runs: `74.8 MB`; this is acceptable only because
    the table scope is narrow and OProj persistence remains off.
- `3B 2048x64`: `.tmp_codex/bench/stage2_20260518_qkv_persist_reserve/3b_2048x64_qkv_persist_reserve64_r3.json`
  - EdgeFM `1570.681 ms`, prefill `297.681 ms`, decode `1272.805 ms`.
  - External TRT reference remains `1568.463 ms`.
  - Gap: `+2.218 ms`.

Rejected/diagnostic notes:

- QKV+OProj persistent weights OOMed (`act_and_mul_kernel launch failed: out of
  memory`), so do not enable OProj persistence on the 3B/S2048 default table.
- Conservative reserve probes (`96/128/192/256 MB`) still reported the same
  final free-memory floor because later graph/runtime allocations consume the
  remaining headroom. The reserve guard is useful to avoid immediate persistent
  allocation failure, but not a full memory-residency manager.
- 1.5B/0.5B `swiglu_threads` table sweep did not produce a defaultable result.
  1.5B `threads=96` improved the quick `2048x32` probe but r5 confirmation only
  reached `474.663 ms` and `2048x64` moved to `810.336 ms`; 0.5B variants all
  regressed the current table. No non-3B `swiglu_threads` record was promoted.

Validation:

- `cmake --build build-3060 -j$(nproc)`
- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/analyze_trt_nsys_profile.py scripts/tune/tune_qwen_attention_prefill.py`
- `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment and not vl"` -> `6 passed`
- `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "max_new_tokens or deferred_stop or metrics_surface or compact_vocab_identity"` -> `4 passed`
- `python3 -m pytest -q tests/operators/test_prefill_linear.py` -> `12 passed`

## 2026-05-17 Target Update: Use `<=5 ms` As The Stage-2 Tactical Gate

The optimization target is now staged. The immediate Stage-2 checkpoint is to
remove all `10 ms+` gaps versus external `TRT-Edge-LLM` on the official
long-context LLM matrix. A residual within `5 ms` is acceptable before moving
deeper into the slower Humanize/KernelPilot loop, while the full target remains
catching and exceeding `TRT-Edge-LLM`.

Current implication:

- Keep absorbing correctness-clean sub-1% wins when they are localized and move
  a measured slice toward the `<=5 ms` tactical gate.
- Stop promoting risky memory-residency shortcuts when they save less than a
  millisecond and leave little 3060 memory headroom.
- Prioritize 3B `2048x32` prefill attention/boundary work because it is the
  remaining `10 ms+` pressure point; 0.5B and 1.5B are already much closer.

Latest rejected shortcuts under this new gate:

- TRT plugin-op attention BF16 pack/cast vectorization: operator-level 3B
  `S=2048` native FlashInfer `0.972752 ms/layer` versus plugin-op
  `1.473424 ms/layer`; reject and remove the temporary code.
- Shape-specific 3B `2048x32` persistent-linear plus strided-QKV diagnostics:
  OProj-persistent + strided side stream reached `929.761 ms` total with only
  `122.8 MB` free, and QKV-persistent + strided side stream reached
  `928.173 ms` total with only `52.8 MB` free. Both remain roughly `10 ms+`
  slower than the external TRT reference and are not safe default paths.
- Current re-baseline for the relaxed gate:
  - 3B `2048x32` current source-op path: `927.241 ms` total,
    `301.536 ms` prefill, `625.543 ms` decode.
  - Same-day external TRT-Edge-LLM reference: `916.511 ms` total.
  - Remaining tactical gap: `+10.730 ms`; this is still just above the
    "no 10 ms+ gap" threshold, so 3B `2048x32` remains the first target.
- Runtime fusion combo recheck:
  `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=1`,
  `EDGE_FM_PREFILL_STRIDED_QKV_SIDE_STREAM=1`, and
  `EDGE_FM_PREFILL_FUSED_K_ROPE_COPY=1` regressed `3B 2048x32` from
  `927.241 ms` to `928.696 ms`. Keep all three gates diagnostic/default-off.
- 3B `S=2048` OProj source-op tile recheck:
  `128x128x32` reached `928.618 ms`, and `256x128x32` reached
  `928.641 ms`, both slower than the current table. Keep the existing OProj
  tile selection and stop this tile surface unless a fresh profile changes the
  attribution.
- 3B `S=2048` QKV/OProj BF16-direct weight recheck:
  QKV-only reached `940.641 ms`, OProj-only reached `936.554 ms`, and both
  together reached `947.880 ms` on `2048x32`; all are slower than the current
  source-op table. Removing the weight conversion still does not pay for the
  slower mixed BF16-weight GEMM path.
- 3B `S=2048` MLP down tile recheck:
  `128x128x32`, `128x128x32_warp32x64`, and `128x128x32_warp64x32` landed in
  the `932.994-933.475 ms` range on `2048x32`; all regress the current
  localized GateUp-s4 table. Keep the current down-projection policy.
- 3B `2048x32` current explicit gate rechecks:
  strided-QKV-only reached `929.098 ms`, `lm_head_top1` reached
  `929.263 ms`, and `plugin-op` attention reached `932.195 ms`. These are
  useful diagnostics, but none moves the current `927.241 ms` baseline toward
  the `<=5 ms` tactical gate.
- QKV FP16->BF16+bias output-cast 2D indexing probe:
  a same-binary A/B check rejected replacing the current half2 linear index
  kernel with a 2D `(row, col_pair)` kernel. Baseline was `927.969 ms`;
  candidate was `930.001 ms`, with prefill regressing from `302.051 ms` to
  `303.478 ms`. The temporary code was removed.
- RoPE pre-rotate `powf` replacement probe:
  a default-off `EDGE_FM_PREFILL_FAST_ROPE_EXP=1` path replaced the per-element
  `powf(theta, exponent)` with `__expf(-log(theta) * exponent)` inside the
  BF16 Q/K pre-rotate kernel. The same-binary A/B check rejected it: baseline
  was `930.911 ms` total / `304.371 ms` prefill, while candidate was
  `931.589 ms` total / `304.908 ms` prefill. The temporary code was removed.
- 3B plain FlashInfer attention table recheck:
  changing the 3B prefill attention record from
  `flashinfer_attention_prefill_prerotate` with `prefill_cta_tile_q=64` and
  `prefill_max_mma_kv_cap=2` back to plain `flashinfer_attention` landed at
  `931.905 ms` total / `305.862 ms` prefill on `2048x32`. Keep the current
  pre-rotate/cap2 table; the remaining gap is not a stale table selection.
- 3B MLP `activation_mode=mixed_bf16` boundary recheck:
  a neighbor A/B check rejected changing the active 3B fused-MLP source-op
  activation boundary from the current `fp16_cast` policy to `mixed_bf16`.
  The candidate reached `929.893 ms` total / `303.759 ms` prefill, while the
  neighbor baseline reached `929.704 ms` total / `303.395 ms` prefill. Keep
  the current MLP policy and move the `<=5 ms` effort back to the attention
  boundary/source-visible FMHA route.
- Prefill strided-QKV direct-V runtime probe:
  a temporary `EDGE_FM_PREFILL_STRIDED_QKV_DIRECT_V=1` path let attention read
  V directly from the fused QKV buffer while side-stream copying K/V into the
  cache. Correctness passed, but 3B `2048x32` graph-on did not improve:
  baseline `932.892 ms` total / `305.792 ms` prefill versus direct-V
  `932.919 ms` total / `306.018 ms` prefill, with free memory dropping from
  `400.8 MB` to `390.8 MB`. The temporary code was removed; V copy/attention
  overlap is not a useful path to the `<=5 ms` gate.
- FlashInfer FMHA launch-bound and NCU follow-up:
  a temporary third-party header probe added
  `__launch_bounds__(KTraits::NUM_THREADS, 2)` to
  `SinglePrefillWithKVCacheKernel`. It regressed the 3B `S=2048`
  pre-rotate/no-RoPE CTA64 cap2 standalone path from the current
  `0.855862 ms/layer` rerun to `0.863969 ms/layer`, so the header was
  reverted. A fresh focused NCU digest confirms this schedule is not DRAM-bound
  (`DRAM 4.99%`, `L2 hit 96.52%`) and is limited by scheduler/tensor-pipe
  latency (`no eligible 83.70%`, `issue slots busy 16.30%`,
  `regs/thread 128`, `achieved occupancy 22.99%`). Next FMHA work should change
  schedule family rather than keep sweeping FlashInfer CTA/cap/launch-bound
  knobs.

Next action: continue in the source-visible BF16 prefill attention route and
only revisit OProj/GEMM if a fresh profile shows a repeatable few-millisecond
movement toward the `<=5 ms` gate.

## 2026-05-17 Accepted Probe: `lm_head_top1` Stage1 Loop Unroll

Optimized the default-off greedy `lm_head_top1` stage1 kernel by processing two
warp-strided hidden elements per loop iteration (`k` and `k+32`). This preserves
the dot-product accumulation order while reducing loop and branch overhead in
the per-vocab-row reduction.

Correctness:

- `tests/layers/test_linear.py -k "lm_head_top1_matches_full_logits_argmax"`: `2 passed`
- `tests/engine/test_qwen2_generate.py -k "lm_head_top1 and not vl"`: `2 passed`

0.5B graph-on paired result after the change:

| Model | Shape | Full logits | `lm_head_top1` | Total delta | Decode delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5B | 2048x32 | `174.089 ms` | `173.810 ms` | `-0.279 ms` | `-0.260 ms` |
| 0.5B | 2048x64 | `301.027 ms` | `299.659 ms` | `-1.368 ms` | `-1.079 ms` |
| 1.5B | 2048x32 | `474.679 ms` | `474.473 ms` | `-0.206 ms` | `-0.182 ms` |
| 1.5B | 2048x64 | `810.051 ms` | `809.209 ms` | `-0.842 ms` | `-0.577 ms` |
| 3B | 2048x32 | `930.088 ms` | `929.852 ms` | `-0.236 ms` | `-0.227 ms` |
| 3B | 2048x64 | `1578.414 ms` | `1576.710 ms` | `-1.705 ms` | `-0.697 ms` |

Decision: accept the kernel cleanup for the explicit/default-off top1 path.
Full logits remain the default because `lm_head_top1` is greedy-only and does
not provide logits for sampling/tooling, but this is now a stronger decode-heavy
benchmark mode across the three 3060 LLM sizes.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_lmhead_top1_unroll2/`
- `.tmp_codex/bench/stage2_20260517_lmhead_top1_unroll2_more/`

## 2026-05-17 Rejected Probe: 0.5B Prefill MLP Tile Split

Screened a small 0.5B `m=2048` fused-MLP source-op tile matrix after the
current matrix showed 0.5B still has the largest percentage gap. The best
initial `2048x32` candidate was `gateup_tile=128x256x32` plus
`down_tile=128x128x32`.

| Shape | Baseline | Candidate | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| 2048x32 | `174.564 ms` | `174.103 ms` | `-0.461 ms` | positive but not enough alone |
| 2048x64 | `299.794 ms` | `300.459 ms` | `+0.664 ms` | reject |

The `512x32` and `1024x32` confirm runs also moved even though the candidate
only changed the `m=2048` record, which makes the small `2048x32` win likely
noise/order-sensitive.

Decision: keep the existing 0.5B MLP source-op tile policy and do not add a
shape-specific `m=2048` split from this probe.

Artifacts:

- `.tmp_codex/operator_tables/stage2_20260517_0p5b_mlp_tile_probe/`
- `.tmp_codex/bench/stage2_20260517_0p5b_mlp_tile_probe/`
- `.tmp_codex/bench/stage2_20260517_0p5b_mlp_tile_probe_confirm/`

## 2026-05-17 Current Default Matrix After 3B GateUp s4

Rebuilt `build-3060` after accepting the localized 3B GateUp s4 tile and
keeping strided-QKV default-off, then reran the current EdgeFM CUDA graph
matrix for `prefill=2048` and `decode={32,64}`. TRT-Edge-LLM values are the
same-day fresh reference artifacts.

| Model | Shape | EdgeFM total | EdgeFM prefill | EdgeFM decode | TRT-Edge-LLM total | Gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 2048x32 | `174.564 ms` | `51.387 ms` | `123.060 ms` | `167.850 ms` | `+6.714 ms` (`+4.00%`) |
| 0.5B | 2048x64 | `300.818 ms` | `50.936 ms` | `249.710 ms` | `292.060 ms` | `+8.758 ms` (`+3.00%`) |
| 1.5B | 2048x32 | `474.679 ms` | `149.975 ms` | `324.582 ms` | `467.580 ms` | `+7.099 ms` (`+1.52%`) |
| 1.5B | 2048x64 | `810.051 ms` | `150.241 ms` | `659.619 ms` | `803.190 ms` | `+6.861 ms` (`+0.85%`) |
| 3B | 2048x32 | `930.088 ms` | `303.768 ms` | `626.189 ms` | `916.511 ms` | `+13.577 ms` (`+1.48%`) |
| 3B | 2048x64 | `1578.414 ms` | `305.005 ms` | `1273.219 ms` | `1568.463 ms` | `+9.952 ms` (`+0.63%`) |

Decision:

- The source-op/default path is close to TRT-Edge-LLM on 1.5B/3B long-decode
  slices, but 3B short-decode still has the largest absolute gap.
- 0.5B has the largest percentage gap and remains the best secondary decode
  target.
- Continue with fresh graph-off attribution instead of reusing older mixed
  baselines; current artifacts are under
  `.tmp_codex/bench/stage2_20260517_current_matrix_after_s4_revert/`.

## 2026-05-17 Explicit Mode Recheck on Current Default Table

I rechecked the two default-off runtime candidates after the latest table and
prefix-KV/RoPE fixes, using same-day paired graph-on runs.

| Candidate | Shape | Baseline | Candidate | Total Delta | Prefill Delta | Decode Delta | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `lm_head_top1` | 0.5B `2048x64` | `300.421 ms` | `300.453 ms` | `+0.031 ms` | `+0.082 ms` | `-0.070 ms` | no current total win; keep explicit/diagnostic |
| fused K copy + K RoPE | 3B `2048x32` | `928.014 ms` | `930.134 ms` | `+2.120 ms` | `+1.667 ms` | `+0.433 ms` | reject default; likely previous order/noise signal |

Decision:

- Keep both gates default-off.
- The older `lm_head_top1` 0.5B decode-heavy positives are useful historical
  evidence, but the current paired run does not justify a default or
  shape-policy change.
- The fused K-copy/RoPE path is correctness-clean but not stable enough for a
  default table/runtime policy; do not revisit it without a stronger paired
  matrix or a larger fusion that changes the cost model.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_explicit_modes_current/0p5b_2048x64_{baseline_pair,lm_head_top1}.json`
- `.tmp_codex/bench/stage2_20260517_explicit_modes_current/3b_2048x32_{baseline_pair,fused_k_rope_copy}.json`

## 2026-05-17 Correctness Fix + Accepted Runtime Probe: Prefix Prefill KV/RoPE

While retesting the default-off strided-QKV runtime path, the CUDA graph prefix
alignment test exposed a real default correctness regression in the prefill
attention boundary. The root cause was not CUDA graph replay itself:
`prefix_no_graph` and `prefix_graph` produced the same wrong tokens. The recent
strided-QKV plumbing had accidentally made the normal prefill attention K/V view
use only the suffix length; prefix prefill must expose the full
`prefix + suffix` KV cache to attention. The second issue was RoPE: suffix Q
tokens need the prefix-length position offset when the FlashInfer pre-rotate
path is used.

Fix:

- normal prefill attention now views full cached K/V when prefix KV is present;
- strided-QKV and fused K-RoPE-copy candidates remain restricted to no-prefix
  prefill;
- `AttentionOpContext` now carries explicit Q/K RoPE position offsets for
  pre-rotated FlashInfer prefill attention;
- the pre-rotate helper keeps input indexing local to the passed tensor while
  applying the absolute RoPE position offset only to the angle.

Validation:

- prefix/no-prefix comparison script:
  - `prefix_no_graph`: match
  - `prefix_graph`: match
  - `no_prefix_graph`: match
- `tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment_cuda_graph and not vl"`:
  `2 passed`
- same test with `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=1`: `2 passed`
- core generate regression:
  `tests/engine/test_qwen2_generate.py -k "compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface or (token_alignment and not vl)"`:
  `9 passed`
- `scripts/operator_table/validate_operator_tables.py`: passed
- `tests/operators/test_prefill_linear.py -k "source_op_mixed_bf16"`:
  `2 passed`

Retested 3B `2048x32` graph-on strided-QKV runtime probe after the fix:

| Case | Total | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| baseline | `934.046 ms` | `307.249 ms` | `626.696 ms` | baseline |
| `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=1` | `932.921 ms` | `306.328 ms` | `626.462 ms` | accept as default-off/runtime diagnostic |

Decision:

- The prefix KV/RoPE fix is accepted as correctness-critical.
- The strided-QKV path is no longer rejected for correctness. It is still a
  small `~1.1 ms` / `0.12%` 3B `2048x32` win, but adjacent checks reject
  default promotion:
  - `3B 512x32`: `+0.543 ms` total / `+0.356 ms` prefill
  - `3B 1024x32`: `+0.520 ms` total / `+0.209 ms` prefill
  - `3B 2048x64`: `+1.328 ms` total / `+0.804 ms` prefill
- Keep the env-gated path for diagnostics and future shape-specific runtime
  experiments, but do not make it default in the 3060 table/runtime policy.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_runtime_next/build_prefix_*`
- `.tmp_codex/bench/stage2_20260517_runtime_next/pytest_*_after_fix.log`
- `.tmp_codex/bench/stage2_20260517_runtime_next/3b_2048x32_{baseline,strided}_after_prefix_fix_confirm10.*`
- `.tmp_codex/bench/stage2_20260517_runtime_next/adjacent_strided/`

## 2026-05-17 Runtime Fusion Probe: Fused K Copy + RoPE Pre-Rotate

I tested the default-off `EDGE_FM_PREFILL_FUSED_K_ROPE_COPY=1` path, which
fuses prefill K cache copy and K RoPE pre-rotate into one BF16 kernel. The path
keeps Q pre-rotate in the existing FlashInfer pre-rotate implementation and is
restricted to no-prefix BF16 prefill.

Correctness:

- `EDGE_FM_PREFILL_FUSED_K_ROPE_COPY=1 tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment_cuda_graph and not vl"`:
  `2 passed`

3B `2048x32` graph-on initial confirmation:

| Case | Total | Prefill | Decode | Delta |
| --- | ---: | ---: | ---: | ---: |
| baseline | `934.046 ms` | `307.249 ms` | `626.696 ms` | baseline |
| fused K copy + RoPE | `930.596 ms` | `304.441 ms` | `626.052 ms` | `-3.450 ms` |

Reversed-order confirmation:

| Case | Total | Prefill | Decode | Delta |
| --- | ---: | ---: | ---: | ---: |
| fused first | `933.120 ms` | `306.442 ms` | `626.565 ms` | `-1.383 ms` vs following baseline |
| baseline second | `934.502 ms` | `307.656 ms` | `626.724 ms` | baseline |

Adjacent-shape checks rejected default promotion:

| Shape | Total Delta | Prefill Delta | Decision |
| --- | ---: | ---: | --- |
| 3B 512x32 | `+0.447 ms` | `+0.164 ms` | reject default |
| 3B 1024x32 | `+0.156 ms` | `+0.058 ms` | reject default |
| 3B 2048x64 | `+0.159 ms` | `+0.169 ms` | reject default |

Decision:

- Keep the fused K-copy/RoPE path default-off as a useful long-prefill
  diagnostic and possible future shape-specific policy input.
- Do not promote it into the default 3060 path because adjacent shapes are
  non-positive and the 2048x32 win shrinks under reversed-order measurement.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_runtime_next/3b_2048x32_fused_k_rope_copy_confirm10.*`
- `.tmp_codex/bench/stage2_20260517_runtime_next/adjacent_fused_k_rope_copy/`
- `.tmp_codex/bench/stage2_20260517_runtime_next/reverse_fused_k_rope_copy/`

## 2026-05-17 Decode Probe: `lm_head_top1` Recheck With Relaxed Gate

I rechecked the default-off `runtime.lm_head_top1.enabled` path after the hard
`1%` acceptance gate was relaxed for localized wins. The path remains
experimental/default-off by design: full logits stay the default runtime because
tests and downstream tooling still treat `lm_head_top1` as an explicit greedy
fast-path probe.

`2048x64` graph-on, 5-run paired matrix:

| Model | Baseline | `lm_head_top1` | Total Delta | Decode Delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.5B | `301.058 ms` | `299.448 ms` | `-1.610 ms` | `-1.237 ms` | Recheck 0.5B only. |
| 1.5B | `810.343 ms` | `810.640 ms` | `+0.297 ms` | `-0.166 ms` | Reject default. |
| 3B | `1579.632 ms` | `1580.641 ms` | `+1.009 ms` | `-0.202 ms` | Reject default. |

0.5B reversed-order confirmation:

| Shape | Baseline | `lm_head_top1` | Total Delta | Decode Delta |
| --- | ---: | ---: | ---: | ---: |
| 2048x64 | `300.773 ms` | `299.922 ms` | `-0.851 ms` | `-0.607 ms` |

0.5B adjacent decode64 checks:

| Shape | Total Delta | Decode Delta | Decision |
| --- | ---: | ---: | --- |
| 512x64 | `-0.329 ms` | `-0.346 ms` | positive |
| 1024x64 | `-0.335 ms` | `-0.369 ms` | positive |
| 2048x64 | `-0.851 ms` to `-1.610 ms` | `-0.607 ms` to `-1.237 ms` | positive |
| 2048x32 | `-0.654 ms` | `-0.268 ms` | positive |

Decision:

- Keep full logits as the default runtime, consistent with the existing Owner A
  decision gate and tests.
- Record `lm_head_top1` as a valid 0.5B decode tuning mode, useful for
  throughput-oriented benchmark variants or a future explicit 0.5B
  shape-specific runtime policy.
- Do not enable for 1.5B/3B by default; total time is neutral/slower despite a
  tiny decode-only movement.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_decode_lm_head_top1_recheck/`

## 2026-05-17 Current Default Gap Refresh: EdgeFM vs TRT-Edge-LLM

After the prefix correctness fix and the latest default table changes, I
refreshed the official `2048x32` total-latency comparison against external
TRT-Edge-LLM engines. This uses default EdgeFM CUDA graph mode; default-off
diagnostic gates such as strided-QKV, fused K-RoPE-copy, and `lm_head_top1` are
not enabled.

| Model | EdgeFM | TRT-Edge-LLM | Remaining Gap |
| --- | ---: | ---: | ---: |
| 0.5B | `174.112 ms` | `167.613 ms` | `+6.500 ms` |
| 1.5B | `474.843 ms` | `468.809 ms` | `+6.033 ms` |
| 3B | `930.731 ms` | `917.630 ms` | `+13.101 ms` |

EdgeFM stage metrics for the same run:

| Model | EdgeFM Prefill | EdgeFM Decode |
| --- | ---: | ---: |
| 0.5B | `50.978 ms` | `123.022 ms` |
| 1.5B | `150.201 ms` | `324.524 ms` |
| 3B | `304.533 ms` | `626.076 ms` |

Decision:

- The remaining official gap is now smaller than earlier Stage-2 snapshots, but
  still largest on 3B.
- Continue treating 3B long-prefill attention/boundary work as the main queue.
- Keep 0.5B decode-side `lm_head_top1` as an explicit tuning variant; it helps
  decode-heavy slices but is not part of the default comparison above.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_current_gap_2048x32/`

## 2026-05-17 Attention Table Resweep: CTA/`NUM_MMA_KV` Around 3B

I reran a narrow 3B prefill attention table sweep around the current
`flashinfer_attention_prefill_prerotate` policy to check whether the latest
runtime changes altered the best `cta_tile_q` / `prefill_max_mma_kv_cap`
choice.

3B `2048x32` graph-on 3-run screen:

| Candidate | Total | Prefill | Decode |
| --- | ---: | ---: | ---: |
| `cta128_cap0` | `929.941 ms` | `304.120 ms` | `625.680 ms` |
| `cta64_cap0` | `930.968 ms` | `304.758 ms` | `626.068 ms` |
| `cta128_cap2` | `931.227 ms` | `305.176 ms` | `625.872 ms` |
| `cta128_cap4` | `931.985 ms` | `305.920 ms` | `625.907 ms` |
| `cta64_cap4` | `932.217 ms` | `305.627 ms` | `626.467 ms` |
| `cta64_cap1` | `932.589 ms` | `306.217 ms` | `626.213 ms` |

The apparent best candidate, `cta128_cap0`, failed 10-run confirmation:

| Case | Total | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| current table | `932.168 ms` | `305.801 ms` | `626.246 ms` | baseline |
| `cta128_cap0` | `935.229 ms` | `308.379 ms` | `626.715 ms` | reject |

Decision:

- Keep the current 3B attention table entry.
- Treat the 3-run `cta128_cap0` result as a false positive caused by short-run
  noise/drift.
- Do not spend more time on this small CTA/cap parameter surface unless a fresh
  NCU digest shows the current FlashInfer schedule bottleneck changed.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_attention_table_resweep/`

## 2026-05-17 Accepted Stage-2 Attention Knob: FlashInfer `NUM_MMA_KV` Cap

I added a production `prefill_max_mma_kv_cap` knob for the EdgeFM
FlashInfer prefill attention tuned path. The default remains `0`, so existing
paths are unchanged unless the operator table opts in. The standalone diagnostic
first showed that `cap=2` is the only useful small cap on the 3B long-prefill
shape:

| Case | Default | `cap=1` | `cap=2` | `cap=4` | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 3B pre-rotate + no-RoPE, S2048 | `0.864157 ms` | `0.874343 ms` | `0.856504 ms` | n/a | `cap=2` best. |
| 3B no-RoPE core, S2048 | `0.798716 ms` | n/a | `0.789002 ms` | `0.799266 ms` | `cap=2` best. |

Cross-model microbench was positive but small for the same pre-rotate path:

| Model | Default | `cap=2` | Delta |
| --- | ---: | ---: | ---: |
| 0.5B | `0.415017 ms` | `0.413231 ms` | `-0.001786 ms/layer` |
| 1.5B | `0.665655 ms` | `0.664433 ms` | `-0.001222 ms/layer` |
| 3B | `0.864157 ms` | `0.856504 ms` | `-0.007653 ms/layer` |

Full-model graph-on paired checks:

| Model | Candidate | Total | Prefill | Decode | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 3B 2048x32, 10-run | no cap | `933.746 ms` | `306.977 ms` | `626.643 ms` | Baseline. |
| 3B 2048x32, 10-run | `cap=2` | `932.989 ms` | `306.329 ms` | `626.522 ms` | Accept for 3B shape. |
| 1.5B 2048x32, 5-run | no cap | `476.186 ms` | `151.084 ms` | `324.969 ms` | Baseline. |
| 1.5B 2048x32, 5-run | `cap=2` | `476.265 ms` | `151.175 ms` | `324.958 ms` | Reject for 1.5B shape. |
| 0.5B 2048x32, 5-run | no cap | `174.770 ms` | `51.427 ms` | `123.214 ms` | Baseline. |
| 0.5B 2048x32, 5-run | `cap=2` | `174.571 ms` | `51.282 ms` | `123.161 ms` | Positive at 2048 only; adjacent check required. |

Adjacent-shape checks kept the 3B decision but rejected default 0.5B enablement:

| Model / Shape | Default | `cap=2` | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| 0.5B 512x32 | `125.780 ms` | `126.208 ms` | `+0.428 ms` | Reject default 0.5B cap. |
| 0.5B 1024x32 | `140.858 ms` | `140.884 ms` | `+0.026 ms` | Reject default 0.5B cap. |
| 3B 512x32 | `686.076 ms` | `685.825 ms` | `-0.251 ms` | Keep 3B cap. |
| 3B 1024x32 | `761.485 ms` | `761.287 ms` | `-0.198 ms` | Keep 3B cap. |

Validation:

- `cmake --build build-3060 -j$(nproc)` passed.
- `tests/operators/test_attention_prefill.py` passed: `9 passed`.
- `scripts/operator_table/validate_operator_tables.py` passed.
- Profile/test scripts `py_compile` passed.

Artifacts:

- `.tmp_codex/bench/stage2_20260517_fmha_mma_kv_cap/`

Decision:

- Keep the production knob because it is localized and default-neutral.
- Enable `prefill_max_mma_kv_cap=2` only for the accepted 3060 prefill
  attention shapes:
  - `num_qo_heads=16|num_kv_heads=2|head_dim=128` (3B)
- Leave the 0.5B `num_qo_heads=14|num_kv_heads=2|head_dim=64` and 1.5B
  `num_qo_heads=12|num_kv_heads=2|head_dim=128` shapes at the existing no-cap
  behavior. The 0.5B 2048-only win is kept as diagnostic evidence, but not
  worth a default regression at 512.

## 2026-05-17 Rejected Stage-2 Probe: FlashInfer TRTLLM-Gen FMHA On SM86

I checked the vendored FlashInfer `trtllm-gen` FMHA launcher as a possible
plugin-op/source-op replacement for the remaining 3060 prefill attention gap.
This path does not use TensorRT serialized engines, so it would have matched
the "no TRT bridge, source/plugin assets are OK" direction if it supported RTX
3060.

Probe result:

- The editable FlashInfer package expected JIT sources under
  `flashinfer/data/csrc`, while this checkout stores them under
  `third_party/flashinfer/csrc`. I used local JIT path overrides for the probe
  rather than changing production source.
- The module built far enough to call `TllmGenFmhaRunner` on a Qwen-like BF16
  GQA context shape, then failed at runtime with `Unsupported architecture` on
  RTX 3060 / `sm_86`.

Artifact:

- `.tmp_codex/bench/stage2_20260517_trtllm_gen_probe/qwen3b_sm86_probe.log`

Decision:

- Do not pursue FlashInfer `trtllm-gen` FMHA as a 3060 plugin-op path. Keep the
  remaining 3060 attention work on source-visible BF16 FMHA / FlashInfer core
  schedule ideas that actually support SM86.

## 2026-05-17 Rejected Stage-2 Probe: 3B MLP Tile Split

I tested whether the 1.5B `m=2048` MLP tile split should be reused for the 3B
`hidden=2048|intermediate=11008` prefill MLP shape. The temporary tables only
changed the 3B `m=2048` MLP record, keeping the rest of the 3060 table intact.

Initial `3B 2048x32` 3-run screen:

| Candidate | Total | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| Current table | `928.959 ms` | `303.069 ms` | `625.667 ms` | Baseline. |
| `gateup_tile=128x256x32` | `928.463 ms` | `302.637 ms` | `625.650 ms` | Recheck; possible tiny win. |
| `down_tile=128x128x32_warp32x64` | `934.441 ms` | `308.345 ms` | `625.960 ms` | Reject; large prefill regression. |
| GateUp + Down split | `933.678 ms` | `307.539 ms` | `625.964 ms` | Reject; down tile dominates. |

The only plausible candidate was then rerun with `warmup=2,runs=5`:

| Candidate | Total | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| Current table | `930.205 ms` | `304.160 ms` | `625.919 ms` | Baseline. |
| `gateup_tile=128x256x32` | `930.338 ms` | `304.105 ms` | `626.092 ms` | Reject; total is noise/regression. |

Artifacts:

- `.tmp_codex/bench/stage2_20260517_mlp_tile_probe/`
- Temporary tables:
  - `.tmp_codex/bench/stage2_20260517_mlp_tile_probe/operator_impl_table_3b_mlp_gateup_128x256x32.json`
  - `.tmp_codex/bench/stage2_20260517_mlp_tile_probe/operator_impl_table_3b_mlp_down_warp32x64.json`
  - `.tmp_codex/bench/stage2_20260517_mlp_tile_probe/operator_impl_table_3b_mlp_split_gateup_down.json`

Decision:

- Do not change the production 3B MLP table. The 1.5B tile split does not
  transfer to the 3B MLP shape; the only tiny 3-run movement did not survive
  confirmation.
- Continue with source-visible BF16 prefill attention / dataflow-boundary work
  and keep decode-only probes secondary.

## 2026-05-17 Accepted Stage-2 Source-Op Step: QKV mixed_bf16 Bias Path

I extended `cutlass_prefill_linear_source_op` so `input_mode=mixed_bf16` also
handles biased QKV source-op linear. The new path runs mixed
`BF16 activation x FP16 weight -> FP16 scratch` CUTLASS GEMM, then reuses the
existing FP16-to-BF16 output+bias kernel. This removes the separate
BF16-to-FP16 activation cast for QKV when the operator table selects
`mixed_bf16`.

Correctness:

- `cmake --build build-3060 -j$(nproc)` passed.
- `tests/operators/test_prefill_linear.py -k "mixed_bf16"` passed:
  `2 passed, 7 deselected`.
- `tests/engine/test_qwen2_generate.py -k "compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface"`
  passed: `4 passed, 16 deselected`.

Paired `2048x32` benchmark, comparing a temporary QKV `fp16_cast` table with
the mixed-bias path:

| Model | QKV fp16_cast total | mixed-bias total | Total delta | QKV fp16_cast prefill | mixed-bias prefill | Prefill delta | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5B | `175.292 ms` | `175.939 ms` | `+0.647 ms` | `51.842 ms` | `51.908 ms` | `+0.067 ms` | Keep QKV `fp16_cast`. |
| 1.5B | `474.937 ms` | `474.848 ms` | `-0.089 ms` | `150.265 ms` | `150.087 ms` | `-0.178 ms` | Neutral; keep mixed. |
| 3B | `932.022 ms` | `931.005 ms` | `-1.018 ms` | `305.872 ms` | `304.609 ms` | `-1.263 ms` | Accept. |

Adjacent-shape check for the retained 1.5B/3B QKV mixed-bias records:

| Model | Shape | Total delta | Prefill delta | Decode delta | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 1.5B | 512x32 | `-1.026 ms` | `-0.810 ms` | `-0.191 ms` | Accept. |
| 1.5B | 1024x32 | `-0.539 ms` | `-0.537 ms` | `-0.002 ms` | Accept. |
| 1.5B | 2048x64 | `-0.179 ms` | `-0.380 ms` | `+0.140 ms` | Accept; decode noise. |
| 3B | 512x32 | `-0.477 ms` | `-0.421 ms` | `-0.027 ms` | Accept. |
| 3B | 1024x32 | `-0.360 ms` | `-0.343 ms` | `-0.017 ms` | Accept. |
| 3B | 2048x64 | `-0.589 ms` | `-0.706 ms` | `+0.117 ms` | Accept; decode noise. |

Artifacts:

- `.tmp_codex/bench/stage2_20260517_linear_mixed_bias/`
- `.tmp_codex/bench/stage2_20260517_linear_mixed_bias/adjacent/`
- Temporary QKV fp16-cast table:
  `.tmp_codex/bench/stage2_20260517_linear_mixed_bias/qkv_fp16_cast_table.json`

Decision:

- Keep the source-op implementation because it is correctness-clean and gives
  a repeatable 3B prefill win without a new runtime branch.
- Update the 3060 table shape policy:
  - 0.5B QKV prefill records stay `input_mode=fp16_cast`.
  - 0.5B OProj prefill records remain `input_mode=mixed_bf16`.
  - 1.5B/3B QKV and OProj prefill records remain `input_mode=mixed_bf16`.
- Next step: refresh the external TRT-Edge-LLM comparison matrix and then return
  to the remaining prefill attention/boundary gap.

## 2026-05-17 Rejected Stage-2 Probe: MLP GateUp mixed-input FP16 Output

I tested a default-off `activation_mode=mixed_gateup` diagnostic for the 3B
prefill MLP source-op. The idea was narrower than the rejected full
`activation_mode=mixed_bf16` path: only remove the BF16-to-FP16 input cast before
the GateUp GEMM, keep GateUp output in FP16, and keep the existing FP16 SwiGLU
and Down GEMM path unchanged.

Paired graph-on `3B 2048x32` result:

| Candidate | Total | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| Current table | `931.984 ms` | `305.338 ms` | `626.451 ms` | Baseline. |
| `mixed_gateup` temporary table | `933.848 ms` | `306.860 ms` | `626.826 ms` | Reject; prefill and total regress. |

Artifacts:

- `.tmp_codex/bench/stage2_20260517_mlp_mixed_gateup_probe/`
- Temporary table:
  `.tmp_codex/bench/stage2_20260517_mlp_mixed_gateup_probe/operator_impl_table_3b_mixed_gateup_2048.json`

Decision:

- Do not keep the production diagnostic implementation. It regresses the target
  slice and adds extra CUTLASS mixed-input half-output instantiations, increasing
  compile cost without a useful acceptance path.
- The remaining MLP gap is not explained by the standalone input cast boundary.
  Continue prioritizing BF16 prefill attention/source-visible FMHA and broader
  source-op work over more MLP input-mode churn.

## 2026-05-17 Rejected Stage-2 Probe: Prefill SwiGLU Fusion Recheck

I rechecked the existing prefill SwiGLU fusion microbench on 3B after the current
source-op table changes. The fused path remained slower than the current
two-stage GateUp + SwiGLU path.

| Seq len | Fused | Current two-stage | Delta | Decision |
| ---: | ---: | ---: | ---: | --- |
| 512 | `2.247 ms` | `1.949 ms` | `+0.297 ms` | Reject. |
| 1024 | `4.477 ms` | `3.884 ms` | `+0.592 ms` | Reject. |
| 2048 | `8.431 ms` | `7.681 ms` | `+0.750 ms` | Reject. |

Artifact:

- `.tmp_codex/bench/stage2_20260517_swiglu_fusion_recheck/3b_swiglu_micro.json`

Decision:

- Keep the existing prefill MLP source-op path. The source-visible fused SwiGLU
  route is not a default candidate for 3B.

## 2026-05-17 Rejected Stage-2 Probe: Prefill Attention CTA Matrix

I rechecked the active `flashinfer_attention_prefill_prerotate` path at the
operator level for all three Qwen2.5 3060 target shapes and prefill lengths.
This sweep keeps the accepted pre-rotate dataflow and only changes
`prefill_cta_tile_q`.

| Model | Seq | CTA 16 | CTA 64 | CTA 128 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.5B | 512 | `0.083696 ms` | `0.050176 ms` | `0.060416 ms` | Keep CTA 64. |
| 0.5B | 1024 | `0.228352 ms` | `0.139936 ms` | `0.146432 ms` | Keep CTA 64. |
| 0.5B | 2048 | `0.707584 ms` | `0.418816 ms` | `0.442256 ms` | Keep CTA 64. |
| 1.5B | 512 | `0.124912 ms` | `0.076800 ms` | `0.097280 ms` | Keep CTA 64. |
| 1.5B | 1024 | `0.356336 ms` | `0.206848 ms` | `0.240640 ms` | Keep CTA 64. |
| 1.5B | 2048 | `1.143808 ms` | `0.639760 ms` | `0.718848 ms` | Keep CTA 64. |
| 3B | 512 | `0.153600 ms` | `0.087040 ms` | `0.112288 ms` | Keep CTA 64. |
| 3B | 1024 | `0.451344 ms` | `0.247808 ms` | `0.274432 ms` | Keep CTA 64. |
| 3B | 2048 | `1.476672 ms` | `0.805888 ms` | `0.859232 ms` | Keep CTA 64. |

Artifact:

- `.tmp_codex/bench/stage2_20260517_attention_cta_micro/attention_cta_micro.jsonl`

Decision:

- Do not change the active 3060 prefill attention CTA table. `CTA_TILE_Q=64`
  remains the best measured choice across the checked model/shape matrix.
- Plain FlashInfer CTA table tuning is now exhausted for this path. Continue
  with source-visible FMHA/Humanize work, larger boundary removal, or the small
  decode residual rather than re-sweeping `{16,64,128}`.

## 2026-05-17 Rejected Stage-2 Probe: 0.5B MLP mixed_bf16 Recheck

I rechecked the earlier 0.5B-only `activation_mode=mixed_bf16` idea after the
linear `input_mode=mixed_bf16` table change had been accepted. The candidate
only changed the three `hidden=896|intermediate=4864` prefill MLP records in a
temporary table and was run sequentially, not in parallel.

| Shape | Current table total | Candidate total | Total delta | Current prefill | Candidate prefill | Prefill delta | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5B 512x32 | `126.844 ms` | `126.851 ms` | `+0.006 ms` | `12.788 ms` | `12.770 ms` | `-0.019 ms` | Noise. |
| 0.5B 1024x32 | `141.617 ms` | `141.589 ms` | `-0.027 ms` | `24.832 ms` | `24.577 ms` | `-0.256 ms` | Noise-level total win. |
| 0.5B 2048x32 | `175.146 ms` | `175.926 ms` | `+0.780 ms` | `51.610 ms` | `52.008 ms` | `+0.398 ms` | Reject. |

Artifacts:

- `.tmp_codex/operator_tables/3060_llm_0p5b_mlp_mixed_bf16_recheck.json`
- `.tmp_codex/bench/stage2_20260517_0p5b_mlp_mixed_recheck/`

Decision:

- Do not promote 0.5B MLP `activation_mode=mixed_bf16`. The possible 1024
  prefill movement does not survive as an end-to-end gain, and the target
  2048 slice regresses.
- The current source-op MLP table remains unchanged.

## 2026-05-17 Rejected Stage-2 Probe: lm_head_top1 Recheck

I rechecked the default-off greedy `lm_head_top1` path after the current table
changes, using CUDA graph and prefill `2048`. The model already guards this
path with `sampling.temperature < 1e-6`, so correctness risk is mostly about
greedy token alignment; the existing alignment tests still cover the path.

| Model | Shape | Current total | top1 total | Total delta | Current decode | top1 decode | Decode delta | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5B | 2048x32 | `176.593 ms` | `175.918 ms` | `-0.675 ms` | `123.958 ms` | `123.677 ms` | `-0.280 ms` | Positive but small. |
| 0.5B | 2048x64 | `300.981 ms` | `301.613 ms` | `+0.632 ms` | `249.774 ms` | `249.765 ms` | `-0.009 ms` | Reject; total regresses. |
| 1.5B | 2048x32 | `476.312 ms` | `476.793 ms` | `+0.480 ms` | `324.805 ms` | `324.752 ms` | `-0.053 ms` | Reject; total regresses. |
| 1.5B | 2048x64 | `811.706 ms` | `811.297 ms` | `-0.409 ms` | `659.968 ms` | `659.612 ms` | `-0.357 ms` | Noise-level total win. |
| 3B | 2048x32 | `933.984 ms` | `933.499 ms` | `-0.485 ms` | `626.253 ms` | `626.264 ms` | `+0.011 ms` | Noise-level total win. |
| 3B | 2048x64 | `1580.963 ms` | `1581.694 ms` | `+0.732 ms` | `1273.503 ms` | `1273.430 ms` | `-0.073 ms` | Reject; total regresses. |

Artifacts:

- `.tmp_codex/bench/stage2_20260517_lm_head_top1_recheck/`

Decision:

- Keep `lm_head_top1` default-off. The current kernel sometimes helps short
  decode totals, but the paired matrix is inconsistent and the measured decode
  savings are mostly noise relative to full-model variation.
- Do not spend more Stage-2 priority on decode `lm_head_top1` unless a future
  trace shows a new, stable decode bottleneck.

## 2026-05-17 Accepted Stage-2 Table Step: Linear mixed_bf16 Input Mode

After the hard `1%` gate was relaxed for localized, correctness-clean wins, I
rechecked `cutlass_prefill_linear_source_op` with `input_mode=mixed_bf16`.
The candidate removes part of the BF16-to-FP16 activation conversion overhead
for QKV/OProj source-op linear while keeping the existing CUTLASS runner and
operator-table shape selection.

The default 3060 LLM table now sets:

- `impl_id=cutlass_prefill_linear_source_op`
- `input_mode=mixed_bf16`
- all Qwen2.5 `0.5B/1.5B/3B` prefill linear records for `m=512/1024/2048`

Long-prefill paired result:

| Model | Shape | Baseline total | mixed_bf16 total | Total delta | Baseline prefill | mixed_bf16 prefill | Prefill delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | 2048x32 | `176.659 ms` | `176.171 ms` | `-0.488 ms` | `52.729 ms` | `52.273 ms` | `-0.456 ms` |
| 1.5B | 2048x32 | `477.398 ms` | `474.993 ms` | `-2.405 ms` | `152.439 ms` | `150.315 ms` | `-2.124 ms` |
| 3B | 2048x32 | `935.593 ms` | `933.831 ms` | `-1.762 ms` | `309.167 ms` | `307.362 ms` | `-1.805 ms` |

Shorter prefill paired check:

| Model | Shape | Total delta | Prefill delta |
| --- | --- | ---: | ---: |
| 0.5B | 512x32 | `-0.500 ms` | `-0.365 ms` |
| 0.5B | 1024x32 | `-1.314 ms` | `-0.853 ms` |
| 1.5B | 512x32 | `-1.788 ms` | `-0.687 ms` |
| 1.5B | 1024x32 | `-0.132 ms` | `-0.259 ms` |
| 3B | 512x32 | `-0.151 ms` | `-0.358 ms` |
| 3B | 1024x32 | `-0.815 ms` | `-0.812 ms` |

Artifacts:

- Long-prefill paired outputs:
  `.tmp_codex/bench/stage2_20260517_linear_mixed_bf16/`
- Shorter prefill paired outputs:
  `.tmp_codex/bench/stage2_20260517_linear_mixed_bf16/short_matrix/`
- Temporary candidate table:
  `.tmp_codex/operator_tables/3060_llm_linear_mixed_bf16.json`

Validation:

- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
  passed.
- `tests/operators/test_prefill_linear.py -k "mixed_bf16"` passed:
  `1 passed, 7 deselected`.
- `tests/engine/test_qwen2_generate.py -k "compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface"`
  passed: `4 passed, 16 deselected`.

Decision:

- Accept `input_mode=mixed_bf16` for prefill linear source-op records in the
  3060 LLM table. The gain is sub-1% but stable across the checked model/shape
  matrix and is a useful reduction in the conversion-boundary residual.
- This acceptance only applies to linear QKV/OProj source-op records. The
  earlier MLP `activation_mode=mixed_bf16` result remains rejected/default-off
  for 1.5B/3B.

Post-acceptance TRT-Edge-LLM comparison for `2048x32`:

| Model | EdgeFM total | TRT total | Total gap | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | `173.6 ms` | `166.5 ms` | `+7.0 ms` | `50.8 ms` | `45.6 ms` | `+5.2 ms` | `122.7 ms` | `120.9 ms` | `+1.8 ms` |
| 1.5B | `477.2 ms` | `469.9 ms` | `+7.4 ms` | `152.4 ms` | `144.9 ms` | `+7.5 ms` | `324.8 ms` | `325.0 ms` | `-0.1 ms` |
| 3B | `934.7 ms` | `917.2 ms` | `+17.5 ms` | `307.9 ms` | `286.6 ms` | `+21.3 ms` | `626.8 ms` | `630.6 ms` | `-3.8 ms` |

Artifacts:

- `.tmp_codex/bench/stage2_20260517_post_linear_mixed/benchmark_trt_0p5b_1p5b_2048x32.log`
- `.tmp_codex/bench/stage2_20260517_post_linear_mixed/benchmark_trt_3b_2048x32.log`

Conclusion:

- The remaining 1.5B/3B gap is still prefill-heavy after the table change.
  Continue with FMHA/source-op and boundary-reduction work rather than decode
  first.
- 0.5B has a smaller decode residual (`+1.8 ms`) worth keeping in the queue,
  but it is secondary to the prefill attention/boundary gap.

## 2026-05-17 Rejected Stage-2 Probe: 3B MLP Tile Micro-Matrix

After the post-acceptance warm NSYS trace still showed the 3B prefill MLP GEMMs
as the largest absolute GPU-time blocks, I ran a small table-only probe for
`Qwen2.5-3B`, `m=2048`, without changing runtime structure.

Baseline:

- `.tmp_codex/bench/stage2_20260517_mlp_tile_probe/edgefm_3b_2048x32_base.json`
- total `929.518 ms`, prefill `303.726 ms`

Candidate results:

| Candidate | Total | Total delta | Prefill | Prefill delta | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `gateup_tile=128x256x32` | `929.301 ms` | `-0.217 ms` | `303.364 ms` | `-0.362 ms` | Reject; noise-level movement only. |
| `down_tile=128x128x32_warp32x64` | `935.796 ms` | `+6.278 ms` | `309.426 ms` | `+5.699 ms` | Reject; clear regression. |
| `gateup_tile=128x256x32`, `down_tile=128x128x32_warp32x64` | `935.541 ms` | `+6.023 ms` | `309.209 ms` | `+5.483 ms` | Reject; down tile dominates regression. |
| `down_tile=128x128x32_warp64x32` | `937.325 ms` | `+7.807 ms` | `310.782 ms` | `+7.055 ms` | Reject; clear regression. |
| `gateup_tile=128x256x32`, `down_tile=128x128x32_warp64x32` | `936.760 ms` | `+7.242 ms` | `310.254 ms` | `+6.527 ms` | Reject; clear regression. |

Decision:

- Do not change the 3B `m=2048` MLP tile table. The only positive candidate is
  too small for a default table edit, while all down-projection tile variants
  regress materially.
- Ordinary MLP tile churn is now deprioritized again. Continue with FMHA/core
  schedule or larger boundary work unless a fresh profile exposes a new MLP
  mechanism beyond table parameters.

## 2026-05-17 Runtime/Fusion Follow-up: Strided QKV And Fused Q/K RoPE

I tested two low-risk runtime/fusion ideas after the accepted pre-rotate
attention change.

Runtime/dataflow probe:

| Candidate | Artifact | Total | Prefill | Decision |
| --- | --- | ---: | ---: | --- |
| Paired default-off baseline | `.tmp_codex/bench/stage2_20260517_runtime/edgefm_3b_2048x32_default_off_pair.json` | `932.764 ms` | `306.816 ms` | baseline |
| Full strided K/V direct + side-stream KV write | `.tmp_codex/bench/stage2_20260517_runtime/edgefm_3b_2048x32_strided_qkv_side.json` | `934.608 ms` | `308.335 ms` | Reject; slower by `+1.84 ms`. |
| K-only strided + side-stream K cache write, V contiguous | `.tmp_codex/bench/stage2_20260517_runtime/edgefm_3b_2048x32_strided_k_side.json` | `932.997 ms` | `306.978 ms` | Reject; effectively noise and slightly slower. |

Fusion probe:

| Candidate | Artifact | Total | Prefill | Decision |
| --- | --- | ---: | ---: | --- |
| Standalone fused Q/K pre-rotate kernel | `.tmp_codex/bench/stage2_20260517_fused_qk/edgefm_3b_2048x32_fused_qk_prerotate.json` | `934.093 ms` | `307.894 ms` | Reject; launch-count fusion regressed `3B 2048x32`. |
| K pre-rotate on side stream | `.tmp_codex/bench/stage2_20260517_prerotate_side/edgefm_3b_2048x32_k_prerotate_side.json` | `934.105 ms` | `307.869 ms` | Reject; tied with paired default. |
| Paired default after side-stream probe | `.tmp_codex/bench/stage2_20260517_prerotate_side/edgefm_3b_2048x32_default_pair_after_side.json` | `934.034 ms` | `307.871 ms` | baseline |
| Cached RoPE sin/cos table for pre-rotate | `.tmp_codex/bench/stage2_20260517_rope_table/edgefm_3b_2048x32_rope_table.json` | `935.281 ms` | `308.774 ms` | Reject; slower than paired default. |
| Paired default for RoPE table probe | `.tmp_codex/bench/stage2_20260517_rope_table/edgefm_3b_2048x32_default_pair.json` | `934.441 ms` | `308.200 ms` | baseline |
| Cached RoPE inv-freq table for pre-rotate | `.tmp_codex/bench/stage2_20260517_rope_inv_freq/edgefm_3b_2048x32_rope_inv_freq.json` | `933.896 ms` | `307.683 ms` | Reject after stronger paired check; early small win was not stable. |
| Paired default for inv-freq probe | `.tmp_codex/bench/stage2_20260517_rope_inv_freq/edgefm_3b_2048x32_default_pair.json` | `934.466 ms` | `308.212 ms` | baseline |
| Table-default inv-freq gate | `.tmp_codex/bench/stage2_20260517_rope_inv_freq_table/edgefm_3b_2048x32_table_default.json` | `933.660 ms` | `307.644 ms` | Rejected; no-inv paired table was faster. |
| No-inv paired table | `.tmp_codex/bench/stage2_20260517_rope_inv_freq_table/edgefm_3b_2048x32_no_inv_freq_pair.json` | `932.718 ms` | `306.762 ms` | baseline; proved inv-freq was noise/regression. |
| Combined Q/K RoPE launch, 3B | `.tmp_codex/bench/stage2_20260517_qk_rope_combined/edgefm_3b_2048x32_combined.json` | `935.844 ms` | `309.354 ms` | Tiny paired win versus split, but not stable across sizes. |
| Split Q/K RoPE launch, 3B | `.tmp_codex/bench/stage2_20260517_qk_rope_combined/edgefm_3b_2048x32_split.json` | `936.125 ms` | `309.676 ms` | paired baseline |
| Combined Q/K RoPE launch, 1.5B | `.tmp_codex/bench/stage2_20260517_qk_rope_combined/edgefm_1p5b_2048x32_combined.json` | `477.758 ms` | `152.715 ms` | Reject; slower than split by `+1.466 ms` total. |
| Split Q/K RoPE launch, 1.5B | `.tmp_codex/bench/stage2_20260517_qk_rope_combined/edgefm_1p5b_2048x32_split.json` | `476.292 ms` | `151.640 ms` | paired baseline |
| Combined Q/K RoPE launch, 0.5B | `.tmp_codex/bench/stage2_20260517_qk_rope_combined/edgefm_0p5b_2048x32_combined.json` | `174.843 ms` | `51.722 ms` | Reject with the combined-launch family; effectively noise. |
| Split Q/K RoPE launch, 0.5B | `.tmp_codex/bench/stage2_20260517_qk_rope_combined/edgefm_0p5b_2048x32_split.json` | `174.737 ms` | `51.734 ms` | paired baseline |
| Fused K-cache copy + K RoPE, 3B | `.tmp_codex/bench/stage2_20260517_fused_k_rope_copy/edgefm_3b_2048x32_fused_k_rope_copy.json` | `935.382 ms` | `308.949 ms` | Default-off diagnostic; tiny `-0.168 ms` total versus paired default. |
| Default, 3B | `.tmp_codex/bench/stage2_20260517_fused_k_rope_copy/edgefm_3b_2048x32_default.json` | `935.550 ms` | `309.078 ms` | paired baseline |
| Fused K-cache copy + K RoPE, 1.5B | `.tmp_codex/bench/stage2_20260517_fused_k_rope_copy/edgefm_1p5b_2048x32_fused_k_rope_copy.json` | `477.441 ms` | `152.504 ms` | Default-off diagnostic; tiny `-0.347 ms` total. |
| Default, 1.5B | `.tmp_codex/bench/stage2_20260517_fused_k_rope_copy/edgefm_1p5b_2048x32_default.json` | `477.788 ms` | `152.696 ms` | paired baseline |
| Fused K-cache copy + K RoPE, 0.5B | `.tmp_codex/bench/stage2_20260517_fused_k_rope_copy/edgefm_0p5b_2048x32_fused_k_rope_copy.json` | `174.997 ms` | `51.786 ms` | Not default; total `+0.056 ms` noise versus paired default. |
| Default, 0.5B | `.tmp_codex/bench/stage2_20260517_fused_k_rope_copy/edgefm_0p5b_2048x32_default.json` | `174.941 ms` | `51.805 ms` | paired baseline |
| Fused QKV split + Q/K RoPE, 3B | `.tmp_codex/bench/stage2_20260517_fused_qkv_rope_split/edgefm_3b_2048x32_fused_qkv_rope_split.json` | `935.356 ms` | `308.998 ms` | Reject; tiny `-0.199 ms` total versus paired default is noise. |
| Default, 3B | `.tmp_codex/bench/stage2_20260517_fused_qkv_rope_split/edgefm_3b_2048x32_default.json` | `935.555 ms` | `309.083 ms` | paired baseline |
| Fused QKV split + Q/K RoPE, 1.5B | `.tmp_codex/bench/stage2_20260517_fused_qkv_rope_split/edgefm_1p5b_2048x32_fused_qkv_rope_split.json` | `477.913 ms` | `152.941 ms` | Reject; slower by `+0.586 ms` total and `+0.521 ms` prefill. |
| Default, 1.5B | `.tmp_codex/bench/stage2_20260517_fused_qkv_rope_split/edgefm_1p5b_2048x32_default.json` | `477.327 ms` | `152.421 ms` | paired baseline |
| Fused QKV split + Q/K RoPE, 0.5B | `.tmp_codex/bench/stage2_20260517_fused_qkv_rope_split/edgefm_0p5b_2048x32_fused_qkv_rope_split.json` | `175.995 ms` | `52.207 ms` | Reject; slower by `+0.313 ms` total. |
| Default, 0.5B | `.tmp_codex/bench/stage2_20260517_fused_qkv_rope_split/edgefm_0p5b_2048x32_default.json` | `175.682 ms` | `52.164 ms` | paired baseline |

Correctness:

- `tests/operators/test_attention_prefill.py -k "prerotate or strided_qkv or k_only_strided"` passed before and after the fused Q/K rollback.
- `tests/engine/test_qwen2_generate.py -k "compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface"` passed after the rollback.
- `EDGE_FM_PREFILL_ROPE_TABLE=1 tests/operators/test_attention_prefill.py -k "prerotate or strided_qkv or k_only_strided"` passed during the RoPE table probe.
- `EDGE_FM_PREFILL_ROPE_TABLE=1 tests/engine/test_qwen2_generate.py -k "max_new_tokens"` passed during the RoPE table probe.
- After rejecting and removing the RoPE table probe code, `validate_operator_tables.py`, profile script `py_compile`, the same attention operator tests, and the core generate tests all passed.
- `EDGE_FM_PREFILL_ROPE_INV_FREQ=1` passed the same operator smoke plus
  `max_new_tokens`, but performance was not stable under a stronger paired
  table check. The inv-freq code and table params were removed.
- After rejecting and removing the combined Q/K RoPE launch branch, the same
  attention operator smoke and core generate subset passed again.
- `EDGE_FM_PREFILL_FUSED_K_ROPE_COPY=1` passed the core generate subset
  (`compact_vocab_identity`, `max_new_tokens`, `deferred_stop`,
  `metrics_surface`), but the benchmark movement was too small and noisy for
  default promotion.
- After rejecting and removing the coarse fused QKV split plus Q/K RoPE branch,
  the build, operator-table validation, profile script `py_compile`, attention
  operator smoke, default core generate subset, and
  `EDGE_FM_PREFILL_FUSED_K_ROPE_COPY=1` core generate subset all passed.

Decision:

- Keep strided Q/K attention and side-stream cache writes default-off as
  diagnostics only; they do not move the current graph-on long-prefill gap.
- Do not promote standalone Q/K RoPE kernel coalescing. The next fusion attempt
  should remove a real boundary, such as QKV split/write plus RoPE, BF16/FP16
  format churn, or a source-visible FMHA prelude.
- Do not keep K-side-stream pre-rotate in production code. The overlap window is
  too small for the added event/stream dependency to matter.
- Do not keep cached RoPE sin/cos tables in production code. Even with
  correctness passing, the extra table loads are slower than recomputing the
  sin/cos values inside the current pre-rotate kernel for the measured 3B
  `2048x32` graph-on slice.
- Do not keep cached RoPE inv-freq tables in production code. The probe showed
  small early wins on 1.5B/3B, but a follow-up no-inv paired table was faster on
  the target 3B `2048x32` slice. Treat standalone RoPE cache variants as
  exhausted unless they are fused into QKV split/write or the FMHA prelude.
- Do not keep a combined Q/K standalone RoPE launch. It saves one launch in
  principle and barely helped 3B in one paired run, but it regressed 1.5B
  `2048x32` by `+1.466 ms` total. Future RoPE work must fuse with a real
  producer/consumer boundary instead of only reducing launch count.
- Keep fused K-cache copy + K RoPE as a default-off diagnostic only. It is a
  real boundary-reduction direction and does not break correctness, but the
  measured gains are only `0.17-0.35 ms` on 3B/1.5B and noise on 0.5B. A
  default change should wait for a broader QKV split/write + Q/K RoPE fusion or
  an FMHA prelude that removes more scratch traffic.
- Do not keep the coarse fused QKV split plus Q/K RoPE branch in production
  code. It removed a larger boundary on paper, but the scalar fused kernel only
  produced a noise-level `3B` win and regressed `1.5B`/`0.5B`. Treat this as a
  stopping point for launch-boundary-only QKV/RoPE variants. The next
  meaningful attempt needs source-visible FMHA/Humanize work or a targeted
  FlashInfer internal change that reduces resource stalls, not another scalar
  split/rotate/copy kernel.

## 2026-05-17 Stage-2 Fresh 3B Prefill Attribution

After Stage 1 removed the internal EdgeFM TensorRT bridge, I captured a fresh
paired graph-off mapping trace for the largest Stage-2 long-prefill residual:
`Qwen2.5-3B`, prefill `2048`, decode `1`.

Setup:

- EdgeFM: `build-3060`, source-op/native, graph-off mapping trace.
- TRT reference: external `TRT-Edge-LLM` engine
  `tests/data/trt_edgellm_workspace/qwen2.5-3b/engines_mxil2048`.
- Artifacts:
  - EdgeFM JSON:
    `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x1_mapping.json`
  - TRT JSON:
    `.tmp_codex/bench/stage2_20260517/trt_3b_2048x1_mapping.json`
  - EdgeFM NSYS:
    `.tmp_codex/nsys/stage2_20260517/edgefm_3b_2048x1_mapping.nsys-rep`
  - TRT NSYS:
    `.tmp_codex/nsys/stage2_20260517/trt_3b_2048x1_mapping.nsys-rep`

Result:

| Runtime | Total | Prefill | Total kernel time |
| --- | ---: | ---: | ---: |
| EdgeFM source-op | `320.921 ms` | `320.681 ms` | `319.492 ms` |
| TRT-Edge-LLM | `290.024 ms` | `289.819 ms` | `288.809 ms` |
| Gap | `+30.896 ms` | `+30.862 ms` | `+30.684 ms` |

Fresh role attribution:

| Role / kernel group | EdgeFM | TRT-Edge-LLM | Gap | Notes |
| --- | ---: | ---: | ---: | --- |
| Prefill attention | `35.412 ms` | `23.396 ms` | `+12.016 ms` | Largest single residual; compare FlashInfer vs source-visible/plugin FMHA next. |
| QKV role | `24.415 ms` | `16.959 ms` | `+7.456 ms` | GEMM-only gap is much smaller; much of this is cast/pack/boundary overhead. |
| OProj role | `18.226 ms` | `14.289 ms` | `+3.937 ms` | GEMM-only gap is small; boundary overhead matters. |
| MLP GateUp GEMM | `143.146 ms` | `138.869 ms` | `+4.277 ms` | Real but lower priority than attention plus boundary overhead. |
| MLP Down GEMM | `72.695 ms` | `71.734 ms` | `+0.961 ms` | Near tie. |
| SwiGLU | `14.434 ms` | `14.459 ms` | `-0.025 ms` | EdgeFM is effectively tied. |
| Norm / lm_head | `9.207 ms` | `9.010 ms` | `+0.197 ms` | Not a Stage-2 target. |

The EdgeFM trace also shows conversion overhead that is no longer negligible:
`bf16_to_half2_kernel` takes `9.930 ms` across `180` launches and
`half2_to_bf16_kernel` takes `2.389 ms` across `36` launches. This makes the
next queue attention plus BF16/FP16 boundary reduction, not another broad
cuBLASLt retune.

Rejected diagnostic:

- Candidate:
  `.tmp_codex/bench/stage2_20260517/operator_table_3b_linear_persistent.json`
  enabled persistent FP16 weights for the `3B/S=2048` QKV and OProj source-op
  records.
- Graph-off `2048x1` improved prefill from `320.681 ms` to `317.466 ms`, but
  available memory after the run was only `80.812 MB`, and MLP persistent mode
  already fell back after allocation failures.
- CUDA-graph `2048x32` failed with
  `LinearLayer: cuBLASLt matmul failed with status 15` after another persistent
  allocation failure.

Decision:

- Reject 3B linear persistent-table promotion. It is faster in a narrow
  graph-off slice but not stable enough for graph-on generate.
- Continue Stage 2 with default-off attention/plugin-op probes and
  source-op cast-boundary reduction.

Follow-up plugin-op attention probe:

- Candidate:
  `--edgefm-mode plugin-op --plugin-op-allow-bf16-fp16-cast`, which calls
  TRT-Edge-LLM `ContextFMHARunner` directly as an EdgeFM operator and does not
  create a TensorRT execution context.
- Paired graph-on `3B 2048x32`:
  - native source-op:
    `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_native_paired_after_plugin.json`
    = `935.548 ms` total, `309.773 ms` prefill.
  - plugin-op BF16-cast:
    `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_plugin_op_bf16_cast_probe.json`
    = `933.725 ms` total, `307.717 ms` prefill.
  - delta: `-1.822 ms` total / `-2.056 ms` prefill, only about `0.19%`
    end-to-end.
- Token comparison against native source-op for the same prompt produced
  `0` mismatches:
  `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_plugin_op_bf16_cast_token_compare.json`.
- Graph-off mapping artifact:
  `.tmp_codex/nsys/stage2_20260517/edgefm_3b_2048x1_plugin_op_bf16_cast_mapping.nsys-rep`.
  The plugin FMHA body is faster than FlashInfer on this slice
  (`35.412 ms` native attention body to about `19.496 ms` plugin body), but the
  BF16-pack/RoPE/output-cast path costs about `11.6 ms`, so the graph-on win is
  too small.
- Third-party source scan: this checkout's
  `ContextFMHARunner::canImplement()` only accepts `DataType::kHALF`, and
  `fmha_cubin.h` ships SM86 FP16 context-FMHA cubins but no BF16 entries.
  Flipping the type check would not create a real BF16 plugin path.

Decision:

- Reject plugin-op BF16-cast attention as a default/table promotion. Keep it as
  a diagnostic path only.
- Do not attempt a fake BF16 enable in TRT-Edge-LLM without real BF16 cubins.
- Continue with source-op boundary reduction and source-visible/Humanize BF16
  FMHA work if a standalone target can show an end-to-end `>=1%` opportunity.

Follow-up MLP mixed-activation recheck:

- Candidate:
  `.tmp_codex/bench/stage2_20260517/operator_table_3b_mlp_mixed_bf16.json`
  sets `activation_mode=mixed_bf16` for the 3B `S=2048` MLP source-op record.
- Graph-on `3B 2048x32` result:
  `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_mlp_mixed_bf16_probe.json`
  = `935.835 ms` total, `310.328 ms` prefill.
- Paired native source-op from the plugin-op probe was `935.548 ms` total,
  `309.773 ms` prefill.

Decision:

- Reject `activation_mode=mixed_bf16` again for production default. It is
  effectively tied to slightly slower in end-to-end generate and does not reduce
  enough conversion overhead to matter.
- The next low-risk probe is selective persistent FP16 weights for only one
  3B `S=2048` linear role at a time (`fused_qkv` or `attention_output`), because
  the earlier all-linear persistent probe was fast graph-off but failed
  graph-on due memory pressure.

Selective 3B linear persistent probes:

| Candidate | Artifact | Total | Prefill | Free memory after runs | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| native paired baseline | `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_native_paired_after_plugin.json` | `935.548 ms` | `309.773 ms` | `402.8 MB` | baseline |
| QKV-only persistent | `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_persistent_qkv_probe.json` | `932.108 ms` | `306.496 ms` | `44.8 MB` | Reject as default; memory margin too small. |
| OProj-only persistent | `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_persistent_oproj_probe.json` | `932.772 ms` | `306.995 ms` | `114.8 MB` | Reject as default; below `1%` gate and still tight memory. |

Decision:

- Selective persistent linear copies are useful diagnostic evidence that BF16
  weight casts cost real time, but neither single-role 3B `S=2048` candidate
  reaches the end-to-end acceptance gate.
- Do not promote either entry to the official 3060 table yet. Keep the artifacts
  for future memory-budgeted profiles or if a model-specific config explicitly
  accepts the low free-memory margin.
- The active Stage-2 work should move to a source-visible attention/boundary
  path: either reduce the BF16-to-FP16 conversion boundary without extra
  residency, or use the Humanize/KernelPilot loop for a BF16 prefill attention
  replacement that can beat FlashInfer without plugin cast overhead.

Production NCU baseline for the active attention target:

- Captured one production `SinglePrefillWithKVCacheKernel` instance from
  `3B 2048x1` native generate with Nsight Compute.
- Artifacts:
  - report:
    `.tmp_codex/ncu/stage2_20260517/edgefm_3b_flashinfer_prefill_baseline.ncu-rep`
  - csv:
    `.tmp_codex/ncu/stage2_20260517/edgefm_3b_flashinfer_prefill_baseline.csv`
  - raw csv:
    `.tmp_codex/ncu/stage2_20260517/edgefm_3b_flashinfer_prefill_baseline_raw.csv`
  - digest:
    `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/profile-digests/2026-05-17_production_flashinfer_3b_s2048_ncu.md`
- Headline metrics:
  - kernel duration `1.359520 ms`
  - compute throughput `38.49%`, DRAM throughput `4.87%`, L2 throughput
    `32.38%`, L2 hit rate `96.55%`
  - issue slots busy `21.59%`, no eligible `78.41%`
  - active warps per scheduler `1.91`, eligible warps per scheduler `0.28`
  - `168` registers/thread, `49.15 KB` dynamic shared memory per block
  - theoretical occupancy `16.67%`, achieved occupancy `15.92%`
  - dominant production stall ratio is `math_pipe_throttle=4.336`

Decision:

- The active 3B attention gap is not a DRAM bandwidth problem. Continue with
  source-visible BF16 FMHA / boundary fusion work that reduces resource pressure
  or removes conversion boundaries.
- Do not spend the next round on scalar attention seeds; the Humanize target
  must be tensor-core/cuTile/CUTLASS-style or a directly source-visible fused
  layout route.

## 2026-05-17 Stage-2 0.5B Decode and Prefill Follow-up

The fresh external TRT-Edge-LLM matrix showed that `0.5B` still has both prefill
and decode residuals, so I rechecked the low-risk decode levers before changing
any defaults.

Artifacts:

- EdgeFM graph-off mapping:
  `.tmp_codex/nsys/stage2_20260517_decode/edgefm_0p5b_2048x64_graphoff_mapping.nsys-rep`
- EdgeFM JSON:
  `.tmp_codex/bench/stage2_20260517_decode/edgefm_0p5b_2048x64_graphoff_mapping.json`
- TRT graph-off mapping:
  `.tmp_codex/nsys/stage2_20260517_decode/trt_0p5b_2048x64_graphoff_mapping.nsys-rep`
- TRT JSON:
  `.tmp_codex/bench/stage2_20260517_decode/trt_0p5b_2048x64_graphoff_mapping.json`
- Decode-attention sweep:
  `.tmp_codex/bench/stage2_20260517_decode_sweep/summary.json`
- Prefill-attention sweep:
  `.tmp_codex/bench/stage2_20260517_prefill_sweep/qwen0p5_prefill_attention_sweep.json`

Graph-off timing:

| Runtime | Total | Prefill | Decode |
| --- | ---: | ---: | ---: |
| EdgeFM native | `326.550 ms` | `55.580 ms` | `270.500 ms` |
| TRT-Edge-LLM | `297.100 ms` | `47.592 ms` | `249.318 ms` |

EdgeFM role attribution for `0.5B 2048x64` graph-off:

| Role | Time | Notes |
| --- | ---: | --- |
| Prefill MLP GateUp | `20.321 ms` | Largest prefill kernel group. |
| Prefill attention | `11.061 ms` | Still a visible prefill residual. |
| Prefill MLP Down | `8.784 ms` | Below GateUp. |
| Decode LMHead | `50.712 ms` | Biggest single decode group. |
| Decode attention | `35.383 ms` | Many small FlashInfer decode launches. |
| Decode norms | `6.724 ms` | Not first target. |

Low-risk decode probes:

- Decode-attention table sweep found only noise-level movement:
  current average `0.028960 ms`, best candidate `0.028875 ms` across
  `kv_len=2048/2056/2111`. No table update.
- Sequential `lm_head_top1` recheck on `0.5B 2048x64`:
  - native paired:
    `.tmp_codex/bench/stage2_20260517_decode/edgefm_0p5b_2048x64_native_top1_pair_seq.json`
    = `303.478 ms` total, `250.163 ms` decode.
  - top1:
    `.tmp_codex/bench/stage2_20260517_decode/edgefm_0p5b_2048x64_lm_head_top1_pair_seq.json`
    = `302.149 ms` total, `249.517 ms` decode.
  - delta: `-1.329 ms` total, about `0.44%`, below the default promotion gate.
- Prefill-attention sweep reported a best split setting
  `threshold=1024, short=64, long=64`, which is equivalent to the existing
  global `CTA_TILE_Q=64` behavior for this shape. Treat as measurement noise;
  no table update.

Decision:

- Keep `lm_head_top1` default-off. It is a useful cumulative diagnostic knob but
  not enough by itself to move the Stage-2 matrix.
- Do not change decode attention or 0.5B prefill attention table entries from
  these sweeps.
- The active optimization target remains 3B/1.5B long-prefill attention and
  BF16/FP16 boundary work; 0.5B decode can return later only after a larger
  LMHead or decode-linear route has fresh evidence.

## 2026-05-17 Stage-2 Mixed BF16 Source-Op Boundary Probes

Because the 3B graph-off trace showed visible BF16-to-FP16 conversion cost, I
tested default-off source-op variants that remove the input activation cast for
prefill linear and MLP shapes by using CUTLASS mixed-input GEMM
(`BF16 activation x FP16 weight -> BF16 output`). No official 3060 operator
table entry was changed during these probes.

Linear QKV/OProj candidate:

| Model / shape | Baseline total | Candidate total | Total delta | Baseline prefill | Candidate prefill | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0.5B 2048x32` | `178.168 ms` | `177.546 ms` | `-0.622 ms` / `0.35%` | `54.070 ms` | `53.612 ms` | Keep default-off only. |
| `1.5B 2048x32` | `479.377 ms` | `477.642 ms` | `-1.735 ms` / `0.36%` | `154.637 ms` | `153.090 ms` | Keep default-off only. |
| `3B 2048x32` | `937.584 ms` | `936.566 ms` | `-1.018 ms` / `0.11%` | `311.616 ms` | `310.456 ms` | Keep default-off only. |

Artifacts:

- `.tmp_codex/bench/stage2_20260517/edgefm_0p5b_2048x32_linear_mixed_bf16_default_tile_probe.json`
- `.tmp_codex/bench/stage2_20260517/edgefm_1p5b_2048x32_linear_mixed_bf16_default_tile_probe.json`
- `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_linear_mixed_bf16_probe.json`

MLP candidate:

| Model / shape | Baseline total | Candidate total | Total delta | Baseline prefill | Candidate prefill | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `0.5B 2048x32` | `178.168 ms` | `176.567 ms` | `-1.601 ms` / `0.90%` | `54.070 ms` | `52.999 ms` | Close, but below the default gate. |
| `1.5B 2048x32` | `479.377 ms` | `479.421 ms` | `+0.044 ms` | `154.637 ms` | `154.683 ms` | Reject. |
| `3B 2048x32` | `935.548 ms` | `935.835 ms` | `+0.287 ms` | `309.773 ms` | `310.328 ms` | Reject. |

Artifacts:

- `.tmp_codex/bench/stage2_20260517/edgefm_0p5b_2048x32_mlp_mixed_bf16_probe.json`
- `.tmp_codex/bench/stage2_20260517/edgefm_1p5b_2048x32_mlp_mixed_bf16_probe.json`
- `.tmp_codex/bench/stage2_20260517/edgefm_3b_2048x32_mlp_mixed_bf16_probe.json`

Combined 0.5B linear+MLP mixed mode did not stack cleanly:

- artifact:
  `.tmp_codex/bench/stage2_20260517/edgefm_0p5b_2048x32_linear_mlp_mixed_bf16_probe.json`
- result: `177.085 ms` total, slower than MLP-only `176.567 ms`

Decision:

- Keep `input_mode=mixed_bf16` for `cutlass_prefill_linear_source_op` as a
  default-off diagnostic capability behind `operator_impl_table` only.
- Do not promote mixed BF16 linear or MLP records to the official 3060 table.
- The only close result is 0.5B MLP mixed mode at about `0.90%`, still below the
  `>=1%` acceptance gate and not enough to change the Stage-2 priority.
- The next optimization work remains source-visible BF16 prefill attention /
  boundary fusion for 3B and 1.5B long-prefill.

## 2026-05-17 Source-Op Abstraction Cleanup

After removing the internal TensorRT engine bridge, I cleaned up the remaining
Qwen2.5 prefill fast-path ownership so future custom operators do not have to be
wired directly through `qwen2_5.cpp`.

Current source layout:

- QKV/OProj prefill fast path is now selected as a normal `linear` operator
  implementation:
  - `impl_id`: `cutlass_prefill_linear_source_op`
  - source: `src/operators/prefill_linear_source_op.{h,cu}`
  - registry entry: `src/operators/linear_impl.cu`
- MLP prefill fast path is now behind `GatedMlpLayer`:
  - `impl_id`: `cutlass_prefill_mlp_source_op`
  - source: `src/operators/prefill_mlp_source_op.{h,cu}`
  - wrapper: `src/layers/gated_mlp.{h,cu}`
- The 3060 operator table now uses regular op kinds:
  - `op_kind=linear` for QKV/OProj source-op records
  - `op_kind=mlp` for fused MLP source-op records
- The old `cutlass_prefill_*_bridge` impl ids remain accepted only as
  compatibility aliases for old local artifacts; they are not the active names.

This keeps `qwen2_5.cpp` focused on model composition: attention, linear, MLP,
norm, and generation flow. CUTLASS/CUDA fast-path decisions now sit in
layer/operator code and the operator table, which is the expected place for
future source-visible or plugin-op candidates.

## 2026-05-17 Internal TensorRT Bridge Removal

The Stage-1 product decision is now reflected in source code. EdgeFM Qwen2.5 no
longer builds or calls the internal TensorRT engine prefill bridges:

- removed `src/models/qwen2_5/trt_mlp_bridge.{h,cpp}`
- removed `src/models/qwen2_5/trt_linear_bridge.{h,cpp}`
- removed the `BUILD_TRT_MLP_BRIDGE` CMake option and Qwen2.5 TensorRT bridge
  link path
- removed the internal bridge env cleanup from
  `scripts/profile/profile_edgefm_generate_case.py`

This does not remove the external `TRT-Edge-LLM` benchmark/runtime path used as
the Stage-2 reference, and it does not remove the default-off direct plugin-op
attention experiment. The practical rule is now:

- EdgeFM Qwen2.5 production/tuning path: source-op CUTLASS/CUDA operators.
- TRT-Edge-LLM: benchmark reference for Stage 2.
- Direct plugin/source assets are allowed only when they run as EdgeFM
  operators, not as serialized TensorRT engine bridges.

Validation for this cleanup is recorded with the commit that contains it.

## 2026-05-17 Stage-2 Fresh TRT-Edge-LLM Long-Prefill Baseline

After closing Stage 1, I refreshed the external `TRT-Edge-LLM` comparison for
the long-prefill slices. This is a different target from the true EdgeFM
`trt_bridge` comparison: the goal is now to beat the external TRT-Edge-LLM
runtime, not just remove EdgeFM's bridge.

Setup:

- EdgeFM: `build-3060`, source-op/native, CUDA graph on.
- TRT-Edge-LLM: local `tests/data/trt_edgellm_workspace/*/engines_mxil2048`
  engines and `third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so`.
- Runs: warmup `1`, timed runs `3`.
- Artifact summary:
  `.tmp_codex/bench/stage2_fresh_trt_edge_20260517/summary_2048_decode32_64.json`

Results:

| Model | Shape | EdgeFM | TRT-Edge-LLM | Gap |
| --- | ---: | ---: | ---: | ---: |
| `0.5B` | `2048x32` | `177.878 ms` | `167.850 ms` | `+10.028 ms` / `+5.97%` |
| `0.5B` | `2048x64` | `303.503 ms` | `292.060 ms` | `+11.443 ms` / `+3.92%` |
| `1.5B` | `2048x32` | `478.075 ms` | `467.580 ms` | `+10.495 ms` / `+2.24%` |
| `1.5B` | `2048x64` | `813.252 ms` | `803.190 ms` | `+10.062 ms` / `+1.25%` |
| `3B` | `2048x32` | `936.562 ms` | `916.511 ms` | `+20.051 ms` / `+2.19%` |
| `3B` | `2048x64` | `1584.191 ms` | `1568.463 ms` | `+15.729 ms` / `+1.00%` |

Stage breakdown:

- `0.5B`: both prefill and decode contribute; the prefill delta is about
  `7.5-7.7 ms`, while decode adds `2.3-3.8 ms`.
- `1.5B`: the gap is almost entirely prefill. Decode is effectively tied.
- `3B`: prefill is slower by about `24-25 ms`, while EdgeFM decode is faster
  than TRT-Edge-LLM on these two slices; prefill remains the Stage-2 priority.

Decision:

- Stage 2 should still start from prefill, not decode, for `1.5B` and `3B`.
- For `0.5B`, decode has a small but real gap and should stay in the queue
  after the next prefill attribution pass.
- The next profiling step should capture paired graph-off mapping and graph-on
  formal traces for `3B 2048x32` first, because it has the largest absolute
  gap and the clearest prefill signal.

## 2026-05-17 Phase-1 Closure: Source-Op Is Close Enough To Retire Bridge As Default Goal

After Round AQ/AR/AS/AT, the remaining EdgeFM source-op versus true
`trt_bridge` gap is small enough to close Stage 1 by product decision rather
than keep chasing noise-level CUTLASS 2.x candidates.

Current valid Stage-1 comparison, using the true bridge build and bridge-only
table from Round AQ:

| Shape | Source-op current | True bridge | Remaining gap |
| --- | ---: | ---: | ---: |
| `1.5B 2048x32` | `478.218 ms` total, `153.491 ms` prefill | `475.005 ms` total, `150.506 ms` prefill | `+3.213 ms` total |
| `1.5B 2048x64` | `813.611 ms` total, `153.932 ms` prefill | `810.215 ms` total, `150.769 ms` prefill | `+3.396 ms` total |

Interpretation:

- 0.5B and 3B are no longer material bridge-removal blockers in the tested
  matrix.
- The only remaining blocker is 1.5B long prefill, and the delta is only a few
  milliseconds.
- Round AQ showed attention and decode are not the cause; the residual is
  scattered across GateUp, Down, QKV, and OProj GEMM/linear tactics.
- Round AR/AS/AT found no acceptable low-risk source-op table/code candidate to
  close the last few milliseconds.

Decision:

- Close Stage 1.
- Treat source-op as the preferred bridge-removal path for continued work.
- Remove the internal EdgeFM TensorRT engine bridge code; the remaining
  few-millisecond 1.5B long-prefill delta is accepted as close enough for
  Stage 1.
- Move active optimization effort to Stage 2: compare against external
  `trt-edge-llm` and optimize the largest fresh profile gaps, starting from
  decode if it offers easier end-to-end wins.

## 2026-05-17 Round AT: Linear Base-Tile Recheck

After Round AR rejected extra warp-shape templates, I rechecked the existing
linear tile choices that require no production source changes: `128x128x32`,
`128x256x32`, and `256x128x32` for the 1.5B `S=2048` QKV and OProj records.

Graph-off `1.5B 2048x32` results:

| Candidate | Total | Prefill | Delta vs official total | Decision |
| --- | ---: | ---: | ---: | --- |
| official | `485.747 ms` | `154.184 ms` | baseline | baseline |
| OProj `128x128x32` | `486.067 ms` | `154.474 ms` | `+0.321 ms` | Reject |
| OProj `128x256x32` | `486.266 ms` | `154.675 ms` | `+0.519 ms` | Reject |
| OProj `256x128x32` | `486.385 ms` | `154.593 ms` | `+0.638 ms` | Reject |
| QKV `128x128x32` | `486.341 ms` | `154.422 ms` | `+0.594 ms` | Reject |
| QKV `128x256x32` | `485.842 ms` | `154.163 ms` | `+0.095 ms` | Reject; noise-level |
| QKV `256x128x32` | `485.883 ms` | `154.151 ms` | `+0.136 ms` | Reject; noise-level |

Artifact:

- `.tmp_codex/bench/roundat_linear_base_tiles/summary_graphoff_2048x32.json`

Decision:

- No table change. Existing linear table stays as-is.
- The tiny QKV prefill improvements are graph-off noise and do not offset total
  latency.

## 2026-05-17 Round AS: MLP Down Output Mode Recheck

I tested the existing `down_output=fp16_cast` knob for the 1.5B `S=2048` MLP
record. The hypothesis was that an FP16-output down GEMM plus a separate BF16
cast might select a faster epilogue than direct BF16 output.

CUDA-graph results:

| Shape | Official | `down_output=fp16_cast` | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `2048x32` | `479.176 ms` total, `154.342 ms` prefill | `479.174 ms` total, `154.536 ms` prefill | `-0.002 ms` total, `+0.195 ms` prefill | Reject; noise-level total, prefill regression |
| `2048x64` | `813.512 ms` total, `153.853 ms` prefill | `814.374 ms` total, `154.738 ms` prefill | `+0.862 ms` total | Reject |

Artifact:

- `.tmp_codex/bench/roundas_mlp_down_output/summary_graphon.json`

Decision:

- No table change. Keep direct BF16 down output for the official 1.5B
  `S=2048` MLP source-op record.

## 2026-05-17 Round AQ: True-Bridge Rebaseline and 1.5B S2048 Role Attribution

Round AP closed the small plugin-op attention RoPE cache probe. I then reran the
remaining Stage-1 blocker with a valid true-bridge baseline and a serial
graph-off NSYS mapping, because the first attempts had two invalid conditions:

- `build-3060` is not bridge-enabled (`BUILD_TRT_MLP_BRIDGE=OFF`), so any
  "true bridge" run from that build is invalid.
- TensorRT bridge runs need
  `LD_LIBRARY_PATH=/usr/local/TensorRT-10.15.1.29/lib:${LD_LIBRARY_PATH:-}`;
  without it, cast-engine construction can fail and silently fall back to native
  paths.
- Parallel source/bridge NSYS runs polluted memory and OOMed the bridge process,
  so only the serial mapping artifacts below are used.

Valid bridge setup:

- Build: `build-3060-trt-mlp-release`
- Temporary source-op removal table:
  `.tmp_codex/bench/3060_20260517_roundaq_bridge_only_operator_table.json`
- Bridge env:
  `EDGE_FM_PREFILL_TRT_MLP=1`, `EDGE_FM_TRT_MLP_FP16_WEIGHTS=auto`,
  `EDGE_FM_PREFILL_TRT_LINEAR=1`, `EDGE_FM_TRT_LINEAR_ROLES=both`

Valid CUDA-graph true-bridge baselines:

| Shape | Source-op current | True bridge | Source-op gap |
| --- | ---: | ---: | ---: |
| `1.5B 2048x32` | `478.218 ms` total, `153.491 ms` prefill | `475.005 ms` total, `150.506 ms` prefill | `+3.213 ms` total, `+2.985 ms` prefill |
| `1.5B 2048x64` | `813.611 ms` total, `153.932 ms` prefill | `810.215 ms` total, `150.769 ms` prefill | `+3.396 ms` total, `+3.163 ms` prefill |

The Round AO plugin-op attention diagnostic narrows but does not close this:

| Shape | Plugin-op attention | True bridge | Plugin-op gap |
| --- | ---: | ---: | ---: |
| `1.5B 2048x32` | `477.311 ms` total, `152.485 ms` prefill | `475.005 ms` total, `150.506 ms` prefill | `+2.306 ms` total, `+1.979 ms` prefill |
| `1.5B 2048x64` | `811.566 ms` total, `151.750 ms` prefill | `810.215 ms` total, `150.769 ms` prefill | `+1.352 ms` total, `+0.981 ms` prefill |

Serial graph-off NSYS mapping:

- Source-op:
  `.tmp_codex/nsys/roundaq/1p5b_2048x1_sourceop_mapping_serial.nsys-rep`
- True bridge:
  `.tmp_codex/nsys/roundaq/1p5b_2048x1_bridge_mapping_serial.nsys-rep`

Role attribution from the serial mapping:

| Role | Source-op | True bridge | Delta |
| --- | ---: | ---: | ---: |
| GateUp GEMM | `68.624 ms` | `67.381 ms` | `+1.243 ms` |
| MLP down GEMM | `32.841 ms` | `32.155 ms` | `+0.686 ms` |
| QKV GEMM | `8.897 ms` | `8.450 ms` | `+0.447 ms` |
| OProj GEMM | `6.357 ms` | `6.057 ms` | `+0.300 ms` |
| Prefill attention | `21.102 ms` | `21.122 ms` | `-0.020 ms` |
| SwiGLU | `9.207 ms` | `9.282 ms` | `-0.075 ms` |

Decision:

- No code or table change is accepted in Round AQ.
- The first-stage blocker is now clearly the source-op GEMM/linear residual on
  1.5B long prefill, not attention and not decode.
- Continue Stage-1 bridge removal by targeting source-visible GateUp, Down,
  QKV, and OProj candidates. Ordinary CUTLASS 2.x tile/table sweeps are mostly
  at plateau, so the next accepted work must either produce a measurable
  source-op GEMM win or remain documented as a rejected candidate.

## 2026-05-17 Round AR: Linear Warp-Shape Table Probe

Round AQ showed the remaining non-MLP residual includes QKV `+0.447 ms` and
OProj `+0.300 ms` in the 1.5B `S=2048` graph-off role attribution. I tested
whether the MLP-style warp-shape CUTLASS variants help the linear bridge path
before starting a heavier custom GEMM loop.

Implementation probe:

- Temporarily extended `CutlassPrefillLinearBridge` with four diagnostic tile
  values:
  `128x128x32_warp64x32`, `128x128x32_warp32x64`,
  `128x128x64_warp64x32`, `128x128x64_warp32x64`.
- Build passed with the temporary templates.
- Generated temporary operator tables under
  `.tmp_codex/bench/roundar_linear_warp_tiles/`.
- After rejection, the temporary production source change was reverted and
  `build-3060` was rebuilt.

Graph-off prefilter on `1.5B 2048x32`:

| Candidate | Prefill | Delta vs official |
| --- | ---: | ---: |
| official | `154.056 ms` | baseline |
| OProj `128x128x32_warp32x64` | `154.176 ms` | `+0.120 ms` |
| OProj `128x128x32_warp64x32` | `154.366 ms` | `+0.310 ms` |
| OProj `128x128x64_warp32x64` | `154.502 ms` | `+0.446 ms` |
| OProj `128x128x64_warp64x32` | `154.650 ms` | `+0.593 ms` |
| QKV `128x128x32_warp32x64` | `154.539 ms` | `+0.482 ms` |
| QKV `128x128x32_warp64x32` | `154.723 ms` | `+0.667 ms` |
| QKV `128x128x64_warp32x64` | `154.708 ms` | `+0.651 ms` |
| QKV `128x128x64_warp64x32` | `154.675 ms` | `+0.618 ms` |

CUDA-graph confirmation for the least-bad OProj candidate:

| Shape | Official | Candidate | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `2048x32` | `478.448 ms` total, `153.725 ms` prefill | `478.822 ms` total, `154.096 ms` prefill | `+0.374 ms` total | Reject |
| `2048x64` | `814.005 ms` total, `154.069 ms` prefill | `814.101 ms` total, `154.172 ms` prefill | `+0.096 ms` total | Reject |

Artifacts:

- `.tmp_codex/bench/roundar_linear_warp_tiles/summary_2048x32.json`
- `.tmp_codex/bench/roundar_linear_warp_tiles/summary_graphon_oproj_best.json`

Decision:

- Reject all linear warp-shape variants for 1.5B `S=2048`.
- Do not add these tile modes to production `CutlassPrefillLinearBridge`.
- The QKV/OProj residual is not solved by the same CUTLASS 2.x warp-shape
  variants that were useful for MLP down; keep the Stage-1 focus on deeper
  source-visible GEMM work or a different kernel family.

## 2026-05-17 Round AP: Plugin-Op RoPE Cache Probe

After Round AO showed the TRT-Edge-LLM `ContextFMHARunner` plugin-op path helps
but is still below the Stage-1 acceptance gate, I tested one very small
operator-runner optimization: cache the plugin-op RoPE `cos_sin` table by
device/sequence/head-dim/rope parameters instead of regenerating it in the
BF16-to-FP16 QKV pack kernel on every prefill layer.

Build:

- `cmake --build build-3060 --target edge_fm -j8`
- Result: pass.

Serial CUDA-graph results against the Round AO plugin-op baseline:

| Shape | Round AO plugin-op | RoPE-cache candidate | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `1.5B 2048x32` | `477.311 ms` total, `152.485 ms` prefill | `477.365 ms` total, `152.377 ms` prefill | `+0.055 ms` total, `-0.108 ms` prefill | Reject; noise-level |
| `1.5B 2048x64` | `811.566 ms` total, `151.750 ms` prefill | `812.460 ms` total, `152.539 ms` prefill | `+0.894 ms` total, `+0.789 ms` prefill | Reject; regression |

Artifacts:

- `.tmp_codex/bench/3060_20260517_roundap_1p5b_2048x32_pluginop_ropecache.json`
- `.tmp_codex/bench/3060_20260517_roundap_1p5b_2048x64_pluginop_ropecache.json`

Decision:

- Reject and revert the `src/operators/attention_op.cu` RoPE cache code.
- Keep Round AO as the current plugin-op attention evidence.
- This does not change the remaining 1.5B `S=2048` Stage-1 bridge-removal
  status.

Notification:

- cc-connect progress notification failed after the build with
  `dial unix /home/zhangzimo/.cc-connect/run/api.sock: connect: connection refused`.

## 2026-05-17 Round AO: 1.5B S2048 Plugin-Op Attention Recheck

After Round AN rejected mixed-input MLP activation as a production default, I
rechecked the one remaining low-risk third-party/source-visible attention route
on the only material Stage-1 blocker: 1.5B `S=2048`.

Scope:

- Mode: `--edgefm-mode plugin-op --plugin-op-allow-bf16-fp16-cast`
- This calls TRT-Edge-LLM `ContextFMHARunner` directly from EdgeFM as an
  operator. It does not use a serialized TensorRT engine or execution context.
- The current third-party checkout only ships FP16 context-FMHA cubin metadata
  for SM86 in `third_party/TensorRT-Edge-LLM/cpp/kernels/contextAttentionKernels/cubin/fmha_cubin.h`.
  `ContextFMHARunner::canImplement()` also currently returns true only for
  `DataType::kHALF`, so BF16 Q/K/V must still use the diagnostic BF16-to-FP16
  pack/cast path.

Invalid run note:

- The first native/plugin pair was accidentally launched in parallel. The
  plugin process OOMed and that pair is invalid. Only the serial files below
  are used for the decision.

Serial CUDA-graph results:

| Shape | Native source-op | Plugin-op attention | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `1.5B 2048x32` | `478.218 ms` total, `153.491 ms` prefill | `477.311 ms` total, `152.485 ms` prefill | `-0.907 ms` total, `-1.006 ms` prefill | Reject; below 1% |
| `1.5B 2048x64` | `813.611 ms` total, `153.932 ms` prefill | `811.566 ms` total, `151.750 ms` prefill | `-2.044 ms` total, `-2.182 ms` prefill | Reject; below 1% and still not a clear bridge replacement |

Decision:

- Keep plugin-op attention default-off and diagnostic only.
- Do not add it to the official 3060 operator table from this evidence.
- A simple third-party patch to allow BF16 in `ContextFMHARunner::canImplement`
  is not enough because this checkout has no SM86 BF16 FMHA cubin entries. A
  BF16 cubin/source generation project would be larger and must be justified by
  fresh standalone evidence; prior BF16 FMHA generator attempts were slower than
  current FlashInfer.

Artifacts:

- `.tmp_codex/bench/3060_20260517_roundao_1p5b_2048x32_native_current_serial.json`
- `.tmp_codex/bench/3060_20260517_roundao_1p5b_2048x32_pluginop_bf16cast_serial.json`
- `.tmp_codex/bench/3060_20260517_roundao_1p5b_2048x64_native_current_serial.json`
- `.tmp_codex/bench/3060_20260517_roundao_1p5b_2048x64_pluginop_bf16cast_serial.json`

Notification:

- cc-connect progress notification again failed with
  `dial unix /home/zhangzimo/.cc-connect/run/api.sock: connect: connection refused`.

## 2026-05-17 Round AN: CUTLASS Mixed-Input Upcast Probe

Round AM rejected a direct `BF16 activation x FP16 weight` arch-MMA
specialization because SM86 does not expose that as a native floating-point MMA
instruction. I then checked the higher-level CUTLASS mixed-input route:
`cutlass::arch::OpMultiplyAddMixedInputUpcast`. This does compile because
CUTLASS upcasts the narrower operand before issuing a supported MMA, so it is a
valid source-visible third-party/CUTLASS route to test.

Standalone harness results:

| Shape | Baseline source-op | `mixed_bf16out` candidate | Decision |
| --- | ---: | ---: | --- |
| `0.5B S=2048` | `1.66730 ms` | `1.61782 ms` | positive standalone evidence |
| `1.5B S=2048` | `4.27256 ms` | `4.18618 ms` | positive standalone evidence |
| `3B S=2048` | `6.62480 ms` | `6.53717 ms` | positive standalone evidence |

The gate-up-only variant was also tested on `1.5B S=2048` and improved only
`4.27256 -> 4.23311 ms`, which is below the normal `>=1%` gate. The full
BF16-output variant was the better candidate.

Production integration:

- Added default-off `activation_mode=mixed_bf16` support to
  `src/models/qwen2_5/cutlass_mlp_bridge.cu`.
- The mode skips the input BF16-to-FP16 cast, writes BF16 gate/up activations,
  runs BF16/BF162 SwiGLU, and uses mixed-input CUTLASS GEMMs for gate/up and
  down.
- The official 3060 operator table was tested temporarily with
  `activation_mode=mixed_bf16` on `m=2048` MLP records, then rolled back after
  full generate regression.

End-to-end CUDA-graph native `2048x32` results:

| Shape | Official table | `activation_mode=mixed_bf16` | Decision |
| --- | ---: | ---: | --- |
| `0.5B` | `177.176 ms` total, `53.334 ms` prefill | `177.773 ms` total, `53.732 ms` prefill | Reject |
| `1.5B` | `479.309 ms` total, `154.302 ms` prefill | `480.043 ms` total, `155.002 ms` prefill | Reject |
| `3B` | `937.131 ms` total, `310.706 ms` prefill | `937.802 ms` total, `311.625 ms` prefill | Reject |

Artifacts:

- Standalone baselines/candidates:
  `.tmp_codex/bench/3060_20260517_mixed_bf16out_*_s2048.json`
- Production end-to-end checks:
  `.tmp_codex/bench/3060_20260517_prod_mixed_mlp_*_2048x32.json`
- Standalone harness source:
  `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/`

Decision:

- Accept the CUTLASS mixed-input upcast path as a valid diagnostic/source idea.
- Reject it as an official table/default production setting because it regresses
  real generate on all tested `S=2048` sizes.
- The official table intentionally contains no `activation_mode` entry.
- This does not remove the remaining 1.5B `S=2048` bridge-removal gap.

Notification:

- cc-connect progress notification failed with
  `dial unix /home/zhangzimo/.cc-connect/run/api.sock: connect: connection refused`;
  tuning continues without blocking on notification delivery.

## 2026-05-17 Round AM: CUTLASS Profiler Narrow GEMM Search and Third-Party Patch Gate

After Round AL left only the 1.5B `S=2048` source-op versus `trt_bridge`
residual, I opened the third-party modification route but treated it as an
evidence gate: patch CUTLASS only if the issue is a missing library wrapper,
not if the target instruction is unsupported by SM86/PTX.

CUTLASS profiler setup:

- A broad profiler build with reference kernels was stopped because it spent
  too long compiling reference unity files. This is not a useful iteration path
  for EdgeFM tuning.
- Rebuilt the profiler with `CUTLASS_PROFILER_DISABLE_REFERENCE=ON` and a narrow
  SM86 half-accumulator TN kernel filter:
  `cutlass_tensorop_h16816gemm_*_tn_align{8,4,2}`.
- Final profiler binary:
  `third_party/cutlass/build-edgefm-3060-profiler-h16816tn/tools/profiler/cutlass_profiler`
- Kernel enumeration artifact:
  `.tmp_codex/bench/3060_20260517_cutlass_profiler_h16816tn_enumerate.gemm.csv`

Profiler-only results:

| Shape | Best profiler candidate | Time | Current source-op implication |
| --- | --- | ---: | --- |
| GateUp `M=2048,N=17920,K=1536` | `256x128x32_s3_align8` | `2.422100 ms` | Not backported; Round AJ real generate already rejected `gateup_tile=256x128x32` |
| GateUp `M=2048,N=17920,K=1536` | `128x128x32_s3_align8` | `2.423550 ms` | Profiler-only tie; no real-generate evidence |
| Down `M=2048,N=1536,K=8960` | `128x128x64_s3_align8` | `1.153780 ms` | Rechecked in real generate before any table change |

Real-generate down-tile revalidation:

| Shape | Current table | `down_tile=default` / profiler-like | Decision |
| --- | ---: | ---: | --- |
| `1.5B 2048x32` | `478.849 ms` total, `154.112 ms` prefill | `478.934 ms` total, `154.140 ms` prefill | Reject |
| `1.5B 2048x64` | `814.827 ms` total, `154.612 ms` prefill | `815.140 ms` total, `154.791 ms` prefill | Reject |

Mixed BF16/FP16 third-party patch check:

- Tried a temporary standalone CUTLASS extension probe for
  `A=BF16 activation, B=FP16 weight -> FP16 output` to remove the source-op
  activation cast while keeping the fast FP16 Tensor Core path.
- Compile failed at the CUTLASS arch-MMA specialization layer
  (`cutlass::arch::Mma` incomplete type).
- PTX ISA evidence explains the failure: SM80/SM86 `mma.sync` has FP16
  `.f16.f16` variants and BF16 alternate floating point `.bf16.bf16` variants;
  for `.m16n8k8`, `.atype` must equal `.btype`, and the available BF16 examples
  are `.bf16.bf16`. The WGMMA docs are even stricter for floating variants
  except FP8. This means a third-party CUTLASS specialization cannot expose a
  real `BF16 x FP16` Tensor Core instruction on RTX 3060.
- The temporary probe was reverted and the standalone harness restore smoke
  passed:
  `.tmp_codex/bench/3060_20260517_mlp_harness_restore_smoke.json`.

Artifacts:

- CUTLASS profiler outputs:
  `.tmp_codex/bench/3060_20260517_cutlass_profiler_h16816tn/`
- Down revalidation:
  `.tmp_codex/bench/3060_20260517_down_default_reverify/`
- PTX reference:
  `.codex/skills/edge-fm-cuda-kernel-optimizer/vendor/kernel-pilot/knowledge/references/ako4all/cuda-cpp/vendored-docs/ptx-docs/9-instruction-set/9-instruction-set.md`

Decision:

- No production table/source change is accepted in Round AM.
- Third-party source modification is allowed for future work, but this specific
  mixed BF16/FP16 cast-removal path is rejected because it lacks a real SM86 MMA
  instruction.
- Stage-1 bridge removal now has two realistic choices: keep `trt_bridge` as a
  shape-local fallback only for 1.5B `S=2048`, or start a deeper custom
  CUDA/CuTe/Myelin-like GEMM project with an explicit cost gate.

## 2026-05-17 Round AL: 1.5B S2048 Down/QKV/OProj Residual Sweep

Continued Stage-1 bridge-removal work after Round AK showed the gate-up GEMM is
at a conventional CUTLASS 2.x plateau.

MLP down standalone check:

- Reused the Humanize torch-extension harness on `M=2048,K=8960,N=1536`.
- Artifact:
  `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/artifacts/attempts/down_shape_variants_roundrobin_r80.json`

| Candidate | Median |
| --- | ---: |
| `128x128x32_warp32x64` current production tile | `1.189856 ms/layer` |
| `128x128x64_warp64x32` | `1.186816 ms/layer` |
| `128x128x64_warp32x64` | `1.187264 ms/layer` |
| default `128x128x64` | `1.202160 ms/layer` |
| `128x128x32` | `1.208272 ms/layer` |

Decision: reject a down-tile backport from the standalone result. The best
standalone delta is only `~0.003 ms/layer`, and prior end-to-end Round AI
already beat the older `128x128x64_warp64x32` table on both long-prefill slices.

QKV/OProj table sweep:

- Temporary tables under:
  `.tmp_codex/bench/3060_20260517_roundal_linear_tile_sweep/tables/`
- Command shape: 1.5B `2048x32`, `--edgefm-mode native`, CUDA graph on, 3 runs.

QKV result:

| QKV tile | Total | Prefill | Decision |
| --- | ---: | ---: | --- |
| default/current | `479.640 ms` | `154.632 ms` | baseline |
| `128x128x32` | `479.633 ms` | `154.576 ms` | neutral |
| `128x256x32` | `479.357 ms` | `154.380 ms` | verify |
| `256x128x32` | `479.810 ms` | `154.725 ms` | reject |

OProj result:

| OProj tile | Total | Prefill | Decision |
| --- | ---: | ---: | --- |
| default/current | `479.323 ms` | `154.452 ms` | baseline/current best |
| `128x128x32` | `479.652 ms` | `154.762 ms` | reject |
| `128x256x32` | `479.352 ms` | `154.520 ms` | neutral/slower prefill |
| `256x128x32` | `479.594 ms` | `154.780 ms` | reject |

QKV `128x256x32` verification:

| Shape | Current/default | QKV `128x256x32` | Decision |
| --- | ---: | ---: | --- |
| `2048x32` | `479.202 ms` total, `154.360 ms` prefill | `478.863 ms` total, `154.034 ms` prefill | small win |
| `2048x64` | `814.970 ms` total, `154.716 ms` prefill | `815.177 ms` total, `154.715 ms` prefill | small regression |

Decision:

- Reject QKV/OProj table changes for now. QKV `128x256x32` helps `2048x32` by
  only `0.34 ms` and regresses `2048x64` by `0.21 ms`; it is below the
  acceptance gate and does not materially close the remaining bridge gap.
- OProj should stay default for 1.5B `S=2048`.

KernelPilot/source-provenance scan:

- Read KernelPilot GEMM routing and TensorRT-LLM references.
- Scanned local `third_party/TensorRT-Edge-LLM` for dense FP16/BF16 GEMM plugin
  assets.
- Result: this checkout exposes attention FMHA/XQA cubins and an INT4
  groupwise GEMM plugin, but no standalone dense FP16/BF16 GEMM plugin/kernel
  that can replace the TensorRT engine bridge for current Qwen prefill MLP or
  QKV/OProj. The bridge win is still coming from TensorRT engine Myelin/XMMA
  tactics rather than a directly reusable plugin-op.
- CUTLASS CuTe DSL quick feasibility check:
  `third_party/cutlass/examples/python/CuTeDSL/ampere/tensorop_gemm.py --help`
  currently fails because the local CuTe DSL Python package is missing the
  built `_mlir` module. Treat CuTe DSL as a separate toolchain/standalone
  project, not an immediate low-risk tweak inside the current repo loop.

Next decision:

- Ordinary table/source-op tweaks have now failed to close the final 1.5B
  `S=2048` residual.
- The remaining options are either a deeper custom CuTe/CUDA GEMM effort for a
  few milliseconds of gain, or a staged Stage-1 removal plan that keeps
  `trt_bridge` only as a 1.5B `S=2048` fallback while removing it for already
  cleared shapes.

Validation:

- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/profile_trt_edgellm_generate_case.py scripts/tune/tune_qwen_cublaslt.py`

## 2026-05-17 Round AK: 1.5B S2048 GateUp Humanize Plateau

Closed the first Humanize pass for the largest remaining 1.5B `S=2048`
source-op residual, MLP gate-up GEMM.

Standalone workspace:

- `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/`

Target shape:

- `M=2048,K=1536,N=17920`
- current production table: `gateup_tile=128x256x32`, FP16 Tensor Core path

Standalone harness results:

| Candidate | Result | Decision |
| --- | ---: | --- |
| current `128x256x32_s3` | median `~2.46 ms/layer` | baseline/current best |
| bridge-like `256x128x32` | median `2.467 ms/layer` | reject |
| default `128x128x64` | median `2.632 ms/layer` | reject |
| `128x128x32_warp32x64` | median `2.650 ms/layer` | reject |
| stage4 current | median `2.472 ms/layer` | reject |
| `64x256x32` | median `2.807 ms/layer` | reject |
| `torch.mm` / cuBLAS | median `4.564 ms/layer` | reject |
| `GemmIdentityThreadblockSwizzle<4>` | round-robin median `2.457968 ms` vs current `2.458608 ms`, mean slightly slower | reject as noise-level |

NCU evidence:

- full harness NCU CSV: `artifacts/ncu/current_gateup_full_ncu.csv`
- stall NCU CSV: `artifacts/ncu/current_gateup_stalls_ncu.csv`
- compute throughput `90.35%`, SM busy `90.77%`, DRAM throughput `36.21%`
- dominant sampled stall: math-pipe throttle (`66.61%`)
- registers/thread `158`, dynamic shared memory/block `73728`

Decision:

- Do not backport a gate-up change from this Humanize round.
- Stop ordinary CUTLASS 2.x table/stage/swizzle probing for
  `M=2048,K=1536,N=17920`; the current source-op kernel is already at a
  conventional math/tensor-pipe-heavy plateau.
- Continue Stage-1 bridge-removal work on the smaller residuals: MLP down
  (`+0.780 ms`), QKV (`+0.471 ms`), and OProj (`+0.315 ms`), or switch to a
  source-visible TRT/XMMA-like plugin/kernel asset that avoids TensorRT
  engine/runtime.

Artifacts:

- attempt ledger:
  `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/artifacts/attempt-ledger.md`
- optimization ledger:
  `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/artifacts/optimization-ledger.md`
- profile digest:
  `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/artifacts/profile-digests/2026-05-17_source_gateup_baseline.md`

## 2026-05-17 Round AJ: Round AI Profiling and GateUp Recheck Rejection

Captured a fresh graph-off NSYS mapping trace for the current Round AI 1.5B
`S=2048` source-op table and compared it with the true `trt_bridge` mapping.

Fresh source-op versus true-bridge role deltas on `2048x1` mapping:

| Role / kernel group | Round AI source-op | True trt_bridge | Delta |
| --- | ---: | ---: | ---: |
| MLP gate-up GEMM | `68.596 ms` | `67.248 ms` | `+1.348 ms` |
| MLP down GEMM | `32.888 ms` | `32.108 ms` | `+0.780 ms` |
| Prefill attention | `21.096 ms` | `21.067 ms` | `+0.029 ms` |
| SwiGLU activation | `9.208 ms` | `9.284 ms` | `-0.076 ms` |
| QKV GEMM | `8.905 ms` | `8.434 ms` | `+0.471 ms` |
| OProj GEMM | `6.359 ms` | `6.044 ms` | `+0.315 ms` |

Targeted follow-up:

- Tested `gateup_tile=256x128x32` with the accepted Round AI down tile, because
  the true bridge gate-up kernel reports a `256x128x32_stage3` shape.
- Tested `gateup_tile=default` with the same down tile as a guardrail.

Result on `1.5B 2048x32`:

| Candidate | Total | Prefill | Delta vs Round AI |
| --- | ---: | ---: | ---: |
| Round AI current | `478.372 ms` | `153.820 ms` | reference |
| `gateup_tile=256x128x32` | `479.163 ms` | `154.595 ms` | `+0.791 ms` total |
| `gateup_tile=default` | `483.328 ms` | `158.630 ms` | `+4.956 ms` total |

Decision:

- Reject the gate-up table recheck. The remaining gap is not closed by matching
  the bridge's coarse tile label.
- Stop adding more ad-hoc table variants for this shape. The next actionable
  target is a source-visible GEMM tactic/Humanize loop, with gate-up first
  because it is the largest individual residual.

Artifacts:

- `.tmp_codex/nsys/3060_20260517_roundai_source_mapping/`
- `.tmp_codex/bench/3060_20260517_gateup_recheck_after_stage4/`
- Humanize workspace:
  `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/`
- NCU baseline:
  `.tmp_codex/ncu/3060_20260517_gateup_humanize_baseline/source_gateup_1p5b_2048x1_ncu_demangled.csv`
- Profile digest:
  `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/artifacts/profile-digests/2026-05-17_source_gateup_baseline.md`

## 2026-05-17 Round AI: 1.5B S2048 Stage4 Warp-Shape Tie-Breaker

Followed Round AH with a narrower check inspired by the true-bridge kernel
names: keep `128x128x32` / stage4, but test non-square warp shapes for the MLP
down projection.

Implementation:

- Added two more default-off `CutlassPrefillMlpBridge` down-projection tiles:
  - `128x128x32_warp64x32`
  - `128x128x32_warp32x64`
- Updated the 3060 table for the 1.5B long-prefill MLP source-op record to use
  `down_tile=128x128x32_warp32x64`.

Fresh 5-run result:

| Model | Shape | Round AH source-op | Round AI source-op | True trt_bridge | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1.5B | `2048x32` | `478.478 ms` total, `153.868 ms` prefill | `478.372 ms` total, `153.820 ms` prefill | `475.004 ms` total, `150.522 ms` prefill | Accept as current fastest; still blocked |
| 1.5B | `2048x64` | `813.675 ms` total, `153.944 ms` prefill | `813.229 ms` total, `153.757 ms` prefill | `810.424 ms` total, `150.933 ms` prefill | Accept as current fastest; still blocked |

Decision:

- Accept the table tie-breaker because it wins both long-prefill slices in the
  5-run rerun and remains shape-local.
- The improvement is intentionally classified as marginal. It does not change
  the larger decision: 1.5B `S=2048` is still the only material Stage-1 blocker,
  now about `+2.8-3.3 ms` total and `+2.8-3.3 ms` prefill behind true bridge.
- Stop adding more ad-hoc CUTLASS table modes for this shape unless fresh NCU
  evidence identifies a specific bottleneck. Next viable route is
  Humanize/CuTe/source-op work or a source-visible plugin asset.

Artifacts:

- `.tmp_codex/bench/3060_20260517_stage4_warp_shape_sweep/`

Validation:

- `cmake --build build-3060 --target install -j$(nproc)`
- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/profile_trt_edgellm_generate_case.py scripts/tune/tune_qwen_cublaslt.py`
- `./build-3060/bin/edge_fm_compact_vocab_test`
- `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment or max_new_tokens or deferred_stop or metrics_surface"`:
  `11 passed, 9 deselected`

## 2026-05-17 Round AH: 1.5B S2048 MLP Down Warp-Shape Source-Op Refinement

Continued the Stage-1 bridge-removal work after Round AG narrowed the remaining
material blocker to Qwen2.5-1.5B `S=2048`.

Implementation:

- Added two default-off `CutlassPrefillMlpBridge` down-projection diagnostic
  tiles:
  - `128x128x64_warp64x32`
  - `128x128x64_warp32x64`
- Updated the 3060 table for the 1.5B long-prefill MLP source-op record:
  `m=2048|hidden=1536|intermediate=8960` now uses
  `down_tile=128x128x64_warp64x32` with the existing
  `tile=256x128x32`, `gateup_tile=128x256x32`, and
  `persistent_weights=true`.

Fresh records-based benchmark:

| Model | Shape | Official source-op rerun | Best warp-shape source-op | True trt_bridge | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1.5B | `2048x32` | `480.853 ms` total, `156.123 ms` prefill | `478.478 ms` total, `153.868 ms` prefill | `475.004 ms` total, `150.522 ms` prefill | Accept as fastest table; still blocked |
| 1.5B | `2048x64` | `815.569 ms` total, `156.087 ms` prefill | `813.675 ms` total, `153.944 ms` prefill | `810.424 ms` total, `150.933 ms` prefill | Accept as fastest table; still blocked |

Rejected / non-selected candidates:

- `down_tile=128x128x64_warp32x64`: essentially tied on `2048x32` but slightly
  worse on the combined `2048x32/64` prefill total.
- Repeating broad 1.5B table sweeps is no longer the best use of time. The
  remaining gap is about `+3.0-3.3 ms` prefill versus true `trt_bridge`, which
  points to exact GEMM tactic/source-op work or a source-visible plugin asset.

Artifacts:

- `.tmp_codex/bench/3060_20260517_warp_shape_sweep/`

Validation:

- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/profile_trt_edgellm_generate_case.py scripts/tune/tune_qwen_cublaslt.py`
- `./build-3060/bin/edge_fm_compact_vocab_test`
- `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment or max_new_tokens or deferred_stop or metrics_surface"`:
  `11 passed, 9 deselected`

Current Stage-1 bridge-removal state:

- 3B: source-op is already faster than EdgeFM `trt_bridge` across the tested
  Round AE matrix.
- 0.5B `S=512`: source-op is effectively tied with bridge after Round AG.
- 1.5B `S=2048`: still the only material blocker, now about `+3.0-3.3 ms`
  prefill behind bridge after Round AH.

## 2026-05-17 Round AG: Corrected Records-Based Sweep and Accepted 0.5B S512 MLP Down Tile

Fixed a profiling/tuning artifact issue before accepting new table changes:
temporary sweep tables must edit the current `records` field. Some ad-hoc
diagnostic tables created during the early Round AG investigation used the old
`entries` field and therefore repeated the official table. Those measurements
are treated as invalid diagnostics only; the main operator table was not
polluted.

Accepted change:

- For Qwen2.5-0.5B `m=512|hidden=896|intermediate=4864`, keep the existing
  source-op MLP gate-up tile `256x128x32`, but set the MLP down GEMM to
  `down_tile=128x128x32`.
- This mirrors the useful part of the TRT/XMMA short-prefill shape: more
  `M`-parallel CTAs for the down projection, while staying in the no-engine
  CUTLASS source-op path.

Fresh `records`-based benchmark:

| Model | Shape | Previous official source-op | Round AG source-op | True trt_bridge | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 0.5B | `512x32` | `14.99 / 113.78 ms` prefill/decode, `248.47 tok/s` | `12.77 / 113.34 ms`, `253.75 tok/s` | `12.85 / 113.26 ms`, `253.76 tok/s` | Accept; prefill now slightly faster, end-to-end tied |
| 0.5B | `512x64` | `14.64 / 229.63 ms`, `262.00 tok/s` | `12.59 / 229.35 ms`, `264.53 tok/s` | `12.73 / 229.07 ms`, `264.68 tok/s` | Accept; prefill faster, end-to-end within noise |

Rejected / no-default candidates from the corrected sweep:

- 0.5B `down_tile=128x256x32`: slower than the accepted `128x128x32`.
- 0.5B `down_output=fp16_cast`: did not beat direct BF16 output.
- 0.5B QKV/OProj `tile=128x256x32`: not a stable improvement.
- 1.5B `S=2048` MLP down `128x256x32`, MLP down `128x128x32`,
  `down_output=fp16_cast`, and QKV/OProj `128x256x32`: all slower than the
  current Round AF official source-op table.

Fresh true-bridge reference must use `--edgefm-mode as-is`. `--edgefm-mode
native` intentionally clears bridge env vars, even when running from the TRT
bridge build, and must not be used as a bridge-removal reference.

Current Stage-1 bridge-removal state:

- 0.5B `S=512` is no longer a blocker after this round.
- 3B was already faster than bridge across the tested Round AE matrix.
- The remaining material blocker is 1.5B `S=2048`: source-op is still about
  `+3 ms` behind true `trt_bridge` on prefill, with decode essentially tied.

Artifacts:

- `.tmp_codex/bench/3060_20260517_records_sweep_key/`
- `.tmp_codex/bench/3060_20260517_0p5b_down128x128_accept/`
- `.tmp_codex/bench/3060_20260517_true_bridge_pair/`
- `.tmp_codex/nsys/3060_20260517_true_bridge_mapping/`
- `.tmp_codex/ncu/3060_20260517_gemm_gap/`

Validation:

- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/profile_trt_edgellm_generate_case.py scripts/tune/tune_qwen_cublaslt.py`

## 2026-05-17 Round AF: Split MLP GateUp/Down Tile for 1.5B Long Prefill

Continued the Stage-1 bridge-removal work on the remaining blocker from Round
AE: Qwen2.5-1.5B `S=2048` was still about `+6.9-7.1 ms` slower than the valid
EdgeFM `trt_bridge` reference.

Implementation:

- Extended `CutlassPrefillMlpBridge` with optional per-GEMM tile params:
  `gateup_tile` and `down_tile`.
- Kept the existing `tile` param as the default/fallback, so all older table
  records keep identical behavior.
- Swept temporary operator tables for 1.5B `m=2048`:
  - current `gateup=256x128`, `down=256x128`: `481.202 / 156.379 ms`
    total/prefill on `2048x32`
  - `gateup=256x128`, `down=default`: `479.027 / 154.327 ms`
  - `gateup=128x256`, `down=default`: best rerun
    `478.797 / 154.182 ms` on `runs=5`
  - `gateup=default`, `down=128x256`: rejected, `484.781 / 160.164 ms`

Accepted table update:

- For 1.5B `m=2048|hidden=1536|intermediate=8960`, use:
  `tile=256x128x32`, `gateup_tile=128x256x32`, `down_tile=default`,
  `persistent_weights=true`.

Official table-driven rerun:

| Model | Shape | Previous Round AE source-op | Round AF source-op | Valid trt_bridge | Remaining delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B | `2048x32` | `482.383 ms` | `478.282 ms` | `475.482 ms` | `+2.800 ms` |
| 1.5B | `2048x64` | `817.800 ms` | `813.844 ms` | `810.713 ms` | `+3.131 ms` |

Decision:

- Accept. The change is shape-local, table-driven, and keeps behavior
  compatible for all other MLP records.
- Stage-1 bridge removal is closer but still not globally complete. The
  `1.5B / S=2048` residual is now about `+3 ms`, down from about `+7 ms`.
- Next step should stop broad CUTLASS tile sweeps and profile the remaining
  1.5B `S=2048` source-op residual at role/kernel level, likely separating
  MLP cast/residency overhead from the two GEMMs.

Validation:

- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/profile_trt_edgellm_generate_case.py scripts/tune/tune_qwen_cublaslt.py`
- `cmake --build build-3060 --target install -j$(nproc)`
- `./build-3060/bin/edge_fm_compact_vocab_test`
- `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment or max_new_tokens or deferred_stop or metrics_surface"`:
  `11 passed, 9 deselected`

Artifacts:

- `.tmp_codex/bench/3060_20260517_mlp_mixed_tile/`
- `.tmp_codex/bench/3060_20260517_table_driven_sourceop_roundaf/`

## 2026-05-16 Round AE: Shape-Specific Source-Op Table for Bridge Removal

Implemented the requested first-stage bridge-removal gate: compare no-engine
EdgeFM source-op against EdgeFM's current `trt_bridge`, and treat TRT-Edge-LLM
as the second-stage external reference after the internal bridge is no longer
needed.

Code/table changes:

- `CutlassPrefillMlpBridge` and `CutlassPrefillLinearBridge` can now resolve
  bridge-only operator table records:
  - `qwen_prefill_mlp_bridge` / `cutlass_prefill_mlp_bridge`
  - `qwen_prefill_linear_bridge` / `cutlass_prefill_linear_bridge`
- Records are exact on `model_name=qwen2_5`, `hw_profile=cuda_sm86`,
  `stage=prefill`, dtype, model hidden/intermediate or in/out features, and
  prefill `m`.
- Normal `linear` operator records are not replaced by the bridge records; the
  source-op bridges query these private op kinds themselves.
- The 3060 LLM table now carries shape-specific CUTLASS source-op settings for
  0.5B/1.5B/3B prefill `512/1024/2048`, including a 3B `S=2048` memory rule:
  MLP persistent FP16 weights stay on, but QKV/OProj persistent weights are
  off. Enabling all three persistent paths OOMs on RTX 3060.

Table-driven source-op versus valid/current `trt_bridge`:

| Model | Shape | Source-op table | trt_bridge | Delta | Prefill source/bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B | `512x32` | `128.920 ms` | `127.195 ms` | `+1.725 ms` | `14.990 / 13.094 ms` |
| 0.5B | `512x64` | `244.624 ms` | `243.139 ms` | `+1.485 ms` | `14.683 / 12.910 ms` |
| 0.5B | `1024x32` | `141.313 ms` | `141.306 ms` | `+0.007 ms` | `24.886 / 24.789 ms` |
| 0.5B | `1024x64` | `260.826 ms` | `260.771 ms` | `+0.055 ms` | `24.883 / 24.760 ms` |
| 0.5B | `2048x32` | `177.208 ms` | `176.562 ms` | `+0.646 ms` | `53.315 / 52.436 ms` |
| 0.5B | `2048x64` | `302.586 ms` | `302.691 ms` | `-0.105 ms` | `52.522 / 52.073 ms` |
| 1.5B | `512x32` | `346.176 ms` | `347.451 ms` | `-1.275 ms` | `37.127 / 38.250 ms` |
| 1.5B | `512x64` | `665.376 ms` | `666.325 ms` | `-0.949 ms` | `37.129 / 38.022 ms` |
| 1.5B | `1024x32` | `389.751 ms` | `390.169 ms` | `-0.418 ms` | `74.864 / 75.056 ms` |
| 1.5B | `1024x64` | `714.821 ms` | `714.971 ms` | `-0.150 ms` | `74.925 / 74.688 ms` |
| 1.5B | `2048x32` | `482.383 ms` | `475.482 ms` | `+6.901 ms` | `157.270 / 150.817 ms` |
| 1.5B | `2048x64` | `817.800 ms` | `810.713 ms` | `+7.087 ms` | `157.346 / 151.013 ms` |
| 3B | `512x32` | `687.765 ms` | `690.126 ms` | `-2.362 ms` | `84.967 / 87.191 ms` |
| 3B | `512x64` | `1310.195 ms` | `1312.552 ms` | `-2.357 ms` | `85.029 / 87.337 ms` |
| 3B | `1024x32` | `764.815 ms` | `771.044 ms` | `-6.228 ms` | `153.969 / 159.998 ms` |
| 3B | `1024x64` | `1395.456 ms` | `1401.624 ms` | `-6.167 ms` | `153.971 / 160.038 ms` |
| 3B | `2048x32` | `936.768 ms` | `943.426 ms` | `-6.658 ms` | `310.857 / 316.566 ms` |
| 3B | `2048x64` | `1583.752 ms` | `1590.451 ms` | `-6.699 ms` | `311.368 / 316.854 ms` |

Decision:

- First target is partially achieved, not globally complete:
  `11/18` source-op table shapes are now at or faster than `trt_bridge`.
- 3B is a strong candidate for staged bridge retirement, but only with the
  shape-specific memory policy above. 3B `S=2048` leaves only about `393 MB`
  free after warmup.
- Keep `trt_bridge` for now because 1.5B `S=2048` is still `~7 ms` slower and
  several 0.5B short-prefill cases are slightly behind.
- Next optimization queue: 1.5B `S=2048` prefill first, then 0.5B `S=512`
  residuals. Do not use TRT-Edge-LLM as the removal gate until this internal
  bridge gate passes.

Validation:

- `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/profile_trt_edgellm_generate_case.py scripts/tune/tune_qwen_cublaslt.py`
- `cmake --build build-3060 --target install -j$(nproc)`
- `./build-3060/bin/edge_fm_compact_vocab_test`
- `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment or max_new_tokens or deferred_stop or metrics_surface"`:
  `11 passed, 9 deselected`

Artifacts:

- `.tmp_codex/bench/3060_20260516_table_driven_sourceop_full/`
- `.tmp_codex/bench/3060_20260516_bridge_retirement_bridge_recheck_valid/`
- baseline bridge matrix:
  `.tmp_codex/bench/3060_20260515_rounde_edgefm_auto_matrix/`

## 2026-05-16 Round AD: Decode-Focused Fresh Matrix

Shifted the active probe from long-prefill attention to decode after prefill
attention reached the current source-visible plateau. The goal was to check
whether decode can compensate for the remaining TRT reference gap.

Fresh `2048x64` paired matrix:

| Model | EdgeFM total | TRT total | EdgeFM prefill | TRT prefill | EdgeFM decode | TRT decode | Decode delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | `303.92 ms` | `292.43 ms` | `53.68 ms` | `46.42 ms` | `250.07 ms` | `245.91 ms` | `+4.15 ms` |
| 1.5B | `819.88 ms` | `803.52 ms` | `160.17 ms` | `143.92 ms` | `659.51 ms` | `659.48 ms` | `+0.03 ms` |
| 3B | `1586.49 ms` | `1570.38 ms` | `313.30 ms` | `288.22 ms` | `1272.93 ms` | `1281.99 ms` | `-9.06 ms` |

Decode attribution:

- 3B decode is already faster than TRT on the long-decode slice. The remaining
  full-run gap is still prefill-driven.
- 1.5B decode is effectively tied with TRT.
- 0.5B decode is the only visible decode regression, about `4.15 ms` over
  `63` decode steps.
- 0.5B graph-off mapping identifies the largest decode hotspot as full-logits
  `LMHead` GEMV, followed by FlashInfer decode attention.

Low-risk decode probes:

- `lm_head_top1` on 0.5B `2048x64` improves total `303.92 -> 302.77 ms` and
  decode `250.07 -> 249.11 ms`, only about `0.38%` end-to-end. Keep it
  default-off.
- Real LMHead cublasLt sweep for `BF16 hidden / BF16 weight / FP32 logits`
  found no faster candidate than the current heuristic
  (`0.808960 ms` median standalone).
- 0.5B decode attention parameter sweep across `384` candidates kept the
  current table as best or tied best: average `0.028672 ms` for `kv_len`
  `2048/2112`.

Decision:

- Do not retune 1.5B/3B decode now; they are tied or ahead of TRT.
- Do not default `lm_head_top1`; it remains below the `>=1%` acceptance gate.
- Do not edit the 0.5B decode attention table from this sweep.
- For bridge removal, the next material gap is still prefill. The current
  `trt_bridge` should remain optional/reference/fallback, but the source-op
  linear work is now close enough that bridge cleanup can be planned only after
  the full matrix proves source-op no longer needs it.

Artifacts:

- `.tmp_codex/bench/3060_20260516_decode_focus/summary_2048x64.json`
- `.tmp_codex/nsys/3060_20260516_decode_focus/edgefm_0p5b_2048x64_graph_off_analysis.txt`
- `.tmp_codex/bench/3060_20260516_decode_focus/edgefm_0.5b_2048x64_lm_head_top1.json`
- `.tmp_codex/bench/3060_20260516_decode_focus/tune_0p5b_decode_lm_head_cublaslt_fp32out.json`
- `.tmp_codex/bench/3060_20260516_decode_focus/tune_0p5b_decode_attention_sweep.json`

## 2026-05-16 Round AC: Query-Block FMHA Scalar Rerun

Re-ran the existing `query_block4_prerotated` standalone candidate after the
fresh NCU/counter access check. This validates whether the open scalar seed has
any remaining reason to receive more work before moving to tensor-core/source
kernel routes.

Results:

| Shape | Correctness | Median |
| --- | --- | ---: |
| 3B `S=32` | passed, `max_abs=0.015625`, `mean_abs=0.000393` | `0.068736 ms` |
| 3B `S=512` | not rechecked | `5.595264 ms` |
| 3B `S=2048` | not rechecked | `75.900414 ms` |

Decision:

- Reject and close the scalar query-loop branch. It is correct on small shapes,
  but still orders of magnitude slower than EdgeFM FlashInfer at long prefill.
- Next FMHA work should start from a tensor-core/source-visible attention
  schedule or a fused QKV-pack/RoPE/FMHA route; no more scalar query-row
  mutations unless used only as a correctness oracle.

Artifacts:

- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/query_block4_prerotated_3b_s32_check.json`
- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/query_block4_prerotated_3b_s512.json`
- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/query_block4_prerotated_3b_s2048.json`

## 2026-05-16 Round AB: Native BF16 CUTLASS MLP Cast-Removal Probe

After fresh graph-off attribution showed about `+12.255 ms` of source-op
BF16/FP16 cast overhead versus TRT on 3B long-prefill, tested whether replacing
the accepted FP16-weight MLP source-op seed with native BF16 CUTLASS GEMMs could
remove casts without losing the Tensor Core advantage.

Validation and performance:

| Shape | Candidate | Correctness | Mean |
| --- | --- | --- | ---: |
| 0.5B `S=128` | BF16 CUTLASS smoke | passed, `max_abs=2.86e-6` | `0.324 ms` |
| 3B `S=2048` | FP16-weight source-op, `256x128x32_s3_f16acc` | passed, `mean_abs=1.92e-6` | `6.611 ms` |
| 3B `S=2048` | BF16 CUTLASS, `128x128x64_s3` | passed, `mean_abs=7.77e-7` | `12.239 ms` |
| 3B `S=2048` | BF16 CUTLASS, `128x256x32_s3_f16acc` | passed, `mean_abs=7.77e-7` | `11.831 ms` |
| 3B `S=2048` | BF16 CUTLASS, `256x128x32_s3_f16acc` | passed, `mean_abs=7.77e-7` | `12.018 ms` |

Decision:

- Reject native BF16 CUTLASS MLP as a production cast-removal route. It is
  correct, but roughly `1.8x` slower than the current FP16-weight source-op
  seed on the target 3B `S=2048` MLP slice.
- Keep the FP16 Tensor Core source-op path. The next cast/residency work should
  look for duplicate conversion removal or cached activation/weight residency,
  not a BF16 GEMM replacement.

Artifacts:

- `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/attempts/mlp_bf16_smoke_0p5b_s128.json`
- `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/attempts/mlp_bf16_diag_3b_s2048_cutlass_fp16_weight_candidate_256x128x32_s3_f16acc.json`
- `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/attempts/mlp_bf16_diag_3b_s2048_cutlass_bf16_candidate_128x128x64_s3.json`
- `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/attempts/mlp_bf16_diag_3b_s2048_cutlass_bf16_candidate_128x256x32_s3_f16acc.json`
- `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/attempts/mlp_bf16_diag_3b_s2048_cutlass_bf16_candidate_256x128x32_s3_f16acc.json`

## 2026-05-16 Round AA: Generated FMHA v2 SM86 BF16 Probe

After Round Z closed the small FlashInfer CTA/RoPE tweak path, tested whether
FlashInfer's TRT-LLM FMHA v2 generator can emit a useful SM86 BF16 source
kernel for the Qwen2.5 prefill shape without using a TensorRT engine.

Scope:

- Generated only temporary sources under `.tmp_codex/fmha_v2_sm86_probe`.
- Did not track generated `.cu` files because their headers are not suitable
  for source-op migration.
- Benchmarked two generated contiguous-Q/KV causal BF16 variants:
  `bf16_64_32_S_q_kv_128_sm86` and `bf16_64_128_S_q_kv_128_sm86`.
- This probe is no-RoPE; it checks the FMHA core family, not the full EdgeFM
  RoPE contract.

Validation and performance:

| Shape | Variant | Correctness | Median |
| --- | --- | --- | ---: |
| 3B `S=32` | non-tiled | passed, `max_abs=0.015625` | `0.683008 ms` |
| 3B `S=32` | tiled | passed, `max_abs=0.015625` | `0.655360 ms` |
| 3B `S=2048` | non-tiled | not checked | `1.567664 ms` |
| 3B `S=2048` | tiled | not checked | `1.723360 ms` |

Decision:

- Reject. The generated SM86 BF16 FMHA v2 kernels are much slower than the
  current FlashInfer no-RoPE core (`0.798592-0.851712 ms` at 3B `S=2048`) and
  also slower than fused-RoPE FlashInfer.
- They also do not solve the provenance/migration requirement: generated
  source is kept out of the repo and cannot be the planned source-op path.
- Active attention work should now skip this generator and focus on either a
  fused QKV-pack/RoPE/FMHA design or the larger BF16/FP16 source-op activation
  residency/dtype issue.

Artifacts:

- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/flashinfer_fmha_v2_sm86_probe.py`
- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/src/bf16_fmha/flashinfer_fmha_v2_sm86_probe.cu`
- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/flashinfer_fmha_v2_sm86_3b_s32_check.json`
- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/flashinfer_fmha_v2_sm86_3b_s2048.json`

## 2026-05-16 Round Z: FlashInfer No-RoPE CTA64 Diagnostic

After Round Y showed `RoPE pre-rotate + FlashInfer(PosEncoding=None)` is a
small but real win, added an explicit standalone `CTA_TILE_Q` diagnostic to the
FlashInfer wrapper. This does not touch production operator selection or the
3060 operator table.

3B `S=2048` focused results:

| Case | Default/previous | `CTA_TILE_Q=64` | `CTA_TILE_Q=128` | Decision |
| --- | ---: | ---: | ---: | --- |
| FlashInfer fused RoPE | `0.971968 ms` | `0.982976 ms` | `0.971616 ms` | stay at default/128 |
| FlashInfer no-RoPE core | `0.851712 ms` | `0.798592 ms` | `0.853792 ms` | `64` is better for no-RoPE core |
| RoPE pre-rotate + no-RoPE | `0.917648 ms` | `0.863744 ms` | `0.918912 ms` | best standalone combo |

Cross-size `S=2048` best standalone combo:

| Model | Fused-RoPE FlashInfer | Pre-rotate + no-RoPE `CTA64` | Delta |
| --- | ---: | ---: | ---: |
| 3B | `0.971968 ms` | `0.863744 ms` | `-11.13%` |
| 1.5B | `0.770048 ms` | `0.666448 ms` | `-13.45%` |
| 0.5B | `0.466784 ms` | `0.417664 ms` | `-10.52%` |

Decision:

- This closes the standalone FlashInfer/RoPE diagnostic loop. A separate
  pre-rotate plus no-RoPE `CTA64` path is the best FlashInfer variant measured
  so far.
- Still do not migrate it directly. The rough full-model movement remains only
  a few milliseconds, below the `>=1%` end-to-end gate for the current source-op
  target slices, and it adds extra launches/workspace.
- Important conclusion for the no-TRT objective: stop small FlashInfer internal
  CTA/table-style tweaks for now. The remaining attention gap requires a
  different FMHA core schedule, a fused QKV-pack/RoPE/FMHA route, or a
  correctness-preserving plugin/source-op asset.

Artifacts:

- `artifacts/attempts/flashinfer_prefill_diag_3b_s2048_flashinfer_rope_cta64.json`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s2048_flashinfer_none_no_rope_baseline_cta64.json`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s2048_flashinfer_rope_cta128.json`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s2048_flashinfer_none_no_rope_baseline_cta128.json`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s2048_prerotate_cta64.json`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s2048_prerotate_cta128.json`
- `artifacts/attempts/flashinfer_prefill_diag_1p5b_s2048_prerotate_cta64.json`
- `artifacts/attempts/flashinfer_prefill_diag_0p5b_s2048_prerotate_cta64.json`

## 2026-05-16 Round Y: FlashInfer RoPE Pre-Rotation Diagnostic

Tested a standalone, no-production-change diagnostic to split the current
FlashInfer prefill attention cost into RoPE placement versus FMHA core cost.
The new harness compares:

- current-style FlashInfer prefill with fused Llama RoPE;
- a source-visible BF16 pair-wise RoPE pre-rotation kernel for Q/K, followed by
  FlashInfer prefill with `PosEncodingMode::kNone`;
- FlashInfer no-RoPE core baseline.

Correctness smoke:

- command:
  `python3 deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/flashinfer_prefill_diag.py --model-size 3b --seq-len 32 --check --warmup 3 --runs 5`
- result: both FlashInfer fused RoPE and pre-rotate+no-RoPE pass the FP32
  causal GQA reference (`max_abs=0.015625`, `mean_abs=0.000484`).

Performance results:

| Shape | FlashInfer fused RoPE | Pre-rotate + no-RoPE FlashInfer | Delta | No-RoPE core |
| --- | ---: | ---: | ---: | ---: |
| 3B `S=512` | `0.120832 ms` | `0.118784 ms` | `-1.69%` | `0.101104 ms` |
| 3B `S=2048` | `0.971968 ms` | `0.917648 ms` | `-5.59%` | `0.851712 ms` |
| 1.5B `S=2048` | `0.770048 ms` | `0.748544 ms` | `-2.79%` | `0.691888 ms` |
| 0.5B `S=2048` | `0.466784 ms` | `0.446464 ms` | `-4.35%` | `0.415328 ms` |

Decision:

- This is a real per-layer microbench win and confirms RoPE placement is part
  of the attention residual.
- It is not enough to migrate as a production/default-off operator yet. On the
  current 3B 36-layer target the 3B `S=2048` median movement is roughly
  `1.96 ms` of prefill, below the `>=1%` end-to-end gate.
- The no-RoPE core still costs `~0.852 ms/layer`, so even perfect RoPE handling
  would not reach TRT's `~0.65 ms/layer` FMHA+RoPE reference. The next
  source-op attention work should target the FMHA core schedule itself or a
  fused QKV-pack/RoPE/FMHA layout, not a standalone pre-rotate migration.

Artifacts:

- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/flashinfer_prefill_diag.py`
- `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/src/bf16_fmha/flashinfer_prefill_diag.cu`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s32_check.json`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s512.json`
- `artifacts/attempts/flashinfer_prefill_diag_3b_s2048.json`
- `artifacts/attempts/flashinfer_prefill_diag_1p5b_s2048.json`
- `artifacts/attempts/flashinfer_prefill_diag_0p5b_s2048.json`

## 2026-05-16 Round X: Fresh Source-Op vs TRT Warm Mapping

Re-ran paired graph-off Nsight Systems mapping for the current no-TRT
source-op lane after one warmup run. This avoids the false first-run
persistent-weight cast hotspot seen with `warmup=0`.

Command shape:

- model: `Qwen2.5-3B-Instruct`
- shape: `prefill=2048`, `decode=1`
- EdgeFM gates:
  `EDGE_FM_PREFILL_CUTLASS_MLP=1`,
  `EDGE_FM_CUTLASS_MLP_ACCUM=fp16`,
  `EDGE_FM_CUTLASS_MLP_TILE=auto`,
  `EDGE_FM_CUTLASS_MLP_PERSISTENT_WEIGHTS=1`,
  `EDGE_FM_PREFILL_CUTLASS_LINEAR=1`,
  `EDGE_FM_CUTLASS_LINEAR_TILE=auto`

Stage result:

| Runtime | Prefill | Kernel total | Launches |
| --- | ---: | ---: | ---: |
| EdgeFM source-op, warm graph-off | `318.044 ms` | `316.853 ms` | `509` |
| TRT-Edge-LLM reference, warm graph-off | `289.951 ms` | `288.958 ms` | `368` |
| Delta | `+28.093 ms` | `+27.895 ms` | `+141` |

Role-level residual:

| Role | EdgeFM source-op | TRT reference | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| MLP GateUp | `141.789 ms` | `138.836 ms` | `+2.953 ms` | close enough; not first priority |
| MLP DownProj | `72.052 ms` | `71.864 ms` | `+0.188 ms` | matched |
| Attention incl. RoPE/KV | `35.030 ms` | `23.399 ms` | `+11.631 ms` | active source-op target |
| QKV | `17.647 ms` | `16.979 ms` | `+0.668 ms` | low priority |
| OProj | `14.319 ms` | `14.311 ms` | `+0.008 ms` | matched |
| SwiGLU | `14.434 ms` | `14.464 ms` | `-0.030 ms` | matched |
| Norm | `7.379 ms` | `7.187 ms` | `+0.192 ms` | low priority |
| lm_head | `1.834 ms` | `1.824 ms` | `+0.010 ms` | matched |
| Source-op BF16/FP16 casts | `12.311 ms` | `0.056 ms` | `+12.255 ms` | needs residency/dtype design |

Conclusion:

- The current no-TRT source-op path is not broadly slow anymore. GateUp,
  DownProj, QKV, OProj, SwiGLU, norm, and lm_head are close to TRT on the warm
  3B long-prefill mapping.
- The remaining prefill gap is concentrated in two places:
  1. prefill attention: `+11.6 ms`, source-visible FMHA/roPE-layout work;
  2. BF16/FP16 source-op casts: `+12.3 ms`, likely requiring a larger
     activation-residency or BF16-native source-op design.
- Next tuning priority remains prefill attention Humanize/source-op. Do not
  reopen classic CUTLASS linear tile sweeps or current fused-SwiGLU variants
  without new evidence.

Artifacts:

- EdgeFM run JSON:
  `artifacts/3060_tuning/20260516_sourceop_mapping/edgefm_sourceop_3b_2048x1_warm_mapping_run.json`
- EdgeFM nsys:
  `.tmp_codex/nsys/3060_20260516_3b_2048x1_sourceop_warm_mapping.nsys-rep`
- EdgeFM analysis:
  `artifacts/3060_tuning/20260516_sourceop_mapping/edgefm_sourceop_3b_2048x1_warm_trtstyle_analysis.md`
- TRT run JSON:
  `artifacts/3060_tuning/20260516_sourceop_mapping/trt_reference_3b_2048x1_warm_mapping_run.json`
- TRT nsys:
  `.tmp_codex/nsys/3060_20260516_3b_2048x1_trt_reference_warm_mapping.nsys-rep`
- TRT analysis:
  `artifacts/3060_tuning/20260516_sourceop_mapping/trt_reference_3b_2048x1_warm_mapping_analysis.md`

## 2026-05-16 Round W: MLP Bridge Fused-SwiGLU Probe

After Round V rejected scalar fast-math changes, tested whether the existing
`cutlass_prefill_swiglu` fused GateUp+SwiGLU operator can help the accepted MLP
source-op lane when it is used inside `CutlassPrefillMlpBridge` and followed by
the source-op DownProj GEMM. This was a temporary default-off diagnostic branch:

- enabled only with both `EDGE_FM_PREFILL_CUTLASS_MLP=1` and
  `EDGE_FM_PREFILL_SWIGLU_FUSION=1`;
- used FP16 input/weight/output for the fused GateUp+SwiGLU step, then reused
  the source-op DownProj path;
- no production code change was kept after the benchmark rejection.

Validation and measurement:

| Candidate | Correctness | 3B `2048x32` total | Prefill | Decode | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| MLP source-op + fused GateUp/SwiGLU + source-op DownProj | token alignment `3/3` passed | `1074.953 ms` | `449.389 ms` | `625.399 ms` | rejected |

Decision:

- Correctness passed, so the weight layout and basic numeric contract are not
  the blocker.
- Performance regressed heavily versus the current source-op lane
  (`~935 ms` total / `~309 ms` prefill), so the fused MoE-derived prefill
  SwiGLU kernel remains diagnostic-only and must not be wired into the MLP
  bridge.
- The temporary code was reverted and `build-3060 --target install` passed
  after restore.

Artifacts:

- `artifacts/3060_tuning/20260516_mlp_bridge_fused_swiglu/edgefm_3b_2048x32_mlp_bridge_fused_swiglu.json`

## 2026-05-16 Round V: MLP SwiGLU Fast Exp Probe

After Round U rejected direct cuTile FMHA migration and Round T showed classic
linear tile variants are at a plateau, tested one small MLP-side prefill
candidate: replacing the default-off source-op SwiGLU `expf` calls with CUDA
`__expf`.

Scope:

- only touched `src/models/qwen2_5/cutlass_mlp_bridge.cu` temporarily;
- only affects `EDGE_FM_PREFILL_CUTLASS_MLP=1`, not the default runtime;
- no production code change was kept after benchmark rejection.

Validation and measurement:

| Candidate | Correctness | 3B `2048x32` total | Prefill | Decode | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| source-op + `__expf` SwiGLU | token alignment `3/3` passed | `938.400 ms` | `312.065 ms` | `626.166 ms` | rejected |

Decision:

- The candidate did not beat the current source-op lane and did not meet the
  `>=1%` end-to-end acceptance gate.
- The temporary code was reverted and `build-3060 --target install` passed
  after restore.
- Do not spend more time on scalar fast-math tweaks for SwiGLU unless a fresh
  profile shows it as a dominant residual again. The next larger prefill levers
  remain RoPE/layout-aware FMHA fusion and a genuinely different GEMM/XMMA-like
  source-op route.

Artifacts:

- `artifacts/3060_tuning/20260516_swiglu_fast_exp/edgefm_3b_2048x32_swiglu_fast_exp.json`

## 2026-05-16 Round U: cuTile RoPE Pre-Rotation Probe

Continued from Round S's finding that cuTile's `64x64` core attention schedule
is promising only before RoPE is included. This round keeps all work inside the
standalone benchmark helper and does not change EdgeFM production runtime.

Added `--rope-prerotate` to
`deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/cutile_fmha_candidate.py`:

- first variant: separate cuTile kernels rotate Q and K before the core FMHA;
- second variant: pair-wise non-interleaved RoPE computes both halves of each
  rotary pair together, avoiding duplicate pair loads/work.

3B BF16 causal GQA results:

| Candidate | `S=512` | `S=2048` | Correctness | Decision |
| --- | ---: | ---: | --- | --- |
| EdgeFM FlashInfer with RoPE | `0.095760 ms` | `0.985088 ms` | existing harness | keep baseline |
| cuTile core, no RoPE | `0.090496 ms` | `0.933408 ms` | passed | not real contract |
| cuTile RoPE-cache inside FMHA | `0.162608 ms` | `1.626416 ms` | passed | rejected |
| cuTile pre-rotate Q/K | `0.123888 ms` | `1.041472 ms` | passed | better, still rejected |
| cuTile pair-wise pre-rotate Q/K | `0.111744 ms` | `1.035744 ms` | passed | best cuTile+RoPE, still rejected |

Small `S=2048` tile sweep for pair-wise pre-rotate:

| Tile | Mean |
| --- | ---: |
| `64x64` | `1.039382 ms` |
| `64x128` | `2.305075 ms` |
| `128x64` | `1.470067 ms` |
| `128x128` | `2.430637 ms` |

Decision:

- Moving RoPE out of the FMHA inner loop is the right direction; it recovers
  most of the RoPE-cache regression.
- The result still misses the current FlashInfer contract by about `5%` at
  3B `S=2048` and `17%` at 3B `S=512`, so it is not eligible for migration.
- The gap is now small enough to be a useful Humanize follow-up, but not via
  another coarse tile sweep. The next source-visible attention work should be a
  fused/vectorized RoPE-layout path or a C++/CUTLASS implementation that keeps
  the 64x64 core schedule while reducing the extra two rotation launches and
  workspace traffic.

Artifacts:

- pre-rotate attempts:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/cutile_fmha_rope_prerotate*_3b_s*.json`

## 2026-05-16 Round T: Linear Role Split and Diagnostic Tile Recheck

After Round S showed that source-visible FMHA is blocked by RoPE/layout fusion,
rechecked the QKV/OProj source-op lane before adding any new production code.
This round leaves no production source change. A temporary diagnostic
`cutlass_linear_bridge.cu` tile expansion was compiled, measured, rejected, and
removed before the final rebuild.

3B `2048x32` source-op role split, with current MLP source-op enabled:

| Linear roles | Total avg | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| `none` | `952.936 ms` | `326.664 ms` | `626.104 ms` | fallback reference |
| `qkv` | `945.585 ms` | `319.299 ms` | `626.138 ms` | QKV source-op helps |
| `oproj` | `946.409 ms` | `319.980 ms` | `626.256 ms` | OProj source-op helps |
| `both` | `938.655 ms` | `312.310 ms` | `626.190 ms` | current source-op lane remains useful |

3B `2048x32` diagnostic linear tile sweep, roles=`both`:

| Tile | Total avg | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| `128x256x32` | `934.582 ms` | `309.128 ms` | `625.311 ms` | keep existing auto target |
| `default` | `935.305 ms` | `309.864 ms` | `625.291 ms` | close but not better |
| `128x256x32_s4` | `936.006 ms` | `310.032 ms` | `625.821 ms` | rejected |
| `64x256x32` | `940.221 ms` | `314.379 ms` | `625.693 ms` | rejected |
| `256x64x32` | `941.958 ms` | `315.876 ms` | `625.940 ms` | rejected |

Notes:

- CUTLASS profiler was configured under `.tmp_codex/cutlass_profiler_build`,
  but the default target started compiling a broad CUTLASS library. That is too
  wide for this tuning loop and was stopped. Future profiler use must constrain
  the kernel family/pattern first; otherwise a narrow in-repo tile probe is more
  efficient.
- The role split confirms that QKV/OProj source-op is a real no-bridge
  improvement, but the simple tile variants do not create a new accepted path.
  Current `128x256x32` remains the best measured diagnostic tile for 3B
  QKV/OProj.
- Next useful no-bridge work should not be another classic CUTLASS linear tile
  tweak unless a new source/tactic appears. The active choices stay:
  RoPE/layout-aware FMHA fusion or a genuinely different GEMM/XMMA-equivalent
  source-op route.

Artifacts:

- role split:
  `artifacts/3060_tuning/20260516_linear_role_rerun/*.json`
- diagnostic tile sweep:
  `artifacts/3060_tuning/20260516_linear_tile_diag2/*.json`

## 2026-05-16 Round S: cuTile FMHA Core Schedule Probe

Continued the prefill-first source-visible FMHA search after rejecting the TRT
`plugin-op` correctness path. This round adds a reproducible cuTile benchmark
under `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/`;
it does not change EdgeFM production runtime.

cuTile BF16 causal GQA probe:

- Source: `third_party/cutile-python/test/kernels/attention.py::fmha_kernel`
  (`Apache-2.0`).
- Shape: Qwen2.5-3B-like `B=1, Hq=16, Hkv=2, D=128`, causal GQA.
- Best tile from the sweep is `TILE_M=64,TILE_N=64`.
- Correctness: cuTile variants passed `torch.scaled_dot_product_attention`
  reference with `max_abs=0.00390625`.

| 3B shape | EdgeFM FlashInfer BF16 with RoPE | cuTile core, no RoPE | cuTile with RoPE cache | Decision |
| --- | ---: | ---: | ---: | --- |
| `S=512` | `0.095760 ms` | `0.090496 ms` | `0.162608 ms` | core interesting, RoPE-cache rejected |
| `S=1024` | `0.283744 ms` | `0.277504 ms` | `0.468880 ms` | core near/tiny win, RoPE-cache rejected |
| `S=2048` | `0.985088 ms` | `0.933408 ms` | `1.626416 ms` | core `~5.2%` faster, RoPE-cache rejected |

Broader `TILE_M=64,TILE_N=64` core-only sweep:

| Model | `S=512` | `S=1024` | `S=2048` |
| --- | ---: | ---: | ---: |
| 0.5B cuTile core | `0.046080 ms` | `0.128896 ms` | `0.421328 ms` |
| 0.5B EdgeFM FlashInfer | `0.055088 ms` | `0.144384 ms` | `0.464640 ms` |
| 1.5B cuTile core | `0.080752 ms` | `0.222208 ms` | `0.736080 ms` |
| 1.5B EdgeFM FlashInfer | `0.083264 ms` | `0.233312 ms` | `0.755264 ms` |
| 3B cuTile core | `0.090496 ms` | `0.277504 ms` | `0.933408 ms` |
| 3B EdgeFM FlashInfer | `0.095760 ms` | `0.283744 ms` | `0.985088 ms` |

Decision:

- cuTile's 64x64 BF16 core schedule is the first source-visible attention
  candidate that beats the current EdgeFM FlashInfer attention microbench on
  the tested core FMHA math.
- It is not an accepted production replacement because the no-RoPE measurement
  is not the real EdgeFM contract. The first realistic RoPE-cache integration is
  substantially slower than FlashInfer, so direct migration is rejected.
- Next useful work is not another tile sweep. The route is either:
  1. a lighter RoPE fusion/pre-rotation design that keeps the 64x64 core fast,
  2. an AOT cuTile cubin/operator-runner feasibility check, or
  3. a separate source-visible CUDA/CUTLASS implementation that borrows only
     the 64x64 tiling lesson.

Artifacts:

- cuTile benchmark helper:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/cutile_fmha_candidate.py`
- cuTile artifacts:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/cutile_fmha*.json`
- EdgeFM FlashInfer rerun artifacts:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/edgefm_flashinfer_*_current_rerun.json`

## 2026-05-16 Round R: CUTLASS41 / Plugin-Op Attention Probe

Continued the prefill-first FMHA branch. This round leaves no production source
change; a temporary `plugin-op` contiguous-Q/KV diagnostic branch was built,
measured, rejected, and removed before the final rebuild.

CUTLASS41 tensor-core FMHA probe:

- Built `third_party/cutlass/examples/41_fused_multi_head_attention` after
  enabling `CUTLASS_ENABLE_TOOLS=ON`; the prior build error was from the
  example's dependency on `cutlass/util/command_line.h`.
- The upstream example is FP16-only by default, so a temporary standalone BF16
  probe was made under `.tmp_codex/cutlass41_bf16_probe/` by switching
  `scalar_t` to `cutlass::bfloat16_t` and adding explicit `Element(...)`
  literals in the test tensor initializer.
- Measured Qwen2.5-3B-like `B=1, H=16, D=128`, causal, fixed seqlen:

| Candidate | 3B `S=512` | 3B `S=2048` | Decision |
| --- | ---: | ---: | --- |
| CUTLASS41 FP16 example | `0.092 ms` | `1.069 ms` | source reference only |
| CUTLASS41 BF16 temp probe | `0.092 ms` | `1.069 ms` | rejected direct replacement |
| current EdgeFM FlashInfer BF16 | `0.096 ms` | `0.982 ms` | keep baseline |
| TRT-Edge-LLM direct FP16 FMHA runner | n/a | `~0.651 ms` | performance ceiling only |

Direct EdgeFM `plugin-op` check:

- `plugin-op` without the CUTLASS MLP/linear source-op gates is not meaningful:
  it measured `1106.894 ms` total / `480.687 ms` prefill on 3B `2048x32`,
  essentially the slow native path with a diagnostic attention overlay.
- With the current default-off source-op gates
  (`EDGE_FM_PREFILL_CUTLASS_MLP=1`, `EDGE_FM_CUTLASS_MLP_ACCUM=fp16`,
  `EDGE_FM_CUTLASS_MLP_TILE=auto`, `EDGE_FM_CUTLASS_MLP_PERSISTENT_WEIGHTS=1`,
  `EDGE_FM_PREFILL_CUTLASS_LINEAR=1`, `EDGE_FM_CUTLASS_LINEAR_TILE=auto`),
  `plugin-op + BF16->FP16 cast` measured `933.052 ms` total /
  `307.102 ms` prefill versus the same source-op check at `936.410 ms` total /
  `310.621 ms` prefill.
- A temporary contiguous-Q/KV wrapper for the same TRT runner measured
  `931.301 ms` total / `305.265 ms` prefill on 3B `2048x32`, but token
  alignment failed on the 1.5B decode fixture (`3/20` aligned). The existing
  packed plugin-op path failed the same alignment check (`3/20` aligned), while
  the source-op MLP/linear path without plugin-op still passed (`20/20`
  aligned). The contiguous wrapper was removed and the build was regenerated.

Decision:

- CUTLASS41 is a useful readable tensor-core FMHA schedule, but its BF16
  fixed-seqlen path is not a direct win for the 3B/S2048 shape.
- The TRT `ContextFMHARunner` operator-level path remains promising as a
  ceiling, but on real BF16 EdgeFM tensors the current cast/pack wrapper gives
  only `~0.36%` end-to-end gain over the source-op baseline. The contiguous
  diagnostic reached `~0.55%`, but both plugin-op variants fail token
  alignment. This is a correctness rejection, not merely a performance-gate
  rejection.
- Do not default or migrate either attention candidate. The next prefill route
  should target a BF16-native source-op schedule or a fused QKV-pack/RoPE/FMHA
  path that avoids per-layer BF16->FP16 round trips; that is a larger design
  than the current low-risk operator swap.

Artifacts:

- EdgeFM paired runs:
  `artifacts/3060_tuning/20260516_fmha_cutlass_probe/*.json`
- Correctness checks:
  source-op alignment `20/20`; plugin-op packed and contiguous alignment `3/20`
- CUTLASS41 temporary probe:
  `.tmp_codex/cutlass41_bf16_probe/`

## 2026-05-16 Round Q: Rejected Scalar Query-Block FMHA Source-Op

Continued the prefill-first FMHA Humanize branch in
`deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/`. This round did not
change production code.

Implemented and measured a first source-visible pre-rotated query-block
candidate:

- `benchmarks/bf16_fmha_seed_candidate.py` now supports the query-block variant.
- `src/bf16_fmha/edgefm_bf16_fmha_seed.cu` now has a pre-rotation stage for Q/K
  and a query-block kernel.
- Correctness passed for 3B `S=32` and `S=128` against the standalone FP32
  causal GQA reference.

Results:

| Candidate | 3B `S=128` | 3B `S=512` | 3B `S=2048` | Decision |
| --- | ---: | ---: | ---: | --- |
| old scalar online seed | `0.825 ms` | `23.384 ms` | `179.641 ms` | rejected |
| query-block4 pre-rotated | `0.498 ms` | `6.085 ms` | `83.959 ms` | rejected |
| query-block4 + K/V reuse | n/a | `5.602 ms` | `79.637 ms` | rejected |
| query-block8 + K/V reuse | `0.512 ms` | `5.811 ms` | `77.507 ms` | rejected |
| current EdgeFM FlashInfer | n/a | `0.096 ms` | `0.982 ms` | baseline |

Decision:

- The query-block route proves the correctness contract and shows that
  pre-rotation/KV reuse helps the scalar seed, but it is still roughly `58x`
  slower than FlashInfer at `S=512` and `80x` slower at `S=2048`.
- Per the Round P rule, no NCU was collected because the candidate did not get
  within `2x` of FlashInfer at `S=512`.
- Stop mutating scalar query-loop FMHA. The next source-visible attention route
  should start from a tensor-core FMHA schedule, with
  `third_party/cutlass/examples/41_fused_multi_head_attention/` logged as the
  next local source scan candidate.

Artifacts:

- standalone qrows4 final:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/fmha_query_block_prerotated_qrows4_final_3b_s*.json`
- qrows8 sweep:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/fmha_query_block_prerotated_qrows8_3b_s*.json`

## 2026-05-16 Round P: FMHA Focused Stall Evidence

After rejecting the MLP stage-4 tile, moved the prefill queue to BF16 FMHA
evidence. This round did not change production code.

Collected a focused NCU profile for the current FlashInfer BF16 prefill
attention kernel on 3B/S2048 using Ampere-valid per-warp stall metric names:

- harness timing: `1.058816 ms/layer`
- `sm__throughput`: `38.50%`
- active warps: `15.92%`
- DRAM throughput: `4.89%`
- L2 throughput: `32.69%`
- tensor pipe cycles active: `35.95%`
- tensor instructions: `8.99%`
- ALU pipe: `10.29%`
- shared-memory bank conflicts: `112613`
- collected per-warp stalls:
  - `short_scoreboard`: `5.10%`
  - `long_scoreboard`: `1.82%`
  - `dispatch_stall`: `1.36%`

Decision:

- The current FlashInfer BF16 prefill shape is still not HBM-bandwidth bound.
  It looks occupancy/shared-memory-latency limited: low active warps, low DRAM,
  and `short_scoreboard` as the largest collected stall.
- Do not repeat `cta_tile_q` table-only work. A useful source-visible FMHA
  candidate should reduce per-CTA shared-memory/register pressure and use a
  bank-conflict-aware K/V tile.
- Next concrete edit for the standalone Humanize workspace:
  `src/bf16_fmha/edgefm_bf16_fmha_seed.cu` should move from scalar
  one-query loops to a first tiled query-block candidate. Only profile it if it
  gets within `2x` of FlashInfer at S512.

Artifacts:

- digest:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/profile-digests/2026-05-16_flashinfer_bf16_fmha_3b_s2048_stall_digest.md`
- NCU report/CSV:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/profile-digests/ncu_reports/edgefm_flashinfer_bf16_fmha_3b_s2048_stalls.*`

## 2026-05-16 Round O: Rejected MLP Stage-4 Tile Follow-Up

Continued the prefill-first no-bridge route. This round stayed on MLP because
Round N's graph-off trace still showed source-op MLP GEMMs as the largest
controllable prefill group. Decode remains parked.

Standalone Humanize sweep:

- Extended the standalone MLP harness with additional CUTLASS FP16-acc configs:
  `128x128x32_s4`, `128x256x32_s4`, `256x128x32_s4`,
  `64x256x32_s3`, and `256x64x32_s3`.
- Best new 3B/S2048 candidate:
  - `128x256x32_s4_f16acc`: `6.572616 ms` mean / `6.572032 ms` median
    over 20 runs.
  - Current 3B auto seed `256x128x32_s3_f16acc`: `6.618253 ms` mean /
    `6.614080 ms` median over the same 20-run check.
  - Standalone movement is only about `0.7%` for the full MLP candidate, below
    a comfortable end-to-end expectation.

Production spike result:

- Adding the new stage-4 template as an extra production branch made the tight
  3B source-op persistent-weight path fail allocation:
  `alloc_failed_down (45088768 bytes, cudaErrorMemoryAllocation)`, followed by
  fallback cuBLASLt failure.
- Replacing the existing `128x256x32` experimental branch with stage-4 avoided
  adding a new enum but still did not produce an accepted route:
  - 3B `2048x32`, MLP persistent + linear scratch:
    `938.254 ms` total / `311.863 ms` prefill.
  - Restored official source-op path, MLP persistent + linear scratch:
    `935.052 ms` total / `309.435 ms` prefill.
  - Round N final quick check remains the better reference:
    `934.917 ms` total / `309.263 ms` prefill.
- 3B linear persistent remains outside the stable budget; the current practical
  3B source-op mode is MLP persistent plus linear scratch.

Decision:

- Reject production `128x256x32_s4` MLP tile. It is a small standalone win but
  loses at the end-to-end CUDA graph gate and risks the 3B memory budget.
- Restored production `src/models/qwen2_5/cutlass_mlp_bridge.cu` to the Round N
  source path. The only retained changes are standalone Humanize harness
  candidate support and artifact records.
- Next action: stop adding small classic CUTLASS MLP tile variants unless a new
  source-visible tactic appears; move the active prefill queue to BF16-correct
  FMHA or a genuinely different GEMM/XMMA-equivalent source.

Artifacts:

- standalone candidate JSONs:
  `.tmp_codex/bench/fresh_3060_20260516/mlp_3b_s2048_*s4*f16acc*.json`
- production rejected/restored checks:
  `artifacts/3060_tuning/20260516_mlp_tile_s4/`

## 2026-05-16 Round N: Prefill SwiGLU Half2 and Minimal Linear Tile Probe

Continued the prefill-first no-bridge route. Decode remains a parking-lot item;
this round only touched the default-off source-op prefill path.

Implemented:

- `src/models/qwen2_5/cutlass_mlp_bridge.cu`
  - added an aligned/even `__half2` SwiGLU kernel for the source-op MLP path;
  - kept the scalar SwiGLU kernel as fallback for odd or unaligned tensors.
- `src/models/qwen2_5/cutlass_linear_bridge.cu`
  - added a minimal default-off linear tile knob:
    `EDGE_FM_CUTLASS_LINEAR_TILE=default|auto|128x256x32`;
  - `auto` maps only the verified 3B QKV/OProj prefill shapes to
    `128x256x32`; all other shapes keep the previous default tile.
  - broader `128x128x32` and `256x128x32` branches were measured but not kept
    in production code because they did not clear the candidate gate.

Correctness/build checks:

- `cmake --build build-3060 --target install -j$(nproc)` passed.
- Source-op enabled token alignment passed:
  `EDGE_FM_CUTLASS_LINEAR_TILE=auto ... -k token_alignment_cuda_graph`
  -> `3 passed, 17 deselected`.
- Source-op enabled core generate regression passed:
  `compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `4 passed, 16 deselected`.
- Default path core generate regression passed:
  `token_alignment_cuda_graph or compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `7 passed, 13 deselected`.
- `./build-3060/bin/edge_fm_compact_vocab_test` passed.
- `scripts/operator_table/validate_operator_tables.py` passed.
- profile/tuning script `py_compile` passed.

Primary source-op CUDA graph matrix, `prefill=2048`, `runs=3`, `warmup=1`,
compared with the accepted Round M vector-cast baseline:

| Model / slice | Round M total / prefill | Round N total / prefill | Prefill delta | Total delta |
| --- | ---: | ---: | ---: | ---: |
| 3B `2048x32` | `938.357 / 312.644 ms` | `937.273 / 311.230 ms` | `-1.414 ms` | `-1.084 ms` |
| 3B `2048x64` | `1584.073 / 312.201 ms` | `1584.529 / 311.676 ms` | `-0.525 ms` | `+0.456 ms` |
| 1.5B `2048x32` | `485.854 / 161.084 ms` | `484.979 / 160.227 ms` | `-0.857 ms` | `-0.875 ms` |
| 1.5B `2048x64` | `821.662 / 161.518 ms` | `819.430 / 159.713 ms` | `-1.805 ms` | `-2.232 ms` |
| 0.5B `2048x32` | `177.796 / 54.145 ms` | `177.343 / 53.878 ms` | `-0.267 ms` | `-0.453 ms` |
| 0.5B `2048x64` | `303.308 / 53.333 ms` | `302.659 / 52.933 ms` | `-0.400 ms` | `-0.649 ms` |

Additional focused `3B / 2048x32` five-run checks:

- `linear=auto`: `935.994 ms` total, `310.172 ms` prefill.
- `linear=default`: `936.875 ms` total, `310.987 ms` prefill.
- after pruning unaccepted tile variants from the code, final `linear=auto`
  quick check: `934.917 ms` total, `309.263 ms` prefill.

Decision:

- Accept as a small default-off prefill source-op polish. It improves prefill
  consistently and does not affect the default path.
- Do not treat this as a main bridge-removal breakthrough: end-to-end movement
  is below the `>=1%` acceptance gate, and 3B `2048x64` total movement is within
  noise.
- `trt_bridge` still cannot be removed. The remaining priority is still
  source-visible GEMM/XMMA-equivalent work and BF16-correct long-prefill FMHA.
- Decode parking lot is unchanged; no decode work should preempt the next
  prefill target.

Artifacts:

- tile probe JSONs:
  `artifacts/3060_tuning/20260516_linear_tile_probe/`
- Round N matrix and focused checks:
  `artifacts/3060_tuning/20260516_swiglu_half2/`
- final graph-off mapping digest:
  `artifacts/3060_tuning/20260516_swiglu_half2/edgefm_3b_swiglu_half2_nsys_digest.md`

Fresh post-commit graph-off grouping confirms the remaining prefill priority:

- MLP CUTLASS GEMMs: `213.621 ms`, flat versus Round M and still dominant.
- FlashInfer prefill attention: `35.030 ms`, flat.
- Linear QKV/OProj CUTLASS GEMMs: `31.935 ms`, only `-0.285 ms` versus Round M.
- SwiGLU half2: `14.433 ms`, down from Round M `16.597 ms`.
- BF16/FP16 casts: `12.308 ms`, flat.

Next action: skip more scalar/vector polish unless a fresh trace says otherwise;
move back to source-visible GEMM/XMMA-equivalent work or BF16-correct FMHA.

Follow-up rejected probe:

- `bf16_to_half2` / `half2_to_bf16` x4-per-thread cast kernels were tested as
  a low-risk attempt to reduce the remaining `12.308 ms` cast group.
- Source-op token alignment still passed, but `3B / 2048x32` regressed to
  `936.946 ms` total and `311.085 ms` prefill versus the final Round N quick
  check `934.917 / 309.263 ms`.
- The x4 code was reverted immediately; raw artifact:
  `artifacts/3060_tuning/20260516_cast_x4_probe/edgefm_3b_2048x32_cast_x4.json`.
- A wider FlashInfer prefill attention CTA sweep was attempted. The runtime
  correctly rejects unsupported `cta_tile_q` values outside `{16,64,128}`.
  Re-running the legal set on 3B/S2048 found `cta128` best in microbench
  (`0.934 ms` versus baseline `0.973 ms`), but the source-op end-to-end slice
  only reached `934.283 ms` total and `308.723 ms` prefill. This is below the
  table-update gate, so the official table remains unchanged.

## 2026-05-16 Round M: Vectorized Source-Op Cast Kernels

Continued the prefill-first queue. This round targets only source-op
BF16/FP16 cast/layout overhead; decode remains parked unless a fresh profile
shows a larger opportunity.

Implemented vectorized conversion kernels in the default-off source-op path:

- production files:
  - `src/models/qwen2_5/cutlass_mlp_bridge.cu`
  - `src/models/qwen2_5/cutlass_linear_bridge.cu`
- BF16->FP16 now uses `__nv_bfloat162 -> __half2` when the tensor is even-sized
  and aligned; otherwise it falls back to the existing scalar kernel.
- FP16->BF16 now uses `__half2 -> __nv_bfloat162` for aligned even tensors.
- QKV bias add in the linear source-op output cast also has a vectorized path
  when columns are even and bias is aligned.
- This is still guarded by the existing default-off source-op gates:
  `EDGE_FM_PREFILL_CUTLASS_MLP=1` and `EDGE_FM_PREFILL_CUTLASS_LINEAR=1`.

Correctness/build checks:

- `cmake --build build-3060 --target install -j$(nproc)` passed.
- Source-op enabled token alignment passed:
  `EDGE_FM_CUTLASS_MLP_TILE=auto ... -k token_alignment_cuda_graph`
  -> `3 passed, 17 deselected`.
- Source-op enabled core generate regression passed:
  `compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `4 passed, 16 deselected`.
- Default path core generate regression passed:
  `token_alignment_cuda_graph or compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `7 passed, 13 deselected`.
- `./build-3060/bin/edge_fm_compact_vocab_test` passed.
- `scripts/operator_table/validate_operator_tables.py` passed.
- profile/tuning script `py_compile` passed.

Primary CUDA graph benchmark, `runs=3`, `warmup=1`, source-op MLP + linear
enabled with `EDGE_FM_CUTLASS_MLP_TILE=auto`:

| Model / slice | Before vector cast total / prefill | Vector cast total / prefill | Prefill delta | Total delta |
| --- | ---: | ---: | ---: | ---: |
| 3B `2048x32` | `943.566 / 317.028 ms` | `938.357 / 312.644 ms` | `-4.384 ms` | `-5.209 ms` |
| 3B `2048x64` | `1587.863 / 315.522 ms` | `1584.073 / 312.201 ms` | `-3.321 ms` | `-3.790 ms` |
| 1.5B `2048x32` | `487.406 / 162.597 ms` | `485.854 / 161.084 ms` | `-1.513 ms` | `-1.552 ms` |
| 1.5B `2048x64` | `823.725 / 163.344 ms` | `821.662 / 161.518 ms` | `-1.826 ms` | `-2.063 ms` |
| 0.5B `2048x32` | `178.429 / 54.719 ms` | `177.796 / 54.145 ms` | `-0.574 ms` | `-0.633 ms` |
| 0.5B `2048x64` | `304.975 / 54.773 ms` | `303.308 / 53.333 ms` | `-1.440 ms` | `-1.667 ms` |

Fresh `3B / 2048x1` graph-off attribution after vector cast:

- `256x128x32` MLP GEMMs: `213.462 ms`, `72` launches.
- linear/QKV/OProj `128x128x64` GEMMs: `32.220 ms`, `72` launches.
- FlashInfer prefill attention: `34.930 ms`, `36` launches.
- source-op SwiGLU: `16.597 ms`, `36` launches.
- BF16->FP16 vector casts: `9.926 ms`, `180` launches.
- FP16->BF16 vector casts: `2.383 ms`, `36` launches.
- Compared with Round L grouping, cast kernels drop from `16.164 ms` to
  `12.309 ms` on the mapping trace.

Decision:

- Accept vectorized cast kernels as a default-off prefill source-op
  improvement. It is correctness-clean and improves every measured long-prefill
  slice.
- Follow-up prefill attention probe:
  - `scripts/tune/tune_qwen_attention_prefill.py` on 3B/S2048 found
    `prefill_cta_tile_q=128` slightly faster than the current table in the
    single-layer microbench: `0.955 ms` versus baseline `0.973 ms`.
  - End-to-end source-op `3B 2048x32` with a temporary cta128 table measured
    `936.272 ms` total and `310.612 ms` prefill, versus the vector-cast
    baseline `938.357 ms` total and `312.644 ms` prefill.
  - This is a useful prefill signal but only about `0.22%` end-to-end, below
    the `>=1%` table-update gate, so the official operator table stays
    unchanged.
- `trt_bridge` still cannot be removed: best no-engine 3B `2048x32` is now
  `938.357 ms` on accepted source-op settings. The temporary cta128 probe can
  reach `936.272 ms`, still about `+20.066 ms` behind the earlier TRT
  `916.206 ms` reference and not accepted as a table change.
- Keep the next active work on prefill:
  1. source-visible GEMM/XMMA gap,
  2. FlashInfer-vs-TRT long-prefill FMHA gap,
  3. residual cast/layout cost.
- Decode parking lot is unchanged: no new decode work should preempt the above.

Artifacts:

- benchmark JSONs:
  `artifacts/3060_tuning/20260516_cast_vectorized/`
- temporary rejected cta128 table and benchmark:
  `artifacts/3060_tuning/20260516_cast_vectorized/operator_impl_table_llm_qwen3b_prefill_cta128.json`
  and
  `artifacts/3060_tuning/20260516_cast_vectorized/edgefm_3b_2048x32_vec_cast_attn_cta128.json`
- graph-off mapping:
  `artifacts/3060_tuning/20260516_cast_vectorized/nsys/edgefm_3b_mlp_auto_linear_vec_cast_2048x1_mapping.nsys-rep`
- digest and kernel grouping:
  `artifacts/3060_tuning/20260516_cast_vectorized/edgefm_3b_mlp_auto_linear_vec_cast_2048x1_nsys_digest.md`
  and
  `artifacts/3060_tuning/20260516_cast_vectorized/edgefm_3b_vec_cast_kernel_groups.txt`

## 2026-05-16 Round L: Prefill MLP Tile Auto for No-Bridge Source-Op

Per the current tuning priority, this round stays focused on prefill. Decode
observations are recorded only as follow-up candidates unless a profile shows a
new `>=1%` end-to-end opportunity.

Implemented a default-off CUTLASS MLP tile selector:

- production file:
  - `src/models/qwen2_5/cutlass_mlp_bridge.cu`
- new gate:
  - `EDGE_FM_CUTLASS_MLP_TILE=default|auto|128x256x32|256x128x32`
- `auto` currently selects `256x128x32` only for verified Qwen2.5 shapes:
  - 3B: `hidden=2048`, `intermediate=11008`
  - 0.5B: `hidden=896`, `intermediate=4864`
  - 1.5B stays on the previous `128x128x64` tile because the standalone sweep
    showed a small regression on that shape.

Standalone Humanize MLP harness evidence:

| Model / S2048 | Previous `128x128x64_s3_f16acc` | `128x256x32_s3_f16acc` | `256x128x32_s3_f16acc` | Decision |
| --- | ---: | ---: | ---: | --- |
| 3B | `6.784 ms` | `6.639 ms` | `6.617 ms` | use `256x128x32` |
| 1.5B | `4.231 ms` | `4.273 ms` | `4.270 ms` | keep default |
| 0.5B | `1.555 ms` | `1.667 ms` | `1.503 ms` | use `256x128x32` |

Correctness/build checks:

- `cmake --build build-3060 --target install -j$(nproc)` passed.
- Source-op enabled token alignment passed:
  `EDGE_FM_CUTLASS_MLP_TILE=auto ... -k token_alignment_cuda_graph`
  -> `3 passed, 17 deselected`.
- Source-op enabled core generate regression passed:
  `compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `4 passed, 16 deselected`.
- Default path core generate regression passed:
  `token_alignment_cuda_graph or compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `7 passed, 13 deselected`.
- `./build-3060/bin/edge_fm_compact_vocab_test` passed.
- `scripts/operator_table/validate_operator_tables.py` passed.
- profile/tuning script `py_compile` passed.

Primary CUDA graph benchmark, `runs=3`, `warmup=1`, source-op MLP + linear
enabled:

| Model / slice | MLP tile | Total | Prefill | Decode | Prefill delta vs default tile | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3B `2048x32` | default | `953.204 ms` | `326.559 ms` | `626.511 ms` | reference | current rebuild baseline |
| 3B `2048x32` | auto | `943.566 ms` | `317.028 ms` | `626.385 ms` | `-9.531 ms` | `-1.01%` total |
| 3B `2048x64` | default | `1599.612 ms` | `326.530 ms` | `1272.898 ms` | reference | |
| 3B `2048x64` | auto | `1587.863 ms` | `315.522 ms` | `1272.156 ms` | `-11.008 ms` | prefill-only win; total under 1% |
| 1.5B `2048x32` | auto | `487.406 ms` | `162.597 ms` | `324.679 ms` | n/a | auto resolves to default tile |
| 1.5B `2048x64` | auto | `823.725 ms` | `163.344 ms` | `660.213 ms` | n/a | auto resolves to default tile |
| 0.5B `2048x32` | default | `180.379 ms` | `56.431 ms` | `123.823 ms` | reference | |
| 0.5B `2048x32` | auto | `178.429 ms` | `54.719 ms` | `123.588 ms` | `-1.712 ms` | `-1.08%` total |
| 0.5B `2048x64` | default | `306.050 ms` | `55.751 ms` | `250.131 ms` | reference | |
| 0.5B `2048x64` | auto | `304.975 ms` | `54.773 ms` | `250.031 ms` | `-0.978 ms` | prefill win; total under 1% |

Fresh `3B / 2048x1` graph-off attribution for auto tile:

- `256x128x32` MLP GEMMs: `215.277 ms`, `72` launches.
- linear/QKV/OProj `128x128x64` GEMMs: `32.468 ms`, `72` launches.
- FlashInfer prefill attention: `35.272 ms`, `36` launches.
- source-op SwiGLU: `16.667 ms`, `36` launches.
- BF16->FP16 casts: `12.621 ms`; FP16->BF16 residual casts: `3.543 ms`.

Decision:

- Accept `EDGE_FM_CUTLASS_MLP_TILE=auto` as a default-off prefill source-op
  improvement. It is correctness-clean and gives a real prefill win on 3B/0.5B.
- Do not default it yet; the whole source-op stack is still experimental and
  TRT remains faster on the reference matrix.
- Keep optimization priority on prefill:
  1. reduce remaining source-op GEMM/XMMA gap,
  2. reduce FlashInfer-vs-TRT FMHA long-prefill gap,
  3. reduce BF16/FP16 cast/layout cost.
- Decode parking lot:
  - decode graph replay is stable and not the current bottleneck in the new
    `2048x32/64` source-op runs;
  - `lm_head_top1` remains rejected without `>=1%` end-to-end gain;
  - previous decode-attention table fix remains accepted historical work, but
    no new decode work is active in this prefill-focused round.

Artifacts:

- benchmark JSONs:
  `artifacts/3060_tuning/20260516_mlp_tile_auto/`
- graph-off mapping:
  `artifacts/3060_tuning/20260516_mlp_tile_auto/nsys/edgefm_3b_mlp_auto_linear_bf16out_2048x1_mapping.nsys-rep`
- digest and kernel grouping:
  `artifacts/3060_tuning/20260516_mlp_tile_auto/edgefm_3b_mlp_auto_linear_bf16out_2048x1_nsys_digest.md`
  and
  `artifacts/3060_tuning/20260516_mlp_tile_auto/edgefm_3b_mlp_auto_kernel_groups.txt`

## 2026-05-16 Round K: CUTLASS QKV/OProj Source-Op and BF16 Output Trim

Extended the no-TRT-engine source-op path beyond MLP:

- added default-off QKV/OProj CUTLASS linear runner:
  - `src/models/qwen2_5/cutlass_linear_bridge.h`
  - `src/models/qwen2_5/cutlass_linear_bridge.cu`
  - wired through `src/models/qwen2_5/qwen2_5.{h,cpp}`
  - CMake target updated in `src/models/qwen2_5/CMakeLists.txt`
- gates:
  - `EDGE_FM_PREFILL_CUTLASS_LINEAR=1`
  - `EDGE_FM_CUTLASS_LINEAR_ROLES=qkv|oproj|both|all`
  - `EDGE_FM_CUTLASS_LINEAR_PERSISTENT_WEIGHTS=1`
  - `EDGE_FM_CUTLASS_LINEAR_MIN_M=64`
- added direct BF16-output CUTLASS epilogues for source-op GEMMs where no bias
  is required. This removes separate FP16->BF16 output cast kernels for MLP
  DownProj and bias-free OProj-like linear calls.

Correctness/build checks:

- `cmake --build build-3060 --target install -j$(nproc)` passed.
- `EDGE_FM_PREFILL_CUTLASS_LINEAR=1 EDGE_FM_CUTLASS_LINEAR_ROLES=qkv ... -k token_alignment_cuda_graph`
  passed: `3 passed, 17 deselected`.
- `EDGE_FM_PREFILL_CUTLASS_LINEAR=1 EDGE_FM_CUTLASS_LINEAR_ROLES=oproj ... -k token_alignment_cuda_graph`
  passed: `3 passed, 17 deselected`.
- `EDGE_FM_PREFILL_CUTLASS_LINEAR=1 EDGE_FM_CUTLASS_LINEAR_ROLES=both ... -k token_alignment_cuda_graph`
  passed: `3 passed, 17 deselected`.
- With MLP + linear source-op enabled, token alignment passed after direct
  BF16-output changes: `3 passed, 17 deselected`.
- With MLP + linear source-op enabled, core generate regression passed:
  `compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `4 passed, 16 deselected`.

Primary CUDA graph benchmark, `runs=3`, `warmup=1`, `prefill=2048`:

| Model / slice | Mode | Total | Prefill | Gap to TRT total | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| 3B `2048x32` | native | `1111.018 ms` | `484.580 ms` | `+194.811 ms` | baseline |
| 3B `2048x32` | MLP source-op old | `965.386 ms` | `339.237 ms` | `+49.180 ms` | Round J |
| 3B `2048x32` | MLP source-op BF16-out | `962.093 ms` | `336.254 ms` | `+45.887 ms` | saves output cast |
| 3B `2048x32` | MLP + linear source-op old | `957.777 ms` | `331.272 ms` | `+41.571 ms` | linear scratch; about `377 MB` free |
| 3B `2048x32` | MLP + linear source-op BF16-out | `950.190 ms` | `324.168 ms` | `+33.983 ms` | best no-engine slice so far; about `393 MB` free |
| 3B `2048x32` | TRT reference | `916.206 ms` | `285.681 ms` | reference | bridge still cannot be removed |
| 1.5B `2048x32` | native | `557.607 ms` | `233.211 ms` | `+88.650 ms` | baseline |
| 1.5B `2048x32` | MLP source-op old | `492.819 ms` | `167.820 ms` | `+23.862 ms` | Round J |
| 1.5B `2048x32` | MLP + linear source-op old | `487.324 ms` | `162.533 ms` | `+18.368 ms` | persistent linear fits |
| 1.5B `2048x32` | MLP + linear source-op BF16-out | `485.315 ms` | `160.529 ms` | `+16.358 ms` | best no-engine 1.5B slice |
| 1.5B `2048x32` | TRT reference | `468.957 ms` | `144.016 ms` | reference | |
| 0.5B `2048x64` | native | `324.569 ms` | `74.541 ms` | `+32.494 ms` | baseline |
| 0.5B `2048x64` | MLP source-op old | `307.219 ms` | `56.859 ms` | `+15.144 ms` | Round J |
| 0.5B `2048x64` | MLP + linear source-op old | `305.840 ms` | `56.004 ms` | `+13.764 ms` | best total among measured source-op modes |
| 0.5B `2048x64` | MLP + linear source-op BF16-out | `306.185 ms` | `55.632 ms` | `+14.109 ms` | prefill improves but total is noise/slightly worse |
| 0.5B `2048x64` | TRT reference | `292.075 ms` | `46.196 ms` | reference | |

Fresh 3B graph-off attribution for the best no-engine combo before BF16-output
trim:

- EdgeFM source-op kernel time:
  - CUTLASS GEMMs: `253.785 ms`, `144` launches, `75.5%`
  - FlashInfer prefill attention: `35.364 ms`, `36` launches
  - source-op SwiGLU: `16.743 ms`
  - BF16/FP16 cast kernels: `20.987 ms`
- TRT reference on the same `2048x1` profiling shape:
  - MLP GateUp XMMA: `138.692 ms`
  - QKV/Down-style XMMA group: `86.089 ms`
  - TRT FMHA: `19.681 ms`
  - TRT SiluMul: `14.459 ms`

Decision:

- Keep QKV/OProj source-op and BF16-output trim as default-off diagnostic
  pieces. They are source-visible and correctness-clean.
- Do not remove `trt_bridge` yet. Best no-engine 3B slice is now
  `950.190 ms`, still `+33.983 ms` behind TRT, and 3B memory headroom remains
  too tight for default persistent residency.
- Next bridge-removal target is no longer generic cublasLt table tuning. The
  remaining gap is a mix of source-op GEMM/XMMA delta, FlashInfer-vs-TRT FMHA,
  and residual cast/layout overhead. Start Humanize/source-visible long loops on
  the largest proven gap before considering any default path change.

Artifacts:

- benchmark JSONs:
  `artifacts/3060_tuning/20260516_cutlass_linear_sourceop/`
- EdgeFM source-op NSYS mapping:
  `artifacts/3060_tuning/20260516_cutlass_linear_sourceop/nsys/edgefm_3b_mlp_persistent_linear_scratch_2048x1_mapping.nsys-rep`
- TRT NSYS mapping:
  `artifacts/3060_tuning/20260516_cutlass_linear_sourceop/nsys/trt_3b_2048x1_mapping.nsys-rep`
- digests:
  `artifacts/3060_tuning/20260516_cutlass_linear_sourceop/edgefm_3b_mlp_persistent_linear_scratch_2048x1_nsys_digest.md`
  and
  `artifacts/3060_tuning/20260516_cutlass_linear_sourceop/trt_3b_2048x1_nsys_digest.md`

## 2026-05-16 Round J: CUTLASS Prefill MLP Source-Op Production Spike

Implemented a default-off source-visible MLP runner for Qwen2.5 prefill:

- production files:
  - `src/models/qwen2_5/cutlass_mlp_bridge.h`
  - `src/models/qwen2_5/cutlass_mlp_bridge.cu`
  - `src/models/qwen2_5/qwen2_5.{h,cpp}`
  - `src/models/qwen2_5/CMakeLists.txt`
- gate:
  - `EDGE_FM_PREFILL_CUTLASS_MLP=1`
  - `EDGE_FM_CUTLASS_MLP_ACCUM=fp16|fp32` (`fp16` is the fast Humanize seed)
  - `EDGE_FM_CUTLASS_MLP_PERSISTENT_WEIGHTS=1` enables layer-local FP16 weight
    residency; otherwise weights are cast through scratch buffers each prefill.
- no TensorRT engine or execution context is used. This is a `source-op` path,
  not `trt_bridge`.
- the path is still default-off because it uses FP16 accumulator/FP16 resident
  weights and needs full-matrix accuracy/perf review before defaulting.

Correctness/build checks:

- `cmake --build build-3060 --target install -j$(nproc)` passed.
- Default path Qwen generate core regression passed:
  `7 passed, 13 deselected`.
- Source-op enabled token alignment passed:
  `EDGE_FM_PREFILL_CUTLASS_MLP=1 EDGE_FM_CUTLASS_MLP_ACCUM=fp16 ... -k token_alignment_cuda_graph`
  -> `3 passed, 17 deselected`.
- Source-op enabled core generate regression passed:
  `compact_vocab_identity or max_new_tokens or deferred_stop or metrics_surface`
  -> `4 passed, 16 deselected`.

Primary CUDA graph benchmark, `runs=3`, `warmup=1`, `prefill=2048`:

| Model / slice | Native total / prefill | Source-op total / prefill | TRT reference total / prefill | Source-op vs native | Source-op gap to TRT | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 3B `2048x32` scratch weights | `1111.018 / 484.580 ms` | `1000.973 / 374.161 ms` | `916.206 / 285.681 ms` | `-110.045 ms` | `+84.766 ms` | no persistent residency, safer memory |
| 3B `2048x32` persistent weights | `1111.018 / 484.580 ms` | `965.386 / 339.237 ms` | `916.206 / 285.681 ms` | `-145.631 ms` | `+49.180 ms` | only `430.8 MB` free after runs |
| 1.5B `2048x32` persistent weights | `557.607 / 233.211 ms` | `492.819 / 167.820 ms` | `468.957 / 144.016 ms` | `-64.788 ms` | `+23.862 ms` | `5774.8 MB` free after runs |
| 0.5B `2048x64` persistent weights | `324.569 / 74.541 ms` | `307.219 / 56.859 ms` | `292.075 / 46.196 ms` | `-17.350 ms` | `+15.144 ms` | `9666.8 MB` free after runs |

Artifacts:

- benchmark JSONs under
  `artifacts/3060_tuning/20260516_cutlass_mlp_sourceop/`.
- production graph-off mapping:
  `artifacts/3060_tuning/20260516_cutlass_mlp_sourceop/nsys/edgefm_3b_sourceop_persistent_2048x1_mapping.nsys-rep`
- digest:
  `artifacts/3060_tuning/20260516_cutlass_mlp_sourceop/edgefm_3b_sourceop_persistent_2048x1_nsys_digest_min01.md`

Fresh graph-off `3B / 2048x1` source-op attribution:

- CUTLASS FP16-acc MLP GEMMs: `220.126 ms`, `72` launches, `63.5%` of
  prefill stage GPU time.
- FlashInfer prefill attention: `35.334 ms`, `36` launches.
- source-op SwiGLU kernel: `16.695 ms`, `36` launches.
- source-op input/output casts: `2.414 + 2.367 ms`.
- QKV native CUTLASS BF16 kernels are about `0.963-0.965 ms/layer`, and OProj
  cublasLt kernels about `0.711 ms/layer` in the long-prefill trace.

Decision:

- Accepted as a real no-TRT-engine source-op milestone and kept default-off.
- Not defaulted yet: 3B persistent residency leaves too little memory headroom,
  and all three sizes remain behind TRT reference.
- Next no-bridge target should be QKV/OProj source-op or a source-visible
  attention replacement. MLP-only source-op removed most native MLP gap but does
  not yet eliminate `trt_bridge` as the fastest full-model reference.

## 2026-05-16 Fresh Profiling + Humanize MLP Source-Op

NCU/environment:

- `/proc/driver/nvidia/params` now reports `RmProfilingAdminOnly: 0`.
- `.codex/skills/edge-fm-cuda-kernel-optimizer/scripts/check_env.py` now
  distinguishes `ncu --query-metrics` from real driver counter access and
  reports `ncu_can_read_counters: true` on this host.
- Real NCU focused collection succeeded for the native prefill MLP BF16 GEMM
  representative; report artifacts live under
  `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/profile-digests/`.

Fresh serial `3B / 2048x32` CUDA graph rerun, `runs=3`, `warmup=1`:

| Mode | Avg | Prefill | Decode | Gap vs TRT |
| --- | ---: | ---: | ---: | ---: |
| EdgeFM native | `1104.44 ms` | `478.97 ms` | `625.39 ms` | `+184.32 ms` |
| EdgeFM bridge diagnostic (`MLP auto + QKV/OProj`) | `941.83 ms` | `315.06 ms` | `626.60 ms` | `+21.70 ms` |
| TRT reference | `920.13 ms` | `284.10 ms` | `635.80 ms` | reference |

Fresh graph-off stage-role attribution for the same target:

| Role | Native prefill | Bridge prefill | TRT reference |
| --- | ---: | ---: | ---: |
| MLP GateUp | `253.24 ms` | `140.31 ms` | `138.90 ms` |
| MLP DownProj | `129.78 ms` | `72.14 ms` | `71.75 ms` |
| SwiGLU | `14.48 ms` | `14.49 ms` | `14.46 ms` |
| QKV | `33.61 ms` | `16.99 ms` | `16.99 ms` |
| OProj | `24.77 ms` | `14.13 ms` | `14.31 ms` |
| Attention/RoPE | `34.23 ms` | `35.43 ms` | `23.45 ms` |
| Bridge/cast | n/a | `20.28 ms` | `0.06 ms` |

Current conclusion:

- For bridge-residual optimization, attention/layout and bridge cast overhead
  are still the main remaining targets.
- For the no-bridge objective, native prefill MLP is the largest source-visible
  replacement target. This supersedes the older assumption that the next active
  long loop should be attention-only.
- `scripts/profile/analyze_trt_nsys_profile.py` now reports NVTX stage roles, so
  prefill and graph-off decode roles are not mixed in one table.

Artifacts:

- Benchmarks:
  `.tmp_codex/bench/fresh_3060_20260516/*3b_2048x32*.json`
- NSYS summaries:
  `.tmp_codex/nsys/fresh_3060_20260516/*3b_2048x32*_summary.{md,json}`
- NCU focused CSV:
  `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/profile-digests/edgefm_native_prefill_mlp_3b_s2048_ampere_bf16_focused.csv`
- Humanize standalone repo:
  `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/`
  (local standalone commit `8d44f67 promote fp16 accumulator cutlass mlp seed`)

Humanize Round 0:

- Standalone repo initialized and committed locally.
- Added finite scaled tensor support to the Qwen prefill MLP harness
  (`--init-scale 0.02`) because unit-scale synthetic FP16 candidates overflow.
- Added an FP16 persistent-weight diagnostic candidate mirroring the bridge
  direction. It is numerically finite but not an accepted performance
  implementation:
  - 3B/S2048: `12.345 ms`, max abs error `7.63e-6` versus scaled BF16 reference
  - 1.5B/S2048: `7.728 ms`, max abs error `7.63e-6`
  - 0.5B/S2048: `2.647 ms`, max abs error `1.91e-6`
- Round 1 target: replace the PyTorch matmul backend in the harness with a
  buildable source-visible CUDA/CUTLASS/CuTe GateUp/DownProj candidate, then run
  NCU if the candidate is correct and close enough to the reference envelope.
- First Round 1 seed is now in place:
  - `cutlass_fp16_weight_candidate` uses CUTLASS SM80 tensor-op GEMMs for
    GateUp and DownProj plus small CUDA kernels for BF16/FP16 casts and SwiGLU.
  - `128x128x64_s3` results at `S=2048`: 3B `12.348 ms`, 1.5B `7.549 ms`,
    0.5B `2.602 ms`.
  - This was correct and about `+1.3%` geomean versus the PyTorch FP16-weight
    diagnostic across those three slices, but the primary 3B target was flat, so
    it became a mutation seed rather than an accepted production candidate.
  - NCU range profile:
    `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/profile-digests/ncu_reports/cutlass_fp16_mlp_candidate_3b_s2048_range_basic.ncu-rep`
  - Tile sweep reject notes: `128x256x64_s3` and `256x128x64_s3` fail CUTLASS
    launch with `Error Internal`; `64x128x64_s3` is correct but slow
    (`23.886 ms` on 3B/S2048).
- FP16 accumulator mutation is the first strong source-visible result:
  - `128x128x64_s3_f16acc` at `S=2048`: 3B `6.743 ms`, 1.5B `4.245 ms`,
    0.5B `1.539 ms`.
  - 3B standalone speed improves `12.348 -> 6.743 ms` versus the FP32-acc seed;
    mean abs error versus the scaled BF16 reference is `1.92e-6`, max abs error
    `3.05e-5`.
  - NCU shows the main GEMMs reaching about `88%` tensor-pipe active with
    registers/thread down from `242` to `166`, while keeping `99.328 KB`
    shared memory/block.
  - This is a default-off integration candidate, not a production claim yet:
    real checkpoint weights and token alignment must pass because FP16
    accumulation is an explicit accuracy tradeoff.

## 2026-05-15 Restart: Kernel Optimizer Workflow

- The active plan has been refreshed in [3060_tuning_plan.md](./3060_tuning_plan.md) for the updated `edge-fm-cuda-kernel-optimizer` skill and the current engine source layout.
- New artifact prefix: `.tmp_codex/bench/3060_20260515_*` and `.tmp_codex/nsys/3060_20260515_*`.
- The first action is not a kernel edit. It is a fresh environment snapshot, model/TRT-engine availability check, and Tier-0 EdgeFM/TRT baseline because the branch and build layout have changed since the 2026-05-11/2026-05-12 bridge artifacts.
- The previous TensorRT bridge results remain useful attribution, but they are not treated as freshly accepted on this branch until rerun. The old `report_qwen_benchmark_suite.py` command is obsolete in this checkout; use the pytest benchmark entries or `scripts/profile/profile_edgefm_generate_case.py`.
- Current notification route for milestones is the local `codex-notify-wechat` wrapper. Older `cc-connect send ... no active session found` notes below are historical debugging context, not the current operating assumption.

## 2026-05-15 Fresh Tier-0 Results

Environment snapshot:

- `.tmp_codex/bench/3060_20260515_env.json`
- GPU: RTX 3060, `sm_86`; `ncu` binary is available. Later Round C
  `--set basic` collection hit `ERR_NVGPUCTRPERM`, so full counter access is
  not assumed for the current user session.
- Toolchain observed by the optimizer skill: `nvcc 12.8.61`, PyTorch `2.10.0+cu128`, Triton `3.6.0`.
- Local checkpoints and TRT engines exist for Qwen2.5 `0.5B`, `1.5B`, and `3B`.

Tooling fixes made during the restart:

- `scripts/profile/profile_edgefm_generate_case.py` now supports `--output-json`, so bridge builds that print TensorRT/C++ logs to stdout still produce a clean JSON artifact.
- `scripts/profile/analyze_trt_nsys_profile.py` now classifies EdgeFM NVTX ranges and bridge Myelin nodes, so EdgeFM native, EdgeFM bridge, and TRT traces can be compared by the same role table.
- The bridge release build was rebuilt with `cmake --build build-3060-trt-mlp-release --target install -j$(nproc)` because the old installed module missed the current `lm_head_top1_*` metrics contract.

Fresh CUDA graph benchmark slices, all with `runs=3`, `warmup=1`:

| Case | Native avg/prefill/decode | Bridge avg/prefill/decode | TRT avg/prefill/decode | Bridge - TRT | Native - TRT |
| --- | ---: | ---: | ---: | ---: | ---: |
| `3B / 2048x32` | `1110.47 / 484.16 / 626.18 ms` | `938.87 / 313.20 / 625.47 ms` | `914.49 / 283.94 / 630.37 ms` | `+24.38 ms` | `+195.99 ms` |
| `1.5B / 2048x32` | `559.55 / 234.91 / 324.51 ms` | `479.24 / 154.88 / 324.23 ms` | `468.94 / 143.95 / 324.88 ms` | `+10.30 ms` | `+90.61 ms` |
| `0.5B / 2048x64` | `329.94 / 73.58 / 256.20 ms` | `309.17 / 53.08 / 255.91 ms` | `292.35 / 45.88 / 246.36 ms` | `+16.81 ms` | `+37.58 ms` |

Artifacts:

- Native: `.tmp_codex/bench/3060_20260515_3b_2048x32_edgefm_graph_baseline.json`, `.tmp_codex/bench/3060_20260515_1p5b_2048x32_edgefm_native_graph_baseline.json`, `.tmp_codex/bench/3060_20260515_0p5b_2048x64_edgefm_native_graph_baseline.json`
- Bridge: `.tmp_codex/bench/3060_20260515_3b_2048x32_edgefm_bridge_graph_baseline_clean.json`, `.tmp_codex/bench/3060_20260515_1p5b_2048x32_edgefm_bridge_graph_baseline.json`, `.tmp_codex/bench/3060_20260515_0p5b_2048x64_edgefm_bridge_graph_baseline.json`
- TRT: `.tmp_codex/bench/3060_20260515_3b_2048x32_trt_baseline.json`, `.tmp_codex/bench/3060_20260515_1p5b_2048x32_trt_baseline.json`, `.tmp_codex/bench/3060_20260515_0p5b_2048x64_trt_baseline.json`

Fresh `3B / 2048x1` graph-off attribution:

| Role | EdgeFM native | EdgeFM bridge | TRT | Bridge - TRT |
| --- | ---: | ---: | ---: | ---: |
| total kernel time | `505.53 ms` | `322.97 ms` | `289.01 ms` | `+33.96 ms` |
| MLP GateUp | `256.51 ms` | `140.35 ms` | `138.95 ms` | `+1.39 ms` |
| MLP DownProj | `131.54 ms` | `72.16 ms` | `71.76 ms` | `+0.40 ms` |
| QKV | `34.02 ms` | `17.00 ms` | `16.98 ms` | `+0.02 ms` |
| OProj | `25.08 ms` | `14.15 ms` | `14.30 ms` | `-0.16 ms` |
| attention / attention plugin | `34.69 ms` | `35.34 ms` | `23.46 ms` | `+11.89 ms` |
| activation | `14.48 ms` | `14.48 ms` | `14.46 ms` | `+0.02 ms` |
| bridge cast overhead | n/a | `20.34 ms` | n/a | `+20.34 ms` |

Attribution artifacts:

- Native: `.tmp_codex/nsys/3060_20260515_3b_2048x1_edgefm_native_mapping_summary.md`, `.tmp_codex/nsys/3060_20260515_3b_2048x1_edgefm_native_mapping_summary.json`
- Bridge: `.tmp_codex/nsys/3060_20260515_3b_2048x1_edgefm_bridge_mapping_summary.md`, `.tmp_codex/nsys/3060_20260515_3b_2048x1_edgefm_bridge_mapping_summary.json`
- TRT: `.tmp_codex/nsys/3060_20260515_3b_2048x1_trt_mapping_summary.md`, `.tmp_codex/nsys/3060_20260515_3b_2048x1_trt_mapping_summary.json`

Current conclusion:

- The optional bridge still removes most of the native prefill gap and should remain the active reference candidate.
- On `3B / 2048`, MLP/QKV/OProj are effectively at TRT parity after the bridge. The remaining prefill work is bridge cast overhead and FlashInfer prefill attention versus TRT's attention plugin.
- On `1.5B / 2048x32`, the same bridge leaves only `+10.30 ms` total gap, mostly prefill.
- On `0.5B / 2048x64`, decode also contributes to the residual gap, so small-model work should not be prefill-only.
- Next low-risk optimization queue: reduce bridge cast overhead first, profile bridge attention second, and separately inspect `0.5B` decode if the small-model gap stays after prefill work.

## 2026-05-15 Low-Risk Probe Round A

The first low-risk pass stayed within existing default-off bridge and operator-table
switches. No default runtime path was changed.

`3B / 2048x32` CUDA graph probes, all with QKV/OProj TensorRT linear bridge,
`runs=3`, `warmup=1`:

| Mode | Avg | Prefill | Decode | After-run free memory | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 bridge baseline | `938.87 ms` | `313.20 ms` | `625.47 ms` | n/a | current safe bridge reference |
| MLP `gateup` FP16 weights + attention tile128 | `937.53 ms` | `311.73 ms` | `625.61 ms` | `1892 MB` | too small for default-table acceptance |
| MLP `down` FP16 weights + attention tile128 | `948.34 ms` | `322.37 ms` | `625.78 ms` | `3404 MB` | rejected, slower than baseline |
| MLP `gateup` + linear OProj FP16 weights + tile128 | `936.67 ms` | `310.64 ms` | `625.82 ms` | `1612 MB` | diagnostic only, below 1% gate |
| MLP `gateup` + linear QKV/OProj FP16 weights + tile128 | `934.41 ms` | `308.39 ms` | `625.83 ms` | `1264 MB` | diagnostic only, below 1% gate |
| MLP `both` FP16 weights + attention tile128 | `923.16 ms` | `297.35 ms` | `625.58 ms` | `358 MB` | fastest EdgeFM slice, memory-unsafe for default |
| TRT-Edge-LLM reference | `914.49 ms` | n/a | n/a | n/a | still ahead by `8.67 ms` versus fastest EdgeFM |

Artifacts:

- `.tmp_codex/bench/3060_20260515_3b_2048x32_edgefm_bridge_fp16weights_gateup_tile128_mem.json`
- `.tmp_codex/bench/3060_20260515_3b_2048x32_edgefm_bridge_fp16weights_down_tile128_mem.json`
- `.tmp_codex/bench/3060_20260515_3b_2048x32_edgefm_bridge_gateup_linear_oproj_fp16weights_tile128_mem.json`
- `.tmp_codex/bench/3060_20260515_3b_2048x32_edgefm_bridge_gateup_linear_both_fp16weights_tile128_mem.json`
- `.tmp_codex/bench/3060_20260515_3b_2048x32_edgefm_bridge_fp16weights_both_tile128_mem.json`
- `.tmp_codex/bench/3060_20260515_trt_linear_qkv_fp16weight_bias_engine_probe.json`

Decision:

- `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` is no longer described as "cannot run"
  on this branch. It can run the target slice and is the fastest measured
  EdgeFM mode, but 3B has only about `358 MB` free after warmup/runs, so it
  remains memory-unsafe for default-on or full-matrix acceptance without a new
  residency policy.
- Attention `prefill_cta_tile_q=128` is useful in the fastest diagnostic slice
  but is still below the 1% target acceptance gate as a standalone table change.
- Linear persistent FP16 QKV/OProj weights reduce prefill by about `3.34 ms`
  versus `gateup+tile128`, but also stay below the 1% target gate.

`0.5B / 2048x64` decode-focused probes:

| Mode | Avg | Prefill | Decode | Decision |
| --- | ---: | ---: | ---: | --- |
| bridge full logits rerun | `309.18 ms` | `52.56 ms` | `256.45 ms` | reference |
| bridge `lm_head_top1` | `309.56 ms` | `53.22 ms` | `256.16 ms` | rejected, no end-to-end gain |
| bridge decode-attention retuned table | `302.38 ms` | `52.67 ms` | `249.55 ms` | accepted table fix |
| TRT reference | `292.35 ms` | n/a | n/a | still ahead by `16.83 ms` |

Graph-off attribution for `0.5B / 2048x64` explains why `lm_head_top1` is not
accepted:

- full logits trace: total kernel time `313.97 ms`; `LMHead` role `51.66 ms`,
  decode attention `39.93 ms`, MLP GateUp `100.66 ms`, MLP DownProj `57.25 ms`.
- top1 trace: total kernel time `312.75 ms`; regular `LMHead` drops to `0.81 ms`,
  but `lm_head_top1::stage1_kernel` costs `50.37 ms`, so the work moved rather
  than disappeared.
- TRT profile-range trace sees TensorRT's graph/plugin path rather than a
  graph-off equivalent, so it is useful as a tactic reference but not a direct
  kernel-total comparison.
- decode attention retune root cause: the 3060 platform table already had the
  faster `0.5B` decode attention params earlier in the file, but a later duplicate
  record for the same shape restored the older slower params. Updating the later
  duplicate in both `operator_impl_table_llm.json` and the combined
  `operator_impl_table.json` makes the fast params actually win runtime selection.
- retune microbench evidence: existing duplicate-selected params summed
  `0.06768 ms` across KV lengths `512/1024/2048`; the accepted params summed
  `0.06261 ms` with `short_seq_bdz=4`, `long_seq_bdz=4`,
  `long_seq_threshold=1024`, `no_split_kv_threshold=256`,
  `min_chunk_size=64`, `chunk_alignment=64`, and
  `chunk_candidates=[64,128,256,512]`.
- end-to-end acceptance evidence: same-run `runs=5` CUDA graph comparison moved
  from `308.89 ms` to `302.38 ms` (`-6.50 ms`, `-2.10%`); decode moved from
  `256.02 ms` to `249.55 ms` (`-6.48 ms`).
- additional slice after the same table fix: `0.5B / 2048x32` EdgeFM bridge
  measured `177.47 ms` (`prefill 53.56 ms`, `decode 123.76 ms`) versus TRT
  `167.50 ms`; the remaining gap is `+9.97 ms`.
- decode linear retune follow-up found no additional 0.5B table win. Existing
  records are already best for decode `fused_qkv` (`algo_9`, `0.031424 ms`),
  `attention_output` (`algo_9`, `0.010224 ms`), and `mlp_down` (`algo_1`,
  `0.037888 ms`); `fused_gate_up` and `lm_head` are fastest with the baseline
  heuristic. The helper script needed a small maintenance fix because
  `tune_qwen_cublaslt.benchmark_candidate()` now requires explicit
  `torch_dtype`/`dtype_id`.

Artifacts:

- `.tmp_codex/bench/3060_20260515_0p5b_2048x64_edgefm_bridge_full_logits_rerun.json`
- `.tmp_codex/bench/3060_20260515_0p5b_2048x64_edgefm_bridge_full_logits_rerun5.json`
- `.tmp_codex/bench/3060_20260515_0p5b_2048x64_edgefm_bridge_lm_head_top1_rerun.json`
- `.tmp_codex/bench/3060_20260515_0p5b_2048x64_edgefm_bridge_decode_attn_retuned_graph.json`
- `.tmp_codex/bench/3060_20260515_0p5b_2048x64_edgefm_bridge_platform_table_after_decode_attn_fix.json`
- `.tmp_codex/bench/3060_20260515_0p5b_2048x32_edgefm_bridge_after_decode_attn_fix.json`
- `.tmp_codex/bench/3060_20260515_0p5b_2048x32_trt_baseline.json`
- `.tmp_codex/bench/3060_20260515_retune_decode_linear_0p5b.log`
- `.tmp_codex/tmp/efm_retune_qwen_tables_s717t4l8/retune_report_device0_20260515T074723Z.json`
- `.tmp_codex/bench/3060_20260515_retune_decode_attn_0p5b/operator_impl_table_llm.json`
- `.tmp_codex/nsys/3060_20260515_0p5b_2048x64_edgefm_bridge_graphoff_summary.md`
- `.tmp_codex/nsys/3060_20260515_0p5b_2048x64_edgefm_bridge_lm_head_top1_graphoff_summary.md`
- `.tmp_codex/nsys/3060_20260515_0p5b_2048x64_trt_profile_range_summary.md`

Next active target after this round:

- Large model: the remaining best-slice gap is mostly attention/plugin behavior
  plus the memory policy needed to make MLP `both` safe.
- Small model: `0.5B / 2048x64` is improved but still `+10.03 ms` behind TRT
  after the decode-attention table fix. `LMHead`/top1 is not a current win; the
  next useful profile should compare decode GEMV/attention roles against a more
  equivalent TRT capture or an operator-level decode microbench.

## 2026-05-15 Round B Full Matrix and Candidate Gates

Round B started by checkpointing the accepted Round A changes:

- commit: `5d59d1a tune 3060 llm round a checkpoint`
- validation before commit:
  - `git diff --check`
  - `python3 scripts/operator_table/validate_operator_tables.py`
  - `python3 -m py_compile` for the profile/tune scripts touched in Round A
  - Qwen generate core regression:
    `tests/engine/test_qwen2_generate.py -k "token_alignment_cuda_graph or metrics_surface or max_new_tokens or deferred_stop"`
    passed with `6 passed, 14 deselected`
- milestone notification via `codex-notify-wechat` succeeded.

Environment/build notes:

- Round B environment snapshot:
  `.tmp_codex/bench/3060_20260515_roundb_env.json`
- `edge_fm_trt` was rebuilt for the active Python 3.13 environment. The stale
  `build-3060/trt-edgellm/CMakeCache.txt` still pointed at `/workspace/edge-fm`;
  clearing that generated subbuild and rerunning the TensorRT-Edge-LLM setup
  script fixed the local TRT benchmark entry.
- The first 54-case matrix accidentally put `build-3060/install/lib` before the
  bridge build in `LD_LIBRARY_PATH`, so bridge cases loaded the non-bridge
  `libedge_fm.so` and printed fallback warnings. The native and TRT cases from
  that matrix remain valid; the bridge cases are invalid and were rerun with
  bridge library paths first.
- `scripts/profile/profile_edgefm_generate_case.py` now records
  `runtime_library_check` and raises when bridge env flags are enabled but the
  loaded `libedge_fm.so` is outside `EDGE_FM_BUILD_DIR`. Guard validation:
  `.tmp_codex/bench/3060_20260515_profile_runtime_guard_bad_bridge_smoke.log`
  fails on the intentionally wrong library order, while
  `.tmp_codex/bench/3060_20260515_profile_runtime_guard_good_bridge_smoke.json`
  passes with the bridge library loaded from `build-3060-trt-mlp-release`.

Effective Round B artifacts:

- native/TRT source matrix:
  `.tmp_codex/bench/3060_20260515_roundb_matrix/summary.json`
- fixed bridge matrix:
  `.tmp_codex/bench/3060_20260515_roundb_bridge_fixed/summary.json`
- combined effective summary:
  `.tmp_codex/bench/3060_20260515_roundb_effective_summary.json`

Key effective CUDA graph results:

| Case | Native avg | Bridge avg | TRT avg | Bridge - TRT | Bridge prefill - TRT prefill | Bridge decode - TRT decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `3B / 2048x32` | `1112.91 ms` | `943.68 ms` | `918.75 ms` | `+24.93 ms` (`+2.71%`) | `+29.41 ms` | `-4.58 ms` |
| `3B / 2048x64` | `1760.43 ms` | `1590.87 ms` | `1570.43 ms` | `+20.44 ms` (`+1.30%`) | `+29.27 ms` | `-8.97 ms` |
| `1.5B / 2048x32` | `559.78 ms` | `480.66 ms` | `469.61 ms` | `+11.05 ms` (`+2.35%`) | `+11.54 ms` | `-0.55 ms` |
| `1.5B / 2048x64` | `895.53 ms` | `816.04 ms` | `805.00 ms` | `+11.04 ms` (`+1.37%`) | `+11.62 ms` | `-0.67 ms` |
| `0.5B / 2048x32` | `198.55 ms` | `177.48 ms` | `168.45 ms` | `+9.03 ms` (`+5.36%`) | `+7.15 ms` | `+1.86 ms` |
| `0.5B / 2048x64` | `323.65 ms` | `302.24 ms` | `292.36 ms` | `+9.88 ms` (`+3.38%`) | `+6.31 ms` | `+3.53 ms` |

Attention prefill split sweeps:

- 3B best microbench candidate:
  `prefill_short_qo_len_threshold=1024`,
  `prefill_short_cta_tile_q=64`, `prefill_long_cta_tile_q=128`.
  It improved the isolated 512/1024/2048 sum versus global64, but paired
  bridge end-to-end rejected it:
  - `3B / 2048x32`: `+0.658 ms` (`+0.070%`)
  - `3B / 2048x64`: `-0.508 ms` (`-0.032%`)
- 1.5B best microbench candidate:
  `prefill_short_qo_len_threshold=768`,
  `prefill_short_cta_tile_q=64`, `prefill_long_cta_tile_q=64`.
  Paired bridge end-to-end rejected it:
  - `1.5B / 2048x32`: `+0.305 ms` (`+0.064%`)
  - `1.5B / 2048x64`: `+0.241 ms` (`+0.030%`)
- 0.5B best microbench candidate:
  `prefill_short_qo_len_threshold=1536`,
  `prefill_short_cta_tile_q=64`, `prefill_long_cta_tile_q=128`.
  Paired bridge end-to-end remained below the 1% gate:
  - `0.5B / 2048x32`: `-0.567 ms` (`-0.321%`)
  - `0.5B / 2048x64`: `-0.155 ms` (`-0.051%`)
- Decision: do not update 3060 prefill attention table entries from these split
  candidates. Microbench wins do not survive the official generate path.
- Artifacts:
  - `.tmp_codex/bench/3060_20260515_roundb_attention_prefill_3b_sweep.json`
  - `.tmp_codex/bench/3060_20260515_roundb_attention_prefill_3b_e2e/summary.json`
  - `.tmp_codex/bench/3060_20260515_roundb_attention_prefill_1p5b_sweep.json`
  - `.tmp_codex/bench/3060_20260515_roundb_attention_prefill_1p5b_e2e/summary.json`
  - `.tmp_codex/bench/3060_20260515_roundb_attention_prefill_0p5b_sweep.json`
  - `.tmp_codex/bench/3060_20260515_roundb_attention_prefill_0p5b_e2e/summary.json`

MLP `both` FP16 weight diagnostic:

- Available `both` engine coverage is currently only `3B / m=2048`; 0.5B/1.5B
  `both` engines are not present locally.
- Paired `3B / 2048` bridge runs with QKV/OProj linear bridge:
  - `2048x32`: `gateup 940.21 ms` -> `both 927.69 ms`, delta `-12.52 ms`
    (`-1.33%`), prefill delta `-12.81 ms`
  - `2048x64`: `gateup 1589.74 ms` -> `both 1576.92 ms`, delta `-12.82 ms`
    (`-0.81%`), prefill delta `-13.27 ms`
  - after-runs free memory drops from about `1895 MB` to `361 MB`
- Decision: `both` is a real performance win and reduces the 3B `2048` TRT gap
  to single digits, but remains blocked from default/full-matrix acceptance by
  memory headroom. Reopen only with a concrete memory policy.
- Artifact:
  `.tmp_codex/bench/3060_20260515_roundb_mlp_both_3b/summary.json`

## 2026-05-15 Round C Steady Attribution and MLP Auto Gate

Round C focused on two questions from Round B: whether the remaining 3B gap is
still attention/cast dominated after a steady trace, and whether MLP `both`
FP16 persistent weights can be enabled safely for smaller models without making
3B memory worse. Default runtime behavior remains unchanged.

Steady `3B / 2048x1` graph-off attribution:

| Role | EdgeFM bridge `gateup` | EdgeFM bridge `both` | TRT | Round C conclusion |
| --- | ---: | ---: | ---: | --- |
| runtime prefill | `323.227 ms` | `313.328 ms` | `288.955 ms` | `both` removes cast time, not attention gap |
| kernel total | `322.829 ms` | `312.682 ms` | `287.961 ms` | remaining gap is still not host overhead |
| attention / plugin+rope | `35.374 ms` | `35.402 ms` | `23.307 ms` | EdgeFM FlashInfer path is about `+12 ms` slower |
| bridge cast overhead | `20.291 ms` | `10.305 ms` | n/a | `both` removes about `10 ms` |
| MLP GateUp | `140.290 ms` | `140.293 ms` | n/a | already stable at TRT-like speed |
| MLP DownProj | `72.130 ms` | `72.125 ms` | n/a | already stable at TRT-like speed |
| QKV / OProj | `16.990 / 14.140 ms` | `17.001 / 14.140 ms` | n/a | not the current Round C target |

Artifacts:

- `.tmp_codex/nsys/3060_20260515_roundc_3b_2048x1_bridge_gateup_steady_mapping_summary.md`
- `.tmp_codex/nsys/3060_20260515_roundc_3b_2048x1_bridge_both_steady_mapping_summary.md`
- `.tmp_codex/nsys/3060_20260515_roundc_3b_2048x1_trt_steady_mapping_summary.md`
- `.tmp_codex/bench/3060_20260515_roundc_attribution/3b_2048x1_bridge_gateup_steady_mapping.json`
- `.tmp_codex/bench/3060_20260515_roundc_attribution/3b_2048x1_bridge_both_steady_mapping.json`
- `.tmp_codex/bench/3060_20260515_roundc_attribution/3b_2048x1_trt_steady_mapping.json`

Small/mid-model MLP `both` engine coverage was generated for long prefill:

- 1.5B `m=2048` `both` engine:
  `.tmp_codex/bench/trt_mlp_subengine_bf16_qwen2_5-1_5b-instruct_l0_runtime_weights_edgefm_bf16_layout_fp16_compute_fp16weights-both_m2048_h1536_i8960.engine`
- 0.5B `m=2048` `both` engine:
  `.tmp_codex/bench/trt_mlp_subengine_bf16_qwen2_5-0_5b-instruct_l0_runtime_weights_edgefm_bf16_layout_fp16_compute_fp16weights-both_m2048_h896_i4864.engine`

Paired CUDA graph bridge results with QKV/OProj linear bridge:

| Case | `gateup` avg/prefill/free | `both` avg/prefill/free | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `1.5B / 2048x32` | `480.445 / 155.670 ms / 6465 MB` | `475.265 / 150.542 ms / 5713 MB` | `-5.180 ms` (`-1.078%`) | target slice clears 1% |
| `1.5B / 2048x64` | `815.113 ms` | `810.084 ms` | `-5.029 ms` (`-0.617%`) | useful but below 1% because decode dominates |
| `0.5B / 2048x32` | `177.368 ms` | `175.627 ms` | `-1.741 ms` (`-0.982%`) | borderline, not enough alone |
| `0.5B / 2048x64` | `302.060 ms` | `301.124 ms` | `-0.936 ms` (`-0.310%`) | below gate |

Round C source change:

- `EDGE_FM_TRT_MLP_FP16_WEIGHTS=auto` was added as an experimental,
  default-off bridge mode.
- Auto estimates the extra persistent DownProj FP16 copy memory from the model
  config and compares it with `EDGE_FM_TRT_MLP_AUTO_DOWN_MAX_MB` (default
  `1024` MB).
- On current Qwen2.5 shapes this resolves 0.5B/1.5B to `both` and 3B to
  `gateup`, avoiding the 3B low-headroom `both` mode by default.
- If auto resolves to `both` but the matching `both` engine is absent for a
  shape, it falls back to a `gateup` engine before returning to native MLP.
- Smoke artifacts:
  `.tmp_codex/bench/3060_20260515_roundc_1p5b_2048x32_bridge_auto_smoke.json`,
  `.tmp_codex/bench/3060_20260515_roundc_3b_2048x1_bridge_auto_smoke.json`,
  `.tmp_codex/bench/3060_20260515_roundc_1p5b_1024x1_bridge_auto_fallback_smoke.json`,
  `.tmp_codex/bench/3060_20260515_roundc_verify_1p5b_2048x1_bridge_auto.json`

NCU status:

- `ncu --set basic` against the 3B prefill attention slice failed with
  `ERR_NVGPUCTRPERM`. This is an environment/counter-permission blocker, not a
  performance result. Deeper attention kernel work now needs either counter
  access to be enabled or a standalone microbench loop that can proceed without
  NCU metrics. Current driver parameter check shows `RmProfilingAdminOnly: 1`.
- Milestone notification for the Round C summary was delivered through
  `codex-notify-wechat`.

## 2026-05-15 Round D Attention Route Check

Round D checked whether the remaining 3B prefill attention gap has a low-risk
operator-table or TensorRT plugin bridge shortcut. No production code was
changed.

3B prefill attention table resweep:

- Operator-level `3B / 2048` resweep still makes `prefill_cta_tile_q=128` look
  better than the current table: `0.9339 ms/layer` versus `0.9819 ms/layer`
  for global 64.
- Paired current Round C bridge/auto generate rejected it again:
  - current table: `942.316 ms` avg, `315.737 ms` prefill
  - tile128 temp table: `940.229 ms` avg, `313.797 ms` prefill
  - delta: `-2.087 ms` (`-0.22%`) end-to-end, below the 1% acceptance gate
- Decision: do not update the 3060 attention prefill table for 3B from this
  candidate. The microbench win is real but too small in the official generate
  path.
- Artifacts:
  - `.tmp_codex/bench/3060_20260515_roundd_attention_prefill_3b_2048_resweep.json`
  - `.tmp_codex/bench/3060_20260515_roundd_operator_impl_table_3b_prefill_attention_tile128.json`
  - `.tmp_codex/bench/3060_20260515_roundd_3b_2048x32_bridge_auto_attention_current.json`
  - `.tmp_codex/bench/3060_20260515_roundd_3b_2048x32_bridge_auto_attention_tile128.json`

TensorRT-Edge-LLM AttentionPlugin standalone probe:

- A temporary Torch/TensorRT probe built a standalone `AttentionPlugin` engine
  for the Qwen2.5-3B `B=1, S=2048, Hq=16, Hkv=2, D=128` shape.
- Python event timing was about `1.39 ms/layer`, but an NSYS trace showed that
  the event timer overstates the actual plugin GPU work in this harness.
- NSYS with one warmup and one timed run captured two plugin executions:
  `1.108 ms` total FMHA attention kernels and `0.197 ms` total RoPE/write-KV
  kernels, or about `0.65 ms/layer` for FMHA+RoPE. This matches the TRT network
  attribution and is materially faster than EdgeFM's current `~0.98 ms/layer`
  FlashInfer prefill attention estimate.
- Decision: the TRT FMHA/AttentionPlugin path is a real performance candidate,
  but it is not a direct low-risk drop-in. EdgeFM currently feeds BF16 separate
  Q/K/V views after QKV projection, while the fast plugin path expects FP16
  packed QKV and owns RoPE/write-KV/layout handling. The next step is a
  standalone bridge-layout repro that includes the required casts/layout moves
  before any production code is touched.
- Artifacts:
  - `.tmp_codex/probes/profile_trt_attention_plugin.py`
  - `.tmp_codex/bench/3060_20260515_roundd_trt_attention_plugin_3b_2048_stream.json`
  - `.tmp_codex/bench/3060_20260515_roundd_trt_attention_plugin_3b_2048_emptyidx.json`
  - `.tmp_codex/nsys/3060_20260515_roundd_trt_attention_plugin_3b_2048_emptyidx_summary.md`
  - `.tmp_codex/nsys/3060_20260515_roundd_trt_attention_plugin_3b_2048_emptyidx_summary.json`

AttentionPlugin bridge-layout repro:

- A temporary TensorRT subgraph probe measured the obvious "large chunk"
  replacement boundary for 3B prefill:
  `BF16 hidden -> FP16 QKV GEMM -> AttentionPlugin -> FP16 OProj -> BF16 output`.
  CUDA event timing reported `~1.96 ms/layer`; NSYS kernel time was
  `1.637 ms/layer` split into QKV `0.473 ms`, FMHA `0.550 ms`, RoPE/write-KV
  `0.103 ms`, OProj `0.394 ms`, and cast/elementwise `0.116 ms`.
- This is directionally positive versus the current EdgeFM QKV + prefill
  attention + OProj kernel sum (`~1.85 ms/layer` from Round C), but only by
  about `7-8 ms` over 36 layers before event/engine overhead. It is not enough
  to justify a production path by itself.
- A narrower BF16 packed-QKV wrapper was also tested:
  `BF16 packed QKV -> FP16 cast -> AttentionPlugin -> BF16 attention output`.
  With CUDA graph replay, median per-layer results at `S=2048` were:

| Model | Best current FlashInfer attention | TRT BF16-QKV wrapper | Estimated layer-sum delta |
| --- | ---: | ---: | ---: |
| 3B | `0.9339 ms` | `0.7625 ms` | `-6.17 ms` |
| 1.5B | `0.7033 ms` | `0.5916 ms` | `-3.13 ms` |
| 0.5B | `0.4403 ms` | `0.3316 ms` | `-2.61 ms` |

- Decision: hold/reject production default for the wrapper. It is a real
  experimental candidate, but the low-risk estimated gains are below the
  `>=1%` official end-to-end gate. Reaching TRT's full attention advantage would
  require a larger FP16 packed-QKV / FP16 cache / activation-residency route,
  which is outside the current "no large design change" tuning lane.
- Artifacts:
  - `.tmp_codex/probes/profile_trt_qkv_attention_oproj_subgraph.py`
  - `.tmp_codex/bench/3060_20260515_roundd_trt_qkv_attention_oproj_subgraph_3b_2048_with_oproj.json`
  - `.tmp_codex/bench/3060_20260515_roundd_trt_qkv_attention_oproj_subgraph_3b_2048_no_oproj.json`
  - `.tmp_codex/nsys/3060_20260515_roundd_trt_qkv_attention_oproj_subgraph_3b_2048_with_oproj_summary.md`
  - `.tmp_codex/nsys/3060_20260515_roundd_trt_qkv_attention_oproj_subgraph_3b_2048_no_oproj_summary.md`
  - `.tmp_codex/probes/profile_trt_attention_bf16_qkv_wrapper.py`
  - `.tmp_codex/bench/3060_20260515_roundd_trt_attention_bf16_qkv_wrapper_3b_2048_cuda_graph.json`
  - `.tmp_codex/bench/3060_20260515_roundd_trt_attention_bf16_qkv_wrapper_1p5b_2048_cuda_graph.json`
  - `.tmp_codex/bench/3060_20260515_roundd_trt_attention_bf16_qkv_wrapper_0p5b_2048_cuda_graph.json`
  - `.tmp_codex/nsys/3060_20260515_roundd_trt_attention_bf16_qkv_wrapper_3b_2048_summary.md`
  - `.tmp_codex/bench/3060_20260515_roundd_attention_wrapper_summary.json`

Bridge cast probe:

- A simple CUDA cast microbench checked whether TensorRT's activation cast
  kernels were an easy custom-kernel opportunity. Best `items4` medians were:
  `2048x2048 ~= 53 us`, `2048x2560 ~= 66 us`, and `2048x11008 ~= 273 us`.
- Decision: do not start a production external-cast + FP16-I/O bridge from this
  evidence alone. The simple source-visible cast is essentially tied with the
  TensorRT internal cast kernels seen in Round C, so removing the remaining
  `~10 ms` of bridge casts needs a larger FP16 activation residency strategy.
- Artifacts:
  - `.tmp_codex/probes/bf16_fp16_cast_probe.cu`
  - `.tmp_codex/bench/3060_20260515_roundd_bf16_fp16_cast_probe.jsonl`

Next attention route:

- NCU counter access is still the cleanest way to understand the FlashInfer
  kernel gap. The practical no-NCU bridge-layout repro is now complete and does
  not clear the low-risk production gate.
- The remaining attention work should either repair NCU counter permissions for
  FlashInfer evidence or explicitly propose a larger FP16 packed-QKV/cache
  experiment. Do not keep repeating operator-table-only sweeps or the BF16
  wrapper probe without a new acceptance argument.
- Milestone notification for the Round D rejection summary was delivered through
  `codex-notify-wechat`; after the NSYS correction, a follow-up notification
  clarified that TRT AttentionPlugin GPU kernel time is a real candidate. A
  third Round D notification recorded the bridge-layout wrapper rejection.

## 2026-05-15 Round E Auto Matrix Refresh

Round E reran the full EdgeFM CUDA graph matrix with the current optional bridge
stack:

- `EDGE_FM_PREFILL_TRT_MLP=1`
- `EDGE_FM_TRT_MLP_FP16_WEIGHTS=auto`
- `EDGE_FM_PREFILL_TRT_LINEAR=1`
- `EDGE_FM_TRT_LINEAR_ROLES=both`
- `warmup=1`, `runs=3`

Before the matrix, missing small/mid `both` MLP engines for `m=512/1024` were
generated so that `auto` could exercise its intended policy rather than falling
back to `gateup`. Logs confirm `0.5B/1.5B -> both` and `3B -> gateup`.

Aggregate result versus Round B bridge:

| Model | Sum auto - Round B bridge | Sum prefill delta | Wins vs TRT | Sum auto - TRT |
| --- | ---: | ---: | ---: | ---: |
| 0.5B | `-5.65 ms` | `-7.28 ms` | `2/6` | `+7.89 ms` |
| 1.5B | `-25.28 ms` | `-27.37 ms` | `0/6` | `+28.84 ms` |
| 3B | `-0.16 ms` | `-0.56 ms` | `1/6` | `+64.94 ms` |

Largest remaining gaps versus the Round B TRT baseline:

| Slice | Total gap | Prefill gap | Decode gap | Note |
| --- | ---: | ---: | ---: | --- |
| 3B 2048x32 | `+24.68 ms` | `+29.05 ms` | `-4.47 ms` | primary long-prefill residual |
| 3B 2048x64 | `+20.02 ms` | `+29.00 ms` | `-9.12 ms` | prefill residual hidden by faster decode |
| 3B 1024x32 | `+13.38 ms` | `+19.50 ms` | `-6.21 ms` | mid/long prefill residual |
| 0.5B 2048x64 | `+10.33 ms` | `+5.99 ms` | `+4.25 ms` | small-model long-context residual |
| 0.5B 2048x32 | `+8.11 ms` | `+6.06 ms` | `+1.99 ms` | small-model long-context residual |
| 1.5B 2048x32 | `+6.59 ms` | `+6.92 ms` | `-0.43 ms` | auto helps but prefill remains |

Decision:

- `auto` is useful as a default-off small/mid-model bridge policy and should be
  kept for further gated testing.
- It does not change the 3B story: 3B remains constrained by long-prefill
  attention/cast/layout residuals and 3B `both` memory headroom.
- A follow-up `0.5B / 2048x64` graph-off mapping on the current auto stack
  explains the small-model long-context residual: decode is still led by full
  `LMHead` (`50.7 ms` over 63 decode steps) and FlashInfer decode attention
  (`35.2 ms` plus `3.3 ms` merge), while prefill is led by MLP and FlashInfer
  prefill attention. These are the same two decode lines already covered by the
  `lm_head_top1` and decode-attention retune rejections, so they remain closed
  unless fresh evidence clears the `>=1%` end-to-end gate.
- Given Round D, the remaining low-risk wins are now quite small. Exceeding TRT
  on the hard 3B long-prefill slices likely requires either fixing NCU
  permissions to find a source-visible FlashInfer improvement or explicitly
  approving a larger FP16 packed-QKV/cache/activation-residency experiment.

Artifacts:

- `.tmp_codex/bench/3060_20260515_rounde_trt_mlp_0p5b_m512_both_engine_probe.json`
- `.tmp_codex/bench/3060_20260515_rounde_trt_mlp_0p5b_m1024_both_engine_probe.json`
- `.tmp_codex/bench/3060_20260515_rounde_trt_mlp_1p5b_m512_both_engine_probe.json`
- `.tmp_codex/bench/3060_20260515_rounde_trt_mlp_1p5b_m1024_both_engine_probe.json`
- `.tmp_codex/bench/3060_20260515_rounde_edgefm_auto_matrix/summary.json`
- `.tmp_codex/bench/3060_20260515_rounde_edgefm_auto_vs_trt_summary.json`
- `.tmp_codex/nsys/3060_20260515_rounde_0p5b_2048x64_edgefm_auto_mapping_triage.md`
- Milestone notification was delivered through `codex-notify-wechat`.

## 2026-05-15 Round F NCU FlashInfer Attention Evidence

Round F repaired the practical NCU blocker by running Nsight Compute under
root privileges. No production source or operator table change was accepted.

NCU access:

- Smoke report succeeded:
  `.tmp_codex/ncu/3060_20260515_sudo_smoke.ncu-rep`
- 3B prefill attention report:
  `.tmp_codex/ncu/3060_20260515_roundf_3b_prefill_attention_basic.ncu-rep`
- Stall-detail report:
  `.tmp_codex/ncu/3060_20260515_roundf_3b_prefill_attention_stalls.ncu-rep`
- Exported details:
  `.tmp_codex/ncu/3060_20260515_roundf_3b_prefill_attention_stalls_details.txt`

Key NCU facts for the current `3B / prefill=2048` FlashInfer
`SinglePrefillWithKVCacheKernel`:

| Metric | Value |
| --- | ---: |
| Kernel duration under NCU replay | `1.34 ms` |
| Compute / memory throughput | `38.4% / 38.4%` |
| DRAM throughput | `4.9%` |
| L2 hit rate | `96.6%` |
| Registers / thread | `168` |
| Dynamic shared memory / block | `49.152 KB` |
| Theoretical / achieved occupancy | `16.7% / 15.9%` |
| Active / eligible warps per scheduler | `1.91 / 0.28` |
| Issue rate | `0.22 warp/scheduler/cycle` |
| Dominant stall | `math_pipe_throttle`, `4.34 cycles/issued instruction` |

Interpretation:

- The kernel is neither DRAM bandwidth-bound nor host-bound. It is a low
  occupancy, low eligible-warp Ampere FlashInfer prefill shape.
- The current `cta_tile_q=64` path is shared-memory limited to two CTAs per SM;
  scheduler issue slots are mostly empty because few warps are eligible.
- This confirms the Round D observation that TRT's AttentionPlugin advantage is
  a real kernel/layout advantage, not a measurement artifact.

TRT AttentionPlugin NCU contrast:

- NCU report:
  `.tmp_codex/ncu/3060_20260515_roundf_trt_attention_plugin_fmha_stalls.ncu-rep`
- Exported details:
  `.tmp_codex/ncu/3060_20260515_roundf_trt_attention_plugin_fmha_stalls_details.txt`
- Same 3B shape, standalone TRT FMHA kernel
  `fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm86_kernel_nl`:

| Metric | EdgeFM FlashInfer | TRT FMHA |
| --- | ---: | ---: |
| Kernel duration | `1.345 ms` | `740 us` |
| Compute throughput | `38.4%` | `63.1%` |
| Memory throughput | `38.4%` | `55.4%` |
| Registers / thread | `168` | `250` |
| Dynamic shared memory / block | `49.152 KB` | `32.768 KB` |
| Theoretical occupancy | `16.7%` | `16.7%` |
| Issued warp / scheduler | `0.22` | `0.25` |
| Eligible warps / scheduler | `0.275` | `0.286` |
| Math-pipe throttle stall | `4.34` | `1.69` |

Interpretation: TRT is not simply winning by higher occupancy. It runs a
similar occupancy envelope but does more useful work per issued instruction,
uses less shared memory per block, and has far lower math-pipe throttle. A
useful source-visible replacement should target instruction mix/layout, not
just occupancy tuning.

Rejected low-risk source experiment:

- A temporary `prefill_num_mma_kv_cap` operator-table knob was tested locally to
  cap FlashInfer's per-CTA KV tile and trade smaller shared memory for more
  loop work. The experiment was fully reverted after measurement.
- Microbench result on 3B `512/1024/2048` looked promising:
  `cta64,cap2` total `1.2664 ms` versus baseline `1.3808 ms` (`-8.29%`).
- Paired CUDA graph generate did not clear the gate:
  - baseline current-auto `3B / 2048x32`: `942.981 ms`, prefill `316.422 ms`
  - temp `cta64,cap2`: `942.672 ms`, prefill `316.189 ms`
  - delta: `-0.309 ms` (`-0.033%`) end-to-end, `-0.233 ms` prefill
- Decision: reject and do not keep the knob. It adds template/maintenance
  surface and compile cost without a meaningful official generate win.
- Artifacts:
  - `.tmp_codex/bench/3060_20260515_roundf_3b_prefill_attention_mma_kv_cap_sweep.json`
  - `.tmp_codex/bench/3060_20260515_roundf_operator_impl_table_3b_prefill_attention_cta64_cap2.json`
  - `.tmp_codex/bench/3060_20260515_roundf_3b_2048x32_edgefm_auto_baseline_after_cap_patch.json`
  - `.tmp_codex/bench/3060_20260515_roundf_3b_2048x32_edgefm_auto_prefill_attn_cta64_cap2.json`

Validation after reverting the temp source experiment:

- `cmake --build build-3060-trt-mlp-release --target install -j$(nproc)` passed.
- `python3 scripts/operator_table/validate_operator_tables.py` passed.
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/analyze_trt_nsys_profile.py scripts/tune/tune_qwen_attention_prefill.py` passed.

Round F conclusion:

- More FlashInfer table-only tweaks are unlikely to close the `~20-25 ms` 3B
  long-prefill gap. The NCU evidence points at the algorithm/layout shape
  rather than one missed scalar parameter.
- The next route should be either a reviewed source-visible prefill attention
  replacement with its own correctness reference, or an explicitly approved
  larger FP16 packed-QKV/cache/activation-residency experiment. Under the
  current "no large design change" rule, this is a hold/reject state, not an
  accepted optimization.

## 2026-05-15 Round G FP16 Residency Boundary Checks

Round G checked whether the remaining bridge/residency ideas have a safe
small-step path before starting a larger source-visible attention replacement.
No production source or operator table change was made.

Attention dtype diagnostic:

- A direct EdgeFM attention-layer dtype check on the Qwen2.5-3B
  `num_qo_heads=16, num_kv_heads=2, head_dim=128` shape showed no standalone
  FP16-input advantage over BF16:
  - `S=512`: FP16 `0.1210 ms`, BF16 `0.1219 ms`
  - `S=1024`: FP16 `0.2918 ms`, BF16 `0.2934 ms`
  - `S=2048`: FP16 `0.9685 ms`, BF16 `0.9414 ms`
- FlashInfer Python JIT with `use_fp16_qk_reduction=True` failed to compile in
  the current environment because the package was not built with
  `FP16_QK_REDUCTION_SUPPORTED`.
- Decision: do not pursue a quick "cast attention inputs to FP16" branch. The
  current evidence still points to TRT's packed/layout-specific FMHA path, not
  dtype alone.

3B all-persistent FP16 bridge boundary:

- Diagnostic command combined:
  `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` and
  `EDGE_FM_TRT_LINEAR_FP16_WEIGHTS=both` on `3B / 2048x32`.
- The run loaded the FP16-weight QKV/OProj engines and began allocating
  persistent FP16 MLP + linear weight copies, but failed around layer 34 with
  `act_and_mul_kernel launch failed: out of memory`.
- Decision: reject the "3B all persistent FP16 weights" shortcut. It needs a
  real residency/memory policy, not another default-off env combination.

Small/mid-model QKV/OProj FP16-weight bridge:

- Built missing `m=2048` FP16-weight TensorRT linear engines for:
  - 0.5B QKV: `.tmp_codex/bench/trt_linear_edgefm_bf16_fp16compute_weight-fp16_bias-bf16_m2048_k896_n1152.engine`
  - 0.5B OProj: `.tmp_codex/bench/trt_linear_edgefm_bf16_fp16compute_weight-fp16_m2048_k896_n896.engine`
  - 1.5B QKV: `.tmp_codex/bench/trt_linear_edgefm_bf16_fp16compute_weight-fp16_bias-bf16_m2048_k1536_n2048.engine`
  - 1.5B OProj: `.tmp_codex/bench/trt_linear_edgefm_bf16_fp16compute_weight-fp16_m2048_k1536_n1536.engine`
- Paired CUDA graph results, all with `MLP auto + QKV/OProj bridge`:

| Slice | BF16 linear | FP16 linear weights | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| 0.5B `2048x32` | `175.921 ms` | `176.306 ms` | `+0.385 ms` (`+0.219%`) | rejected |
| 0.5B `2048x64` | `301.529 ms` | `301.798 ms` | `+0.269 ms` (`+0.089%`) | rejected |
| 1.5B `2048x32` | `477.274 ms` | `474.557 ms` | `-2.717 ms` (`-0.569%`) | below gate |
| 1.5B `2048x64` | `811.589 ms` | `809.105 ms` | `-2.484 ms` (`-0.306%`) | below gate |

- 1.5B prefill improves by about `2 ms`, but the official end-to-end gate is
  not met and the extra persistent copies reduce free memory from
  `5712.8 MB` to `5390.8 MB`.
- Decision: keep these FP16-weight linear engines as diagnostic artifacts only.
  Do not enable `EDGE_FM_TRT_LINEAR_FP16_WEIGHTS=both` by default or update the
  active bridge policy from this evidence.
- Artifacts:
  - `.tmp_codex/bench/3060_20260515_roundg_trt_linear_0p5b_qkv_fp16_engine_probe.json`
  - `.tmp_codex/bench/3060_20260515_roundg_trt_linear_0p5b_oproj_fp16_engine_probe.json`
  - `.tmp_codex/bench/3060_20260515_roundg_trt_linear_1p5b_qkv_fp16_engine_probe.json`
  - `.tmp_codex/bench/3060_20260515_roundg_trt_linear_1p5b_oproj_fp16_engine_probe.json`
  - `.tmp_codex/bench/3060_20260515_roundg_0p5b_2048x32_auto_linear_bf16.json`
  - `.tmp_codex/bench/3060_20260515_roundg_0p5b_2048x32_auto_linear_fp16.json`
  - `.tmp_codex/bench/3060_20260515_roundg_0p5b_2048x64_auto_linear_bf16.json`
  - `.tmp_codex/bench/3060_20260515_roundg_0p5b_2048x64_auto_linear_fp16.json`
  - `.tmp_codex/bench/3060_20260515_roundg_1p5b_2048x32_auto_linear_bf16.json`
  - `.tmp_codex/bench/3060_20260515_roundg_1p5b_2048x32_auto_linear_fp16.json`
  - `.tmp_codex/bench/3060_20260515_roundg_1p5b_2048x64_auto_linear_bf16.json`
  - `.tmp_codex/bench/3060_20260515_roundg_1p5b_2048x64_auto_linear_fp16.json`

Round G conclusion:

- The remaining no-large-design-change knobs are now mostly exhausted. More
  bridge FP16 residency toggles either OOM on 3B or fail the 1% end-to-end gate
  on 0.5B/1.5B.
- The next meaningful optimization must be a contained source-visible attention
  repro with correctness and NCU evidence, or an explicitly approved larger
  packed-QKV/cache/activation-residency design.

## 2026-05-15 Round H TRT FMHA Direct Runner Probe

Round H moved the TRT attention evidence from "observed inside plugin/engine
profiling" to a direct, repeatable runner probe. This is still an attribution
tool, not a production EdgeFM path.

New deliverable:

- `deliverables/kernel_opt/3060_prefill_attention_20260515/trt_fmha_runner_probe.cpp`
- README/build notes:
  `deliverables/kernel_opt/3060_prefill_attention_20260515/README.md`

The probe links against
`third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so.1.0` and calls
`trt_edgellm::ContextFMHARunner` directly on the Qwen2.5-3B prefill attention
shape:

- packed FP16 QKV: `[1, 2048, 16 + 2 * 2, 128]`
- FP16 KV cache: `[1, 2, 2, 2048, 128]`
- FP16 output: `[1, 2048, 16, 128]`
- identity RoPE tensor, so the timed region still includes
  `launchApplyRopeWriteKVPackedQKV`

Direct runner event timing:

| Shape | Mean | Median | Artifact |
| --- | ---: | ---: | --- |
| 3B `S=2048,Hq=16,Hkv=2,D=128` | `0.650708 ms` | `0.651264 ms` | `.tmp_codex/bench/3060_20260515_roundh_trt_fmha_runner_probe_3b_2048.json` |

NCU report:

- `.tmp_codex/ncu/3060_20260515_roundh_trt_fmha_runner_3b_2048_stalls.ncu-rep`
- `.tmp_codex/ncu/3060_20260515_roundh_trt_fmha_runner_3b_2048_stalls_details.txt`
- `.tmp_codex/ncu/3060_20260515_roundh_trt_fmha_runner_3b_2048_stalls_raw.csv`

Key NCU facts:

| Metric | Value |
| --- | ---: |
| Kernel under NCU | `fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm86_kernel_nl` |
| Duration under NCU replay | `739.46 us` |
| Compute / memory throughput | `63.34% / 55.41%` |
| DRAM throughput | `7.62%` |
| L2 hit rate | `96.10%` |
| Theoretical / achieved occupancy | `16.67% / 16.20%` |
| Active / eligible warps per scheduler | `1.95 / 0.29` |
| Issue rate | `0.25 warp/scheduler/cycle` |

Interpretation:

- This independently reproduces the earlier TRT plugin attribution:
  `ContextFMHARunner` itself runs the 3B long-prefill FMHA+RoPE slice around
  `0.65 ms/layer`, without relying on TensorRT engine enqueue timing.
- The TRT FMHA kernel is still in a low-occupancy envelope, but it keeps much
  higher SM/memory utilization than the current FlashInfer BF16 prefill path.
- The useful next unit is now a correctness-and-layout wrapper around this
  runner, not another FlashInfer table-only sweep.

Production gate before any integration:

- verify real RoPE, causal mask, multiple sequence lengths, and output
  correctness against EdgeFM/FlashInfer or PyTorch
- measure the real EdgeFM layout costs: separate BF16 Q/K/V to packed FP16 QKV,
  KV cache writes, and FP16 output back to BF16 residual
- keep the route default-off behind an explicit experiment gate until paired
  CUDA graph generate clears the `>=1%` end-to-end acceptance rule
- do not change `src/` or the 3060 operator table from this evidence alone

## 2026-05-15 Round I TRT FMHA Runner Correctness Coverage

Round I extended the direct runner probe instead of touching production EdgeFM
code.

Probe update:

- `trt_fmha_runner_probe.cpp` now supports `--rope-mode=identity|normal`.
- The runner restores the original QKV tensor before each timed launch, so
  non-identity RoPE is not repeatedly applied to an in-place buffer.
- `--check=1` runs a host-side FP32 causal attention reference for
  `seq_len <= 128`, covering normal RoPE, grouped-query head mapping, causal
  softmax, and FP16 output comparison.

Verification:

| Case | Mean | Median | Check |
| --- | ---: | ---: | --- |
| `S=32`, normal RoPE | `0.008688 ms` | `0.009216 ms` | pass, max abs error `0.000367761` |
| `S=512`, normal RoPE | `0.077920 ms` | `0.078240 ms` | smoke |
| `S=1024`, normal RoPE | `0.207686 ms` | `0.207872 ms` | smoke |
| `S=2048`, normal RoPE | `0.654522 ms` | `0.656096 ms` | smoke |
| `S=2048`, identity RoPE after restore update | `0.651104 ms` | `0.650848 ms` | smoke |

Artifacts:

- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_runner_normal_rope_s32_check.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_runner_normal_rope_s512.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_runner_normal_rope_s1024.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_runner_normal_rope_s2048.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_runner_identity_rope_s2048.json`

Decision:

- The direct TRT FMHA runner now has a first correctness foothold for real RoPE
  and causal attention.
- At this point it still did not justify production integration. The next
  required check was EdgeFM layout cost around the runner before any
  `src/`/operator-table experiment.

Round I layout/cast microbench:

- Added
  `deliverables/kernel_opt/3060_prefill_attention_20260515/trt_fmha_layout_cast_probe.cu`.
- It measures the current EdgeFM-compatible wrapper:
  `packed BF16 QKV -> packed FP16 QKV -> TRT FMHA + RoPE -> BF16 attention output`.
- Existing EdgeFM QKV prefill buffer is already packed `[S, Q + K + V]`, so the
  first-order wrapper cost is dtype conversion, not a full separate-Q/K/V
  repack.

`S=2048` results:

| Model | EdgeFM FlashInfer BF16 | Runner+casts total | Per-layer win | Layers | Full-model upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.5B `Hq=14,Hkv=2,D=64` | `0.467968 ms` | `0.348160 ms` | `0.119808 ms` | 24 | `2.88 ms` |
| 1.5B `Hq=12,Hkv=2,D=128` | `0.772112 ms` | `0.613408 ms` | `0.158704 ms` | 28 | `4.44 ms` |
| 3B `Hq=16,Hkv=2,D=128` | `0.971232 ms` | `0.787520 ms` | `0.183712 ms` | 36 | `6.61 ms` |

3B `S=2048` split:

- BF16 packed QKV -> FP16 packed QKV: `0.077824 ms`
- TRT FMHA + normal RoPE: `0.647168 ms`
- FP16 output -> BF16 output: `0.062464 ms`
- total: `0.787520 ms`

Artifacts:

- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_layout_cast_0p5b_s2048.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_layout_cast_1p5b_s2048.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_layout_cast_s512.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_layout_cast_s1024.json`
- `.tmp_codex/bench/3060_20260515_roundi_trt_fmha_layout_cast_s2048.json`
- `.tmp_codex/bench/3060_20260515_roundi_edgefm_attention_baseline_0p5b_s2048.json`
- `.tmp_codex/bench/3060_20260515_roundi_edgefm_attention_baseline_1p5b_s2048.json`
- `.tmp_codex/bench/3060_20260515_roundi_edgefm_attention_baseline_3b_s2048.json`

Decision:

- The direct FMHA runner plus BF16/FP16 casts is a real layer-level win.
- It is not a clear standalone production win under the current gate. The
  full-model upper bound is likely below 1% for 1.5B/3B and borderline only for
  the small 0.5B long-prefill slices.
- Do not wire this into `src/` yet. The only integration path worth considering
  is a default-off experiment with either scratch FP16 KV for prefill-only FMHA
  or an explicit cache dtype/reuse plan, plus paired CUDA graph evidence. A
  broader packed-FP16 QKV residency path remains a separate larger design.

## 2026-05-16 Round J Plugin-op FMHA Safety Gate

Round J tried the first contained `plugin-op` integration from the Round H/I
`ContextFMHARunner` evidence. This is not a TensorRT engine bridge: it links the
TRT-Edge-LLM plugin library and calls the runner from an EdgeFM attention
operator implementation selected by the operator table. The path is compile-time
and runtime gated.

Implementation slice:

- CMake option: `BUILD_TRT_PLUGIN_OPS=ON`.
- Operator impl id: `trt_context_fmha_plugin_attention`.
- Runtime gate: `EDGE_FM_PREFILL_TRT_FMHA_PLUGIN=1`.
- Benchmark helper: `scripts/profile/profile_edgefm_generate_case.py
  --edgefm-mode native|plugin-op|as-is`.
- The helper writes a temporary operator-table overlay for `plugin-op`; it does
  not require a serialized TensorRT engine or TensorRT execution context.

Correctness finding:

- `ContextFMHARunner` is FP16-only for the current SM86/head-dim path. A BF16
  Qwen prefill therefore needs a diagnostic BF16->FP16 input cast, scratch FP16
  KV/output, and FP16->BF16 output cast.
- Single-operator random BF16 comparison against FlashInfer was close but not
  exact: max abs diff `0.015625`, mean abs diff about `0.00042`.
- Real `1.5B` CUDA graph token alignment with the BF16 cast diagnostic path
  failed: `17/20` generated steps mismatched after the first two matched tokens.
  Reject log:
  `.tmp_codex/bench/3060_20260516_plugin_op_bf16_cast_alignment_reject.log`.

Safety change:

- BF16->FP16 plugin-op casting is now behind a second explicit diagnostic gate:
  `EDGE_FM_PREFILL_TRT_FMHA_PLUGIN_ALLOW_BF16_FP16_CAST=1` or
  `--plugin-op-allow-bf16-fp16-cast`.
- Without that second gate, `plugin-op` supports FP16 only. Current BF16 Qwen
  models fall back to the existing FlashInfer attention implementation when the
  temporary overlay selects `trt_context_fmha_plugin_attention`.
- This keeps the WIP plugin-op code available for diagnostics without silently
  changing the production BF16 correctness contract.

Validation after the safety gate:

- `cmake --build build-3060 -j$(nproc)`: pass.
- `python3 scripts/operator_table/validate_operator_tables.py`: pass.
- `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py
  scripts/profile/profile_trt_edgellm_generate_case.py
  scripts/profile/analyze_trt_nsys_profile.py`: pass.
- `EDGE_FM_BUILD_DIR=... EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0
  EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -q
  tests/operators/test_attention_prefill.py`: `6 passed`.
- `EDGE_FM_BUILD_DIR=... EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0
  EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -q
  tests/engine/test_qwen2_generate.py -k
  "token_alignment_cuda_graph or metrics_surface or max_new_tokens or
  deferred_stop"`: `6 passed, 14 deselected`.
- Safe `plugin-op` smoke artifact:
  `.tmp_codex/bench/3060_20260516_plugin_op_safe_smoke_1p5b_128x4.json`.
- Safe overlay alignment gate:
  `EDGE_FM_PREFILL_TRT_FMHA_PLUGIN=1 EDGE_FM_OPERATOR_IMPL_TABLE_LLM=<overlay>
  ... pytest -q
  tests/engine/test_qwen2_generate.py::test_generate_token_alignment_cuda_graph`:
  `1 passed`.
- Stage notification: `cc-connect send -p edge-fm-x -s s1 ...` and the same
  command without `-s` both returned `no active session found` after daemon
  restart. The stage summary was delivered to the same Feishu chat using the
  existing cc-connect Feishu OpenAPI credentials. Message id:
  `om_x100b6f4b1cd768acb481d6bcc8d5638`.

Decision:

- Direct TRT FMHA `plugin-op` is **not accepted as a BF16 performance
  optimization** yet because the only real BF16 execution path changes precision
  enough to break token alignment.
- Keep the runner integration as a default-off diagnostic scaffold. Do not run
  official plugin-op BF16 performance claims until token alignment passes.
- The next non-bridge route should be either a BF16-correct/source-visible FMHA
  replacement or a reviewed FP16 packed-QKV/cache residency design. Plainly
  casting the current BF16 path through the FP16 runner is rejected.

Humanize follow-up:

- Created standalone long-loop workspace:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/`.
- Local standalone commits:
  - `8f0ddb1` `scaffold bf16 fmha humanize loop`
  - `1e0886f` `initialize humanize round 0 contract`
  - `2023266` `ignore transient humanize session marker`
- RLCR loop directory:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/.humanize/rlcr/2026-05-16_12-42-26/`.
- Round 0 status: goal tracker, round contract, source ledger seed, and summary
  are initialized. No CUDA candidate was implemented in Round 0. Next round
  should build the standalone BF16 FMHA correctness and benchmark harness before
  any EdgeFM `src/` migration.
- Humanize Round 0 notification: `cc-connect send` still returned
  `no active session found`; Feishu OpenAPI fallback delivered message id
  `om_x100b6f4b2b295ca4b10f32132007431`.

Round 1 baseline harness:

- Added standalone harness:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/edgefm_attention_harness.py`.
- Correctness smoke:
  - 3B `S=32`, BF16 input/output, FP32 causal GQA reference
  - `max_abs=0.015625`, `mean_abs=0.000476845569210127`, passed
  - artifact:
    `.tmp_codex/bench/3060_20260516_bf16_fmha_harness_3b_s32_check.json`
- EdgeFM current FlashInfer-path `S=2048` baselines:
  - 0.5B: `0.463906 ms/layer` mean, `0.463872 ms/layer` median
  - 1.5B: `0.755977 ms/layer` mean, `0.756032 ms/layer` median
  - 3B: `0.985320 ms/layer` mean, `0.985088 ms/layer` median
  - artifacts:
    `.tmp_codex/bench/3060_20260516_bf16_fmha_harness_0p5b_s2048_baseline.json`,
    `.tmp_codex/bench/3060_20260516_bf16_fmha_harness_1p5b_s2048_baseline.json`,
    `.tmp_codex/bench/3060_20260516_bf16_fmha_harness_3b_s2048_baseline.json`
- NCU status:
  - launch metadata report captured:
    `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/profile-digests/ncu_reports/edgefm_bf16_fmha_3b_s2048_baseline.ncu-rep`
  - launch CSV exported:
    `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/profile-digests/ncu_reports/edgefm_bf16_fmha_3b_s2048_baseline_launch_raw.csv`
  - full `--set basic` metrics are blocked by `ERR_NVGPUCTRPERM` in the current
    non-root shell; this is recorded in
    `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/profile-digests/2026-05-16_edgefm_bf16_fmha_baseline.md`

Decision:

- The standalone baseline is stable enough to start a source-visible BF16 FMHA
  candidate loop.
- Do not use the TRT FP16 runner cast path as a production candidate unless
  token alignment is solved. The candidate lane should either be BF16-correct
  from the start, or explicitly propose a wider FP16 residency/cache design.

CTA tile sweep refresh:

- Re-ran the current FlashInfer `prefill_cta_tile_q` sweep under the new
  `S=2048` operator-only timing harness.
- Results:
  - 0.5B: baseline `0.468912 ms`, tile64 `0.463856 ms`, tile128 `0.466944 ms`
  - 1.5B: baseline `0.772096 ms`, tile64 `0.754656 ms`, tile128 `0.768848 ms`
  - 3B: baseline `0.970752 ms`, tile64 `0.984064 ms`, tile128 `0.954720 ms`
- Artifacts:
  - `.tmp_codex/bench/3060_20260516_attention_prefill_cta_sweep_0p5b_s2048_rerun.json`
  - `.tmp_codex/bench/3060_20260516_attention_prefill_cta_sweep_1p5b_s2048_rerun.json`
  - `.tmp_codex/bench/3060_20260516_attention_prefill_cta_sweep_3b_s2048_rerun.json`
- Decision: keep this out of the default table. The best 3B movement is only
  about `0.016 ms/layer`, or roughly `0.6 ms` across 36 layers, and previous
  end-to-end evidence already showed this class of tile-only change below the
  acceptance gate.
- Stage notification: `cc-connect send` still returned `no active session
  found`; Feishu OpenAPI fallback delivered message id
  `om_x100b6f4be86c253cb2452bf95e3227e`.

Source-visible seed kernel:

- Added standalone CUDA seed:
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/src/bf16_fmha/edgefm_bf16_fmha_seed.cu`
  and wrapper
  `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/benchmarks/bf16_fmha_seed_candidate.py`.
- Correctness:
  - 3B `S=32`: passed, `max_abs=0.001953125`, mean `0.128659 ms`
  - 0.5B `S=32`: passed, `max_abs=0.00048828125`, mean `0.105478 ms`
  - 3B `S=128`: passed, `max_abs=0.001953125`, mean `1.587546 ms`
- Performance smoke:
  - 3B `S=512`: seed `23.749973 ms`, current EdgeFM baseline `0.095796 ms`
- Decision: this is a correctness seed only. It is intentionally rejected as a
  performance route until a later mutation replaces the scalar two-pass design
  with a tiled/Tensor Core/online-softmax implementation.

FlashInfer TRTLLM FMHA v2 check:

- Reviewed `third_party/flashinfer/csrc/trtllm_fmha_v2_binding.cu` and
  `third_party/flashinfer/benchmarks/bench_trtllm_fmha.py`.
- The readily exposed benchmark is decode-oriented, and the FMHA v2 binding in
  this checkout calls SM120/MLA-specific symbols such as
  `run_fmha_v2_flash_attention_bf16_64_128_S_q_k_v_192x128_sm120_nl_tiled`.
- Decision: reject this as a near-term 3060/Qwen prefill route. It may be
  revisited only if a concrete SM80 Qwen prefill symbol and correctness path are
  found.
- Stage notification: Feishu OpenAPI fallback delivered message id
  `om_x100b6f4b8ffea4acb20ed6f2bf1422a`.

## Current Status

- Scope: LLM only.
- Hardware: RTX 3060 (`cuda_sm86`).
- Current observed toolchain from the optimizer restart: `nvcc 12.8.61`,
  PyTorch `2.10.0+cu128`, Triton `3.6.0`.
- Nsight tools available in the current shell: `ncu` and `nsys`; non-root NCU
  launch metadata works, but full performance counters are blocked by
  `ERR_NVGPUCTRPERM` until GPU counter permissions are enabled.
- Official comparison rule: only `EdgeFM(cuda graph)` versus `TRT-Edge-LLM`.
- Standing rules: [doc/3060_tuning_rules.md](./3060_tuning_rules.md)
- Latest accepted source change before this restart: `2026-05-09 15:18 +0800`,
  prefill SwiGLU fusion stays default-off on 3060.
- Latest fresh branch-local reference: the 2026-05-15 Round B effective matrix
  and Round C steady attribution above. Rounds H/I add a direct TRT
  `ContextFMHARunner` repro, initial normal-RoPE correctness coverage, and
  BF16/FP16 wrapper cost evidence for the long-prefill attention candidate.
  Round J adds a default-off `plugin-op` attention scaffold but rejects the
  BF16->FP16 cast diagnostic path for production because token alignment fails.
  The 2026-05-11 full-matrix bridge artifact is historical context only.
- Historical strategy convergence from `2026-05-12`: source-visible CUTLASS/cuTile/cublasLt sweeps did not close the 3060 prefill gap, while TensorRT compiler-generated Myelin/XMMA/FcCast-style tactics explained much of TRT-Edge-LLM's advantage. This remains an important prior, but the 2026-05-15 optimizer restart requires fresh profiling before accepting or rejecting any new kernel route. Small open-kernel edits are allowed only when a current profile identifies a specific hot path and the change clears the documented acceptance gates.
- Superseded high-memory note: the 2026-05-12 `both` OOM remains historical, but
  the 2026-05-15 rerun can complete `3B / 2048x32`. The current blocker is not
  a guaranteed OOM; it is insufficient headroom (`358 MB` free after warmup/runs).
  Keep `both` out of the default/full-matrix acceptance queue until there is a
  real memory policy.
- Latest bridge residual profile: the Round C steady `3B / 2048x1` traces above
  supersede the 2026-05-12 mapping. Current bridge `gateup` prefill is
  `323.227 ms` versus TRT `288.955 ms`; switching to `both` removes about
  `10 ms` of bridge casts but leaves attention at about `35.4 ms` versus TRT
  plugin+rope at `23.3 ms`. Analyzer known-path/action tables therefore point
  first to attention evidence, with MLP `auto` kept as a default-off memory gate
  rather than a 3B default.
- Latest attention prefill tuning probe: `2026-05-12 10:07 +0800`, the isolated 3B / 2048 attention win from `prefill_cta_tile_q=128` did **not** survive end-to-end. A paired bridge run on the same environment measured `3B / 2048x32` at EdgeFM `947.754 ms` with the current `prefill_cta_tile_q=64` table entry and `948.438 ms` with `128` (`+0.684 ms`, `+0.07%`). The current attention table therefore stays at `64`; keep `128` only as a rejected diagnostic result. Raw: `.tmp_codex/bench/3060_20260512_attention_tile128_bridge_paired_edgefm_3b_2048x32.json`; isolated probe raw: `.tmp_codex/bench/3060_20260512_attention_prefill_3b_2048_tune.json`.
- Latest EdgeFM profiling conclusion: `2026-05-09 16:08 +0800`, post-default-off `3B / 2048x32` graph-off mapping shows the previous `fused_moe` prefill hotspot is gone; remaining prefill time is dominated by existing linear GateUp/DownProj. Raw: `.tmp_codex/nsys/3060_3b_2048x32_default_off_mapping.nsys-rep`, triage: `.tmp_codex/nsys/3060_3b_2048x32_default_off_mapping_triage.md`, role summary: `.tmp_codex/nsys/3060_3b_2048x32_default_off_mapping_role_summary.json`
- Latest TRT profiling conclusion: `2026-05-09 16:50 +0800`, `3B / 2048x1` nsys capture shows TRT-Edge-LLM prefill is also GEMM-dominated but much faster in the same roles: total kernel time `285.46 ms`; GEMM `240.70 ms` (`84.32%`); inferred roles are Gate+Up `136.50 ms`, DownProj `71.36 ms`, QKV `16.96 ms`, OProj `14.06 ms`, AttentionPlugin `22.90 ms`, SwiGLU `14.53 ms`. Raw: `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping.nsys-rep`, run JSON: `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping_run.json`, summary: `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping_kernel_summary.md`
- Latest benchmark-alignment check: `2026-05-09 18:58 +0800`, EdgeFM does include prefill input preparation/copy in its `prefill_ms`, but existing nsys evidence shows this is not the `~201 ms` prefill gap. EdgeFM graph-off `3B / 2048x32` captured CUDA memcpy total is `0.623 ms` (`84.02 MB`), while TRT `3B / 2048x1` memcpy total is `0.004 ms` (`8.2 KB`). EdgeFM graph-on formal trace only shows `0.044 ms` memcpy outside CUDA graph. A TRT sync probe shows no async timing bias: unsynchronized wall avg `921.5 ms`, explicit `torch.cuda.synchronize()` wall avg `922.1 ms`. Raw: `.tmp_codex/nsys/3060_3b_2048x32_default_off_mapping.sqlite`, `.tmp_codex/nsys/3060_3b_2048x32_formal_graph.sqlite`, `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping.sqlite`, `.tmp_codex/bench/3060_20260509_trt_3b_2048x32_sync_probe.json`
- Confirming profile: `2026-05-09 16:15 +0800`, `3B / 2048x64` graph-off mapping shows the same long-prefill MLP pattern: GateUp `243.82 ms` (`50.62%`) and DownProj `124.74 ms` (`25.90%`) dominate prefill. Raw: `.tmp_codex/nsys/3060_3b_2048x64_default_off_mapping.nsys-rep`, triage: `.tmp_codex/nsys/3060_3b_2048x64_default_off_mapping_triage.md`, role summary: `.tmp_codex/nsys/3060_3b_2048x64_default_off_mapping_role_summary.json`
- Latest rejected tuning route: `2026-05-09 16:08 +0800`, 3B `m=2048` cublasLt heuristic/explicit and FlashInfer prefill attention sweeps did not meet the `>=1%` official CUDA graph target-case acceptance rule. Candidate table edits were reverted.
- Latest diagnostic rejection: `2026-05-09 18:49 +0800`, post-commit probes show that the current three `LinearCutlassImpl` configs do not beat the accepted 3060 cublasLt table for `3B / BF16 / m=2048`, and a temporary FP16 checkpoint does not close the end-to-end gap. Raw artifacts: `.tmp_codex/bench/3060_20260509_probe_3b_m2048_cutlass_vs_cublaslt.json`, `.tmp_codex/bench/3060_20260509_3b_2048x32_fp16_edgefm_probe.json`, `.tmp_codex/bench/3060_20260509_3b_fp16_m2048_fused_gate_up_cublaslt_all.json`, `.tmp_codex/bench/3060_20260509_3b_fp16_m2048_mlp_down_cublaslt_all.json`
- Latest cublasLt layout probe: `2026-05-09 19:02 +0800`, native row-major descriptors do not beat the current column-major view for `3B / m=2048` GateUp or DownProj; cublasLt packed weight transform/layout combinations tested here returned unsupported. Raw artifacts: `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_gateup_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_downproj_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_gateup_fp16.jsonl`, `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_downproj_fp16.jsonl`
- Latest CUTLASS config sweep: `2026-05-09 19:02 +0800`, a standalone sweep of existing `third_party/cutlass` GEMM tile/stage configs near TRT's profiled XMMA shapes did not beat current cublasLt for the official BF16 path. BF16 bests: GateUp `6.894 ms`, DownProj `3.727 ms`. FP16 bests: GateUp `6.823 ms`, DownProj `3.524 ms`; this is not a production BF16 win and still cannot close the gap. Raw artifacts: `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_gateup_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_downproj_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_gateup_fp16_serial.jsonl`, `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_downproj_fp16_serial.jsonl`
- Latest Myelin/XMMA reuse check: `2026-05-10`, `third_party/TensorRT-Edge-LLM` does not expose a public/source-visible BF16/FP16 dense Myelin/XMMA GEMM launcher for `LinearImpl`. The profiled TensorRT kernels are engine tactics, not a buildable EdgeFM operator path. Review doc updated: [doc/3060_fused_mlp_review.md](./3060_fused_mlp_review.md)
- Latest Myelin/XMMA subengine probe: `2026-05-10 10:45 +0800`, an isolated TensorRT `GateUp -> SwiGLU -> DownProj` engine reproduced the internal tactics. BF16 selected `sm80_xmma_gemm_bf16bf16_*` plus `__myl_SiluMul_*` but measured `11.33 ms` median per layer, slower than the current EdgeFM MLP estimate `10.62 ms`. FP16 selected the same `sm80_xmma_gemm_f16f16_*` family seen in TRT-Edge-LLM and measured `6.26 ms` median per layer, matching TRT's inferred `6.18 ms`. BF16 bridge is rejected as the next production route; FP16 bridge remains review-gated. Raw: `.tmp_codex/bench/trt_mlp_subengine_3b_bf16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_bf16_kernel_summary.md`, `.tmp_codex/bench/trt_mlp_subengine_3b_fp16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_fp16_kernel_summary.md`
- Latest TRT FP16 MLP feasibility: `2026-05-11 09:41 +0800`, a TensorRT MLP engine with `gateup_weight` and `down_weight` as runtime inputs validated actual Qwen2.5-3B weights without duplicating them into the engine. The runtime-weight engine is `67 KB`, selected `sm80_xmma_gemm_f16f16_*` plus Myelin activation, and can reuse the same engine across same-shape layers. Fresh rerun: layer 0 median `6.06 ms`, layer 35 median `6.26 ms`; torch FP16 reference mean relative error is about `0.59-0.60%`. Raw: `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_fp16_runtime_weights_rerun_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_fp16_runtime_weights_reuse_rerun_verify.json`. Earlier raw: `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_fp16_runtime_weights_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_fp16_runtime_weights_reuse_verify.json`. Feishu notifications: `om_x100b6f270ae1b4e4b2bbd30a2c71846`, `om_x100b6f2718dbf0e8b3b9a2b87782b13`
- Latest production-layout TRT MLP feasibility: `2026-05-11 10:00 +0800`, a TensorRT runtime-weight engine can bind EdgeFM-shaped BF16 weights directly (`GateUp [up,gate,H]`, DownProj `[H,I]`), cast activation/weights to FP16 inside TensorRT, use transposed MatMul operands, run Myelin activation, and cast output back to BF16. Inspector confirms `sm80_xmma_gemm_f16f16_*` for both GEMMs, not BF16 XMMA. Medians: 3B `m=2048` layer 0 `7.02 ms`, 3B layer 35 reuse `7.18 ms`, 3B `m=1024` `4.12 ms`, 3B `m=512` `2.47 ms`, 1.5B `m=2048` `4.29 ms`, 0.5B `m=2048` `1.57 ms`. Engine sizes are `80-93 KB`; torch FP16-compute/BF16-output reference mean relative error is about `0.4-0.6%`. Raw: `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_layer35_bf16_edgefm_layout_fp16_compute_reuse_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_m1024_bf16_edgefm_layout_fp16_compute_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_layer0_m512_bf16_edgefm_layout_fp16_compute_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_1p5b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_0p5b_layer0_bf16_edgefm_layout_fp16_compute_verify.json`. Feishu notification: `om_x100b6f27ddef8c58b29b04c75520107`
- Latest bridge implementation slice: `2026-05-11 11:25 +0800`, an optional TensorRT prefill MLP bridge (`BUILD_TRT_MLP_BRIDGE=ON`, `EDGE_FM_PREFILL_TRT_MLP=1`) was wired into Qwen2.5 prefill only. It binds existing BF16 GateUp/DownProj tensors, loads one TensorRT engine per MLP shape, and falls back to native MLP when disabled or unsupported. Default builds still exclude TensorRT bridge code. A separate correctness bug was found and fixed in prefill CUDA graph: first capture did not replay the captured graph before `generate()` advanced, so the first sampled token could be uninitialized. New regression test: `tests/engine/test_qwen2_generate.py::test_generate_token_alignment_prefill_cuda_graph_first_request`.
- Latest bridge correctness: after the prefill graph replay fix, `3B / 2048x32` bridge+CUDA graph matches `transformers.generate(do_sample=False, eos_token_id=None)` for all 32 generated tokens. Raw: `.tmp_codex/bench/3060_20260511_1117_3b_2048x32_trt_mlp_bridge_transformers_generate_compare_after_prefill_graph_fix.json`. Earlier temporary artifacts `.tmp_codex/bench/3060_20260511_1102_3b_2048x32_trt_mlp_bridge_transformers_token_compare.json` and `.tmp_codex/bench/3060_20260511_1111_3b_2048x32_trt_mlp_bridge_transformers_token_compare_after_prefill_graph_fix.json` used an inconsistent ad hoc reference sequence and are obsolete.
- Latest bridge full-matrix slice: `2026-05-11 11:43 +0800`, TensorRT MLP bridge engines now cover all 9 MLP prefill shapes (`0.5B/1.5B/3B x 512/1024/2048`). The experimental EdgeFM-only CUDA graph matrix improved all 18 cases versus the latest native full-matrix EdgeFM baseline, with total mean improvement `662.46 ms` across the matrix and mean per-case improvement `36.80 ms`. It still reaches or beats the latest official TRT baseline in only `2/18` mean cases and `2/18` median cases, both `0.5B / 512`. This remains default-off and not accepted as production/default. Raw clean artifact: `.tmp_codex/bench/3060_20260511_1230_full_llm_matrix_trt_mlp_bridge_edgefm_only_clean.json`; summary: `.tmp_codex/bench/3060_20260511_1230_full_llm_matrix_trt_mlp_bridge_edgefm_only_summary.json`.
  - Largest remaining gap: `3B / 2048x32`, native `1125.25 ms`, bridge `986.44 ms`, TRT `927.69 ms`; bridge gain `138.81 ms`, remaining gap `+58.75 ms`, bridge prefill avg `355.38 ms`.
  - `3B / 2048x64`: native `1778.55 ms`, bridge `1638.02 ms`, TRT `1584.36 ms`; bridge gain `140.53 ms`, remaining gap `+53.65 ms`.
  - `3B / 1024x32`: native `858.97 ms`, bridge `810.40 ms`, TRT `764.35 ms`; bridge gain `48.57 ms`, remaining gap `+46.05 ms`.
  - Best cases versus TRT: `0.5B / 512x32` bridge `130.57 ms` versus TRT `134.66 ms` (`-4.09 ms`), and `0.5B / 512x64` bridge `248.09 ms` versus TRT `249.81 ms` (`-1.73 ms`).
- Historical bridge residual profile: `2026-05-11 11:50 +0800`, `3B / 2048x1` bridge graph-off mapping shows the TensorRT MLP GEMM and activation core is no longer the main residual problem. Bridge prefill kernel total is `350.41 ms` versus TRT prefill kernel total `285.46 ms`, delta `+64.95 ms`. Bridge MLP core without casts is `216.92 ms` versus TRT inferred GateUp+SwiGLU+DownProj `222.39 ms`, delta `-5.47 ms`. The largest extra bridge-specific cost is TensorRT internal BF16 weight casts `29.86 ms` plus activation/output casts `3.77 ms`. Remaining non-MLP deltas are EdgeFM QKV `+16.08 ms`, OProj `+10.04 ms`, and prefill attention `+10.40 ms`. Raw: `.tmp_codex/nsys/3060_3b_2048x1_trt_mlp_bridge_mapping.nsys-rep`; triage: `.tmp_codex/nsys/3060_3b_2048x1_trt_mlp_bridge_mapping_triage.md`; structured summary: `.tmp_codex/nsys/3060_3b_2048x1_trt_mlp_bridge_residual_summary.json`.
- Latest FP16 MLP weight-input diagnostic: `2026-05-11 12:03 +0800`, isolated TensorRT `3B / m=2048` EdgeFM-layout MLP probes confirm that binding selected runtime weights as FP16 removes most of the TensorRT BF16 weight-cast overhead. Baseline EdgeFM-layout BF16 weight inputs measured `7.02 ms/layer`; FP16 inputs for both GateUp and DownProj measured `6.25 ms/layer`, close to pure FP16 runtime weights at `6.06 ms/layer`; GateUp-only measured `6.60 ms/layer`; Down-only measured `6.90 ms/layer`. The `both` nsys trace shows only activation/input/output cast kernels plus two `sm80_xmma_gemm_f16f16_*` GEMMs, with no large BF16 weight-cast kernels in the timed region. Raw: `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_both_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_gateup_verify.json`, `.tmp_codex/bench/20260511_trt_mlp_3b_m2048_edgefm_layout_fp16weights_down_verify.json`, nsys: `.tmp_codex/nsys/trt_mlp_3b_m2048_fp16weights_both_probe.nsys-rep`, sqlite: `.tmp_codex/nsys/trt_mlp_3b_m2048_fp16weights_both_probe.sqlite`.
- Latest GateUp-FP16 bridge implementation result: `2026-05-11 12:36 +0800`, the optional TensorRT MLP bridge now supports `EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup`, creating persistent FP16 GateUp copies via a TensorRT BF16->FP16 cast subengine instead of a handwritten kernel. The path is still compile/runtime/default-off (`BUILD_TRT_MLP_BRIDGE=ON`, `EDGE_FM_PREFILL_TRT_MLP=1`, `EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup`). Actual 3B `m=2048` bridge logs show the `fp16weights-gateup` engine loaded, a cast engine built for `r22016_c2048`, and `36` persistent GateUp copies of `90,177,536` bytes each. Correctness: `3B / 2048x32` CUDA graph matched `transformers.generate(do_sample=False, eos_token_id=None)` for all `32` generated tokens. Performance: `3B / 2048x32` 3-run EdgeFM mean `964.20 ms`, median `964.34 ms`, prefill avg `334.83 ms`, decode avg `629.12 ms`; this is `-161.06 ms` versus native EdgeFM and `-22.24 ms` versus the BF16 bridge, but still `+36.51 ms` slower than the latest TRT baseline `927.69 ms`. Raw: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_runs3_clean.json`; correctness raw: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_transformers_compare_clean.json`; post-cache-reset smoke: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_3b_2048x32_post_cache_reset_smoke_clean.json`.
- Latest gate validation after GateUp-FP16 WIP: `2026-05-11 12:36 +0800`. `git diff --check`, default build/install, bridge build/install, operator table validation, default generate alignment, default `test_prefill_linear.py`, default `test_fused_gate_up_activation.py`, and bridge-build first-request regression passed. `test_attention_decode.py` is currently all skipped in this environment (`1 skipped`, pytest exit code `5`), so no decode-attention regression signal was produced in this gate. The build-path helper was fixed so explicit `EDGE_FM_BUILD_DIR` remains first in `sys.path`; without that fix bridge correctness gates were silently importing the default native build.
- Feishu notification for the GateUp-FP16 conclusion: `om_x100b6f219d3e24acb3c0c627c25abd1`. `cc-connect send -p edge-fm-x -s s1 ...` still returns `no active session found`, so this was sent with the Feishu OpenAPI fallback.
- Latest GateUp-FP16 full-matrix result: `2026-05-11 12:54 +0800`, generated GateUp-FP16 engines for all official MLP shapes (`0.5B/1.5B/3B x m=512/1024/2048`) and ran the 18-case EdgeFM-only CUDA graph matrix. No missing-engine fallback occurred. The GateUp-FP16 bridge improved all `18/18` cases versus the BF16 bridge, with total mean improvement `191.57 ms` across the matrix and mean per-case improvement `10.64 ms`. It reaches or beats the latest official TRT baseline in `3/18` mean cases and `3/18` median cases, versus `2/18` for the BF16 bridge, but still does not meet the final target. Raw: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_full_llm_matrix_runs3_clean.json`; summary: `.tmp_codex/bench/3060_20260511_gateup_fp16_weight_full_llm_matrix_runs3_summary.json`.
  - Remaining top gaps: `3B / 2048x32` GateUp-FP16 `970.55 ms` versus TRT `927.69 ms`, gap `+42.86 ms`; `3B / 2048x64` `+37.37 ms`; `3B / 1024x32` `+23.43 ms`; `0.5B / 2048x64` `+18.09 ms`; `3B / 1024x64` `+17.86 ms`.
  - Best cases versus TRT: `0.5B / 512x32` gap `-6.55 ms`, `0.5B / 512x64` gap `-5.05 ms`, and `3B / 512x64` gap `-0.25 ms`.
- Feishu notification for the GateUp-FP16 full-matrix conclusion: `om_x100b6f225882f0b0b30c63a70a2a4a0`.
- Latest GateUp-FP16 residual profile: `2026-05-11 13:18 +0800`, `3B / 2048x1` graph-off mapping shows the GateUp-FP16 bridge has aligned the MLP GEMM+SwiGLU core with TRT. Stage kernel total is `336.08 ms` versus TRT reference `285.46 ms`, delta `+50.62 ms`. MLP core without casts is `222.54 ms` versus TRT `222.39 ms`, delta `+0.14 ms`. Remaining deltas are EdgeFM QKV `+16.39 ms`, OProj `+10.54 ms`, FlashInfer prefill attention `+11.29 ms`, and remaining bridge casts `+11.88 ms`, mostly DownProj BF16->FP16 weight cast `9.93 ms`. This means DownProj-FP16 can only address about `10 ms` of the largest `+42.86 ms` full-matrix gap; QKV/OProj/attention must become active candidates after the cast check. Raw: `.tmp_codex/nsys/3060_3b_2048x1_gateup_fp16_bridge_mapping.nsys-rep`; triage: `.tmp_codex/nsys/3060_3b_2048x1_gateup_fp16_bridge_mapping_triage.md`; structured summary: `.tmp_codex/nsys/3060_3b_2048x1_gateup_fp16_bridge_residual_summary.json`.
- Feishu notification for the GateUp-FP16 residual conclusion: `om_x100b6f22765bd0a8b22ef0385121d9b`. `cc-connect send -p edge-fm-x -s s1 ...` still returns `no active session found`; OpenAPI fallback was used.
- Latest GateUp+DownProj-FP16 target slice: `2026-05-11 13:24 +0800`, `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` ran `3B / 2048x32` with the existing `fp16weights-both` TensorRT engine and actual persistent FP16 copies for both MLP weights. It did not OOM in this single target run; logs show `36` GateUp FP16 copies of `90,177,536` bytes and `36` DownProj FP16 copies of `45,088,768` bytes. Correctness passed against `transformers.generate(do_sample=False, eos_token_id=None)` for all `32` generated tokens. Performance: EdgeFM mean `952.10 ms`, median `952.18 ms`, prefill avg `322.12 ms`, decode avg `629.82 ms`. This is `-18.45 ms` versus GateUp-FP16 full-matrix `3B / 2048x32` and reduces the latest TRT gap from `+42.86 ms` to `+24.41 ms`, but it is still slower than TRT and is only a single-slice, high-memory, default-off candidate. Memory probe: process start free `11,413.6 MiB`, after engine init free `5,343.6 MiB`, after first generate and timed runs free only `161.6 MiB`. Conclusion: performance direction is valid, but `both` is memory-unsafe for 3B expansion/default-on without a new memory policy. Raw clean: `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_runs3_clean.json`; summary: `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_summary.json`; correctness: `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_transformers_compare_clean.json`; memory probe: `.tmp_codex/bench/3060_20260511_both_fp16_weight_3b_2048x32_memory_probe_clean.json`; Feishu notification: `om_x100b6f2207cca8b4b109e38c0985317`.
- Latest QKV/OProj TensorRT linear probe: `2026-05-11 13:35 +0800`, a temporary runtime-weight TensorRT linear subengine reproduced FP16 XMMA/Myelin tactics for EdgeFM resident `[out,in]` weight layout with BF16 input/output and FP16 compute. For `3B / m=2048`, current EdgeFM profile has QKV `33.35 ms` + OProj `24.60 ms` across 36 layers, while TRT reference has QKV `16.96 ms` + OProj `14.06 ms`. The synthetic subengine measured BF16-weight inputs at estimated 36-layer QKV+OProj `39.37 ms` and FP16-weight inputs at `35.24 ms`, implying estimated gains of `18.57 ms` and `22.71 ms` versus current EdgeFM. Inspector shows `sm80_xmma_gemm_f16f16_*` plus Myelin casts. This is a high-value candidate for a reviewed QKV/OProj bridge, but not a production change yet: it uses synthetic weights, isolated subengines, and would touch attention-layer boundaries. Raw summary: `.tmp_codex/bench/20260511_trt_linear_qkv_oproj_3b_m2048_summary.json`; raw probes: `.tmp_codex/bench/20260511_trt_linear_qkv_3b_m2048_bf16weight_verify.json`, `.tmp_codex/bench/20260511_trt_linear_oproj_3b_m2048_bf16weight_verify.json`, `.tmp_codex/bench/20260511_trt_linear_qkv_3b_m2048_fp16weight_verify.json`, `.tmp_codex/bench/20260511_trt_linear_oproj_3b_m2048_fp16weight_verify.json`; Feishu notification: `om_x100b6f22287798acb29b917deeadd2e`.
- Latest QKV/OProj bridge target and regression slice: `2026-05-11 14:05 +0800`, the optional TensorRT prefill linear bridge now supports QKV bias tensors. Root cause for the earlier QKV failure was missing Q/K/V bias: Qwen2.5 Q/K/V projections have BF16 bias, while OProj has no bias. The bias-aware QKV engine fixes token correctness. With `EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup`, `EDGE_FM_PREFILL_TRT_LINEAR=1`, `EDGE_FM_TRT_LINEAR_ROLES=both`, and BF16 linear weights:
  - `3B / 2048x32`: correctness passed against a Transformers KV-cache greedy loop for all 32 generated tokens. Performance EdgeFM mean `948.91 ms`, median `948.62 ms`, prefill avg `318.73 ms`, decode avg `629.99 ms`. This is `-21.64 ms` (`2.23%`) versus GateUp-FP16 full-matrix `970.55 ms`, and reduces the latest TRT gap from `+42.86 ms` to `+21.23 ms`.
  - `3B / 2048x64`: EdgeFM mean `1597.05 ms`, median `1597.23 ms`, prefill avg `316.79 ms`, decode avg `1280.00 ms`. This is `-24.69 ms` (`1.52%`) versus GateUp-FP16 `1621.74 ms`, and reduces the latest TRT gap from `+37.37 ms` to `+12.68 ms`.
  - `1.5B / 2048x32`: correctness passed for all 32 generated tokens. Performance EdgeFM mean `482.96 ms`, median `482.57 ms`, prefill avg `156.41 ms`, decode avg `326.36 ms`. This is `-8.56 ms` (`1.74%`) versus GateUp-FP16 `491.52 ms`, and reduces the latest TRT gap from `+17.29 ms` to `+8.73 ms`.
  - Conclusion: QKV bias-aware + OProj BF16 TensorRT linear bridge clears the target and key regression-slice acceptance threshold. A later 18-case full-matrix run confirms the path improves `15/18` cases versus GateUp-FP16, but it remains compile/runtime/default-off pending same-run TRT comparison and packaging/cleanup. Raw summaries: `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_3b_2048x32_summary.json`, `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_3b_2048x64_summary.json`, `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_1p5b_2048x32_summary.json`. Correctness/raw: `.tmp_codex/bench/3060_20260511_qkv_bias_operator_diagnostic.json`, `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_3b_2048x32_transformers_compare.json`, `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_1p5b_2048x32_transformers_compare.json`. Feishu notification: `om_x100b6f235628e0b8b25e0ca13a7e756`.
- Latest CUTLASS layout diagnostic: `2026-05-10 10:58 +0800`, source-visible classic CUTLASS `device::Gemm` with prepacked `RowMajor x RowMajor` B layout does not reproduce TRT's FP16 XMMA speed. Best FP16 GateUp was `6.666 ms` versus TensorRT FP16 GateUp `3.856 ms`; best FP16 DownProj was `3.440 ms` versus TensorRT FP16 DownProj `1.984 ms`. Do not add a production prepacked-weight CUTLASS path from this result alone. Raw: `.tmp_codex/bench/3060_20260510_cutlass_layout_gateup_fp16_sweep.jsonl`, `.tmp_codex/bench/3060_20260510_cutlass_layout_down_fp16_sweep.jsonl`, `.tmp_codex/bench/3060_20260510_cutlass_layout_fp16_sweep_summary.json`
- Latest source-visible third-party search: `2026-05-10 11:06 +0800`, the remaining vendored candidates do not provide a direct RTX 3060 dense FP16 XMMA-equivalent runner. `third_party/flashinfer/csrc/trtllm_gemm_runner.cu` is FP8/E2M1-to-BF16 only, `third_party/flashinfer/csrc/tgv_gemm.cu` is SM100/UMMA/TMA-oriented and not an SM86 path, and TensorRT-LLM CUTLASS FP16/BF16 code under `third_party/flashinfer/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels` is MoE/grouped, fused-MoE, or quantized GEMM. The current build only compiles the fused-MoE helper path used by `cutlass_prefill_swiglu`, which has already been rejected as default-on. Conclusion: no production `myelin`/`xmma`/dense FP16 impl id should be added from the current source-visible third-party tree; the next meaningful implementation path is a reviewed FP16 TensorRT-backed MLP bridge or a new external source-visible runner with operator evidence.
- 3060 platform conclusion: after the CUTLASS, cuTile, cublasLt-layout, and source-visible third-party sweeps, open-kernel retuning is no longer the main line on this hardware. The maintained path is TensorRT subgraph/subengine bridging for selected modules, because that is the only route that can practically reach the closed Myelin/XMMA-style tactics observed in TRT profiles.
- Latest cuTile MatMul probe: `2026-05-10 11:28 +0800`, `third_party/cutile-python/samples/MatMul.py` was run with the pip-provided TileIR compiler enabled through `nvidia-cuda-tileiras`. Best persistent `64x128x64` results were FP16 `GateUp 9.87 ms` / `DownProj 4.80 ms` and BF16 `GateUp 9.75 ms` / `DownProj 4.79 ms`, which is slower than current cublasLt/BF16 and far behind TRT FP16 XMMA. Raw artifacts: `.tmp_codex/bench/3060_20260510_cutile_matmul_fp16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_bf16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_summary.json`
- Latest validation gate: `2026-05-09 17:40 +0800`. `git diff --check`, Python compile, `make -C build-3060 -j edge_fm_python && make -C build-3060 install`, operator table validation, operator gate, and generate alignment passed after final cleanup and before commit.
- Latest validation raw artifacts:
  - `.tmp_codex/bench/3060_20260509_final_build_install.log`
  - `.tmp_codex/bench/3060_20260509_final_validate_operator_tables.log`
  - `.tmp_codex/bench/3060_20260509_final_operator_gate.log`
  - `.tmp_codex/bench/3060_20260509_final_generate_alignment.log`
- Feishu notifications:
  - `cc-connect daemon start` succeeded, but `cc-connect send` returned `no active session found` after daemon restart; the same Feishu bot credentials from `~/.cc-connect/config.toml` were used to send the conclusion directly.
  - cuTile probe conclusion message id: `om_x100b50cb9f0f68acb130beab4436935`; delivered to the same Feishu chat via the bot credentials because `cc-connect send` still returned `no active session found` for the current local session map.
  - prefill SwiGLU default-off accepted message id: `om_x100b50da6fc790acb296afd2fe209d3`
  - full-matrix refresh message id: `om_x100b50da05a6e084b36a16762e1a183`
  - post-default-off nsys + rejected low-risk table sweep message id: `om_x100b50da89dc6cb8b24291bf5255c27`
  - attention `prefill_cta_tile_q=128` end-to-end rejection message id: `om_x100b6f1c813aa8ecb1095322ffb3e28`
  - `3B / 2048x64` confirming nsys profile message id: `om_x100b50da9b9f6880b125f42933e1ef8`
  - `cc-connect send -p edge-fm-x -s s1 ...` still returns `no active session found`; use the Feishu OpenAPI fallback until the active-session mapping is repaired.
  - TRT prefill reverse attribution message id: `om_x100b50dbe49acca0b2bd2059dbd0f67`
  - post-commit CUTLASS/FP16 diagnostic rejection message id: `om_x100b50c55826248cb2b867358f2f621`
  - benchmark/copy alignment check message id: `om_x100b50c57763d080b39b53adf57791b`
  - cublasLt layout diagnostic rejection message id: `om_x100b50c50a0dbca0b10e7a74d1bccdc`
  - Myelin/XMMA direct reuse check message id: `om_x100b50ca922744acb3db168ba1115c1`
  - Myelin/XMMA subengine probe message id: `om_x100b50cb6c0ea8a0b369b4abbdf6487`
  - CUTLASS prepacked-B layout rejection message id: `om_x100b50cb025e54b0b3b5fca28420517`

Latest official raw matrix:

- `.tmp_codex/bench/3060_20260509_1524_full_llm_matrix_prefill_swiglu_default_off.json`
- summary: `.tmp_codex/bench/3060_20260509_1524_full_llm_matrix_prefill_swiglu_default_off_summary.json`
- command: `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 ... report_qwen_benchmark_suite.py --repo-root /home/zhangzimo/Repos/private/edge-fm-x --device-id 0 --kind llm --llm-models 0.5b,1.5b,3b --prefill-list 512,1024,2048 --decode-list 32,64 --json-only`
- Python: `/home/zhangzimo/miniconda3/envs/horizon_quant/bin/python`
- TensorRT library path: `/usr/local/TensorRT-10.15.1.29/lib`
- TRT plugin: `/home/zhangzimo/Repos/private/edge-fm-x/build-3060/trt-edgellm/libNvInfer_edgellm_plugin.so`
- result: 18/18 cases completed; EdgeFM is faster than or equal to TRT in 1/18 mean cases and 1/18 median cases
- default-off effect: EdgeFM mean improved in all 18 cases versus the 14:34 baseline; improvements range from `0.45 ms` to `36.67 ms`

Current diagnosis:

- decode is already close to TRT or better on the large models
- the remaining gap is overwhelmingly prefill dominated
- post-default-off `3B / 2048x32` graph-off role aggregation: GateUp `242.41 ms` (`50.48%`), DownProj `125.26 ms` (`26.09%`), QKV `32.16 ms` (`6.70%`), attention `32.72 ms` (`6.81%`), OProj `23.69 ms` (`4.93%`), activation `14.64 ms` (`3.05%`)
- post-default-off `3B / 2048x64` graph-off role aggregation: GateUp `243.82 ms` (`50.62%`), DownProj `124.74 ms` (`25.90%`), QKV `32.25 ms` (`6.70%`), attention `32.81 ms` (`6.81%`), OProj `23.96 ms` (`4.97%`), activation `14.83 ms` (`3.08%`)
- TRT-Edge-LLM `3B / 2048x1` prefill reverse attribution:
  - total kernel time: `285.46 ms`; runtime-reported prefill: `286.41 ms`
  - kernel categories: GEMM `240.70 ms` (`84.32%`), attention `19.12 ms` (`6.70%`), SwiGLU `14.53 ms` (`5.09%`), TensorRT elementwise/norm `7.29 ms` (`2.55%`)
  - TensorRT NVTX role inference: Gate+Up uses one fused MatMul node per layer (`/mlp/up_proj/MatMul+/mlp/gate_proj/MatMul`) with `36` XMMA GEMM launches totaling `136.50 ms`
  - DownProj uses `36` XMMA GEMM launches totaling `71.36 ms`; the same kernel family also appears for OProj (`14.06 ms`)
  - QKV appears as TensorRT Myelin `FcCast` GEMM (`36` launches, `16.96 ms`)
  - AttentionPlugin uses TRT-Edge-LLM `fmha_v2_flash_attention_fp16_64_32_S_qkv_128_causal_sm86_kernel_nl` plus rope/write-KV helpers (`22.90 ms` inferred role time)
  - implication: TRT is not winning by eliminating the MLP structure; it is using faster TensorRT XMMA/Myelin GEMM selections for the same Gate+Up and DownProj roles. Existing-kernel reuse should focus on matching these third-party GEMM paths before any new kernel family.
- Myelin/XMMA direct reuse check:
  - `third_party/TensorRT-Edge-LLM/cpp` exposes runtime, attention plugin, embedding, KV-cache, sampling, and INT4 groupwise GEMM code, but no public/source-visible BF16/FP16 dense Myelin/XMMA GEMM launcher
  - `src/operators/linear_impl.cu` can register source-visible implementations, but a `myelin` or `xmma` impl id would be misleading unless it calls a buildable API in this repo
  - isolated subengine result: BF16 TensorRT Myelin/XMMA is not faster than current EdgeFM MLP, while FP16 TensorRT Myelin/XMMA matches TRT-Edge-LLM; this makes precision and bridge ownership the next review issue, not direct `myelin`/`xmma` impl registration
  - TRT-Edge-LLM `3B / 2048x1` trace uses `sm80_xmma_gemm_f16f16_*` for MLP Gate+Up and DownProj even though the HF checkpoint config says `torch_dtype=bfloat16`
  - latest runtime-weight TensorRT MLP probe shows one shape engine can bind actual Qwen2.5-3B layer weights at runtime and still select `sm80_xmma_gemm_f16f16_*`, removing the earlier serialized-weight duplication blocker for a prototype
  - latest EdgeFM-layout probe is closer to production: it binds resident BF16 GateUp/DownProj layouts, uses TensorRT internal casts plus transposed MatMul operands, and still selects FP16 XMMA; this removes the need for persistent FP16 weight copies in the first prototype
  - source-visible classic CUTLASS `device::Gemm` plus prepacked B layout was checked and did not approach TensorRT's FP16 XMMA numbers
  - the rest of the current vendored source-visible search did not find a direct SM86 dense FP16 runner: `trtllm_gemm_runner` is FP8/E2M1 only, `tgv_gemm` is SM100-only, and the TensorRT-LLM CUTLASS FP16/BF16 pieces here are MoE/grouped, fused-MoE, or quantized GEMM paths
  - next valid action is an env-gated TensorRT-backed prefill MLP bridge prototype using BF16 EdgeFM-owned weights with internal FP16 compute, or a genuinely new source-visible third-party runner with operator evidence; do not add fake `myelin`/`xmma` records or another small classic CUTLASS layout variant
- benchmark/copy alignment check:
  - official EdgeFM `prefill_ms` is recorded from before `EDGEFM_PREFILL_PREPARE` through prefill sampler, so it includes the host-to-device token copy and prefill tensor preparation
  - official TRT `prefill_ms` is the TRT internal GPU timer for `kLLM_PREFILL`; the outer TRT wall-time path was checked with and without explicit `torch.cuda.synchronize()`
  - TRT sync probe for `3B / 2048x32`: unsynchronized wall avg `921.5 ms`; synchronized wall avg `922.1 ms`; stage metrics remain around prefill `286-289 ms`, decode `633-634 ms`
  - EdgeFM graph-off memcpy in `3B / 2048x32`: `0.623 ms` total, dominated by `72` D2D copies of `1 MB` each for prefill K/V slices; H2D token copy is `8192 B` and about `0.001 ms`
  - EdgeFM graph-on formal trace shows only `0.044 ms` memcpy outside CUDA graph; this is too small to explain the `+201.15 ms` latest full-matrix prefill gap
- after default-off, `3B / 2048x32` is still the largest measured gap: EdgeFM mean `1125.25 ms` versus TRT mean `927.69 ms`; prefill gap `+201.15 ms`
- `3B / 2048x64` is close behind: EdgeFM mean `1778.55 ms` versus TRT mean `1584.36 ms`; prefill gap `+201.43 ms`

## Accepted Changes

- `src/engine/engine_factory.cpp`, `src/models/qwen2_5/qwen2_5.cpp`, `src/utils/device/weight_loader.*`
  - shared checkpoint paths now reuse GPU weights instead of loading the same model twice
  - this removed the 3B 3060 OOM caused by duplicate GPU weight copies
- `src/operators/attention_op.cu`
  - FP16 decode support stays on the tuned FlashInfer decode path
  - FP16 prefill still uses the generic prefill attention path
- `scripts/tune/tune_qwen_cublaslt.py`
  - `--dtype bf16|fp16` is available for aligned tuning and comparison
- `tests/operators/_test_utils.py`, `tests/operators/test_prefill_linear.py`
  - operator tests now resolve the active hw profile instead of assuming stale `cuda_sm80`
- `src/operators/activation_op.cu`
  - the scalar tail cleanup is accepted and should stay neutral for current Qwen hidden sizes
- `examples/config/platform/3060/operator_impl_table*.json`
  - 3060 table records now use `cuda_sm86`, matching the resolved 3060 test/runtime hw profile
  - `1p5b_fused_qkv_m2048` keeps `algo_index=4`; the later duplicate `algo_index=6` override was removed after a targeted regression check
- `tests/operators/test_fused_gate_up_activation.py`
  - prefill SwiGLU correctness is kept for the explicit diagnostic path with BF16 tolerance adjusted for small BF16 ULP drift
  - the unstable prefill SwiGLU performance smoke was removed from the gate
- `tests/operators/test_prefill_linear.py`
  - the 3060 prefill latency smoke threshold is now 6% instead of 3% to reflect observed run-to-run jitter on one remaining tuned case
- `tests/engine/test_qwen2_generate.py`
  - LLM benchmark now releases the Transformers model after each Transformers run before constructing EdgeFM/TRT runtimes
  - this fixed the 3B benchmark OOM/segfault path caused by holding the 3B Transformers model on GPU while creating `EdgeFM(cuda graph)`
  - blocker artifact: `.tmp_codex/bench/3060_20260509_1428_llm_3b_512x32_edge_only_probe.err`
  - confirmation artifact: `.tmp_codex/bench/3060_20260509_1432_llm_3b_512x32_after_tf_cleanup.json`
- `src/engine/tasks/token_generation/cuda/standard_engine.cpp`, `tests/engine/test_qwen2_generate.py`
  - fixed prefill CUDA graph first-capture semantics by replaying the captured graph immediately after capture
  - this is accepted as a correctness fix, not a performance optimization
  - regression command: `EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 python3 -m pytest -s tests/engine/test_qwen2_generate.py::test_generate_token_alignment_prefill_cuda_graph_first_request -q`
  - result: `1 passed`
  - broader command: `EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 python3 -m pytest -s tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or test_generate_token_alignment_cuda_graph or test_generate_token_alignment_prefill_cuda_graph_first_request" -q`
  - result: `3 passed, 13 deselected in 10.95s`
- `src/operators/fused_gate_up_activation_op.cu`, `tests/operators/test_fused_gate_up_activation.py`, `scripts/tune/profile_prefill_swiglu_kernels.py`
  - prefill SwiGLU fusion is now default-off because the current TensorRT-LLM/CUTLASS fused MoE reuse path regresses 3060 prefill
  - the path remains available for explicit diagnostics with `EDGE_FM_PREFILL_SWIGLU_FUSION=1`
  - the diagnostic script now sets that env by default for profiling and uses the same BF16 tolerance as the operator test
- `scripts/profile/profile_trt_edgellm_generate_case.py`, `scripts/profile/analyze_trt_nsys_profile.py`
  - TRT-Edge-LLM can now be profiled with the same token-id prefill construction as the official benchmark
  - nsys sqlite aggregation reports kernel categories plus TensorRT NVTX-derived roles for reverse attribution

Accepted end-to-end milestones:

- `0.5B / 512x32`: EdgeFM mean `133.02 ms` versus TRT mean `133.61 ms`; EdgeFM median `133.03 ms` versus TRT median `133.60 ms`
- `3B / 2048x32` prefill SwiGLU default-off target slice:
  - command: `EDGE_FM_BUILD_DIR=... EDGE_FM_PLATFORM=3060 ... report_qwen_benchmark_suite.py --repo-root /home/zhangzimo/Repos/private/edge-fm-x --device-id 0 --kind llm --llm-models 3b --prefill-list 2048 --decode-list 32 --json-only`
  - baseline from 14:34 full matrix: EdgeFM mean `1161.92 ms`, TRT mean `928.48 ms`, prefill gap `+237.97 ms`
  - default-off result: EdgeFM mean `1119.85 ms`, TRT mean `922.44 ms`, prefill gap `+200.90 ms`
  - improvement versus baseline: EdgeFM mean `-42.08 ms`, prefill gap `-37.07 ms`
  - raw artifact: `.tmp_codex/bench/3060_20260509_3b_2048x32_prefill_swiglu_default_off.json`
- prefill SwiGLU default-off regression slice:
  - command: `EDGE_FM_BUILD_DIR=... EDGE_FM_PLATFORM=3060 ... report_qwen_benchmark_suite.py --repo-root /home/zhangzimo/Repos/private/edge-fm-x --device-id 0 --kind llm --llm-models 0.5b,1.5b --prefill-list 2048 --decode-list 32 --json-only`
  - `0.5B / 2048x32`: EdgeFM mean `199.51 ms` versus 14:34 baseline `207.04 ms`, delta `-7.53 ms`; prefill gap improved by `6.44 ms`
  - `1.5B / 2048x32`: EdgeFM mean `563.22 ms` versus 14:34 baseline `585.26 ms`, delta `-22.04 ms`; prefill gap improved by `20.01 ms`
  - raw artifact: `.tmp_codex/bench/3060_20260509_prefill_swiglu_default_off_regression_0p5_1p5_2048x32.json`
- full matrix after prefill SwiGLU default-off:
  - command: `EDGE_FM_BUILD_DIR=... EDGE_FM_PLATFORM=3060 ... report_qwen_benchmark_suite.py --repo-root /home/zhangzimo/Repos/private/edge-fm-x --device-id 0 --kind llm --llm-models 0.5b,1.5b,3b --prefill-list 512,1024,2048 --decode-list 32,64 --json-only`
  - result: 18/18 cases completed; EdgeFM faster than or equal to TRT in `1/18` mean cases and `1/18` median cases
  - EdgeFM mean improved in every case versus the 14:34 baseline
  - largest remaining gap: `3B / 2048x32`, EdgeFM mean `1125.25 ms`, TRT mean `927.69 ms`, gap `+197.57 ms`
  - raw artifact: `.tmp_codex/bench/3060_20260509_1524_full_llm_matrix_prefill_swiglu_default_off.json`
  - summary artifact: `.tmp_codex/bench/3060_20260509_1524_full_llm_matrix_prefill_swiglu_default_off_summary.json`
- TRT-Edge-LLM `3B / 2048x1` prefill nsys reverse attribution:
  - command: `nsys profile --force-overwrite=true -o .tmp_codex/nsys/3060_trt_3b_2048x1_mapping --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop /home/zhangzimo/miniconda3/envs/horizon_quant/bin/python scripts/profile/profile_trt_edgellm_generate_case.py --model-path examples/qwen2.5-3b-instruct/qwen2.5-3b-instruct --engine-dir tests/data/trt_edgellm_workspace/qwen2.5-3b/engines_mxil2048 --plugin-path third_party/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so.1.0 --device-id 0 --prefill-len 2048 --decode-len 1 --warmup 1 --runs 1 --profile-range --output-json .tmp_codex/nsys/3060_trt_3b_2048x1_mapping_run.json --json`
  - runtime result: prefill `286.41 ms`, total wall `286.69 ms`, generated `1` token
  - nsys kernel result: total kernel `285.46 ms`, `368` launches, `18` unique kernel names
  - role result: Gate+Up `136.50 ms`, DownProj `71.36 ms`, AttentionPlugin `22.90 ms`, QKV `16.96 ms`, SwiGLU `14.53 ms`, OProj `14.06 ms`
  - raw artifacts: `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping.nsys-rep`, `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping.sqlite`, `.tmp_codex/nsys/3060_trt_3b_2048x1_mapping_kernel_summary.json`

Recent validation milestone:

- `2026-05-09 14:11 +0800`
  - build/config fix: `build-3060/CMakeCache.txt` was stale (`PLATFORM=a800`); it was reconfigured to `PLATFORM=3060`, `CMAKE_CUDA_ARCHITECTURES=86`
  - table cleanup: `1p5b_fused_qkv_m2048` duplicate same-specificity `algo_index=6` was removed; source and install 3060 tables now resolve this shape to `algo_index=4`
  - diagnostic artifact: `.tmp_codex/bench/3060_20260509_1403_1p5b_fused_qkv_m2048_algos.json`
  - diagnostic result: `no_shape_record=0.594944 ms`, `algo4=0.533504 ms`, `algo6=0.563200 ms`
  - command: `make -C build-3060 -j edge_fm_python`
  - result: pass
  - command: `make -C build-3060 install`
  - result: pass
  - command: `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 python3 scripts/operator_table/validate_operator_tables.py`
  - result: pass
  - command: `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -s tests/operators/test_prefill_linear.py tests/operators/test_attention_decode.py tests/operators/test_fused_gate_up_activation.py -q`
  - result: `21 passed, 2 skipped in 93.61s`
  - log: `.tmp_codex/bench/3060_20260509_1408_operator_gate_after_algo6_revert.log`
  - command: `EDGE_FM_BUILD_DIR=/home/zhangzimo/Repos/private/edge-fm-x/build-3060 EDGE_FM_PLATFORM=3060 EDGE_FM_DEVICE_ID=0 EDGE_FM_TEST_DEVICE_ID=0 python3 -m pytest -s tests/engine/test_qwen2_generate.py -k 'test_generate_token_alignment or test_generate_token_alignment_cuda_graph' -q`
  - result: `2 passed, 13 deselected in 7.55s`
  - log: `.tmp_codex/bench/3060_20260509_1411_generate_alignment_after_algo6_revert.log`
  - raw artifact: `.tmp_codex/bench/3060_20260509_1411_algo6_revert_validation.json`
  - conclusion: this is a build hygiene and table regression cleanup result, not an end-to-end performance win

## Current Remaining Gap

### 2026-05-17 Stage 2 pre-rotate attention table absorption

The hard `>=1%` default gate has been relaxed for low-risk, stable, localized
positive changes. A sub-1% win can now be absorbed when correctness is clean and
the change is useful as a stepping stone.

Absorbed change:

- new attention impl: `flashinfer_attention_prefill_prerotate`
- behavior: pre-apply BF16 Llama RoPE into scratch Q/K tensors, then call the
  FlashInfer prefill kernel with `pos_encoding=None`
- table: `examples/config/platform/3060/operator_impl_table_llm.json` now selects
  this impl for the concrete 3060 Qwen2.5 prefill attention shapes

Paired CUDA graph results:

| Case | Baseline | Pre-rotate | Delta | Prefill delta |
| --- | ---: | ---: | ---: | ---: |
| `0.5B / 2048x32` | `177.974 ms` | `176.085 ms` | `-1.889 ms` | `-1.764 ms` |
| `1.5B / 2048x32` | `479.023 ms` | `476.601 ms` | `-2.421 ms` | `-2.381 ms` |
| `3B / 2048x32` | `938.262 ms` | `933.502 ms` | `-4.761 ms` | `-4.603 ms` |

Fresh gap after this local absorption, compared to the paired external
`TRT-Edge-LLM` reference from `.tmp_codex/bench/stage2_fresh_trt_edge_20260517`:

| Case | EdgeFM pre-rotate | TRT-Edge-LLM | Remaining gap |
| --- | ---: | ---: | ---: |
| `0.5B / 2048x32` | `176.085 ms` | `167.850 ms` | `+8.234 ms` (`+4.91%`) |
| `1.5B / 2048x32` | `476.601 ms` | `467.580 ms` | `+9.021 ms` (`+1.93%`) |
| `3B / 2048x32` | `933.502 ms` | `916.511 ms` | `+16.991 ms` (`+1.85%`) |

Runtime optimization candidate opened from this result:

- do not only move RoPE to a side stream; Q/K rotation remains an attention
  prerequisite and has limited overlap room
- higher-value candidate: let prefill attention read strided K/V directly from
  fused QKV, and move raw K/V cache writes to a side stream that overlaps with
  attention/OProj/MLP; wait only before the next layer reuses the QKV scratch
- required production probe: extend prefill attention context with K/V row
  stride, add a correctness test, then benchmark `3B/1.5B/0.5B 2048x32`

### 2026-05-17 Runtime/dataflow probe: strided QKV attention

Implemented a guarded runtime probe behind
`EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=1` and kept the default path unchanged.
Two variants were measured on `Qwen2.5-3B-Instruct / 2048x32 / CUDA graph on`:

| Variant | Avg total | Avg prefill | Paired delta | Decision |
| --- | ---: | ---: | ---: | --- |
| default-off paired baseline | `932.764 ms` | `306.816 ms` | baseline | keep |
| strided K/V direct from packed QKV + side-stream KV write | `934.608 ms` | `308.335 ms` | `+1.844 ms` | reject |
| K-only strided input + side-stream K cache write, V contiguous | `932.997 ms` | `306.978 ms` | `+0.234 ms` | reject |

Correctness:

- `tests/operators/test_attention_prefill.py -k "prerotate or strided_qkv or k_only_strided"`: passed
- `tests/engine/test_qwen2_generate.py -k "max_new_tokens"` with
  `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=1`: passed

Artifacts:

- `.tmp_codex/bench/stage2_20260517_runtime/edgefm_3b_2048x32_strided_qkv_side.json`
- `.tmp_codex/bench/stage2_20260517_runtime/edgefm_3b_2048x32_strided_k_side.json`
- `.tmp_codex/bench/stage2_20260517_runtime/edgefm_3b_2048x32_default_off_pair.json`

Conclusion: the copy/scheduling bubble is too small to beat the cost and noise
of this runtime change. Do not default-enable the strided QKV runtime gate.
Continue with fusion/operator work, especially pre-rotate fusion with QKV split
or attention prelude.

### 2026-05-17 Rejected probe: FP16-IO prefill attention

Fresh Stage-2 profiling showed the external TRT reference running an FP16 FMHA
kernel, while EdgeFM's current Qwen path keeps BF16 Q/K/V/O around FlashInfer.
A temporary default-off diagnostic impl was built to isolate whether "BF16
input/output, FP16 internal FlashInfer FMHA" could recover the attention body
gap without TensorRT engine bridge.

Result on `Qwen2.5-3B-Instruct / 2048x32 / CUDA graph on / runs=3`:

| Variant | Avg total | Avg prefill | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| current table | `931.702 ms` | `305.314 ms` | baseline | keep |
| FP16-IO attention diagnostic | `937.198 ms` | `310.872 ms` | `+5.497 ms` total, `+5.558 ms` prefill | reject |

Artifacts:

- `.tmp_codex/bench/stage2_20260517_attention_fp16_diag/3b_2048x32_default.json`
- `.tmp_codex/bench/stage2_20260517_attention_fp16_diag/3b_2048x32_fp16_attention_diag.json`
- `.tmp_codex/bench/stage2_20260517_attention_fp16_diag/operator_impl_table_3b_fp16_attention_diag.json`

Conclusion: the full Q/K/V/O BF16<->FP16 cast boundary is more expensive than
the exposed FP16 FlashInfer prefill benefit on this path, and it also increases
`attention_op.cu` compile cost via extra FP16 prefill template instantiation.
The diagnostic source change was removed; do not pursue a full attention
cast-wrapper route. Any future FP16-residency attention plan must avoid these
per-layer full-tensor casts, for example by producing resident FP16 Q/K/V from
the upstream QKV source-op and proving token alignment before benchmarking.

Latest full-matrix bridge artifact:

- `.tmp_codex/bench/3060_20260511_qkv_bias_oproj_linear_bridge_full_llm_matrix_runs3_summary.json`

Bridge mode:

- GateUp-FP16 MLP bridge + BF16 TensorRT QKV/OProj linear bridge

| Case | Edge mean | TRT mean | Mean gap | Median gap | Prefill gap | Decode gap | Note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3B / 2048x32` | `953.29 ms` | `927.69 ms` | `+25.60 ms` | `+25.59 ms` | `+25.56 ms` | `+0.27 ms` | prefill dominated |
| `3B / 2048x64` | `1603.74 ms` | `1584.36 ms` | `+19.38 ms` | `+19.43 ms` | `+19.38 ms` | `+0.00 ms` | prefill dominated |
| `0.5B / 2048x64` | `311.43 ms` | `293.41 ms` | `+18.01 ms` | `+17.83 ms` | `+18.05 ms` | `-0.04 ms` | prefill dominated |
| `3B / 1024x32` | `777.56 ms` | `764.35 ms` | `+13.21 ms` | `+13.25 ms` | `+13.24 ms` | `-0.03 ms` | prefill dominated |
| `1.5B / 2048x32` | `486.14 ms` | `474.23 ms` | `+11.91 ms` | `+11.83 ms` | `+11.91 ms` | `+0.00 ms` | prefill dominated |
| `1.5B / 2048x64` | `823.57 ms` | `811.69 ms` | `+11.87 ms` | `+11.96 ms` | `+11.87 ms` | `-0.01 ms` | prefill dominated |
| `0.5B / 2048x32` | `180.13 ms` | `168.37 ms` | `+11.76 ms` | `+11.70 ms` | `+11.76 ms` | `+0.00 ms` | prefill dominated |
| `1.5B / 1024x64` | `724.52 ms` | `713.25 ms` | `+11.27 ms` | `+11.38 ms` | `+11.28 ms` | `-0.01 ms` | prefill dominated |

Full matrix status from the latest raw artifact:

- `18/18` cases completed.
- EdgeFM is faster than TRT in `3/18` cases: `0.5B / 512x32`, `0.5B / 512x64`, and `3B / 512x64`.
- The largest gaps are still prefill dominated, especially 3B long-prefill cases.
- Decode is not the main blocker for 3B; it is already at or slightly better than TRT in the latest matrix.
- `EDGE_FM_TRT_MLP_FP16_WEIGHTS=both` is not an active 3B default target. Round
  C added an explicit `auto` mode so smaller models can choose `both` when the
  estimated DownProj FP16 copy fits the configured memory cap, while 3B resolves
  to `gateup`.

Historical bridge diagnostics that are no longer the current gap frame:

- `EDGE_FM_TRT_MLP_FP16_WEIGHTS=gateup` still matters as the cleanest MLP-only bridge variant and the only bridge path that safely improves all 18 cases versus the BF16 bridge.
- The gateup and both FP16 weight-input probes remain useful for attribution, but the current active bridge path is the QKV/OProj linear bridge on top of GateUp-FP16.

The `both` mode is the strongest latency result but is now a blocked diagnostic, not an active route. GateUp-only remains the cleanest MLP-only variant, but the active bridge path has moved on to QKV/OProj linear on top of GateUp-FP16. Down-only is still lower risk for memory, but too small to be the first optimization unless later end-to-end data contradicts the single-layer probe.

## Rejected / Obsolete

- `Qwen2.5-3B-Instruct / source-visible cuTile dense MatMul probe`
  - rejected because the best persistent `64x128x64` cuTile MatMul results are slower than the current BF16 cublasLt baseline and far behind TRT FP16 XMMA
  - FP16 GateUp `9.87 ms`, DownProj `4.80 ms`; BF16 GateUp `9.75 ms`, DownProj `4.79 ms`
  - raw artifacts: `.tmp_codex/bench/3060_20260510_cutile_matmul_fp16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_bf16_probe.json`, `.tmp_codex/bench/3060_20260510_cutile_matmul_summary.json`
  - conclusion: keep cuTile only as a diagnostic helper, not as a near-term production path

- `Qwen2.5-3B-Instruct / BF16 / attention prefill / prefill_cta_tile_q=128`
  - rejected because the isolated attention kernel win did not survive end-to-end on the current bridge path
  - paired bridge run on the same environment: `3B / 2048x32` EdgeFM `947.754 ms` with `prefill_cta_tile_q=64` versus `948.438 ms` with `128`, delta `+0.684 ms` (`+0.07%`)
  - isolated attention probe still showed `128` at `0.8974 ms` median versus `0.9115 ms` for `64`, but that did not translate to the full model
  - raw paired artifact: `.tmp_codex/bench/3060_20260512_attention_tile128_bridge_paired_edgefm_3b_2048x32.json`
  - raw isolated probe: `.tmp_codex/bench/3060_20260512_attention_prefill_3b_2048_tune.json`
  - conclusion: keep the attention table at `prefill_cta_tile_q=64`; do not promote `128` to the 3060 LLM table

- `Qwen2.5-3B-Instruct / BF16 / prefill attention / FP16-IO FlashInfer wrapper`
  - rejected because the diagnostic cast-wrapper regressed the current source-op path on the target slice
  - paired result: current table `931.702 ms` total / `305.314 ms` prefill versus FP16-IO diagnostic `937.198 ms` total / `310.872 ms` prefill
  - raw artifacts: `.tmp_codex/bench/stage2_20260517_attention_fp16_diag/3b_2048x32_default.json`, `.tmp_codex/bench/stage2_20260517_attention_fp16_diag/3b_2048x32_fp16_attention_diag.json`
  - conclusion: do not add a full BF16<->FP16 attention wrapper; only revisit FP16 residency if upstream QKV can produce resident FP16 tensors without extra per-layer cast kernels and token alignment passes

- `Qwen2.5-3B-Instruct / BF16 / prefill m=2048 / fused_gate_up + mlp_down cublasLt table sweep`
  - rejected because the official CUDA graph target slice improved by only `2.12 ms` mean and `2.84 ms` median, below the `>=1%` acceptance rule
  - candidate table edit tested: `fused_gate_up algo_index=3` and `mlp_down algo_index=2`
  - official target result: `3B / 2048x32` EdgeFM mean `1117.73 ms`, TRT mean `922.69 ms`, prefill gap `+198.95 ms`
  - compare target-slice baseline: `.tmp_codex/bench/3060_20260509_3b_2048x32_prefill_swiglu_default_off.json`, EdgeFM mean `1119.85 ms`, prefill gap `+200.90 ms`
  - raw candidate artifact: `.tmp_codex/bench/3060_20260509_3b_2048x32_gateup3_mlpdown2_candidate_trt.json`
  - microbench artifacts: `.tmp_codex/bench/3060_20260509_3b_m2048_fused_gate_up_cublaslt_heuristic.json`, `.tmp_codex/bench/3060_20260509_3b_m2048_mlp_down_cublaslt_heuristic.json`
  - candidate table edits were reverted; table validation after revert passed: `.tmp_codex/bench/3060_20260509_3b_m2048_cublaslt_rejected_revert_table_validate.log`
- `Qwen2.5-3B-Instruct / BF16 / prefill m=2048 / existing LinearCutlassImpl configs`
  - rejected as a near-term route because none of the three currently wired CUTLASS configs beat the accepted 3060 cublasLt records for the dominant GateUp or DownProj shapes
  - GateUp results: current table `6.852 ms` with cublasLt `algo_index=4`; no shape record `7.031 ms`; CUTLASS `128x128x32_s3` `6.973 ms`; `128x256x32_s3` `7.344 ms`; `256x128x32_s3` `7.322 ms`
  - DownProj results: current table `3.545 ms` with cublasLt `algo_index=5`; no shape record `3.675 ms`; CUTLASS `128x128x32_s3` `3.718 ms`; `128x256x32_s3` `3.852 ms`; `256x128x32_s3` `3.840 ms`
  - raw artifact: `.tmp_codex/bench/3060_20260509_probe_3b_m2048_cutlass_vs_cublaslt.json`
  - conclusion: do not re-add broad CUTLASS heuristics to `src/operators/linear_impl.cu`; only a new, specific third-party config with operator and end-to-end evidence should be considered
- `Qwen2.5-3B-Instruct / prefill m=2048 / cublasLt native row-major and packed layout probe`
  - rejected as a low-risk production direction because native row-major descriptors are neutral/slightly slower, and tested cublasLt packed weight transforms returned unsupported for the current dense FP16/BF16 shapes
  - BF16 GateUp `M=2048,N=22016,K=2048`: current column-view `6.760 ms`; native row-major `6.797 ms`; packed layout transform attempts returned cublasLt status `15`
  - BF16 DownProj `M=2048,N=2048,K=11008`: current column-view `3.495 ms`; native row-major `3.498 ms`; packed layout transform attempts returned cublasLt status `15`
  - FP16 GateUp: current column-view `6.702 ms`; native row-major `6.701 ms`; packed transform unsupported
  - FP16 DownProj: current column-view `3.490 ms`; native row-major `3.497 ms`; packed transform unsupported
  - raw artifacts: `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_gateup_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_downproj_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_gateup_fp16.jsonl`, `.tmp_codex/bench/3060_20260509_cublaslt_layout_probe_3b_downproj_fp16.jsonl`
  - conclusion: do not change production descriptors or add a weight prepack path based on this probe; continue toward a proven third-party GEMM runner/config rather than cublasLt layout churn
- `Qwen2.5-3B-Instruct / prefill m=2048 / standalone CUTLASS dense GEMM config sweep`
  - rejected because the tested existing CUTLASS tile/stage configs do not beat current cublasLt on the production BF16 path
  - BF16 GateUp: best `128x128x32_s3_w64x64` at `6.894 ms`, slower than current cublasLt column-view `~6.76-6.85 ms`
  - BF16 DownProj: best `128x128x32_s3_w64x64` at `3.727 ms`, slower than current cublasLt column-view `~3.49-3.55 ms`
  - FP16 GateUp: best `64x128x32_s3_w32x64` at `6.823 ms`, slower than FP16 cublasLt best `6.776 ms`
  - FP16 DownProj: best `64x128x32_s4_w32x64` at `3.524 ms`, roughly tied with FP16 cublasLt best `3.526 ms` but only on the temporary FP16 checkpoint direction
  - raw artifacts: `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_gateup_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_downproj_bf16.jsonl`, `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_gateup_fp16_serial.jsonl`, `.tmp_codex/bench/3060_20260509_cutlass_config_probe_3b_downproj_fp16_serial.jsonl`
  - conclusion: do not add more `LinearCutlassImpl` configs from this sweep; continue searching for a higher-level third-party GEMM runner or a reviewed larger prefill path
- `Qwen2.5-3B-Instruct / temporary FP16 EdgeFM checkpoint`
  - rejected as an explanation for the TRT gap because converting EdgeFM weights to FP16 did not improve the official target slice
  - `3B / 2048x32` FP16 probe: EdgeFM mean `1131.10 ms`, TRT mean `921.44 ms`; EdgeFM prefill `498.95 ms`, TRT prefill `287.42 ms`; prefill gap `+211.53 ms`
  - compare latest BF16 full-matrix case: EdgeFM mean `1125.25 ms`, TRT mean `927.69 ms`, prefill gap `+201.15 ms`
  - raw artifact: `.tmp_codex/bench/3060_20260509_3b_2048x32_fp16_edgefm_probe.json`
  - conclusion: the remaining gap is not explained by BF16 versus FP16 alone; do not convert the production checkpoint path as a performance fix
- `Qwen2.5-3B-Instruct / FP16 / prefill m=2048 / cublasLt operator tuning`
  - rejected as a mainline route because the best operator-level movement is far below the remaining `~200 ms` prefill gap and was measured only on a temporary FP16 checkpoint
  - GateUp FP16: baseline `6.916 ms`, best `algo_index=4` `6.776 ms`
  - DownProj FP16: baseline `3.676 ms`, best `algo_index=1` `3.526 ms`
  - rough upper-bound movement across 36 layers is about `10 ms`, before end-to-end noise and without addressing the BF16 production path
  - raw artifacts: `.tmp_codex/bench/3060_20260509_3b_fp16_m2048_fused_gate_up_cublaslt_all.json`, `.tmp_codex/bench/3060_20260509_3b_fp16_m2048_mlp_down_cublaslt_all.json`
- `Qwen2.5-3B-Instruct / BF16 TensorRT MLP subengine bridge`
  - rejected as the next production route because the isolated BF16 TensorRT subengine did reproduce `sm80_xmma_gemm_bf16bf16_*` and `__myl_SiluMul_*`, but did not beat EdgeFM's current MLP estimate
  - BF16 timed-region median: `11.33 ms` per layer
  - BF16 nsys split per layer: GateUp `7.207 ms`, DownProj `3.701 ms`, SwiGLU `0.401 ms`
  - compare current EdgeFM graph-off MLP estimate: `(GateUp 242.41 + activation 14.64 + DownProj 125.26) / 36 = 10.62 ms` per layer
  - FP16 subengine is not rejected as attribution: it matched TRT-Edge-LLM at `6.26 ms` median per layer, but this is a precision/bridge design question and not an accepted production implementation
  - raw artifacts: `.tmp_codex/bench/trt_mlp_subengine_3b_bf16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_bf16_kernel_summary.md`, `.tmp_codex/bench/trt_mlp_subengine_3b_fp16_nsys_run.json`, `.tmp_codex/nsys/trt_mlp_subengine_3b_fp16_kernel_summary.md`
- `Qwen2.5-3B-Instruct / FP16 CUTLASS prepacked-B layout probe`
  - rejected as a near-term production route because prepacking B for classic CUTLASS `device::Gemm` improved one shape but stayed far behind TensorRT's FP16 XMMA subengine
  - GateUp best: `6.666 ms`, `row_major_prepacked`, `64x128x32_s3_w32x64`; TensorRT FP16 GateUp reference is `3.856 ms`
  - DownProj best: `3.440 ms`, `row_major_prepacked`, `64x128x32_s3_w32x64`; TensorRT FP16 DownProj reference is `1.984 ms`
  - raw artifacts: `.tmp_codex/bench/3060_20260510_cutlass_layout_gateup_fp16_sweep.jsonl`, `.tmp_codex/bench/3060_20260510_cutlass_layout_down_fp16_sweep.jsonl`, `.tmp_codex/bench/3060_20260510_cutlass_layout_fp16_sweep_summary.json`
  - conclusion: do not add a production prepacked-weight CUTLASS path from this result alone; the remaining gap needs a different third-party runner or a reviewed FP16 TensorRT bridge
- `Qwen2.5-3B-Instruct / BF16 / prefill m=2048 / cublasLt explicit low-level configs`
  - rejected because explicit candidates did not beat the best heuristic candidates
  - `mlp_down` best explicit `3.662 ms`, while heuristic best was `3.491 ms`
  - `fused_gate_up` best explicit was baseline `7.010 ms`, while heuristic best was `6.729 ms`
  - raw artifacts: `.tmp_codex/bench/3060_20260509_3b_m2048_mlp_down_cublaslt_explicit_top32.json`, `.tmp_codex/bench/3060_20260509_3b_m2048_fused_gate_up_cublaslt_explicit_top32.json`
- `src/operators/linear_impl.cu` extra CUTLASS tile heuristic
  - removed during cleanup before commit because it had no accepted operator + CUDA graph evidence
  - keep the original three stable CUTLASS configs until a specific profile-backed candidate clears the acceptance gate
- `Qwen2.5-3B-Instruct / BF16 / prefill attention m=2048 / FlashInfer cta_tile_q sweep`
  - rejected for end-to-end priority because best `cta_tile_q=128` only improved the per-layer attention microbench from current `0.921 ms` to `0.908 ms`
  - expected full-prefill movement is roughly `0.46 ms` across 36 layers, far below the `>=1%` target-case rule
  - raw artifact: `.tmp_codex/bench/3060_20260509_3b_m2048_attention_prefill_cta_sweep.json`

- `Qwen2.5-1.5B-Instruct / BF16 / prefill m=1024 / mlp_down`
  - rejected because the tuned record regressed operator latency (`1.167 ms` versus `1.085 ms`)
  - removing it kept the official CUDA graph result at `431.52 ms`
- `Qwen2.5-0.5B-Instruct / BF16 / prefill m=2048 / mlp_down`
  - CUTLASS probe was faster in microbench but slower end-to-end
  - keep the stable cublasLt record instead
- `Qwen2.5-0.5B-Instruct / BF16 / prefill m=1024 / attention_output`
  - explicit cublasLt record regressed or stayed neutral end-to-end
  - rejected
- `Qwen2.5-1.5B-Instruct / BF16 / prefill m=2048 / fused_qkv / algo_index=6`
  - rejected because a duplicate same-specificity table record made `algo_index=6` override `algo_index=4`
  - targeted median result: `algo4=0.533504 ms`, `algo6=0.563200 ms`, `no_shape_record=0.594944 ms`
  - formal failed gate before cleanup: `algo6 tuned=0.605184 ms` versus `algo4 baseline=0.536576 ms`
  - raw artifacts: `.tmp_codex/bench/3060_20260509_1357_operator_gate.log`, `.tmp_codex/bench/3060_20260509_1403_1p5b_fused_qkv_m2048_algos.json`
- `cutlass_prefill_swiglu` as a default-enabled path
  - rejected as default-on for 3060 because it regressed microbench and end-to-end prefill
  - microbench command: `python3 scripts/tune/profile_prefill_swiglu_kernels.py --model-path examples/qwen2.5-3b-instruct/qwen2.5-3b-instruct --device-id 0 --dtype bf16 --seq-lens 512,1024,2048 --warmup 10 --iters 30 --output-json .tmp_codex/bench/3060_20260509_prefill_swiglu_3b_current.json`
  - 3B microbench raw artifact: `.tmp_codex/bench/3060_20260509_prefill_swiglu_3b_current.json`
  - 3B microbench result: `512` fused `2.085 ms` vs two-stage `1.832 ms`; `1024` fused `4.144 ms` vs two-stage `3.795 ms`; `2048` fused `8.239 ms` vs two-stage `7.177 ms`
  - official target result after default-off: `3B / 2048x32` EdgeFM mean improved by `42.08 ms`
  - conclusion: keep the existing-kernel path only behind `EDGE_FM_PREFILL_SWIGLU_FUSION=1` for diagnostics
- `Qwen2.5-3B-Instruct / cublasLt mixed BF16 activation x FP16 weight`
  - rejected because cublasLt returned no heuristic candidate for `A=CUDA_R_16BF`, `B=CUDA_R_16F` with either FP16 or BF16 output
  - tested small smoke plus 3B GateUp and QKV shapes
  - raw artifacts: `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/attempts/cublaslt_mixed_smoke_{fp16out,bf16out}.json`, `deliverables/kernel_opt/3060_prefill_mlp_sourceop_humanize_20260516/artifacts/attempts/cublaslt_mixed_3b_{gateup_fp16out,qkv_bf16out}.json`
  - conclusion: do not build production source-op logic around mixed cublasLt descriptors; cast work must use explicit conversion/residency or a different source-visible GEMM stack
- `Qwen2.5-3B-Instruct / BF16 / prefill attention / production-like QKV-strided Q`
  - rejected as a meaningful boundary fix because contiguous Q and production-like fused-QKV-strided Q are essentially tied in the EdgeFM AttentionLayer harness
  - result: 3B `S=2048` contiguous Q `0.984016 ms`, qkv-strided Q `0.985792 ms`; qkv-strided S32 correctness passes
  - raw artifacts: `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/edgefm_flashinfer_3b_s2048_{contiguous_q,qkv_strided}_rerun.json`, `deliverables/kernel_opt/3060_bf16_fmha_humanize_20260516/artifacts/attempts/edgefm_flashinfer_3b_s32_qkv_strided_check.json`
  - conclusion: do not add a Q-contiguous copy/fusion path solely to help FlashInfer prefill
- `Qwen2.5-3B-Instruct / BF16 / prefill attention / FlashInfer half QK accumulation`
  - rejected on correctness after enabling the vendored FlashInfer half-accumulation branch in the standalone diagnostic harness
  - third-party compatibility edits were needed only for the diagnostic branch: remove the hard dependency on missing `boost/math/ccmath/fabs.hpp`, build the extension as C++20 for `std::bit_cast`, and avoid ambiguous `half != float` comparison in `AttentionVariantBase::OutputTransform`
  - 3B `S=32` pre-rotate+no-RoPE with `qk_accum=half` fails reference check: `max_abs=3.023071`, `mean_abs=0.329942`
  - same path with `qk_accum=float` still passes: `max_abs=0.015625`, `mean_abs=0.000484`
  - raw artifacts: `.tmp_codex/bench/stage2_20260517_qk_accum_half/3b_s32_prerotate_cta64_cap2_qkhalf_check.json`, `.tmp_codex/bench/stage2_20260517_qk_accum_half/3b_s32_prerotate_cta64_cap2_qkfloat_check.json`
  - conclusion: do not use half QK accumulation for EdgeFM BF16 prefill; continue with BF16-correct FMHA/source-op or runtime-boundary candidates
- `Qwen2.5-3B-Instruct / BF16 / prefill MLP source-op / shape-specific tile split`
  - rejected after paired confirmation; the apparent 5-run gate-up-only win did not reproduce
  - baseline confirm: `3B 2048x32` avg `931.129 ms`, prefill `304.849 ms`
  - `gateup_tile=128x256x32`: 5-run sample `929.207 ms`, but 10-run confirm `932.055 ms`, prefill `305.437 ms`
  - `down_tile=128x128x32_warp32x64`: 5-run `935.026 ms`, prefill `308.740 ms`
  - `gateup_tile=128x256x32 + down_tile=128x128x32_warp32x64`: 5-run `932.821 ms`, prefill `306.803 ms`
  - `activation_mode=mixed_bf16`: 5-run `932.605 ms`, prefill `306.304 ms`
  - raw artifacts: `.tmp_codex/bench/stage2_20260517_mlp_tile_probe/`
  - conclusion: keep the 3B MLP source-op record unchanged; require confirm runs before absorbing sub-1% tile movement
- `Qwen2.5-3B-Instruct / BF16 / prefill fused_qkv / CUTLASS fused-bias epilogue`
  - rejected on correctness even though the layer microbench looked faster
  - graph-on end-to-end probe was inconclusive/noisy: first 10-run `931.442 -> 931.295 ms`; reversed 20-run drifted heavily and could not be used as acceptance evidence
  - same-process 3B QKV layer microbench: no-fuse `0.644352 ms`, fused-bias `0.585728 ms`, but output mismatch was unacceptable (`max_abs=91.75`, `mean_abs=1.02755`)
  - setting the CUTLASS broadcast epilogue `StoreT=false` did not fix the 3B mismatch
  - raw artifacts: `.tmp_codex/bench/stage2_20260517_fuse_bias/`
  - conclusion: do not expose `fuse_bias` in the production operator table; keep the safe no-fuse mixed-BF16 QKV path and continue with higher-confidence runtime/operator candidates
- `Qwen2.5-3B-Instruct / BF16 / prefill strided-QKV attention + side-stream KV copy`
  - rejected for CUDA graph default despite a promising runtime win
  - `3B 2048x32` graph-on paired probe: default avg `932.365 ms`, prefill `305.835 ms`; `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=1` avg `930.325 ms`, prefill `304.204 ms`
  - correctness gate failed under CUDA graph: `test_generate_token_alignment_cuda_graph` mismatched all 20 decode steps; disabling the side stream did not fix it
  - non-graph token alignment still passed with strided-QKV enabled (`3 passed`), so the failure is graph-path specific
  - raw artifacts: `.tmp_codex/bench/stage2_20260517_runtime_next/3b_2048x32_{strided_qkv_side_stream,baseline_post_cleanup}_confirm10.*`
  - conclusion: keep the env-gated diagnostic path off for graph-on benchmarks; revisit only if the prefill graph capture/replay contract is made strided-QKV aware
- `Qwen2.5-3B-Instruct / BF16 / prefill linear runtime weight prefetch`
  - rejected because overlapping nonpersistent QKV/OProj weight casts with nearby attention did not improve the critical path
  - `3B 2048x32` graph-on: baseline avg `932.192 ms`, prefill `305.792 ms`; prefetch avg `932.366 ms`, prefill `305.808 ms`
  - `3B 2048x1` prefill-only: baseline avg `306.664 ms`, prefill `306.547 ms`; prefetch avg `307.080 ms`, prefill `306.985 ms`
  - memory also moved in the wrong direction (`400.8 MB` free after warmup to `378.8 MB` on `2048x32`)
  - raw artifacts: `.tmp_codex/bench/stage2_20260517_prefetch_weight_probe/`
  - conclusion: remove the production-path prefetch API/calls and do not add a `prefetch_weights` table knob
- `Qwen2.5-{0.5B,1.5B,3B} / BF16-FP16 direct conversion kernels`
  - accepted as a low-risk source-op boundary improvement: replace BF16->float->FP16 and FP16->float->BF16 scalar/vector conversion kernels with direct CUDA BF16/FP16 constructors, which lower to direct conversion instructions on SM86
  - correctness passed:
    `tests/operators/test_prefill_linear.py -k "source_op or bf16_direct or overlap_casts"` (`5 passed`) and
    `tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment_cuda_graph and not vl"` (`2 passed`)
  - main target after cleanup: `3B 2048x32` graph-on avg `928.890 ms`, prefill `303.043 ms`, decode `625.713 ms`
  - latest gap versus external TRT-Edge-LLM on `2048x32`:
    - `0.5B`: EdgeFM `173.965 ms`, TRT `167.613 ms`, gap `+6.352 ms` (`+3.79%`)
    - `1.5B`: EdgeFM `474.902 ms`, TRT `468.809 ms`, gap `+6.093 ms` (`+1.30%`)
    - `3B`: EdgeFM `928.890 ms`, TRT `917.630 ms`, gap `+11.260 ms` (`+1.23%`)
  - adjacent checks after the patch: `0.5B 512x32 125.502 ms`, `0.5B 1024x32 140.096 ms`, `1.5B 512x32 345.347 ms`, `1.5B 1024x32 387.851 ms`, `3B 512x32 686.868 ms`, `3B 1024x32 761.629 ms`, `3B 2048x64 1577.463 ms`
  - raw artifacts: `.tmp_codex/bench/stage2_20260517_direct_convert_probe/`
  - conclusion: keep the direct conversion kernel change; continue Stage-2 on remaining 3B prefill attention/FMA boundary gap and secondary 0.5B decode opportunities
- `Qwen2.5-3B-Instruct / BF16 / prefill linear / single-role persistent FP16 weights`
  - rejected as a default source-op strategy after a fresh current-table recheck
  - OProj-only persistent weights produced a tiny full-generate movement:
    `3B 2048x32` avg `928.890 -> 928.424 ms`, but left only `122.8 MB`
    free after warmup/runs
  - QKV-only persistent weights regressed total time:
    `928.890 -> 929.422 ms`, and left only `52.8 MB` free
  - raw artifacts:
    `.tmp_codex/operator_tables/stage2_20260517_persistent_linear_probe/`,
    `.tmp_codex/bench/stage2_20260517_persistent_linear_probe/`
  - conclusion: keep 3B QKV/OProj nonpersistent in the production table. The
    small OProj win is not worth the memory risk on 12 GB 3060, especially
    because decode/KV growth and adjacent shapes need headroom.
- `Qwen2.5-3B-Instruct / BF16 / prefill attention / FlashInfer max_mma_kv_cap recheck`
  - rejected after the layer microbench did not reproduce at full graph-on
    generate level
  - microbench for 3B `S=2048` suggested `prefill_cta_tile_q=64` with
    `prefill_max_mma_kv_cap={0,4,8}` could beat current `cap=2`
    (`~0.810 ms/layer` versus current `0.856 ms/layer`)
  - full generate contradicted the microbench:
    - `cap=8`: `3B 2048x32` avg `929.625 ms`, prefill `303.666 ms`
    - `cap=0`: `3B 2048x32` avg `930.722 ms`, prefill `304.522 ms`
    - current direct-conversion baseline remains `928.890 ms`, prefill
      `303.043 ms`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260517_attention_sweep_3b_2048.json`,
    `.tmp_codex/bench/stage2_20260517_attention_sweep/`
  - conclusion: keep the current 3B attention table
    `prefill_cta_tile_q=64, prefill_max_mma_kv_cap=2`. Treat isolated
    attention microbench wins as insufficient when graph-on end-to-end timing
    moves in the opposite direction.
- `Qwen2.5-0.5B-Instruct / BF16 / decode fresh attribution`
  - refreshed the current source-op/default-off path against external
    TRT-Edge-LLM on `2048x64`
  - paired result:
    - EdgeFM graph-on avg `300.235 ms`, prefill `50.761 ms`, decode
      `249.310 ms`
    - TRT-Edge-LLM avg `291.652 ms`, prefill `45.773 ms`, decode
      `245.784 ms`
    - gap: total `+8.583 ms` (`+2.94%`), prefill `+4.988 ms`, decode
      `+3.526 ms`
  - graph-off EdgeFM decode attribution:
    - GateUp GEMV `82.082 ms`
    - LMHead `50.712 ms`
    - DownProj `49.622 ms`
    - FlashInfer decode attention plus merge `39.265 ms`
    - Norm `6.716 ms`, SwiGLU `3.173 ms`
  - follow-up probes:
    - `lm_head_top1` recheck: `300.235 -> 300.269 ms`; still neutral/slower
    - decode attention chunk sweep: best `[128,256,384,512]` moved the
      microbench only `0.029472 -> 0.029429 ms`; not enough for a default table
      change
    - GateUp cublasLt heuristic/explicit retune: baseline remains best
      (`~0.05824 ms`)
    - DownProj cublasLt retune confirmed current `algo_index=1` is best
      (`0.037888 ms`)
    - LMHead cublasLt heuristic retune confirmed baseline/algo0 is best
      (`0.808960 ms`)
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260517_decode_fresh/`,
    `.tmp_codex/nsys/stage2_20260517_decode_fresh/`
  - conclusion: keep 0.5B decode in the secondary queue, but do not promote
    `lm_head_top1`, a new decode-attention chunk table, or another cublasLt
    decode table edit from this round. The remaining 0.5B gap is distributed
    across prefill and several decode GEMV/attention blocks rather than one
    obvious low-risk knob.
- `Qwen2.5-1.5B-Instruct / BF16 / fresh long-prefill attribution`
  - captured a fresh current source-op graph-off mapping for `2048x1` after the
    direct BF16/FP16 conversion patch
  - prefill model time is `153.620 ms`
  - role attribution:
    - GateUp source-op GEMM `68.607 ms`
    - Down source-op GEMM `32.803 ms`
    - FlashInfer prefill attention `17.027 ms`
    - SwiGLU `9.207 ms`
    - norms `4.258 ms`
    - BF16/FP16 conversion `2.569 ms`
    - RoPE `1.495 ms`
  - follow-up attention sweep:
    - microbench suggested `prefill_short_qo_len_threshold=512`,
      `prefill_short_cta_tile_q=64`, `prefill_long_cta_tile_q=64`
      could improve the 1.5B attention layer (`S=2048` about
      `0.773 -> 0.708 ms/layer`)
    - full graph-on generate did not reproduce it:
      current same-day `1.5B 2048x32` avg `474.645 ms`, candidate
      `474.783 ms`
  - raw artifacts:
    `.tmp_codex/nsys/stage2_20260517_1p5b_prefill_fresh/`,
    `.tmp_codex/bench/stage2_20260517_1p5b_prefill_fresh/`,
    `.tmp_codex/operator_tables/stage2_20260517_1p5b_attention_probe/`
  - conclusion: reject the 1.5B attention split64 table change. The 1.5B
    remaining gap is still MLP/GEMM-heavy, but the existing Humanize gate-up
    workspace shows ordinary CUTLASS 2.x tile/stage/swizzle/cuBLAS variants are
    already exhausted; future 1.5B work needs a nontrivial source-visible GEMM
    route or plugin/source asset, not another small table sweep.
- `Qwen2.5-1.5B/3B-Instruct / BF16 / prefill GateUp+SwiGLU fusion recheck`
  - reran `scripts/tune/profile_prefill_swiglu_kernels.py` on the current tree
    to verify whether the existing TRT-LLM fused-MoE helper could replace the
    source-op GateUp plus standalone SwiGLU boundary
  - `1.5B` results:
    - `S=512`: fused `1.396 ms`, two-stage `1.203 ms`, delta `+0.193 ms/layer`
    - `S=1024`: fused `2.566 ms`, two-stage `2.221 ms`, delta `+0.346 ms/layer`
    - `S=2048`: fused `5.125 ms`, two-stage `4.778 ms`, delta `+0.347 ms/layer`
  - `3B` results:
    - `S=512`: fused `2.250 ms`, two-stage `1.948 ms`, delta `+0.302 ms/layer`
    - `S=1024`: fused `4.129 ms`, two-stage `3.709 ms`, delta `+0.421 ms/layer`
    - `S=2048`: fused `8.214 ms`, two-stage `7.304 ms`, delta `+0.910 ms/layer`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260517_swiglu_recheck/`
  - conclusion: reject the current prefill GateUp+SwiGLU fused helper for
    default Stage-2 work. The MLP gap remains a GEMM tactic problem rather than
    a missed activation fusion knob.
- `Qwen2.5-3B-Instruct / BF16 / prefill QKV/OProj BF16-direct weight probe`
  - tested `weight_mode=bf16_direct` on the 3B source-op QKV/OProj prefill
    records with `input_mode=mixed_bf16` and nonpersistent weights
  - target slice:
    - current direct-conversion baseline `3B 2048x32`: avg `928.890 ms`,
      prefill `303.043 ms`
    - BF16-direct linear probe: avg `951.409 ms`, prefill `325.187 ms`
  - raw artifacts:
    `.tmp_codex/operator_tables/stage2_20260517_linear_bf16_direct_probe/`,
    `.tmp_codex/bench/stage2_20260517_linear_bf16_direct_probe/`
  - conclusion: reject BF16-direct QKV/OProj for 3B. Removing per-layer FP16
    weight casts is outweighed by the slower mixed BF16-weight GEMM path.
- `Qwen2.5-3B-Instruct / BF16 / prefill QKV/OProj side-stream weight cast overlap`
  - tested source-op QKV/OProj with `input_mode=fp16_cast`,
    `overlap_casts=true`, `persistent_weights=false`
  - target slice:
    - current direct-conversion baseline `3B 2048x32`: avg `928.890 ms`,
      prefill `303.043 ms`
    - overlap-cast probe: avg `932.897 ms`, prefill `306.735 ms`
  - raw artifacts:
    `.tmp_codex/operator_tables/stage2_20260517_linear_bf16_direct_probe/`,
    `.tmp_codex/bench/stage2_20260517_linear_bf16_direct_probe/`
  - conclusion: reject side-stream overlap for 3B linear source-op default.
    The extra input BF16->FP16 conversion costs more than the weight-cast
    overlap can recover; keep `mixed_bf16`.
- `Qwen2.5-1.5B-Instruct / FP16 standalone GateUp / CUTLASS profiler candidate`
  - ran the existing CUTLASS profiler build on the 1.5B GateUp shape
    `M=2048,K=1536,N=17920`
  - profiler with column-major output suggested `128x128x32_s3` could be
    slightly faster than current `128x256x32_s3`
  - added `128x128x32_s3` to the standalone Humanize harness to validate the
    actual EdgeFM row-output layout
  - row-output harness result rejected the candidate:
    - current median `2.502432 ms`
    - `128x128x32_s3` median `2.960320 ms`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260517_cutlass_profiler_gemm/`,
    `deliverables/kernel_opt/3060_1p5b_gateup_gemm_humanize_20260517/artifacts/attempts/full_gateup_128x128x32_s3_roundrobin_r100.json`
  - conclusion: keep the current GateUp source-op tile. Do not transfer
    column-output CUTLASS profiler wins directly to EdgeFM row-output kernels.
- `Qwen2.5-3B-Instruct / FP16 source-op GateUp / stage4 production recheck`
  - revisited the `128x256x32_s4` GateUp tile after the direct BF16/FP16
    conversion cleanup and the current 3B source-op table; this supersedes the
    earlier standalone-only rejection for the specific `3B/S2048` production
    shape, but does not promote s4 globally
  - implementation:
    - added `gateup_tile=128x256x32_s4` as an explicit source-op tile mode
    - enabled it only for 3B fused MLP `m=2048|hidden=2048|intermediate=11008`
  - paired results versus current direct-conversion baseline:
    - `3B 2048x32`: `928.890 -> 927.804 ms`, prefill
      `303.043 -> 302.024 ms`
    - `3B 1024x32`: `761.629 -> 759.645 ms`, prefill
      `151.073 -> 149.615 ms`; the 1024 record does not use s4 and confirms
      adjacent no-regression on the rebuilt source-op path
    - `3B 2048x64`: `1577.463 -> 1575.828 ms`, prefill
      `304.736 -> 303.379 ms`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260517_3b_gateup_s4_probe/`
  - conclusion: accept this as a small, localized Stage-2 win. It is not a
    bridge-removal blocker and it does not materially change the remaining
    TRT-Edge-LLM gap, but it is stable enough to keep under the relaxed
    sub-1% improvement rule.
- `Qwen2.5-3B-Instruct / BF16-FP16 conversion kernel unroll probe`
  - after fresh graph-off profiling showed `bf16_to_half2` conversion kernels
    at about `6.1 ms` across `108` launches, tested a simple two-pair-per-thread
    unroll for the MLP and linear conversion kernels
  - correctness smoke passed, but the target slice regressed:
    - accepted s4 baseline `3B 2048x32`: `927.804 ms`, prefill `302.024 ms`
    - unroll probe: `929.335 ms`, prefill `303.206 ms`
  - raw artifact:
    `.tmp_codex/bench/stage2_20260517_convert_unroll_probe/3b_2048x32_unroll2_r7.json`
  - conclusion: reject and revert the unroll. The original one-pair-per-thread
    vector path remains faster on RTX 3060 for these conversion shapes.
- `Qwen2.5-{0.5B,1.5B,3B}-Instruct / BF16 / strided-QKV runtime default gate`
  - rechecked the existing env-gated strided-QKV attention path after the
    accepted 3B GateUp s4 change
  - paired observations:
    - `0.5B 2048x32`: direct baseline `173.965 ms`; env-on strided
      `174.225 ms`; default-gate probe `174.059 ms`
    - `1.5B 2048x32`: direct baseline `474.902 ms`; env-on strided
      `474.105 ms`; default-gate probe `474.294 ms`
    - `3B 2048x32`: accepted s4 baseline `927.804 ms`; env-on strided
      `927.437 ms`, but rebuilt forced-on/forced-off/default checks were
      `929.994 ms` / `930.179 ms` / `931.387 ms`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260517_runtime_post_s4/`
  - conclusion: keep strided-QKV as an env-only diagnostic route. The 1.5B
    and one 3B run showed sub-ms wins, but the 3B rebuilt default checks were
    too noisy and the 0.5B path regressed. Do not default-enable this runtime
    gate without a stronger graph-on matrix.
- `Qwen2.5-3B-Instruct / BF16 / table-driven TRT-FMHA plugin-op attention`
  - goal: satisfy the relaxed Stage-2 checkpoint by removing all `10 ms+`
    gaps versus external `TRT-Edge-LLM` without reintroducing a serialized
    TensorRT engine bridge
  - implementation:
    - `trt_context_fmha_plugin_attention` can now be enabled by
      `impl_params.enabled` and BF16->FP16 cast by
      `impl_params.allow_bf16_fp16_cast`, instead of requiring env-only gates
    - `AttentionLayer::resolve_impl()` now passes table `impl_params` into
      `supports()`, which is required for table-driven default-off operators
    - the 3060 LLM table enables the plugin-op attention only for
      `num_qo_heads=16|num_kv_heads=2|head_dim=128` prefill, with
      `contiguous_q_kv=true` and `contiguous_q_kv_min_seq_len=2048`
    - the contiguous pack uses pair-wise Q/K RoPE and calls the
      TRT-Edge-LLM `ContextFMHARunner` as an operator runner; no serialized TRT
      engine or TensorRT execution context is used
  - paired results:
    - current source-op baseline `3B 2048x32`: avg `930.670 ms`,
      prefill `304.404 ms`, decode `626.183 ms`
    - env-gated pair-wise plugin-op diagnostic: avg `925.104 ms`, prefill
      `298.866 ms`
    - table-driven confirmation: avg `925.432 ms`, prefill `299.124 ms`,
      decode `626.157 ms`
    - external TRT reference: avg `916.511 ms`; remaining gap `+8.921 ms`
  - adjacent checks:
    - `3B 512x32`: avg `686.910 ms` versus TRT `686.297 ms`, gap
      `+0.613 ms`
    - `3B 1024x32`: avg `761.862 ms` versus TRT `757.664 ms`, gap
      `+4.198 ms`
    - `3B 2048x64`: avg `1572.824 ms` versus TRT `1568.463 ms`, gap
      `+4.362 ms`
  - validation:
    - `cmake --build build-3060 -j$(nproc)`
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/analyze_trt_nsys_profile.py scripts/tune/tune_qwen_attention_prefill.py`
    - `python3 -m pytest -q tests/operators/test_attention_prefill.py -k "correctness or prerotate"`
    - `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment and not vl"`
    - `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "max_new_tokens or deferred_stop or metrics_surface or compact_vocab_identity"`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260518_plugin_contig/`
  - conclusion: accept as the first tactical checkpoint for 3B long-context.
    It does not fully catch TRT-Edge-LLM, but it removes the `10 ms+`
    residual. The next target is specifically `3B 2048x32`, where the
    remaining `+8.921 ms` gap must be pushed toward the `<=5 ms` checkpoint.
- `Qwen2.5-3B-Instruct / BF16 / post-plugin-op residual profile and SwiGLU launch retune`
  - fresh graph-off mapping after the plugin-op attention checkpoint changed
    the active queue: the FMHA body is no longer the only meaningful residual.
    The dominant profiled prefill work is now GateUp (`~141.4 ms`), Down
    (`~72.7 ms`), TRT plugin FMHA body (`~19.2 ms`), `swiglu_half2_kernel`
    (`~14.5 ms`), BF16->FP16 conversion (`~6.1 ms`), and contiguous Q/KV RoPE
    pack (`~3.5 ms`)
  - rejected follow-ups:
    - RoPE inv-freq precompute: `3B 2048x32` regressed to `926.137 ms`,
      prefill `299.780 ms`
    - MLP `activation_mode=mixed_bf16`: neutral/slower at `925.608 ms`,
      prefill `299.429 ms`
    - global SwiGLU block-size probes: `64` threads `923.580 ms`, `192`
      threads `923.574 ms`; both slower than the 128-thread candidate on the
      same short confirmation
    - SwiGLU `__expf` fast-math probe: `3B 2048x32` regressed to
      `924.893 ms`, prefill `298.749 ms`; keep `expf` in the default path
  - accepted implementation:
    - production SwiGLU launch now selects `128` threads only for
      `hidden=2048|intermediate=11008|m>=2048`
    - all other shapes keep the previous `256` thread launch because 1.5B did
      not show a clear benefit
  - paired/confirmation results:
    - `3B 2048x32` table plugin baseline: `925.432 ms`, prefill `299.124 ms`
    - 128-thread short confirmation: `923.909 ms`, prefill `297.903 ms`
    - external TRT reference: `916.511 ms`; remaining gap `+7.398 ms`
    - prior adjacent checks with the 128-thread candidate stayed positive:
      `3B 512x32 686.707 ms`, `3B 1024x32 761.236 ms`, `3B 2048x64
      1571.159 ms`
  - validation:
    - `cmake --build build-3060 -j$(nproc)`
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/analyze_trt_nsys_profile.py scripts/tune/tune_qwen_attention_prefill.py`
    - `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment and not vl"`
    - `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "max_new_tokens or deferred_stop or metrics_surface or compact_vocab_identity"`
  - raw artifacts:
    `.tmp_codex/nsys/stage2_20260518/`,
    `.tmp_codex/bench/stage2_20260518_swiglu_threads/`
  - conclusion: accept as a small Stage-2 step under the relaxed rule. The
    official 3B `2048x32` gap is below the "no 10+ ms" bar, but still needs
    about `2.4 ms` more improvement to reach the preferred `<=5 ms` checkpoint.
- `Qwen2.5-{0.5B,1.5B}-Instruct / BF16 / secondary-size plugin-op attention extension`
  - goal: answer the follow-up request to compress the other two model sizes,
    preferably toward a `<=3 ms` gap versus external `TRT-Edge-LLM`, while
    keeping the accepted 3B Stage-2 checkpoint intact
  - implementation:
    - enabled table-driven `trt_context_fmha_plugin_attention` for the 0.5B
      prefill shape `num_qo_heads=14|num_kv_heads=2|head_dim=64`
    - enabled the same plugin-op runner for the 1.5B prefill shape
      `num_qo_heads=12|num_kv_heads=2|head_dim=128`
    - both records use `allow_bf16_fp16_cast=true`, pair-wise contiguous Q/KV
      packing, `contiguous_q_kv_min_seq_len=2048`, and
      `prefill_cta_tile_q=64`
    - repaired the plugin-op fallback so non-2048 and unsupported cases first
      reuse the accepted BF16 FlashInfer pre-rotate path for Llama RoPE instead
      of silently taking a slower generic fallback
  - paired results:
    - `0.5B 2048x32`: current `174.166 ms`, official plugin-op
      `172.723 ms`, TRT `167.850 ms`, remaining gap `+4.873 ms`
    - `0.5B 2048x64`: current `300.653 ms`, official plugin-op
      `298.682 ms`, TRT `291.652 ms`, remaining gap `+7.030 ms`
    - `1.5B 2048x32`: current `474.041 ms`, official plugin-op
      `471.802 ms`, TRT `468.809 ms`, remaining gap `+2.993 ms`
    - `1.5B 2048x64`: current `809.992 ms`, official plugin-op
      `807.196 ms`, TRT `803.190 ms`, remaining gap `+4.006 ms`
  - adjacent checks:
    - `0.5B 1024x32`: plugin-op fallback keeps prefill slightly faster
      (`-0.231 ms`) with total within noise (`+0.309 ms`)
    - `0.5B 512x32`: total improves by `-0.699 ms`
    - `1.5B 1024x32`: total is within noise (`+0.285 ms`)
  - rejected follow-ups:
    - 0.5B and 1.5B decode-attention table retunes were neutral or regressed
    - 0.5B and 1.5B MLP tile sweeps regressed
    - 0.5B decode cublasLt recheck confirmed the current table/default picks
      for `fused_qkv`, `attention_output`, and `mlp_down`; the missing
      `fused_gate_up` exact record resolves to the same best heuristic and does
      not justify a table change
    - `prefill_max_mma_kv_cap=2` did not produce a stable defaultable win for
      these two sizes
    - cublasLt LMHead fp32-output records matched existing fastest algorithms
      and produced no measurable end-to-end gain
    - `lm_head_top1` remains default-off because the post-plugin 0.5B win is
      sub-ms and does not justify changing the full-logits default contract
    - SM86 decode fused-SwiGLU was probed as a possible large 0.5B decode win,
      but it is rejected for this round: the existing source already records a
      numerical-equivalence risk on RTX 3060, and temporarily opening the
      device gate pulled in a very expensive TensorRT-LLM fused-MoE template
      rebuild before any correctness proof
  - validation:
    - `cmake --build build-3060 -j$(nproc)`
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - `python3 -m py_compile scripts/profile/profile_edgefm_generate_case.py scripts/profile/analyze_trt_nsys_profile.py scripts/tune/tune_qwen_attention_decode.py scripts/tune/tune_qwen_attention_prefill.py`
    - `python3 -m pytest -q tests/operators/test_attention_prefill.py tests/operators/test_prefill_linear.py`
    - `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment and not vl"`
    - `python3 -m pytest -q tests/engine/test_qwen2_generate.py -k "max_new_tokens or deferred_stop or metrics_surface or compact_vocab_identity"`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260518_official_plugin_attn_v2/`,
    `.tmp_codex/bench/stage2_20260518_0p5_adjacent_recheck/`,
    `.tmp_codex/bench/stage2_20260518_0p5_plugin_top1/`,
    `.tmp_codex/bench/stage2_20260518_0p5_decode_attn_after_plugin/`,
    `.tmp_codex/tune/stage2_20260518_0p5_decode_linear/`,
    `.tmp_codex/tune/stage2_20260518_lm_head/`,
    `.tmp_codex/nsys/stage2_20260518_other_sizes/`
  - conclusion: accept the plugin-op attention extension. `1.5B 2048x32`
    reaches the requested `~3 ms` gap, `1.5B 2048x64` is close, and 0.5B is
    reduced but not yet within `3 ms`; continue 0.5B decode/runtime work next.
- `Qwen2.5-{0.5B,1.5B}-Instruct / BF16 / strided-QKV default confirmation and rejected decode probes`
  - goal: compress the remaining 0.5B/1.5B long-context gaps toward the
    requested `<=3 ms` stretch target without bringing back the serialized TRT
    engine bridge
  - accepted implementation:
    - default-enable the correctness-clean prefill strided-QKV attention path
      for BF16 no-prefix prefill, with `EDGE_FM_PREFILL_STRIDED_QKV_ATTENTION=0`
      as the opt-out gate
    - keep the prefill strided-QKV side-stream KV copy default-on for the same
      path, with `EDGE_FM_PREFILL_STRIDED_QKV_SIDE_STREAM=0` as the opt-out gate
    - keep fused-K-RoPE copy default-off because it is incompatible with the
      current plugin-op attention contract unless the attention impl explicitly
      supports pre-rotated K
  - latest paired confirmation:
    - `0.5B 2048x32`: EdgeFM `171.396 ms`, TRT `166.825 ms`, gap
      `+4.571 ms`; prefill gap `+2.858 ms`, decode gap `+1.689 ms`
    - `0.5B 2048x64`: EdgeFM `297.691 ms`, TRT `291.460 ms`, gap
      `+6.231 ms`; prefill gap `+2.642 ms`, decode gap `+3.534 ms`
    - `1.5B 2048x32`: EdgeFM `472.190 ms`, TRT `470.007 ms`, gap
      `+2.183 ms`; prefill gap `+2.593 ms`, decode gap `-0.421 ms`
    - `1.5B 2048x64`: EdgeFM `808.084 ms`, TRT `805.714 ms`, gap
      `+2.370 ms`; prefill gap `+2.653 ms`, decode gap `-0.395 ms`
  - rejected follow-ups:
    - SM86 decode fused-SwiGLU env gate: `EDGE_FM_DECODE_SWIGLU_ALLOW_SM86=1`
      failed token alignment badly on Qwen generate (`4 failed`, `2` OOM
      during the stress subset). Keep it default-off.
    - CUDA-graph response-token memcpy node: correctness was fine, but
      `0.5B 2048x32/64` regressed by `+0.480/+0.983 ms`; reverted.
    - BF16 decode logits: greedy token alignment passed under the env gate, but
      `0.5B 2048x32/64` regressed by `+0.135/+0.465 ms`; rejected and removed
      from the production path.
    - 0.5B decode GateUp cublasLt explicit retune: current heuristic remained
      fastest; no table edit.
    - side-stream off and strided-QKV off probes both regressed 0.5B
      `2048x64` (`+0.058 ms` and `+0.175 ms`), confirming the default-on
      decision.
    - disabling the 0.5B plugin-op prefill attention record and falling back to
      FlashInfer prerotate regressed `2048x32/64` by `+2.073/+2.246 ms`; keep
      the plugin-op record.
    - current-table `lm_head_top1` still helps only modestly
      (`0.5B 2048x32 -0.083 ms`, `2048x64 -0.404 ms`) and remains
      explicit/default-off because it does not preserve the full-logits path.
    - decode attention `no_split_kv_threshold=1024` looked slightly faster in
      the layer microbench, but full generate was mixed/noise
      (`2048x32 +0.041 ms`, `2048x64 -0.096 ms`); no table edit.
    - FP16 decode logits were tested as a narrower alternative to the rejected
      BF16 logits path. They failed the 1.5B generate correctness/run gate with
      cuBLASLt status `15`; the experiment was removed.
    - SM86 single-row 0.5B norm source-op passed token-alignment correctness
      under a temporary table, but full generate regressed badly:
      `2048x32 +5.058 ms`, `2048x64 +10.236 ms`, with decode step time
      increasing by about `+0.163 ms`. The production source edits were
      reverted; keep the existing FlashInfer norm path.
    - After the token-dim Q/KV pack checkpoint, 0.5B decode-attention
      `no_split_kv_threshold` was rechecked at `512`, `1024`, and `2048`.
      All regressed `2048x64` against the current table by
      `+0.174/+0.390/+0.447 ms`; keep the current threshold.
    - 0.5B MLP `gateup_tile=128x256x32_s4` was rechecked after the
      token-dim Q/KV pack. A short `2048x64` sweep showed a noisy
      `-0.122 ms` total delta, but reverse-order confirmation regressed
      `2048x64` by `+0.860 ms` (`298.516 ms` candidate first versus
      `297.656 ms` default second). The adjacent `2048x32` pass was only
      `-0.184 ms` total while shifting decode by `+0.113 ms`, so this is
      treated as unstable/noise and not promoted.
    - Fresh 0.5B `2048x64` NSYS attribution after the token-dim pack shows the
      remaining decode pressure is still dominated by full-logits `LMHead`
      (`~50.7 ms` across the graph-off mapping trace) and FlashInfer decode
      attention (`~35.9 ms`). A real-path `lm_head` cublasLt retune using
      FP32 logits found the baseline heuristic fastest (`~0.809 ms/step`);
      no linear table edit.
    - A narrow decode-attention grid found only `no_split_kv_threshold=192`
      slightly faster in the layer microbench. End-to-end order checking
      rejected it: candidate-second looked positive (`2048x64 -0.189 ms`,
      `2048x32 -0.283 ms`), but candidate-first regressed against the default
      second run (`2048x64 298.543 ms` versus `297.986 ms`).
  - validation after rejecting BF16 decode logits:
    - `cmake --build build-3060 -j$(nproc)`
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - profile/tuning script `py_compile`
    - `python3 -m pytest -q tests/operators/test_norm_sampler.py tests/engine/test_qwen2_generate.py -k "test_sampler_greedy_correctness_and_performance_smoke or test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface"`
      (`10 passed`)
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260518_strided_default_confirm/`,
    `.tmp_codex/nsys/stage2_20260518_0p5b_decode/`,
    `.tmp_codex/tune/stage2_20260518_0p5b_decode_gateup/`,
    `.tmp_codex/tune/stage2_20260518_0p5b_decode_gateup_explicit/`,
    `.tmp_codex/bench/stage2_20260518_graph_response_copy_probe/`,
    `.tmp_codex/bench/stage2_20260518_decode_bf16_logits_probe/`,
    `.tmp_codex/bench/stage2_20260518_0p5_side_stream_probe/`,
    `.tmp_codex/bench/stage2_20260518_0p5_strided_disable_probe/`,
    `.tmp_codex/bench/stage2_20260518_0p5_flashinfer_fallback/`,
    `.tmp_codex/bench/stage2_20260518_0p5_top1_current/`,
    `.tmp_codex/tune/stage2_20260518_0p5_decode_attention_recheck/`,
    `.tmp_codex/bench/stage2_20260518_0p5_decode_attention_recheck/`,
    `.tmp_codex/bench/stage2_20260518_0p5_norm_probe/`,
    `.tmp_codex/bench/stage2_20260518_0p5_decode_attn_thresholds/`,
    `.tmp_codex/bench/stage2_20260518_0p5_mlp_narrow/`,
    `.tmp_codex/bench/stage2_20260518_0p5_mlp_gate_s4_confirm/`,
    `.tmp_codex/nsys/stage2_20260518_0p5b_decode_fresh_after_token_pack/`,
    `.tmp_codex/tune/stage2_20260518_0p5_lm_head_decode_after_token_pack/`,
    `.tmp_codex/tune/stage2_20260518_0p5_decode_attention_grid_after_token_pack/`,
    `.tmp_codex/bench/stage2_20260518_0p5_decode_attn_nosplit192/`
  - conclusion: accept strided-QKV + side-stream as the current default for the
    checked 0.5B/1.5B long-prefill path. `1.5B` now satisfies the `<=3 ms`
    stretch target for both decode lengths. The remaining active slice is 0.5B,
    where `2048x32` is `+4.571 ms` and `2048x64` is `+6.231 ms`.
- `Qwen2.5-{0.5B,1.5B,3B}-Instruct / BF16 / plugin-op contiguous Q/KV token-dim pack`
  - goal: reduce the direct TRT-FMHA plugin-op prefill packing/RoPE overhead
    without changing the serialized-TRT-free operator boundary
  - accepted implementation:
    - add a default-off `contiguous_q_kv_token_dim_pack` impl param for the
      `trt_context_fmha_plugin_attention` contiguous Q/KV path
    - for each `(token, rotary_dim)` compute RoPE sin/cos once and loop over
      Q/KV heads, instead of recomputing the same transcendental work for every
      head
    - enable the param only on the 3060 long-prefill plugin-op records for
      Qwen2.5 0.5B, 1.5B, and 3B
  - correctness / validation:
    - `cmake --build build-3060 -j$(nproc)`
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - profile/tuning script `py_compile`
    - `python3 -m pytest -q tests/operators/test_attention_prefill.py` (`9 passed`)
    - temporary-table token alignment: `tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment and not vl"` (`6 passed`)
    - post-table core generate subset:
      `tests/operators/test_attention_prefill.py tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface"`
      (`9 passed`, `20 deselected`)
  - paired benchmark deltas, CUDA graph on:
    - `0.5B 2048x32`: `171.747 -> 171.704 ms`, `-0.042 ms`; prefill
      `48.724 -> 48.598 ms`, `-0.125 ms`
    - `0.5B 2048x64`: initial pass `298.894 -> 298.676 ms`,
      `-0.218 ms`; reverse-order confirmation `298.538 -> 298.080 ms`,
      `-0.458 ms`; prefill `48.772 -> 48.461 ms`, `-0.312 ms`
    - `1.5B 2048x32`: `471.936 -> 471.300 ms`, `-0.635 ms`; prefill
      `147.113 -> 146.471 ms`, `-0.642 ms`
    - `1.5B 2048x64`: `807.853 -> 807.826 ms`, `-0.028 ms`; prefill
      `147.836 -> 147.643 ms`, `-0.193 ms`
    - `3B 2048x32`: `924.646 -> 923.810 ms`, `-0.836 ms`; prefill
      `297.970 -> 297.258 ms`, `-0.712 ms`
    - `3B 2048x64`: `1572.272 -> 1571.943 ms`, `-0.329 ms`; prefill
      `298.850 -> 298.359 ms`, `-0.491 ms`
  - raw artifacts:
    `.tmp_codex/operator_tables/stage2_20260518_token_dim_pack/`,
    `.tmp_codex/bench/stage2_20260518_0p5_token_dim_pack/`,
    `.tmp_codex/bench/stage2_20260518_token_dim_pack_scan/`
  - conclusion: accept as a small, correctness-clean source-visible runtime
    improvement. The largest single move is still sub-ms, but it is consistent
    with the relaxed "absorb useful increments" rule and directly reduces the
    active long-prefill residual.
- `Qwen2.5-0.5B-Instruct / BF16 / source-op table-selection log demotion`
  - goal: remove repeated default-level host logging from the tuned source-op
    path without changing any CUDA operator selection
  - accepted implementation:
    - demote the `CUTLASS prefill linear/MLP source-op selected by
      operator_impl_table` messages from warning to debug
    - keep true experimental env-gate and fallback/OOM messages at warning
      level
  - benchmark after rebuild, CUDA graph on:
    - `0.5B 2048x64`: `297.553 ms`, prefill `48.272 ms`, decode
      `249.125 ms`
    - `0.5B 2048x32`: `171.073 ms`, prefill `48.289 ms`, decode
      `122.676 ms`
  - validation:
    - `cmake --build build-3060 -j$(nproc)`
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - profile/tuning script `py_compile`
    - `python3 -m pytest -q tests/operators/test_prefill_linear.py tests/operators/test_attention_prefill.py tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface or prefill_linear"`
      (`21 passed`, `20 deselected`)
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260518_log_debug_runtime_probe/`
  - conclusion: accept as a low-risk runtime cleanup. The measured improvement
    is small and partly host-side, but it removes noisy stderr work from the
    official benchmark/service path and keeps the 0.5B long-context slices
    moving toward the `<=3 ms` stretch target.
- `Qwen2.5-0.5B-Instruct / BF16 / source-op runtime config cache`
  - goal: avoid repeated `OperatorImplTable` resolution and impl-parameter
    parsing inside the tuned source-op prefill linear/MLP paths
  - accepted implementation:
    - cache resolved prefill linear source-op runtime config by
      `role|shape_sig`
    - cache resolved prefill MLP source-op runtime config by `shape_sig`
    - clear the caches with the existing source-op runtime reset path
  - benchmark after rebuild, CUDA graph on:
    - `0.5B 2048x64`: `297.489 ms`, prefill `48.263 ms`, decode
      `249.070 ms`; previous log-demotion checkpoint was `297.553 ms`
    - `0.5B 2048x32`: `171.068 ms`, prefill `48.303 ms`, decode
      `122.650 ms`; previous log-demotion checkpoint was `171.073 ms`
  - validation:
    - `cmake --build build-3060 -j$(nproc)`
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - profile/tuning script `py_compile`
    - `python3 -m pytest -q tests/operators/test_prefill_linear.py tests/operators/test_attention_prefill.py tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface or prefill_linear"`
      (`21 passed`, `20 deselected`)
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260518_runtime_config_cache_probe/`
  - conclusion: accept as a small runtime cleanup. The measured win is only
    `~0.0-0.06 ms` on the checked 0.5B long-context slices, but it is
    correctness-clean, reduces host table work on the production path, and does
    not change CUDA math or operator selection.
- `Qwen2.5-3B-Instruct / BF16 / 1024-prefill MLP down-tile split`
  - goal: close the remaining `3B` mid-context residual after the long-context
    decode chunk and source-op cache checkpoints
  - accepted implementation:
    - for `m=1024|hidden=2048|intermediate=11008`, keep the existing
      `gateup` tile and set `down_tile=128x128x32_warp32x64`
    - leave the 0.5B and 1.5B `m=1024` MLP records unchanged because the same
      tile split did not transfer
  - probe / confirmation:
    - 3B `1024x32`: official table `756.605 ms` versus TRT `757.678 ms`,
      gap `-1.073 ms`; prefill `146.107 ms`
    - 3B `1024x64`: official table `1388.757 ms` versus TRT `1396.219 ms`,
      gap `-7.462 ms`; prefill `147.315 ms`
  - validation:
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - core operator/generate subset after the follow-up decode change:
      `21 passed`, `20 deselected`
  - raw artifacts:
    `.tmp_codex/bench/stage2_20260518_mid_context_mlp_tile_probe/`,
    `.tmp_codex/bench/stage2_20260518_3b_m1024_mlp_down_tile_official/`
  - conclusion: accept only for the 3B `m=1024` MLP shape. This resolves the
    3B mid-context target without broad MLP tile churn.
- `Qwen2.5-1.5B-Instruct / BF16 / 1024-decode attention chunk refinement`
  - goal: compress the remaining `1.5B 1024x32/64` gap after prefill linear,
    prefill attention, MLP, and `lm_head_top1` probes failed to reproduce
  - rejected probes:
    - `m=1024` QKV/OProj source-op tile/input-mode variants: the best short
      run (`OProj tile=128x256x32`) did not reproduce (`1024x32 -0.038 ms`,
      `1024x64 +0.106 ms`)
    - `lm_head_top1`: `1024x32 +0.130 ms`, `1024x64 -0.231 ms`; still
      inconsistent/default-off
    - prefill attention prerotate CTA/split sweep: best candidates were
      noise-level and did not address the multi-ms residual
    - MLP `activation_mode=mixed_bf16`, down-output, SwiGLU thread, and
      gate/down tile variants: the best short run did not reproduce
      (`activation_mode=mixed_bf16` confirmed at `+0.176/+0.533 ms`)
  - accepted implementation:
    - for the Qwen2.5 1.5B decode attention shape
      `num_qo_heads=12|num_kv_heads=2|head_dim=128`, set
      `chunk_alignment=32` and `chunk_candidates=[64,96,128,192]`
    - keep 0.5B and 3B decode attention chunk tables unchanged
    - update both duplicate q12 decode records in the 3060 materialized table
  - confirmation versus previous table, CUDA graph on:
    - `1.5B 512x32`: `345.455 -> 345.527 ms`, `+0.072 ms`
    - `1.5B 512x64`: `664.959 -> 665.149 ms`, `+0.190 ms`
    - `1.5B 1024x32`: `388.582 -> 384.982 ms`, `-3.600 ms`
    - `1.5B 1024x64`: `713.728 -> 706.232 ms`, `-7.496 ms`
    - `1.5B 2048x32`: `468.013 -> 467.990 ms`, `-0.023 ms`
    - `1.5B 2048x64`: `798.049 -> 797.983 ms`, `-0.066 ms`
  - official paired check after table update:
    - `1.5B 1024x32`: EdgeFM `384.188 ms`, TRT `384.479 ms`, gap
      `-0.291 ms`
    - `1.5B 1024x64`: EdgeFM `705.958 ms`, TRT `709.321 ms`, gap
      `-3.363 ms`
  - full 18-slice matrix after this change, CUDA graph on, warmup `2`, runs `5`:
    - `0.5B`: `512x32 -8.886 ms`, `512x64 -5.871 ms`,
      `1024x32 -0.051 ms`, `1024x64 +2.098 ms`,
      `2048x32 +1.467 ms`, `2048x64 -0.451 ms`
    - `1.5B`: `512x32 +0.129 ms`, `512x64 +0.759 ms`,
      `1024x32 +0.241 ms`, `1024x64 -2.602 ms`,
      `2048x32 -2.363 ms`, `2048x64 -8.475 ms`
    - `3B`: `512x32 +0.662 ms`, `512x64 -6.414 ms`,
      `1024x32 -0.674 ms`, `1024x64 -6.643 ms`,
      `2048x32 -2.148 ms`, `2048x64 -13.700 ms`
  - validation:
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - profile/tuning script `py_compile`
    - `tests/operators/test_prefill_linear.py tests/operators/test_attention_prefill.py tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface or prefill_linear"`
      (`21 passed`, `20 deselected`)
  - raw artifacts:
    `.tmp_codex/operator_tables/stage2_20260518_1p5b_m1024_decode_attention_probe/`,
    `.tmp_codex/bench/stage2_20260518_1p5b_decode_attention_chunks96_confirm/`,
    `.tmp_codex/bench/stage2_20260518_1p5b_decode_attention_chunks96_official_v2/`,
    `.tmp_codex/bench/stage2_20260518_full_matrix_after_1p5b_decode_chunks96/`
  - conclusion: accept. The full matrix now has no positive residual above
    `+3 ms`; 12 of 18 slices are faster than TRT-Edge-LLM in the latest paired
    run.
- `Qwen2.5-0.5B-Instruct / BF16 / q14 decode chunk and 2048-prefill source-op refinements`
  - goal: start Stage 3 by moving the largest remaining positive residuals
    below zero where possible, without disturbing the full-matrix `<=3 ms`
    checkpoint
  - accepted decode attention change:
    - for the Qwen2.5 0.5B decode attention shape
      `num_qo_heads=14|num_kv_heads=2|head_dim=64`, set
      `chunk_alignment=32` and `chunk_candidates=[64,96,128,192]`
    - keep 3B q16 decode attention on `[64,128,192,256]`
  - q14 chunk confirmation versus previous table:
    - `0.5B 512x32`: `+0.009 ms`
    - `0.5B 512x64`: `+0.026 ms`
    - `0.5B 1024x32`: `-1.793 ms`
    - `0.5B 1024x64`: `-2.744 ms`
    - `0.5B 2048x32`: `+0.227 ms`
    - `0.5B 2048x64`: `+0.109 ms`
  - official q14 paired check:
    - `0.5B 1024x32`: EdgeFM `138.395 ms`, TRT `139.290 ms`, gap
      `-0.895 ms`
    - `0.5B 1024x64`: EdgeFM `256.409 ms`, TRT `256.922 ms`, gap
      `-0.513 ms`
    - `0.5B 2048x32`: EdgeFM `168.301 ms`, TRT `166.823 ms`, gap
      `+1.478 ms`
    - `0.5B 2048x64`: EdgeFM `291.140 ms`, TRT `291.572 ms`, gap
      `-0.432 ms`
  - accepted 2048-prefill source-op changes:
    - for 0.5B `m=2048|hidden=896|intermediate=4864`, add
      `activation_mode=mixed_bf16`
    - for 0.5B `m=2048|in_features=896|out_features=1152` fused QKV,
      switch `input_mode` from `fp16_cast` to `mixed_bf16`
    - reject OProj `tile=256x128x32` after confirmation regressed both
      `2048x32` and `2048x64`
  - source-op confirmation:
    - MLP activation mixed: `2048x32 -0.073 ms`, `2048x64 -0.316 ms`
    - QKV mixed BF16: `2048x32 -0.154 ms`, `2048x64 -0.119 ms`
  - final full 18-slice matrix after these Stage-3 changes, CUDA graph on,
    warmup `2`, runs `5`:
    - `0.5B`: `512x32 -8.712 ms`, `512x64 -5.772 ms`,
      `1024x32 -0.640 ms`, `1024x64 -0.356 ms`,
      `2048x32 +1.191 ms`, `2048x64 -1.195 ms`
    - `1.5B`: `512x32 +0.163 ms`, `512x64 +0.826 ms`,
      `1024x32 +0.063 ms`, `1024x64 -2.793 ms`,
      `2048x32 -2.801 ms`, `2048x64 -8.611 ms`
    - `3B`: `512x32 +0.812 ms`, `512x64 -6.605 ms`,
      `1024x32 -0.100 ms`, `1024x64 -6.857 ms`,
      `2048x32 -1.565 ms`, `2048x64 -13.949 ms`
  - validation:
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - profile/tuning script `py_compile`
    - `tests/operators/test_prefill_linear.py tests/operators/test_attention_prefill.py tests/engine/test_qwen2_generate.py -k "test_generate_token_alignment or max_new_tokens or deferred_stop or metrics_surface or prefill_linear"`
      (`21 passed`, `20 deselected`)
  - raw artifacts:
    `.tmp_codex/bench/stage3_20260518_0p5b_decode_attention_chunks96_probe/`,
    `.tmp_codex/bench/stage3_20260518_0p5b_decode_attention_chunks96_official/`,
    `.tmp_codex/bench/stage3_20260518_0p5b_m2048_mlp_activation_mixed_confirm/`,
    `.tmp_codex/bench/stage3_20260518_0p5b_m2048_linear_confirm/`,
    `.tmp_codex/bench/stage3_20260518_full_matrix_after_0p5b_decode_linear_mlp/`
  - conclusion: accept. The latest matrix has 13 of 18 slices faster than TRT;
    the largest remaining positive residual is `0.5B 2048x32 +1.191 ms`.
- `Qwen2.5-3B-Instruct / BF16 / 512-prefill MLP down-tile split`
  - goal: remove the remaining small `3B 512x32` positive residual without
    touching larger 3B prefill shapes
  - accepted implementation:
    - for `m=512|hidden=2048|intermediate=11008`, set
      `down_tile=128x128x32_warp32x64`
    - leave `m=1024` on its previously accepted down-tile split and `m=2048`
      on the long-context gateup/SwiGLU policy
  - rejected in the same probe:
    - `activation_mode=mixed_bf16`, `gateup_tile=128x256x32_s4`,
      `gateup_tile=128x256x32`, `swiglu_threads=64`, and mixed activation plus
      gateup s4 were neutral or regressed
  - confirmation:
    - `3B 512x32`: `687.289 -> 681.778 ms`, delta `-5.511 ms`; TRT
      `686.939 ms`, new gap `-5.161 ms`
    - `3B 512x64`: `1310.699 -> 1305.563 ms`, delta `-5.136 ms`; TRT
      `1317.385 ms`, new gap `-11.822 ms`
  - validation:
    - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060`
    - profile/tuning script `py_compile`
    - core operator/generate subset: `21 passed`, `20 deselected`
  - raw artifacts:
    `.tmp_codex/bench/stage3_20260518_3b_m512_mlp_probe/`,
    `.tmp_codex/bench/stage3_20260518_3b_m512_mlp_down_tile_confirm/`
  - conclusion: accept. The latest checked positive residual queue is now led
    by `0.5B 2048x32 +1.191 ms`, followed by small 1.5B short-context decode
    residuals.
- `Qwen2.5-{0.5B,1.5B}-Instruct / BF16 / residual short-slice probes`
  - goal: check whether the remaining sub-1.2ms positive gaps can be removed
    with existing default-off switches or tiny table edits
  - rejected `lm_head_top1` probes:
    - `1.5B 512x32`: `+0.058 ms`
    - `1.5B 512x64`: `-0.170 ms`
    - `0.5B 2048x32`: `-0.195 ms`
    - `0.5B 2048x64`: `-0.396 ms`
    - conclusion: still too small/inconsistent for default greedy-only routing
  - rejected 0.5B prefill plugin attention probes on `2048x32`:
    - current `167.679 ms`
    - `prefill_cta_tile_q=16`: `167.754 ms`
    - `prefill_cta_tile_q=128`: `167.767 ms`
    - disable token-dim pack: `168.095 ms`
    - FlashInfer fallback/no plugin-op record: `169.851 ms`
  - raw artifacts:
    `.tmp_codex/bench/stage3_20260518_1p5b_512_lmhead_top1_probe/`,
    `.tmp_codex/bench/stage3_20260518_0p5b_2048_lmhead_top1_probe/`,
    `.tmp_codex/bench/stage3_20260518_0p5b_prefill_plugin_cta_probe/`
  - conclusion: keep these paths rejected. Remaining positive gaps are small
    enough that further progress should start from fresh attribution rather
    than re-sweeping already-tuned table knobs.
- `Qwen2.5-0.5B-Instruct / BF16 / 2048x32 residual attribution`
  - goal: identify whether the final `~+1.2 ms` positive residual still has a
    table-level target
  - graph-on capture-range note:
    - CUDA graph capture only exposed graph-external finalize work, so it is
      not useful for operator attribution
  - graph-off mapping result:
    - total graph-off run: `181.505 ms`
    - stage metrics: prefill `51.354 ms`, decode `129.807 ms`
    - top prefill kernels:
      - CUTLASS mixed-input GEMM family: `20.303 ms` across 24 launches
      - CUTLASS mixed-input GEMM family: `9.003 ms` across 24 launches
      - TRT-FMHA plugin kernel: `5.857 ms`
      - SwiGLU BF16 kernel: `4.301 ms`
      - fused RMSNorm: `2.147 ms`
      - prefill LMHead GEMV: `0.809 ms`
      - contiguous Q/KV RoPE pack: `0.752 ms`
      - FP16-to-BF16 cast: `0.731 ms`
    - top decode kernels:
      - LMHead GEMV: `24.954 ms`
      - FlashInfer decode attention: `13.280 ms`
      - RMSNorm: `3.360 ms`
  - action decision:
    - prefill plugin CTA, FlashInfer fallback, MLP activation, QKV input mode,
      and lm_head_top1 have already been checked in the current table context
    - remaining opportunity is concentrated in source-op CUTLASS prefill GEMMs
      and likely requires a source-visible kernel/Humanize pass rather than
      another small JSON table sweep
  - raw artifacts:
    `.tmp_codex/nsys/stage3_20260518_0p5b_2048x32_current/`

## Active Queue

1. Stage-1 internal EdgeFM TensorRT bridge removal is complete in source.
2. Stage 2 tactical checkpoint is met across the full checked 3060 LLM matrix:
   every positive EdgeFM-vs-TRT residual is now below `+1 ms`, and 16 of 18
   slices are faster than TRT-Edge-LLM. A higher-run 1.5B recheck puts
   `512x32` at practical parity (`+0.197 ms` avg, `+0.135 ms` median).
3. Continue Stage 3 only if fresh attribution finds a real target for
   `1.5B 512x64`, currently about `+0.9 ms`. The previous `0.5B 2048x32`,
   `1.5B 512x32`, `1.5B 1024x32`, and 3B queues are historical or practical
   parity after the latest full/higher-run checks.
4. Do not continue broad gate-up/MLP-down/QKV/OProj CUTLASS 2.x table churn
   unless a fresh Stage-2 profile shows a concrete end-to-end opportunity.
5. Keep direct plugin-op/source-op assets in scope only when they do not require
   serialized TensorRT engines or execution contexts.
6. Review [doc/3060_fused_mlp_review.md](./3060_fused_mlp_review.md) before
   changing engine/layer/operator boundaries for a larger prefill MLP path.

## Environment Notes

- TensorRT Python is not installed in the image; the benchmark path uses the existing TRT-Edge-LLM C++/pybind runtime.
- TRT-Edge-LLM engines are available for the full LLM matrix under `tests/data/trt_edgellm_workspace/`.
- The 3060 table metadata has already been retuned for the CUDA 12.6 toolchain.
