# Qwen3.5 Phase 2 Current State

- Updated: `2026-05-23T18:36:00+08:00`
- Scope: EdgeFM-only optimization. TRT comparison remains blocked until a TensorRT-Edge-LLM Qwen3.5 linear-attention port exists.
- Scope status: Iter121 Qwen3.5 decode GateUp+SwiGLU warp GEMV kernel accepted after standalone direction evidence, production build, operator-table validation, Qwen3.5 runtime ops, fresh-dump token alignment, and 0.8B/2B graph-on gates. Iter122 profiles that new top visible hotspot and records it at the memory ceiling (`95.34%` DRAM throughput). Iter117 records a LMHead exact top1 byte-reduction blocker: simple row-norm/Cauchy pruning skips `0 / 248320` rows even with the final true best logit. Iter118 rejects a standalone LMHead BF16x2/launch-shape sweep: all variants are exact, but the best path is only `~0.2%` faster than scalar24 on 0.8B/2B, so it is not worth production migration. Iter119 rechecks the existing SM86 decode fused SwiGLU path for 0.8B: token alignment passes, but graph-on latency is unchanged (`189.794 -> 189.778 ms`) and 2B remains a known token-failure blocker, so no production branch is added. Iter120 rejects the existing CUTLASS prefill SwiGLU env path because it is token-safe but slower.
- Long-loop status: closure pass complete. KernelPilot knowledge review confirms the remaining high-value paths are compressed/quantized GEMV, a new exact LMHead indexing/search method, larger fused-projection design, or runtime/scheduler redesign. Those require a separate algorithm/precision contract, so this phase should converge and be committed.
- Current code path: Qwen3.5 CUDA graph enabled; greedy temperature=0 uses Iter93 default `lm_head_top1` unless runtime explicitly disables it. GatedDeltaNet prefill uses q/k precompute, decay precompute, and Iter105 precomputed recurrent sequence `value_tile=32`, `threads=256` with shared-memory q/k staging, qmem-output algebra, and vectorized state decay/update; decode uses Iter112 fused `conv1d kernel4 state update + gated_delta_sequence_from_ab + gated_rmsnorm` fast path for `seq_len==1,key_dim=value_dim=128,kernel=4`, falls back to Iter111/Iter91 paths for unsupported decode shapes, Iter78 static full-tile fallback for 128/128 non-single-token shapes, and the Iter64 generic `from_ab` kernel fallback for other shapes. Decode cublasLt records now include Iter107 Qwen3.5/RTX3060 2B `mlp_down` `6144->2048` (`algo_index=1`) and generic linear `2048->2048` (`algo_index=2`), Iter114 full-attention `attention_output 2048->2048` (`algo_index=2`) and `linear 2048->512` (`algo_index=11`) records, plus Iter116 0.8B `mlp_down 3584->1024` and `linear 2048->1024` explicit `algo_id=13/tile_id=0/stages_id=0/custom_option=13` records. Iter121 adds non-default Qwen3.5/SM86 table records for `edgefm_decode_swiglu_warp` on 0.8B `1024->3584` and 2B `2048->6144` decode GateUp+SwiGLU shapes. The matching 0.8B `attention_output 2048->1024` tactic is rejected in combination because it breaks token alignment. Iter71 uses in-place residual add (`hidden += mixer_output`); Iter73/76 specialize non-gated large-hidden RMSNorm for hidden `1024/2048`; small-hidden RMSNorm uses Iter49 one-warp-per-row.

## Current Correctness

- Latest full C++ kernel/table gate after Iter121: build passed; `scripts/operator_table/validate_operator_tables.py --platform 3060` passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `17 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Latest table-only gate after Iter116: `scripts/operator_table/validate_operator_tables.py --platform 3060` passed; `scripts/tune/tune_qwen_cublaslt.py` py_compile passed; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Qwen2.5 guard after Iter121 shared operator change: `tests/engine/test_qwen2_generate.py -k "token_alignment or kvcache" -s` -> `8 passed, 12 deselected`.
- Latest operator-table gate after Iter108 rejection: temporary table validation passed; after reverting the temporary record, `scripts/operator_table/validate_operator_tables.py --platform 3060` passed.
- Qwen2.5 guard after Iter100: `tests/engine/test_qwen2_generate.py -k "token_alignment or kvcache" -s` -> `8 passed, 12 deselected`; Qwen2.5 `512/32` graph-on smoke measured 0.5B `127.077 ms` and 1.5B `346.059 ms`, matching the docs baseline/noise band.
- Long-prefill graph/regular drift remains fixed by Iter39 token-boundary barriers; graph reuse stays enabled.
- Iter113 rejection gate after revert: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `17 passed`.
- Iter115 rejection gate: `EDGE_FM_DECODE_SWIGLU_ALLOW_SM86=1` made the isolated decode fused SwiGLU operator test pass, but Qwen3.5 generate failed token alignment for 2B, so the path remains default-off.

## Iter118 LMHead Standalone Gate

| Shape | Scalar24 mean | Best candidate | Best mean | Decision |
|---|---:|---|---:|---|
| 0.8B vocab 248320 hidden 1024 | 1.4995 ms | `bf162_w24` | 1.4970 ms | Reject, `~0.2%` |
| 2B vocab 248320 hidden 2048 | 2.9854 ms | `bf162_w32` | 2.9786 ms | Reject, `~0.2%` |

Artifact: `deliverables/kernel_opt/qwen3_5_lmhead_top1_iter118/benchmarks/lmhead_top1_iter118_baseline.json`. All variants match scalar24 and `torch.mv(...).argmax()` tokens in standalone. No production `src/` code was changed because the improvement is below the migration threshold and confirms the LMHead DRAM-roofline/algorithmic-ceiling conclusion.

## Iter119 SM86 SwiGLU Recheck

- `EDGE_FM_DECODE_SWIGLU_ALLOW_SM86=1` with Qwen3.5 0.8B regular fresh-dump alignment: passed.
- 0.8B p128/d32 graph-on benchmark: baseline `189.794 ms`, env-enabled `189.778 ms`, noise-level delta.
- Decision: keep default-off. The path is still unsafe for 2B and not measurably useful for 0.8B.

## Iter120 Prefill SwiGLU Recheck

- `EDGE_FM_PREFILL_SWIGLU_FUSION=1` passed Qwen3.5 0.8B/2B regular and graph-on token alignment.
- Representative graph-on benchmarks regressed: 0.8B p512/d128 `768.245 -> 769.714 ms`, 2B p512/d128 accepted baseline `1648.430 -> 1652.397 ms`.
- Decision: keep env default-off. Existing CUTLASS prefill SwiGLU fusion is correct but slower for Qwen3.5 on RTX 3060.

## Iter121 GateUp+SwiGLU Warp

- Standalone artifact: `deliverables/kernel_opt/qwen3_5_gateup_swiglu_iter121/`; best fused candidate beat the torch two-step reference for both visible shapes, then production transfer was gated by token alignment.
- Production path: non-default `edgefm_decode_swiglu_warp` GEMV+SwiGLU implementation under `fused_gate_up_activation`; selected only by exact Qwen3.5/SM86 decode operator-table records for 0.8B `1024->3584` and 2B `2048->6144`.
- Correctness: build passed; operator-table validation passed; Qwen3.5 runtime ops -> `17 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Decision: accept. The gain is small but consistent, table-scoped, and does not alter the generic default implementation selection.

| Model | Shape | Previous accepted | Iter121 | Delta | Decode step |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128, decode 32 | 189.299 | 188.130 | -0.62% | 5.430 |
| 0.8B | 128, decode 128 | 715.516 | 711.708 | -0.53% | 5.448 |
| 2B | 128, decode 32 | 407.973 | 405.948 | -0.50% | 12.020 |
| 2B | 128, decode 128 | 1572.494 | 1563.125 | -0.60% | 12.046 |

## Iter122 GateUp+SwiGLU Ceiling

- Post-Iter121 NSYS graph-off mapping: `edgefm_decode_swiglu_warp_kernel<16>` is the top visible 2B decode kernel at `110.853 ms` across `744` launches (`29.3%` of decode GPU time), followed by `lm_head_top1::stage1_kernel<bf16>` at `92.260 ms` (`24.4%`).
- NCU first matching 2B launch: `153.70 us`, memory/DRAM throughput `95.34%`, compute throughput `23.99%`, achieved occupancy `93.18%`, block/grid `512/384`, registers/thread `33`.
- Decision: no production change. For the current exact BF16 GEMV algorithm this path is `at_ceiling`; further material gain requires reducing bytes read or changing the contract/layout (custom compressed/quantized GEMV, a larger fused-projection design, or an accepted approximation).

Artifact: `profiles/qwen3_5_nsys_action_iter121.md`, `profiles/ncu/qwen3_5_2b_decode_swiglu_warp_iter122.ncu-rep`.

## Long-Loop Closure

- Report: `long_loop/closure/LONG_LOOP_CLOSURE.md`.
- Profile digest: `long_loop/closure/artifacts/profile-digests/gateup_swiglu_iter122.md`.
- Source idea ledger: `long_loop/closure/artifacts/source-idea-ledger.md`.
- Decision: converge this phase. Continuing exact-BF16 local kernel tuning has low expected ROI; future work should be a new spec for quantized/compressed GEMV, exact LMHead indexing, larger fused projection, or runtime scheduling.

## Current Performance Facts

| Case | Avg ms | Prefill ms | Decode step ms | Notes |
|---|---:|---:|---:|---|
| 0.8B p128/d32 graph-on latest accepted | 188.130 | 19.647 | 5.430 | Iter121 decode GateUp+SwiGLU warp |
| 0.8B p128/d128 graph-on latest accepted | 711.708 | 19.387 | 5.448 | Iter121 decode GateUp+SwiGLU warp |
| 0.8B p512/d128 graph-on latest accepted | 766.434 | 63.815 | 5.530 | Iter116 safe 0.8B decode linear tactics |
| 0.8B p1024/d128 graph-on latest accepted | 830.181 | 127.535 | 5.530 | Iter116 safe 0.8B decode linear tactics |
| 2B p128/d32 graph-on latest accepted | 405.948 | 33.126 | 12.020 | Iter121 decode GateUp+SwiGLU warp |
| 2B p128/d128 graph-on latest accepted | 1563.125 | 32.922 | 12.046 | Iter121 decode GateUp+SwiGLU warp |
| 2B p512/d128 graph-on latest accepted | 1648.430 | 103.371 | 12.163 | Iter114 full-attention decode cublasLt table records |
| 2B p1024/d128 graph-on latest accepted | 1748.435 | 202.427 | 12.171 | Iter114 full-attention decode cublasLt table records |
| 0.8B p128/d32 graph-on previous accepted | 192.596 | 19.590 | 5.576 | Iter112 decode conv1d + GatedDelta + gated RMSNorm fusion |
| 2B p128/d32 graph-on previous accepted | 408.857 | 33.148 | 12.115 | Iter112 decode conv1d + GatedDelta + gated RMSNorm fusion |
| 2B p512/d128 graph-on previous accepted | 1654.142 | 104.378 | 12.200 | Iter112 full matrix |
| 2B p1024/d128 graph-on previous accepted | 1754.552 | 204.283 | 12.204 | Iter112 full matrix |
| 0.8B p128/d32 graph-on previous accepted | 192.986 | 19.603 | 5.589 | Iter105 P0 qmem-output algebra |
| 2B p128/d32 graph-on previous accepted | 421.638 | 33.017 | 12.532 | Iter105 P0 qmem-output algebra |
| 2B p1024/d128 graph-on validation | 1803.137 | 202.372 | 12.602 | Iter105 P0 qmem-output algebra |
| 0.8B p128/d32 graph-on previous accepted | 194.618 | 20.800 | 5.603 | Iter100 P0 tile32 shared q/k + vectorized state update |
| 2B p128/d32 graph-on previous accepted | 423.158 | 34.189 | 12.543 | Iter100 P0 tile32 shared q/k + vectorized state update |
| 2B p1024/d128 graph-on previous validation | 1812.822 | 211.914 | 12.603 | Iter100 P0 tile32 shared q/k + vectorized state update |
| 0.8B p128/d32 graph-on previous accepted | 197.393 | 23.408 | 5.608 | Iter98 P0 shared q/k + vectorized state update |
| 2B p128/d32 graph-on previous accepted | 425.986 | 36.949 | 12.545 | Iter98 P0 shared q/k + vectorized state update |
| 2B p1024/d128 graph-on previous validation | 1829.887 | 229.596 | 12.598 | Iter98 P0 shared q/k + vectorized state update |
| 0.8B p128/d32 graph-on previous accepted | 205.148 | 31.394 | 5.600 | Iter97 P0 vectorized state decay/update |
| 2B p128/d32 graph-on previous accepted | 433.192 | 44.654 | 12.529 | Iter97 P0 vectorized state decay/update |
| 2B p1024/d128 graph-on previous validation | 1897.618 | 296.433 | 12.605 | Iter97 P0 vectorized state decay/update |
| 0.8B p128/d32 graph-on previous accepted | 207.594 | 33.835 | 5.601 | Iter93 default greedy top1 |
| 2B p128/d32 graph-on previous accepted | 435.528 | 46.989 | 12.529 | Iter93 default greedy top1 |
| 2B p1024/d128 graph-on previous validation | 1916.334 | 316.512 | 12.595 | Iter93 default greedy top1 |
| 0.8B p128/d32 graph-on previous accepted | 207.644 | 33.831 | 5.602 | Iter91 decode from_ab column-fused |
| 2B p128/d32 graph-on previous accepted | 435.832 | 46.905 | 12.541 | Iter91 decode from_ab column-fused |
| 2B p1024/d128 graph-on previous validation | 1920.326 | 317.660 | 12.617 | Iter91 decode from_ab column-fused |
| 0.8B p128/d32 graph-on previous accepted | 209.432 | 33.837 | 5.660 | Iter89 decode from_ab precomputed q/k norms |
| 2B p128/d32 graph-on previous accepted | 436.955 | 46.847 | 12.580 | Iter89 decode from_ab precomputed q/k norms |
| 2B p1024/d128 graph-on previous validation | 1923.887 | 316.278 | 12.656 | Iter89 decode from_ab precomputed q/k norms |
| 0.8B p128/d32 graph-on previous accepted | 210.717 | 33.930 | 5.698 | Iter87 decode from_ab single-token parallel q/k l2norm |
| 2B p128/d32 graph-on previous accepted | 438.904 | 47.170 | 12.631 | Iter87 decode from_ab single-token parallel q/k l2norm |
| 2B p1024/d128 graph-on previous validation | 1932.613 | 318.427 | 12.708 | Iter87 decode from_ab single-token parallel q/k l2norm |
| 0.8B p128/d32 graph-on previous accepted | 210.924 | 33.358 | 5.724 | Iter82 decode from_ab single-token static |
| 2B p128/d32 graph-on previous accepted | 439.496 | 46.846 | 12.662 | Iter82 decode from_ab single-token static |
| 2B p1024/d128 graph-on previous validation | 1936.460 | 317.730 | 12.744 | Iter82 decode from_ab single-token static |
| 0.8B p128/d32 graph-on previous accepted | 211.329 | 33.381 | 5.735 | Iter78 decode from_ab static full tile |
| 2B p128/d32 graph-on previous accepted | 440.283 | 47.032 | 12.681 | Iter78 decode from_ab static full tile |
| 2B p1024/d128 graph-on previous validation | 1939.423 | 318.522 | 12.760 | Iter78 decode from_ab static full tile |
| 0.8B p128/d32 graph-on previous accepted | 214.112 | 33.485 | 5.822 | Iter76 fixed-hidden static stride |
| 2B p128/d32 graph-on previous accepted | 443.138 | 47.149 | 12.770 | Iter76 fixed-hidden static stride |
| 2B p1024/d128 graph-on previous validation | 1950.131 | 318.923 | 12.841 | Iter76 fixed-hidden static stride |
| 0.8B p128/d32 graph-on previous accepted | 215.235 | 33.169 | 5.869 | Iter71 in-place residual add |
| 2B p128/d32 graph-on previous accepted | 446.619 | 46.996 | 12.887 | Iter71 in-place residual add |
| 2B p1024/d128 graph-on previous validation | 1962.376 | 317.062 | 12.953 | Iter71 in-place residual add |
| 0.8B p128/d32 graph-on previous accepted | 217.065 | 33.491 | 5.917 | Iter64 decode from_ab |
| 2B p128/d32 graph-on previous accepted | 447.830 | 47.078 | 12.923 | Iter64 decode from_ab |
| 2B p1024/d128 graph-on previous validation | 1969.855 | 318.835 | 12.997 | Iter64 warmup-2/runs-5 |

## Iter116 0.8B Graph-On Gates

| Model | Prefill | Graph on avg ms | Graph on prefill ms | Graph on decode step ms | Baseline |
|---|---:|---:|---:|---:|---|
| 0.8B | 128, decode 32 | 189.299 | 19.567 | 5.471 | Iter112 192.596 / 5.576 |
| 0.8B | 128, decode 128 | 715.516 | 19.213 | 5.481 | Iter112 729.227 / 5.587 |
| 0.8B | 512, decode 128 | 766.434 | 63.815 | 5.530 | Iter112 780.646 / 5.637 |
| 0.8B | 1024, decode 128 | 830.181 | 127.535 | 5.530 | Iter112 845.858 / 5.645 |

Iter116 adds only Qwen3.5/RTX3060 decode operator-table records for 0.8B safe shapes: `mlp_down 3584->1024` and generic `linear 2048->1024`, both using explicit cublasLt `algo_id=13,tile_id=0,stages_id=0,custom_option=13`. The matching 0.8B `attention_output 2048->1024` tactic was individually token-safe, but `mlp_down+attention_output` and all three records together diverged at decode token index 8, so it is rejected. Correctness gate passed before acceptance: operator-table validation passed, tune script py_compile passed, and Qwen3.5 0.8B/2B fresh-dump generate passed (`13 passed`).

## Iter93 Matrix

| Model | Prefill | Graph off avg ms | Graph on avg ms | Graph on prefill ms | Graph on decode step ms |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128 | 789.715 | 745.260 | 33.145 | 5.605 |
| 0.8B | 512 | 883.238 | 838.082 | 120.420 | 5.649 |
| 0.8B | 1024 | 1011.374 | 967.389 | 248.543 | 5.658 |
| 2B | 128 | 1686.481 | 1643.354 | 47.119 | 12.567 |
| 2B | 512 | 1807.941 | 1762.474 | 160.922 | 12.608 |
| 2B | 1024 | 1964.777 | 1920.495 | 319.083 | 12.607 |

Iter93 improves all six Iter91 graph-on matrix cases and all six graph-off matrix cases. Graph-on geomean ratio versus Iter91 is `0.99826` (~`0.17%` faster). The main purpose is also to move default Qwen3.5 greedy LMHead onto the top1 path whose stage1 NCU is already at the memory roofline (`DRAM 97.46%`).

## Iter114 2B Graph-On Gates

| Model | Prefill | Graph on avg ms | Graph on prefill ms | Graph on decode step ms | Baseline |
|---|---:|---:|---:|---:|---|
| 2B | 128, decode 32 | 407.973 | 33.014 | 12.091 | Iter112 408.857 / 12.115 |
| 2B | 128, decode 128 | 1572.494 | 32.999 | 12.119 | Iter112 1576.995 / 12.152 |
| 2B | 512, decode 128 | 1648.430 | 103.371 | 12.163 | Iter112 1654.142 / 12.200 |
| 2B | 1024, decode 128 | 1748.435 | 202.427 | 12.171 | Iter112 1754.552 / 12.204 |

Iter114 adds only Qwen3.5/RTX3060 decode operator-table records for 2B full-attention projections: `attention_output 2048->2048` uses `algo_index=2`, and `linear 2048->512` uses `algo_index=11` for `k_proj/v_proj`. Tuner scans rejected `qwen3_linear_qkv 2048->6144` and `self_attn_q 2048->4096` because baseline remained best. Correctness gate passed before acceptance: operator-table validation passed, tune script py_compile passed, and Qwen3.5 0.8B/2B fresh-dump generate passed (`13 passed`).

## Iter112 Matrix

| Model | Prefill | Graph off avg ms | Graph on avg ms | Graph on prefill ms | Graph on decode step ms |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128 | 769.482 | 729.227 | 19.358 | 5.587 |
| 0.8B | 512 | 822.184 | 780.646 | 64.329 | 5.637 |
| 0.8B | 1024 | 887.597 | 845.858 | 128.498 | 5.645 |
| 2B | 128 | 1616.000 | 1576.995 | 33.272 | 12.152 |
| 2B | 512 | 1693.494 | 1654.142 | 104.378 | 12.200 |
| 2B | 1024 | 1793.715 | 1754.552 | 204.283 | 12.204 |

Iter112 fuses Qwen3.5 decode-only depthwise conv1d kernel4 state update with the Iter111 GatedDelta + gated RMSNorm fused path. Versus Iter111, graph-on deltas are 0.8B `-0.09%/-0.11%/-0.01%` and 2B `-0.04%/-0.06%/-0.05%`; graph-off improves all six cases by `0.13%-0.40%`. This closes the simple local decode conv/GatedDelta/RMSNorm launch-fusion tier.

## Iter112 Acceptance Gate

| Evidence | Iter111 / baseline | Iter112 | Decision |
|---|---:|---:|---|
| Operator tests | 16 passed | 17 passed | Accepted |
| Fresh-dump generate | 13 passed | 13 passed | Accepted |
| 0.8B p128/d32 graph-on avg | 192.630 ms | 192.596 ms | Accepted |
| 2B p128/d32 graph-on avg | 409.155 ms | 408.857 ms | Accepted |
| 0.8B p128/d128 graph-on avg | 729.895 ms | 729.227 ms | Accepted |
| 2B p128/d128 graph-on avg | 1577.651 ms | 1576.995 ms | Accepted |

Correctness gate passed before acceptance: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `17 passed`; `EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 tests/engine/test_qwen3_5_generate.py -s` -> `13 passed`.

## Iter111 Matrix

| Model | Prefill | Graph off avg ms | Graph on avg ms | Graph on prefill ms | Graph on decode step ms |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128 | 772.554 | 729.895 | 19.366 | 5.592 |
| 0.8B | 512 | 824.057 | 781.488 | 64.366 | 5.643 |
| 0.8B | 1024 | 890.105 | 845.920 | 128.360 | 5.647 |
| 2B | 128 | 1618.101 | 1577.651 | 33.320 | 12.157 |
| 2B | 512 | 1696.311 | 1655.172 | 104.519 | 12.207 |
| 2B | 1024 | 1797.104 | 1755.485 | 204.336 | 12.211 |

Iter111 fuses Qwen3.5 decode-only `gated_delta_sequence_from_ab` with the immediately following gated RMSNorm for `seq_len=1,key_dim=value_dim=128`. Versus Iter107 full matrix, graph-on deltas are 0.8B `-0.17%/+0.03%/-0.06%` and 2B `-0.07%/-0.02%/-0.05%`; graph-off improves all six cases by `0.04%-0.44%`. The gain is small but correctness-clean and scoped to Qwen3.5-local operator/model code.

## Iter111 Acceptance Gate

| Evidence | Iter107 / baseline | Iter111 | Decision |
|---|---:|---:|---|
| Operator tests | 15 passed | 16 passed | Accepted |
| Fresh-dump generate | 13 passed | 13 passed | Accepted |
| 0.8B p128/d32 graph-on avg | 193.590 ms | 192.630 ms | Accepted |
| 2B p128/d32 graph-on avg | 409.160 ms | 409.155 ms | Neutral |
| 0.8B p128/d128 graph-on avg | 731.119 ms | 729.895 ms | Accepted |
| 2B p128/d128 graph-on avg | 1578.680 ms | 1577.651 ms | Accepted |

Correctness gate passed before acceptance: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `16 passed`; `EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 tests/engine/test_qwen3_5_generate.py -s` -> `13 passed`.

## Iter107 Matrix

| Model | Prefill | Graph off avg ms | Graph on avg ms | Graph on prefill ms | Graph on decode step ms |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128 | 775.974 | 731.119 | 19.260 | 5.603 |
| 0.8B | 512 | 827.588 | 781.274 | 63.809 | 5.646 |
| 0.8B | 1024 | 890.803 | 846.403 | 127.624 | 5.658 |
| 2B | 128 | 1618.779 | 1578.680 | 33.109 | 12.168 |
| 2B | 512 | 1697.754 | 1655.575 | 103.926 | 12.216 |
| 2B | 1024 | 1797.839 | 1756.324 | 203.540 | 12.224 |

Iter107 improves all six 2B graph-on/off matrix cases versus Iter105 with Qwen3.5 2B decode linear cuBLASLt tactic records. 2B graph-on p128/p512/p1024 improved `2.72%-3.06%`; 2B graph-off improved `2.91%-3.22%`. 0.8B is effectively unchanged because the accepted records target only 2B shape signatures.

## Iter107 Acceptance Gate

| Evidence | Iter105 / baseline | Iter107 | Decision |
|---|---:|---:|---|
| 2B decode `mlp_down` tuner | 0.093184 ms | 0.080896 ms | Accepted |
| 2B decode generic linear `2048->2048` tuner | 0.035840 ms | 0.032768 ms | Accepted |
| 2B p128/d32 graph-on avg | 421.638 ms | 409.160 ms | Accepted |
| 2B p128/d128 graph-on avg | 1628.511 ms | 1578.680 ms | Accepted |
| 2B p512/d128 graph-on avg | 1706.069 ms | 1655.575 ms | Accepted |
| 2B p1024/d128 graph-on avg | 1805.514 ms | 1756.324 ms | Accepted |
| 0.8B p128/d128 graph-on avg | 730.513 ms | 731.119 ms | Noise / unchanged |

Correctness gate passed before acceptance: operator-table validation passed; `EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 tests/engine/test_qwen3_5_generate.py -s` -> `13 passed`.

## Post-Iter112 Hotspots

2B p128/d32 graph-off mapping trace after Iter112:

| Stage | Kernel / family | Time | Share | Status |
|---|---|---:|---:|---|
| Decode | `lm_head_top1::stage1_kernel<bf16>` | 92.262 ms | 24.2% | Iter110 measured DRAM 97.60%; needs exact byte-reduction/search algorithm |
| Decode | GateUp dense GEMV family | ~4.68 ms/layer | 1.2%/layer | Iter109 found largest explicit cuBLASLt alternatives baseline-best |
| Decode | `conv1d_gated_delta_sequence_from_ab_gated_rmsnorm_single_token_128` | 5.893 ms | 1.5% | Iter112 accepted; simple local launch fusion tier closed |
| Decode | fixed-hidden RMSNorm 2048 | 4.099 ms | 1.1% | Iter113 512-thread retune rejected |

Report: `profiles/qwen3_5_nsys_action_iter112.md`; trace: `profiles/nsys/qwen3_5_2b_p128_d32_graph_off_iter112_mapping.nsys-rep`.

SM86 fused SwiGLU note: Iter115 verified the existing TRT-LLM SM80 decode fused SwiGLU path is not safe to enable on RTX 3060 for Qwen3.5. It passed the isolated layer equivalence test under `EDGE_FM_DECODE_SWIGLU_ALLOW_SM86=1`, but Qwen3.5 2B token alignment diverged at decode token index 8, so this remains a correctness blocker rather than a performance candidate.

## Iter105 Matrix

| Model | Prefill | Graph off avg ms | Graph on avg ms | Graph on prefill ms | Graph on decode step ms |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128 | 773.950 | 730.513 | 19.172 | 5.599 |
| 0.8B | 512 | 825.816 | 781.354 | 64.048 | 5.645 |
| 0.8B | 1024 | 890.332 | 846.122 | 127.552 | 5.656 |
| 2B | 128 | 1672.714 | 1628.511 | 33.095 | 12.560 |
| 2B | 512 | 1750.783 | 1706.069 | 104.044 | 12.612 |
| 2B | 1024 | 1851.745 | 1805.514 | 203.569 | 12.611 |

Iter105 improves all six graph-on and graph-off matrix cases versus Iter100. The change uses `q^T(S + k*delta) = q^T S + (q^T k) * delta` to remove the post-update output state-read pass while keeping the final recurrent state update path unchanged.

## Iter105 Acceptance Gate

| Evidence | Iter100 / baseline | Iter105 | Decision |
|---|---:|---:|---|
| Production P0 NCU duration | 0.803 ms | 0.706 ms | Accepted |
| 0.8B p128/d32 graph-on avg | 194.618 ms | 192.986 ms | Accepted |
| 2B p128/d32 graph-on avg | 423.158 ms | 421.638 ms | Accepted |
| 2B p1024/d128 graph-on avg | 1812.822 ms | 1803.137 ms | Accepted |
| 2B p1024/d128 prefill | 211.914 ms | 202.372 ms | Accepted |

Correctness gate passed before acceptance: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; `EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 tests/engine/test_qwen3_5_generate.py -s` -> `13 passed`.

## Iter100 Matrix

| Model | Prefill | Graph off avg ms | Graph on avg ms | Graph on prefill ms | Graph on decode step ms |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128 | 778.738 | 733.114 | 20.478 | 5.609 |
| 0.8B | 512 | 831.857 | 785.403 | 68.211 | 5.645 |
| 0.8B | 1024 | 899.167 | 855.628 | 136.865 | 5.658 |
| 2B | 128 | 1672.990 | 1629.689 | 34.192 | 12.561 |
| 2B | 512 | 1756.129 | 1710.639 | 108.585 | 12.612 |
| 2B | 1024 | 1860.306 | 1814.472 | 212.906 | 12.608 |

Iter100 improves all six graph-on and graph-off matrix cases versus Iter98. The change rebalances the precomputed recurrence from tile16 to tile32 after shared q/k staging made the larger value tile transfer cleanly to full generate.

## Iter100 Acceptance Gate

| Evidence | Iter98 / baseline | Iter100 | Decision |
|---|---:|---:|---|
| Production P0 NCU duration | 0.953 ms | 0.803 ms | Accepted |
| 0.8B p128/d32 graph-on avg | 197.393 ms | 194.618 ms | Accepted |
| 2B p128/d32 graph-on avg | 425.986 ms | 423.158 ms | Accepted |
| 2B p1024/d128 graph-on avg | 1829.887 ms | 1812.822 ms | Accepted |
| 2B p1024/d128 prefill | 229.596 ms | 211.914 ms | Accepted |

Correctness gate passed before acceptance: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; `EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 tests/engine/test_qwen3_5_generate.py -s` -> `13 passed`. Qwen2.5 token/KV regression and 0.5B/1.5B performance smoke also passed after the Qwen3.5-local change.

## Iter98 Matrix

| Model | Prefill | Graph off avg ms | Graph on avg ms | Graph on prefill ms | Graph on decode step ms |
|---|---:|---:|---:|---:|---:|
| 0.8B | 128 | 779.852 | 735.102 | 22.957 | 5.605 |
| 0.8B | 512 | 840.038 | 795.678 | 77.974 | 5.648 |
| 0.8B | 1024 | 920.492 | 875.656 | 156.370 | 5.661 |
| 2B | 128 | 1676.677 | 1632.677 | 36.746 | 12.564 |
| 2B | 512 | 1766.954 | 1720.722 | 118.477 | 12.614 |
| 2B | 1024 | 1880.147 | 1833.518 | 231.807 | 12.610 |

Iter98 improves all six graph-on and graph-off matrix cases versus Iter93. The largest gains are in prefill because shared-memory q/k staging removes repeated normalized q/k global loads inside the P0 precomputed recurrence kernel.

## Iter98 Acceptance Gate

| Evidence | Iter97 / baseline | Iter98 | Decision |
|---|---:|---:|---|
| Standalone P0 event avg | 1.207 ms | 0.699 ms | Accepted |
| Standalone P0 NCU duration | 1.50 ms | 0.953 ms | Accepted |
| 0.8B p128/d32 graph-on avg | 205.148 ms | 197.393 ms | Accepted |
| 2B p128/d32 graph-on avg | 433.192 ms | 425.986 ms | Accepted |
| 2B p1024/d128 graph-on avg | 1897.618 ms | 1829.887 ms | Accepted |
| 2B p1024/d128 prefill | 296.433 ms | 229.596 ms | Accepted |

Correctness gate passed before acceptance: `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; `EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 tests/engine/test_qwen3_5_generate.py -s` -> `13 passed`. Qwen2.5 token/KV regression also passed after the Qwen3.5-local change.

## Iter97 Acceptance Gate

| Evidence | Iter93 / baseline | Iter97 | Decision |
|---|---:|---:|---|
| Standalone P0 event avg | 1.458 ms | 1.207 ms | Accepted |
| Standalone P0 NCU duration | 1.84 ms | 1.50 ms | Accepted |
| 0.8B p128/d32 graph-on avg | 207.594 ms | 205.148 ms | Accepted |
| 2B p128/d32 graph-on avg | 435.528 ms | 433.192 ms | Accepted |
| 2B p1024/d128 graph-on avg | 1916.334 ms | 1897.618 ms | Accepted |
| 2B p1024/d128 prefill | 316.512 ms | 296.433 ms | Accepted |

Correctness gate passed before acceptance: `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; `EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 tests/engine/test_qwen3_5_generate.py -s` -> `13 passed`. The production transfer only vectorizes P0 precomputed recurrent state decay/update in `src/operators/qwen3_5/qwen3_5_ops.cu`; runtime scheduling and model state semantics were not changed.

## Latest P0 NCU

P0 precomputed recurrence, shape `seq_len=128, heads=16, key_dim=128, value_dim=128`:

| Kernel | Duration | Compute throughput | Memory throughput | Achieved occupancy | Eligible warps/scheduler | No eligible | Grid | Block | Reg/thread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Iter105 qmem-output algebra | 0.706 ms | 33.72% | 55.33% | 38.31% | 0.41 | 68.48% | 64 | 256 | 40 |
| Iter100 production tile32 shared q/k + vectorized update | 0.803 ms | 35.90% | 48.65% | 38.20% | 0.36 | 71.25% | 64 | 256 | 40 |
| Iter98 shared q/k + vectorized state update | 0.953 ms | 42.08% | 49.51% | 76.33% | 0.60 | 68.87% | 128 | 256 | 38 |
| Iter97 vectorized state update | 1.50 ms | 37.35% | n/a | 75.98% | 0.31 | 78.23% | 128 | 256 | n/a |
| Iter66 barrier-safe baseline | 1.84 ms | n/a | n/a | n/a | n/a | n/a | 128 | 256 | n/a |

Iter105 is faster because it removes the post-update output state-read pass with a token-stable recurrence algebra rewrite. NCU still flags small-grid and barrier stalls, so raw hardware-peak utilization is not a valid acceptance target for this recurrence-bound kernel.

## Latest LMHead NCU

Qwen3.5 2B greedy `lm_head_top1::stage1_kernel`, first matching launch:

| Kernel | Duration | DRAM throughput | Memory throughput | SM throughput | Active warps | Issue active | Grid | Block | Reg/thread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Iter110 top1 stage1 | 2.981 ms | 97.60% | 97.60% | 61.85% | 97.79% | 28.91% | 10347 | 768 | 31 |

This path is at the measured memory-bandwidth ceiling for the current algorithm. Further LMHead optimization must reduce bytes read or change the search algorithm; launch-shape/shared-hidden retuning is closed.

Iter117 feasibility check: simple exact row-norm pruning is not viable for Qwen3.5. For 0.8B first 6 generated steps and 2B first 3 generated steps, even a perfect Cauchy upper-bound pass using the final true best logit pruned `0 / 248320` vocab rows; a practical seed from top-4096 row norms also pruned `0` rows. Artifacts: `analysis/qwen3_5_0p8b_lmhead_prune_feasibility_iter117.json` and `analysis/qwen3_5_2b_lmhead_prune_feasibility_iter117.json`.

## Latest Decode NCU

First matching decode `gated_delta_sequence_from_ab_single_token_128_kernel<bf16,bf16>` launch, 0.8B p128/d4:

| Kernel | Duration | SM throughput | Memory throughput | Achieved occupancy | Eligible warps/scheduler | No eligible | Grid | Block | Reg/thread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Iter91 single-token column-fused | 13.89 us | n/a | 42.04% | 8.18% | 0.12 | 88.97% | 64 | 128 | 48 |
| Iter89 single-token precompute q/k norms | 17.54 us | n/a | 33.59% | 35.14% | 0.32 | 78.82% | 64 | 256 | 48 |
| Iter87 single-token parallel l2norm | 20.74 us | 25.39% | 28.74% | 32.72% | 0.40 | 74.84% | 64 | 256 | 39 |
| Iter82 single-token static | 26.37 us | 22.02% | 22.24% | 33.77% | 0.31 | 79.88% | 64 | 256 | 39 |
| Iter78 static full tile | 27.20 us | 20.94% | 21.35% | 33.40% | 0.33 | 78.88% | 64 | 256 | 48 |
| Iter64 generic from_ab | 33.63 us | 23.98% | 17.75% | n/a | 0.39 | 72.17% | 64 | 256 | n/a |

Iter91 fuses each value column's decay/delta/update/output work and cuts the Iter89 first launch from `17.54 us` to `13.89 us`. It is still tiny-grid/low-wave limited, but it removes several block barriers and memory passes.

## Latest Hotspot Map

Latest mapping traces are Iter116 0.8B/2B p128/d32 graph-off NSYS. The 0.8B profile reflects the accepted Iter116 safe linear records; the 2B profile confirms the post-Iter114 dense-linear state. Graph-on formal trace hides most replay internals, so graph-off mapping remains the attribution source and graph-on remains the acceptance gate.

| Stage | Hotspot | Time | Share | Decision |
|---|---|---:|---:|---|
| 0.8B decode | `lm_head_top1::stage1_kernel<bf16>` | 46.174 ms | 26.3% decode | Iter110 2B NCU shows same family at DRAM `97.60%`; exact byte-reduction/search algorithm required |
| 0.8B decode | fused conv/GatedDelta/gated RMSNorm | 5.944 ms | 3.4% decode | Iter112 accepted; simple local launch-fusion tier closed |
| 0.8B decode | fixed-hidden RMSNorm 1024 | 3.733 ms | 2.1% decode | Iter73/76 accepted; later add/RMSNorm and retune candidates rejected |
| 0.8B decode | FlashInfer single decode attention | 2.958 ms | 1.7% decode | Iter86 Qwen3.5 tuned attention shape rejected across 0.8B/2B |
| 0.8B decode | residual add | 2.296 ms | 1.3% decode | Iter71 accepted; Iter81/96 follow-up local fusions rejected |
| 0.8B prefill | `gated_delta_sequence_precomputed_kernel<bf16,true,32>` | 9.402 ms | 46.5% prefill | Iter105 accepted; recurrence-aware current-layout ceiling |
| 2B decode | `lm_head_top1::stage1_kernel<bf16>` | 92.265 ms | 24.2% decode | Iter110 NCU DRAM `97.60%`, active warps `97.79%`; at bandwidth ceiling |
| 2B decode | `edgefm_decode_swiglu_warp_kernel<16>` GateUp GEMV+SwiGLU | 110.853 ms | 29.3% decode GPU | Iter122 NCU DRAM `95.34%`; at current exact-BF16 memory ceiling |
| 2B decode | GateUp library GEMV alternatives | N/A | N/A | Iter109/Iter116 explicit searches found fused_gate_up baseline-best before Iter121; larger compressed/custom GEMV design required for more |
| 2B decode | fused conv/GatedDelta/gated RMSNorm | 5.922 ms | 1.6% decode | Iter112 accepted; simple local launch-fusion tier closed |
| 2B decode | fixed-hidden RMSNorm 2048 | 4.122 ms | 1.1% decode | Iter113 retune rejected |

## Accepted Since Iter39

- Iter39: token-boundary barrier fix for recurrent sequence correctness.
- Iter43: optional `lm_head_top1` stage1 `24` warps/block; Iter93 later made it default for Qwen3.5 greedy.
- Iter49: one-warp small-hidden RMSNorm/gated-RMSNorm; current accepted small-hotspot implementation.
- Iter59: prefill-only GatedDeltaNet precomputed recurrent sequence `tile=16/thread=256`, accepted after Iter39 barrier fix made it token-safe.
- Iter64: decode-only `gated_delta_sequence_from_ab`, accepted as a small Qwen3.5-local win after alloc-restore removed long-prefill noise.
- Iter71: in-place residual add, accepted after cleanup correctness and all six graph-on matrix cases improved versus Iter69.
- Iter73: fixed-hidden large RMSNorm for hidden `1024/2048`, accepted after 15 operator tests, fresh-dump alignment, and 6/6 matrix improvement.
- Iter76: fixed-hidden RMSNorm static stride `256`, accepted after 15 operator tests, fresh-dump alignment, NCU, and 6/6 matrix improvement.
- Iter78: decode `gated_delta_sequence_from_ab_full_tile_128`, accepted after 15 operator tests, fresh-dump alignment, NCU, and 6/6 matrix improvement versus Iter76.
- Iter82: decode `gated_delta_sequence_from_ab_single_token_128`, accepted after 15 operator tests, fresh-dump alignment, NCU, and 12/12 graph-on/off matrix improvement versus Iter78.
- Iter87: decode `gated_delta_sequence_from_ab_single_token_128` parallel q/k l2norm, accepted after 15 operator tests, fresh-dump alignment, NCU, and 12/12 graph-on/off matrix improvement versus Iter82.
- Iter89: decode `gated_delta_sequence_from_ab_single_token_128` precomputed q/k normalized vectors, accepted after 15 operator tests, fresh-dump alignment, NCU, and 12/12 graph-on/off matrix improvement versus Iter87.
- Iter91: decode `gated_delta_sequence_from_ab_single_token_128` 128-thread column-fused schedule, accepted after 15 operator tests, fresh-dump alignment, NCU, and 12/12 graph-on/off matrix improvement versus Iter89.
- Iter93: Qwen3.5 greedy default `lm_head_top1` policy, accepted after 15 operator tests, fresh-dump alignment, profile-script unit coverage, and 12/12 graph-on/off matrix improvement versus Iter91.
- Iter97: P0 precomputed recurrent state decay/update vectorization, accepted after standalone correctness/NCU, 15 operator tests, fresh-dump 0.8B/2B alignment, and target graph-on benchmark wins on 0.8B p128/d32, 2B p128/d32, and 2B p1024/d128.
- Iter98: P0 precomputed shared q/k staging, accepted after standalone correctness/NCU, 15 operator tests, fresh-dump 0.8B/2B alignment, Qwen2.5 token/KV regression, and 12/12 graph-on/off matrix improvement versus Iter93.
- Iter100: P0 precomputed `value_tile=32` production transfer, accepted after 15 operator tests, fresh-dump 0.8B/2B alignment, production NCU, Qwen2.5 token/KV/performance guard, and 12/12 graph-on/off matrix improvement versus Iter98.
- Iter105: P0 qmem-output algebra, accepted after standalone correctness, 15 operator tests, fresh-dump 0.8B/2B alignment, production NCU (`802.50 us -> 705.70 us`), and 12/12 graph-on/off matrix improvement versus Iter100.
- Iter107: Qwen3.5 2B decode dense-linear cuBLASLt tactics, accepted after operator-table validation, fresh-dump 0.8B/2B alignment, target p128/d32/p512/d128/p1024/d128 benchmark wins, and 2B graph-on/off full-matrix improvement versus Iter105.
- Iter116: Qwen3.5 0.8B decode dense-linear cublasLt explicit tactics, accepted after operator-table validation, tune-script py_compile, fresh-dump 0.8B/2B alignment, and 0.8B graph-on p128/p512/p1024 d128 benchmark wins. The accepted safe pair is `mlp_down 3584->1024` plus generic `linear 2048->1024`; the matching `attention_output 2048->1024` tactic is rejected in combination because it diverges at decode token index 8.
- Iter117: LMHead exact top1 row-norm pruning feasibility check, accepted as a blocker/evidence update. Both 0.8B and 2B prune `0` rows under the optimistic perfect-bound test, so simple exact byte-reduction is not a viable local candidate.
- Iter121: Qwen3.5 decode GateUp+SwiGLU warp kernel, accepted after standalone direction evidence, production build, operator-table validation, Qwen3.5 runtime ops, fresh-dump 0.8B/2B alignment, and 0.8B/2B p128 graph-on benchmark wins. The new implementation is non-default and selected only by exact Qwen3.5/SM86 table records unless explicitly enabled by env.

## Recent Rejections

- Iter109: Qwen3.5 2B explicit cuBLASLt search for large decode linear shapes found no production candidate. `fused_gate_up 2048->12288` baseline `0.154624 ms` beat the best explicit `0.157696 ms`; `qwen3_linear_qkv 2048->6144` baseline `0.081216 ms` beat `0.084992 ms`; `self_attn_q 2048->4096` baseline `0.057344 ms` beat `0.061136 ms`. No operator-table change was made.
- Iter108: Qwen3.5 2B decode `linear` `2048->512` cuBLASLt record `algo_index=11` was locally faster only for `self_attn_v` (`0.013136 -> 0.012576 ms`, `4.45%`), while `self_attn_k`/`linear_a`/`linear_b` were baseline or noise. It was token-correct (`13` generate tests), but rejected and reverted because 2B p128/d32 graph-on moved only `409.160 -> 409.071 ms` (`0.02%`, noise). The temporary operator-table record was removed and table validation passed.
- Iter106: Qwen3.5 0.8B decode `mlp_down` cuBLASLt record `algo_index=2` was locally faster in the tuner (`0.034816 -> 0.034400 ms`) and token-correct (`13` generate tests), but rejected and reverted because 0.8B p128/d32 graph-on regressed `192.986 -> 193.411 ms`. The temporary operator-table record was removed and table validation passed.
- Iter104: P0 precomputed `tile32/thread512` was standalone-correct and slightly faster than current `thread256` (`0.606 ms` vs `0.620 ms` event time), but rejected and reverted because production 0.8B p128/d32 graph-on regressed `194.618 -> 195.215 ms` and prefill `20.800 -> 21.528 ms`; post-revert operator tests passed (`15`).
- Iter103: P0 fused update+output retested after Iter100 tile32 was standalone-correct and faster (`0.544 ms` vs current `0.620 ms` event time), but rejected and reverted because production 0.8B p128/d32 graph-on regressed `194.618 -> 198.066 ms` and prefill `20.800 -> 24.460 ms`. Revert validation passed (`15` operator tests, `13` generate tests), and 0.8B p128/d32 recovered to `194.311 ms`.
- Iter102: LMHead top1 shared-hidden staging was token-correct (`test_linear.py -k top1` -> `4 passed`, Qwen3.5 operator tests -> `15 passed`) but rejected and reverted. It measured 0.8B p128/d32 graph-on `194.416 ms` versus Iter100 `194.618 ms`, only `0.10%` faster and within noise, while repeating the already rejected Iter21 idea and touching the common LMHead top1 path.
- Iter99: P0 fused update+output was standalone-correct and faster (`0.699 ms -> 0.643 ms` event, NCU `0.953 ms -> 0.875 ms`) but rejected and reverted because production generate regressed: 0.8B p128/d32 `197.393 -> 200.697 ms`, 2B p128/d32 `425.986 -> 428.665 ms`, and 2B p1024/d128 `1829.887 -> 1863.548 ms`; post-revert operator tests passed.
- Iter96: Qwen3.5-local fixed-hidden fused add+rmsnorm for the post-attention pair was token-correct (`17` operator tests, `13` generate tests) but rejected and reverted. It improved 0.8B p128/d32 `207.594 -> 207.292 ms`, but regressed 2B p128/d32 `435.528 -> 435.998 ms` and 2B p1024/d128 `1916.334 -> 1921.390 ms`; post-revert validation passed (`15` operator tests, `13` generate tests).
- Iter95: P0 precomputed intermediate tile standalone sweep found tile24 fastest in repro (`1.411 ms` event time; NCU `1.80 ms`, grid/block `96/256`, achieved occupancy `56.26%`), but the production transfer regressed target graph-on generate. 0.8B p128/d32 moved `207.594 -> 207.871 ms`, 2B p128/d32 `435.528 -> 436.289 ms`, and 2B p1024/d128 `1916.334 -> 1948.154 ms`; reverted to tile16 and revalidated (`15` operator tests, `13` generate tests).
- Iter86: Qwen3.5 FlashInfer decode tuned-shape support for `num_qo_heads=8,num_kv_heads=2,head_dim=256` was token-correct with the temporary operator table, but rejected and reverted. It was slightly positive only on 0.8B p128/d32 (`210.924 -> 210.703 ms`) and regressed 2B p128/d32 (`439.496 -> 439.848 ms`) and 2B p1024/d128 (`1936.460 -> 1938.112 ms`); post-revert validation passed (`15` operator tests, `13` generate tests).
- Iter85: single-token `from_ab` tile64 was token-correct but rejected and reverted. It regressed all checked graph-on cases: 0.8B p128/d32 `210.924 -> 214.712 ms`, 2B p128/d32 `439.496 -> 443.617 ms`, and 2B p1024/d128 `1936.460 -> 1953.464 ms`; post-revert validation passed (`15` operator tests, `13` generate tests).
- Iter84: Qwen3.5 LMHead cuBLASLt tactic retune found the current baseline best for both tested shapes. Best non-baseline explicit candidates were slower (`1.512 ms` vs `1.503 ms` for 0.8B, `3.012 ms` vs `2.991 ms` for 2B), so no operator-table change was made.
- Iter81: fixed-elements add `1024/2048` was token-correct but rejected and reverted. It regressed 0.8B p128/d32 (`211.329 -> 211.636 ms`), was flat on 2B p128/d32, and only helped 2B p1024/d128; post-revert validation passed (`15` operator tests, `13` generate tests).
- Iter80: post-Iter78 `lm_head_top1` default-enable gate rejected. It remains token-correct and optional, but current graph-on gains are only `0.287%` on 0.8B p128/d32, `0.294%` on 2B p128/d32, and `0.268%` on 2B p1024/d128, below the 1% default gate.
- Iter75: fixed-hidden RMSNorm warp-shuffle reduction passed operator tests but failed Qwen3.5 fresh-dump token alignment on 0.8B at decode index 18 (`99550` vs `98846`); reverted to Iter73 shared-memory reduction and revalidated (`15` operator tests, `13` generate tests).
- Iter68: decode `gated_delta_sequence_from_ab` tile64 was token-correct but rejected: 0.8B/2B short p128/d32 were marginally better (`216.601 ms`, `447.790 ms`), but 2B p1024/d128 regressed to `1971.804 ms` vs Iter64 `1969.855 ms`; reverted to Iter64 tile32 and revalidated.
- Iter67: decode `gated_delta_sequence_from_ab` tile16 was token-correct but rejected: 0.8B improved slightly (`216.711 ms`) while 2B regressed (`448.004 ms` vs Iter64 `447.830 ms`); reverted to Iter64 tile32 and revalidated.
- Iter61/62/63: precomputed recurrent sequence retile attempts were token-correct but slower than Iter59 tile16/thread256.

## Ceiling / Plateau Notes

- Target policy refresh: `95%` now means a documented operator-specific ceiling, not raw hardware peak for every kernel. Raw SOL is only a diagnostic target when the kernel has enough work/parallelism and a matching roofline model. Recurrence-bound, tiny-grid, and CUDA-graph replay-chain kernels use a token-safe standalone/replay ceiling plus full-generate acceptance.
- Iter91 NCU for `gated_delta_sequence_from_ab_single_token_128_kernel<bf16,bf16>`: `13.89 us`, memory/DRAM throughput `42.04%`, achieved occupancy `8.18%`, eligible warps/scheduler `0.12`, no-eligible `88.97%`. This improves Iter89 by removing barriers and memory passes, but remains low-wave/tiny-grid limited.
- Iter76 NCU for decode fixed-hidden RMSNorm: `3.84 us`, SM throughput `0.49%`, memory throughput `1.29%`, achieved occupancy `16.55%`, grid `1`, block `256`; this path is tiny-grid/launch limited, and the accepted win came from preserving the reduction tree while lowering loop overhead.
- Iter67/Iter68 closed the decode `from_ab` retile path: tile16 helps 0.8B only and tile64 regresses the 2B long case, so tile32 remains the stable fallback tile.
- Optional top1 stage1 NCU is memory-roofline limited: DRAM `97.46%`, achieved occupancy `94.50%`; no more launch-shape tuning planned without a new algorithm.
- Iter93 changed the default decision because the objective shifted from a 1% end-to-end gate to hotspot operator ceilings: Qwen3.5 greedy now defaults to top1, whose stage1 path is memory-roofline limited, while explicit runtime `lm_head_top1=false` still allows full-logits fallback.
- Iter94 `from_ab` column-fused-64 standalone candidate was exact but flat/slower than the accepted 128-thread production schedule (`7.387 us` vs `7.382 us` in repro), so it was not transferred.
- Iter95 confirms that prefill standalone event-time wins are not sufficient for acceptance: tile24 was slightly better than tile16 in isolated repro/NCU, but full Qwen3.5 generate regressed, especially 2B long-prefill. Iter100 later moved to tile32 only after shared q/k staging changed the production transfer behavior and all full target benchmarks improved.
- Iter103 confirms the same for fused update/output after tile32: the standalone repro improved again, but full generate regressed, so the Iter100 separate vectorized update plus output-dot phase is the current production ceiling for this layout.
- Iter96 confirms the add+rmsnorm fusion path remains a cross-model reject even after fixed-hidden RMSNorm and top1 default changes; keep the separate add plus fixed-hidden RMSNorm path.
- Iter107 confirms dense-linear cuBLASLt tactic records are worth accepting only when the local tuner margin is large and shape-family transfer is broad. The 0.8B `mlp_down` `~1.2%` local win was rejected in Iter106, while the 2B `mlp_down`/generic-linear `~9%-15%` local wins transferred to `~2.7%-3.2%` full-matrix 2B gains.
- Iter108 closes the remaining small 2B `2048->512` linear tactic as a local no-op: even the best `self_attn_v` candidate had only `4.45%` local margin and produced a `0.02%` p128/d32 graph-on delta, so it is below the production-table acceptance threshold.
- Iter109 closes pure cuBLASLt explicit tactic search for the largest remaining 2B decode linear shapes. `fused_gate_up`, `qwen3_linear_qkv`, and `self_attn_q` are baseline-best under the measured library tactic ceiling; further dense-linear wins need an algorithmic change rather than a table record.
- Iter121/122 closes the low-risk GateUp GEMV+SwiGLU path for the current exact BF16 algorithm: the Qwen3.5-safe warp GEMV transfers to `~0.5%-0.6%` p128 graph-on gains, and NCU then shows the accepted kernel at DRAM `95.34%`. Larger GateUp wins need fewer bytes or a different contract, not another launch-shape tweak.
- Iter110 confirms the default greedy `lm_head_top1` stage1 is at the DRAM bandwidth ceiling (`97.60%`) with high active-warps utilization (`97.79%`). Further LMHead gains require reading fewer bytes or changing the search problem; launch-shape/shared-hidden variants are closed.
- P0 precomputed recurrent tiling is currently accepted at Iter105 `tile32/thread256` with qmem-output algebra, after tile8, tile16/thread128, tile64/thread128, Iter95 tile24, Iter99/103 fused update+output, and Iter104 thread512 either regressed or failed to transfer. Iter97 vectorized state decay/update (`1.84 ms -> 1.50 ms` standalone NCU), Iter98 moved q/k loads into shared memory (`1.50 ms -> 0.953 ms` standalone NCU), Iter100 reduced the production P0 NCU to `0.803 ms`, and Iter105 reduced it again to `0.706 ms` while preserving token alignment.
- P0's remaining raw hardware SOL gap is still not a raw-peak failure target. This kernel is recurrence-bound with token barriers and limited waves; the valid ceiling for this phase is the best token-safe current-layout standalone/production result plus full-generate acceptance. Iter105 is now the current-layout ceiling candidate. Further gains require refreshed NSYS/NCU evidence; likely next steps are a new P0 recurrence layout/barrier algorithm or a different hotspot if P0 no longer dominates.
- Large-hidden RMSNorm and recurrent math/order edits remain token-sensitive; Iter75 proves even tolerance-safe RMSNorm reduction-order changes can drift tokens, so future norm changes need either bitwise-preserving order or full fresh-dump gates before benchmark.

## Next Action

- Active resumed work: local kernel phase is closed after Iter122. The local P0, pure dense-linear table tuning, LMHead launch-shape/row-norm pruning, existing SM86 SwiGLU, existing prefill SwiGLU, GateUp GEMV+SwiGLU, fixed RMSNorm/add, and FlashInfer tuned-shape paths are either accepted at operator-specific ceiling or blocked with evidence. Remaining material work is a larger standalone/Humanize+KernelPilot search: compressed/custom GEMV+activation/projection design, larger decode-chain fusion, or a true LMHead byte-reduction/indexing method.
- Production transfer rule: migrate only if the standalone/profile candidate is correct and faster, then pass Qwen3.5 0.8B/2B fresh-dump alignment and affected graph-on benchmarks. If two consecutive standalone candidates are correct but weak, expand KernelPilot source research before the next edit.
- Keep every accepted production change gated by `tests/operators/test_qwen3_5_runtime_ops.py` and Qwen3.5 0.8B/2B fresh-dump generate alignment.
