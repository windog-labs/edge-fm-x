# Qwen3.5 Phase 2 Current State

- Updated: `2026-05-22T14:46:14+08:00`
- Scope: EdgeFM-only optimization. TRT comparison remains blocked until a TensorRT-Edge-LLM Qwen3.5 linear-attention port exists.
- Scope status: local kernel ceiling/blocker phase completed; no new P0/LMHead long-loop is in progress.
- Current code path: Qwen3.5 CUDA graph enabled; greedy temperature=0 uses Iter93 default `lm_head_top1` unless runtime explicitly disables it. GatedDeltaNet prefill uses q/k precompute, decay precompute, and Iter97 precomputed recurrent sequence `value_tile=16`, `threads=256` with vectorized state decay/update; decode uses Iter91 `gated_delta_sequence_from_ab_single_token_128` fast path with 128-thread column-fused decay/delta/update/output for `seq_len==1,key_dim=value_dim=128`, Iter78 static full-tile fallback for 128/128 non-single-token shapes, and the Iter64 generic `from_ab` kernel fallback for other shapes. Iter71 uses in-place residual add (`hidden += mixer_output`); Iter73/76 specialize non-gated large-hidden RMSNorm for hidden `1024/2048`; decode conv+state fused kernel is enabled; small-hidden RMSNorm uses Iter49 one-warp-per-row.

## Current Correctness

- Latest full gate after Iter97 acceptance: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Shared-path regression after Iter93: `tests/engine/test_qwen2_generate.py -k "token_alignment or kvcache" -s` -> `8 passed, 12 deselected`.
- Long-prefill graph/regular drift remains fixed by Iter39 token-boundary barriers; graph reuse stays enabled.

## Current Performance Facts

| Case | Avg ms | Prefill ms | Decode step ms | Notes |
|---|---:|---:|---:|---|
| 0.8B p128/d32 graph-on latest accepted | 205.148 | 31.394 | 5.600 | Iter97 P0 vectorized state decay/update |
| 2B p128/d32 graph-on latest accepted | 433.192 | 44.654 | 12.529 | Iter97 P0 vectorized state decay/update |
| 2B p1024/d128 graph-on validation | 1897.618 | 296.433 | 12.605 | Iter97 P0 vectorized state decay/update |
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

## Latest NCU

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

Latest mapping trace is Iter94 graph-off NSYS, 0.8B p128/d32, post-Iter93. P0 timings in this table predate Iter97 and should be read together with the Iter97 acceptance gate above.

| Stage | Hotspot | Time | Share | Decision |
|---|---|---:|---:|---|
| Prefill | `gated_delta_sequence_precomputed_kernel<bf16,true,16>` | 24.273 ms | 69.2% | Iter97 standalone proof accepted; current remaining gap is recurrence/layout blocker |
| Decode | `lm_head_top1::stage1_kernel<bf16>` | 46.174 ms | 25.5% | Memory-roofline path; previous NCU DRAM `97.46%`, achieved occupancy `94.50%` |
| Decode | `gated_delta_sequence_from_ab_single_token_128_kernel` | 4.931 ms | 2.7% | Iter91 accepted; Iter94 column-fused-64 repro rejected as flat |
| Decode | fixed-hidden RMSNorm | 3.340 ms | 1.8% | Iter76 accepted; Iter75 warp-reduce rejected because token drifted |
| Decode | FlashInfer attention | 2.958 ms | 1.6% | Iter86 Qwen3.5 tuned-shape table was token-correct but cross-model regressed; keep default |
| Decode | `add_kernel<bf16>` | 2.127 ms | 1.2% | Iter81 fixed add rejected; keep generic add |

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

## Recent Rejections

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
- Iter95 confirms that prefill standalone event-time wins are not sufficient for acceptance: tile24 was slightly better than tile16 in isolated repro/NCU, but full Qwen3.5 generate regressed, especially 2B long-prefill. Keep tile16 as the production P0 layout unless a new candidate improves full target benchmarks.
- Iter96 confirms the add+rmsnorm fusion path remains a cross-model reject even after fixed-hidden RMSNorm and top1 default changes; keep the separate add plus fixed-hidden RMSNorm path.
- P0 precomputed recurrent tiling is locally converged at Iter59 after tile8, tile16/thread128, tile64/thread128, and Iter95 tile24 production transfer all regressed. Iter97 is the final accepted local edit in the current layout: vectorized state decay/update improved standalone NCU `1.84 ms -> 1.50 ms` and target graph-on cases while preserving token alignment.
- P0's remaining raw hardware SOL gap is a documented blocker, not a failed optimization target. This kernel is recurrence-bound with token barriers and limited waves; the valid ceiling for this phase is the best token-safe current-layout standalone/production result plus full-generate acceptance. Reaching a materially higher ceiling now requires a new state layout or algorithmic recurrence rewrite in a separate long-loop.
- Large-hidden RMSNorm and recurrent math/order edits remain token-sensitive; Iter75 proves even tolerance-safe RMSNorm reduction-order changes can drift tokens, so future norm changes need either bitwise-preserving order or full fresh-dump gates before benchmark.

## Next Action

- Stage stop point: the requested local ceiling/blocker phase is complete. Current material hotspots are either at a documented operator-specific ceiling or have rejected local candidates: P0 prefill is at the best token-safe current-layout result after Iter97, default greedy LMHead is on the memory-roofline top1 path, `from_ab` column-fused-64 is flat, fixed RMSNorm/attention/add have rejected cross-model candidates.
- Do not continue optimization automatically from here. The next meaningful step, if resumed later, is a larger standalone/Humanize + KernelPilot search for a new algorithmic route, most likely P0 prefill layout/barrier reduction or an LMHead/top1 algorithmic replacement. Keep any production transfer gated by Qwen3.5 0.8B/2B fresh-dump alignment and the 12-case matrix.
- Keep every accepted production change gated by `tests/operators/test_qwen3_5_runtime_ops.py` and Qwen3.5 0.8B/2B fresh-dump generate alignment.
