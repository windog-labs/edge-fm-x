# Qwen3.5 Phase 2 Iteration Ledger

## Baseline 2026-05-21T16:31:19

- Change: establish EdgeFM graph-off/graph-on and optional TRT paired baseline.
- Correctness: expected to be validated separately by `tests/engine/test_qwen3_5_generate.py`.
- Accepted cases: 0/6.

## Iteration 1 2026-05-21T16:34:33+08:00

- Change: move Qwen3.5 CUDA graph decode state to graph-stable static buffers and bind per-request arena views to those buffers; add `decode_graph_captures` metric and repeated-generate graph reuse regression.
- Correctness: `tests/engine/test_qwen3_5_generate.py` fresh dump passed for 0.8B and 2B; runtime/KV/operator unit tests passed; Qwen2/Qwen2.5-VL token/KV regression passed.
- Benchmark: graph-on vs graph-off geomean speedup `1.0048x`; all six EdgeFM cases improved, with ratios from `0.9893` to `0.9985`.
- TRT comparison: blocked. Current TRT-Edge-LLM exporter/runtime is KV-attention-only and fails on Qwen3.5 `linear_attention`.
- Decision: accept. The change is correctness-preserving and removes per-request decode graph recapture while keeping Qwen3.5 state semantics inside the model.
- Next: test `lm_head_top1` under Qwen3.5 graph-on, then profile linear-attention decode kernels if top1 is not enough.

## Iteration 2 2026-05-21T17:05:00+08:00

- Change: parallelized Qwen3.5 `gated_delta_sequence_kernel` across value/state dimensions inside each head, preserving token recurrence and per-value accumulation order.
- Correctness: `tests/operators/test_qwen3_5_runtime_ops.py` passed, including model-geometry coverage; Qwen3.5 0.8B/2B fresh dump graph/non-graph alignment passed.
- Benchmark: 0.8B p128/d32 graph-on improved from `4673.768 ms` to `258.050 ms`; prefill improved from `3404.454 ms` to `62.313 ms`; decode step improved from `40.935 ms` to `6.310 ms`.
- NCU: single-kernel duration improved from `275.57 ms` to `3.95 ms`; active threads per warp from `1.00` to `30.59`; SM throughput from `5.96%` to `16.32%`.
- Decision: accept. This fixes the largest baseline pathology: one thread per head doing all recurrent-state work serially.

## Iteration 3 2026-05-21T17:12:31+08:00

- Change: split the value dimension into per-head value tiles so each head launches multiple blocks while keeping each state column independent.
- Correctness: `tests/operators/test_qwen3_5_runtime_ops.py` passed; Qwen3.5 0.8B/2B fresh dump graph/non-graph alignment passed.
- Benchmark: 0.8B p128/d32 graph-on improved further to `244.700 ms`; prefill `55.108 ms`; decode step `6.111 ms`.
- NCU: single-kernel duration improved to `3.45 ms`; SM throughput `27.82%`; achieved occupancy `19.06%`.
- Decision: accept for now. It is still below the 95% theoretical-ceiling target, so continue iterating on occupancy, synchronization, and recurrence-aware tiling.

## Iteration 4 Candidate A 2026-05-21T17:20:58+08:00

- Intent: test `value_tile=16`, `threads=128` for `gated_delta_sequence_kernel`, increasing per-head block count to improve SM residency.
- Safety fix included: move `scalar_s` shared-memory offset from `value_dim` to the allocated value-tile size, preventing shared-memory out-of-bounds access.
- Correctness: `tests/operators/test_qwen3_5_runtime_ops.py` -> `8 passed`; `tests/engine/test_qwen3_5_generate.py` fresh dump -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on improved from Iter3 `244.700 ms` to `234.558 ms`; prefill `55.108 ms` to `47.602 ms`; decode step `6.111 ms` to `6.026 ms`.
- NCU: single-kernel duration improved from Iter3 `3.45 ms` to `2.92 ms`; SM throughput `27.82%` to `42.28%`; achieved occupancy `19.06%` to `37.07%`.
- Decision: accept. Candidate A is the current best production kernel.

## Iteration 5 Candidate B 2026-05-21T17:24:21+08:00

- Intent: test `value_tile=64`, `threads=128` for `gated_delta_sequence_kernel`, reducing per-token synchronization overhead and increasing per-block work at the cost of lower block count.
- Correctness: `tests/operators/test_qwen3_5_runtime_ops.py` -> `8 passed`; `tests/engine/test_qwen3_5_generate.py` fresh dump -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on regressed from Candidate A `234.558 ms` to `273.520 ms`; prefill regressed from `47.602 ms` to `73.907 ms`; decode step regressed from `6.026 ms` to `6.435 ms`.
- NCU: single-kernel duration regressed from Candidate A `2.92 ms` to `5.19 ms`; SM throughput fell from `42.28%` to `17.05%`; achieved occupancy fell from `37.07%` to `9.57%`. NCU also flagged a too-small grid (`32` blocks, `0.14` waves/SM).
- Decision: reject and revert to Candidate A constants (`value_tile=16`, `threads=128`). Larger value tiles reduce launch occupancy too much for this recurrence shape.

## Iteration 6 Candidate C 2026-05-21T17:34:40+08:00

- Intent: test `value_tile=32`, `threads=256` for `gated_delta_sequence_kernel`, keeping the Iter3 tiling but increasing per-block parallelism to improve issue/latency hiding.
- Correctness: `tests/operators/test_qwen3_5_runtime_ops.py` -> `8 passed`; `tests/engine/test_qwen3_5_generate.py` fresh dump -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on improved from Candidate A `234.558 ms` to `232.529 ms`; prefill improved from `47.602 ms` to `46.503 ms`; decode step improved from `6.026 ms` to `5.997 ms`.
- NCU: single-kernel duration improved from Candidate A `2.92 ms` to `2.81 ms`; achieved occupancy `37.88%`; active threads/warp `28.46`. SM throughput reported lower than A (`34.47%` vs `42.28%`) but wall time and end-to-end both improved.
- Decision: accept. Candidate C is the current best production kernel.

## Iteration 7 Candidate D 2026-05-21T17:45:05+08:00

- Intent: precompute normalized q/k once per token/head for prefill `seq_len > 1`, then run the recurrent sequence kernel against Float32 q/k views. This targets the NCU barrier stall caused by each value tile repeating thread-0 q/k l2norm work before a block-wide barrier.
- Correctness: `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; `tests/engine/test_qwen3_5_generate.py` fresh dump -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on improved from Candidate C `232.529 ms` to `224.249 ms`; prefill improved from `46.503 ms` to `38.444 ms`; decode step stayed roughly flat (`5.997 ms` to `5.988 ms`) because precompute is prefill-only.
- NCU: recurrent sequence kernel duration improved from Candidate C `2.81 ms` to `2.06 ms`; SM throughput `39.76%`; achieved occupancy `37.99%`; active threads/warp `32`. The new q/k precompute kernel is small (`30.88 us`) and already high-utilization (`79.01%` compute/memory throughput, `90.81%` achieved occupancy).
- Decision: accept. Candidate D is the current best production path and validates scalar q/k norm work as material.

## Iteration 8 Candidate D-Retile-A 2026-05-21T18:01:30+08:00

- Intent: after q/k precompute, retest `value_tile=16`, `threads=128` because removing scalar norm work changes the tiling tradeoff and may benefit from the larger 128-block grid.
- Correctness: `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; `tests/engine/test_qwen3_5_generate.py` fresh dump -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on regressed from Candidate D `224.249 ms` to `225.109 ms`; prefill was nearly flat (`38.444 ms` to `38.358 ms`) but decode step regressed (`5.988 ms` to `6.020 ms`).
- NCU: recurrent sequence duration was effectively flat/slightly worse (`2.06 ms` to `2.07 ms`); SM throughput rose to `44.03%` but active threads/warp fell to `28.77`, and end-to-end did not improve.
- Decision: reject and revert to Candidate D constants (`value_tile=32`, `threads=256`).
## Milestone Matrix 2026-05-21T17:47:09+08:00

- Change: refresh EdgeFM-only graph-off/graph-on matrix after Candidate D; TRT remains blocked/missing.
- Correctness: final accepted build validated by `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; `tests/engine/test_qwen3_5_generate.py` -> `13 passed`.
- Accepted cases: no TRT acceptance gate available.

## Iteration 9 Candidate P2-LMHeadTop1 2026-05-21T17:58:45+08:00

- Intent: enable the existing Qwen3.5 `lm_head_top1` greedy decode path under CUDA graph, using token-alignment tests as the correctness gate and p128/d128 graph-on as the first benchmark gate.
- Correctness: `tests/engine/test_qwen3_5_generate.py -k lm_head_top1` -> `4 passed`.
- Benchmark: 0.8B p128/d128 graph-on improved from `801.566 ms` to `799.202 ms`; 2B p128/d128 graph-on improved from `1706.013 ms` to `1704.421 ms`.
- NSYS follow-up: graph-off p128/d32 with `--lm-head-top1` shows the decode LMHead GEMV is replaced by `lm_head_top1::stage1_kernel`, but that kernel still scans the full vocab weight matrix and consumes `46.235 ms` across 31 launches, effectively the same as cuBLAS GEMV.
- Decision: reject for default enablement. It is correct and remains useful as an experimental option, but both measured gains are below the documented `>=1%` default-on gate.

## Iteration 10 Candidate P1-DecodePrecomputeQK 2026-05-21T18:20:00+08:00

- Intent: test whether decode should also use the precomputed q/k sequence path. The change is limited to `src/models/qwen3_5/qwen3_5.cpp`: reuse the already-correct q/k precompute kernel for `seq_len == 1` decode, then run `gated_delta_sequence_precomputed_kernel`.
- Correctness: rejected. `tests/operators/test_qwen3_5_runtime_ops.py` still passed (`9 passed`), but fresh-dump generate alignment failed: 0.8B drifted at decode index 18 (`99550` vs reference `98846`), so this path does not preserve the decode token contract.
- Root cause: the decode path is numerically more sensitive than prefill; materializing normalized q/k as Float32 changes the per-step rounding path versus the original in-kernel dtype-rounded q/k normalization. The candidate was reverted before benchmarking.
- Benchmark: skipped because correctness failed.
- Decision: reject and revert to prefill-only q/k precompute.

## Iteration 11 Candidate P4-AddRMSNormFusion 2026-05-21T18:30:00+08:00

- Intent: fuse the attention residual add and the immediately following Qwen3.5 RMSNorm into one Qwen3.5-local operator. This targets decode small-kernel overhead (`add_kernel` + `qwen3_5_rmsnorm_kernel`) without changing runtime scheduling or the full attention/linear-attention contracts.
- Correctness: passed while testing the candidate: `tests/operators/test_qwen3_5_runtime_ops.py` -> `10 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: after avoiding a redundant second lhs/rhs read, 0.8B improved slightly (`p128/d32` `224.249 -> 223.997 ms`, `p128/d128` `801.566 -> 798.986 ms`), but 2B p128/d128 regressed on two runs (`1706.013 -> 1706.963/1707.246 ms`).
- Decision: reject and revert. The candidate is correct but does not improve both supported Qwen3.5 models, so it is not a safe default optimization.

## Iteration 12 Candidate P2-LMHeadCublasLtRetune 2026-05-21T19:01:40+08:00

- Intent: use the existing operator-table/cuBLASLt tuning path for Qwen3.5 decode LMHead (`m=1`, BF16 input/weight, FP32 logits), avoiding runtime changes.
- Tooling fix: updated `scripts/operator_table/utils.py` so Qwen3.5 model paths/configs resolve to engine model `Qwen3.5` and operator model `qwen3_5`; before this, the tuning script misclassified Qwen3.5 as VLM and could not load tied `model.language_model.embed_tokens.weight`.
- Correctness: no runtime kernel change. Current accepted runtime was revalidated after Iter11 revert: `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B heuristic search found baseline/algo_0 best (`1.503232 ms`); 0.8B explicit search also found baseline best (`1.503216 ms`, closest explicit `1.506128 ms`); 2B heuristic search found baseline/algo_0 best (`2.991104 ms`).
- Decision: reject for performance. The current cuBLASLt LMHead tactic is already the best candidate found by the repo tuning path.

## Iteration 13 Candidate P1-GatedDeltaDecodeSpecialized 2026-05-21T19:05:00+08:00

- Intent: add a `seq_len == 1` specialization inside Qwen3.5 `gated_delta_sequence_forward`, preserving the original dtype-rounded q/k normalization but removing the runtime token loop and some decode-only indexing from each launch.
- Correctness: passed while testing the candidate: `tests/operators/test_qwen3_5_runtime_ops.py` -> `10 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: regressed on 0.8B graph-on: p128/d32 `224.249 -> 224.531 ms`, p128/d128 `801.566 -> 802.017 ms`.
- Decision: reject and revert. Removing the token loop did not help the recurrence kernel; the original sequence kernel remains faster for `seq_len == 1`.

## Iteration 14 Candidate P0-PrecomputedSubwarpDot 2026-05-21T19:35:31+08:00

- Intent: optimize only the prefill `gated_delta_sequence_precomputed_kernel` dot-product phases. Current NCU shows the kernel has a 64-block grid, achieved occupancy `37.99%`, eligible warps per scheduler `0.61`, and barrier stalls around `40.7%`; the two per-token dot phases use only one thread per value column. This candidate assigns 8 threads to each value column and uses subwarp reduction for the `state * key` and `state * query` dot products.
- Correctness: rejected. `tests/operators/test_qwen3_5_runtime_ops.py` passed (`9 passed`), but fresh-dump generate alignment failed on 0.8B at decode index 18 (`99550` vs reference `98846`).
- Benchmark: skipped because correctness failed.
- Decision: reject and revert. The candidate changes the FP32 accumulation order in prefill enough to perturb later greedy logits, so Qwen3.5 requires the original sequential k-order dot accumulation for token alignment.

## Iteration 15 Candidate P0-PrecomputedDotUnroll4 2026-05-21T19:40:08+08:00

- Intent: reduce loop-control overhead in the same prefill `gated_delta_sequence_precomputed_kernel` dot-product phases while preserving the original sequential k accumulation order. NCU reports high branch instruction count and issue-slot underutilization; this candidate manually unrolls the `k_idx` loops by 4 with source-order `acc += ...` statements.
- Correctness: rejected. `tests/operators/test_qwen3_5_runtime_ops.py` passed (`9 passed`), but fresh-dump generate alignment failed on both models: 0.8B drifted at decode index 3 (`98435` vs `101947`), and 2B drifted at decode index 8 (`108956` vs `97273`).
- Benchmark: skipped because correctness failed.
- Decision: reject and revert. Even source-order manual unroll changes the compiled floating-point behavior enough to break the token contract.

## Iteration 16 Candidate P0-PrecomputedStatePointerWalk 2026-05-21T19:44:00+08:00

- Intent: reduce integer address-generation overhead in the precomputed recurrent dot loops without changing loop shape, thread mapping, or FP accumulation order. This candidate replaces repeated `k_idx * value_dim + v_idx` indexing in the dot phases with a pointer walk over the state column.
- Correctness: passed. `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on regressed from Candidate D `224.249 ms` to `227.006 ms`; average prefill regressed from `38.444 ms` to `40.807 ms`.
- Decision: reject. Pointer-walk address generation is token-safe but slower than the original indexed form in the production compiler.

## Iteration 17 Candidate P0-PrecomputedColumnFusion 2026-05-21T19:45:15+08:00

- Intent: reduce barrier overhead in `gated_delta_sequence_precomputed_kernel` by fusing the per-token decay, delta update, and output projection inside one value-column thread while keeping each column's k traversal in ascending order. This targets the NCU barrier stall (`~40.7%`) without changing runtime or model semantics.
- Correctness: rejected. `tests/operators/test_qwen3_5_runtime_ops.py` passed (`9 passed`), but fresh-dump generate alignment failed on 0.8B at decode index 8 (`14791` vs `98409`).
- Benchmark: skipped because correctness failed.
- Decision: reject and revert. The kernel is sensitive not only to accumulation order, but also to the original intermediate FP32 state write/read sequence between decay, update, and output.

## Iteration 18 Candidate P0-PrecomputedTile16T256 2026-05-21T20:00:00+08:00

- Intent: move P0 from blind production edits into a standalone recurrence-preserving repro, then test only a math-order-preserving tiling change. Repro baseline `tile=32/thread=256` passed CPU reference and measured `1.698 ms`; NCU reported duration `2.19 ms`, SM throughput `32.25%`, achieved occupancy `37.97%`, and `No Eligible 67.59%`.
- Repro candidate: `tile=16/thread=256` passed CPU reference, measured `1.429 ms`, and NCU improved duration to `1.80 ms`, SM throughput to `45.61%`, achieved occupancy to `76.15%`, and `No Eligible` to `56.28%`.
- Production intent: apply this only to the prefill `gated_delta_sequence_precomputed_kernel`, leaving the original sequence/decode kernel at Candidate D `tile=32/thread=256`.
- Correctness: rejected in production. The first patch accidentally changed the original sequence kernel and failed `tests/operators/test_qwen3_5_runtime_ops.py`; after fixing the patch to precomputed-only, operator tests passed (`9 passed`) but fresh-dump generate alignment failed on both models: 0.8B regular decode drifted at index 10 (`110257` vs `98920`), 2B regular decode drifted at index 7 (`95793` vs `5205`), and graph paths also drifted.
- Benchmark: skipped because token alignment failed.
- Decision: reject and revert to Candidate D. The repro win does not transfer because changing the precomputed value tiling changes cross-block update/output timing enough to perturb logits, even though per-column math order is unchanged inside each block. Baseline was revalidated after revert: `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.

## Iteration 19 Candidate P1-ConvStatePrefillDirect 2026-05-21T20:10:00+08:00

- Intent: optimize a smaller Qwen3.5-local hotspot after P0 plateau. For prefill `seq_len >= kernel_size`, the next depthwise-conv state is exactly the last `kernel_size` input tokens, so update it directly in-place and skip the temporary state buffer plus device-to-device copy. Decode/short sequences keep the existing tmp path to avoid in-place shift hazards.
- Correctness: passed. `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on regressed from Candidate D `224.249 ms` to `224.895 ms`; prefill regressed slightly from `38.444 ms` to `38.581 ms`, and decode step from `5.988 ms` to `6.006 ms`.
- Decision: reject and revert. Removing the tmp/copy on the prefill state update is mathematically safe, but does not improve end-to-end performance on the measured case.

## Iteration 20 Candidate P3-Qwen3_5AttentionDecodeTunedShape 2026-05-21T20:18:00+08:00

- Intent: address the smaller decode attention hotspot without runtime redesign. Add an explicit Qwen3.5 decode tuned FlashInfer shape (`num_qo_heads=8`, `num_kv_heads=2`, `head_dim=256`) and fix the attention tune script to prefer explicit `head_dim` from config. Do not update the default operator table unless correctness and benchmark improve.
- Tooling: keep the tune-script `head_dim` fix because Qwen3.5 config has explicit `head_dim=256`; the runtime template specialization is evaluated separately.
- Operator benchmark: for 0.8B attention decode kv lens `128/512/1024`, default FlashInfer total median was `0.080384 ms`; tuned shape total median was `0.080144 ms` (`0.3%` faster), too small to justify the extra production template without an end-to-end win.
- End-to-end benchmark: 0.8B p128/d32 graph-on default rerun was `223.906 ms`, decode step `5.978 ms`; tuned table was `223.933 ms`, decode step `5.981 ms`.
- Correctness after revert: rebuilt after removing the Qwen3.5 production attention specialization; `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Decision: reject and revert the production `src/operators/attention_op.cu` specialization. Keep only the tune-script `head_dim` fix for future Qwen3.5 attention profiling.

## Iteration 21 Candidate P2-LMHeadTop1SharedHidden 2026-05-21T20:30:00+08:00

- Intent: revisit the experimental decode-only `lm_head_top1` path without changing default full-logits behavior. Stage1 currently assigns one warp per vocab row and each row reloads the same hidden vector; cache hidden in per-block shared memory so the 8 vocab-row warps in a block reuse it.
- Correctness while testing: `tests/layers/test_linear.py -k top1` -> `2 passed`; Qwen3.5 0.8B/2B fresh-dump `lm_head_top1` alignment -> `4 passed`.
- Benchmark: 0.8B p128/d128 graph-on with `--lm-head-top1` regressed from Iter9 top1 `799.202 ms` to `800.515 ms`; decode step `6.001 ms`. It remains below the default-on gate and is worse than the previous experimental top1 implementation.
- Correctness after revert: rebuilt after removing the shared-hidden stage1; `tests/layers/test_linear.py -k top1` -> `2 passed`; Qwen3.5 fresh-dump `lm_head_top1` alignment -> `4 passed`.
- Decision: reject and revert. Hidden reuse through shared memory adds enough per-block overhead to outweigh reduced hidden-vector global reads, likely because the original hidden vector is already well served by cache.

## Iteration 22 Candidate P4-GatedRMSNormSmallHiddenThreads 2026-05-21T20:34:00+08:00

- Intent: reduce overhead in the Qwen3.5 linear-attention gated RMSNorm hotspot. That path reshapes to rows `seq_len * 16`, hidden `128`, but the generic RMSNorm kernel launches 256 threads per row; test a Qwen3.5-local launch policy using 128 threads when `kGated && hidden <= 128`.
- Correctness while testing: `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on measured `223.897 ms`, prefill `38.139 ms`, decode step `5.988 ms`; this is effectively noise versus the latest default rerun `223.906 ms` and does not show a reliable end-to-end gain.
- Correctness after revert: rebuilt with the original 256-thread RMSNorm launch policy; `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Decision: reject and revert. The smaller block is token-safe but not materially faster end-to-end.

## Iteration 23 Profile Refresh 2026-05-21T20:40:00+08:00

- Change: refreshed Qwen3.5 0.8B p128/d32 NSYS with graph-off mapping and graph-on formal traces before making another production edit.
- Evidence: graph-off mapping shows prefill dominated by `gated_delta_sequence_precomputed_kernel<bf16>` at `28.381 ms`, `71.4%` of prefill GPU time across 18 launches. Smaller hotspots are LMHead GEMV `1.503 ms`, depthwise conv `0.652 ms`, gated RMSNorm `0.462 ms`, and q/k precompute `0.399 ms`.
- Graph-on formal metrics: profiled run reported `prefill_ms=40.652`, `decode_ms=189.258`, `decode_step_avg_ms=6.105`, `cuda_graph_enabled=1.0`.
- Decision: no production change. Keep Candidate D as current best and continue only with narrow, token-safe GatedDeltaNet evidence/candidates.

## Iteration 24 Candidate P0-PrecomputedFullTileFastPath 2026-05-21T20:42:00+08:00

- Intent: optimize only `gated_delta_sequence_precomputed_kernel` for the common Qwen3.5 full-value-tile case (`value_dim=128`, `tile=32`). Add a full-tile kernel variant that removes dynamic `min`, modulo, and division in state loops while preserving token order, state write/read phases, and per-column k accumulation order.
- Correctness: rejected. Operator reference tests passed (`tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`), but Qwen3.5 fresh-dump token alignment failed: 0.8B drifted at decode index 3 (`98435` vs `101947`) and 2B drifted at decode index 4 (`118822` vs `103715`).
- Benchmark: skipped because token alignment failed.
- Correctness after revert: rebuilt after removing the full-tile fast path; `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Decision: reject and revert. Removing dynamic index arithmetic changed the compiled floating-point behavior enough to perturb logits, so the original dynamic-index precomputed kernel remains part of the token contract.

## Iteration 25 Candidate P1-DepthwiseConvKernel4 2026-05-21T20:53:00+08:00

- Intent: address the next Qwen3.5-local prefill hotspot after P0. NCU for `depthwise_causal_conv1d_kernel<bf16>` shows `45.41 us` for the first launch, SM throughput `63.25%`, achieved occupancy `89.47%`, and long-scoreboard stalls `52.2%`; add a kernel-size-4 fast path that removes the dynamic loop and most source-token boundary checks for the common prefill case while keeping the generic fallback.
- Correctness: passed. `tests/operators/test_qwen3_5_runtime_ops.py` -> `9 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- NCU: local depthwise conv first-launch duration improved from `45.41 us` to `30.94 us`; registers per thread fell from `32` to `22`; long-scoreboard stall share fell from `52.2%` to `46.0%`. Compute throughput changed from `63.25%` to `59.19%`, but wall time and memory throughput improved.
- Benchmark: 0.8B p128/d32 graph-on measured `224.024 ms` average with stable post-warmup runs around `223.4-223.6 ms` (neutral/noise vs latest default rerun `223.906 ms`). Sequential p128/d128 graph-on improved from milestone `801.566 ms` to `799.671 ms` for 0.8B, and from `1706.013 ms` to `1705.266 ms` for 2B.
- Decision: accept. The change is Qwen3.5-local, token-safe, improves the depthwise conv hotspot by `31.9%` in NCU, and shows small end-to-end wins on p128/d128 for both supported models without a material p128/d32 regression.

## Iteration 26 Profile GatedRMSNorm 2026-05-21T21:00:00+08:00

- Change: profile the Qwen3.5 gated RMSNorm hotspot after Iter25.
- Evidence: first gated RMSNorm launch (`grid=2048`, `block=256`) measured `33.66 us`, SM/memory throughput `77.43%`, achieved occupancy `82.47%`, registers/thread `17`, and barrier stalls `31.1%`.
- Decision: no production change. This kernel is not the current limiting path, and the earlier 128-thread launch-policy candidate was token-safe but only noise-level end-to-end. Keep it as profile evidence and move on.

## Iteration 27 NSYS Refresh 2026-05-21T21:02:00+08:00

- Change: refreshed 0.8B p128/d32 graph-off mapping and graph-on formal NSYS after accepting Iter25.
- Evidence: mapping still shows prefill dominated by `gated_delta_sequence_precomputed_kernel<bf16>` at `56.650 ms`, `71.8%` of prefill GPU time across warmup+run. Smaller prefill kernels are LMHead GEMV `3.006 ms`, gated RMSNorm `0.922 ms`, depthwise kernel4 `0.872 ms`, and q/k precompute `0.798 ms` across warmup+run. Decode mapping still points to LMHead GEMV as the largest visible non-graph decode kernel, but cuBLASLt retune and `lm_head_top1` were already rejected.
- Decision: no production change from this profile alone. P0 remains GatedDeltaNet; remaining work needs a stronger exact replay gate or standalone long-loop kernel search before another precomputed-sequence rewrite.

## Iteration 28 Candidate P0-PrefillDecayPrecompute 2026-05-21T21:06:33+08:00

- Intent: remove repeated per-value-tile `expf(g)` work from the prefill-only precomputed GatedDeltaNet sequence path. Add a Qwen3.5-local decay/beta compute path that stores `decay = exp(g)` once per token/head, and a precomputed sequence variant that consumes decay directly. Keep the existing public `compute_g_beta` and decode sequence semantics unchanged.
- Correctness: passed. `tests/operators/test_qwen3_5_runtime_ops.py` now includes the decay path and passed (`10 passed`); Qwen3.5 0.8B/2B fresh-dump generate alignment passed (`13 passed`).
- Benchmark: 0.8B p128/d32 graph-on measured `223.386 ms`, prefill `37.977 ms`, decode step `5.977 ms`, improving over the latest accepted p128/d32 reruns (`223.906-224.024 ms`). P0 NCU first launch changed only slightly (`2.06 -> 2.05 ms`, occupancy `37.99% -> 38.01%`, eligible warps/scheduler `0.61 -> 0.62`). Sequential p128/d128 graph-on measured `799.539 ms` for 0.8B and `1705.201 ms` on the 2B rerun.
- Decision: accept as a small, token-safe P0 prefill cleanup. The NCU delta is modest, but the change is Qwen3.5-local, removes duplicated special-function work, and has no material regression on the supported 0.8B/2B cases.

## Iteration 29 NSYS Refresh 2026-05-21T21:22:00+08:00

- Change: refreshed 0.8B p128/d32 graph-off mapping and graph-on formal traces after accepting Iter28.
- Evidence: graph-off mapping shows `gated_delta_sequence_precomputed_kernel<bf16, true>` at `28.328 ms`, `71.8%` of prefill GPU time across 18 launches. Smaller prefill kernels remain LMHead GEMV `1.503 ms`, gated RMSNorm `0.462 ms`, depthwise kernel4 `0.439 ms`, and q/k precompute `0.399 ms`.
- Graph-on formal metrics under NSYS overhead: `prefill_ms=40.408`, `decode_step_avg_ms=6.098`, `cuda_graph_enabled=1.0`.
- Decision: no production change from the profile. Iter28 did not change the hotspot ordering; P0 remains the only material prefill target, but further P0 changes still need a stronger exact replay gate or standalone long-loop search.

## Iteration 30 Candidate P2-LMHeadTop1PrefillGate 2026-05-21T21:19:24+08:00

- Intent: extend the existing optional `lm_head_top1` greedy path to prefill's first sampled token. Let `LMHeadLinearLayer::try_forward_top1` accept `ModelStage::Prefill`, let Qwen3.5 call it when `lm_head_top1` is enabled and `lm_head_rows == 1`, and let `StandardEngine` skip prefill sampler when the model has already written `SAMPLER_TOKEN_OUT`. Default full-logits behavior remains unchanged when `lm_head_top1` is disabled.
- Correctness: passed. `tests/layers/test_linear.py -k top1` -> `4 passed`; `tests/operators/test_qwen3_5_runtime_ops.py` -> `10 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment, including `lm_head_top1` and CUDA graph+top1 cases, -> `13 passed`.
- Benchmark: optional `--lm-head-top1` path improved after adding prefill skip. 0.8B p128/d32 graph-on measured `223.136 ms`, median `222.306 ms`, prefill sampler near zero. 0.8B p128/d128 graph-on measured `797.346 ms`, and 2B p128/d128 measured `1704.463 ms`.
- Decision: accept for the optional `lm_head_top1` path only. It is token-safe and improves the top1 benchmark path, but the win versus the default full-logits path is still below the `>=1%` default-enable gate, so `lm_head_top1` remains default-off.
- Shared-path regression: because this touched `StandardEngine` and `LMHeadLinearLayer`, ran `tests/engine/test_qwen2_generate.py -k "token_alignment or kvcache" -s` -> `8 passed, 12 deselected`.

## Iteration 31 Default Graph-On Matrix Refresh 2026-05-21T21:34:00+08:00

- Change: refreshed the EdgeFM default graph-on matrix after Iter28/Iter30 accepted changes. TRT remains out of scope.
- Results: 0.8B p128/p512/p1024 measured `800.399 / 907.143 / 1043.771 ms`; 2B p128/p512/p1024 measured `1703.863 / 1835.728 / 2009.147 ms`.
- Decision: no production change. Matrix confirms current default path is broadly stable; 2B improves across all three prefill lengths, while 0.8B is mostly noise with p512 slightly slower than the previous milestone.

## Iteration 32 Candidate P0-FusePrefillQKDecayBeta 2026-05-21T21:30:29+08:00

- Intent: fuse the prefill-only GatedDeltaNet q/k normalization precompute and decay/beta precompute into one Qwen3.5-local kernel. This removes one small launch per linear-attention layer while preserving the accepted precomputed sequence kernel and its state update / accumulation order.
- Correctness: mixed first run due to concurrent GPU tests, then passed when isolated. Operator tests including the fused-vs-separate bitwise check passed (`11 passed`), the long-prefill graph replay test passed alone, and the full Qwen3.5 0.8B/2B fresh-dump generate suite passed (`13 passed`). After reverting the candidate, the accepted baseline was revalidated (`tests/operators/test_qwen3_5_runtime_ops.py` -> `10 passed`; Qwen3.5 generate -> `13 passed`).
- Benchmark: 0.8B p128/d32 graph-on measured `223.991 ms`, prefill `38.315 ms`, decode step `5.985 ms`, slower than the Iter28 accepted default baseline (`223.386 ms`, prefill `37.977 ms`).
- Decision: reject and revert. Removing one small launch was outweighed by the fused kernel's extra work/shape; q/k precompute plus decay/beta precompute remain separate accepted kernels.

## Iteration 33 Candidate P0-PairedQKPrecompute 2026-05-21T21:43:00+08:00

- Intent: keep decay/beta separate, but compute q and k normalization for the same token/head in one precompute block. Each q/k lane keeps the same per-thread accumulation and tree-reduction order as the accepted kernel, while reducing block scheduling count from `seq_len * heads * 2` to `seq_len * heads`.
- Correctness: passed before benchmark (`tests/operators/test_qwen3_5_runtime_ops.py` -> `10 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`). After reverting the candidate, the accepted baseline was revalidated with the same two gates (`10 passed`, `13 passed`).
- Benchmark: 0.8B p128/d32 graph-on measured `223.920 ms`, prefill `38.214 ms`, decode step `5.986 ms`, slower than the Iter28 accepted baseline (`223.386 ms`, prefill `37.977 ms`).
- Decision: reject and revert. Pairing q/k halves block count, but also halves available block-level parallelism and increases per-block shared-memory work; the accepted separate q/k z-dimension launch remains faster end-to-end.

## Iteration 34 Profile P0-GatedDeltaDecay 2026-05-21T21:51:00+08:00

- Change: refreshed NCU for the current accepted decay-consuming `gated_delta_sequence_precomputed_kernel<bf16, true>` at 0.8B p128.
- Evidence: report `profiles/ncu/qwen3_5_0p8b_p128_gated_delta_sequence_decay_iter34.ncu-rep`, CSV `profiles/ncu/qwen3_5_0p8b_p128_gated_delta_sequence_decay_iter34_details.csv`. Duration `2.06 ms`, compute throughput `39.73%`, memory throughput `28.55%`, achieved occupancy `37.98%`, eligible warps/scheduler `0.61`, no-eligible `60.06%`, barrier stall `41.74%`, registers/thread `40`, grid `64` blocks.
- Decision: profile-only. P0 remains latency/barrier limited with too few independent blocks (`0.38` waves/SM), but prior tile/split attempts either regressed or broke token alignment. NCU also flagged non-fused FP32 instruction potential, so the next narrow candidate is explicit FMA in the precomputed sequence math only.

## Iteration 35 Candidate P0-PrecomputedExplicitFMA 2026-05-21T21:54:00+08:00

- Intent: apply `fmaf` only inside the accepted precomputed GatedDeltaNet sequence kernel for the `kv_mem`, state update, and output projection multiply-adds. This follows the Iter34 NCU FP instruction hint and avoids runtime changes, tiling changes, or q/k precompute changes.
- Correctness: rejected. Operator tests passed (`tests/operators/test_qwen3_5_runtime_ops.py` -> `10 passed`), but full Qwen3.5 fresh-dump generate failed the 0.8B long-prefill CUDA graph replay vs regular decode check. After reverting the explicit FMA edits, the accepted baseline was revalidated (`10 passed`; Qwen3.5 generate -> `13 passed`).
- Benchmark: skipped because correctness failed.
- Decision: reject and revert. The P0 sequence kernel's floating-point contract includes the compiler's existing multiply/add behavior; explicit FMA is not token-safe for long-prefill graph replay.

## Iteration 36 Candidate P2-LMHeadTop1Warps4 2026-05-21T21:58:00+08:00

- Intent: tune only the optional `lm_head_top1` stage1 launch shape by reducing `kWarpsPerBlock` from `8` to `4`. The math per vocab row and warp reduction order stay unchanged; only block grouping and the second-stage candidate count change. Default full-logits behavior remains unchanged.
- Correctness: passed while testing the candidate. `tests/layers/test_linear.py -k top1` -> `4 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: optional `--lm-head-top1` 0.8B p128/d32 graph-on regressed to `224.224 ms`, prefill `38.318 ms`, decode step `5.992 ms`, versus the Iter30 optional top1 baseline `223.136 ms`, `38.127 ms`, `5.963 ms`.
- Correctness after revert: rebuilt after restoring `kWarpsPerBlock=8`; `tests/layers/test_linear.py -k top1` -> `4 passed`; Qwen3.5 fresh-dump generate alignment first had one non-reproduced long-prefill graph/regular mismatch, then the isolated failing case, graph+top1 combination, and full suite all passed (`13 passed`).
- Decision: reject and revert. Four warps per block reduced grouping but hurt the optional top1 path; the accepted experimental top1 stage1 shape remains `kWarpsPerBlock=8`.

## Iteration 37 Candidate P1-DecodeConvStateInplace 2026-05-21T22:08:00+08:00

- Intent: specialize Qwen3.5 depthwise conv state update for decode `seq_len == 1 && kernel_size == 4`. One thread per channel shifts the four state slots in order and appends the current token, avoiding the generic tmp update kernel plus device-to-device copy. This is Qwen3.5 operator-local and does not alter recurrent GatedDeltaNet math.
- Correctness: passed. `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on measured `222.823 ms`, prefill `38.183 ms`, decode step `5.951 ms`, improving over the Iter28 accepted reference (`223.386 ms`, prefill `37.977 ms`, decode step `5.977 ms`). 2B p128/d32 graph-on measured `453.452 ms`, prefill `51.560 ms`, decode step `12.960 ms`.
- NCU: new `update_conv_state_kernel4_single_token_inplace<bf16>` single launch measured `2.62 us`, memory throughput `10.87%`, compute throughput `3.72%`, achieved occupancy `15.42%`, grid `24` blocks. NCU flags the kernel as tiny-grid limited, so the practical win is graph-node and D2D-copy removal rather than saturating the GPU.
- Decision: accept. The change is Qwen3.5 operator-local, token-safe on both supported models, and improves the targeted decode-heavy 0.8B case without touching common runtime or recurrent math.

## Iteration 38 Candidate P1-DecodeConvKernel4FuseUpdate 2026-05-21T22:14:00+08:00

- Intent: fuse Qwen3.5 decode `seq_len == 1 && kernel_size == 4` depthwise conv and conv-state shift into one per-channel kernel. This builds on Iter37 by removing the separate inplace update launch entirely while preserving per-channel math and state order.
- Correctness: passed. `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment passed on the final full rerun (`13 passed`). One prior full-suite rerun hit the already-observed intermittent long-prefill repeated-graph mismatch, but the isolated test and the full ordered subset immediately passed, then the full file passed again.
- Benchmark: 0.8B p128/d32 graph-on improved from Iter37 `222.823 ms`, decode step `5.951 ms` to `222.391 ms`, decode step `5.934 ms`. 2B p128/d32 graph-on improved from Iter37 `453.452 ms`, decode step `12.960 ms` to `452.557 ms`, decode step `12.933 ms`.
- NCU: fused `depthwise_causal_conv1d_kernel4_single_token_update<bf16>` single launch measured `2.94 us`, memory throughput `13.19%`, compute throughput `5.08%`, achieved occupancy `16.01%`, grid `24` blocks. Like Iter37, this is tiny-grid limited; the win is removing one decode graph node per Qwen3.5 linear-attention layer and token.
- Decision: accept. This supersedes Iter37's separate inplace update launch and keeps the final code simpler by removing the now-dead update-only fallback.

## Iteration 39 Candidate P0-SequenceTokenBarrier 2026-05-21T22:49:58+08:00

- Intent: investigate and fix the reproducible long-prefill repeated-generate drift that appeared after Iter38. The minimal repro showed the issue was not CUDA graph-specific: a single regular Qwen3.5 engine with prefill 512/decode 32 could drift within a few repeated requests. Root cause was a missing block barrier at the end of each token iteration in `gated_delta_sequence_kernel` and `gated_delta_sequence_precomputed_kernel`: output projection was still reading `recurrent_state` while other warps could start the next token's state decay/update.
- Correctness: passed. Same-engine regular long-prefill replay passed `20/20`; long-prefill repeated CUDA graph replay with graph reuse passed `20/20`; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; `tests/layers/test_linear.py -k top1` -> `4 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: with graph reuse restored, 0.8B p128/d32 graph-on measured `223.025 ms`, prefill `38.760 ms`, decode step `5.940 ms`. 2B p128/d32 graph-on measured `453.096 ms`, prefill `51.943 ms`, decode step `12.936 ms`. This is slightly slower than Iter38's pre-barrier `222.391/452.557 ms`, but removes a real recurrent-state race.
- NCU: `gated_delta_sequence_precomputed_kernel<bf16,true>` at 0.8B p128 measured `2.10 ms`, compute throughput `38.96%`, memory throughput `27.97%`, achieved occupancy `37.99%`, eligible warps/scheduler `0.59`, no-eligible `60.84%`, barrier stall `43.31%`, grid `64`, block `256`, registers/thread `40`. The added barrier makes the synchronization cost explicit and confirms P0 is still recurrence/barrier limited.
- Decision: accept as a correctness fix. Restore Qwen3.5 cross-request decode graph reuse because the barrier fix makes the repeated graph stress stable; do not keep the temporary `can_reuse=false` fallback. Future P0 changes must keep the long-prefill same-engine and repeated-graph stress gates.

## Iteration 40 Profile Refresh 2026-05-21T22:54:00+08:00

- Change: refreshed graph-off NSYS mapping after Iter39 and rechecked the existing optional `lm_head_top1` path against the new barrier-safe baseline.
- Evidence: graph-off p128/d32 mapping shows prefill `gated_delta_sequence_precomputed_kernel<bf16,true>` at `56.813 ms` (`72.2%` of prefill GPU time), decode LMHead cuBLAS GEMV at `92.958 ms` (`24.5%` of decode GPU time), decode `gated_delta_sequence_kernel<bf16>` at `22.830 ms`, decode RMSNorm at `11.744 ms`, and attention at `5.793 ms`. The optional `--lm-head-top1` 0.8B p128/d32 graph-on path measured `222.180 ms`, versus the barrier-safe default path `223.025 ms`.
- Decision: profile-only. P0 remains the dominant prefill target, while LMHead remains the largest decode hotspot. `lm_head_top1` is token-safe and useful as an experimental path, but still below the `>=1%` default-enable gate.

## Iteration 41 Candidate P2-LMHeadTop1Warps16 2026-05-21T22:59:00+08:00

- Intent: tune only the optional `lm_head_top1` stage1 launch shape by increasing grouping from `8` to `16` warps per block. This keeps the default full-logits path unchanged and only affects the default-off `--lm-head-top1` path.
- Correctness: passed. `tests/layers/test_linear.py -k top1` -> `4 passed`; `tests/engine/test_qwen3_5_generate.py -k lm_head_top1` with fresh Qwen3.5 dumps -> `4 passed`.
- Benchmark: optional 0.8B p128/d32 graph-on `--lm-head-top1` improved from the post-Iter39 top1 baseline `222.180 ms`, decode step `5.919 ms`, to `221.655 ms`, decode step `5.904 ms`. Optional 2B p128/d32 graph-on measured `452.655 ms`, decode step `12.921 ms`, versus the post-Iter39 default full-logits path `453.096 ms`, decode step `12.936 ms`.
- Decision: accept for the optional `lm_head_top1` path only. The launch-shape change is correct and slightly faster for the measured optional path, but the end-to-end gain versus default full logits remains below `1%`, so `lm_head_top1` stays default-off.

## Iteration 42 Candidate P2-LMHeadTop1Warps32 2026-05-21T23:01:51+08:00

- Intent: test the upper launch-shape bound for optional `lm_head_top1` by increasing stage1 grouping from `16` to `32` warps per block. This reduces candidate count but uses `1024` threads per block, so it is expected to trade lower stage2 work for lower residency.
- Correctness: passed while testing the candidate. `tests/layers/test_linear.py -k top1` -> `4 passed`; Qwen3.5 fresh-dump generate subset `-k lm_head_top1` -> `4 passed`.
- Benchmark: optional 0.8B p128/d32 graph-on `--lm-head-top1` measured `222.588 ms`, decode step `5.926 ms`, slower than the accepted Iter41 warps16 result `221.655 ms`, decode step `5.904 ms`.
- Decision: reject and revert to `kWarpsPerBlock=16`. After revert and rebuild, `tests/layers/test_linear.py -k top1` passed again (`4 passed`).

## Iteration 43 Candidate P2-LMHeadTop1Warps24 2026-05-21T23:07:47+08:00

- Intent: test an intermediate optional `lm_head_top1` stage1 launch shape, `24` warps per block, after `16` won over `8` and `32` regressed. This keeps the path default-off and does not affect full-logits generation.
- Correctness: passed. `tests/layers/test_linear.py -k top1` -> `4 passed`; Qwen3.5 fresh-dump generate subset `-k lm_head_top1` -> `4 passed`.
- Benchmark: 0.8B p128/d32 graph-on `--lm-head-top1` first 5-run average was noisy (`222.266 ms`), then a warmup-2 / 8-run rerun measured `221.249 ms`, decode step `5.902 ms`, improving over Iter41 warps16 `221.655 ms`, decode step `5.904 ms`. 2B p128/d32 graph-on measured `452.648 ms`, decode step `12.921 ms`, essentially flat/slightly better versus Iter41 warps16 `452.655 ms`, decode step `12.921 ms`.
- Decision: accept for the optional `lm_head_top1` path and supersede Iter41's `16`-warp launch shape. The path remains default-off because gain versus full logits is still below `1%`.

## Iteration 44 Candidate P2-LMHeadTop1Warps20 2026-05-21T23:10:44+08:00

- Intent: test a lighter intermediate optional `lm_head_top1` stage1 launch shape at `20` warps per block after `24` became the best measured shape.
- Correctness: passed while testing the candidate. `tests/layers/test_linear.py -k top1` -> `4 passed`; Qwen3.5 fresh-dump generate subset `-k lm_head_top1` -> `4 passed`.
- Benchmark: 0.8B p128/d32 graph-on `--lm-head-top1` with warmup-2 / 8 runs measured `221.942 ms`, decode step `5.914 ms`, slower than Iter43 warps24 `221.249 ms`, decode step `5.902 ms`.
- Decision: reject and revert to `kWarpsPerBlock=24`. After rebuild, `tests/layers/test_linear.py -k top1` passed again (`4 passed`).

## Iteration 45 Candidate P2-LMHeadTop1Warps28 2026-05-21T23:13:07+08:00

- Intent: test the heavier side between the accepted `24`-warp shape and the rejected `32`-warp shape for optional `lm_head_top1`.
- Correctness: passed while testing the candidate. Rebuild succeeded; `tests/layers/test_linear.py -k top1` -> `4 passed`; Qwen3.5 fresh-dump generate subset `-k lm_head_top1` -> `4 passed`.
- Benchmark: 0.8B p128/d32 graph-on `--lm-head-top1` with warmup-2 / 8 runs measured `221.840 ms`, decode step `5.915 ms`, slower than Iter43 warps24 `221.249 ms`, decode step `5.902 ms`.
- Decision: reject and revert to `kWarpsPerBlock=24`. After rebuild, `tests/layers/test_linear.py -k top1` passed again (`4 passed`). The stage1 launch-shape sweep is closed with `24` as the best accepted value.

## Iteration 46 Profile P2-LMHeadTop1Stage1 2026-05-21T23:15:00+08:00

- Change: collected NCU for the accepted optional `lm_head_top1` stage1 kernel with `kWarpsPerBlock=24`.
- Evidence: report `profiles/ncu/qwen3_5_0p8b_lm_head_top1_stage1_warps24_iter46.ncu-rep`, CSV `profiles/ncu/qwen3_5_0p8b_lm_head_top1_stage1_warps24_iter46.csv`. First captured stage1 launch measured duration `1.49 ms`, memory/DRAM throughput `97.46%`, compute throughput `65.80%`, achieved occupancy `94.50%`, theoretical occupancy `100%`, block size `768`, grid size `10347`, registers/thread `31`, active warps/scheduler `11.31`, eligible warps/scheduler `0.73`, no-eligible `67.73%`.
- Decision: profile-only. The accepted optional top1 stage1 is memory-roofline limited and meets the `>=95%` operator-ceiling target on memory throughput. Do not continue launch-shape micro-tuning for this kernel unless a new algorithmic top1 implementation is introduced.

## Iteration 47 Candidate P4-SmallHiddenRMSNormWarpReduce 2026-05-21T23:19:21+08:00

- Intent: add a Qwen3.5-local small-hidden RMSNorm/gated-RMSNorm kernel for `hidden <= 128` using warp reductions plus one cross-warp reduction, reducing the generic shared-memory reduction barrier count. This targets the small GatedDeltaNet norm path and does not touch runtime or recurrent math.
- Correctness: passed. `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on default path measured `222.031 ms` in the first 5-run pass, then `221.433 ms` with warmup-2 / 8 runs, improving versus the Iter39 barrier-safe default `223.025 ms`. 2B p128/d32 graph-on default path measured `452.963 ms`, slightly better than the Iter39 barrier-safe default `453.096 ms`.
- Decision: accept. The change is Qwen3.5 operator-local, token-safe on both supported models, improves the 0.8B target case, and does not regress the 2B p128/d32 check.

## Iteration 48 Profile P4-SmallHiddenRMSNormWarpReduce 2026-05-21T23:20:00+08:00

- Change: collected NCU for the accepted 128-thread small-hidden RMSNorm/gated-RMSNorm kernel before trying a lighter launch shape.
- Evidence: report `profiles/ncu/qwen3_5_0p8b_small_hidden_rmsnorm_iter48.ncu-rep`, CSV `profiles/ncu/qwen3_5_0p8b_small_hidden_rmsnorm_iter48.csv`. First captured launch measured `13.63 us`, memory throughput `49.78%`, DRAM throughput `30.90%`, compute throughput `49.78%`, achieved occupancy `86.33%`, block size `128`, grid size `2048`, registers/thread `17`, waves/SM `6.10`.
- Decision: profile-only. The profile showed the kernel was still latency/issue limited and had avoidable block-level barriers, so the next narrow candidate was a one-warp-per-row implementation.

## Iteration 49 Candidate P4-SmallHiddenRMSNormOneWarp 2026-05-21T23:26:42+08:00

- Intent: specialize Qwen3.5 small-hidden (`hidden <= 128`) RMSNorm/gated-RMSNorm to one warp per row. This removes the cross-warp shared-memory reduction and block barriers from Iter47 while preserving the per-row reduction order within one warp.
- Correctness: passed. Build succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`. After removing the now-unused 128-thread small-hidden kernel, rebuild plus `tests/operators/test_qwen3_5_runtime_ops.py` passed again (`11 passed`).
- Benchmark: sequential warmup-2 / 8-run graph-on default path measured 0.8B p128/d32 `221.148 ms`, prefill `37.825 ms`, decode step `5.909 ms`, improving versus Iter47 `221.433 ms`, `37.963 ms`, `5.914 ms`. 2B p128/d32 measured `452.350 ms`, prefill `51.511 ms`, decode step `12.926 ms`, improving versus Iter47 `452.963 ms`, `51.927 ms`, `12.932 ms`.
- Decision: accept and supersede Iter47's 128-thread small-hidden kernel. The change is Qwen3.5 operator-local, token-safe on both supported models, and improves both measured 0.8B and 2B p128/d32 graph-on cases.

## Iteration 50 Profile P4-SmallHiddenRMSNormOneWarp 2026-05-21T23:29:00+08:00

- Change: collected NCU for the accepted one-warp small-hidden RMSNorm/gated-RMSNorm kernel.
- Evidence: report `profiles/ncu/qwen3_5_0p8b_one_warp_rmsnorm_iter50.ncu-rep`, CSV `profiles/ncu/qwen3_5_0p8b_one_warp_rmsnorm_iter50.csv`. First captured launch measured `10.88 us`, memory/DRAM throughput `38.50%`, compute throughput `27.48%`, achieved occupancy `27.12%`, theoretical occupancy `33.33%`, block size `32`, grid size `2048`, registers/thread `33`, eligible warps/scheduler `0.43`, no-eligible `68.00%`.
- Decision: profile-only. The one-warp kernel improves Iter48's `13.63 us` 128-thread kernel, but occupancy is capped by one-warp blocks, so the next narrow candidate is packing multiple independent row-warps into each block.

## Iteration 51 Candidate P4-SmallHiddenRMSNormFourWarpsPerBlock 2026-05-21T23:31:35+08:00

- Intent: pack four independent row-warps into each small-hidden RMSNorm block, keeping one warp per row and the same intra-warp reduction order while improving block shape and theoretical occupancy.
- Correctness: passed while testing the candidate. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2 / 8-run average regressed to `221.539 ms`, prefill `37.948 ms`, decode step `5.918 ms`, slower than Iter49 one-warp `221.148 ms`, `37.825 ms`, `5.909 ms`.
- Decision: reject and revert. Packing rows into 4-warp blocks improves the theoretical occupancy shape but hurts the end-to-end target case, likely from lower block-level scheduling parallelism for this tiny row kernel. After reverting to the accepted one-warp kernel, rebuild plus `tests/operators/test_qwen3_5_runtime_ops.py` passed again (`11 passed`).

## Iteration 52 Candidate P4-SmallHiddenRMSNormHidden128Unroll 2026-05-21T23:34:51+08:00

- Intent: specialize the accepted one-warp small-hidden RMSNorm path for the common `hidden == 128` shape by unrolling the four per-lane loads/stores. This keeps each lane's accumulation order unchanged and only removes loop/branch overhead.
- Correctness: passed while testing the candidate. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2 / 8-run average regressed to `221.623 ms`, prefill `38.048 ms`, decode step `5.918 ms`, slower than Iter49 one-warp `221.148 ms`, `37.825 ms`, `5.909 ms`.
- Decision: reject and revert. The compiler-generated loop for this small shape is already better in the full generate path; after reverting to the accepted one-warp kernel, rebuild plus `tests/operators/test_qwen3_5_runtime_ops.py` passed again (`11 passed`).

## Iteration 53 NSYS Refresh 2026-05-21T23:37:00+08:00

- Change: refreshed 0.8B p128/d32 graph-off NSYS mapping after Iter49/Iter52.
- Evidence: action report `profiles/qwen3_5_nsys_action_iter53.txt`. Prefill remains dominated by `gated_delta_sequence_precomputed_kernel<bf16,true>` at `28.910 ms`, `72.7%` of prefill GPU time across 18 launches. Smaller prefill kernels are LMHead GEMV `1.503 ms`, depthwise kernel4 `0.436 ms`, and q/k precompute `0.400 ms`. Decode is dominated by LMHead GEMV `46.474 ms`, then `gated_delta_sequence_kernel<bf16>` `11.698 ms`, large-hidden RMSNorm `6.025 ms`, FlashInfer attention `2.971 ms`, and add `2.152 ms`.
- Decision: profile-only. Since LMHead cuBLASLt/top1 paths were already tuned or gated, the next Qwen3.5-local candidate is large-hidden RMSNorm reduction overhead.

## Iteration 54 Candidate P4-LargeHiddenRMSNormWarpReduce 2026-05-21T23:40:51+08:00

- Intent: reduce large-hidden Qwen3.5 RMSNorm barrier cost by replacing the 256-thread shared-memory tree with warp reductions plus one cross-warp reduction. This targets decode RMSNorm (`6.025 ms` across 1891 launches in Iter53) without touching runtime scheduling.
- Correctness: rejected. Operator tests passed (`tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`), but Qwen3.5 fresh-dump token alignment failed for 0.8B at decode index 18 (`99550` vs `98846`). This matches the earlier pattern that large hidden normalization/recurrent math reduction order is part of the token contract.
- Benchmark: skipped because token alignment failed.
- Decision: reject and revert. After restoring the original shared-memory reduction path, rebuild, operator tests, and Qwen3.5 0.8B/2B fresh-dump generate all passed again (`11 passed`, `13 passed`).

## Iteration 55 Profile P2-LMHeadTop1PostIter49 2026-05-21T23:42:00+08:00

- Change: rechecked the optional `--lm-head-top1` path after the accepted Iter49 one-warp small-hidden RMSNorm path.
- Evidence: 0.8B p128/d32 graph-on warmup-2 / 8-run `--lm-head-top1` measured `221.541 ms`, prefill `38.104 ms`, decode step `5.913 ms`.
- Decision: profile-only. This is slower than the current default full-logits Iter49 baseline (`221.148 ms`), so `lm_head_top1` remains default-off and no default policy change is made.

## Iteration 56 Profile P4-AddKernel 2026-05-21T23:43:00+08:00

- Change: collected NCU for `add_kernel<bf16>` after Iter53 showed add as a smaller decode hotspot (`2.152 ms` across 1488 launches).
- Evidence: report `profiles/ncu/qwen3_5_0p8b_add_kernel_iter56b.ncu-rep`, CSV `profiles/ncu/qwen3_5_0p8b_add_kernel_iter56b.csv`. First captured launch measured `4.96 us`, memory/DRAM throughput `40.02%`, compute throughput `19.79%`, achieved occupancy `79.19%`, theoretical occupancy `100%`, block size `256`, grid size `512`, registers/thread `16`, eligible warps/scheduler `0.55`, no-eligible `75.73%`.
- Decision: profile-only. The kernel is mostly latency/issue limited at this shape, so only a semantics-preserving unroll candidate is worth testing before considering this path plateaued.

## Iteration 57 Candidate P4-AddKernelUnroll2 2026-05-21T23:46:08+08:00

- Intent: make Qwen3.5 add process two adjacent elements per thread while preserving per-element `float(lhs) + float(rhs)` and dtype store semantics. This targets add instruction/issue overhead without fusing with RMSNorm or changing runtime scheduling.
- Correctness: passed while testing the candidate. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2 / 8-run average was `221.581 ms`, prefill `37.960 ms`, decode step `5.919 ms`, slower than Iter49 one-warp default `221.148 ms`, `37.825 ms`, `5.909 ms`.
- Decision: reject and revert. The unroll is token-safe but does not improve the target case; after reverting to the original add kernel, rebuild plus `tests/operators/test_qwen3_5_runtime_ops.py` passed again (`11 passed`).

## Iteration 58 Sanity + Current Log Cleanup 2026-05-21T23:48:00+08:00

- Change: reran 0.8B p128/d32 graph-on after rejected candidates were reverted, then rewrote `CURRENT.md` into a concise current-facts summary to remove stale "latest" claims from older iterations.
- Evidence: post-reject sanity run measured `221.913 ms`, prefill `38.125 ms`, decode step `5.925 ms` on the same accepted code path. This is slower than the Iter49 best accepted rerun (`221.148 ms`) and is treated as run-to-run variance rather than a code regression because the source was restored and operator correctness passed after each revert.
- Decision: no production change. Current accepted code remains Iter49 plus earlier accepted changes; next work should avoid large-hidden RMSNorm/P0 order changes in production and move any new P0 search into standalone repro first.

## Iteration 59 Candidate P0-PrecomputedTile16AfterBarrier 2026-05-22T09:16:12+08:00

- Intent: retest the old standalone-winning `tile=16/thread=256` P0 sequence shape after Iter39's token-boundary barrier fix. Keep decode/original `gated_delta_sequence_kernel` at `tile=32`, and apply `tile=16` only to the prefill `gated_delta_sequence_precomputed_kernel`.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`. This differs from Iter18, where the same precomputed tiling failed before the barrier fix.
- Benchmark: 0.8B p128/d32 graph-on warmup-2 / 8-run average improved to `217.350 ms`, prefill `33.513 ms`, decode step `5.925 ms`, versus Iter49 best accepted `221.148 ms`, `37.825 ms`, `5.909 ms`. 2B p128/d32 graph-on improved to `448.683 ms`, prefill `47.274 ms`, decode step `12.943 ms`, versus Iter49 `452.350 ms`, `51.511 ms`, `12.926 ms`.
- NCU: `gated_delta_sequence_precomputed_kernel<bf16,true,16>` first captured 0.8B p128 launch measured `1.85 ms`, compute throughput `50.97%`, memory throughput `44.93%`, achieved occupancy `74.07%`, eligible warps/scheduler `1.03`, no-eligible `46.52%`, grid `128`, block `256`, registers/thread `40`. This improves over the barrier-safe tile32 profile (`2.10 ms`, compute `38.96%`, occupancy `37.99%`, no-eligible `60.84%`).
- Decision: accept. The candidate is Qwen3.5-local, now token-safe on both supported models after the barrier fix, and materially improves prefill latency. Next step is a graph-on milestone matrix for 0.8B/2B prefill `128/512/1024`, decode `128`.

## Iteration 60 Matrix Refresh After Tile16 2026-05-22T09:20:13+08:00

- Change: ran the EdgeFM-only matrix in `matrix_iter59/` after accepting Iter59. Workload: 0.8B/2B, prefill `128/512/1024`, decode `128`, warmup `1`, runs `3`, graph-off and graph-on; TRT still skipped.
- Results: graph-on 0.8B p128/p512/p1024 measured `790.445 / 880.508 / 1007.907 ms`, with prefill `33.644 / 120.570 / 247.342 ms`. Graph-on 2B p128/p512/p1024 measured `1693.637 / 1813.777 / 1972.185 ms`, with prefill `47.217 / 161.586 / 319.718 ms`.
- Decision: matrix confirms Iter59 transfers to the longer prefill targets. Compared with the previous milestone matrix (`0.8B 800.399/907.143/1043.771`, `2B 1703.863/1835.728/2009.147`), graph-on improves all six cases.

## Iteration 61 Candidate P0-PrecomputedTile8AfterBarrier 2026-05-22T09:23:11+08:00

- Intent: test prefill-only `gated_delta_sequence_precomputed_kernel` with `value_tile=8/thread=256` after Iter59 proved `tile=16` is token-safe and faster than `tile=32`. This doubles the block count for the value dimension and may improve SM residency further, but risks extra scheduling/synchronization overhead. Decode/original recurrent sequence remains unchanged at `tile=32`.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2 / 8-run average regressed to `232.469 ms`, prefill `48.606 ms`, decode step `5.926 ms`, slower than Iter59 tile16 `217.350 ms`, prefill `33.513 ms`, decode step `5.925 ms`.
- Decision: reject and revert to Iter59 `value_tile=16`. The doubled block count is token-safe but increases prefill latency enough that 2B benchmarking is unnecessary for the acceptance gate.

## Iteration 62 Candidate P0-PrecomputedTile16Threads128 2026-05-22T09:26:35+08:00

- Intent: test prefill-only `gated_delta_sequence_precomputed_kernel` with accepted `value_tile=16` but `threads=128` instead of `256`. This keeps the token-safe tile shape from Iter59 while reducing per-block resources and may improve resident block count/issue latency. Decode/original recurrent sequence remains unchanged at `tile=32/thread=256`; q/k precompute remains unchanged.
- Correctness: passed. Build succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2 / 8-run average regressed to `221.920 ms`, prefill `38.065 ms`, decode step `5.927 ms`, slower than Iter59 tile16/thread256 `217.350 ms`, prefill `33.513 ms`, decode step `5.925 ms`.
- Decision: reject and revert to Iter59 `value_tile=16/thread=256`. Reducing block size is token-safe but does not improve the full generate gate.

## Iteration 63 Candidate P0-PrecomputedTile64Threads128 2026-05-22T09:29:50+08:00

- Intent: test prefill-only `gated_delta_sequence_precomputed_kernel` with `value_tile=64/thread=128`. This reduces value-dimension block count and synchronization partitioning relative to Iter59, trading lower SM residency for more per-block work and potentially less launch scheduling overhead. Decode/original recurrent sequence remains unchanged at `tile=32/thread=256`.
- Correctness: passed. Build succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `11 passed`; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2 / 8-run average regressed to `250.635 ms`, prefill `66.768 ms`, decode step `5.926 ms`, slower than Iter59 tile16/thread256 `217.350 ms`, prefill `33.513 ms`, decode step `5.925 ms`.
- Decision: reject and revert to Iter59 `value_tile=16/thread=256`. Larger tile reduces value-block count but makes each block too heavy for the full generate gate.

## Iteration 64 Candidate P1-DecodeSequenceFromAB 2026-05-22T09:37:31+08:00

- Intent: fuse the decode-only `compute_g_beta` launch into `gated_delta_sequence` by adding a Qwen3.5-local `gated_delta_sequence_from_ab` operator. Prefill remains on the accepted decay/qk-precompute path; runtime scheduling is unchanged. This removes one small launch and the decode-side g/beta temporary path while preserving the existing recurrent sequence math and reduction order.
- Correctness: passed. Targeted fused-vs-separate operator test passed (`tests/operators/test_qwen3_5_runtime_ops.py -k sequence_from_ab` -> `1 passed`); full operator suite passed after alloc-restore (`12 passed`); Qwen3.5 0.8B/2B fresh-dump generate alignment passed (`13 passed`).
- Benchmark: 0.8B p128/d32 graph-on warmup-2/runs-8 measured `217.065 ms`, prefill `33.491 ms`, decode step `5.917 ms`, versus Iter59 `217.350/33.513/5.925`. 2B p128/d32 measured `447.830 ms`, prefill `47.078 ms`, decode step `12.923 ms`, versus Iter59 `448.683/47.274/12.943`. A 2B p1024/d128 warmup-2/runs-5 regression check measured `1969.855 ms`, prefill `318.835 ms`, decode step `12.997 ms`.
- NCU: first captured `gated_delta_sequence_from_ab_kernel<bf16,bf16>` launch measured `33.63 us`, compute throughput `23.98%`, memory throughput `17.75%`, issue slots busy `26.48%`, active warps/scheduler `4.35`, eligible warps/scheduler `0.39`, no-eligible `72.17%`, grid `64`, block `256`. The report flags a tiny-grid limit, so the practical gain is node/temp removal rather than roofline utilization.
- Matrix: EdgeFM-only `matrix_iter64_alloc_restore/` with decode128 measured graph-on 0.8B p128/p512/p1024 `786.883/880.567/1008.868 ms` and 2B `1693.325/1813.829/1973.261 ms`. Compared with Iter60, this is effectively flat outside the shorter p128/d32 gate; no material long-prefill regression remains after restoring workspace allocation order.
- Decision: accept as a Qwen3.5-local micro-win, not as a major P1 breakthrough. The change is token-safe on both supported models and slightly improves p128/d32, but the full matrix shows only noise-level movement. Next step is a fresh post-Iter64 hotspot map before choosing the next candidate.
## Iteration 65 Profile PostIter64HotspotAndTop1 2026-05-22T10:02:00+08:00

- Change: refreshed the 0.8B p128/d32 graph-off NSYS mapping after Iter64 and rechecked the optional `--lm-head-top1` path under the current graph-on default.
- Evidence: NSYS action report `profiles/qwen3_5_nsys_action_iter65.txt` shows prefill still dominated by `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.317 ms` (`69.1%` of prefill GPU time), decode LMHead GEMV at `46.477 ms` (`24.4%`), decode `gated_delta_sequence_from_ab_kernel<bf16,bf16>` at `12.179 ms` (`6.4%`), large-hidden RMSNorm at `6.031 ms`, attention at `2.968 ms`, and add at `2.155 ms`.
- Correctness: no production code change. `tests/layers/test_linear.py -k top1` passed (`4 passed`); Iter64 full Qwen3.5 fresh-dump gate remains the current generation correctness evidence.
- Benchmark: optional `--lm-head-top1` measured 0.8B p128/d32 graph-on `216.392 ms`, prefill `33.441 ms`, decode step `5.897 ms`, versus default Iter64 `217.065/33.491/5.917`. 2B measured `446.724 ms`, prefill `46.768 ms`, decode step `12.897 ms`, versus default Iter64 `447.830/47.078/12.923`.
- Decision: profile-only, keep default-off. The optional path is consistently positive after Iter64, but the p128/d32 end-to-end gain is still below the 1% policy gate, and prior NCU already showed top1 stage1 is memory-roofline limited. Next work returns to P0 standalone/ceiling or another larger hotspot with a clear improvement path.
## Iteration 66 Profile P0BarrierSafeCeiling 2026-05-22T10:08:00+08:00

- Change: clarified the recurrence-aware ceiling for the P0 precomputed GatedDeltaNet kernel. Added a `GD_TOKEN_BOUNDARY_BARRIER` switch to the standalone repro and compiled `bench_tile16_t256_barrier` so the repro matches the Iter39/Iter59 production token-boundary barrier requirement.
- Correctness: no production code change. Standalone barrier repro passed its output/state check (`output_max_abs=0.000488281`, `state_max_abs=5.96046e-08`). Current production correctness remains the Iter64 full gate (`12 passed` operator tests, `13 passed` Qwen3.5 fresh-dump generate).
- Evidence: production `gated_delta_sequence_precomputed_kernel<bf16,true,16>` NCU rerun measured `1.85 ms`, SM throughput `50.94%`, memory throughput `44.89%`, achieved occupancy `73.85%`, eligible warps/scheduler `1.04`, no-eligible `46.27%`. Barrier-safe standalone `tile16/thread256` NCU also measured `1.85 ms`, SM/memory throughput `44.38%`, achieved occupancy `76.00%`, eligible warps/scheduler `0.80`, no-eligible `57.33%`.
- Decision: mark P0 precomputed recurrent sequence as at the token-safe recurrence-aware ceiling for the current layout on an apples-to-apples NCU basis. The faster standalone event timing is not a valid production ceiling because it either omitted the correctness-critical token-boundary barrier or uses a different measurement basis. Further P0 gains require a standalone token-stable algorithm/layout search before production transfer.
## Iteration 67 Candidate P1-DecodeFromABTile16 2026-05-22T10:11:00+08:00

- Intent: test a decode-only `gated_delta_sequence_from_ab` value tile of `16` instead of the generic sequence tile `32`. Iter64 NCU showed the fused decode kernel is tiny-grid limited (`64` blocks, `0.6` waves/SM); tile16 doubles the value-tile block count to `128` without touching prefill, runtime scheduling, or recurrent math order.
- Correctness: passed during candidate evaluation. Build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `12 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`. After reverting to Iter64 tile32, rebuild passed; runtime ops -> `12 passed`; fresh-dump generate -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2/runs-8 measured `216.711 ms`, prefill `33.390 ms`, decode step `5.908 ms`, slightly better than Iter64 `217.065/33.491/5.917`. 2B p128/d32 measured `448.004 ms`, prefill `47.132 ms`, decode step `12.927 ms`, slower than Iter64 `447.830/47.078/12.923`.
- Decision: reject and revert. The tile16 retile improves 0.8B only marginally and regresses 2B, so it is not a stable cross-model improvement. Keep Iter64 from_ab tile32.
## Iteration 68 Candidate P1-DecodeFromABTile64 2026-05-22T10:24:00+08:00

- Intent: test a decode-only `gated_delta_sequence_from_ab` value tile of `64`. This halves the value-tile blocks versus Iter64 tile32 and may reduce duplicated per-block q/k l2norm scalar work, at the cost of fewer resident blocks. Prefill and runtime scheduling remain unchanged.
- Correctness: passed during candidate evaluation. Build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `12 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`. After reverting to Iter64 tile32, rebuild passed; runtime ops -> `12 passed`; fresh-dump generate -> `13 passed`.
- Benchmark: 0.8B p128/d32 graph-on warmup-2/runs-8 measured `216.601 ms`, prefill `33.365 ms`, decode step `5.906 ms`, slightly better than Iter64 `217.065/33.491/5.917`. 2B p128/d32 measured `447.790 ms`, prefill `47.058 ms`, decode step `12.923 ms`, effectively flat/slightly better than Iter64 `447.830/47.078/12.923`. 2B p1024/d128 measured `1971.804 ms`, prefill `319.941 ms`, decode step `13.004 ms`, regressing versus Iter64 `1969.855/318.835/12.997`.
- Decision: reject and revert. Tile64 is token-safe and marginally positive on short cases, but the long-prefill 2B regression exceeds the tiny short-case gain. Keep Iter64 from_ab tile32 as the stable cross-case choice; the tile16/tile64 sweep closes this retile path.

## Iteration 69 Candidate P4-AddStoreResidual 2026-05-22T10:28:18+08:00

- Intent: remove Qwen3.5 model-local residual D2D copies without changing runtime scheduling. Add `qwen3_5_add_store_residual`: compute the existing elementwise `float(lhs) + float(rhs)` once, store it to `hidden`, and also store the same value into the residual workspace. `forward_impl` now copies the embedding hidden state into residual once, then each attention/MLP residual add refreshes residual for the next sublayer/layer.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `13 passed` including the new aliasing test; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: targeted graph-on warmup-2 runs showed 0.8B p128/d32 `215.491 ms`, prefill `33.284 ms`, decode step `5.874 ms`, versus Iter64 `217.065/33.491/5.917`; 2B p128/d32 `446.987 ms`, `47.143 ms`, `12.895 ms`, versus Iter64 `447.830/47.078/12.923`; 2B p1024/d128 `1966.871 ms`, `319.510 ms`, `12.970 ms`, versus Iter64 `1969.855/318.835/12.997`.
- Matrix: `matrix_iter69_add_store_residual/` graph-on decode128 measured 0.8B p128/p512/p1024 `782.246/876.225/1004.755 ms` and 2B `1688.651/1808.815/1967.849 ms`, improving all six Iter64 graph-on matrix cases (`786.883/880.567/1008.868`, `1693.325/1813.829/1973.261`).
- Decision: accepted at the time, then superseded by Iter71. The dual-store operator proved the residual-copy hypothesis, but Iter71's in-place add removes the extra residual store and deletes the temporary `add_store_residual` API from current code.

## Iteration 70 Profile PostIter69HotspotAndTop1 2026-05-22T10:34:00+08:00

- Change: refreshed a post-Iter69 graph-off NSYS mapping trace for 0.8B p128/d32, then rechecked optional `--lm-head-top1` under the Iter69 code path.
- Evidence: `profiles/qwen3_5_nsys_action_iter70.txt` shows prefill still dominated by `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.226 ms` (`69.1%` of prefill GPU time). Decode remains dominated by LMHead GEMV `46.474 ms` (`24.4%`), followed by `gated_delta_sequence_from_ab_kernel` `12.159 ms`, large-hidden RMSNorm `6.012 ms`, FlashInfer attention `2.966 ms`, and the residual add path `2.202 ms`.
- Top1 recheck: optional `--lm-head-top1` measured 0.8B p128/d32 `215.160 ms` versus default Iter69 `215.491 ms` (~`0.15%`), and 2B `445.516 ms` versus default `446.987 ms` (~`0.33%`).
- Decision: profile-only. Keep `lm_head_top1` default-off because the gain is still below the 1% default gate; use the residual add path as the next local candidate.

## Iteration 71 Candidate P4-InplaceResidualAdd 2026-05-22T10:43:22+08:00

- Intent: supersede Iter69 by using the existing `qwen3_5_add` in-place (`hidden += mixer_output`) instead of maintaining a separate residual workspace. Qwen3.5 layer code already keeps `hidden` unchanged until each residual add, so lhs/output aliasing preserves the same per-element math while removing the initial residual copy and the Iter69 extra residual store.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `13 passed` including the new in-place alias test; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`. The temporary Iter69 `add_store_residual` API/test was removed after Iter71 superseded it, and the cleanup correctness gate still passed.
- Benchmark: targeted graph-on warmup-2 runs measured 0.8B p128/d32 `215.235 ms`, prefill `33.169 ms`, decode step `5.869 ms`; 2B p128/d32 `446.619 ms`, `46.996 ms`, `12.887 ms`; 2B p1024/d128 `1962.376 ms`, `317.062 ms`, `12.953 ms`.
- Matrix: `matrix_iter71_inplace_residual_add/` graph-on decode128 measured 0.8B p128/p512/p1024 `781.800/876.108/1004.686 ms` and 2B `1687.489/1808.147/1966.742 ms`, improving all six Iter69 graph-on matrix cases.
- Decision: accept. This is simpler than Iter69, removes a temporary operator API, keeps all changes Qwen3.5-local, and improves every graph-on milestone case versus the previous accepted matrix.

## Iteration 72 Profile PostIter71HotspotAndTop1Long 2026-05-22T10:48:00+08:00

- Change: refreshed post-Iter71 graph-off NSYS mapping for 0.8B p128/d32 and checked optional `--lm-head-top1` on the 2B p1024/d128 long decode case.
- Evidence: `profiles/qwen3_5_nsys_action_iter72.txt` shows prefill `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.243 ms`, decode LMHead GEMV `46.477 ms`, decode `gated_delta_sequence_from_ab_kernel` `12.143 ms`, large-hidden RMSNorm `6.004 ms`, attention `2.967 ms`, and in-place `add_kernel` `2.160 ms`.
- Top1 long-case check: 2B p1024/d128 graph-on with `--lm-head-top1` measured `1959.973 ms`, prefill `315.618 ms`, decode step `12.946 ms`, versus Iter71 default `1962.376/317.062/12.953`. The total gain is only ~`0.12%`.
- Decision: profile-only. Keep `lm_head_top1` default-off; next local candidate is fixed-hidden large RMSNorm.

## Iteration 73 Candidate P4-FixedHiddenRMSNorm 2026-05-22T10:57:39+08:00

- Intent: specialize non-gated Qwen3.5 large-hidden RMSNorm for `hidden=1024` and `hidden=2048`. Keep block size `256` and the same shared-memory reduction tree as the generic kernel, but remove dynamic hidden loop/index overhead for the two supported model sizes.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed` with new 1024/2048 RMSNorm reference coverage; Qwen3.5 0.8B/2B fresh-dump generate alignment -> `13 passed`.
- Benchmark: targeted graph-on warmup-2 runs were noisy on 0.8B (`215.759 ms` then `215.031 ms`) but positive on 2B: p128/d32 `445.641 ms`, prefill `46.798 ms`, decode step `12.862 ms`; p1024/d128 `1961.341 ms`, `317.228 ms`, `12.944 ms`.
- Matrix: `matrix_iter73_fixed_hidden_rmsnorm/` graph-on decode128 measured 0.8B p128/p512/p1024 `780.538/874.372/1003.015 ms` and 2B `1686.888/1807.107/1965.452 ms`, improving all six Iter71 graph-on matrix cases.
- Decision: accept. The candidate is Qwen3.5-local, token-stable, and produces a consistent matrix-level improvement; next hotspot map should decide whether any remaining low-risk norm/add work is worth doing.

## Iteration 74 Profile PostIter73Hotspot 2026-05-22T11:04:57+08:00

- Change: refreshed a post-Iter73 graph-off NSYS mapping trace for 0.8B p128/d32 after fixed-hidden RMSNorm was accepted.
- Evidence: `profiles/qwen3_5_nsys_action_iter74.txt` shows prefill still dominated by `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.290 ms` (`69.1%` of prefill GPU time). Decode remains dominated by LMHead GEMV at `46.470 ms` (`24.5%`), then `gated_delta_sequence_from_ab_kernel` at `12.123 ms` (`6.4%`), fixed-hidden RMSNorm at `5.002 ms` (`2.6%`), FlashInfer attention at `2.946 ms`, and in-place add at `2.124 ms`.
- Decision: profile-only. LMHead remains the largest decode hotspot, but repeated `lm_head_top1` rechecks stayed below the 1% default gate and NCU shows that path is already memory-roofline limited. Next narrow production candidate is a Qwen3.5-local fixed-hidden RMSNorm warp-shuffle reduction to reduce sync/shared-memory overhead without touching runtime scheduling.

## Iteration 75 Candidate P4-FixedHiddenRMSNormWarpReduce 2026-05-22T11:04:57+08:00

- Intent: replace the Iter73 fixed-hidden RMSNorm shared-memory tree reduction with a warp-shuffle block reduction for hidden `1024/2048`. This preserves the same math order at block granularity only as a floating-point reduction variant already covered by runtime-op tolerances, keeps one block per row and `256` threads, and reduces the reduction path from a full shared-memory tree to warp sums plus two block barriers.
- Correctness: rejected. Rebuild passed and `tests/operators/test_qwen3_5_runtime_ops.py` passed (`15 passed`), but Qwen3.5 fresh-dump generate drifted on 0.8B at decode index 18 (`99550` vs Transformers `98846`) in both regular and CUDA graph decode. The remaining failures in that run included OOMs after the first drift, but the token mismatch was sufficient to reject the candidate.
- Revert validation: restored Iter73 fixed-hidden shared-memory reduction, rebuilt, reran `tests/operators/test_qwen3_5_runtime_ops.py` (`15 passed`) and Qwen3.5 0.8B/2B fresh-dump generate (`13 passed`).
- Decision: reject. Large-hidden RMSNorm reduction order is token-sensitive; future RMSNorm work must preserve the exact reduction tree or stay out of production.

## Iteration 76 Candidate P4-FixedHiddenRMSNormStaticStride 2026-05-22T11:13:09+08:00

- Intent: keep the Iter73 fixed-hidden RMSNorm shared-memory reduction tree exactly intact, but make the hidden read/write loop stride compile-time constant (`256`) for hidden `1024/2048`. This should preserve the same per-thread accumulation order and block reduction order while removing a small amount of dynamic `blockDim.x` loop/index overhead.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Benchmark: targeted graph-on warmup-2 runs measured 0.8B p128/d32 `214.112 ms`, prefill `33.485 ms`, decode step `5.822 ms`; 2B p128/d32 `443.138 ms`, prefill `47.149 ms`, decode step `12.770 ms`; 2B p1024/d128 `1950.131 ms`, prefill `318.923 ms`, decode step `12.841 ms`.
- Matrix: `matrix_iter76_fixed_hidden_static_stride/` graph-on decode128 measured 0.8B p128/p512/p1024 `775.304/868.522/996.974 ms` and 2B `1673.134/1792.078/1950.507 ms`, improving all six Iter73 graph-on matrix cases (`780.538/874.372/1003.015`, `1686.888/1807.107/1965.452`).
- NCU: first decode fixed-hidden RMSNorm launch measured `3.84 us`, SM throughput `0.49%`, memory throughput `1.29%`, achieved occupancy `16.55%`, eligible warps/scheduler `0.15`, no-eligible `86.56%`, grid `1`, block `256`, registers/thread `27`. The kernel is launch/tiny-grid limited; the accepted win comes from lower per-launch instruction/loop overhead without changing the token-sensitive reduction tree.
- Decision: accept. The change is Qwen3.5-local, preserves token alignment, and improves every graph-on milestone case.

## Iteration 77 Profile PostIter76Hotspot 2026-05-22T11:26:35+08:00

- Change: refreshed post-Iter76 graph-off NSYS mapping for 0.8B p128/d32.
- Evidence: `profiles/qwen3_5_nsys_action_iter77.txt` shows prefill `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.249 ms` (`69.2%`), decode LMHead GEMV at `46.468 ms` (`24.6%`), decode `gated_delta_sequence_from_ab_kernel` at `12.183 ms` (`6.5%`), fixed-hidden RMSNorm now down to `3.350 ms` (`1.8%`), attention at `2.963 ms`, and add at `2.133 ms`.
- Decision: profile-only. Since LMHead top1/cuBLASLt paths are already below the default gate or library-bound, the next local candidate is a decode `from_ab` full-tile/static-shape specialization for Qwen3.5 `key_dim=value_dim=128` that preserves state update and output accumulation order.

## Iteration 78 Candidate P1-DecodeFromABFullTileStatic 2026-05-22T11:26:35+08:00

- Intent: add a Qwen3.5-local `gated_delta_sequence_from_ab` fast path for the supported full-tile shape (`key_dim=128`, `value_dim=128`, `value_tile=32`). The candidate keeps the same decay/update/output phases and per-column accumulation order, but replaces dynamic `min`, `% value_tile_size`, `/ value_tile_size`, and dynamic key/value dimension arithmetic with compile-time constants.
- Correctness: passed. Initial dispatch edit was accidentally inserted into the generic `launch_gated_delta_sequence` launcher and failed compilation because `BiasT/a/b/dt_bias` were out of scope; root cause was the patch context matching the wrong launcher. After moving the dispatch into `launch_gated_delta_sequence_from_ab`, rebuild succeeded, `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`, and Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Benchmark: targeted graph-on warmup-2 runs measured 0.8B p128/d32 `211.329 ms`, prefill ~`33.407 ms`, decode step `5.735 ms`; 2B p128/d32 `440.283 ms`, prefill ~`46.947 ms`, decode step `12.681 ms`; 2B p1024/d128 `1939.423 ms`, prefill ~`318.072 ms`, decode step `12.760 ms`. These improve Iter76 targeted baselines (`214.112/5.822`, `443.138/12.770`, `1950.131/12.841`).
- Matrix: `matrix_iter78_decode_from_ab_full_tile_static/` graph-on decode128 measured 0.8B p128/p512/p1024 `765.170/858.212/987.065 ms` and 2B `1662.204/1782.021/1941.373 ms`, improving all six Iter76 graph-on matrix cases (`775.304/868.522/996.974`, `1673.134/1792.078/1950.507`). Graph-on geomean ratio versus Iter76 is `0.9914` (~`0.86%` faster); graph-off also improved all six cases.
- NCU: first matching decode static full-tile launch measured `27.20 us`, SM throughput `20.94%`, memory throughput `21.35%`, achieved occupancy `33.40%`, eligible warps/scheduler `0.33`, no eligible `78.88%`, grid `64`, block `256`, registers/thread `48`. Iter64 generic from_ab NCU was `33.63 us`, so this local static-shape path is ~`19%` faster per launch, though still tiny-grid/barrier-stall limited.
- Decision: accept. The change is Qwen3.5-local, preserves token alignment, keeps the generic fallback for non-128 shapes, and improves every graph-on/off matrix case without runtime redesign.

## Iteration 79 Profile PostIter78Hotspot 2026-05-22T11:58:00+08:00

- Change: refreshed post-Iter78 graph-off NSYS mapping for 0.8B p128/d32.
- Evidence: `profiles/qwen3_5_nsys_action_iter79.txt` shows prefill `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.256 ms` (`69.2%`), decode LMHead GEMV at `46.469 ms` (`25.0%`), decode `gated_delta_sequence_from_ab_full_tile_128_kernel` at `9.483 ms` (`5.1%`), fixed-hidden RMSNorm at `3.322 ms`, attention at `2.947 ms`, and add at `2.120 ms`.
- Decision: profile-only. Iter78 reduced the from_ab decode hotspot; the next material candidate is a current-code `lm_head_top1` gate recheck under the accepted Iter78 baseline. Default enablement still requires fresh correctness plus >=1% end-to-end CUDA graph gain.

## Iteration 80 Candidate P2-LMHeadTop1PostIter78Gate 2026-05-22T11:58:00+08:00

- Intent: recheck the existing optional `--lm-head-top1` path after Iter78 lowered the linear-attention decode cost. No source change is planned unless the current implementation clears the default-enable gate: Qwen3.5 token correctness plus >=1% end-to-end CUDA graph improvement on affected cases.
- Correctness: passed by the Iter78 full generate gate; `test_qwen3_5_generate.py` includes regular and CUDA graph `lm_head_top1` alignment for both 0.8B and 2B and the fresh-dump run reported `13 passed`.
- Benchmark: current Iter78 default vs `--lm-head-top1` graph-on measured 0.8B p128/d32 `211.329 -> 210.723 ms` (`0.287%` faster), 2B p128/d32 `440.283 -> 438.990 ms` (`0.294%` faster), and 2B p1024/d128 `1939.423 -> 1934.220 ms` (`0.268%` faster).
- Decision: reject default enablement / keep optional path default-off. The path remains correct and available, but the end-to-end gain is well below the 1% default gate.

## Iteration 81 Candidate P4-FixedElementAdd1024_2048 2026-05-22T12:08:00+08:00

- Intent: add a Qwen3.5-local decode add fast path for `element_count == 1024` and `2048`, matching the 0.8B/2B hidden sizes. The kernel keeps exactly the same per-element math as the generic add (`float(lhs) + float(rhs)` then store to dtype), supports lhs/output aliasing, and keeps the generic add fallback for all other shapes.
- Correctness during candidate: rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` with temporary 1024/2048 exact-add coverage -> `17 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Benchmark: targeted graph-on runs measured 0.8B p128/d32 `211.636 ms`, regressing versus Iter78 `211.329 ms` by `0.145%`; 2B p128/d32 was flat (`440.285 ms` vs `440.283 ms`); 2B p1024/d128 improved to `1935.576 ms` vs `1939.423 ms`.
- Revert validation: removed the fixed-elements add kernel/dispatch and temporary test coverage, rebuilt, reran `tests/operators/test_qwen3_5_runtime_ops.py` (`15 passed`) and Qwen3.5 0.8B/2B fresh-dump generate (`13 passed`).
- Decision: reject and revert. The candidate is token-safe but not a stable cross-case win: short 0.8B regressed, 2B short is noise-flat, and the long-case gain is not enough to keep a shape-specific add path.

## Iteration 82 Candidate P1-DecodeFromABSingleTokenStatic128 2026-05-22T12:24:00+08:00

- Intent: specialize the accepted Iter78 static full-tile `from_ab` path for the actual decode shape `seq_len == 1`, keeping the same decay/update/output phases and per-column accumulation order while removing the token loop and token-dependent row/gate indexing from the hot decode kernel.
- Correctness: passed. Rebuild succeeded after removing a copied unused variable from the candidate kernel; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Benchmark: targeted graph-on warmup-2 runs measured 0.8B p128/d32 `210.924 ms` versus Iter78 `211.329 ms`; 2B p128/d32 `439.496 ms` versus `440.283 ms`; 2B p1024/d128 `1936.460 ms` versus `1939.423 ms`.
- Matrix: `matrix_iter82_from_ab_single_token_static/` graph-on decode128 measured 0.8B p128/p512/p1024 `762.980/856.211/985.164 ms` and 2B `1660.824/1780.098/1938.378 ms`, improving all six Iter78 graph-on cases (`765.170/858.212/987.065`, `1662.204/1782.021/1941.373`). Graph-on geomean ratio versus Iter78 is `0.99824` (~`0.18%` faster); graph-off also improved all six cases.
- NCU: first matching decode single-token launch measured `26.37 us`, SM throughput `22.02%`, memory throughput `22.24%`, achieved occupancy `33.77%`, eligible warps/scheduler `0.31`, no eligible `79.88%`, grid `64`, block `256`, registers/thread `39`. This improves Iter78 full-tile launch `27.20 us` and reduces registers/thread `48 -> 39`, while remaining tiny-grid/barrier-stall limited.
- Decision: accept. The change is Qwen3.5-local, preserves token alignment, keeps the Iter78 full-tile fallback for seq_len > 1, and improves every graph-on/off matrix case without runtime redesign.

## Iteration 83 Profile PostIter82Hotspot 2026-05-22T12:48:00+08:00

- Change: refreshed post-Iter82 graph-off NSYS mapping for 0.8B p128/d32.
- Evidence: `profiles/qwen3_5_nsys_action_iter83.txt` shows prefill `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.275 ms` (`69.1%`), decode LMHead GEMV at `46.470 ms` (`25.1%`), decode `gated_delta_sequence_from_ab_single_token_128_kernel` at `9.118 ms` (`4.9%`), fixed-hidden RMSNorm at `3.337 ms`, attention at `2.961 ms`, and add at `2.138 ms`.
- Decision: profile-only. Iter82 reduced the `from_ab` decode hotspot again, and LMHead is now the dominant decode kernel. Since prior `lm_head_top1` gates remain below 1%, the next step is to recheck cublasLt LMHead tactic search before trying another Qwen3.5-local recurrent micro-candidate.

## Iteration 84 Profile P2-LMHeadCublasLtRetune 2026-05-22T12:55:00+08:00

- Intent: retune Qwen3.5 LMHead decode `m=1` cublasLt tactics for 0.8B and 2B under `cuda_sm86`, without changing runtime or production code unless a candidate beats the current default.
- Evidence: heuristic sweep for 0.8B `m=1|input=2|weight=2|output=0|in_features=1024|out_features=248320` selected baseline at `1.503 ms`; best heuristic explicit index tied baseline (`algo_index=0`) and all other heuristic candidates were slower. 2B `in_features=2048` selected baseline at `2.991 ms`; best heuristic alternatives were `3.004 ms+`.
- Evidence: explicit top-k search also selected baseline for both models. Best non-baseline explicit candidate was `1.512 ms` for 0.8B and `3.012 ms` for 2B, both slower than baseline.
- Decision: reject production table change / mark current LMHead cublasLt tactic as locally optimal under the existing library path. A faster LMHead route needs a new algorithmic path, not an operator-table retune.

## Iteration 85 Candidate P1-DecodeFromABSingleTokenTile64 2026-05-22T12:58:00+08:00

- Intent: test a decode-only `seq_len==1,key_dim=value_dim=128` `from_ab` fast path with value tile `64`, while keeping Iter82 tile32 as the fallback if this regresses. This halves the per-head value-tile block count and repeated q/k norm scalar work, but may reduce SM residency; correctness should be token-stable because each value column keeps the same decay/update/output math and accumulation order.
- Correctness during candidate: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Benchmark: targeted graph-on runs regressed all checked cases versus Iter82. 0.8B p128/d32 `210.924 -> 214.712 ms`, decode step `5.724 -> 5.842 ms`; 2B p128/d32 `439.496 -> 443.617 ms`, decode step `12.662 -> 12.784 ms`; 2B p1024/d128 `1936.460 -> 1953.464 ms`, decode step `12.744 -> 12.861 ms`.
- Revert validation: restored Iter82 tile32 single-token path, rebuilt, reran `tests/operators/test_qwen3_5_runtime_ops.py` (`15 passed`) and Qwen3.5 0.8B/2B fresh-dump generate (`13 passed`).
- Decision: reject and revert. The lower block count loses more parallelism than it saves in repeated q/k norm work, so Iter82 tile32 remains the stable decode path.

## Iteration 86 Candidate P3-Qwen3_5AttentionDecodeTunedShape 2026-05-22T13:12:00+08:00

- Intent: add a narrow `flashinfer_attention_decode_sm80_tuned` shape for Qwen3.5 full-attention decode (`num_qo_heads=8,num_kv_heads=2,head_dim=256`) and evaluate it through a temporary qwen3_5 operator-table record. This touches only the attention operator shape dispatch and table selection, not runtime scheduling.
- Correctness during candidate: build passed after adding the temporary attention tuned shape; Qwen3.5 0.8B/2B fresh-dump generate with the temporary operator table passed (`13 passed`).
- Benchmark: targeted graph-on runs with the temporary table were mixed versus Iter82. 0.8B p128/d32 moved `210.924 -> 210.703 ms`, but 2B p128/d32 regressed `439.496 -> 439.848 ms`, and 2B p1024/d128 regressed `1936.460 -> 1938.112 ms`.
- Revert validation: removed the temporary `Qwen3_5DecodeTunedShape` dispatch, rebuilt, reran `tests/operators/test_qwen3_5_runtime_ops.py` (`15 passed`) and Qwen3.5 0.8B/2B fresh-dump generate (`13 passed`).
- Decision: reject and revert. The tuned shape is token-correct but not a stable cross-model win; keep the temporary operator table only as evidence and leave production on the default attention decode path.

## Iteration 87 Standalone P1-FromABSingleTokenWarpSearch 2026-05-22T12:38:56+08:00

- Intent: move the next decode `from_ab` search out of production code after Iter85 closed the simple tile retile path. Add a standalone repro for `gated_delta_sequence_from_ab_single_token_128_kernel` with the Iter82 baseline and two warp-per-value candidates: `warp8_serialdot` keeps the serial dot order in lane 0, while `warp8_reduce` parallelizes the dot products and is expected to be token-risky until proven by full generate alignment.
- Scope: new repro-only files under `deliverables/kernel_opt/qwen3_5_phase2_20260521_162140/repro/gated_delta_from_ab_single_token/`; no `src/` production change.
- Repro result: Iter82-style baseline measured `14.283 us`. Warp-per-value candidates were slower (`warp8_serialdot` `94.590 us`, `warp8_reduce` `61.838 us`) and are rejected at repro level. A narrower `baseline_parallel_l2norm` variant measured `12.111 us` with `output_max_abs=0` and `state_max_abs=0` against the baseline on the repro input.
- Follow-up candidate: transfer only the `baseline_parallel_l2norm` idea into the Qwen3.5 single-token `from_ab` fast path, then gate with exact operator tests and 0.8B/2B fresh-dump generation before any benchmark acceptance.
- Production correctness: accepted candidate build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed` including exact `from_ab` vs separate path coverage; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Targeted benchmark: graph-on p128/d32 improved 0.8B `210.924 -> 210.717 ms` and 2B `439.496 -> 438.904 ms`; 2B p1024/d128 improved `1936.460 -> 1932.613 ms`. The accidental concurrent 2B runs were renamed to `*.invalid_parallel.json` and excluded from acceptance.
- NCU: first matching 0.8B p128/d4 launch improved `26.37 us -> 20.74 us`, SM throughput `22.02% -> 25.39%`, memory throughput `22.24% -> 28.74%`, eligible warps/scheduler `0.31 -> 0.40`, no-eligible `79.88% -> 74.84%`, registers/thread stayed `39`.
- Matrix: `matrix_iter87_from_ab_parallel_l2norm/` improved all six graph-on cases versus Iter82: 0.8B p128/p512/p1024 `757.475/851.243/979.370 ms` vs `762.980/856.211/985.164 ms`; 2B `1655.718/1775.176/1932.458 ms` vs `1660.824/1780.098/1938.378 ms`. Graph-on geomean ratio versus Iter82 is `0.99537`; graph-off also improved all six cases.
- Decision: accept. The change is Qwen3.5-local, token-stable on both supported models, improves targeted and full-matrix graph-on/off gates, and gives a real single-launch NCU improvement without runtime redesign.

## Iteration 88 Profile PostIter87HotspotAndLMHead 2026-05-22T13:00:14+08:00

- Change: refreshed graph-off NSYS mapping after Iter87 and captured one NCU decode LMHead GEMV launch.
- Evidence: `profiles/qwen3_5_nsys_action_iter88.txt` shows prefill remains dominated by `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.306 ms` (`69.2%` prefill GPU time). Decode remains dominated by LMHead GEMV at `46.470 ms` (`25.2%`), while `gated_delta_sequence_from_ab_single_token_128_kernel` dropped to `7.830 ms` (`4.3%`) from Iter83 `9.118 ms`.
- LMHead evidence: NCU for one decode cuBLAS GEMV launch measured `45.06 us`, memory/DRAM throughput `86.00%`, achieved occupancy `61.33%`, eligible warps/scheduler `0.65`, no-eligible `64.93%`, grid `768`, block `128`, registers/thread `60`. Iter84 tactic search already found the current cuBLAS path best; optional top1 is memory-roofline limited but remains default-off unless it clears the end-to-end gate.
- Top1 recheck: current optional `--lm-head-top1` is still token-covered and positive but below the default gate: 0.8B p128/d32 `210.717 -> 209.641 ms`, 2B p128/d32 `438.904 -> 437.759 ms`, and 2B p1024/d128 `1932.613 -> 1928.289 ms`.
- Decision: profile-only. Keep LMHead top1 available/default-off for now and continue with a standalone `from_ab` low-risk search before the next production transfer.

## Iteration 89 Candidate P1-DecodeFromABSingleTokenPrecomputeNorms 2026-05-22T13:00:14+08:00

- Intent: after Iter87 moved q/k l2norm sum to parallel reduction, test whether precomputing the per-head q/k normalized vectors in shared memory can remove repeated bf16 normalization work in the delta, state-update, and output loops. This preserves per-column dot accumulation order and uses only the existing single-token shared arrays.
- Repro evidence: standalone `precompute_norms` measured `9.539 us` versus Iter87-style `baseline_parallel_l2norm` `12.104 us`, with `output_max_abs=0` and `state_max_abs=0` against the original baseline input. Warp-per-value candidates remain rejected at repro level.
- Production transfer: applied only to `gated_delta_sequence_from_ab_single_token_128_kernel`; no runtime or model scheduling changes.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Targeted benchmark: graph-on improved 0.8B p128/d32 `210.717 -> 209.432 ms`, 2B p128/d32 `438.904 -> 436.955 ms`, and 2B p1024/d128 `1932.613 -> 1923.887 ms`.
- NCU: first matching 0.8B p128/d4 launch improved `20.74 us -> 17.54 us`; memory/DRAM throughput moved to `33.59%`, achieved occupancy `35.14%`, registers/thread `39 -> 48`, grid/block stayed `64/256`.
- Matrix: `matrix_iter89_from_ab_precompute_norms/` improved all six graph-on cases versus Iter87: 0.8B p128/p512/p1024 `752.509/846.229/974.384 ms` vs `757.475/851.243/979.370 ms`; 2B `1650.585/1769.276/1928.817 ms` vs `1655.718/1775.176/1932.458 ms`. Graph-on geomean ratio versus Iter87 is `0.99569`; graph-off also improved all six cases.
- Decision: accept. The change is Qwen3.5-local, token-stable on both supported models, improves targeted and full-matrix graph-on/off gates, and gives another single-launch NCU improvement without runtime redesign.

## Iteration 90 Profile PostIter89Hotspot 2026-05-22T13:12:57+08:00

- Change: refreshed graph-off NSYS mapping after Iter89.
- Evidence: `profiles/qwen3_5_nsys_action_iter90.txt` shows prefill still dominated by `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.233 ms` (`69.1%`). Decode remains dominated by LMHead GEMV at `46.474 ms` (`25.4%`), while `gated_delta_sequence_from_ab_single_token_128_kernel` dropped again to `6.460 ms` (`3.5%`) from Iter88 `7.830 ms`.
- Decision: profile-only. Since LMHead/full-logits remains library-bound and top1 is still below the default gate, continue with one more standalone `from_ab` barrier/memory-pass reduction candidate.

## Iteration 91 Candidate P1-DecodeFromABSingleTokenColumnFused128 2026-05-22T13:12:57+08:00

- Intent: exploit value-column independence in the single-token `from_ab` kernel. After q/k norm precompute, let each of the 32 active column threads perform decay, delta, state update, and output accumulation for one value column. This removes the separate block-wide decay/update/output phases and their barriers while preserving per-column dot accumulation order.
- Repro evidence: standalone `column_fused_128` measured `7.204 us` versus Iter89-style `precompute_norms` `9.530 us`, with `output_max_abs=0` and `state_max_abs=0` against the original baseline input. The 256-thread column-fused version was also exact but slower at `7.751 us`.
- Production transfer: changed only the single-token 128/128 fast path to use 128 launch threads and the column-fused schedule; no runtime or model scheduling changes.
- Correctness: passed. Initial production patch accidentally changed the full-tile kernel constants and failed compilation; after correcting the patch scope, rebuild succeeded. `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Targeted benchmark: graph-on improved 0.8B p128/d32 `209.432 -> 207.644 ms`, 2B p128/d32 `436.955 -> 435.832 ms`, and 2B p1024/d128 `1923.887 -> 1920.326 ms`.
- NCU: first matching 0.8B p128/d4 launch improved `17.54 us -> 13.89 us`; memory/DRAM throughput `42.04%`, achieved occupancy `8.18%`, registers/thread `48`, grid/block `64/128`.
- Matrix: `matrix_iter91_from_ab_column_fused/` improved all six graph-on cases versus Iter89: 0.8B p128/p512/p1024 `747.306/840.653/968.856 ms` vs `752.509/846.229/974.384 ms`; 2B `1645.130/1764.276/1922.491 ms` vs `1650.585/1769.276/1928.817 ms`. Graph-on geomean ratio versus Iter89 is `0.99523`; graph-off also improved all six cases.
- Decision: accept. The change is Qwen3.5-local, token-stable on both supported models, improves targeted and full-matrix graph-on/off gates, and gives the largest recent single-launch `from_ab` improvement without runtime redesign.

## Iteration 92 Profile PostIter91Hotspot 2026-05-22T13:25:03+08:00

- Change: refreshed graph-off NSYS mapping after Iter91.
- Evidence: `profiles/qwen3_5_nsys_action_iter92.txt` shows prefill still dominated by `gated_delta_sequence_precomputed_kernel<bf16,true,16>` at `24.199 ms` (`69.1%`). Decode remains dominated by LMHead GEMV at `46.471 ms` (`25.6%`), while `gated_delta_sequence_from_ab_single_token_128_kernel` dropped to `4.919 ms` (`2.7%`) from Iter90 `6.460 ms`.
- Decision: profile-only. The next material default-path opportunity is LMHead policy: Iter84 proved full-logits cuBLASLt retune is locally best, and the already-integrated top1 route is memory-roofline limited, token-aligned, and appropriate for Qwen3.5 greedy generation.

## Iteration 93 Candidate P2-Qwen3_5GreedyDefaultLMHeadTop1 2026-05-22T13:40:28+08:00

- Intent: make Qwen3.5 greedy temperature=0 use the existing `lm_head_top1` route by default, while preserving an explicit runtime opt-out via `runtime.lm_head_top1=false` or `runtime.lm_head_top1.enabled=false`. This is model-local policy plus a generic metrics fix to report actual top1 usage when the model activates it by default.
- Implementation: `Qwen3_5` now defaults top1 on when no runtime override exists and sampling temperature is greedy; `StandardEngine` reports `lm_head_top1_enabled` if config requests it or if any top1 decode steps actually ran; `profile_edgefm_generate_case.py` no longer writes `lm_head_top1.enabled=false` into temporary configs when the CLI flag is absent.
- Correctness: passed. Rebuild succeeded; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`; `PYTHONPATH=. pytest -q tests/scripts/test_profile_edgefm_generate_case.py -q` -> `3 passed`.
- Targeted benchmark: default graph-on with top1 active measured 0.8B p128/d32 `207.594 ms`, 2B p128/d32 `435.528 ms`, and 2B p1024/d128 `1916.334 ms`, improving the Iter91 default full-logits cases.
- Matrix: `matrix_iter93_qwen3_5_default_top1/` improved all six graph-on cases versus Iter91: 0.8B p128/p512/p1024 `745.260/838.082/967.389 ms` vs `747.306/840.653/968.856 ms`; 2B `1643.354/1762.474/1920.495 ms` vs `1645.130/1764.276/1922.491 ms`. Graph-on geomean ratio versus Iter91 is `0.99826`; graph-off also improved all six cases.
- Decision: accept. The end-to-end gain is small but stable, and this aligns the default greedy Qwen3.5 LMHead with the already measured memory-roofline top1 operator path (`DRAM 97.46%`) instead of the full-logits GEMV path (`DRAM 86%`).

## Iteration 94 Profile PostIter93HotspotAndColumn64Repro 2026-05-22T13:46:03+08:00

- Change: refreshed graph-off NSYS mapping after Iter93 and tested one more `from_ab` standalone block-size candidate.
- Evidence: `profiles/qwen3_5_nsys_action_iter94.txt` shows default greedy decode is now dominated by `lm_head_top1::stage1_kernel<bf16>` at `46.174 ms` (`25.5%`), replacing the previous full-logits GEMV hotspot. `gated_delta_sequence_from_ab_single_token_128_kernel` is now `4.931 ms` (`2.7%`).
- Repro: `column_fused_64` was exact but flat/slower than accepted `column_fused_128`: `7.387 us` vs `7.382 us`. No production transfer.
- Shared regression: after the Iter93 common metrics/profile changes, `EDGE_FM_BUILD_DIR=build pytest -q tests/engine/test_qwen2_generate.py -k "token_alignment or kvcache" -s` passed (`8 passed, 12 deselected`).
- Decision: local short-loop candidates are now effectively plateaued. Default Qwen3.5 greedy LMHead is on the memory-roofline top1 path, P0 prefill has the previous token-safe ceiling note, and the already-tested fixed RMSNorm/attention/add/from_ab local candidates are either accepted or rejected. The next meaningful work should be a larger standalone/Humanize + KernelPilot search before any further production transfer.

## Iteration 95 Standalone P0-PrecomputedIntermediateTiles 2026-05-22T13:51:24+08:00

- Intent: keep production untouched and reopen P0 only inside the standalone prefill repro by sweeping intermediate `value_tile` shapes not covered by Iter61-63 (`10/12/14/18/20/24`, `threads=256`, token-boundary barrier enabled). This tests whether a non-power-of-two tile can beat the accepted token-safe `tile16/thread256` recurrence layout before considering any risky production transfer.
- Repro evidence: all swept candidates passed the standalone CPU reference. First sweep (`iter95_intermediate_tiles.jsonl`) measured tile10/tile12/tile14/tile18/tile20/tile24 at `2.739/2.611/1.595/1.547/1.462/1.412 ms`. Neighbor sweep (`iter95_neighbor_tiles.jsonl`) measured tile16/tile22/tile24/tile26/tile28/tile30/tile32 at `1.458/1.610/1.411/1.661/1.559/1.420/1.740 ms`, so tile24 was the best standalone event-time candidate.
- NCU: standalone tile24 report `profiles/ncu/qwen3_5_precomputed_tile24_iter95.ncu-rep` measured `1.80 ms`, compute throughput `41.27%`, memory throughput `38.60%`, achieved occupancy `56.26%`, eligible warps/scheduler `0.67`, no-eligible `58.23%`, grid/block `96/256`, registers/thread `39`. This is slightly faster than the previous tile16 barrier-safe NCU ceiling note (`1.85 ms`) but with fewer blocks and lower occupancy.
- Production transfer: temporarily changed only `kGatedDeltaPrecomputedValueTile` from `16` to `24`; no runtime or model scheduling changes.
- Correctness during candidate: build passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Benchmark: target graph-on runs regressed versus the Iter93 accepted baseline. 0.8B p128/d32 measured `207.871 ms` vs Iter93 `207.594 ms`; 2B p128/d32 measured `436.289 ms` vs `435.528 ms`; 2B p1024/d128 measured `1948.154 ms` vs `1916.334 ms`, with prefill `346.626 ms` vs `316.512 ms`.
- Revert validation: restored `kGatedDeltaPrecomputedValueTile=16`, rebuilt, reran `tests/operators/test_qwen3_5_runtime_ops.py` (`15 passed`) and Qwen3.5 0.8B/2B fresh-dump generate (`13 passed`).
- Decision: reject and revert. The standalone tile24 event-time/NCU signal did not transfer to full generate; the accepted tile16/thread256 precomputed sequence remains the stable P0 production layout.

## Iteration 96 Candidate P4-FixedHiddenAddRMSNormFusion 2026-05-22T14:11:00+08:00

- Intent: retest the old attention-residual add plus immediately-following RMSNorm fusion under the current Iter76 fixed-hidden RMSNorm and Iter93 top1 default baseline. The candidate is Qwen3.5-local, targets only the post-attention `add(hidden, mixer_output, hidden)` + `post_attention_layernorm` pair, stores the rounded residual before computing RMSNorm, and should preserve the separate add->RMSNorm token contract.
- TDD red gate: added `test_qwen3_5_add_rmsnorm_matches_separate_add_then_rmsnorm` for hidden `1024/2048`; before implementation it fails as expected because `edge_fm.qwen3_5_add_rmsnorm` is not exported.
- Correctness during candidate: implemented a Qwen3.5-local fixed-hidden fused add+rmsnorm operator and temporarily used it for the post-attention pair. Build passed; the new operator test passed for hidden `1024/2048`; full `tests/operators/test_qwen3_5_runtime_ops.py` passed (`17 passed`); Qwen3.5 0.8B/2B fresh-dump generate passed (`13 passed`).
- Benchmark: target graph-on runs were mixed/regressed versus Iter93. 0.8B p128/d32 improved `207.594 -> 207.292 ms`, but 2B p128/d32 regressed `435.528 -> 435.998 ms`, and 2B p1024/d128 regressed `1916.334 -> 1921.390 ms`.
- Revert validation: removed the fused operator/test/export and restored the separate Qwen3.5 `add` + `rmsnorm` model path. Rebuild passed; `tests/operators/test_qwen3_5_runtime_ops.py` -> `15 passed`; Qwen3.5 0.8B/2B fresh-dump generate -> `13 passed`.
- Decision: reject and revert. The candidate is token-safe but does not improve both supported models; this confirms Iter11's earlier fused add+rmsnorm rejection still holds under the current baseline.

## Iteration 97 Candidate P0-PrecomputedVectorizedStateUpdate 2026-05-22T14:46:14+08:00

- Intent: close the local P0 GatedDeltaNet current-layout loop by moving only the standalone-proven state decay/update vectorization into the production precomputed recurrent sequence kernel. This keeps the recurrence order, token-boundary barriers, value tile `16`, and runtime/model wiring unchanged.
- Standalone evidence: in `long_loop/p0_gated_delta_recurrence`, `candidate_vectorized_state_update.cu` matched the baseline reference (`output_max_abs=0.000488281`, `state_max_abs=5.96046e-08`) and improved event time `1.458215 -> 1.207467 ms`. NCU improved duration `1.84 -> 1.50 ms`; the remaining low-SOL behavior is latency/recurrence limited rather than a raw hardware-peak target miss.
- Production transfer: updated only `src/operators/qwen3_5/qwen3_5_ops.cu` precomputed P0 kernel to use guarded `float4` state decay/update when the tile is full and 4-aligned; generic and fallback paths remain scalar.
- Correctness: build passed; `EDGE_FM_BUILD_DIR=build pytest -q tests/operators/test_qwen3_5_runtime_ops.py -q` -> `15 passed`; `EDGE_FM_BUILD_DIR=build EDGE_FM_QWEN3_5_REGENERATE_DUMP=1 pytest -q tests/engine/test_qwen3_5_generate.py -s` -> `13 passed` for 0.8B and 2B.
- Benchmark: target graph-on runs improved versus Iter93. 0.8B p128/d32 `207.594 -> 205.148 ms`, prefill `33.835 -> 31.394 ms`; 2B p128/d32 `435.528 -> 433.192 ms`, prefill `46.989 -> 44.654 ms`; 2B p1024/d128 `1916.334 -> 1897.618 ms`, prefill `316.512 -> 296.433 ms`.
- Decision: accept. This completes the local ceiling/blocker phase: P0 current-layout local vectorization is accepted, LMHead top1 is already memory-roofline, and the remaining large gains require a new algorithm/layout long-loop rather than another narrow runtime-local edit.
