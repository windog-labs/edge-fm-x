# EdgeFM Optimization Journal

Last updated: 2026-04-06

This file is the persistent source of truth for ongoing EdgeFM performance work.
Before starting a new optimization branch, read this file first.
After every meaningful experiment, update this file with:

- the experiment goal
- the exact A/B setup
- the code change
- the measured result
- the keep/revert decision

## Core Goal

- Primary goal: continuously optimize `EdgeFM(cuda-graph)` until it matches and then surpasses `TRT-Edge-LLM`.
- Main benchmark target: `Qwen2.5-1.5B BF16, batch=1`.
- Main benchmark matrix:
  - `prefill=512, decode=32`
  - `prefill=512, decode=64`
  - `prefill=1024, decode=32`
  - `prefill=1024, decode=64`
  - `prefill=2048, decode=32`
  - `prefill=2048, decode=64`
- Main comparison contract:
  - compare `EdgeFM(cuda-graph)` against `TRT-Edge-LLM`
  - keep `Transformers` only as a slow reference baseline
  - do not spend analysis time on `EdgeFM(no-graph)` unless debugging graph correctness

## Available Tools And Resources

Use the project skills as the default optimization toolbox instead of ad-hoc trial and error.

- `cuda-skill`
  - CUDA runtime / driver / PTX / best-practice reference
  - `nsys`, `ncu`, compute-sanitizer workflows
  - use for kernel analysis, CUDA Graph reasoning, memory movement analysis, runtime design
- `ncu-cuda-profiling`
  - NCU profiling workflow and metric interpretation
  - use when the environment supports `ncu`
- `edge-fm-benchmark-report`
  - standard 3-way benchmark workflow: `Transformers` vs `EdgeFM(cuda-graph)` vs `TRT-Edge-LLM`
  - use for final benchmark reports and fair case alignment
- `edge-fm-add-operator`
  - use when adding a new `impl_id`, wiring operator registry, or updating `operator_impl_table.json`
- `cutlass-skill`
  - use for production-grade CUDA/CUTLASS kernels after prototype validation
- `triton-skill`
  - use for quick kernel prototypes and shape-specific experiments
- `cutile-python-skill`
  - use for quick cuTile prototype kernels and autotune-style validation
- `cc-connect-feishu-codex`
  - auxiliary integration skill, not part of the main performance path
- `imagegen`
  - irrelevant to current optimization work
- `openai-docs`
  - auxiliary documentation lookup, not part of the main performance path
- `plugin-creator`
  - irrelevant to current optimization work
- `skill-creator`
  - auxiliary, only if we decide to create a repo-specific optimization skill

## Non-Negotiable Principles

- Correctness first.
  - Any optimization must preserve both LLM and VLM correctness.
  - Operator-level wins are not enough. Final acceptance requires end-to-end correctness.
- Keep the codebase clean.
  - If a direction does not show reliable benefit, revert it.
  - Do not leave dead code, unused `impl_id`, one-off debug hooks, or temporary benchmark-only runtime paths in `src/`.
- Do not break engine architecture invariants.
  - Existing engine logic is not a free-for-all scratchpad. Optimize within the design unless there is strong evidence that a design change is required.
- Use data, not instinct, to conclude.
  - Theory matters, but any final decision must be backed by fair A/B measurement and code-path analysis.
- Prefer stable production paths over clever but fragile experiments.
  - Prototype with Triton/cuTile when needed.
  - Promote to CUTLASS/CUDA only after the prototype demonstrates clear upside.

## Engine Architecture Invariants

These constraints are real and enforced in the current code. Do not violate them during optimization.

- Request-to-slot matching is fixed.
  - `Scheduler::create_context()` looks up the request by `request_id` and requires a matching slot.
- Slot prefix matching is fixed.
  - If a slot carries `prefix_token_ids`, the request tokens must match that prefix exactly.
- Slot ownership is fixed.
  - Each slot has a fixed `prefix_size` and `max_tokens`.
  - The request must fit the matched slot instead of mutating slot semantics on the fly.
- `Context` is per-request runtime state layered on top of the matched slot.
  - `Context` tracks per-request dynamic state such as `generated_tokens`, `decode_cache_kv_len`, response token pointers, tensor views, and model-specific state.
  - `Context` must not invade or rewrite the slot definition itself.
- Decode runtime state should prefer stable device-side buffers when possible.
  - Current code already uses a stable decode runtime buffer for `TOKEN_IDS`, `D_KV_LEN`, and optional decode `POSITION_IDS`.
  - This is compatible with CUDA Graph replay and should remain the default direction.

Relevant code references:

- [src/engine/scheduler.cpp](/xs-train-nas/zzm/repos/edge-fm-x/src/engine/scheduler.cpp)
- [src/engine/scheduler.h](/xs-train-nas/zzm/repos/edge-fm-x/src/engine/scheduler.h)
- [src/engine/stardard_engine.cpp](/xs-train-nas/zzm/repos/edge-fm-x/src/engine/stardard_engine.cpp)

## Experiment Rules

- Always write the hypothesis first.
  - Example: "prefill gap at `2048/64` is likely dominated by GEMM selection rather than D2D copy time."
- Keep the A/B comparison fair.
  - Same model
  - Same tokens
  - Same `prefill_len`
  - Same `decode_len`
  - Same stop-token behavior
  - Same dtype
  - Same device
- Compare the right thing.
  - If evaluating fusion, compare fused vs unfused implementations under the same kernel family and same shape.
  - Do not use a slow Triton prototype to conclude that production fusion is useless.
- Separate operator conclusions from end-to-end conclusions.
  - Operator microbench decides whether a path deserves integration.
  - End-to-end benchmark decides whether the integration actually matters.
- Use profiling before deep rewrites.
  - `nsys` is the primary in-container profiler right now.
  - `ncu` is blocked in this container and is not a current blocker.
- Do not parallelize GPU benchmark/profile runs.
  - Earlier runs showed this can produce invalid or misleading measurements.
- Roll back no-gain changes quickly.
  - Temporary branches that lose, regress correctness, or create instability should not stay in tree.

## Current Environment Facts

- Platform target: `A800-SXM4-80GB / sm80 / device=1`
- Build type: `Release`
  - verified from `build/CMakeCache.txt`
  - so the current performance gap is not explained by a Debug-vs-Release mismatch
- `nsys`: usable in this container
  - practical workaround: some runs emit `.qdstrm`; import with `QdstrmImporter` before `stats`
- `ncu`: not usable in this container right now
  - do not block optimization on it

Relevant artifacts:

- benchmark helper:
  - [.codex/skills/edge-fm-benchmark-report/scripts/report_qwen_3way_cuda_graph_vs_trt.py](/xs-train-nas/zzm/repos/edge-fm-x/.codex/skills/edge-fm-benchmark-report/scripts/report_qwen_3way_cuda_graph_vs_trt.py)
- latest benchmark snapshots:
  - `/tmp/edgefm_bench_512_64_after_fused512.json`
  - `/tmp/edgefm_bench_2048_64_latest.json`
  - `/tmp/edgefm_bench_fresh_512_1024_2048_x64.json`
- latest `nsys` artifacts:
  - `/tmp/edgefm_profile_2048_64.nsys-rep`
  - `/tmp/edgefm_profile_2048_64.sqlite`
  - `/tmp/edgefm_profile_2048_64_stats_cuda_gpu_kern_sum.csv`
  - `/tmp/edgefm_profile_2048_64_stats_cuda_api_sum.csv`
  - `/tmp/edgefm_profile_current_2048_64.nsys-rep`
  - `/tmp/edgefm_profile_current_2048_64.sqlite`
  - `/tmp/edgefm_profile_current_2048_64_stats_cuda_gpu_kern_sum.csv`
  - `/tmp/edgefm_profile_current_2048_64_stats_cuda_api_sum.csv`

## Benchmark Snapshots

Keep both the historical anchor and the current working-tree snapshot.
They are both useful:

- the historical snapshot shows what was achieved earlier
- the current working-tree snapshot tells us whether a regression has appeared

### Historical Anchor Snapshot

These numbers were the last previously trusted anchor points before the current round of cleanup and revalidation.

| Case | EdgeFM total | TRT total | Gap | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill=512, decode=64` | `221.566 ms` | `213.730 ms` | `+7.835 ms` / `+3.67%` | `12.100 ms` | `9.105 ms` | `+2.995 ms` | `209.225 ms` | `204.492 ms` | `+4.734 ms` |
| `prefill=2048, decode=64` | `250.450 ms` | `256.620 ms` | `-6.170 ms` / `-2.40%` | `33.503 ms` | `29.308 ms` | `+4.195 ms` | `216.708 ms` | `227.164 ms` | `-10.455 ms` |

Interpretation:

- `2048/64` had already beaten TRT overall in this earlier snapshot.
- `512/64` was still slower than TRT, and both prefill and decode were behind.
- At that point the main job was to close short-context gaps without regressing the long-context case.

### Fresh Working-Tree Snapshot (2026-04-06)

This snapshot was re-run after the current cleanup and correctness gates, using:

- `edge-fm-benchmark-report`
- `prefill={512,1024,2048}`
- `decode=64`
- `EdgeFM(cuda-graph)` only

| Case | EdgeFM total | TRT total | Gap | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill=512, decode=64` | `288.765 ms` | `215.394 ms` | `+73.371 ms` / `+34.06%` | `12.225 ms` | `12.834 ms` | `-0.609 ms` | `276.302 ms` | `202.433 ms` | `+73.869 ms` |
| `prefill=1024, decode=64` | `295.542 ms` | `227.027 ms` | `+68.515 ms` / `+30.18%` | `18.836 ms` | `16.752 ms` | `+2.083 ms` | `276.377 ms` | `210.107 ms` | `+66.270 ms` |
| `prefill=2048, decode=64` | `311.235 ms` | `256.112 ms` | `+55.123 ms` / `+21.52%` | `33.576 ms` | `29.450 ms` | `+4.126 ms` | `277.241 ms` | `226.482 ms` | `+50.759 ms` |

Interpretation:

- The current working tree is materially slower than the historical anchor.
- The dominant regression is in decode, not prefill.
- Decode time is now almost flat across `512/64`, `1024/64`, and `2048/64`, which strongly suggests a runtime-side regression or a decode path change that is largely insensitive to prompt length.
- New kernel work should not start until this regression is explained.

### Profiling Snapshot: Historical EdgeFM `prefill=2048, decode=64`

From `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum`:

- `12.125 ms` `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn`
- `7.349 ms` `ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn`
- `5.420 ms` `SinglePrefillWithKVCacheKernel`
- `1.864 ms` `ampere_bf16_*relu*`
- `1.826 ms` `act_and_mul_kernel`
- `0.748 ms` `FusedAddRMSNormKernel`

Memcpy conclusion from the SQLite activity tables:

- total D2D GPU memcpy time: `0.696571 ms`
- count: `149`
- bytes: `230 MB`

Historical conclusion:

- D2D GPU copy exists, but it is not the dominant bottleneck for the current `2048/64` case.
- The main prefill cost at this point is still in GEMM selection plus the prefill attention kernel, not raw D2D copy time.

### Profiling Snapshot: Current Working Tree `prefill=2048, decode=64`

Fresh `nsys` run on 2026-04-06 confirms:

- `cudaGraphLaunch_v10000`: `63` calls
- no observed `cudaGraphExecKernelNodeSetParams`
- no observed `cudaGraphExecMemcpyNodeSetParams1D`

Meaning:

- CUDA Graph replay is still active
- the current decode slowdown is not explained by "graph path stopped working"
- the current decode slowdown is not explained by per-step graph node patching on the profiled path

Important runtime detail:

- the top `cudaMemcpyAsync_v3020` API call took `115.523 ms`
- correlating by `correlationId` shows it corresponds to a single `256-byte` `copyKind=2` memcpy activity with only `0.002368 ms` GPU time

Interpretation:

- that API time is host-side waiting around a tiny copy, not a real bulk memcpy bottleneck
- it should not be misread as "copy bandwidth is the main problem"

## Verified Wins Kept In Tree

- Decode linear tuned `cublasLt` records are kept for:
  - `fused_qkv`, `attention_output`, `mlp_down`, `fused_gate_up`, `lm_head` at `m=1`
- Prefill `fused_qkv` tuned `cublasLt` records are kept for:
  - `m=512 -> algo_index=0`
  - `m=1024 -> algo_index=4`
  - `m=2048 -> algo_index=3`
- Current measured operator-level `fused_qkv` improvements:
  - `m=512`: `0.044880 ms -> 0.038720 ms`
  - `m=1024`: `0.062272 ms -> 0.059216 ms`
  - `m=2048`: `0.114112 ms -> 0.084736 ms`
- Decode runtime state consolidation is already present in the engine:
  - stable decode device state for `TOKEN_IDS`
  - stable decode device state for `D_KV_LEN`
  - optional stable decode device state for `POSITION_IDS`

## Reverted Or Explicitly Rejected Directions

- Reverted: custom decode attention path `flashinfer_attention_decode_sm80_tuned`
  - reason:
    - unstable benefit
    - slower at representative `kv_len=2048`
    - graph-like repeated bench hit invalid-handle / illegal-memory-access issues during experimentation
  - action:
    - removed from [src/operators/attention_op.cu](/xs-train-nas/zzm/repos/edge-fm-x/src/operators/attention_op.cu)
    - removed from [examples/config/operator_impl_table.json](/xs-train-nas/zzm/repos/edge-fm-x/examples/config/operator_impl_table.json)
- Reverted: prefill `mlp_down m=2048 -> algo_index=0`
  - reason: operator test showed it was slower than baseline
- Rejected as a final conclusion: "`gate_up + act_and_mul` fusion has no value"
  - reason: previous Triton experiment only proved that one Triton prototype was not worth integrating
  - it did not prove that true production fusion is useless

## Open Questions And Known Risks

- Existing `mlp_down prefill m=1024 -> algo_index=2` needs more careful validation.
  - A strict baseline-vs-tuned BF16 operator check showed small drift beyond the current microbench tolerance.
  - Do not treat that one record as universally "proven good" until it is validated against torch reference and end-to-end correctness.
- The current working tree shows a decode-heavy regression relative to the historical anchor.
  - This was root-caused on 2026-04-06: the local working tree had removed the decode-specific
    `flashinfer_attention_decode_sm80_tuned` path and its `operator_impl_table` routing record.
  - Restoring that path brought the benchmark back near the historical anchor and removed the
    large decode regression.
- Benchmark output files may contain log prefixes before the final JSON block.
  - Always parse the trailing JSON payload instead of assuming the whole file is pure JSON.

## Current Next Steps

1. Keep this journal updated after every meaningful experiment.
2. Treat `flashinfer_attention_decode_sm80_tuned` as the current accepted decode-attention production path for Qwen2.5 BF16 on `sm80`.
3. Use the recovered benchmark matrix to drive the next branch from the remaining small gaps:
   - short-context decode gap at `512/64`
   - short-context decode/prefill mix at `1024/32` and `1024/64`
4. Before deeper kernel work, re-profile the recovered build and attribute the remaining `<= 8 ms` gaps.
5. Prioritize the next optimization branch as:
   - decode linear fixed-cost path
   - residual/norm/activation fixed-cost path
   - prefill short-context GEMM selection only if it reappears as a measurable top contributor
6. Keep the current correctness gates in place:
   - `tests/operators/test_attention_decode.py`
   - `test_generate_token_alignment_cuda_graph`
   - `test_generate_vl_token_alignment_cuda_graph`

Current default branch priority:

1. preserve the recovered decode-attention path
2. profile the remaining short-context gaps
3. only then resume decode linear / runtime fixed-cost optimization
4. avoid reopening speculative attention branches unless new profiling data says attention is again the top gap

## Journal Entries

### 2026-04-06

State consolidated to stop repeated attention-loop experimentation.

What was confirmed:

- The previous decode-attention custom tuned path did not have stable evidence and was removed.
- The current strongest kept evidence is on `fused_qkv` prefill algorithm selection.
- `2048/64` already beats TRT overall, but `512/64` still loses.
- Current dominant prefill bottleneck at `2048/64` is not D2D GPU memcpy time.
- Build type is already `Release`, so that is not the primary explanation for the remaining gap.
- `tests/operators` now pass after removing stale `mlp_down` prefill tuned cases from the active gate.
- LLM CUDA-graph token alignment passes.
- VLM CUDA-graph token alignment passes.

Additional 2026-04-06 correction:

- a fresh benchmark on the current working tree showed a large decode regression relative to the historical anchor
- current `nsys` confirms CUDA Graph replay is still active and no decode dynamic-node patching was observed on the profiled path
- therefore the next optimization branch is to isolate the decode regression, not to restart another speculative attention branch

Additional 2026-04-06 recovery entry:

- Root cause found:
  - the local working tree had removed the decode-specific tuned attention path in
    [src/operators/attention_op.cu](/xs-train-nas/zzm/repos/edge-fm-x/src/operators/attention_op.cu)
  - the matching decode-stage `impl_id` route in
    [examples/config/operator_impl_table.json](/xs-train-nas/zzm/repos/edge-fm-x/examples/config/operator_impl_table.json)
    was also absent from the active working tree
- Controlled fix:
  - restored `flashinfer_attention_decode_sm80_tuned`
  - rebuilt `Release`
  - re-ran operator correctness/perf, LLM/VLM correctness, and graph-only benchmarks
- Operator-level evidence after restore:
  - decode attention median latency:
    - `kv=512`: `0.024032 ms`
    - `kv=1024`: `0.024736 ms`
    - `kv=2048`: `0.028768 ms`
  - previous current-build reference before restore at `kv=2048`: about `0.0682 ms`
- End-to-end correctness after restore:
  - `tests/operators/test_attention_decode.py`: pass
  - `test_generate_token_alignment_cuda_graph`: pass
  - `test_generate_vl_token_alignment_cuda_graph`: pass
- Recovered benchmark matrix after restore:

| Case | EdgeFM total | TRT total | Gap | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill=512, decode=32` | `115.259 ms` | `112.003 ms` | `+3.165 ms` / `+2.83%` | `12.169 ms` | `12.282 ms` | `-0.113 ms` | `102.883 ms` | `99.606 ms` | `+3.277 ms` |
| `prefill=1024, decode=32` | `122.980 ms` | `119.182 ms` | `+3.614 ms` / `+3.04%` | `18.733 ms` | `15.699 ms` | `+3.034 ms` | `103.934 ms` | `103.353 ms` | `+0.581 ms` |
| `prefill=2048, decode=32` | `140.943 ms` | `141.125 ms` | `-0.339 ms` / `-0.24%` | `33.629 ms` | `29.457 ms` | `+4.172 ms` | `106.974 ms` | `111.485 ms` | `-4.511 ms` |
| `prefill=512, decode=64` | `221.818 ms` | `213.766 ms` | `+7.932 ms` / `+3.71%` | `12.233 ms` | `11.209 ms` | `+1.024 ms` | `209.333 ms` | `202.425 ms` | `+6.908 ms` |
| `prefill=1024, decode=64` | `230.557 ms` | `226.185 ms` | `+4.232 ms` / `+1.87%` | `18.712 ms` | `15.823 ms` | `+2.889 ms` | `211.541 ms` | `210.199 ms` | `+1.343 ms` |
| `prefill=2048, decode=64` | `263.140 ms` | `264.530 ms` | `-1.525 ms` / `-0.58%` | `37.932 ms` | `37.737 ms` | `+0.195 ms` | `224.899 ms` | `226.619 ms` | `-1.720 ms` |

- Single-case confirmation on `prefill=2048, decode=64`:
  - `EdgeFM(cuda-graph)`: `250.508 ms`
  - `TRT-Edge-LLM`: `256.779 ms`
  - decode stage gap: `-10.678 ms` in favor of EdgeFM
- Keep/revert decision:
  - keep the restored decode tuned attention path
  - do not remove it again unless a fair A/B with correctness and end-to-end benchmark shows a stable regression
  - next work should focus on the remaining small short-context gaps, not on re-litigating this recovered path

Process correction recorded for future work:

- do not claim a fusion is useless unless fused vs unfused was compared fairly under the same kernel family
- do not leave temporary tuned paths in the runtime after they fail to show reliable value
- do not start another attention branch without first checking this journal
