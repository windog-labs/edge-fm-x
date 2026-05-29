# EdgeFM Development Plan Overview

This document re-organizes the current round of EdgeFM development planning. The goal is not to lay out every technical detail at once, but rather to first clearly organize the background, boundaries, feature overview, owner breakdown, and detailed tasks, so that 3 people can work in parallel.

## 0. Owner A Current Status Correction (2026-05-14)

Below is the Owner A status correction based on the current code implementation. It takes priority over the older descriptions in the historical schedule later in this document:

- Prefix KV is already complete and should no longer be listed as an unfinished item. The current implementation is a contiguous per-request/per-layer KV slot: `KVManager` parses `prefix_token_ids` and offsets the write pointer, `Scheduler` validates the request prefix, `StandardEngine::warmup()` pre-fills the prefix KV, and the real prefill skips the prefix and only writes the suffix.
- Completed Owner A closed loops include: `sampling.max_new_tokens`, device-side token finalize/stop semantics, `last_generate_metrics()`, the basic capability of compact vocab runtime remap/restore, the non-identity compact vocab test, the greedy sampler direct-argmax optimization, the decode breakdown profiler, the default-disabled `lm_head_top1` experimental path, and the DeepGEMM probe-only decision gate.
- Owner A items that are unfinished or still in probe/deferred state: the full benchmark matrix report, the prerequisites for landing the default-disabled DeepGEMM candidate, the FP8/W8A8 artifact/scale contract, the engine-side contiguous INT8 KV, and the multi-model/large-scale acceptance for compact vocab.
- `lm_head_top1` currently has a default-disabled implementation: when `runtime.lm_head_top1.enabled=true` and greedy decode is in use, it can bypass full logits + sampler; unless it is later proven that the CUDA graph target slice yields an end-to-end improvement of >= 1% and that token alignment fully passes, the full-logits default path will continue to be kept.

## 1. Background && Current State

### 1.1 Current Project Structure and Positioning

The current codebase already has a fairly clear CUDA runtime layering:

- `src/engine/`: request lifecycle, scheduler, KV cache, CUDA graph, generate loop.
- `src/models/`: model runtime; currently Qwen2.5 LLM/VLM primarily goes through `src/models/qwen2_5/`.
- `src/layers/`: layer contract, shape/dtype/device validation, weight binding, operator query construction.
- `src/operators/`: operator registry, operator implementation, `operator_impl_table` routing, and the underlying CUDA kernels.
- `src/backends/`: Horizon whole-graph artifact and runtime backend.

The overall design philosophy of the project is not to build an over-generalized large framework, but rather to do small-step, verifiable, and easily revertible incremental development around the existing runtime.

### 1.2 Current Capability Status

| Capability | Current Status | Main Gaps |
| --- | --- | --- |
| Single-request performance optimization | Already has standard generate path, decode CUDA graph, device-side finalize/stop semantics, greedy direct argmax, and decode breakdown metrics | Full benchmark matrix report; the dedicated `m=1` decode optimization has not yet entered the default path |
| W4A16 quantization | Already has INT4 groupwise kernel and the `LinearLayer::forward_int4_groupwise()` test entry point | Not a formal `linear` operator impl, cannot be routed through `operator_impl_table`, end-to-end flow is incomplete |
| W8A8 quantization | The `DType::Int8` base type exists | No activation quant, scale contract, or formal decode-first int8 linear implementation |
| Vocabulary pruning | Already has compact vocab artifact contract, runtime input/output remap, response restore, non-identity test, the TRT-Edge-LLM-style `vocab_map` packaging/validator tool, and the 0.5B real checkpoint packaging smoke | Still lacks multi-model/large-scale acceptance |
| Speculative decoding | `src/engine/experimental/speculative/EagleEngine` only has a prototype header file; the facade refuses to enable it | No draft model runtime, verify loop, accept/reject/commit, or temporary KV workspace |
| KV cache compression | `KVManager` is currently a contiguous per-request/per-layer KV buffer | No compression format, scale metadata, dequant-on-load attention, or contiguous-state compressed buffer integration |
| Prefix KV | Prefix warmup/reuse under contiguous KV slots is already complete | No paged attention, semantic prefix cache lookup, or INT8 KV scale contract |

### 1.3 Current Key Code Facts

- `src/engine/engine_factory.cpp` currently throws directly when `speculative.enabled=true`.
- `src/engine/experimental/speculative/eagle_engine.cpp` is currently an empty file.
- `src/engine/tasks/token_generation/kv_manager.cpp` currently allocates a contiguous KV buffer per request/layer, with no paged attention semantics.
- `src/layers/linear.cu` already supports recognizing `.qweight + .scaling_factors` INT4 groupwise weights.
- `src/operators/linear_impl.cu` currently has the formally registered linear impls mainly as `cublasLt`, `cutlass`, etc.
- `src/layers/sampler.cu` currently has a main path that mainly consumes `temperature` and `seed`.

### 1.4 Framework Design and Development Principles That Must Be Followed

The following principles need to be hard-wired into this round of development; subsequent feature implementations must not bypass any of them:

1. Simplicity first. Prioritize small-step increments that hug the existing structure; do not introduce new framework layers ahead of time just to "look more general".
2. Do not introduce paged attention. The KV cache continues to keep contiguous per-request/per-layer buffers; do not introduce non-contiguous index tables, block tables, or extra address translation layers.
3. Do not build a new general IR or decode framework. The CUDA path continues to keep the current direct runtime structure, and the first version of speculative only adds a minimal branch near the existing `StandardEngine::generate()`.
4. Keep shared boundaries stable. The responsibility chain `engine -> model -> layer -> operator` is not disrupted; the layer continues to be responsible for the contract and operator query, and the operator continues to be selected via the registry and `operator_impl_table`.
5. Quantization only loads already-quantized models. Do not develop offline quantization conversion tools, and do not push complex format conversions into the hot path.
6. Default behavior must not regress. All new capabilities must be explicitly gated; once disabled, the original FP16/BF16 generate behavior and correctness should not change.
7. Benchmark first. Larger design changes must have a unified benchmark methodology, otherwise subsequent gain judgments will be distorted.
8. B/C should be developed as independently as possible. Owner B's and Owner C's tasks should by default be confined to local directories and local interfaces; they should not proactively initiate shared framework refactoring.
9. Shared interface changes must first be aligned with Owner A. Especially public boundaries such as `StandardEngine::generate()`, `Scheduler::create_context()`, `Context`, `Response`, `KVManager`, the `operator_impl_table` schema, and `LinearShapeSignature`.
10. Do lossless or near-lossless capabilities in the first version, and only then do approximate optimizations that change the output distribution.

### 1.5 Current Overall Assessment

- Owner A is the most familiar with the project as a whole and should take on the roles of runtime owner and interface owner.
- Owner B and Owner C are relatively less familiar, so when splitting tasks they should be given local closed loops first, rather than touching a large number of existing shared interfaces.
- The most worthwhile closed loops to prioritize in the first round are: single-request standard path optimization, W4A16 already-quantized model end-to-end, and speculative greedy correctness.
- W8A8, KV cache compression, and the actual integration of DeepGEMM should all be built on the premise that the previous three lines already have a baseline and interface boundaries.

## 2. Development Requirements && Feature Items && Overall Development Plan Overview

### 2.1 Development Requirements for This Round

The development requirements explicitly to be covered in this round include:

1. Implementation of quantization operators, prioritizing `w4a16` support, then advancing `w8a8`.
2. Vocabulary pruning.
3. Speculative decoding.
4. KV cache compression algorithm.
5. Performance optimization and design updates for the single-request scenario.
6. Adding DeepGEMM candidate path evaluation on the Owner A side.

### 2.2 Feature Item Overview

| Feature Item | Main Owner | First-Version Scope | Notes |
| --- | --- | --- | --- |
| Single-request performance optimization and runtime design update | Owner A | benchmark, metrics, `max_new_tokens`, device-side stop flag, token finalize, greedy fast path, `m=1` decode optimization | First priority |
| Vocabulary pruning | Owner A | compact vocab artifact contract, runtime input/output remap, special token retention validation | Assigned to Owner A, no longer floating separately |
| Speculative decoding | Owner B | greedy only, same tokenizer/same vocab, draft-only, target verify, accept/reject/commit | First version only does minimal speculative greedy |
| W4A16 / W8A8 quantization operators | Owner C | W4A16 operator-ization and end-to-end; W8A8 decode-first | Only loads already-quantized models |
| KV cache compression | Owner A + Owner C | Contiguous INT8 KV first; A is responsible for the engine side, C for the attention/kernel side | Advanced later in the second round |
| DeepGEMM candidate path | Owner A leads, Owner C assists | First benchmark and routing design, then decide whether to do lightweight integration | Not used as a default path |

### 2.3 Overall Development Plan Overview

Overall, it is recommended to advance in 4 phases:

| Phase | Core Goal | Main Deliverables |
| --- | --- | --- |
| Phase 0 | Unify boundaries and benchmark methodology | baseline benchmark, metrics, shared interface boundaries, confirmation of the first-version scope of each feature |
| Phase 1 | Each owner first builds their own minimal closed loop | A gets the standard path MVP running; B gets draft-only running; C gets the W4A16 operator MVP running |
| Phase 2 | Connect to the real generate path | A completes the greedy hot path and vocabulary pruning design; B completes verify/accept; C completes W4A16 end-to-end and starts W8A8 |
| Phase 3 | Second features and stabilization | A finalizes the DeepGEMM/`lm_head_top1` decision and verifies the completed Prefix KV; B does metrics/fallback; C completes W8A8, and A+C advance contiguous INT8 KV as appropriate |

### 2.4 Main Outcomes and Trade-offs of the First Round

For the first round, it is recommended to prioritize ensuring the following three closed loops:

1. Single-request standard path optimization, with the default FP16/BF16 correctness not regressing.
2. W4A16 already-quantized models run end-to-end and can be benchmarked stably.
3. The speculative greedy correctness closed loop, with the ability to fall back to standard decode on failure.

If resources are tight in the first round, the following items can be deferred to the second round or treated as stretch goals:

- Full W8A8 completion.
- Truly landing the contiguous INT8 KV cache.
- Truly integrating DeepGEMM into the default runtime.
- More aggressive prefix reuse and approximate compression.

## 3. The Responsibilities, General Work Arrangements, and Collaboration Methods of the Three Owners

### 3.1 Splitting Principles

This round of splitting does not divide features evenly, but rather splits by "who is most suitable to touch the shared boundaries":

- Owner A is the runtime owner and also the shared interface owner.
- Owner B's and Owner C's tasks should be as independent as possible, by default not changing a large number of existing interfaces.
- If B/C find an interface insufficient, they should first raise the minimal requirement, then converge the interface change together with Owner A; it is not recommended that they each directly modify the large framework.

### 3.2 Master Table of the Three Owners' Responsibilities

| Owner | Scope of Responsibility | Default Write Scope | Scope That Should Not Be Proactively Expanded |
| --- | --- | --- | --- |
| Owner A | Single-request performance, vocabulary pruning, runtime design, shared interface boundaries, DeepGEMM evaluation, KV compression engine side | `src/engine/`, part of `src/models/`, benchmark/metrics/config, compact vocab runtime | Does not write quantization kernels, does not write the main body of the speculative algorithm |
| Owner B | The minimal speculative greedy closed loop | `src/engine/experimental/speculative/`, plus the minimal glue code near `StandardEngine::generate()` | Does not refactor the generate framework, does not change the KV contiguous design, does not make wide-ranging changes to the engine API |
| Owner C | W4A16/W8A8 quantization operators, already-quantized model loading compatibility, related correctness/benchmark | `src/layers/linear*`, `src/operators/*linear*`, quant loading/tests | Does not change the scheduler/generate loop, does not develop offline quantization conversion tools |

### 3.3 Shared Interface Collaboration Rules

The following interfaces or schemas, if they are to be changed, must first be aligned with Owner A:

- The main loop semantics of `StandardEngine::generate()`.
- `Scheduler::create_context()` and the request/context lifecycle.
- The tensor naming, tensor slots, and lifecycle constraints within `Context`.
- The write semantics of `Response` and the external token semantics.
- The layout, capacity, and contiguous write pointer rules of `KVManager`.
- The `operator_impl_table` schema.
- The `LinearShapeSignature` schema.
- The semantics of shared config such as `sampling.max_new_tokens`, `speculative.*`, and `kvcache.*`.

Specific requirements for B/C:

- If Owner B needs to change token finalize, response writing, or the target main KV commit flow, align with Owner A first.
- If Owner C needs to change the shared quant contract, shape signature, or operator route schema, align with Owner A first.
- If it is only a local implementation detail, such as the internal logic of a speculative helper, the W4A16 kernel route, or local test completion, it should be done independently within each owner's scope as much as possible.

### 3.4 General Work Arrangement

It is recommended to schedule the first round over 6-8 weeks:

| Period | Owner A | Owner B | Owner C |
| --- | --- | --- | --- |
| Week 1 | baseline benchmark, metrics, `max_new_tokens` semantics, compact vocab contract, DeepGEMM evaluation scope | speculative config and state transition design | W4A16 contract, already-quantized weight naming, INT4 reference |
| Week 2 | lower-synchronization stop check, token finalize plan | draft model load, draft KV context | `w4a16_groupwise` impl skeleton |
| Week 3 | standard path MVP, compact vocab remap POC | draft-only K token generation | W4A16 correctness |
| Week 4 | greedy hot path benchmark, `lm_head_top1`/DeepGEMM decision, compact vocab runtime design convergence | target verify temporary KV workspace | W4A16 loading compat and operator table |
| Week 5 | single-request standard path performance convergence | accept/reject/commit, exact match | W4A16 Qwen2.5 end-to-end |
| Week 6 | prefix reuse or first version of compact vocab landed | metrics/fallback | W8A8 decode prototype |
| Week 7-8 | advance DeepGEMM and engine-side contiguous INT8 KV as results dictate; Prefix KV verification and documentation finalization | CUDA graph verify evaluation | W8A8 correctness/report or INT8 KV attention kernel POC |

## 4. Detailed Task Breakdown

### 4.1 Owner A: Single-Request Performance, Vocabulary Pruning, and Shared Interfaces

#### 4.1.1 Role Positioning

Owner A is the runtime owner and interface owner for this round, responsible for:

- Standard single-request path performance optimization.
- The overall landing of vocabulary pruning.
- The shared interface boundaries related to the other two lines.
- The evaluation of the DeepGEMM candidate path and the decision on whether to integrate it.
- If the KV cache compression phase is entered, responsible for the contiguous buffer integration on the engine/KVManager side.

#### 4.1.2 Core Tasks

1. Baseline and benchmark

- Fix the benchmark matrix: model, prompt len, decode len, CUDA graph on/off, greedy/sampling, FP16/BF16/W4A16/W8A8.
- Unify the output of `prefill_ms`, `decode_ms`, `decode_step_avg_ms`, `tokens/s`.
- Add key NVTX/CUDA event boundaries, at least covering lm_head, sampler, stop check, and decode graph replay.

2. generate semantics cleanup

- Clarify that `kvcache.requests[].max_tokens` still represents KV capacity.
- Introduce or formally enable `sampling.max_new_tokens` to represent the upper limit of a single generation.
- The actual generation upper limit is controlled by `min(max_new_tokens, slot.max_tokens - prompt_len + 1)`.

3. Single-request decode hot path optimization

- Change the stop check from a per-token host copy + stream sync to a device-side stop flag.
- The token finalize kernel simultaneously writes the decode token, the response token, and the stop flag.
- Evaluate a lightweight argmax path under the greedy path.
- The `lm_head_top1` fast path already has a default-disabled experimental entry point; whether to make it the default depends only on benchmark gains and token alignment.
- Drive the dedicated `m=1` decode GEMV optimization, but the quantization kernel here is provided by Owner C in collaboration.

4. Vocabulary pruning

- Define the compact vocab artifact contract:
  - `original_vocab_size`
  - `compact_vocab_size`
  - `old_to_new`
  - `new_to_old`
  - The TRT-Edge-LLM-style `vocab_map.safetensors`, where `vocab_map == new_to_old`
  - `special_token_ids`
  - pruned embedding / `lm_head`
  - the updated `config.json`
- Complete the runtime input remap and output remap.
- Ensure that the external `Response.token_ids()` still returns the original tokenizer ids.
- Pruned tokens raise an error by default; no silent arbitrary mapping is done.
- The focus of the first round is landing the runtime and contract, without requiring very complex tooling first.

5. DeepGEMM candidate path

- Only do benchmark, shape filtering, routing strategy, and default switch strategy.
- Prioritize evaluating the prefill bucket and FP8/W8A8-related dense linear.
- The first version does not use DeepGEMM as the main `m=1` decode solution.
- Unsupported shape/hardware/build artifact must directly fall back.

6. Prefix KV and KV compression engine side

- Prefix KV has already completed config-driven prefix warmup/reuse under the contiguous KV layout: it matches exactly by the `request_id` slot and `prefix_token_ids`, without doing paged attention or non-contiguous layout.
- Prefix KV is not a semantic/approximate prefix cache, nor is it responsible for the INT8 KV scale buffer.
- If contiguous INT8 KV is advanced in the future, Owner A is responsible for integrating the contiguous int8 K/V buffer and scale buffer on the `prepare_kvcache_tensors()` side.

#### 4.1.3 Points That Require Collaboration with Other Owners

- Provide Owner B with stable boundaries for token finalize, response writing, and generated token advancement.
- If `lm_head_top1`, `m=1` GEMV, or DeepGEMM ultimately involve an operator entry point, determine the minimal interface together with Owner C.
- If the `Response`, `Context`, `KVManager`, or `operator_impl_table` schema needs to be changed, Owner A should lead the change.

#### 4.1.4 Phased Plan

| Phase | Goal | Deliverables |
| --- | --- | --- |
| Phase 0 | Boundary confirmation | benchmark matrix, metrics plan, `max_new_tokens` semantics, compact vocab contract, DeepGEMM scope |
| Phase 1 | Standard path MVP | stop flag, token finalize, standard path baseline, compact vocab remap POC |
| Phase 2 | Connect to the real path | greedy hot path optimization, compact vocab runtime design convergence, shared interface boundaries needed by B |
| Phase 3 | Second features | `lm_head_top1`/DeepGEMM decision, Prefix KV verification finalization, first version of compact vocab landed, engine-side contiguous INT8 KV collaboration |

#### 4.1.5 Acceptance Criteria

- Standard greedy output is unchanged.
- Decode step latency does not regress, and key hotspots have explainable benchmarks.
- Compact vocab does not change the external token id semantics.
- If DeepGEMM is integrated, it must be independently switchable and automatically fall back.

### 4.2 Owner B: Speculative Decoding

#### 4.2.1 Role Positioning

Owner B is responsible for the minimal closed loop of speculative greedy. The focus here is correctness and independence, not building a general decode framework.

#### 4.2.2 Core Tasks

1. Configuration and entry point

- Keep `speculative.enabled`.
- The first version only supports `algorithm="greedy"`.
- Add `num_draft_tokens`.
- When `enabled=false`, the standard path is completely unaffected.

2. Draft model integration

- The draft and target must use the same tokenizer.
- The draft and target must have the same vocab size.
- The draft has an independent `Model`, an independent contiguous `KVManager`, and an independent scheduler/context.

3. Draft-only MVP

- Starting from the target's first token, the draft can continuously generate K tokens.
- All draft KV is written into its own contiguous KV buffer.
- A draft-only failure does not affect the standard path.

4. Target verify and accept/reject/commit

- Target verify uses a temporary contiguous K/V workspace.
- It does not directly pollute the main target KV.
- Under greedy, compare draft tokens and target tokens position by position.
- Accepted tokens are written into the response, and the corresponding K/V is committed to the main KV's current write pointer.
- After a rejection, discard the temporary workspace and write the target token.

5. Second-phase capabilities

- acceptance rate, accepted/rejected tokens, draft latency, target verify latency.
- low-acceptance fallback.
- If needed, evaluate CUDA graph verify with a fixed `num_draft_tokens`.

#### 4.2.3 Constraints on Owner B

- Do not refactor into a general decode framework.
- Do not support cross-tokenizer draft models.
- Do not change the main KV contiguous buffer design.
- Keep the default write scope confined as much as possible to `src/engine/experimental/speculative/` and the minimal glue code near `StandardEngine::generate()`.

#### 4.2.4 Changes That Must First Be Aligned with Owner A

- Token finalize / response writing semantics.
- The interaction between the generated token count and the stop check.
- The target main KV commit flow.
- The public interfaces of `Context`, `Response`, `KVManager`, and `Scheduler`.

Owner B is better suited to raising minimal interface requirements and letting Owner A lead the stable changes to the shared boundaries together, rather than doing wide-ranging refactoring of the engine on their own.

#### 4.2.5 Phased Plan

| Phase | Goal | Deliverables |
| --- | --- | --- |
| Phase 0 | Scope confirmation | speculative config, state transition description, minimal draft-only test sample |
| Phase 1 | Independent MVP | draft model load, draft KV context, draft-only K token |
| Phase 2 | True closed loop | target verify, accept/reject/commit, exact match correctness |
| Phase 3 | Stabilization | metrics, fallback, CUDA graph verify evaluation |

#### 4.2.6 Acceptance Criteria

- `speculative.enabled=true` is no longer directly rejected by the facade.
- Greedy speculative output is exactly consistent with standard greedy.
- acceptance/fallback metrics are visible.
- Draft/target KV do not pollute each other.

### 4.3 Owner C: Quantization Operators

#### 4.3.1 Role Positioning

Owner C is responsible for quantization operators, already-quantized checkpoint loading compatibility, and the subsequent W8A8 and optional INT8 KV attention kernel. Owner C is not responsible for offline quantization conversion tools, nor for the main flow of the generate loop.

#### 4.3.2 Core Tasks

1. W4A16 operator-ization

- Add a new `w4a16_groupwise` linear impl.
- Register it into the `LinearOpRegistry`.
- Route it through `operator_impl_table`.
- decode `m=1` goes through GEMV, prefill `m>1` goes through GEMM.

2. Already-quantized model loading compatibility

- Support `<prefix>.qweight`.
- Support `<prefix>.scales` and `<prefix>.scaling_factors`.
- Optionally support `<prefix>.qzeros` and `<prefix>.zeros`.
- Read the necessary metadata, such as `bits`, `group_size`, and `quant_method`.
- When metadata is missing, infer from shape/dtype; if inference fails, raise an error.

3. correctness and benchmark

- Add a Python dequant reference.
- Cover `m=1`, `m=2/4`, and `m=64/128/512`.
- Cover `fused_qkv`, `attention_output`, `fused_gate_up`, `mlp_down`, and optionally `lm_head`.
- Output layer benchmarks, error reports, and token alignment results.

4. W8A8 decode-first

- Assume the checkpoint already contains int8 weights and scale metadata.
- The runtime does activation quant.
- The first version only supports `m=1` decode linear.
- Output back to FP16/BF16 or FP32.
- `lm_head` is evaluated separately and is not enabled by default.

5. Second-phase collaboration items

- Provide a quantization path or operator entry point for Owner A's `m=1` decode GEMV and `lm_head_top1`.
- If Owner A decides to integrate DeepGEMM, provide a minimal `linear` operator binding or build/link support.
- If the KV cache compression phase is entered, provide the INT8 KV attention dequant-on-load kernel.

#### 4.3.3 Constraints on Owner C

- Do not develop offline quantization conversion tools.
- Do not be responsible for the speculative decode loop.
- Do not proactively change the main structure of the scheduler, generate loop, or KVManager.
- Keep the default write scope confined as much as possible to `src/layers/linear*`, `src/operators/*linear*`, model loading, and tests.

#### 4.3.4 Changes That Must First Be Aligned with Owner A

- `LinearShapeSignature` schema changes.
- `operator_impl_table` schema changes.
- Quantization contract changes if they would affect shared config or engine routing logic.
- Any requirement that needs to modify `Context`, `Response`, or `KVManager`.

Owner C is better suited to converging the work within the layer/operator/model-loading scope, leaving the engine-side shared interfaces to be led by Owner A.

#### 4.3.5 Phased Plan

| Phase | Goal | Deliverables |
| --- | --- | --- |
| Phase 0 | Contract and test preparation | W4A16 weight contract, already-quantized naming review, INT4 reference test |
| Phase 1 | W4A16 MVP | `w4a16_groupwise` impl, operator table route, loading compat, layer correctness |
| Phase 2 | W4A16 end-to-end + W8A8 kickoff | Qwen2.5 greedy alignment, latency report, W8A8 decode prototype |
| Phase 3 | Second features | W8A8 correctness/report, DeepGEMM binding collaboration, INT8 KV attention kernel advanced as appropriate |

#### 4.3.6 Acceptance Criteria

- W4A16 can be hit through `operator_impl_table` and run end-to-end.
- Unsupported quant formats raise clear errors.
- W8A8 decode-first has at least layer-level correctness and an error report.
- After disabling the quantization path, the original FP16/BF16 behavior does not regress.

### 4.4 Shared Interface Modifications and Parallel Development Considerations

This part is where things are most likely to go wrong during three-person parallel development and needs a separate reminder:

1. Shared interface modifications are led by Owner A by default.

- Owner B and Owner C can raise requirements, but should not each directly make wide-ranging changes to the engine/shared schema.
- If a change is indeed needed, first converge it into a minimal interface change, then arrange for the owner to collaborate on landing it.

2. PR boundaries should converge by owner as much as possible.

- Owner B's PRs should change only speculative and the minimal glue code as much as possible.
- Owner C's PRs should change only layer/operator/quant loading/test as much as possible.
- Owner A's PRs are responsible for the shared runtime, benchmark, vocabulary pruning, public semantics, and configuration.

3. Combinations not recommended to start in parallel.

- Speculative and KV cache compression should not change the main KV commit flow at the same time.
- W8A8 and `lm_head_top1` should not change the same section of the lm_head path in the same PR.
- DeepGEMM and W8A8 should not simultaneously change the `linear` quant dtype, operator route, and benchmark methodology in the same PR.
- Vocabulary pruning and speculative should not change the response token id semantics at the same time.

4. Things not to do in the first version.

- Do not introduce paged attention.
- Do not introduce a new general decode framework.
- Do not introduce a new general backend framework just for DeepGEMM.
- Do not develop offline quantization conversion tools in the quantization path.
- Do not make approximate KV compression that changes the output distribution the main line of the first version.
