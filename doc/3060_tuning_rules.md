# 3060 Tuning Rules

This document is the standing rule set for every RTX 3060 EdgeFM tuning pass.
It is intentionally narrow and should stay stable over time.

## Scope

- Tune LLM only on RTX 3060.
- Official performance claims are only `EdgeFM(cuda graph)` versus `TRT-Edge-LLM`.
- `graph-off`, `nsys`, and `ncu` are for hotspot attribution only.
- If `ncu` is blocked by `ERR_NVGPUCTRPERM`, record the permission blocker and do not infer hardware-counter conclusions.

## Implementation Rules

- Do not start with a from-scratch kernel. Prefer extending or tuning existing kernels under `3rdparty/` or `third_party/` first, especially the vendored CUTLASS, FlashInfer, cuTile, and TensorRT-LLM kernels, and only add a new kernel family after a review gate.
- For the 3060 stage specifically, do not continue open-ended CUTLASS retuning or start a self-written CUTLASS-style kernel route once `nsys` and operator evidence show a plateau. Stage 1 accepted the source-op path as close enough and removed the internal Qwen2.5 TensorRT engine bridge; further work should target source-visible operators, direct plugin-op assets, or external TRT-Edge-LLM comparison evidence.
- Do not do a large refactor unless the expected performance win is clear and the change is explicitly reviewed first.
- Change one variable per experiment whenever possible.
- Keep code clean. If a candidate fails correctness or does not produce a useful end-to-end gain, remove the temporary test code, debug code, and scripts in the same round.
- Keep runtime decisions small and reversible. Default-off or fallback behavior must remain available for every new fast path.
- Treat TensorRT Myelin/XMMA tactics as opaque unless they are reached through a source-visible API in this repo. Do not register a fake `myelin` or `xmma` operator impl id that cannot be built, tested, and profiled here.
- Do not quietly change the production precision mode just to chase a benchmark. Any BF16->FP16 or FP16->BF16 shift must be called out explicitly, revalidated with the generation gate, and either justified or rejected in the log.
- Do not wire Qwen2.5 prefill fast paths as model-local bridge members. Linear
  and MLP fast paths should be selected through layers, operators, and
  `operator_impl_table` records so future source-op/plugin-op candidates share
  the same boundary.
- Preserve correctness on every accepted change. Use the operator and generation tests as the gate before claiming a tuning result.

## Documentation Rules

- Keep `doc/3060_tuning_plan.md` and `doc/3060_tuning_log.md` current.
- Keep `doc/` user-facing. Temporary debug notes, ad hoc correctness scratchpads, and one-off tuning drafts should be deleted once their conclusions are absorbed into the maintained plan/log/rules or the root `README.md`.
- Delete stale conclusions from the live status sections instead of letting them accumulate.
- Every conclusion must record:
  - date
  - command
  - environment
  - raw artifact path
  - the exact case or shape
- If a result is later rejected, move it to the rejected/obsolete section and remove it from current status.

## Communication Rules

- Every conclusive result must be sent to Feishu through `cc-connect`.
- Include the case, EdgeFM number, TRT number, conclusion, raw artifact, and the relevant doc link.
- Send a test message first when the connection or config changes.

## Benchmark Rules

- The only official matrix is the full 18-case LLM matrix:
  - `Qwen2.5-{0.5B,1.5B,3B}`
  - `prefill={512,1024,2048}`
  - `decode={32,64}`
- `EdgeFM(cuda graph)` must remain the headline result.
- `graph-off` results never replace the official CUDA graph comparison.
- Keep raw benchmark artifacts under `.tmp_codex/bench/` with timestamped names.

## Kernel Strategy

- Prefer existing kernel families first.
- For `fused_gate_up` / SwiGLU work, extend the current TensorRT-LLM / CUTLASS-based path instead of inventing a new implementation style.
- On 3060, TensorRT engine bridging is no longer the preferred implementation strategy for Qwen2.5. Keep external TRT-Edge-LLM as the reference, and only promote source-op or direct plugin-op paths that pass correctness and end-to-end gates.
- Only promote a new kernel path if correctness passes and the end-to-end benchmark shows a real gain.
- If a kernel path is only useful as a diagnostic helper, keep it under `scripts/tune/` or `.tmp_codex/` and do not treat it as the source of truth.
