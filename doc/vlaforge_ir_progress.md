# VLAForge IR Foundation Progress

## Current status

- Branch: `codex/vlaforge-ir-foundation`
- Base: `master@83251d5`
- IR schema: `0.1`
- Last updated: 2026-07-23
- Goal state: active

## Milestone evidence

### G0: semantics and structure

Status: **provisional; core semantics and real-source audit pass, OpenVLA
checkpoint closure pending**.

Implemented:

- immutable SSA module/policy/block/operation representation;
- versioned types for epoch, snapshot, pending state, transaction, action,
  committed action, event, and future;
- clock, input-stream, persistent-state, pure TensorRegion, structured
  control-flow, transaction, commit, reset, async/await semantics;
- stable schema-versioned textual form and deterministic round-trip;
- source contract, semantic specification, textual syntax, trace schema, and
  MLIR mapping;
- generic flow-style and autoregressive-style adapter fixtures with no
  model-named core operations.
- a VLA-only public profile that rejects invented persistent state and defers
  general `while`/`async` scheduling abstractions.

Real-source conclusions:

- SmolVLA persists only its action queue across `select_action` calls; prefix KV
  and solver values remain local SSA in one action-chunk inference.
- OpenVLA's reference `predict_action` is stateless across control ticks and is
  represented by pure bounded generation plus detokenization and action commit.
- Neither adapter required a model-named core opcode.

### G1: verifier and interpreter

Status: **implementation and offline tests pass; broader control-flow and soak
coverage remains**.

Implemented verifier coverage:

- read-before-definition and duplicate SSA values;
- wrong state version clock;
- retention/freshness mismatch;
- double staged write;
- pending-state escape;
- required future not awaited;
- validator dominance and commit condition type;
- action publication before commit;
- zero/double commit;
- authoritative in-place overwrite;
- conflicting async state effects;
- hidden TensorRegion RNG/mutation effects;
- unsafe physical-slot capacity.

Implemented runtime coverage:

- deterministic multi-tick execution;
- immutable state snapshots and transaction staging;
- commit/abort/action publication;
- stale/future input rejection;
- episode reset isolation;
- state/solver/action trace recording and exact/numeric comparison.

### G2: real Python model closure

Status: **partially achieved: SmolVLA passed; OpenVLA pending**.

Current local evidence:

| Model | Checkpoint | Environment | Result |
| --- | --- | --- | --- |
| SmolVLA | `examples/smolvla/SmolVLA-Base/model.safetensors`, SHA256 `7cd549ac...aaca01eb` | isolated `horizon_quant`, LeRobot `8fff0fde`, PyTorch 2.6.0+cu126 | **passed**, 10 solver steps, eager-vs-IR action/solver error 0 |
| OpenVLA | official `openvla/openvla-7b`, revision `47a0ec7f...9ed83f` (download in progress) | isolated Python 3.10, Transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, bitsandbytes 0.49.2 | adapter ready; real gate pending |
| π0/π0.5 | no local checkpoint found | not prepared | held-out, not started |

Deterministic fixtures are explicitly labelled `deterministic_fixture` and do
not count as G2 evidence.

## Test evidence

Command:

```bash
cd /home/zhangzimo/Repos/private/edge-fm-x/vlaforge
python3 -m pytest -q
```

Result:

```text
50 passed in 0.13s
```

Covered suites:

- type and textual serialization;
- verifier positive/negative cases;
- dependency, liveness, and property-style physical-slot planning;
- multi-tick interpreter, stale input, explicit reset/abort, and failed
  transaction;
- queue refill/reuse `if` semantics and exact bounded `for` iteration counts;
- epoch-keyed memoization and state physicalization;
- flow/autoregressive fixture programs;
- inspect/verify/run/diff CLI.

The default suite has two explicit opt-in `real_model` tests when model
environment variables are absent. The SmolVLA test was also executed
separately with the real local checkpoint; it must not be counted as skipped in
the G2 audit. OpenVLA remains pending.

Real SmolVLA evidence:

- report: `artifacts/vlaforge_ir/smolvla/real_model_report.json`;
- trace: `artifacts/vlaforge_ir/smolvla/ir_trace.json`;
- checkpoint: 450M-parameter policy, output `(1, 50, 6)`;
- 10 eager-versus-IR solver errors: all `0.0`;
- final action max absolute error: `0.0`;
- action-queue indices 0–2 max absolute errors: all `0.0`;
- peak CUDA allocation: approximately 918 MiB on RTX 3060.

## Current blockers

No core implementation blocker.

Model validation constraints:

- the official OpenVLA checkpoint is 14.05 GiB and is still downloading through
  a resumable Git-LFS path after direct Hugging Face downloads hit TLS EOF;
- the local RTX 3060 has 12 GiB memory, so the gate uses isolated
  bitsandbytes-NF4 loading without changing the semantic IR.

## Next informative work

1. Finish the pinned OpenVLA checkpoint transfer and execute eager-versus-IR
   token/action comparison in the isolated 4-bit environment.
2. Diagnose memory or dependency failures without adding quantization or
   framework-specific operations to the IR.
3. Run the full offline suite and both opt-in real-model gates, then complete
   the G0-G2 audit.

## Deviations from the development plan

- Python executable semantics precede MLIR/TableGen implementation. This is
  intentional: the goal requires Python closure first, and the mapping to MLIR
  is documented in `vlaforge/spec/mlir-mapping.md`.
- TensorRegion callables are reference Python functions. `torch.export` and AOT
  artifacts are deferred until the real-model source audit establishes stable
  region boundaries.
- The current textual v0.1 payload is canonical structural JSON behind an
  explicit `!vlaforge.ir 0.1` header. It serializes executable IR rather than a
  runtime manifest; an MLIR surface syntax remains a later compatibility step.
