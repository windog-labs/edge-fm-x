# VLAForge IR Foundation Progress

## Current status

- Branch: `codex/vlaforge-ir-foundation`
- Base: `master@83251d5`
- Foundation commits: `792b205`, `3125025`
- Final implementation/evidence commit: `9b91b16`
- IR schema: `0.1`
- Last updated: 2026-07-23
- Goal state: complete; G0, G1, and G2 passed

## Milestone evidence

### G0: semantics and structure

Status: **passed**.

Implemented:

- immutable SSA module/policy/block/operation representation;
- versioned types for epoch, snapshot, pending state, transaction, action,
  committed action, event, and future;
- clock, input-stream, persistent-state, pure TensorRegion, structured
  control-flow, transaction, commit, and reset semantics;
- stable schema-versioned textual form and deterministic round-trip;
- source contract, semantic specification, textual syntax, trace schema, and
  MLIR mapping;
- generic flow-style and autoregressive-style adapter fixtures with no
  model-named core operations;
- a VLA-only public profile that rejects invented persistent state and keeps
  goal-required `while`/`async` constructs compatibility-only rather than
  treating them as paper or deployment abstractions.

Real-source conclusions:

- SmolVLA persists only its action queue across `select_action` calls; prefix KV
  and solver values remain local SSA in one action-chunk inference.
- OpenVLA's reference `predict_action` is stateless across control ticks and is
  represented by pure bounded generation plus detokenization and action commit.
- Neither adapter required a model-named core opcode.

### G1: verifier and interpreter

Status: **passed**.

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

Status: **passed**.

Current local evidence:

| Model | Checkpoint | Environment | Result |
| --- | --- | --- | --- |
| SmolVLA | `examples/smolvla/SmolVLA-Base/model.safetensors`, SHA256 `7cd549ac...aaca01eb` | isolated `horizon_quant`, LeRobot `8fff0fde`, PyTorch 2.6.0+cu126 | **passed**, 10 solver steps, eager-vs-IR action/solver error 0 |
| OpenVLA | official `openvla/openvla-7b`, revision `47a0ec7f...9ed83f`, 3 LFS shards verified by SHA256 | isolated Python 3.10, Transformers 4.40.1, Accelerate 0.29.3, tokenizers 0.19.1, timm 0.9.10, bitsandbytes 0.49.2 | **passed**, 7 tokens exact, action error 0 |
| π0/π0.5 | no local checkpoint found | LeRobot source `8fff0fde` audited | held-out source mapping passes with no new core op; real run not claimed |

Deterministic fixtures are explicitly labelled `deterministic_fixture` and do
not count as G2 evidence.

## Test evidence

Command:

```bash
cd /home/zhangzimo/Repos/private/edge-fm-x
PYTHONPATH="$PWD/vlaforge/python" \
  /home/zhangzimo/.venvs/vlaforge-ir/bin/python -m pytest -q vlaforge/tests
```

Result:

```text
59 passed, 2 skipped in 0.33s
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
environment variables are absent. They were both executed separately with
their real checkpoints and are not counted as skipped in the G2 audit:

```text
SmolVLA: 1 passed in 8.80s
OpenVLA: 1 passed, 5 dependency deprecation warnings in 17.87s
```

Real SmolVLA evidence:

- report: `artifacts/vlaforge_ir/smolvla/real_model_report.json`;
- trace: `artifacts/vlaforge_ir/smolvla/ir_trace.json`;
- checkpoint: 450M-parameter policy, output `(1, 50, 6)`;
- 10 eager-versus-IR solver errors: all `0.0`;
- final action max absolute error: `0.0`;
- action-queue indices 0–2 max absolute errors: all `0.0`;
- peak CUDA allocation: approximately 918 MiB on RTX 3060.

Real OpenVLA evidence:

- report: `artifacts/vlaforge_ir/openvla/real_model_report.json`;
- trace: `artifacts/vlaforge_ir/openvla/ir_trace.json`;
- three shards: 15,082,600,824 bytes total, each SHA256 equal to its Git-LFS
  object ID;
- deterministic RGB coordinate-grid fixture, prompt tokens `(1, 19)`, pixels
  `(1, 6, 224, 224)`;
- generated tokens:
  `31904, 31935, 31852, 31911, 31938, 31865, 31744`;
- token equality: exact;
- maximum eager-versus-IR action error: `0.0`;
- IR trace events: 10;
- peak CUDA allocation: approximately 4,509 MiB on RTX 3060.

Additional executed gates:

```text
SmolVLA fixture CLI: verify passed, 3 ticks, 55 trace events
OpenVLA fixture CLI: verify passed, 3 ticks, 39 events, trace diff passed
Wheel build: vlaforge-0.1.0.dev0-py3-none-any.whl, 57,426 bytes
compileall: passed
git diff --check: passed
```

## Current blockers

None for the scoped IR foundation.

## Next work outside this goal

The C++ AOT runtime, automatic `torch.export` frontend, backend plan emission,
and performance paper experiments remain separate follow-up milestones. They
must build on the frozen VLA profile rather than expanding it preemptively.

Held-out π0/π0.5 source audit:
`doc/reports/vlaforge_ir_pi0_source_audit.md`.

Cross-model coverage and explicit abstraction gaps:
`doc/reports/vlaforge_ir_coverage_and_gaps.md`.

Real-model reports:

- `doc/reports/vlaforge_ir_smolvla_real.md`;
- `doc/reports/vlaforge_ir_openvla_real.md`.

## Deviations from the development plan

- Python executable semantics precede MLIR/TableGen implementation. This is
  intentional: the goal requires Python closure first, and the mapping to MLIR
  is documented in `vlaforge/spec/mlir-mapping.md`.
- TensorRegion callables are reference Python functions. `torch.export` and AOT
  artifacts are deferred; the real-model gates establish stable VLA-level
  region boundaries without claiming frontend automation.
- The current textual v0.1 payload is canonical structural JSON behind an
  explicit `!vlaforge.ir 0.1` header. It serializes executable IR rather than a
  runtime manifest; an MLIR surface syntax remains a later compatibility step.
- After the explicit user request to avoid an over-general IR, the public
  design was narrowed to VLA business semantics: observation snapshots,
  source-proven cross-tick state, pure model regions, bounded generation, and
  validated action commit. No workflow scheduler, distributed event system, or
  general tensor dialect is part of v0.1.
- OpenVLA uses NF4 only as a loading strategy for the 12 GiB test GPU. Eager and
  IR execute the same real checkpoint and quantized model instance; no
  quantization-specific core operation was added.
- The pinned OpenVLA source appends token `29871` without extending a supplied
  attention mask. The deterministic prompt explicitly ends in the expected
  training-time empty token, so official `predict_action()` runs unchanged and
  no model-source patch is carried.
