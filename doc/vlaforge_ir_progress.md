# VLAForge IR Foundation Progress

## Current status

- Branch: `codex/vlaforge-ir-foundation`
- Base: `master@83251d5`
- IR schema: `0.1`
- Last updated: 2026-07-23
- Goal state: active

## Milestone evidence

### G0: semantics and structure

Status: **provisional; deterministic fixtures pass, real-source audit pending**.

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

Remaining evidence:

- audit real SmolVLA and OpenVLA source state/clock/loop/action/reset behavior;
- prove the same core constructs cover their real Python execution boundaries.

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

Status: **not achieved**.

Current local evidence:

| Model | Checkpoint | Environment | Result |
| --- | --- | --- | --- |
| SmolVLA | `examples/smolvla/SmolVLA-Base/model.safetensors`, 906,712,520 bytes | system Python 3.13 has PyTorch 2.10 but no LeRobot | not run through real adapter |
| OpenVLA | no local checkpoint found | no pinned OpenVLA environment | not run |
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
46 passed in 0.10s
```

Covered suites:

- type and textual serialization;
- verifier positive/negative cases;
- dependency, liveness, and property-style physical-slot planning;
- multi-tick interpreter, stale input, reset, and failed transaction;
- epoch-keyed memoization and state physicalization;
- flow/autoregressive fixture programs;
- inspect/verify/run/diff CLI.

Skipped tests: 0.

## Current blockers

No core implementation blocker.

Model validation constraints:

- the current system Python is 3.13, while the existing SmolVLA deployment
  documentation uses a pinned Python 3.10 LeRobot environment;
- the real SmolVLA checkpoint is present, but LeRobot is not installed in the
  current interpreter;
- OpenVLA weights and its pinned legacy dependency environment are not present;
- the local RTX 3060 has 12 GiB memory, which is sufficient for SmolVLA but may
  require quantization/offload or a larger machine for OpenVLA-7B.

## Next informative work

1. Complete positive semantics tests for `if`, `while`, `async/await`, explicit
   abort, and reset.
2. Add an export-audit/real-model adapter contract that records checkpoint,
   dependency, device, and trace provenance.
3. Build an isolated SmolVLA environment without modifying the shared Python
   installation, execute the local checkpoint, and capture eager region traces.
4. Select and pin a reproducible OpenVLA inference environment, then determine
   whether the 12 GiB GPU can execute the official checkpoint without changing
   the semantic adapter.

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

