# OpenPI Prefix Stage-Chain Host Integration, 2026-10-23

## Current Result

Host-only, model-neutral composition is implemented and verified. The actual
authoritative input is [publication002](../../artifacts/orin/20261023-15/published-002/publication.json)
and its three `partition-000/001/002.json` ledgers, not planner preview13.
The existing `split_region_invocation` API creates the repeated call chain.
Actual engine pruning later uses `project_region_inputs` only with a checked
unused-graph-input proof. No shared production source was edited.

The derived [semantic Module](../../artifacts/orin/20261023-14/actual-ledger-replay-002/composed/semantic-module.json)
and [physical Plan](../../artifacts/orin/20261023-14/actual-ledger-replay-002/composed/physical-plan.json)
have ten Regions: six existing actual contracts and four explicitly unbound
prefix stages. Module verification, compilation and Plan verification pass.

| Region | Physical ID | Actual Inputs | Actual Outputs | Binding |
| --- | --- | --- | --- | --- |
| checked state F64->F32 | 0 | unchanged | unchanged | original native DSO |
| checked tokens I64->I32 | 1 | unchanged | unchanged | original native DSO |
| finite | 2 | unchanged | unchanged | original actual TRT |
| native output | 3 | unchanged | F64 action + BOOL accepted | original native DSO |
| solver step | 4 | unchanged | unchanged | actual step006 TRT |
| time | 5 | 0 | unchanged | original actual TRT |
| prefix `__part00` | 6 | 8 | 10 | unbound |
| prefix `__part01` | 7 | 18 | 51 | unbound |
| prefix `__part02` | 8 | 69 | 26 | unbound |
| prefix `__part03` | 9 | 95 | 37 | unbound |

Every later stage retains prior inputs and explicit frontier ports as the real
partition API dictates. A local `source: input` route is not a public input
index. No index is guessed from a shape. The actual stage2 frontier differs
from the planner's 46 cut seeds because inherited graph inputs are not new
produced cut values; no neural work is omitted.

## Preserved Semantics

Public schema digest remains
`8751fa709441f09883da726d2cc2d6b52f7a5894bf5876ee36b6803e2dff9575`.
All original public images/masks/tokens/state/noise and both action outputs
are unchanged. F64 state is still read before the ten-iteration solver loop,
checked-cast once before that loop, and separately supplied in its original
F64 form to native finishing. The source private time noise projection remains
zero-input; public noise remains available to the solver.

Native output remains the original `[native F64, accepted BOOL]` signature,
then source validation, output construction and atomic commit. Full normalized
action remains F32 `[1,50,32]`; native action remains F64 `[1,50,14]`.
All stages are pure, uncached, fresh per invocation, with no cross-chunk pointer
cache or cached visual/text features. Native execution remains outside capture.
All native IDs `0/1/3` are explicitly checked; a shift rejects reuse and requires
a DSO rebuild. The actual stagechain arena is 198,892,800 bytes, a compiler
allocation result, not target memory or performance evidence.

## Source And Lineage

The starting point is immutable root08/v4's real six-bound Module/Plan, including
step006 and actual time/finite TRT engines plus three actual native DSOs.
The graph anchor is complete v5 prefix SHA
`4d0eede245e56282630801993cc824b7aa0b239e2f2ba7733dae26a5b892c847`.
The full selected precision profile SHA is
`b0f417d20d7fd5845af61fa6796e10e6b8f62d81c3e0c470f7c59ec48c9dca2b`.

[Replay receipt](../../artifacts/orin/20261023-14/actual-ledger-replay-002/replay-receipt.json)
records independent use of the existing partition API with the exact actual
ledger selections/aliases. Only its filename checker is replaced by read-only
external-range metadata checks, since the filename checker writes beside source
weights outside this task's scope. Reviewed shape inference, Constant hoisting
and extraction are unchanged. Shape inference is strict, does not seed internal
annotations, and uses only the reviewed BF16Round/1 and LayerNorm/1 contracts.

The composer checks named route origins, all interface types, exact reachable
source NodeProto bytes once, exact literal-Constant-to-same-SSA initializer
hoisting and provably dead DCE. Passthrough alias Identity bridges are checked
explicitly. The first actual split retains all 9,589 reachable source compute
nodes after literal hoisting/dead DCE; subsequent splits preserve that work.
Raw metadata graphs reference the original source payload directory and are
not presented as standalone deployable graphs.

[Published-part proof](../../artifacts/orin/20261023-14/published-audit-001/publication-proof.json)
checks actual compacted graphs against those exact raw outputs. It independently
rehashes every referenced source/destination range using the existing compactor's
bounded streaming helpers, verifies non-address protobuf identity, and reruns
the existing precision-profile generator. No tensor value is decoded.
Stage0/1 are the actual reused `published-001` directories; stage2/3 are
`published-002`. All 810 source dense nodes occur once. Stage weight unions are
2,802,293,696; 2,762,950,656; 2,776,727,552; 2,915,147,776 bytes: total
11,257,119,680 bytes, exactly the source union, zero cross-stage duplication.
All source 4 F32 / 806 F16 dense selections and 165 F32 LayerNorm selections
are preserved, including the declared FP16 dense-result rounding.
This is not proof of LayerNorm/BF16 numerical equivalence or complete quality.

## Actual Stage000 Milestone

Parent's actual native Orin [stage000 build](../../artifacts/orin/20261023-15/stage000-build-001/build.json)
says `built-not-validated`, inner exit 0, same actual stage000 graph and precision
hashes. Its fetched engine is 2,461,615,407 bytes; independently checked local SHA:
`86ffeda4d78ffe6ad355d7195d8dc536f37ac7f235b51d4ec4694942c548a800`.
The layer report names the actual 8 graph inputs and 10 graph outputs.
No context-memory/ABI inspection is supplied yet, so it remains unbound in this
Plan. Stage001 builds serially under the parent; no later build success is
inferred here. No neural enqueue, full model run or target acceptance occurred.

## Tools And Tests

New code is only under `artifacts/orin/20261023-14/`:

- `stagechain.py`: data-driven typed composition and native-ID checks.
- `ledger_envelope.py`, `replay_owner_ledgers.py`: exact actual-ledger replay.
- `stage_binding.py`, `attach_engine.py`: fail-closed real engine attachment and
  source-only Session generation through existing backend/codegen APIs.
- `audit_publication.py`: actual compacted-graph/precision/source-range proof.
- `freeze_base.py`, `prepare_binding.py`: explicit pinned requests.
- `test_stagechain.py`, `verify_delivery.py`, `freeze_delivery.py`: checks and sealing.
- [Interface](../../artifacts/orin/20261023-14/INTERFACE.md): full receipt/declaration
  schema, actual Region IDs, IO requirements and attachment CLI.

[Tests002](../../artifacts/orin/20261023-14/tests-002.xml): 52 passed, including
20 focused tests and 32 existing Region split/interface tests. Structural unit
fixtures never execute providers and are not model evidence. Real-source tests
verify the actual Module/Plan, original six contracts, unchanged ten-step/public/
native commit path, actual step006's 42-tensor binding and three real v3
inspections. Negative tests cover same-shaped route swaps, changed nodes/weights/
types, invalid literal hoists, false inherited cuts, used-input projection,
native ID drift, invalid scalar layout provenance, inner compile failure,
precision drift, and forged compaction metadata. Missing stage engines reject
Session generation before creating its output directory.

## Binding And Next Step

The current fresh request is
[published-audit-001/binding-request.json](../../artifacts/orin/20261023-14/published-audit-001/binding-request.json).
It carries all four actual graph/precision/compaction proposals and four null
runtime bindings. [Binding assessment](../../artifacts/orin/20261023-14/published-binding-audit-001/binding-assessment.json)
is explicitly `incomplete-fail-closed`, 6 actual bindings, 4 missing inspections/
complete receipt sets. Existing source-compatible native providers are retained;
no synthetic provider, dummy input, pending artifact hash or engine is invented.

After the parent supplies actual v3 inspections and complete build/engine/layer/
audit declarations, `attach_engine.py` validates each against the actual stage
proposal and writes a fresh request. Only actually unused graph inputs may be
projected, in inspected engine order; every output is retained. TRT-owned context
memory, scalar-layout derivation provenance, effects, precision, capability,
workspace and actual PID/start/argv are checked. `stage_binding.py generate`
then requires every real binding before emitting native C++ Session source.
That all-bound generation path is not yet exercised because stage receipts are
incomplete; no Session source/binary is claimed to have been generated.

Fixed quality primary is complete normalized action, MSE `<1e-5` and cosine
`>0.9999`, all outputs finite. Native errors remain supplemental; byte-exact
regression remains separate. Overall goal stays active at 2/5, SmolVLA+RDT.

Immediately runnable delivery verification:

```sh
cd /home/zhangzimo/.codex/worktrees/f801/edge-fm-x
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vlaforge/python:artifacts/orin/20261023-14 python3 -B artifacts/orin/20261023-14/verify_delivery.py --manifest artifacts/orin/20261023-14/manifest.json
```

This task performed no SSH, GPU access, target build/run, inference, service
stop, installation or shared source edit. Old root08/v4 and root11 negative
evidence remain unchanged. Published requests/replays are preserved; later
engine receipts must create fresh outputs, not overwrite these files.
