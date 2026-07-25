# VLAForge Invocation IR v0.2 Semantics

## Scope

VLAForge compiles one externally invoked model call. It does not acquire or
synchronize sensors, maintain a physical rate, schedule deadlines, drop
frames, or publish vehicle/robot commands. The host pushes typed values and
calls `Session::Run()`.

A module is:

$$
P = (I, S, R, O, M)
$$

- $I$: static external input ports;
- $S$: authoritative persistent-state slots;
- $R$: pure typed TensorRegions;
- $O$: named output ports and groups;
- $M$: passive invocations.

There is no clock, tick, deadline, or middleware object in this tuple.

## External inputs

An input port declares a stable integer ID, Tensor or Scalar payload, required
or optional/default status, static shape/dtype/layout/device/alignment,
ownership, and optional bounded-profile metadata.

`vla.input.read` returns `(value, InputRevision)`. A host binding may include:

```text
InputStamp(revision?: u64, timestamp_ns?: u64)
```

`revision` identifies data for exact-cache invalidation. `timestamp_ns` is
freshness metadata only; VLAForge never synchronizes it. Missing revision is
replaced by a fresh Session-local revision on every bind/run, so unsafe
cross-Run reuse is impossible. Optional defaults have stable revision zero.

External CPU/CUDA buffers are borrowed until `Run()` returns. The Session does
not free them. A contract mismatch is either an explicit copy/preprocessing
Region or an error; silent dtype/layout/device conversion is illegal.

## Authoritative state

A `StateSlot` contains values that affect later Runs and cannot be discarded:
queue/cursor, previous action, recurrent hidden state, or explicit RNG.

The state transition is:

```text
read_latest -> immutable snapshot -> stage_write -> transaction commit
```

Committed values are identified by `(session, episode, state, version)`.
`StateStore` allocates the next monotonically increasing version during a
successful commit. Abort, execution failure, and validation failure do not
advance any state version. `ResetEpisode(new_episode)` resets or explicitly
carries each slot according to its declaration; it is not a clock transition.

## Derived cache

Derived cache can be invalidated and recomputed: VLM prefix/KV, condition
embedding, or diffusion features. It is not authoritative state.

An exact memoization key is:

$$
K_f = (\text{model},\text{artifact},\text{region},\text{episode},
       \text{InputRevision}^*,\text{StateSnapshot.version}^*)
$$

Model/artifact identity is compile-time constant in a generated Session.
Every transitive external input and authoritative snapshot must appear in the
key. Missing provenance rejects the exact-cache candidate.

Autoregressive decode KV and denoise samples are loop-carried SSA. Exact
condition reuse may use memoization/LICM. Approximate diffusion reuse must use
an explicit guarded-reuse contract and never masquerade as exact cache.

## TensorRegions and control flow

`TensorRegion` is a typed deterministic function over Tensor/Scalar values.
Invocation-local workspace may be internal; external I/O, hidden persistent
mutation, and hidden RNG are illegal.

The core control set is intentionally small:

- `vla.invoke`;
- structured `vla.if`;
- statically bounded `vla.for` with loop-carried SSA;
- `vla.yield` and `vla.return`.

Model-specific routing remains inside a captured TensorRegion when possible.
Cross-artifact selection uses structured branch/variant metadata. A new
extension op is legal only with schema/type verification, reference semantics,
Plan lowering, serialization version, runtime/codegen, and tests.

## Transactional outputs

Outputs are generic named values, not a hard-coded action:

```text
output.create -> output.group -> validate -> txn.commit -> return/read_output
```

`vla.txn.commit` atomically makes all staged state versions and one validated
output group visible. On failed validation, the transaction aborts, state
versions do not advance, and the previous committed output remains the latest
readable result. There is no `vla.action.publish` operation. The host decides
how to consume or publish trajectories, candidates, scores, detections, VQA
tokens, or robot actions.

## Four memory classes

| Class | Lifetime | May be discarded? |
|---|---|---|
| external input/output | host contract / committed output | host-owned rules |
| per-Run temporary/static arena | one invocation | yes after Run |
| authoritative persistent state | across Runs/episodes by policy | no |
| derived cache | across Runs while key is valid | yes, recompute |

Temporary buffers may alias only under Plan liveness and size/alignment
compatibility. Derived cache and persistent state never alias ordinary
temporaries. Persistent state uses a proven ring capacity and logical version
independent of physical slot.

## Verification invariants

The verifier rejects:

- undefined or multiply defined SSA values;
- unknown input/state/region/output identifiers;
- input/output schema or stable-ID mismatch;
- unbounded control flow;
- hidden Region effects;
- state write without an active transaction;
- multiple writes to one state in a transaction;
- pending state/output escaping a block;
- commit without a dominating validator;
- output/state visibility before commit;
- exact cache with incomplete revision/state identity;
- unverified extension opcode;
- legacy clock/tick/publish operations.

Semantic Interpreter, Scheduled Plan, and generated C++ are required to agree
on committed outputs, state versions, and normalized runtime trace.
