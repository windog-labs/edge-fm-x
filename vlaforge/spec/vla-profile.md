# VLAForge Invocation Profile v0.2

VLAForge is a whole-program compiler for one externally invoked, stateful VLA
model call. It is deliberately not a sensor, middleware, or real-time
scheduling framework.

## Required VLA patterns

Every core construct must serve at least one of these deployment patterns:

1. **Stamped external input** — statically typed Tensor/Scalar ports with a
   stable ID and optional `InputRevision`. Multi-camera history, masks,
   bounded agent/map arrays, language tokens, route commands, and ego state
   all use this boundary. The host performs acquisition, synchronization, and
   history assembly.
2. **Authoritative persistent state** — values whose loss changes later Runs:
   chunk queues/cursors, recurrent hidden state, previous action, or explicit
   RNG. A successful transaction allocates the next logical version.
3. **Pure model Regions** — VLM prefix/decode, vision/BEV encoder, action
   expert, diffusion/flow step, scorer, detokenizer, and typed external C++ or
   CUDA preprocessing.
4. **Bounded generation** — finite autoregressive, diffusion, or flow loops
   with explicit loop-carried SSA, plus structured branches for cross-artifact
   fast/slow or expert selection.
5. **Generic transactional results** — one or more named outputs become
   visible atomically with staged state. Outputs may be an action, trajectory,
   candidates and scores, predictions, maps, detections, or auxiliary tokens.
6. **Derived reuse** — exact cache entries are recomputable and keyed by every
   transitive input revision, state snapshot version, episode, model, artifact,
   and Region identity. Guarded approximate reuse is a separate contract.

## Core operations

- `vla.input.read`
- `vla.txn.begin`
- `vla.state.read_latest`
- `vla.snapshot.value`
- `vla.invoke`
- `vla.if`
- `vla.for`
- `vla.yield`
- `vla.state.stage_write`
- `vla.validate`
- `vla.output.create`
- `vla.output.group`
- `vla.txn.commit` / `vla.txn.abort`
- `vla.return`

The passive runtime exposes `Session::Run()` and `ResetEpisode()`. There is no
IR operation for a timer, tick, deadline, sleep, frame drop, topic, or action
publication.

## Adapter templates, not core semantics

Reusable adapters may implement:

- `StatelessTrajectory`
- `ChunkedAction`
- `AutoregressiveTrajectory`
- `DiffusionPlanner`
- `HybridVLMPlanner`
- `MultiTaskDriving`

For example, an action queue and cursor belong to `ChunkedAction`; they are not
assumptions of the core IR. Driving adapters normally return an entire
trajectory or a group of candidates, scores, and auxiliary predictions in one
Run.

## State admission

An adapter declares a persistent `StateSlot` only when the upstream model or
policy wrapper retains that value across externally visible Runs.

- SmolVLA/ACT-style chunk queue and cursor are authoritative state.
- Transformer decode KV used only inside one bounded generation is
  loop-carried SSA.
- A VLM prefix or diffusion condition retained only for acceleration is a
  derived cache.
- OpenVLA-style one-shot action generation has no invented cross-Run queue.

The four memory classes remain distinct: external I/O, per-Run arena,
authoritative persistent state, and derived cache.

## Bounded dynamic inputs

Deployment tensors have a compile-time maximum shape/profile. Runtime
cardinality is represented by a typed `valid_count` and/or mask. The verifier
checks the bound; the runtime never allocates unbounded dynamic storage.

## Extension hierarchy

Use the narrowest extension that expresses a new model:

1. add an Adapter/template composition;
2. add a typed Region or backend/artifact implementation;
3. add optional fixed-shape input/output ports, an output validator, cache
   guard, or artifact variant;
4. only then add a new control/state opcode.

A new opcode requires a versioned schema, type/effect verifier, reference
semantics, Plan lowering, runtime/codegen implementation, serialization, and
positive/negative tests. Model-named opcodes and arbitrary unverified
extension opcodes are forbidden.
