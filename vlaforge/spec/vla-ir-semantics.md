# VLAForge IR v0.1 Semantics

## Status

This document defines the normative semantics of the executable Python IR.
The schema version is `0.1`. The implementation lives under
`python/vlaforge/ir`, and the deterministic semantics are implemented by
`python/vlaforge/interpreter`.

## Program model

A module is a tuple:

$$
P = (C, I, S, R, M)
$$

- $C$: logical clock domains;
- $I$: timestamped input streams;
- $S$: versioned persistent-state declarations;
- $R$: pure tensor regions;
- $M$: policies triggered by a clock.

A runtime configuration is:

$$
\Gamma = (\Sigma, E, T, O)
$$

- $\Sigma$: committed state versions;
- $E$: current clock epochs and input samples;
- $T$: open transactions;
- $O$: externally published actions.

## Logical epochs

An epoch is `(clock, sequence, timestamp_ns, episode)`. `EpochExpr` can select
the current, next, previous, input, solver, or action-chunk epoch. State reads
and staged writes use logical epochs; physical addresses are not part of the
semantic IR.

An input sample is legal when:

1. its episode equals the policy tick episode;
2. its timestamp does not lie in the future;
3. its age satisfies the operation and stream freshness contracts.

## Persistent state

A `StateSlot` is a declaration, not mutable storage. A committed logical value
is identified by:

```text
(session, episode, state, epoch, version)
```

`vla.state.read` yields an immutable `snapshot<state, payload>`.
`vla.snapshot.value` obtains the tensor/scalar payload used by a pure region.
`vla.state.stage_write` creates a `pending<state, payload>` owned by exactly one
transaction. Pending values are invisible to other policy ticks.

The reference store deep-copies payloads at read, stage, and commit boundaries.
This makes accidental Python mutation observably different from an IR state
transition.

## Transactions and actions

A policy tick opens a transaction with `vla.txn.begin`. Writes are accumulated
in a staging map. A successful path:

1. samples inputs and reads committed snapshots;
2. executes pure tensor regions and structured control flow;
3. stages next state versions;
4. creates an uncommitted action;
5. validates the action;
6. commits state and action exactly once;
7. publishes only the committed action.

`vla.txn.commit` is the state/action visibility barrier. `vla.action.publish`
accepts only `committed_action<T>`. An abort discards staged state and publishes
nothing. A physical robot cannot observe a pending action.

## Pure tensor regions

`TensorRegion` is an opaque, typed, pure function:

```text
(tensor/scalar inputs, explicit state payloads, explicit RNG state)
    -> (tensor/scalar outputs, explicit next RNG/state payloads)
```

Mutable buffers, hidden RNG, external I/O, and stateful cache mutation are
illegal region effects. They must be lifted into explicit IR state/effects.

## Structured control flow

- `vla.if` selects one region and yields its results.
- `vla.for` carries one explicit iteration value in v0.1.
- `vla.while` carries explicit values and has a mandatory maximum-iteration
  bound in the reference interpreter.
- `vla.async` executes deterministically in the reference interpreter but
  preserves explicit future/event and read/write effect declarations.
- `vla.await` is the only operation that unwraps a future.

The scheduled IR may later execute asynchronous tasks concurrently. It must
refine the deterministic result and satisfy the same effects and commit order.

## Logical-to-physical state

The semantic IR exposes an unbounded sequence of logical versions. The
physical-slot analysis computes a bounded capacity:

```text
required = max(retention,
               1 + max_in_flight + consumer_lag + fallback_snapshots)
slot(version) = version mod required
```

A requested capacity below `required` is rejected. The physicalization
transformation records the proven capacity but does not change reference
interpreter behavior.

## Verification invariants

The verifier rejects:

- undefined or multiply-defined SSA values;
- unknown clocks, inputs, states, regions, or operations;
- state accesses using the wrong version clock;
- state retention incompatible with freshness;
- state result/payload type mismatches;
- two staged writes to the same state in one transaction;
- pending values escaping through yield/return;
- in-place overwrite of authoritative state;
- non-pure tensor regions;
- conflicting un-awaited asynchronous state accesses;
- commit without a dominating validator;
- commit before required futures are awaited;
- action publication before commit;
- successful paths with zero or multiple commits.

Runtime checks additionally reject stale/future inputs, old-episode state,
failed validation, missing state versions, and reuse of closed transactions.

