# Mapping Invocation IR v0.2 to a Future MLIR Dialect

The Python IR is intentionally shaped so it can migrate to MLIR without
changing its passive invocation semantics.

| Python construct | Proposed MLIR construct |
| --- | --- |
| `Module` | builtin `module` with `vla.schema = "0.2"` |
| `InputPort` | `vla.input` symbol op with stable ID and static contract |
| `OutputPort` | `vla.output` symbol op with stable ID and group |
| `StateSlot` | `vla.state` symbol op with retention/reset attributes |
| `TensorRegion` | `vla.region` symbol op referencing an immutable artifact |
| `Invocation` | `vla.invocation` region op |
| `TensorType` | builtin ranked tensor type plus layout/device attributes |
| `ScalarType` | builtin scalar or a small VLA POD type |
| `InputRevisionType` | `!vla.input_revision` |
| `SnapshotType` | `!vla.snapshot<@state, T>` |
| `PendingType` | `!vla.pending_state<@state, T>` |
| `TransactionType` | `!vla.transaction` |
| `PendingOutputType` | `!vla.pending_output<@port, T>` |
| `PendingOutputGroupType` | `!vla.pending_output_group<@group, ...>` |
| `CommittedOutputGroupType` | `!vla.committed_output_group<@group, ...>` |
| `vla.if` | `scf.if` with typed results |
| `vla.for` | `scf.for` with explicit loop-carried iter args |

Input/output symbols carry the contract attributes that produce the generic C
ABI and model-specific typed wrapper. Runtime `InputStamp` metadata is not a
sensor-time type: only its revision participates in exact cache identity;
timestamp is optional freshness metadata.

State operations implement a VLA state-effect interface distinguishing
read-latest, stage-write, atomic commit, abort, and episode reset.
TensorRegions implement `MemoryEffectOpInterface` as pure at the Semantic IR
boundary. Backend-private workspace is allowed but hidden persistent mutation
or I/O is not.

The following semantics must not be weakened in an MLIR port:

1. external binding is push-only and borrowed until `Run()` returns;
2. a state read yields an immutable logical snapshot with a committed version;
3. staged state and pending outputs cannot escape their transaction;
4. state versions advance only on successful commit;
5. one validated output group and all staged state become visible atomically;
6. exact cache identity includes all transitive revisions and snapshot
   versions;
7. bounded loops use explicit loop-carried SSA;
8. the four memory classes cannot silently alias or change lifetime;
9. adapters cannot introduce unverified dialect operations.

There are intentionally no `ClockDomain`, `EpochType`, `Policy.clock`,
`RunTick`, `ActionType`, `action.publish`, `while`, `async`, or `await`
constructs in the v0.2 profile. Sensor scheduling and command publication
remain host responsibilities.

