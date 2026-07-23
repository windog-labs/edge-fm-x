# Mapping from Python IR v0.1 to a Future MLIR Dialect

The Python IR is intentionally shaped so each construct has a direct MLIR
representation.

| Python construct | Proposed MLIR construct |
| --- | --- |
| `Module` | builtin `module` with `vla.schema` |
| `ClockDomain` | `vla.clock` symbol op |
| `InputStream` | `vla.input` symbol op |
| `StateSlot` | `vla.state` symbol op |
| `TensorRegion` | `vla.region` symbol op referencing exported artifact |
| `Policy` | `vla.policy` region op |
| `TensorType` | builtin ranked tensor type |
| `EpochType` | `!vla.epoch<@clock>` |
| `SnapshotType` | `!vla.snapshot<@state, T>` |
| `PendingType` | `!vla.pending<@state, T>` |
| `TransactionType` | `!vla.transaction` |
| `ActionType` | `!vla.action<T>` |
| `CommittedActionType` | `!vla.committed_action<T>` |
| `FutureType` | `!async.value<T>` or `!vla.future<T>` |
| `vla.for` | `scf.for` with explicit iter args |
| `vla.while` | `scf.while` |
| `vla.if` | `scf.if` |
| `vla.async` / `vla.await` | `async.execute` / `async.await` plus VLA effects |

`StateSlot` fields map to dialect attributes. State operations implement a
`VLAStateEffectInterface` distinguishing read, stage-write, commit, reset, and
publish effects. Tensor regions implement `MemoryEffectOpInterface` as pure.

The following semantics must not be weakened during an MLIR port:

1. a state read yields an immutable logical snapshot;
2. staged state cannot escape its transaction;
3. physical buffers are introduced only after liveness/retention analysis;
4. commit is an external action-effect barrier;
5. epoch and freshness constraints participate in legality checks;
6. model adapters cannot create new dialect operations.

