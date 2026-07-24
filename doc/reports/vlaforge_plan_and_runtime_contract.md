# VLAForge Scheduled Plan and Runtime Contract

## Status

This document records the executable contract between VLA Semantic IR,
internal scheduling, physical memory planning, generated code, and the
lightweight C++ runtime.

- Scheduled Plan lowering/verifier/reference executor: implemented.
- Physical state and static arena: implemented.
- C++ runtime and generated session: pending Milestones E and F.

## Representation boundary

VLA Semantic IR remains the only public, normative semantic representation.
The Scheduled Plan is an internal compiler data structure with schema
`vlaforge.scheduled_plan/1`; the version exists for deterministic build
artifacts and tests, not as a commitment to a user-facing DSL or parser.

The Plan intentionally uses one compact immutable `Task` record instead of a
parallel class hierarchy for every semantic operation. `TaskKind` exposes only
the distinctions required by the verifier, memory planner, and code generator:

- input;
- region;
- bounded loop;
- branch;
- state;
- validation;
- commit;
- publish;
- model-independent control.

This keeps the Plan VLA-focused and prevents it from becoming a general
workflow or tensor compiler IR.

## Core records

`PlanModule`
: The source Semantic IR digest, policies, tasks, blocks, logical buffers,
  state bindings, artifact bindings, and optional physical arena.

`PlanPolicy`
: Integer policy ID, clock, typed external arguments, root block, and optional
  deadline guard.

`PlanBlock`
: Integer block ID, typed block arguments, ordered task IDs, and source path.
  Structured `for`, `while`, and `if` remain nested blocks.

`Task`
: Integer task ID, generic opcode, typed input/output buffer IDs, dependency
  IDs, optional artifact binding, bounded guards, nested block IDs, and source
  semantic operation/location.

`LogicalBuffer`
: Integer buffer ID, IR type, producer, semantic storage class, and source.
  Storage classes distinguish external values, SSA, loop carry, state
  snapshot/pending values, pending actions, and committed actions.

`StateBinding`
: Stable state ID plus retention, in-flight, consumer-lag, fallback-snapshot,
  and eventual physical ring-capacity proof.

`ArtifactBinding`
: Stable artifact ID, source region, backend variant, and eventual artifact
  path. Gate C accepts an explicit `uncompiled/default` binding; later gates
  replace it with a verified artifact contract.

## Deterministic lowering

Lowering performs a preorder walk of each verified Semantic IR policy:

1. Assign policy, block, task, buffer, state, and artifact IDs from zero in
   source order.
2. Convert every SSA result into one typed logical buffer.
3. Derive data dependencies from buffer producers.
4. Add a conservative same-block effect dependency, preserving source order
   until a proven scheduling pass relaxes it.
5. Preserve structured control as nested Plan blocks.
6. Attach freshness guards from input/state contracts.
7. Bind every `vla.invoke` to exactly one artifact variant.
8. Preserve the source opcode and location for trace comparison.

Canonical JSON uses sorted keys and contains no timing or process-dependent
data. Repeated lowering and serialization therefore produce identical bytes
and SHA-256 digests.

Current deterministic examples:

| Program | Tasks | Blocks | Buffers | States | Artifacts | Logical Plan digest |
|---|---:|---:|---:|---:|---:|---|
| SmolVLA fixture | 31 | 4 | 34 | 3 | 8 | `baf4fa58ec1e481064a11bd487799e5e31c9edc9711cb5cca570d84325658e99` |
| OpenVLA fixture | 14 | 2 | 16 | 0 | 4 | `d8c340c2f8e22fa2b5e7bdf6aabb9bc8736294b80dc90a54807dcc0d055b418a` |
| Real SmolVLA program | 28 | 4 | 30 | 2 | 7 | `1517a7b52b457118abd1622232e1d239d39dc0efca68ab0774cc2dfd7959b37e` |
| Real OpenVLA program | 15 | 2 | 18 | 0 | 4 | `42c902291f324125b56d5985b4709303912fffc0c257604c520e5c6603685ab0` |

The real OpenVLA program now contains a fixed six-iteration decode loop after
prefill, yielding seven action tokens in total. KV is an invocation-local loop
carry and the state table remains empty.

## Plan verification

The verifier rejects:

- non-contiguous or duplicate deterministic IDs;
- tasks absent from blocks or reused by multiple blocks;
- unknown dependencies, buffers, blocks, states, or artifacts;
- dependency cycles;
- read-before-produce without a transitive dependency path;
- output/producer mismatches;
- region/artifact mismatches;
- loss of a declared freshness guard;
- zero, negative, or empty loop bounds;
- commit without a validation-produced condition;
- publish of anything other than a committed action;
- physical arena overflow or simultaneous lifetime/memory overlap.

The negative suite mutates valid plans to cover dependency cycles,
read-before-produce, missing artifacts, publish-before-commit, missing
freshness guards, and invalid loop bounds. Physical-capacity and alias tests
are added with Milestone D.

## Reference execution and trace contract

`PlanExecutor` executes Plan tasks directly; it does not reconstruct or call
the Semantic IR interpreter. It reuses only the normative runtime value types
and logical `StateStore`, which prevents state/action behavior from drifting
while the Plan representation is validated.

For both offline model fixtures and three consecutive ticks, tests compare:

- returned committed actions;
- retained logical state versions;
- every normalized trace event;
- region invocation order and loop counts;
- validation, commit, and publish order.

Semantic Interpreter and Plan Executor traces are byte-equivalent after stable
normalization.

## Physical state rings

Each logical state version maps to a bounded slot:

```text
slot(logical_version) = logical_version mod slot_capacity
```

The minimum accepted capacity is:

```text
max(retention,
    1 + max_in_flight + consumer_lag + fallback_snapshots)
```

Supplying a smaller capacity raises `UnsafeStateCapacityError` before an arena
is emitted. Each `StateBinding` records the proof inputs, capacity, slot
stride, alignment, device, and state-arena offset while preserving the logical
state ID and version used by traces.

Current default real SmolVLA layout:

| State | Required slots | Slot stride | Offset | Total |
|---|---:|---:|---:|---:|
| `action_queue` | 5 | 1,216 B | 0 | 6,080 B |
| `queue_cursor` | 5 | 4 B | 6,080 | 20 B |

The state arena is 6,100 bytes. Real OpenVLA has no state arena because the
source policy has no persistent state.

## Logical-buffer liveness and Static Arena

The initial allocator is deliberately conservative:

1. derive every internal logical buffer's lifetime from its producer and all
   transitive structured-task consumers;
2. size static tensors and fixed runtime descriptors;
3. require an explicit override for a dynamic internal tensor;
4. treat dynamic process inputs and block arguments as explicit external/alias
   bindings;
5. allocate one non-aliasing physical range per internal logical buffer in
   stable buffer-ID order;
6. reserve artifact workspace as a distinct `REGION_WORKSPACE` buffer using
   its declared device, size, and alignment;
7. verify bounds, alignment, lifetime coverage, one-to-one logical mapping,
   and absence of simultaneous memory/lifetime overlap.

The first physicalization pass does not reuse addresses. This provides a
simple correctness baseline; lifetime-based cross-cycle reuse is intentionally
deferred to the measured whole-program optimization pass.

| Program | Static arena | Alignment | Allocations | Physical Plan digest |
|---|---:|---:|---:|---|
| SmolVLA fixture | 1,088 B | 64 B | 30 | `374302b140459769db6dcd2b6aa9993aa3726f99dcfa3e5e75ec7090e4fd2615` |
| OpenVLA fixture | 320 B | 64 B | 11 | `7f2bd7e6057c345b5e0d60f9919a1d2283d054327038d166fa72751885c2d191` |
| Real SmolVLA program | 17,024 B | 64 B | 25 | `6d9c634c430e7abffa456b0c501caf2672b706027ad095cc88d6f083ea8b5be4` |
| Real OpenVLA program | 448 B | 64 B | 12 | `632fe0d1829ebed928e8cebf7be9afb0d8f2c3b78111197eb9cc9f1fcf4cce05` |

Opaque Semantic IR values occupy fixed descriptor handles only. The concrete
flattened tensor ABI and its storage/workspace remain part of the bound region
artifact contract; opaque handles do not authorize hidden cross-tick state.

`emit_memory_constants()` produces deterministic C++17 `StateRingDesc` and
`BufferDesc` constexpr tables. The generated header is compiled with
`-Wall -Wextra -Wpedantic -Werror` in the Python test suite.

The C++ `StaticArena` allocates once at session construction, resolves
prevalidated offset/size/alignment slices without allocation, rejects
out-of-bounds or misaligned requests, and supports move-only ownership.
Release and ASan+UBSan CTest exercise the arena.

## C++ runtime boundary

The future generated session will consume only integer IDs and constexpr
tables derived from this verified Plan. The runtime will not parse Semantic IR
or dispatch on model names. Its stable responsibilities are:

- bind typed external input/output views;
- own the static arena and bounded state rings;
- call `RegionExecutable` artifacts by integer ID;
- execute bounded task/control descriptors;
- enforce transaction and action-commit ordering;
- optionally emit the same normalized logical trace IDs.

No JSON parsing, Python callback, dynamic model string lookup, or general
allocator operation is permitted in the tick hot path.
