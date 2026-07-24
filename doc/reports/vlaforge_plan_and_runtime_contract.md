# VLAForge Scheduled Plan and Runtime Contract

## Status

This document records the executable contract between VLA Semantic IR,
internal scheduling, physical memory planning, generated code, and the
lightweight C++ runtime.

- Scheduled Plan lowering/verifier/reference executor: implemented.
- Physical state and static arena: pending Milestone D.
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

| Program | Tasks | Blocks | Buffers | States | Artifacts | Plan digest |
|---|---:|---:|---:|---:|---:|---|
| SmolVLA fixture | 31 | 4 | 34 | 3 | 8 | `7f57ed54c0616841bc1195e294ebebaedde4d65c5263039cb063703c958ead49` |
| OpenVLA fixture | 14 | 2 | 16 | 0 | 4 | `00293b18cf01b354867c6e2971bae595673eef90aea19d1b2c3f860282c471ca` |
| Real SmolVLA program | 28 | 4 | 30 | 2 | 7 | `8d4c00ad2006157650620bf93a552b3b0a6ca039a448d6d3b1172bba57f88b76` |
| Real OpenVLA program | 15 | 2 | 18 | 0 | 4 | `73e46eb3e1532143b78d6e64425bd7464368a515ac6ac4c30b5f4b54fdc04882` |

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
