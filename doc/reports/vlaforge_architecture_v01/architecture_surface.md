# VLAForge architecture and build-surface audit

Status: **passed**.

The production surface contains only passive, caller-driven model invocation semantics. Mentions in negative tests are reported separately and are not runtime implementations.

## Results

| Invariant | Result |
|---|---|
| No tick/clock/deadline/period/jitter | pass |
| No middleware, sensor sync, publish, or internal sleep | pass |
| No core action queue/cursor | pass |
| No Python runtime dependency in production surface | pass |
| Semantic IR opcode set equals frozen v0.2 set | pass |
| VLAForge build has no `.cu`, `.cuh`, or `.ptx` source | pass |
| No source/subdirectory edge escapes `vlaforge/` | pass |
| Repository root builds VLAForge and no retired source tree | pass |

## Build graph

- Audited CMake files: 3
- Declared C/C++ sources: 20
- CUDA source files: 0
- Retired engine/operator source files: 0
- Contract: C++ AOTI backend links CUDA::cudart and executes external compiled artifacts; VLAForge declares no CUDA kernel source

## Old to new migration

| Old surface | New surface | Status |
|---|---|---|
| ClockDomain + period/deadline/jitter + RunTick | caller-owned scheduling + passive Session::Run | removed |
| EpochExpr.current/next | StateStore allocated commit version | removed |
| action.publish / host I/O | transactional named outputs + ReadOutput | removed |
| core ActionQueue | ChunkedAction Adapter authoritative state | adapter-only |
| sensor synchronization / middleware | caller-prepared TensorView/ScalarValue inputs | outside-framework |
| retired custom CUDA engine/operators | verified external AOTI/RegionExecutable artifacts | removed |

## Claim boundary

This audit proves source/build isolation. Runtime correctness, no-libpython linkage, and CUDA execution remain separate clean build and generated-Session gates.
