# VLAForge C++ AOT Progress

## Goal

Continue the frozen VLA Semantic IR v0.1 through an explicit PyTorch frontend,
internal scheduled plan, bounded physical storage, generated C++ session, and
no-Python execution for SmolVLA and OpenVLA. The terminal gates are G3
(two-model C++ correctness) and G4 (three VLA-semantic optimizations with
evidence).

Branch: `codex/vlaforge-plan-cpp-aot`

Base: `2c46ce5`

## Baseline

Recorded on 2026-07-24 before implementation changes.

### Offline

```text
PYTHONPATH="/tmp/vlaforge-test-deps:$PWD/python" python -m pytest -q
59 passed, 2 skipped in 0.58s
```

The two skips are the opt-in real checkpoint gates.

### Real SmolVLA

```text
1 passed in 8.82s
```

Environment and checkpoint paths match `vlaforge/README.md`.

### Real OpenVLA

```text
1 passed, 5 warnings in 18.17s
```

The checkpoint revision remains
`47a0ec7fc4ec123775a391911046cf33cf9ed83f`.

## Infrastructure note

The checked-in worktree has an initialized, intentionally untracked
`.codegraph/codegraph.db`. The local CodeGraph launcher currently points to a
Mach-O arm64 Node executable while this host is Linux x86-64 and therefore
fails with `Exec format error`. No CodeGraph files were modified or staged.
Source inspection continues without treating this tooling mismatch as a
VLAForge functional blocker.

## Gate A: deployment contracts

Status: passed.

Commit: `ace4a4c` (`feat(vlaforge): freeze deployment artifact contracts`).

Implemented:

- `vlaforge.region_artifact/1` immutable artifact contract;
- static and bounded-symbolic shape profiles;
- backend, device, dtype, layout, alignment, and workspace contracts;
- hidden mutation/RNG/external-I/O effect audit;
- normalized relative paths, SHA-256, exact file sizes, and structured
  diagnostics;
- `vlaforge.compile_bundle/1` manifest with all required semantic, plan, state,
  physical-memory, I/O, artifact, generated-code, binary, toolchain, backend,
  and reproducibility fields;
- deterministic canonical JSON and manifest digest;
- complete on-disk bundle hash/size verification;
- versioned pure-C `RegionExecutable` function-table ABI;
- C and C++ ABI smoke executables.

Current evidence:

```text
Python full offline suite: 73 passed, 2 real-model tests skipped
Python compileall: passed
Wheel: vlaforge-0.1.0.dev0-py3-none-any.whl, 73,343 bytes
CMake Release configure/build: passed with -Werror
CMake ASan+UBSan configure/build: passed with -Werror
ASan+UBSan CTest: 2/2 passed
CMake install/export: passed
git diff --check: passed
```

Contract specification:
`vlaforge/spec/deployment-contracts.md`.

## Gate B: restricted PyTorch frontend

Status: passed.

Milestone commit:
`1c6dc0d` (`feat(vlaforge): add explicit PyTorch frontend capture`).

Implemented:

- explicit `torch.export` capture for declared coarse `TensorRegion` callables;
- bounded static/dynamic shape profiles and typed value contracts;
- eager/export numeric comparison and deterministic graph evidence;
- structured, versioned unsupported reports with no eager fallback;
- mutation alias audit that rejects input/module writes while allowing proven
  invocation-local workspaces;
- hidden RNG and external-I/O audit, including deterministic evaluation-mode
  dropout handling;
- mutable closure rejection and evidence-gated persistent state lifting;
- versioned backend compile requests and finalized artifact contracts;
- save/load helpers for verified exported programs;
- versioned real-model frontend audit reports and reproducible audit tools.

Real SmolVLA evidence:

```text
prepare_prefix: 2,965 nodes, export 2.107 s, max abs error 0
solver_step: 2,525 nodes, export 1.569 s, max abs error 0
trim_action_chunk: 5 nodes, export 0.028 s, max abs error 0
effect audit: pass
persistent state: action_queue, queue_cursor
real Semantic IR gate: 1 passed in 8.85 s
```

The real SmolVLA Semantic IR now models queue refill/reuse, cursor evolution,
state staging, and atomic action commit across three control ticks. Prefix KV
and solver sample remain invocation-local SSA values.

Real OpenVLA evidence:

```text
generate_action_tokens_prefill: 4,427 nodes, export 6.586 s, max abs error 0
generate_action_tokens_decode_step: 3,378 nodes, export 4.510 s, max abs error 0
detokenize_action: 17 nodes, export 0.093 s, max abs error 0
effect audit: pass
explicit fixed-loop tokens == BF16 model.generate tokens: true
persistent state: none
real NF4 Semantic IR gate: 1 passed, 5 warnings in 21.14 s
```

The OpenVLA export audit uses the same pinned checkpoint in BF16 on CPU.
PyTorch 2.6 FakeTensor export of the NF4 bitsandbytes `Params4bit` subclass
fails before graph capture; this limitation and the exact failure are retained
in `doc/reports/vlaforge_frontend_real_models.md`. The audit separates prefill,
one cached decode step, and detokenization, then proves that an explicit fixed
seven-token loop matches BF16 `model.generate()`. KV is loop-carried SSA, not a
`StateSlot`.

Current evidence:

```text
Python full offline suite: 85 passed, 2 real-model tests skipped in 3.51 s
Python 3.10 focused frontend/model suite: 17 passed in 1.86 s
Python compileall: passed
Wheel: vlaforge-0.1.0.dev0-py3-none-any.whl, 95,333 bytes
CMake Release configure/build: passed with -Werror
CMake Release CTest: 2/2 passed
CMake install/export: passed
CMake ASan+UBSan configure/build: passed with -Werror
ASan+UBSan CTest: 2/2 passed
git diff --check: passed
```

Detailed report:
`doc/reports/vlaforge_frontend_real_models.md`.

## Gate C: internal Scheduled Execution

Status: passed.

Milestone commit:
`23f61c1` (`feat(vlaforge): add deterministic scheduled execution plan`).

Implemented:

- compact internal `PlanModule` with deterministic integer policy, block,
  task, logical-buffer, state, and artifact IDs;
- one VLA-focused `Task` record with input, region, loop, branch, state,
  validation, commit, publish, and control kinds;
- typed input/output bindings, explicit producer dependencies, bounded
  freshness/deadline guards, artifact variants, and source-op/location maps;
- structured fixed loops and branches as nested Plan blocks;
- canonical deterministic serialization, SHA-256 digest, and round-trip load;
- Semantic IR to Plan lowering with no public Plan DSL;
- verifier for IDs, graph cycles, read-before-produce, artifacts, freshness,
  loop bounds, validation/commit, and commit/publish ordering;
- direct Plan reference executor using Plan tasks rather than reconstructing
  the Semantic Interpreter;
- byte-equivalent normalized Semantic Interpreter/Plan Executor traces across
  three SmolVLA and OpenVLA fixture ticks.

The real OpenVLA Semantic IR was strengthened during this gate. It no longer
wraps Hugging Face `generate()` in one region; it now contains prefill, a fixed
six-iteration decode-step loop, token extraction, and detokenization. The same
NF4 checkpoint matches official `predict_action()` token IDs and action
exactly. KV is invocation-local loop carry and the persistent state table
remains empty.

Deterministic real-program plans:

```text
SmolVLA: 28 tasks, 4 blocks, 30 buffers, 2 states, 7 artifacts
digest: 8d4c00ad2006157650620bf93a552b3b0a6ca039a448d6d3b1172bba57f88b76

OpenVLA: 15 tasks, 2 blocks, 18 buffers, 0 states, 4 artifacts
digest: 73e46eb3e1532143b78d6e64425bd7464368a515ac6ac4c30b5f4b54fdc04882
```

Current evidence:

```text
Plan focused suite: 11 passed
Python full offline suite: 96 passed, 2 real-model tests skipped in 2.49 s
Python 3.10 Plan/model suite: 17 passed in 0.11 s
Real OpenVLA NF4 explicit-loop gate: 1 passed, 5 warnings in 21.14 s
Python compileall: passed
Wheel: vlaforge-0.1.0.dev0-py3-none-any.whl, 111,314 bytes
git diff --check: passed
```

Detailed contract:
`doc/reports/vlaforge_plan_and_runtime_contract.md`.

## Gate D: physical state and Static Arena

Status: passed.

Milestone commit:
`1ce45f4` (`feat(vlaforge): physicalize state and static memory`).

Implemented:

- bounded logical-version to ring-slot mapping;
- capacity proof from retention, max-in-flight, consumer lag, and fallback
  snapshots;
- per-state slot size, alignment, device, offset, and stable logical ID;
- typed storage sizing with explicit overrides for dynamic internal tensors;
- producer/consumer liveness for every internal logical buffer;
- distinct buffer classes for loop carry, state descriptors, region workspace,
  pending action, and committed action;
- deterministic non-aliasing Static Arena baseline;
- explicit artifact workspace buffers and alignment;
- verifier rejection of unsafe capacity, unplanned/duplicate mappings,
  truncated lifetimes, arena overflow, live overlap, and state-ring overlap;
- deterministic C++17 constexpr state/buffer tables with compile test;
- move-only C++ `StaticArena` with bounds/alignment checks and no hot-path
  allocation.

Current real-program layouts:

```text
SmolVLA:
  static arena: 17,152 bytes, alignment 64, allocations 25
  state arena: 6,100 bytes
  action_queue: 5 slots x 1,216 bytes
  queue_cursor: 5 slots x 4 bytes
  physical digest:
    cf0c90793e1831152eec634af192506d30faa7bd5802dfcbdb8377989ef1be0f

OpenVLA:
  static arena: 448 bytes, alignment 64, allocations 12
  persistent state arena: none
  physical digest:
    0ce0443c792434a1d85dd71125cab7f393068e6c4da62e0f001515c8531127ca
```

The baseline allocator intentionally performs no address reuse. Safe
lifetime-based cross-cycle reuse remains a measured Milestone H optimization,
so its benefit can be isolated from correctness.

Current evidence:

```text
Plan/memory focused suite: 22 passed in 0.19 s
Python full offline suite: 107 passed, 2 real-model tests skipped in 2.55 s
Python 3.10 Plan/model suite: 28 passed in 0.14 s
C++ Release CTest: 3/3 passed
C++ ASan+UBSan CTest: 3/3 passed
CMake install/export with static_arena.h: passed
Generated constexpr header C++17 -Werror compile: passed
Python compileall: passed
Wheel: vlaforge-0.1.0.dev0-py3-none-any.whl, 116,570 bytes
git diff --check: passed
```

## Gate E: lightweight C++ Runtime

Status: passed.

Milestone commit:
`f61a24d` (`feat(vlaforge): add preallocated C++ state runtime`).

Implemented:

- fixed `Status`, `Epoch`, `TensorView`, and integer `TraceEvent` contracts;
- preallocated bounded `StateStore` over physical ring descriptors;
- preallocated per-state transaction staging;
- synchronous begin/read/stage/commit/abort/reset semantics;
- exactly-once transaction close and duplicate-stage rejection;
- typed `PendingAction` versus `CommittedAction`;
- `ActionQueue::Publish` accepting only committed actions;
- model-independent generated `Session` interface;
- optional function-pointer trace sink with no string payload;
- multi-tick global allocation-counter test proving no heap allocations after
  runtime construction.

The C++ state smoke test covers:

```text
initialize/read
stage/commit
ring wraparound and overwritten-version rejection
duplicate stage
validation abort without state mutation
explicit abort
double commit/abort rejection
episode reset
committed action publish
ten consecutive allocation-free ticks
```

Current evidence:

```text
Python full offline suite: 107 passed, 2 real-model tests skipped in 2.51 s
C++ Release configure/build with -Werror: passed
C++ Release CTest: 4/4 passed
C++ ASan+UBSan configure/build with -Werror: passed
C++ ASan+UBSan CTest: 4/4 passed
CMake install/export of all runtime headers: passed
Hot-path allocation counter: unchanged across 10 ticks
Forbidden runtime scan: no model names, JSON, Python.h, std::string, or maps
Python compileall: passed
Wheel: vlaforge-0.1.0.dev0-py3-none-any.whl, 116,570 bytes
git diff --check: passed
```

## Gate F: static C++ AOT Codegen

Status: passed.

Milestone commit:
`feat(vlaforge): generate standalone C++ sessions`.

Implemented:

- deterministic physical-Plan to C++17 source generation;
- concrete `Session` with epoch-qualified inputs, reset, bounded tick
  execution, committed-action reads, trace sink, and destruction;
- integer/`constexpr` task, buffer, state, input, clock, and artifact tables;
- fixed-loop emission and explicit rejection of unsupported constructs;
- embedded CPU fixture artifact backend for offline CI;
- standalone runner, clean CMake build, install, and export;
- reproducible `vlaforge codegen` CLI;
- exact Semantic IR/Plan/C++ normalized trace comparison over three ticks;
- optional CUDA AOTInductor `RegionExecutable` backend;
- real `.pt2` export/load/run audit in a no-Python C++ process.

Current evidence:

```text
Generated source-set golden:
  d05684708daa9e96c15d26319bdfdb8fefcca3eb3a57920abfc815e53764ef9d
Three-way fixture trace: 42/42 fixed events exact across 3 ticks
Three-way action comparison: passed, absolute tolerance 1e-6
Clean runner with invalid PYTHONHOME/PYTHONPATH: passed
Runner libpython dependency: none
CUDA AOTI on RTX 3060: max abs error 4.341e-9, 16 outputs
CUDA AOTI opt-in pytest: 1 passed in 18.80 s
Offline Python suite: 111 passed, 3 skipped in 3.67 s
Python 3.10 focused suite: 48 passed in 1.14 s
C++ Release CTest: 4/4 passed
C++ ASan+UBSan CTest: 4/4 passed
Generated clean build/run/install/export: passed
Wheel: 131,057 bytes
Python compileall: passed
git diff --check: passed
```

Detailed report:
`doc/reports/vlaforge_cpp_codegen_aoti.md`.

## Gate G3: two real models, no Python

Status: passed.

SmolVLA:

```text
backend: CUDA AOTInductor
generated source digest:
  352ae0704404984afe5d8243ffc7d79ebffd556e083d143f61502295ff10cab0
Semantic/Plan/C++ execution events: 66/66 exact
reset event: exact
transactions: 0, 1, 2
action_queue and queue_cursor versions: 1, 2, 3
numeric comparisons: 50/50 within explicit BF16 contract
maximum published-action absolute error: 0.010336
```

OpenVLA:

```text
backend: shared CPU TorchScript archive after documented AOTI host OOM
archive size: 15,085,415,106 bytes
archive SHA-256:
  f77f68374187adade017e6f5d9e35ba0d97936f144ca7ae4fc5711b4a4c2eaec
generated source digest:
  cefbb5b403dce15ea675d7f2d0b4696256a8b7f4d6dfae0199b9e877ec111e3d
official/Python/C++ token IDs: exact
region/decode-step tensors: 460/460 exact
Semantic/Plan/C++ execution events: 54/54 exact
transactions: 0, 1, 2
persistent states and state commit events: 0
```

Both clean C++ runners pass with invalid `PYTHONHOME/PYTHONPATH` and neither
links `libpython`. Runtime backends contain no model name or model-specific
branch.

Detailed reports:

- `doc/reports/vlaforge_cpp_smolvla_real.md`
- `doc/reports/vlaforge_cpp_openvla_real.md`

## Gate G4: VLA-specific whole-program optimization

Status: passed.

Implemented:

- transitive Epoch/StateVersion cache-key synthesis with freshness and episode
  invalidation;
- fixed-capacity C++ `EpochVersionCacheGuard`;
- true pure-region temporal LICM plus preheader recognition;
- deterministic lifetime interval packing for cross-cycle static arena reuse;
- positive, forbidden-negative, state/action trace, generated-C++, and
  sanitizer coverage.

Measured real generated-C++ results:

```text
SmolVLA prefix cache steady p99: 50.367 -> 29.825 ms (-40.79%)
OpenVLA prefix cache steady p99: 34.270 -> 3.021 s (-91.18%)
SmolVLA LICM steady p99: 210.534 -> 50.876 ms (-75.83%)
OpenVLA LICM: already prehoisted, measured improvement 0%
SmolVLA static arena: 17,152 -> 15,360 B (-10.45%)
OpenVLA static arena: 448 -> 192 B (-57.14%)
```

All compared action lines, evidence hashes, and non-Region state/transaction/
action traces are exact. SmolVLA arena reduction is below the 20% internal
target because overlapping 6,400-byte solver lifetimes cannot legally alias.
OpenVLA LICM reports no speedup because its prefill is already the
autoregressive loop preheader. These limitations are recorded without
inflating the result.

Detailed report:
`doc/reports/vlaforge_whole_program_optimizations.md`.

## Final regression

Status: passed.

```text
Offline Python: 124 passed, 3 real/CUDA gates deselected in 4.87 s
Focused optimization/codegen: 30 passed in 2.00 s
Real SmolVLA Python checkpoint: 1 passed in 12.12 s
Real OpenVLA Python checkpoint: 1 passed in 29.30 s
CUDA AOTI opt-in audit: 1 passed in 20.86 s
C++ Release CTest: 5/5 passed
C++ ASan+UBSan CTest: 5/5 passed
Generated C++ optimization audit: gate_passed=true
Default SmolVLA/OpenVLA codegen golden digests: unchanged
Python compileall: passed
CMake install/export with epoch_cache.h: passed
Wheel contains all three optimization modules: passed
CLI help smoke: passed
git diff --check: passed
```

The two real generated benchmark runners were also verified with invalid
Python environment variables and no `libpython` dependency. Gate G3 and Gate
G4 are complete; Jetson/vendor backends, π0, closed-loop robot evaluation, and
paper artifact freeze remain intentionally out of scope for this Goal.
