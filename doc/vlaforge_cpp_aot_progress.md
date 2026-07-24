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
`feat(vlaforge): add explicit PyTorch frontend capture`.

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
real NF4 Semantic IR gate: 1 passed, 5 warnings in 19.22 s
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

## Next

1. Lower verified Semantic IR into deterministic internal Scheduled Execution
   Plans for SmolVLA and OpenVLA.
2. Add the Plan verifier, reference executor, trace mapping, and required
   negative tests.
3. Physicalize state rings and temporary tensor storage into a deterministic
   static arena.
