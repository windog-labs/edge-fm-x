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

## Next

1. Implement the restricted PyTorch frontend, export capture, effect audit,
   bounded shape profile, artifact requests, and unsupported reports.
2. Produce real SmolVLA/OpenVLA frontend audit evidence before lowering to the
   scheduled plan.
