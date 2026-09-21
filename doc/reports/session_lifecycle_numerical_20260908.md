# Numerically Bound Session Lifetimes

2026-09-08 13:27 UTC. The initial H100 diagnostic and remote/local independent
checks are complete. A fresh same-source retain/scoped pair is still running.
This is G1/G4 memory evidence, not a formal timing or full-paper acceptance.

## Public Tool

`vlaforge/tools/diagnose_session_lifecycle.py` now carries numerical-worker
initialization from each typed Region binding into the native diagnostic runner.
It runs once after ownership registration and before inputs or any Session.
Legacy protocols stay unconfigured; model names do not appear in dispatch.

Numerical metadata, linked runtime library identities and loop schedules are
frozen. Each cycle checks complete output bytes and actual runtime mappings;
worker bootstrap markers and every declared replay counter are checked.
The memory-policy selector defaults to source metadata and reuses the shared
bundle builder's retain/scoped SDK gates. It records overrides and verifies the
actual CMake selection, without modifying a source model bundle or global cache.

Three bootstrap/counter regressions first failed, then passed. Two memory-policy
selection regressions also first failed. The combined lifecycle/benchmark suite
passes 114 tests. Full CPU regression 025 is recorded separately from CUDA
execution; optional tests remain explicit skips.

## Initial Real H100 Diagnostic

Local root:
`artifacts/edgefm-vla-goal/20260906-044509/lifecycle-bound-h100x4-20260908-001/`.
Remote root uses the same name under H100 `edgefm-vla-goal-20260906/runs/`.
Host is `zzm-h100-x4`, GPU0 `GPU-3251e3fd-849d-6a9d-0d06-7e4290aecec8`.

The existing SmolVLA complete-002 formal bundle/reference is reused at its same
GPU boundary: 16 real episodes, saved noise, complete normalized and official
scaled F32 `[1,50,6]` outputs. The model reference producer is not rerun. The
293-file new local snapshot supplies the diagnostic tool; runtime sources and
model artifacts come unchanged from the origin's frozen benchmark.

Three separate native processes each create/run/destroy five Sessions. All
240 calls / **480 complete tensors / 144000 values** are byte-exact. Each new
required Session captures N10 once and replays all 16 invocations; no ordinary
fallback. All five numerical policy bindings and actual numerical DSO mappings
are checked. No native process maps Python runtime libraries.

The first prepare failed before native execution because OpenSSL was not found.
A separate configure-only probe changed the search root to `/opt/conda`, matching
the origin's successful build. Pipeline 002 uses that setting in a fresh output
directory, retaining all failed files and unchanged source/model packages.

- Original failed pipeline SHA:
  `da1e35347a1aad0295aff4348c858b724b9925e46022fffec7e3f6117b1974d5`.
- Recovered complete pipeline SHA:
  `0fa2adc4f4af699e540e3b6f54d26b41a9377175bb095d5ea845636a6672bb1a`.
- Independent native audit SHA:
  `2e01b06d2fb68d516b5d330a3f8f3340a5f31f15cfc6e8115577d4984ce2e49a`.
- `retrieval-001/report.json`: 413 files locally retrieved, with all 480 tensors
  and allocator snapshots rechecked; not a complete runnable model bundle.

Allocated/active bytes after destruction stay at 34603008 (33 MiB) in all three
policies. Ordinary/shared-stream reserved bytes stay at 1937768448 (1848 MiB).
Required reserved bytes are 1941962752, 1946157056, 1950351360, 1954545664 and
1958739968: **4 MiB more per new Session**, 16 MiB above the first destruction.
This is not zero allocation, and this initial replay path does not prove bounded
reserved-memory retention. Constant active bytes alone do not prove no leak.

## Same-Source Policy Comparison

The separate `lifecycle-memory-h100x4-20260908-001` freezes the new common
memory-policy selector and repeats both `retain` and `scoped-reclaim`, each with
all three loop policies and five fresh Session lifetimes. Same source bundles,
observations, noise and numerical bindings are required; actual compiler flags,
runner identities and output maps are audited independently. Its results are
pending and must not be inferred from the older RTX scoped experiment.

Even a successful finite scoped comparison will not establish unlimited
thread/Session behavior, every backend's workspace ownership, zero device frees,
failure-path reclamation, an E2E latency improvement or Orin/BPU behavior.
