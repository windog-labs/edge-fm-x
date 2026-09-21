# Same-Source Graph Memory Policy Pair

2026-09-08 14:58 UTC. The retain/scoped comparison is complete, independently
audited and locally rechecked. This is bounded Session-lifecycle memory
evidence on H100, not a formal timing result and not final paper acceptance.

## Result

Host: `zzm-h100-x4`, GPU0 `GPU-3251e3fd-849d-6a9d-0d06-7e4290aecec8`.
Both lanes reuse the same verified SmolVLA H100 source bundle, 16 real
observations and saved noise at the same tensor boundary. Only the graph
memory policy differs: the source-bundle default `retain` versus explicit
`scoped-reclaim`, limited to the audited Torch 2.10 provider. Actual CMake
flags, runner binaries and flags files are checked for the selected policy.

Each policy lane ran three loop policies (`off`, `batch-only`, `required`),
five Session create/run/destroy cycles and 16 calls per Session. All
**480 calls / 960 complete F32 tensors / 288000 values** are byte-exact in
both lanes. Numerical worker bindings, runtime-library mappings and replay
counters are verified. No native process maps Python runtime libraries.

After destruction, allocated/active bytes stay at 34,603,008 in both lanes.
`off` and `batch-only` reserved bytes stay constant at 1,937,768,448.

| Required loop lane | Reserved after first destroy | Reserved after fifth destroy | Net growth | Device frees at destroy |
|---|---:|---:|---:|---:|
| retain | 1,941,962,752 | 1,958,739,968 | +16 MiB | 0 |
| scoped-reclaim | 1,937,768,448 | 1,937,768,448 | 0 | 2 per cycle |

So the same complete-model replay lane shows a 4 MiB reserved increase per
new retained Session, and the scoped-reclaim path holds a constant reserved
plateau while performing two checked scoped device frees per graph destroy.
Constant active bytes and a finite plateau still do not prove zero allocation,
zero free, unlimited Session/thread behavior or ownership of every backend
workspace. Retain-mode timing never inherits scoped-mode results.

## Evidence

- Remote pipeline SHA: `d46d0c7d076e052091298b79cfb24f17879b4f1600aceda9d433431918d29c07`.
- Remote independent pair audit SHA:
  `bafebd980163b9f7fe0f4234aea648f29c4110229a994aa0f1b871d5623908c3`.
- Local retrieval SHA:
  `e1d2272246924a8919f8b0a0dae866d178971acd43849f44aca6f11f2f3a2b0d`:
  829 files, all 960 tensors and 288000 values rechecked, including native
  runner binaries and compile flags.
- Memory table: `retrieval-001/memory-lifecycle-table.csv`.

## Remaining Boundary

No timing, energy, power, allocator peak, low-bit, vendor, physical
calibration or board result is inherited. The comparison covers SmolVLA on
H100 under this diagnostic harness. TorchScript and AOTI graph-provider
versions outside the audited Torch 2.10 path need their own validation.
