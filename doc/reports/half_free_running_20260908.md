# FP16 Full-Trajectory Quality Candidate

2026-09-08 23:20 UTC. The public precision path now supports a real
half-precision Linear replacement. On the existing SmolVLA recovered-profile
workload, FP16 passes the predeclared full-action quality gate for all sixteen
real trajectories, including the eight held-out episodes that every earlier
INT8 scale strategy failed.

This is full original-IR CUDA free-running validation, not no-Python native
deployment, not a formal latency/CDF, and not a lossless claim.

## Code

- `vlaforge/python/vlaforge/deployment/half_linear.py`: generic scheduled FP16
  or BF16 Linear replacement with explicit int64 step ABI and nonfinite invalid
  step guard. It records real `aten.linear.default` execution after explicit
  half casts and owns half-precision weight/bias buffers.
- `vlaforge/python/vlaforge/deployment/linear_precision.py`: `precision_kind`
  accepts `int8`, `float16` or `bfloat16`; the graph rewrite, profile check,
  state ownership and scheduler ABI remain model-independent.
- Focused INT8/half precision suite: 78 passed.

## Real Run

Host is the local RTX 3060
`GPU-ce878329-6b58-5666-292d-94185f5e5585`. All original inputs, saved noise,
weights and references are unchanged from the calibration probe. The fixed
first-eight/last-eight partition is reused; no held-out sample is fit. Only the
`to_267 -> linear_115` head edge is replaced. Ten actual scheduler steps run per
trajectory and every quantized next-sample feeds the following step; this is not
teacher forcing.

All 16 complete normalized action chunks are finite and pass
`MSE <= 1e-5` and `cosine >= 0.9999`.

| Metric | Worst all 16 | Worst held-out 8 |
|---|---:|---:|
| MSE | 2.550426e-6 | 2.434350e-6 |
| Minimum cosine | 0.9999993423 | 0.9999994815 |

The worst held-out sample is `sample-000014` from episode 42. None of the four
earlier INT8 strategies passed held-out quality; this FP16 candidate is the
first validated full-trajectory low-precision path in this series.

## BF16 Comparison

The same public lowering was run again with `bfloat16`. All sixteen trajectories
also pass the fixed quality gate. BF16 has a larger but still in-gate error than
FP16, which is useful for the stepwise-precision narrative.

| Precision | Worst all 16 MSE | Worst held-out MSE | Minimum held-out cosine |
|---|---:|---:|---:|
| float16 | 2.550426e-6 | 2.434350e-6 | 0.9999993732 |
| bfloat16 | 5.291621e-6 | 5.109796e-6 | 0.9999986243 |

BF16 protocol SHA: `f86856bf61f7c084e695675a4dbfaa4235af56c1b510bd3fc1db7b8656ea3442`.
BF16 independent audit SHA:
`f7e3f514b11124270f6334522ba67d04c0ee3e08196cb9b4bf01e782d3571cba`.
BF16 supervisor PID 2074915 exited 0 with an empty final GPU owner list. The
first BF16 launch attempt recorded a relative driver path failure and is
preserved under `control-failed-001/`.

## Evidence

- Protocol SHA: `4f6a1bcbe1e21be5b9bd6a56afd3d676ae53da65d6ec7382df15d8ce89ae2a3b`.
- Full campaign result and per-sample raw actions:
  `artifacts/edgefm-vla-goal/20260906-044509/half-free-running-v1/results/`.
- Independent audit SHA:
  `08264a3bfd3671f41de8737e90eab836b4e393e52f738880d714b08da82105a0`.
- Supervisor/telemetry: process PID 2067408, exit 0, final GPU owner list empty.

## Step TorchScript Transport

The rewritten FP16 step Region was also exported with the public TorchScript
helper. The saved `step-half.pt` archive is 907,403,934 bytes, SHA256
`c33e79922bb9f09925bdf7aa5174bfad716694530e9e5f2822e8ce347f768b7e`, and all
11 validation cases (one example plus ten explicit scheduler steps) are
bitwise-equal after save/reload. This is the Python-to-TorchScript boundary;
the full no-Python Session and complete-action native validation are still
pending.

## Remaining Boundary

FP16 here is still one replaced Linear in a Python-interpreted IR experiment;
it has no no-Python Session, no native profiler, no allocator comparison, no
latency table, no vendor/board result. Cosine near one and
MSE below the gate are not physical task success or mathematical exactness.
The next steps are native FP16/BF16 artifacts and same-runtime
speed/memory comparison, then integration into the G5 evidence chain.
