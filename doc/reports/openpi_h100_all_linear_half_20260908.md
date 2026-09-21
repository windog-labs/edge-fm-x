# H100 OpenPI All-Linear FP16 Python Quality

2026-09-09 01:30 UTC. A generic exported-graph transform now lowers every
`aten.linear.default` in an OpenPI step Region to explicit FP16 compute while
preserving the original ABI and output dtype. This extends the low-precision
evidence from SmolVLA to the OpenPI model family on H100.

## Public Lowering

- New API: `lower_exported_all_linear_halves` in
  `vlaforge/deployment/linear_precision.py`.
- It accepts an `ExportedProgram`, rewrites all Linear nodes with half input,
  weight and bias casts, casts the Linear result back to the original float
  dtype, then save/reloads without changing graph signature or state storage.
- Does not require a scheduler/step placeholder and is not model-name
  dispatched.
- Fixture coverage added in `test_linear_precision.py`; the full file passes
  with 20 tests, including FP16/BF16 save/reload ABI checks.

## Real OpenPI Results

Each model used its own H100 v2 capture, a rewritten FP16 step Region and 16
real H100 frames in Python free-running execution. All frames are complete
normalized chunks.

| model | Linear count | worst MSE | worst max-abs | minimum cosine |
|---|---:|---:|---:|---:|
| pi0 | 131 | 1.3312e-06 | 6.9662e-03 | 0.99998424 |
| pi0.5 | 167 | 3.3693e-07 | 4.1522e-03 | 0.99999710 |

All 32 frames pass the existing paper gate `MSE <= 1e-5`,
`cosine >= 0.9999`. The numerical compare with zero tolerance is not
bitwise-exact, so this is not described as lossless.

Correction (2026-09-09): frames 8-15 were selected for this summary after all
16 frames had been evaluated. These are descriptive post-hoc subset statistics,
not a predeclared held-out validation set. No calibration was performed, but
that does not establish an independent held-out quality gate.

| model | subset worst MSE | subset max-abs | subset minimum cosine |
|---|---:|---:|---:|
| pi0 | 1.3228e-06 | 6.9662e-03 | 0.99998424 |
| pi0.5 | 2.8047e-07 | 2.9390e-03 | 0.99999804 |

## Evidence

- H100 pi0 quality JSON:
  `artifacts/edgefm-vla-goal/20260906-044509/h100-pi05-aten-recovery-20260909/openpi-all-linear-half-20260908/pi0/quality.json`
- H100 pi0.5 quality JSON:
  `.../openpi-all-linear-half-20260908/pi05/quality.json`
- Ledgers in the same folders record source/result graph hashes, precision and
  Linear count.
- Large FP16 step archives remain on H100:
  `runs/pi05-h100-aten-recovery-20260909/{pi0,pi05}-openpi-all-linear-fp16-20260908/openpi-step-fp16.pt2`.

## Scope

This is Python full-model free-running quality with no calibration/held-out
split, no native no-Python execution, no latency, no dual-output physical
validation and no Orin/BPU evidence. It is not a selected deployment claim.
