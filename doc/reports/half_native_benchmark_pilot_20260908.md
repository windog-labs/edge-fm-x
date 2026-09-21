# Half Native Same-Runtime Benchmark Pilot

2026-09-08 23:59 UTC. Baseline, FP16 and BF16 native Sessions were measured on
the same local RTX 3060 with the same public benchmark entrypoint, same 16
inputs and same resident model-tensor boundary. This is a one-process pilot
(16 warmup + 32 measured per lane), not the formal five-process x1024 CDF.

Each lane is a no-Python C++ Session. Complete outputs are byte-exact against
the lane's own eager reference (48/48 calls per lane), process maps contain no
Python runtime, and numerical worker initialization succeeded.

## Descriptive Timing

| Lane | Mean ms | p50 ms | p95 ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| same-precision baseline | 93.137304 | 93.058647 | 93.850685 | 94.011325 | 94.011325 |
| FP16 half-head | 94.474481 | 94.394644 | 95.159807 | 95.183190 | 95.183190 |
| BF16 half-head | 94.115055 | 93.996556 | 95.306992 | 96.111507 | 96.111507 |

The FP16/BF16 lanes are not faster in this pilot. Differences are single-process
descriptive values and are not used as a paper speedup claim.

## Allocator Observations

LibTorch allocator counters are lifetime peaks and are never reset; steady
requests cover the measured interval.

| Lane | Allocator requests | Allocator frees | Allocated peak B | Reserved peak B | Active after destroy B |
|---|---:|---:|---:|---:|---:|
| baseline | 410912 | 410912 | 1884473856 | 1925185536 | 9568256 |
| FP16 | 413152 | 413152 | 1884520448 | 1925185536 | 9568256 |
| BF16 | 413152 | 413152 | 1884520448 | 1925185536 | 9568256 |

FP16/BF16 add 2240 allocator requests over 32 measured calls (70 per call) and
46608 bytes to the cumulative allocated peak relative to baseline. Reserved
peak and active-after-destroy values are unchanged. These are not zero
allocation or memory-saving claims.

## Evidence

Root: `artifacts/edgefm-vla-goal/20260906-044509/half-native-benchmark-001/`.

- Protocols: `baseline-protocol.json`, `fp16-protocol.json`,
  `bf16-protocol.json`.
- Prepared/pilot raw evidence under `prepared/<lane>/pilot/00-off/`, including
  `report.json`, `samples.csv`, allocator snapshots and telemetry.
- This pilot does not replace a formal five-process CDF. Full 5x1024 runs are
  the next benchmark step.
