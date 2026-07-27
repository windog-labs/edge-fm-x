# MindDrive eager/direct-AOTI/generated-C++ matrix

Status: **passed**.

| Path | Init mean | First Run | Warm mean [95% CI] | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| eager | 26352.71 ms | 1629.26 ms | 1511.66 [1504.59, 1515.41] ms | 1513.95 ms | 1518.72 ms | 1519.80 ms |
| direct_artifact | 4925.41 ms | 1385.84 ms | 1275.17 [1265.51, 1282.75] ms | 1279.40 ms | 1284.40 ms | 1288.80 ms |
| generated_session | 4104.27 ms | 1394.94 ms | 1279.71 [1275.59, 1283.34] ms | 1281.18 ms | 1284.68 ms | 1288.74 ms |

## Comparison

- Eager/generated warm speedup: 1.181x.
- Generated/direct warm delta: 0.356%.
- Direct and generated execute the same 66 physical AOTI artifacts. Their delta includes Python versus generated C++ composition and state-management boundaries.
- Initialization values are reported but are not directly compared: generated Session initialization begins after C++ input loading, while Python path initialization includes input preparation.
- Output validation here uses aligned scalar probes. Full named-output parity remains backed by the real L4 correctness report.
- This is RTX 3060 Host-CUDA evidence, not Orin evidence.
