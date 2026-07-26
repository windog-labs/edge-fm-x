# MindDrive real L4 Host-CUDA benchmark

Status: **passed**.

| Revision mode | Init mean | First Run mean | Warm mean | p50 | p90 | p99 | Process std |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 4083.01 ms | 1388.93 ms | 1270.38 ms | 1272.01 ms | 1279.18 ms | 1281.42 ms | 8.13 ms |
| same | 4082.14 ms | 1392.75 ms | 260.01 ms | 259.95 ms | 260.69 ms | 261.46 ms | 0.20 ms |
| new | 4075.96 ms | 1395.93 ms | 1279.27 ms | 1280.35 ms | 1282.95 ms | 1284.21 ms | 2.02 ms |
| missing | 4073.88 ms | 1398.62 ms | 1281.75 ms | 1282.90 ms | 1285.59 ms | 1286.19 ms | 0.55 ms |

## Declared memory classes

- External input per invocation: 29493452 bytes
- External output per invocation: 29332 bytes
- Per-Run static arena: 56559808 bytes
- Authoritative state arena: 3351680 bytes
- Derived-cache physical capacity: 39321600 bytes

All cells are five independent processes running the real MindDrive 0.5B generated no-Python C++ Session on RTX 3060 sm_86. `full` cycles the five held-out real frames; `same`, `new`, and `missing` isolate InputRevision behavior on the same frame while authoritative state continues to commit.

This is Host-CUDA evidence, not Orin, cross-GPU, closed-loop, power, thermal, or model-kernel-optimization evidence.
