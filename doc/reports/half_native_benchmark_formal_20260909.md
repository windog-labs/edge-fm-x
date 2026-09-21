# Half Native Formal Benchmark

2026-09-09 00:50 UTC. Baseline, FP16 and BF16 no-Python Sessions each completed
the formal protocol on the same RTX 3060: five independent processes, 128
warmups and 1024 measured calls per process, same 16 real inputs and same
resident model-tensor boundary. The independent analysis recomputed all
aggregate statistics and CDFs from per-process raw `samples.csv`.

## Timing

| Lane | Mean ms | p50 ms | p95 ms | p99 ms | Max ms | Std ms | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 93.351683 | 93.319855 | 94.233526 | 94.774618 | 99.122241 | 0.542799 | 10.712180 |
| FP16 | 94.182185 | 94.237280 | 95.118514 | 95.548861 | 97.499200 | 0.610690 | 10.617719 |
| BF16 | 93.792310 | 93.785164 | 94.548589 | 94.948899 | 105.566804 | 0.503871 | 10.661855 |

FP16 is 0.8896% slower than baseline in mean; BF16 is 0.4719% slower. Neither
half candidate shows a stable native speedup in this ordinary single-Session
path. No outlier is removed.

## Allocator

Lifetime cumulative allocated peak is 1884473856 B for baseline and
1884520448 B for both half lanes (+46608 B). Reserved peak stays 1925185536 B
across all lanes. Steady allocator requests total 65745920 for baseline and
66104320 for FP16/BF16 (+70 per measured call). Active bytes after Session
destruction stay 9568256 B in every lane.

## Evidence

- Independent formal analysis SHA:
  `ef615c48383372763e4a1668b4eaf4096762d4956faefa3862f1d19853bfe074`.
- Formal table CSV SHA:
  `aeb65d79032f2f8b7d18db29d090e29f50737274602ce038a7b5c667c012e07d`.
- CDF PNG SHA:
  `1b171248a257c6953b09df026982a6a587f8ab2f6292eda5b216623af2efcc9e`.
- Raw evidence under `half-native-benchmark-001/prepared/<lane>/runs/`.

## Limitations

This is native ordinary execution only (off policy), no replay, no profiler, no
energy, no board, no vendor comparison, no lossless claim and no Orin/BPU
acceptance. The model is SmolVLA recovered-profile normalized action boundary.
