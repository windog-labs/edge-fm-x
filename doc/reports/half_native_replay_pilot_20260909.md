# Half Native Whole-Loop Replay Pilot

2026-09-09 01:40 UTC. Baseline, FP16 and BF16 bundles were rebuilt with the
public execution-policy variants and piloted on RTX 3060. Each lane ran
off/batch-only/required with 16 warmups + 32 measured calls. All 48 complete
actions per policy are byte-exact against the lane reference; required workers
report `REPLAY_FINAL,11,1,10,48,0` (10 captured steps, 48 replay, no ordinary
fallback).

## Pilot Means

| Lane | Off ms | Batch-only ms | Required ms |
|---|---:|---:|---:|
| Baseline | 93.474704 | 92.211023 | 76.290214 |
| FP16 | 94.431004 | 93.118270 | 76.611065 |
| BF16 | 94.633408 | 92.789239 | 76.779725 |

Required replay shows a pilot mean reduction of about 18-19% relative to the
same lane's off mean. This is execution-policy evidence, not a selected kernel
speedup and not a formal CDF.

## Evidence

- Variants under `half-native-benchmark-001/variants-{baseline,fp16,bf16}/`.
- Pilot evidence under `variants-prepared-{baseline,fp16,bf16}/pilot/`.
- Formal replay CDF is still pending; these 48-call pilots do not replace it.
