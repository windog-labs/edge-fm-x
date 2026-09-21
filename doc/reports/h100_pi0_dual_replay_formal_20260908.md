# H100 pi0 Dual-Output Replay Formal

2026-09-08 23:15 UTC. The H100 pi0 no-Python dual-output replay is now
formally measured across 16 frames and three execution policies, in addition
to the earlier normalized-only and single-frame dual-output results.

## Protocol

- 16 distinct H100 frames; outputs are normalized F32 `[1,50,32]` and native
  F64 `[1,50,14]`.
- 5 independent processes per policy, 128 warmup + 1024 measured calls each.
- 17280 calls and 34560 complete tensors validated.
- Protocol SHA `6015d71e582f4a7628ecc99555a6f0dda8961c69aa77d27d5cc2385f3367c131`.

## Results

| policy | mean ms | p99 ms | chunks/s |
|---|---:|---:|---:|
| off | 107.460 | 122.009 | 9.306 |
| batch-only | 108.077 | 117.210 | 9.253 |
| required | 59.861 | 63.795 | 16.705 |

Required replay reduces mean latency by 44.2949% relative to ordinary
execution. All normalized and physical chunks are byte-exact to the H100
official references and direct full-IR outputs.

## Evidence

- Aggregate report SHA `ac822e8860351d598e28db09b4fb4f64278025b04a31aa572a01472b9b9a8b50`.
- Independent audit JSON:
  `dual-formal-summaries/pi0/pi0-h100-dual-independent-audit.json`
  SHA `3f155ce6f3394af3bb5751beb56a65739005aecdb0df2c4beb7b8823f7180b48`.
- Figures and CDF CSV under
  `dual-formal-summaries/pi0/`.

This is H100 resident model-tensor evidence, not Orin/BPU.
`full_paper_acceptance=false`.
