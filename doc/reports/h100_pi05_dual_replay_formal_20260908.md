# H100 pi0.5 Dual-Output Replay Formal

2026-09-08 23:25 UTC. The H100 pi0.5 no-Python dual-output replay is now
formally measured across 16 frames and three execution policies.

## Protocol

- 16 distinct H100 frames; outputs are normalized F32 `[1,50,32]` and native
  F64 `[1,50,14]`.
- 5 independent processes per policy, 128 warmup + 1024 measured calls each.
- 17280 calls and 34560 complete tensors validated.
- Protocol SHA `97bee86b8c658398f62e165c197a91ba3112a777bb2a0b54a071cae2cf7b25ac`.

## Results

| policy | mean ms | p99 ms | chunks/s |
|---|---:|---:|---:|
| off | 126.277 | 140.064 | 7.919 |
| batch-only | 123.479 | 137.930 | 8.099 |
| required | 64.436 | 69.198 | 15.519 |

Required replay reduces mean latency by 48.9725% relative to ordinary
execution. All normalized and physical chunks are byte-exact to the H100
official references and direct full-IR outputs.

## Evidence

- Aggregate report SHA `b42681b44d6d3a54ddfa2a83c0e54a7accd4d62cbcf15a944fd9d3dfcf557939`.
- Independent audit JSON:
  `dual-formal-summaries/pi05/pi05-h100-dual-independent-audit.json`
  SHA `2ee27314eafa8bcf09cf93874a67f12dc3ccf7f7dafa98072f576f7e1d6614d2`.
- Figures and CDF CSV under
  `dual-formal-summaries/pi05/`.

This is H100 resident model-tensor evidence, not Orin/BPU.
`full_paper_acceptance=false`.
