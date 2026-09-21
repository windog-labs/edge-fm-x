# H100 pi0.5 Native Replay Formal Benchmark

2026-09-08 18:35 UTC. This report closes the H100 normalized-output
three-policy formal replay for pi0.5 that was still pending after the
16-frame native validation. It uses the H100-generated official references
from the same checkpoint/input/noise pack and the already passing ATen
no-Python bundle; it does not mix in the older RTX reference set.

## What Was Run

- Machine: `zzm-h100-x4`, GPU UUID
  `GPU-3251e3fd-849d-6a9d-0d06-7e4290aecec8`, sm_90.
- Artifact: `native-session-009/bundle` rebuilt into
  `pi05-h100-benchmark-variants/{off,batch-only,required}/bundle`.
- Protocol: `vlaforge.session_latency_protocol/2`, 16 distinct real frames,
  one `normalized_action_chunk` F32 `[1,50,32]` output, N=10,
  explicit numerical provider bootstrap, bitwise eager gate,
  paper gates `mse_max=1e-5`, `cosine_min=0.9999`.
- Formal protocol SHA `d47ad2a956064be66873fbe8114813bf2e1d09e8c676320ed3f7da08f9b0cb9b`.
- 5 independent processes per policy, 128 warmup + 1024 measured calls per
  process, 17280 total calls, 17280 complete normalized tensors validated.

## Results

| policy | mean ms | p50 ms | p95 ms | p99 ms | max ms | chunks/s |
|---|---:|---:|---:|---:|---:|---:|
| off | 133.087 | 131.171 | 142.018 | 148.597 | 215.536 | 7.514 |
| batch-only | 135.040 | 136.349 | 144.489 | 152.641 | 209.512 | 7.405 |
| required | 65.300 | 65.349 | 66.903 | 71.753 | 80.505 | 15.314 |

Required whole-loop replay reduces mean latency by 50.9345% relative to the
same-artifact ordinary execution. This is the execution-layer replay path, not
an Agent kernel or memory optimization, and it is not an Orin/BPU result.

## Evidence

- Independent local audit:
  `artifacts/edgefm-vla-goal/20260906-044509/h100-pi05-aten-recovery-20260909/independent-formal-audit.json`
  SHA `f65a27b7911edd3bf3ec2d2d8a4b7c13e0c4ff9052918280cfd1d9f1a207df28`.
  It independently rechecked 278 local frozen files, all 17280 raw output
  calls against the direct/eager references, per-process latency reports,
  aggregate table/CDF rows and no-Python process maps.
- Aggregate report SHA `1f0365e670946e2e5258a0584bf826e9a459e8d8eff86afae6f608a123581761`.
- Latency table SHA `3097ba910ee7952561da4950f4bb89678afe6c125becba20c58a6da59d951329`.
- CDF CSV SHAs: off `b9df0ece...`, batch-only `f15b6629...`,
  required `98c36184...`.
- Figure report SHA `cef52dcca9cf199ed29043dfc4059ff56f835c7a67fed7ae893cd44bf2bfec56`;
  PNG SHA `4bd2cf263212662241bad999f0235ed8aa6f3b8386f3dc161b0ab07d26077433`,
  PDF SHA `a68208c59e2dd08baba48568d59577756ebb89c64b6f27e901b270bf8ba31bed`.

## Boundaries and Remaining Work

The boundary is resident CUDA model tensors through synchronized
`Session::Run` completion; init, H2D/D2H, preprocessing and action handoff are
excluded. This formal result covers the normalized action only. Native
physical/double-output H100 validation, official/vendor baseline tables,
other-models-on-the-same-H100 rows and Orin/BPU cells remain open.
`full_paper_acceptance=false`.
