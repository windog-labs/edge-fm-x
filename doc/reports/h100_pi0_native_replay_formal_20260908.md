# H100 pi0 Native Replay Formal Benchmark

2026-09-08 20:25 UTC. This closes the H100 pi0 normalized three-policy formal
replay after the 16-frame native validation. It uses the same H100-generated
official references, checkpoint and ATen bundle as the 16-frame validation,
so the two results form one H100 evidence chain.

## What Was Run

- Machine: `zzm-h100-x4`, GPU UUID
  `GPU-3251e3fd-849d-6a9d-0d06-7e4290aecec8`, sm_90.
- Artifact: `pi0-native-session-001/bundle`, rebuilt into
  `pi0-h100-benchmark-variants/{off,batch-only,required}/bundle`.
- Protocol: `vlaforge.session_latency_protocol/2`, 16 distinct frames,
  one `normalized_action_chunk` F32 `[1,50,32]` output, N=10,
  explicit numerical provider bootstrap, bitwise eager gate,
  paper gates `mse_max=1e-5`, `cosine_min=0.9999`.
- Protocol SHA `3eb8a871d93fa58f9f5a72be1c0dba28320e2a4fcf9cb4a577d51d2cd8253297`.
- 5 independent processes per policy, 128 warmup + 1024 measured calls per
  process, 17280 total calls, 17280 complete normalized tensors validated.

## Results

| policy | mean ms | p50 ms | p95 ms | p99 ms | max ms | chunks/s |
|---|---:|---:|---:|---:|---:|---:|
| off | 109.108 | 108.276 | 113.572 | 119.294 | 179.245 | 9.165 |
| batch-only | 108.722 | 107.341 | 118.058 | 123.296 | 176.751 | 9.198 |
| required | 59.052 | 58.907 | 60.097 | 63.362 | 77.485 | 16.934 |

Required whole-loop replay reduces mean latency by 45.8777% relative to
same-artifact ordinary execution. This is the execution-layer replay path and
is not an Agent kernel, memory or board result.

## Evidence

- Independent local audit:
  `artifacts/edgefm-vla-goal/20260906-044509/h100-pi05-aten-recovery-20260909/pi0-h100-independent-formal-audit.json`
  SHA `e69b7c99420ecade23c87a2ac4ac8835534ffa697eff06efcece8d7684eaec03`.
  It rechecked 278 local frozen files, all 17280 raw output calls against
  direct/eager references, process maps, aggregate table/CDF and latency.
- Aggregate report SHA `d605e307029847115ca3200ad5965d3fc885798cab705203c345fa16d9ba2b8d`.
- Latency table SHA `f381eb2ea38da65f4e0abc9e5b1e93cebda65ae8f3d7f241ae7fb2cc026d1b7c`.
- Figure report SHA `1cd52eee495da4215b6f81272f50a06cb4301137bfc858b42428be30d12139c6`;
  PNG SHA `be6d7adf56d85d22afaf42a65b3ac3001b19804abed9f0eb32ef68ee05c1ea71`,
  PDF SHA `4466321a89a9539a3450f03743d66c491f9e246ef6ada63f210d154cd820697a`.

## Scope

This is resident model-tensor normalized output, with init/H2D/D2H and
preprocessing excluded. H100 pi0 physical/double-output native validation,
three-policy physical output and Orin/BPU cells remain open.
`full_paper_acceptance=false`.
