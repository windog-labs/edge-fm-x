# SmolVLA J6M Stage-060

Date: 2026-09-11

## Scope

This report contains the completed SmolVLA J6M work requested in the current
round. The only measured model is SmolVLA. RDT, pi0, pi0.5, CogACT, and Qwen are
not run and are outside this report. Orin and `j6m-1` are deferred. All board
measurements use `j6m-2` at `root@10.1.200.209`.

Stage-060 is one SmolVLA action-solver step. It is not a complete prefix plus
N-step solver plus action-trim pipeline. The row is suitable as stage-level
deployment and ablation evidence, not as a complete end-to-end VLA main-table
result.

## Build

| Item | Value |
|---|---|
| Stage | stateful SmolVLA action-solver step |
| Body precision | FP16 |
| Conv/MatMul precision | INT16 |
| Preserved FP32 operators | Softmax, Where, Sin/Cos, ScatterND, and remaining CPU fallbacks |
| Compiler fusion | `skip_fuse_to_hzswish` |
| Target | `nash-m` |
| HBM | `artifacts/j6m-feasibility-20260911/smolvla-stage-060-horizon-nash-m-fp16-linear-int16/horizon-nash-m-fp16-linear-int16/smolvla-step-fp16-linear-int16.hbm` |
| HBM SHA256 | `457c50eb182d7694c3e6b5d79215f9a01208e0488497148c98e2133f840ca7b3` |
| HBM size | 108642048 bytes |
| Compile time | about 25.9 minutes |

The generic operator-level recipe is `--op-qtype Conv=int16` and
`--op-qtype MatMul=int16`. It is not SmolVLA-specific graph surgery.

## Stage Performance

The five-frame profiler result is:

| Metric | Stage-057 FP16 | Stage-060 FP16 + INT16 Conv/MatMul |
|---|---:|---:|
| Mean latency | 8893.975 ms | 90.088 ms |
| Throughput | 0.1124 steps/s | 11.062 steps/s |
| BPU time | 84.503 ms | 31.038 ms |
| CPU fallback time | 8802.582 ms | 57.288 ms |
| HBM size | 410331920 bytes | 108642048 bytes |
| BPU nodes | 425 | 573 |
| VPU nodes | 790 | 573 |
| External CPU nodes | 324 | 384 |

Stage-060 is 98.73x faster than stage-057 on the same `j6m-2` board. All 116
`Conv` and 32 `MatMul` nodes moved from the external CPU path to BPU. The
remaining external CPU work is dominated by 234 cast nodes, 48 `ScatterND`
nodes, 25 `Sin` and 25 `Cos` nodes, and 16 each of `Where`, `Softmax`, and
reduction nodes.

## Continuous Latency CDF

Protocol: five independent `hrt_model_exec infer` processes, 128 warmup calls
per process, 1024 measured calls per process, one thread, no outliers removed.

| Process | Mean ms | p50 ms | p95 ms | p99 ms | Max ms | Std ms |
|---|---:|---:|---:|---:|---:|---:|
| run-1 | 89.683561 | 89.603 | 91.103 | 95.613 | 96.293 | 1.232466 |
| run-2 | 89.654554 | 89.585 | 90.900 | 95.583 | 96.166 | 1.133623 |
| run-3 | 89.750358 | 89.644 | 91.066 | 95.352 | 96.512 | 1.161144 |
| run-4 | 89.796083 | 89.706 | 91.039 | 95.315 | 96.223 | 1.099083 |
| run-5 | 89.805693 | 89.713 | 91.208 | 95.848 | 96.937 | 1.187700 |
| pooled | 89.738050 | 89.645 | 91.068 | 95.593 | 96.937 | 1.165245 |

Evidence:

- Summary: `artifacts/j6m-feasibility-20260911/smolvla-stage-060-horizon-nash-m-fp16-linear-int16/board-evidence/latency-summary/summary.json`
- Raw measured samples: `.../latency-summary/latency-measured.csv`
- Exact-rank CDF data: `.../latency-summary/latency-cdf-all.csv`
- Figure: `.../latency-cdf-figure/latency-cdf.png` and `latency-cdf.pdf`
- Figure report: `.../latency-cdf-figure/report.json`

This CDF measures repeated calls to the same action-solver step. It is a
latency-jitter result, not a sensor-to-action end-to-end distribution.

## Fidelity

The board action tensor was compared with the same H20 reference used by the
prior SmolVLA stage evaluations:

| Metric | Value |
|---|---:|
| Action cosine similarity | 0.9999994983908165 |
| Maximum absolute error | 0.003674030303955078 |
| MSE | 8.980497730075133e-07 |
| RMSE | 0.000947654880749059 |
| Step-index exact | yes |

The strict local float gate `max_abs <= 0.01` and `cosine >= 0.999` passes.
The output is not bit exact. The eight held-out ONNX samples in stage-059 have
action cosine similarity from 0.99997918 to 0.99999752, and every step index is
exact.

Evidence:
`artifacts/j6m-feasibility-20260911/smolvla-stage-060-horizon-nash-m-fp16-linear-int16/reference-parity/report.json`.

## Operator Ablation

The reusable fixed-shape operator probes give the same-platform before/after
view. FP32 is a reference baseline only and is not a deployment candidate.
The optimized column is the calibrated INT8 probe recipe.

| Operator family | FP32 ms | INT8 ms | Speedup | CPU fallback FP32 -> INT8 |
|---|---:|---:|---:|---:|
| Attention core | 29.124 | 0.258 | 112.88x | 5 -> 0 |
| GEMM + LayerNorm + Embedding | 7.385 | 0.315 | 23.44x | 4 -> 1 |
| RMSNorm | 0.875 | 0.433 | 2.02x | 0 -> 0 |
| RoPE | 0.243 | 0.267 | 0.91x | 0 -> 0 |
| Conv + vision | 133.216 | 0.413 | 322.56x | 5 -> 0 |
| Flow-matching step | 0.700 | 0.276 | 2.54x | 2 -> 0 |
| Gated DeltaNet-style step | 0.560 | 0.313 | 1.79x | 2 -> 0 |
| F64 state/output | unsupported | unsupported | - | - |

These probes use synthetic fixed-shape inputs and deterministic calibration.
They prove lowering and placement behavior, not task-quality preservation.
RoPE is a negative result: quantized lowering is slightly slower than the FP32
probe. The actual stage-060 optimizer in the model is FP16 plus INT16
Conv/MatMul, not the separate exploratory INT8 probe recipe.

Source:
`artifacts/j6m-feasibility-20260911/summary-001/operator-support-matrix.md`.

## Negative Boundary

Stage-061 extended INT16 to `Softmax` and `Where`. It failed fidelity:

- cosine similarity: 0.3523425661058557
- maximum absolute error: 7.449563026428223
- MSE: 7.706983383874228

No HBM was compiled for stage-061. `Softmax` and `Where` remain out of the
deployment recipe.

## Full-Chain File Pilot

A `prefix -> 10 step -> finish` chain was executed on `j6m-2` using real
captured inputs. All 12 board executions completed and all output dumps were
parsed successfully:

`artifacts/j6m-feasibility-20260911/smolvla-stage-062-full-chain-20260911/board-chain/sample-000000/full-chain-v1`

The final action result is:

| Metric | Value |
|---|---:|
| Cosine similarity | 0.999903171999195 |
| MSE | 0.0003810012909923388 |
| RMSE | 0.019519254365685662 |
| Maximum absolute error | 0.050837501883506775 |
| Mean absolute error | 0.01609863852150738 |
| Strict gate | failed because max-abs and mean-abs exceed the declared thresholds |

The action cosine is close, but this is not a strict E2E pass. Prefix cache
parity against the H20 trace is the main source of error:

| Prefix metric | Value |
|---|---:|
| Minimum cache cosine | 0.994239330291748 |
| Maximum cache absolute error | 1.310546875 |
| Padding mask exact | false |

The 148.238 s wall time for the complete file-chain command is not an E2E
latency result. It includes model uploads, per-stage SSH/file transfers, model
loads, and model dumps. The sum of the individual board command windows is
about 21.979 s, and even that sum is a file-orchestration pilot rather than a
resident runtime measurement.

## Lowered Prefix Progress

The first lowered Prefix (V2) was compiled and profiled on `j6m-2`:

| Metric | Value |
|---|---:|
| Prefix latency | 1405.032 ms |
| BPU time | 186.637 ms |
| External CPU time | 1215.067 ms |
| `ScatterND` external CPU time | 811.092 ms across 63 nodes |
| `Resize2dBilinearNHWC` time | 221.687 ms |
| HBM size | 462692200 bytes |

V2 is `3.12x` faster than the original Prefix but is not on the critical path
for the final chain because static Scatter remains the largest fallback. A
generic V5 lowering probes each `ScatterND` index tensor and rewrites only
statically verified continuous blocks as `Slice + Concat`. V5 removes all 63
`ScatterND` nodes and passes host parity. Its H20 HMCT `batch=1` output has
SHA256 `262d842a591b29d359ff26cf069e54252296cb6e0bd6184650b1b29d0285c6de`.
HBM compilation also passed after `5091.3 s`; the resulting HBM is
`454521544` bytes with SHA256
`b05a1c4d62d29b37e29f7a347f7003440b73cbdbe6d76b44d67c1491adf30993`.
Offline `hbrt4-disas` reduces the HBM node count from V2's `232` to `64`,
CPU nodes from `157` to `44`, and BPU nodes from `75` to `20`; V5 has no
`InplaceScatterND` or `Pow` nodes. The detailed counts are recorded in
`.../prefix-lowered-v5-b1/offline-placement-summary.json`.
The replacement-chain board run remains pending. At `2026-09-11 22:10 CST`,
`j6m-2` was unreachable from the workstation and both H20 hosts, so no board
result is claimed for V5.

## Remaining Work

The SmolVLA step-level task is complete and the file-chain pilot has executed.
A complete SmolVLA J6M E2E row still requires the lowered Prefix HBM, a new
complete-action fidelity pass, and a resident four-stage runner for formal
latency and CDF evidence. Other VLA models remain deferred by the current user
scope.
