# H20 Host Tensor Formal Results

All five models completed three policies with five independent processes per
policy, 128 warmups and 1024 measured calls per process. All 75 workers, remote
independent audits, retrieved complete-output checks and the shared five-model
summarizer passed. There are 76800 measured latencies and 224640 complete output
tensors including warmups. No measured outlier was removed. All five CDF/tail
PNGs were inspected; PDF and CSV counterparts are retained.

The measured interval includes pageable host model-tensor H2D, fresh binding,
Session run/completion and every declared output D2H. It excludes original RGB
preprocessing, file IO, initialization, reference comparisons and log IO.
This is an additional host-model-tensor result, not original-input E2E or board
performance. The earlier resident-tensor table retains its original meaning.

| Model | Off Mean ms | Batch-only Mean ms | Required Mean ms | Required p99 ms | Required Chunks/s |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 121.939924 | 120.539683 | 47.810259 | 48.941977 | 20.916013 |
| RDT | 177.103208 | 172.187518 | 164.879751 | 166.872083 | 6.065026 |
| pi0 | 143.922713 | 139.762972 | 86.381092 | 90.053202 | 11.576607 |
| pi0.5 | 169.710285 | 177.257704 | 94.886679 | 97.238034 | 10.538887 |
| CogACT | 86.785191 | 85.320263 | 76.242278 | 77.524266 | 13.116082 |

Throughput counts fresh full action chunks, not individual robot control steps.
Each policy uses 5120 measured calls. Complete normalized and native output
bytes match both scoped references: MSE and maximum absolute error are zero;
computed cosine minima are within double-precision rounding of one. The CSV
retains exact values. CogACT also verifies three additional typed outputs.
Native RDT outputs remain BF16, OpenPI and CogACT native outputs remain F64;
there is no lossy F32 serialization of the comparison data.

GPU ownership is exclusive per monitored worker, but host CPU/NAS exclusivity
is not claimed. Several non-replay distributions contain visible process groups.
The distributions and all process means remain intact; this is not evidence of
an interference-free platform limit. SmolVLA retains its recovered statistics
profile. CogACT retains the public-dependency, external-RNG-tape profile and
approximately 7.63B actual parameter elements; it is not the original Meta
configuration or a nominal 3B model. Input coverage remains the archived recorded
observations, not a robot success-rate or general quality benchmark.

## Evidence Locations

Remote host: `zzm-h20-x8-2`.
Remote root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/host-tensor-20260909-001/`.
Each model directory contains `protocol.json`, `lineage.json`, controller records,
`prepared/` raw inputs, all outputs, worker PIDs, telemetry, source/environment
bindings and `independent-formal-audit.json`. The frozen 298-file source snapshot
and source provenance are in the remote root; this snapshot was not overwritten.

| Directory | Controller / Supervisor PID | GPU UUID |
|---|---|---|
| smolvla | 33468 / 33501 | GPU-ae321531-504c-f416-e693-94fa02779130 |
| rdt | 35605 / 35626 | GPU-c66f42bd-8a1c-423b-e2eb-d0cc72b9fe0b |
| pi0 | 33483 / 33583 | GPU-3596876b-e679-6289-f784-c8d98294ed23 |
| pi05 | 31003 / 31032 | GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5 |
| cogact | 39618 / 39647 | GPU-245ec313-eeab-8b25-b02b-047d7e91e31d |

Every controller/supervisor above is terminal with success, not a running job.
Local root:
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/`.
The five `host-<directory>-formal/` archives contain complete data, source metadata,
remote audit, `local-output-audit.json` and `figures/latency-cdf-tail.{png,pdf}`.
Large native payloads stay on NAS and are checked by the bound remote audits.
`host-five-index.json` binds these audits; `host-five-summary/` contains the full
15-row `cuda-session-table.csv` and its recomputed source-bound `report.json`.

| Model | Remote Formal Audit SHA256 | Local Complete-Data Audit SHA256 |
|---|---|---|
| SmolVLA | `653f1e00cab0a501610407a5c62bd652e444a501e5fa3892fbdc6ee13722d1af` | `73dc957a1a4daaf862e7db2a60889e584f6030772d218a67cf5ec60159d107f3` |
| RDT | `f3ae7894a79cb7938d32cb8ef5bc8cf7ae203d93c9be9a6eb7ee8c9298fa1c7a` | `e2c57ab7fbd0e1da18daccd962ecfb19c3c1cf17a1f32910359f307b56aa2c49` |
| pi0 | `17f0a41af9d1415aa7aaa6a3c2940651749b3fd39a0e48e1a612a9864ed4a55e` | `ed037c0da2c511862e75b051ac8cc05b5f01554e39f8bfa107610d383ac2ade7` |
| pi0.5 | `0ad60ef1574d3f590d22f16a7df8f7d03bc7ade70b0edff24796d4db0ba2a2ef` | `304939d867a4039106086901c26e7b180d8cc535f44583b93c4f4c66ba808d65` |
| CogACT | `4583a4ba251806db52f644185a5c740557230da6e57bca5d02fa464f93d12f7b` | `dc1f75fe2442ebd37894dccc7e5ba77331e56b92b7df5f6fed9d885d2e65247c` |

Summary SHA256:
`6a1365250262540895d33443fc674b1a7edf54b5495d96821b79c5d66f4fd694`.
CSV SHA256:
`df39ba0503a4298bfe4caf4491d678fc2284b625a8f9cb8bdf0553235242ef9b`.

## Remaining Boundary

Original RGB/state/text preprocessing and matched official implementation timing
are still pending formal measurement. In a separate `raw-input-20260909-001`
experiment, pi0.5 native and official Python host paths each passed 48 pilot
calls over 16 recorded observations with all inputs and both outputs byte exact.
Native controller/worker PIDs are 73129/73144, official PIDs 74019/74022, on the
same GPU1 UUID above. These pilots are excluded from this formal table and still
need independent audit. The Python host path is explicitly not a no-Python
deployment certificate. Orin/J6M/BPU remain deferred.
