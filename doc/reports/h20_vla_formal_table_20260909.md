# H20 VLA Session Comparison

Five scoped H20 campaigns, fifteen policy rows, 76,800 measured calls and
224,640 complete output tensors were independently rechecked from archived
data. Every policy uses five processes, 128 warmups and 1024 measured calls per
process. This is a resident model-tensor Session comparison, not a sensor-to-action
or Orin/J6M table. No cross-model speedup or pooled cross-model CDF is computed.

| Model | Off Mean ms | Shared Stream Mean ms | Replay Mean ms | Replay p99 ms | Replay Fresh Chunks/s |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 117.632 | 118.380 | 47.113 | 49.798 | 21.226 |
| RDT-1B pipeline | 176.711 | 172.734 | 164.398 | 168.415 | 6.083 |
| pi0 | 139.629 | 139.362 | 84.651 | 86.143 | 11.813 |
| pi0.5 | 180.263 | 169.153 | 95.344 | 113.096 | 10.488 |
| CogACT Base, public-dependency profile | 84.870 | 86.191 | 76.231 | 78.861 | 13.118 |

Complete normalized and native-scale outputs match each campaign's official-code
and same-artifact references byte for byte. CogACT additionally checks raw actions,
the RNG receipt and draw count, retaining exact integer comparisons. The numeric
agreement applies to the recorded workload and profile. It does not establish
robot task accuracy or universal floating-point equivalence.

The off baseline is ordinary execution of the same compiled C++ Session.
It is not a separately measured native-PyTorch or vendor baseline. The boundary
includes model execution, output processing and synchronized Session completion;
input preprocessing, H2D/D2H, initialization, validation/logging and robot transport
are outside the timed interval. The original per-campaign boundary is retained in
the CSV. RDT includes online T5, vision and its full DPMSolver path.

## pi0.5 Closure

The new H20 row is from `pi05-dual-recovery-20260909`, not the previously
misclassified RTX 3060 result. Its controller 4125122 completed all fifteen
workers with exit 0. Full independent audit checked 17,280 calls, including
warmups, and 34,560 complete F32/F64 tensors. Required records ten captured
steps, 1152 replays and zero ordinary fallbacks per process.

Replay mean latency is 47.1082% lower than the same-campaign off mean. Off/replay
standard deviation is 24.307/3.076 ms; off p99 is 300.240 ms. All long-tail
samples remain. Sampled device memory increases from 7397 to 8185 MiB with replay,
an increase of 788 MiB. These observations do not support zero jitter, a universal
sub-1.2 ms deviation, hard real-time execution or memory savings.

The host was shared. In later parts of the campaign, a small operator experiment
ran on a different GPU; CPU/memory/NAS exclusivity is not claimed and no contention
effect is inferred or subtracted. Sixteen frames come from one ALOHA episode.

## Artifacts

Current-worktree root:
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/`.

- `h20-five-summary/cuda-session-table.csv`: all fifteen rows, UUIDs, driver,
  measurement boundary, output metrics, memory and source audit digests.
- `h20-five-summary/report.json`: source file bindings and per-campaign rechecks.
- `h20-five-index.json`: exact locations and hashes of the five original audits.
- `pi05-formal/prepared/runs/`: complete raw samples, outputs, maps and telemetry.
- `pi05-formal/figures/latency-cdf-tail.png` and `.pdf`: actual CDF/tail figure,
  generated from the audited raw samples; PNG visually inspected for content,
  legibility and clipping. Each policy contributes 5120 measured points.
- `pi05-formal/figures/complete-table.csv` and `complete-output-metrics.json`:
  complete pi0.5 timing and per-output fidelity tables.

Hashes:

- Five-model summary: `799d9b6625061793861bfcc68f6ab543732f1406b6d51eeff660857ec59c63f0`.
- Five-model CSV: `c3ea57c739e5b910acc4d68c6ac7a0ad69fd043ab226ec9d2561fca03e5852f4`.
- pi0.5 formal audit: `772c3fd98e2dfc478df6ce7259af16d81ade9e0d97e974e69bff603ff0dd0849`.
- pi0.5 figure report: `06d3367111f361f70bc7774d1896b95ea9cc288c8d104f9edeae52e931376d27`.

Large model payloads remain on NAS; this local evidence archive is not a complete
runtime bundle. Older campaigns and failure evidence remain unchanged.

## Remaining Scope

Full-input measurements and matching official/vendor performance baselines remain
open. SmolVLA uses a recovered statistics profile; RDT and OpenPI each use one
episode's frames. CogACT retains its public-dependency configuration and external
random tape; its installed size is about 7.63B elements, not the nominal 3B label.
These limitations are not removed by successful timing or byte checks.
Operator confirmation and Qwen3.5 formal/deployment work remain separate tasks.
Orin/J6M are deferred as requested and do not receive H20 values.
