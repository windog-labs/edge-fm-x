# pi0.5 Multi-Observation Replay

Platform clarification (2026-09-09): this campaign ran on **RTX 3060**, UUID
`GPU-ce878329-6b58-5666-292d-94185f5e5585`. Both the original protocol and
`prepared-002/runs/00-off/telemetry.jsonl` identify that device. These values
do not establish H20 coverage; the later summary labeling them H20 was wrong.

Status at 2026-09-08 06:35 UTC: fresh off / batch-only / required bundles,
multi-observation pilots, all fifteen formal workers, independent formal audit,
table and inspected CDF/tail figures have completed. The existing 04:34 UTC
controller was resumed without launching a duplicate. All model workers have
exited. This does not complete the available-hardware stage or final paper
acceptance.

Root: `/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/local-pi05-replay-20260908-001/`.
Actual prepared directory is `prepared-002/`, not the failed first `prepared/`.

## Frozen Source

The existing `local-pi05-formal-003/protocol.json` supplies all 16 real ALOHA
frames, saved noise, full normalized F32 `[1,50,32]` and official native F64
`[1,50,14]` references. It is SHA256
`ec00a306f5071a357b193fe8ef05ac793ff78a69fd140d7353b9f4977e33df6e`.
The selected source bundle is `local-pi05-native-output-session-003/bundle`, with
manifest SHA `972b91fc8bbf457a65ebec2acb5dd5a366fbf10aaf4b0dc993e5c4944d59252d`.

The local source was rsynced into this new isolated directory after a dry run.
All 292 implementation hashes are frozen in `variants/source-files.json`.
The public `build_session_variants.py` reconstructs all three policies without
changing AOTI archive, effect, numerical-policy or complete I/O contracts.
No model-name dispatch was added to runtime, IR, Plan or the benchmark tool.

The only source-protocol adjustment is a new task-private AOTI extraction root:
`/dev/shm/edgefm-pi05-replay-20260908-001`. The first prepare failed because this
owned empty directory was created with 0755 rather than the required 0700.
Its failed receipt/log and incomplete output remain unchanged. The recovery
controller verified ownership, no symlink, emptiness and no local GPU owners,
then changed only that directory to 0700. It reused the completed bundles and
prepared into a new directory; no failed stage was relabeled successful.
The initialization helper now explicitly requests mode 0700 for future task
directories. Its original used version remains in `controllers/` with the
matching launch-receipt SHA; the frozen model/runtime source is unchanged.

## Real Result

Each policy ran 16 application warmups + 32 measured calls over all 16 source
frames. All 144 complete calls / 288 full output tensors / 331200 values were
byte-exact against both original official and same-artifact references. MSE and
max-abs are zero; minimum computed cosine is 0.9999999999999998. This is exact
storage agreement on these observations, not a robot task-success proof.

| Policy | Native PID | Pilot mean ms | Runtime loop counter |
| --- | --- | --- | --- |
| off | 1924223 | 360.067169 | No replay counter |
| batch-only | 1924472 | 357.210701 | 0 capture / 0 replay / 48 ordinary |
| required | 1924730 | 347.130829 | 10 steps captured / 48 replay / 0 ordinary |

Required emits `REPLAY_FINAL,13,1,10,48,0`. These pilot means are not formal CDF
or stable speedup claims. Unlike the earlier single-observation three-call test,
this pilot covers every source frame with the complete checked output stage.

Independent audit: `independent-pilot-audit.json`, SHA256
`d33b2d488ff32d11798269327db8cc1fd0879e53c519f1cc31094faecd4cea5c`.
It verifies frozen source and archives, untouched references/noise, raw complete
outputs, actual runtime DSO mappings and numerical bootstrap, independent worker
PIDs, GPU ownership, telemetry, timing samples and all replay counters. Actual
model-process maps contain no Python runtime. The six real-evidence audit tests
also pass, rejecting incomplete/corrupt/missing native F64 output, failed process
and missing replay telemetry. XML SHA256:
`914f9ac72c15eeb3ce4afa1c3e412b05ee157154d0e55c7f5f3fad4c019deeec`.

## Formal Result

Each policy completed five independent processes, each with 128 application
warmups and 1024 measured calls, balanced over all 16 frames. The 15360 formal
latency samples exclude the pilot and warmups. Output checks include all 17280
calls: 34560 complete tensors / 39744000 values are byte-exact against both
original official and same-artifact references. MSE and max-abs are zero;
minimum computed cosine is 0.9999999999999998. Exact agreement is limited to
these observations and this precision, not universal or task-success proof.

| Policy | Mean ms | p50 ms | p95 ms | p99 ms | Max ms | Std ms | Chunks/s | Sampled peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| off | 362.055789 | 362.087333 | 363.249484 | 363.738000 | 364.551185 | 0.769327 | 2.762005 | 7506 |
| batch-only | 360.010064 | 360.032267 | 361.218244 | 361.657842 | 362.688290 | 0.757806 | 2.777700 | 7524 |
| required | 350.053228 | 349.993310 | 351.265406 | 351.884728 | 358.753161 | 0.774295 | 2.856708 | 7626 |

Required reduces mean latency by 3.315114% relative to same-source off; sampled
memory increases by 120 MiB. Each required worker emits
`REPLAY_FINAL,13,1,10,1152,0`, with no ordinary fallback. This is execution-policy
evidence, not an Agent-selected operator gain or a memory-saving result.

Independent formal audit SHA256:
`b50b40f7c889e56a5186987815c91a0556dbb7e5d8fd8c7e096e08f516641f11`.
It rechecks all raw tensors, each CDF rank, aggregate statistics, runtime DSO
mappings, numerical bootstrap, frozen inputs/source/archives, process ownership,
telemetry and replay counters. `formal-closeout-001.json` pins the completed
controller, audit, report, table and figures; `formal-figures-001/` contains
`complete-table.csv`, `complete-output-metrics.json`, and `latency-cdf-tail.png`
and `.pdf`. The PNG was actually viewed and checked for clipping/blank content.
Do not repeat the completed create-only report or formal launch commands.

## Remaining Boundary

Sensor preprocessing, H2D/D2H and
robot transport remain outside the resident model-tensor timing boundary; one
episode's 16 frames are not 16 independent episodes. No vendor baseline, board
run, autonomous RNG, low-bit or physical calibration result is inherited.
Next work is CogACT's complete native replay gate, then the remaining G1/G4/G5
items. Current GPU availability must be rechecked before any later launch.
