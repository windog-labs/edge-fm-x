# SmolVLA Orin Soak, Memory and Power

Date: 2026-10-21, Asia/Shanghai. This supplements the
[formal E2E results](orin_smolvla_formal_e2e_20261021.md), without replacing them.
The five-model Goal remains incomplete. No production code, engine, precision,
weights, input/noise, solver or system clock/power configuration changed.

## Continuous Worker

The exact source-qualified replay binary and seven engine hashes used in the
formal experiment ran as one process, PID `618893`, start ticks `3148182`.
Every request starts from CPU decoded RGB/state/text and saved noise, includes
online native preprocessing, transfers, fresh context and all ten solver steps,
and finishes with complete CPU normalized/native actions. The fixed 16-record
cycle is not an independent trajectory dataset.

| Metric | Observed Value |
|---|---:|
| Independent worker processes | 1 |
| Warmup / measured requests | 128 / 18432 |
| Worker elapsed time | 2100.936864 seconds, 35.016 minutes |
| Sum of measured ModelRun + completion time | 2022.466141 seconds, 33.708 minutes |
| Mean / p50 E2E | 112.835238 / 112.710880 ms |
| p95 / p99 / max E2E | 114.775334 / 115.975244 / 121.945920 ms |
| E2E standard deviation | 1.060070 ms |
| Request service rate | 8.862480 chunks/s |
| Complete action tensors byte exact, including warmup | 37120 |
| Measured complete action tensors | 36864 |
| Replay / ordinary fallback | 18560 / 0 |

ModelRun time includes host launch work and actual completion waits; it is not
an isolated CUDA-kernel busy-time measurement. Initialization, file decoding,
output checking and evidence serialization stay outside the E2E boundary.
All timing rows and output bytes are retained; no slow rows are removed.

This one-worker standard deviation is below 1.2 ms for this workload/window.
The independent five-process formal replay standard deviation remains
**2.034434 ms**, and the formal same-engine speedup remains **1.187995x**.
These different statistics are reported together, not substituted or centered
to make the formal claim pass. The original BF16 H20 supplemental MSE remains
failed; exact equality here is against the source-qualified target baseline.

There are 1959 resource snapshots, 1941 after warmup and 66 visible-owner scans.
Every sampled GPU frequency is 1300500000 Hz; the highest valid thermal sensor
reading is 58.843 C. This does not rule out unobserved sub-sampling transients.
The board civil clock is incorrect; durations and alignment use monotonic time.

## Observed Memory

| Post-Warmup Process Metric | First | Last | Min | Max |
|---|---:|---:|---:|---:|
| RSS (KiB) | 2623448 | 2623448 | 2623448 | 2623448 |
| Anonymous RSS (KiB) | 1784764 | 1784764 | 1784764 | 1784764 |

This is a sampled envelope, approximately 2561.961 MiB RSS and 1742.934 MiB
anonymous RSS, not a proof of zero heap allocations. Largest sampled process
VmHWM is 3113.875 MiB. These shared-memory Jetson values are **not** added to
CUDA requested bytes or interpreted as an isolated physical GPU peak.
Inaccessible process FD directories remain an ownership-coverage limitation.

### Destruction Controls

The prior ambient CUDA-free-memory decline was reproduced with an explicit
red-capable observation command. It detects the original symptom, not a leak:

```bash
python3 artifacts/orin/20261021-00/check_ambient_lifecycle_drop.py \
  --csv artifacts/orin/20261021-00/qualified-fault-replay-001/results/lifecycle-memory.csv \
  --report artifacts/orin/20261021-00/new-memory-observation.json
```

The existing observation returns 2 with consecutive deltas -84157440 and
-85751040 bytes. The new three-cycle controls reuse the original full-model
fault/reset/isolation function and add process RSS, smaps and structured
`malloc_info` snapshots. Top-level glibc totals are parsed without summing
duplicate per-heap totals. All 180 complete action tensors remain byte exact.

| Control | RSS Change After First Cycle (MiB) | Ambient CUDA-Free Change (MiB) |
|---|---:|---:|
| Idle, no Sessions | 0 | -6.40 |
| Original paired-Session cycles | +4.34 | -176.94 |
| Same cycles with diagnostic post-destruction trim | +1.07 | -178.50 |

Diagnostic `malloc_trim(0)` recovers about 351.56--354.21 MiB RSS per cycle,
but does not stop the ambient decrease. The idle and inference windows have
different durations and are not a quantitative causal subtraction. These data
show a CPU allocator-cache component and an attribution gap; they do not prove
either a complete leak or complete leak freedom. Trim was **not** added to
production execution or used to alter the formal/soak measurements.

### Covered CUDA Resources

A separate diagnostic library intercepts listed outermost CUDART allocation,
graph and stream APIs. Its real 1 MiB allocate/free positive control passes.
It adds observation markers to private probe sources, not production code.
Diagnostic timings are not used as performance results.

- Twelve paired-Session lifecycle cycles: 24 Sessions, 180 complete model calls
  and 72 rejected nonfinite-input transactions. All 360 complete action tensors
  remain byte exact, including retained publications and recovery.
- Tracked device acquisitions/releases are 217/217, including the self-test;
  no tracked device bytes remain after destruction.
- Graph and graph-executable acquisitions/releases are 36/36 each; no tracked
  graph handles remain. All observed categories have zero post-first-cycle
  growth, errors, duplicate acquisitions and unknown releases.
- Two shared streams remain constant through all twelve cycles. A separate
  one-Session raw-input probe retains one. This matches the existing exclusive
  lease pool, whose idle streams intentionally live to process exit and are
  bounded by maximum simultaneous leases, not Session lifetime count.
- The frozen native `execution_context.cpp` matches the current source SHA256
  `ae52b5a82fcab7b73827e76ebebb8a042efc812f9bee9e9b2055a67d37c466e0`.
- In a separate 128-warmup/1024-measured raw-input run, every measured ModelRun
  has zero observed covered allocation/free, graph or stream creation/destruction
  events. All 2304 complete action tensors, including warmup, remain byte exact.

These hooks do **not** exhaustively cover private driver/SDK allocations,
general CPU malloc, memory pools or module caches. The small tracked requested-
byte peak must not be published as total model memory. Full private-SDK/driver
attribution of the remaining ambient decline stays open. No unproven allocator
fix, forced device reset or per-request process restart was introduced.

## Power Window

Direct INA3221 sysfs reads were permission-denied, and `sudo -n` required a
password. The existing active `jtop` 4.2.3 daemon provided a read-only telemetry
path without changing permissions or hardware controls. Only the observed late
window is reported: **338 samples over 337.215 seconds**, at worker elapsed
1763.068--2100.283 seconds. Earlier power samples are not reconstructed.

| Rail / Estimate | Mean (W) | Min (W) | Max (W) |
|---|---:|---:|---:|
| VDD_GPU_SOC | 28.401657 | 27.747 | 29.214 |
| VDD_CPU_CV | 2.847337 | 2.657 | 3.863 |
| VIN_SYS_5V0 | 10.552902 | 10.366 | 10.748 |
| Selected non-overlapping input-rail sum | 41.801896 | 40.870 | 43.003 |

The SDK guide documents that VIN_SYS_5V0 already includes VDDQ_VDD2_1V8AO.
That overlapping DDR rail and raw `jtop` total are therefore excluded from this
sum; voltage/current/power units and rounding are checked against real samples.
[NVIDIA R35.4.1 power monitoring](https://docs.nvidia.com/jetson/archives/r35.4.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling),
[jtop power API](https://rnext.it/jetson_stats/reference/jtop.html#jtop.jtop.power).

This is an input-rail estimate including background services, not wall-plug
power, exclusive inference energy, a new power mode or a matched-baseline
efficiency comparison. The daemon running-average fields are not used.

## Evidence and Next Work

All roots below are local under `artifacts/orin/20261021-00/`; original target
copies remain under `/data/datalog/edgefm-orin/20261021-00/`.

- `smolvla-raw-soak-001/{campaign,protocol}.json` and `audit-output/audit.json`.
- `smolvla-raw-soak-001/audit-output/smolvla-soak-{trace,cdf}.{png,pdf}`.
- `memory-boundary-controls-001/{campaign,audit}.json`, every XML/proc boundary
  snapshot and complete action outputs.
- `native-allocation-audit-001/{campaign,audit}.json`, `cuda-api.csv` per probe,
  native diagnostic binaries/module hashes and complete action outputs.
- `smolvla-soak-jtop-power-001/{collection,audit}.json` and `power.jsonl`.
- `observation-diagnostics-tests-003.xml`: eight CPU checks passed; the C++
  ledger test, host syntax checks and real target diagnostic builds also passed.
- `soak-allocation-power-regression.xml`: all 17 combined collection/audit checks
  passed after the follow-up.

Independent audit SHA256 values:

| Audit | SHA256 |
|---|---|
| Continuous worker | `f0d0b4fd818353de78cff25d340a6faf2f7a8930519e42713f62d1bd1919deb5` |
| Covered allocation/resources | `c75bd717b7176f6cc4dfb89b80e46f6d8ca85a7300fef3d7e51044763d9a6dfe` |
| Partial power window | `34e58b6f2719f70a99a0a4c607659d3553374b23c96e3732fd124eafb621290a` |

The formal and soak collectors preserve original worker identity, commands,
inputs/noise/engine hashes, phase results and raw logs. New output paths are
required for reproduction; evidence is never overwritten. The observation-
only and source-qualified-reference limits are intentional, not completion of
all manuscript claims.

For example, after checking current board availability, use a **new** root:

```bash
ssh orin python3 /data/datalog/edgefm-orin/20261020-00/collect_qualified_raw_soak.py \
  --root /data/datalog/edgefm-orin/20261021-00/NEW-SOAK \
  --formal-campaign /data/datalog/edgefm-orin/20261021-00/smolvla-raw-formal-001/campaign.json \
  --formal-audit /data/datalog/edgefm-orin/20261021-00/smolvla-formal-audit-host.json \
  --pilot-campaign /data/datalog/edgefm-orin/20261020-00/qualified-raw-e2e-pilots-001/campaign.json \
  --source-gate /data/datalog/edgefm-orin/20261020-00/source-gate-layernorm-solver-full16-001.json \
  --observations /data/datalog/edgefm-orin/20261020-00/native-observations-001 \
  --environment-probe /data/datalog/edgefm-orin/20261020-00/capture_environment.py
```

The read-only power client must start while that worker is actually live, with
its own new root and `--soak` pointing at the new collection. It stops when that
exact PID/start-time/argv identity disappears. The original main experiment's
three-mode protocol, outputs and metrics must not be overwritten by either run.

Next: RDT complete-model export/TRT support and deployment, then pi0, pi0.5 and
CogACT; same-platform operator/Agent ablations and final claim reconciliation
remain required. Qwen remains P1 and secondary power modes remain after the
five primary model configurations.
