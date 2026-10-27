# Pi0 Formal Orin E2E

Status: **this pi0 native D1-D3 cohort passed the final joint audit**.
`formal_acceptance=true`; `whole_goal_complete=false`. This is not completion
of D4, the five-model Goal, source-BF16 equivalence or robot/task quality.

The authoritative [joint audit](../../artifacts/orin/20261023-42/joint-audit-001.json)
has status `complete-native-cohort-D1-D3-verified` and SHA256
`bb79a25109973dd5a0ea4e7c30ebc9901d7c40d84a6594e702a2b345af245d8c`.
Its eight input hashes match the archived final reports. Historical pilot and
failed/pre-worker cohorts are not merged into this result.

## Scope And Protocol

- Jetson AGX Orin serial `1421524312594`, boot
  `3c801f5d-6897-4d97-b021-3c52db38a750`; TensorRT 8.5.2.2, CUDA 11.4,
  L4T 35.4.1, unchanged existing MAXN, not relabeled 60W.
- Two modes: `ordinary-repeated-candidate` (conventional native TRT Session)
  and `verified-arena-candidate`. Both use the **same eleven frozen
  engine/provider resources**, shared bundle and byte-identical artifact
  contract SHA256 `0e3b85d27710a4c2a3394c5eda493f1b7b30097691c556ba0292a169e6c6060c`.
  The precision profile is unchanged between modes; this is not FP16 vs BF16.
- Both execute three fresh calls of the same source-proved single-view engine,
  the original language consumer and all ten action steps. No camera, text,
  neural iteration or F64 state/native finishing is omitted; no cached encoder
  features or request memoization are used. Public ModuleIO digest remains
  `8751fa709441f09883da726d2cc2d6b52f7a5894bf5876ee36b6803e2dff9575`.
- **No replay/capture optimization is claimed**: both mode declarations have
  `replay=[]`, and request records have `actual_replay=[]`. The only logical
  Plan difference is static arena layout: 205184256 vs 199307264 bytes, a
  5876992-byte reduction. This is not measured process/GPU peak reduction.
- Batch one, five independent processes per mode; each process has 128 warmup
  and 1024 measured requests. Serial order rotates O/A, A/O, O/A, A/O, O/A.
  Total: **10 processes, 11520 requests** = 1280 warmup + 10240 measured;
  each mode has 5760 requests, including 5120 measured.
- Timing is the unaltered driver `e2e_ns`: decoded CPU raw observations through
  online image/state/text/token/noise preparation, transfers/binding, full
  model/solver, finishing and complete CPU action chunks. Asset loading and
  raw decoding, initialization, output checking and logging are outside the
  request clock. No sum of stage times substitutes for E2E.
- Sixteen frozen observations and saved noise sets are cycled with fresh
  computation, not 11520 independent trajectories or robot trials. The named
  native processor profile is identical in both paths; this does not erase
  historical literal-CPU pixel/F64 preprocessing comparison failures.

## Formal Latency

All durations are milliseconds; each row pools all 5120 measured requests.
Percentiles use NumPy linear interpolation, population std uses `ddof=0`.
Chunks/s is `1000 / mean_ms`, not process-wall throughput or control frequency.

| Mode | Mean | p50 | p95 | p99 | Max | Std | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional TRT | 1129.869939 | 1127.602320 | 1163.024701 | 1170.859402 | 1177.885024 | 16.205243 | 0.885058 |
| Verified Arena | 1125.379221 | 1127.359920 | 1146.530686 | 1150.569614 | 1157.528480 | 13.313744 | 0.888589 |

Observed mean difference is **4.490718 ms**, ratio **1.003990404x**:
**0.399040% speedup**, or **0.397454% latency reduction**. This is a small
descriptive difference, not a demonstrated statistically significant benefit.
No significance test or independence assumption over repeated observations is
used to upgrade it. Both pooled standard deviations miss the draft **<1.2 ms**
target. Lower p99/max does not establish deadline success or all-metric jitter
improvement. No slow row is removed, mean-centered or smoothed away.

### Process Variation

Each entry represents one independent process with 1024 measured requests.
Paired speedup is `100 * (ordinary_mean / arena_mean - 1)`; negative rounds
are retained, not replaced with selected faster processes.

| Round | Serial Order | Ordinary Mean | Ordinary Std | Arena Mean | Arena Std | Paired Speedup % |
|---|---|---:|---:|---:|---:|---:|
| 0 | O then A | 1156.167364 | 11.011672 | 1137.453841 | 12.766085 | 1.645212 |
| 1 | A then O | 1120.921817 | 7.714021 | 1135.519644 | 3.680990 | -1.285564 |
| 2 | O then A | 1127.580677 | 8.138445 | 1117.761829 | 9.025830 | 0.878438 |
| 3 | A then O | 1121.891680 | 7.765179 | 1109.581797 | 3.129058 | 1.109417 |
| 4 | O then A | 1122.788156 | 10.727801 | 1126.578994 | 7.800790 | -0.336491 |

Ordinary process means span 1120.921817-1156.167364 ms; Arena spans
1109.581797-1137.453841 ms. Rounds 1 and 4 are slower with Arena, and round 0
has higher Arena within-process std. All per-process percentiles/maxima remain
in the [unaltered summary](../../artifacts/orin/20261023-42/latency-001/summary.json).

![Pi0 pooled raw E2E CDF](../../artifacts/orin/20261023-42/publication-curves-001/cdf-pooled.png)

![Pi0 per-process raw E2E CDF](../../artifacts/orin/20261023-42/publication-curves-001/cdf-processes.png)

The neutral-caption publication [receipt](../../artifacts/orin/20261023-42/publication-curves-001/receipt.json)
is `rendered-from-accepted-unchanged-rank-data`: all twelve pooled/per-process
rank CSVs and four PNG/PDF outputs are pinned; numerical data was not changed.
The older audited `latency-001` plot captions and receipt are left untouched.

## Complete Output Quality

The [whole model-data audit](../../artifacts/orin/20261023-42/model-data-audit-001.json)
checks **23040 full tensors**: 11520 normalized + 11520 native, including
all warmup outputs. All ten processes report all outputs finite and byte-exact
to the frozen **same-Orin conventional TRT reference**, not an H20 reference.

| Complete Output | Type / Shape | Tensor Count | Max MSE / Maxabs | Min Cosine | Byte Exact |
|---|---|---:|---:|---:|---|
| Normalized action | F32 `[1,50,32]` | 11520 | 0 / 0 | 1 | All |
| Native action | F64 `[1,50,14]` | 11520 | 0 / 0 | 1 | All |

The fixed primary gate remains complete normalized **MSE < 1e-5** and
**cosine > 0.9999**. All public outputs must be finite; native-unit errors are
supplemental. This cohort observes zero native error too, without turning that
observation into a relaxed/changed quality policy or a general losslessness claim.
Reference-index SHA256 is
`94b9b4fa8e859039d3c45073e1f0b8b1567534117452639ed6440f84d2f16ac5`.

### H20 Remains Separate

The preserved [offline H20 supplement](../../artifacts/orin/20261023-25/REPORT.md)
compares the sixteen matching original observations/noise sets, not additional
formal timing samples. It remains failed at the original normalized thresholds:

| Supplement | Complete Values | MSE | Cosine |
|---|---:|---:|---:|
| Normalized | 25600 | 0.06826529088 | 0.6685354882 |
| Native | 11200 | 0.004825036517 | 0.9907577346 |

Only **0/16** normalized observations pass both thresholds. Its audit SHA256 is
`714e6a185aa5c276979fbe51c103e4dab2f959957f2821489133bd144b28de86`.
H20 is supplemental, never the primary same-Orin gate or an Orin speedup
denominator. Processor arithmetic, FP16 vs original BF16, platform kernels and
fusion differences are confounded; no causal attribution or official
BF16/source-model equivalence follows from the current formal pass.

## Resources And Services

The [full sampled-resource audit](../../artifacts/orin/20261023-42/resource-audit-001.json)
retains 11026 telemetry and 52419 memory-guard samples, including initialization
and warmup. These are sampled process/platform envelopes, not isolated GPU
allocations or a proof of continuous peaks/leak freedom.

| Mode | Telemetry Samples | Guard Samples | Max Sampled VmRSS KiB | Max Reported VmHWM KiB |
|---|---:|---:|---:|---:|
| Conventional | 5517 | 26271 | 4252292 | 4261096 |
| Arena | 5509 | 26148 | 4254760 | 4261148 |

Minimum admission MemAvailable is 28348216 KiB; telemetry/guard minima are
21851012/21851516 KiB. Frozen 16 GiB admission and 2 GiB guard-floor gates were
not lowered. Process RSS/HWM are not summed with GPU memory or called GPU peak;
these measurements do not show a process peak reduction from the smaller arena.

Resource limitations are preserved explicitly:

- Nominal 1 s telemetry has a largest observed gap of 5.623486 s and a largest
  last-sample-to-worker-exit gap of 4.866858 s. Nominal 0.25 s guard sampling
  reaches a 0.254274 s maximum gap. Process-exit `/proc`/`smaps_rollup` gaps remain.
- All three CV thermal sensors are invalid in all 11026 samples each. Valid
  sampled CPU/GPU temperatures span 45.218-58.375 C / 40.062-56.906 C. Sampled
  CPU/GPU frequencies are 2201600 kHz / 1300500000 Hz; this is not continuous
  proof of absence of throttling.
- Power sensor reads yield 330780 `PermissionError` entries and no readable
  power samples. Missing readings were not replaced by zero; calibrated power,
  energy/request and a 60W operating-point claim remain unavailable.
- 375 owner scans find no visible competing owner, with up to 269 denied
  process inspections. Coverage is same-user device-FD visibility only, not
  exclusive hardware ownership. Tegrastats civil time is known incorrect;
  monotonic envelopes are used, not request-aligned power/thermal attribution.

The [service audit](../../artifacts/orin/20261023-42/service-audit-001.json)
verifies the explicitly authorized two-Java stop/restore window, original
argv/JAR/config/cwd and original manager PID4237/start27537. The observed
`data-collection.jar` PID4129713/start21419985 and starter
PID4129674/start21419872 have new births and match the restoration receipt.
Other manager rows are unchanged during stop/restore. Raw restore order is
starter then data-collection, exactly inspection order; its disclosed in-memory
projection into authorization order changes no raw evidence or hashes.
This proves **service-process restoration, not application readiness, recovered
in-memory state or continuous nonselected-process stability**.

The final [target closure](../../artifacts/orin/20261023-42/target-closure-001.json)
stream-verifies all 216 required/observed pins after sampling, bound to the
same final campaign/spec/protocol and board boot. It covers declared resources,
SDK inputs/headers, processor/tokenizer, raw data/noise and frozen references,
not an exhaustive runtime `dlopen` closure or BF16-fidelity proof.

## Evidence And Recheck

The [archive index](../../artifacts/orin/20261023-42/README.md) links all final
reports, raw worker records, references, service observations and CDFs. The
immutable cohort fingerprints are:

| Input | SHA256 |
|---|---|
| Campaign | `0a93887c8c714153c7fc95641172640e2fd274066f7395bd737f7e440d6b1f44` |
| Spec | `0b200c67f20e808c2e22acc9a355296a4c4f767b691bb214019ed8aa517fe0d8` |
| Protocol | `b363c7017577f518f3dfa35723d62f4ebc11b6a2ba864d6c97b547762888a737` |

Host-only joint recheck, with a fresh output name and unchanged input paths:

```sh
R=artifacts/orin/20261023-42
PYTHONDONTWRITEBYTECODE=1 python artifacts/orin/20261023-40/audit_native_cohort.py \
  --capture "$R/capture" --spec "$R/window/formal-spec.json" \
  --model-audit "$R/model-data-audit-001.json" --resources "$R/resource-audit-001.json" \
  --services "$R/service-audit-001.json" --latency "$R/latency-001/summary.json" \
  --closure "$R/target-closure-001.json" --output "$R/joint-audit-RECHECK.json"
```

Commands for the independent model/resource/service audits and neutral-caption
rendering are indexed in the archive README. Do not overwrite sealed JSON,
requests, binaries, CSVs or plots. The documentation update itself performs
only local evidence reads; no SSH, GPU, model or service operation is run.
