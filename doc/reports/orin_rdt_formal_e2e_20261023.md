# RDT Formal Orin E2E

Status: **formal model data, sampled service/resource window and current restored
service-process contracts passed**. This closes RDT's D1-D3 campaign, not the
five-model Goal or the remaining shared operator/resource/power requirements.

## Scope And Protocol

- Real Jetson AGX Orin, serial `1421524312594`, boot
  `3c801f5d-6897-4d97-b021-3c52db38a750`; TensorRT 8.5.2.2, CUDA 11.4,
  L4T 35.4.1, unchanged existing MAXN (not relabeled 60W).
- Eleven identical frozen engines across conventional/shared/replay modes;
  FLOAT-carrier deployment, FP16 dense and declared numerical islands.
  Full online text/six-view visual encoding and all five original solver steps
  execute in native C++/TensorRT. No target Torch baseline or model compute.
- Batch one. Boundary: decoded CPU camera/state/text/noise through online
  preprocessing, transfers, complete model/solver work, finishing and complete
  consumable CPU actions. Initialization, decoding, logging and checks are outside.
- Each mode has five independent processes, each 128 warmup + 1024 measured.
  Rotated process order; 5120 measured rows per mode, 15360 total. All rows and
  tails are kept; no stage-sum, centering, outlier deletion or old-cohort merge.
- Sixteen recorded observations/noise sets are cycled. Repeated fresh computation
  is not 15360 independent trajectories, autonomous robot trials or control Hz.

## Main Results

All latency values below are milliseconds; throughput is complete chunks/s.

| Mode | Mean | p50 | p95 | p99 | Max | Std | Chunks/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional TRT | 3065.625218 | 3063.296512 | 3102.937584 | 3109.544230 | 3121.115744 | 23.509168 | 0.326198 |
| Shared stream | 3064.553677 | 3071.761888 | 3106.344786 | 3111.908306 | 3122.125760 | 29.516018 | 0.326312 |
| Full-loop replay | 2975.587343 | 2964.070992 | 3006.751920 | 3012.872988 | 3029.676640 | 21.612032 | 0.336068 |

Replay mean speedup is **1.030258858x**, a **2.937015%** latency reduction.
Shared-stream mean improvement is negligible and its p50/p99/std are worse;
those outcomes are retained. These results do not establish the draft's
sub-1.2-ms jitter claim, a vendor Edge-LLM speedup or robot deadline success.
No H20 time participates in this ratio.

![RDT pooled CDF](../../artifacts/orin/20261022-01/formal-cdf-001/e2e-cdf.png)

Pooled and per-process CDF PNG/PDF retain every measured sample. The two-cluster
replay shape and process differences are not smoothed or normalized away.

## Complete Output Quality

All **34560 complete tensors**, including warmup outputs, are byte-exact to
their frozen same-Orin references: 17280 calls, two outputs each.

| Output | Type / Complete Shape | Max MSE / Maxabs | Byte Exact |
|---|---|---:|---|
| Native action | F32 `[1,64,14]` | 0 / 0 | All |
| Unified normalized action | F32 `[1,64,128]` | 0 / 0 | All stored values |

The active-coordinate projection for normalized-action metrics is separately
proved; full storage is still compared, including inactive coordinates. The
fixed primary gates remain MSE < 1e-5 and cosine > 0.9999; minimum reported
cosine is 0.9999999999999999, distinct from the stronger byte-comparison result.
Each replay process records 1152 actual replays/five captured steps/zero ordinary
fallback. Shared ordinary counts are its intended path, not replay failures.

Original H20 BF16 discrepancies remain supplemental and failed where previously
recorded. Same-engine target equality is not original-BF16/PyTorch equivalence,
mathematical losslessness for arbitrary inputs or task/robot success.

## Resources And Restoration

All **46125** retained telemetry snapshots pass the unchanged sampled system
floor; each admission exceeds 16 GiB, and the minimum sampled MemAvailable is
18.966969 GiB. Maximum observed process RSS is 6805.437500 MiB, VmHWM
6811.164063 MiB, maximum valid temperature 61.875 C. CPU/GPU shared memory is
not added twice; RSS/VmHWM are not isolated GPU peaks or complete model footprint.
Permission-limited owners, invalid CV sensors, power gaps and unobserved
between-sample behavior remain explicit. No exhaustive SDK/driver accounting,
leak freedom, zero CPU allocation or matched energy result is inferred.

The exact two authorized Java programs were restored through their original
manager. A fresh read-only observation independently verifies:

| Service | PID | Start Ticks |
|---|---:|---:|
| `xs-ugv-starter-0.0.1-SNAPSHOT.jar` | 3291273 | 17225825 |
| `data-collection.jar` | 3291324 | 17225946 |

Boot, PID/argv/parent, original configuration/JAR hashes and unchanged other
programs during restoration passed. This proves current process contracts,
not application readiness or restoration of earlier in-memory state.
Collector, last worker and guard were separately confirmed absent afterward.

## Reproduction And Evidence

Root: `artifacts/orin/20261022-01/`.

- `formal-retrieval-001/rdt-raw-formal-001/`: complete untouched new cohort.
- `managed-audit-spec-final-001.json`: actual terminal input bindings.
- `final-audit-001/audit.json`: `full-data-and-restoration-verified`, SHA256
  `e793f5c7a1b0a3a63ac003eec8220e00892bfa8d888daf4dc6b1081fb8fa4f6f`.
- `final-audit-001/model-data-audit.json`: full 15-process/output/latency audit,
  SHA256 `c82241104031d2b8e080308d7e833b4c652488e60f24e6aa2a8ff7d740823ec0`.
- `final-audit-001/all-requests.csv`: all raw timing rows, including warmup.
- `formal-cdf-001/`: pooled/per-process PNG/PDF and hashed render manifest.
- `restoration-terminal-001/`: actual final guard/lease and fresh observation.

Exact terminal retrieval/audit/plot commands remain in the
[managed runbook](orin_managed_campaign_audit_20261022.md); this run uses
`restoration-terminal-001` rather than its example restoration directory.
The old interrupted cohort and every negative/pending receipt remain separate.
Required formal model coverage is now **2/5**; pi0, pi0.5, CogACT and shared
remaining requirements are not completed by this report.
