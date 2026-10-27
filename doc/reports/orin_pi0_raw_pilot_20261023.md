# Pi0 Orin Raw E2E Pilot

Status: complete ordinary deployment and same-Orin ordinary/Arena quality pilot
passed. Formal D3 sampling is not complete. The five-model Goal remains open.

## Actual Path

The original four-part prefix path compiled, but original part01 produced NaNs
both inside and outside Session. All 51 complete standalone outputs matched
the Session dump byte for byte, including NaN payloads. The failed replacement
three-view visual producer also returned three entirely nonfinite outputs.
These failures are retained, not relabeled as successful deployment.

A source-preserving repeated-call path is now used. Each of the three visual
tail closures has 1400 nodes. Their operators, attributes, shapes, parameter
payload identities and per-node precision policies are exactly isomorphic.
One actual single-view engine executes three fresh calls, followed by the
original 652-node language consumer: `3 * 1400 + 652 = 4852` original nodes.
No camera or neural stage is omitted, and no precomputed visual features are
used by the raw Session. The full eleven-region Module folds canonically back
to the original all-ten Module. Native physical IDs 0/1/3, public schema8751,
F64 state, all ten action iterations, predicate and atomic publication remain.

The failed and successful engines select different implementation families
for many otherwise identical matrix operators. This is evidence of
composition-sensitive target compilation, not a proven hidden-precision or
vendor-kernel root cause. No new dense FP32 exception, whole-model FP32 path,
quantization search, SDK upgrade or source-parameter change was introduced.

Ordinary and verified-Arena modes use exactly the same eleven engines/providers
and public Module. Their only logical Plan difference is static arena layout:
205184256 versus 199307264 bytes. This is compiler-layout accounting, not
measured process/GPU peak memory. Neither mode is claimed to use replay.

## Complete Raw Pilot

Actual target: AGX Orin serial1421524312594, boot
`3c801f5d-6897-4d97-b021-3c52db38a750`, TRT8.5.2.2/CUDA11.4/L4T35.4.1,
unchanged existing MAXN, not relabeled 60W. Target execution is native C++/TRT;
no target Torch model or Torch baseline was installed or used.

Each mode completed one independent process with 16 warmup and 16 measured
requests. Every call includes fresh online native image/text/state preparation,
H2D, complete visual/language processing, ten action steps, native finishing,
correct-device completion and both full CPU output tensors. Initialization,
file decoding, output verification and logs are outside the request clock.

| Pilot Mode | Measured Requests | Mean E2E ms | Max E2E ms |
| --- | ---: | ---: | ---: |
| Ordinary repeated-call TRT | 16 | 1110.754214 | 1117.745856 |
| Verified Arena | 16 | 1123.260410 | 1132.541312 |

These are pilot-only values, not formal main-table/CDF results. The observed
Arena pilot is slower; that negative outcome is retained. Sixteen recorded
observations/noise sets repeat with fresh inference, not independent robot
trajectories.

Both modes completed 32 accepted requests each. All 128 complete output files
were independently checked by byte count, SHA256, finite classification and
full-array numerical comparison against the frozen same-Orin references.

| Complete Output | Type / Shape | Maximum MSE | Minimum Cosine | Byte Exact |
| --- | --- | ---: | ---: | --- |
| Normalized action | F32 `[1,50,32]` | 0 | 1 | All |
| Native action | F64 `[1,50,14]` | 0 | 1 | All |

The unchanged primary gate is complete normalized MSE<1e-5 and cosine>0.9999;
native errors are supplemental and all public outputs must be finite.
Byte equality for these inputs is not mathematical losslessness for arbitrary
inputs or robot/task success.

## H20 Supplement

The independent offline audit matches all sixteen original observations,
checkpoint/source identities and exact saved noise. Comparing complete Orin
outputs with the frozen H20 official outputs gives:

| Space | Cosine | MSE | RMSE | Maxabs | Meanabs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normalized, all 25600 values | 0.6685354882 | 0.06826529088 | 0.2612762731 | 1.8938393 | 0.1297010668 |
| Native, all 11200 values | 0.9907577346 | 0.004825036517 | 0.0694624828 | 0.3657310086 | 0.04994661581 |

Normalized per-observation results are 0/16 at the original thresholds. These
are preserved failures. By the human-fixed comparison contract, H20 is
supplemental and not the primary same-Orin optimization gate. No H20 latency
enters an Orin speedup. Processor arithmetic, FP16/BF16 compute and platform
implementation differences are confounded; their causal contributions are
not isolated. The successful same-Orin comparison does not establish official
BF16/source-model equivalence or deployment task quality.

## Formal Sampling And Resources

The frozen formal protocol is two configurations, five independent processes
each, 128 warmup and 1024 measured calls per process, rotated serial order.
The first collector launch ended before any worker because available system
memory was below the frozen 16GiB admission threshold. It has zero formal
rows; there is no cohort to merge or promote. The 2GiB safety floor and counts
were not reduced. Other services were not stopped.

At the subsequent read-only inspection, MemAvailable was13700452KiB. The human
then explicitly authorized temporarily stopping only the two Java programs
and restoring them using their original manager after sampling or failure.
A fresh inspection verifies their PID/start/argv, parent manager, JAR and
configuration hashes; new authorization is recorded independently of RDT.

The new `20261023-31/formal-002` cohort has started. The service-window owner
PID3871611/start20094526 owns collector PID3872054/start20096270. The first
native worker PID3872125/start20096646 is live with matching argv. Supervisor
reports both selected programs Exited/PID0; all other managed programs remain
unchanged. Post-stop MemAvailable is36977600KiB. At the 01:31 UTC checkpoint,
42 warmup calls pass online complete-output flags; no formal result is yet
accepted. Independent all-output/resource/CDF audits and restoration validation
remain required. The pinned immutable lease-at-activation snapshot survives
later mutation of the live lease during restoration. The service guard restores
on normal collector exit, failure or handled interruption, with an outer
restoration fallback and six-hour window alarm. This is not resilience to
power loss or SIGKILL, nor an application-health guarantee after restoration.

## Evidence

- `artifacts/orin/20261023-15/final-checkpoint-001.json`: preserved original fault.
- `artifacts/orin/20261023-20/split-002/publication.json`: unchanged partition nodes/weights/precision.
- `artifacts/orin/20261023-21/repeated/proof-001/source-proof.json`: exact shared-callable equivalence.
- `artifacts/orin/20261023-21/repeated/actual-campaign/generated-session-001/`: actual eleven-region binding/generated source.
- `artifacts/orin/20261023-20/repeated-resource-materialization-001.json`: eleven independent copied resources, all hashes verified.
- `artifacts/orin/20261023-20/repeated-readonly-audit-001/build-receipt.json`: actual native/SDK/DSO/source/resource audit.
- `artifacts/orin/20261023-20/repeated-ordinary-raw-pilot-001/` and `repeated-optimized-raw-pilot-001/`: untouched complete pilot outputs.
- `artifacts/orin/20261023-20/same-orin-pair-quality-audit-001.json`: independent complete primary comparison, statuspassed.
- `artifacts/orin/20261023-25/REPORT.md` and `audit-001.json`: H20 supplemental failures and identities.
- `artifacts/orin/20261023-24/formal-spec-002.json`, `formal-001/campaign.json`: frozen protocol and terminal pre-worker memory refusal.
- `artifacts/orin/20261023-24/parent-tests-003.xml`: 54 collector protocol/refusal tests; not model evidence.
- `artifacts/orin/20261023-31/service-authorization-001.json`: current human authority for exactly two Java services.
- `artifacts/orin/20261023-31/services-inspection-001.json`: selected process/manager/config/JAR inspection.
- `artifacts/orin/20261023-31/managed-window-001/formal-spec.json`: fresh protocol, collector and immutable lease snapshot pins.
- `artifacts/orin/20261023-31/observation-001.json`: independently read live identities, service states and first 42 warmup flags.
- `artifacts/orin/20261023-31/preflight-tests-002.xml`: 79 NONMODEL software checks, not neural performance/quality data.
