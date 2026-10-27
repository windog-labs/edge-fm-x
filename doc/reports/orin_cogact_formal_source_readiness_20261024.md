# CogACT Formal Source Readiness

Checkpoint: 2026-10-23 20:03UTC. **Offline source/resource/typed-quality contracts
are qualified, not target activation or a formal model result.** No new model
inference, service interruption, SDK change or target-tool staging occurs.
Required VLA formal coverage remains4/5.

## Resource Contract

The archived collector31and typed auditor103both assume16GiBadmission/2GiBfloor.
CogACT's actual successful pilots use24GiB/4GiB. The new pinned source versions
accept an explicit, model-neutral resource policy and require exact agreement:

| Point | Qualified Current Proposal |
|---|---|
| Per-process admission check |25,165,824KiB |
| Owned guard argv |`--admission-kib 25165824 --floor-kib 4194304` |
| Runtime sampled floor |4,194,304KiB |
| Published protocol fields |Actual declared24GiB/4GiB |
| Independent output audit |Requires the exact declared resource contract |

The pure policy function is embedded identically into both source pins, avoiding
an unpinned runtime side module. Legacy configurations without a policy retain
their original16GiB/2GiBbehavior. Explicit policies reject lower original minima,
inverted limits, booleans/floats/strings, unknown schemas and extra fields. The
current proposal explicitly declares24GiB/4GiB; the legacy default is not an
approval to lower its proven capacity gate.

Whole-file inverse checks recover both original source files exactly. Sampling
counts, worker commands apart from resource values, numerical gates, opaque
byte handling, service-lease checks and cleanup logic are unchanged. Both old
files and their earlier accepted receipts remain intact. There is no model-name
dispatch or precision/quantization change.

Collector source SHA256:
`0ccb442c708157c4bf8dd090c59c59cbb153084203a2e3c3bb7edf3df1770abd`.
Auditor source SHA256:
`1882a4923bd19b9648e07e90c7e26bfadeb127b80539a68c2c60197738edbc57`.

## Actual Pilot Data Compatibility

Both actual archived comparison pilots are reopened:ordinary106and Arena100,
each16observations and80complete public output tensors. All160tensors are
independently checked against the unchanged same-Orin reference, with public
IO types/shapes, full extent/hash, finite F32/F64arrays and exact U8/I64bytes.
Neither integers nor RNG state are cast to floats or assigned cosine metrics.
Normalized MSE0/cosine1and native supplemental0are reproduced. The upstream
published request-trace proofs are checked before these archives are accepted.

The resulting quality report uses the existing generic collector schema. It
describes actual archived pilot data, not new inference, a formal cohort or a
controlled timing pair. The16ordinary and16Arenaobservations are the existing
ones; repeated local audits do not increase observation/trajectory counts.

The collector's real `validate()` accepts the actual report and all93exact-byte
file bindings under a strict local path-rebasing resolver. This checks real
reference payloads, metadata, profile, mode manifest, shared-resource catalog,
Session libraries, runner and source hashes. It does not load the copied ARM64
libraries or execute the runner. The archived-only validation deliberately
excludes an active service lease and actual board resource/owner preflight;
it is not target readiness.

## Activation Remains Forbidden

The saved target template has schema
`edgefm.native_api_formal_spec_template/1`, not the runnable formal-spec schema.
The real collector rejects it before file rebasing. Human authority and service
lease/window pins are null. No runnable archive-validation specification is
saved:the temporary in-memory schema/path adaptation is limited to file/schema
checks and never calls `collect()`.

Three requirements remain before a real campaign:

1. Independent human permission for the proposed at-most-four-hour interruption
   of only the two original Java programs. Original build/pilot authority is not
   converted or renewed into formal scope.
2. Review and freeze the actual original-manager service window, its lease,
   signal/timeout cleanup, source pins and24GiB/4GiBresource integration.
3. Fresh target identity/PID/argv, owner, memory, environment and resource checks
   immediately before an authorized launch.

The later [window mechanism review](orin_cogact_window_mechanism_20261024.md)
adds three real Orin CPU-only transient-user-unit cases for normal exit,
timeout and own-main SIGKILL, with detached-child cleanup and post-hook evidence.
Those diagnostic scripts run on Orin, but the107formal collector/auditor tools
do not. The probes add no inference, original-application stop/start or actual
production lease qualification. The original six-hour build/pilot authority
has expired; separate formal permission and the integration above remain needed.

Counts remain two modes × five processes ×128warmup+1024measured,batch1:
11,520calls/10,240measured/57,600complete tensors. No new formal samples exist.
The approximately7.63Bpublic-dependency identity, source/noise/tape, allten
iterations, same resource pool and Arena-only Plan difference remain unchanged.

## Tests And Provenance

The new source runs83resource/behavior/real-archive checks successfully. They
exercise actual constructed guard arguments and protocol fields without a
subprocess, refusal at the old16GiBavailability, rejection of3GiBat the new
4GiBfloor, exact protocol/auditor matching, typed full-cohort fixtures, corrupted
private archives and actual pilot/collector compatibility. All synthetic
collections remain NONMODEL, failed or non-formal.

The original source independently passes all47checks, including its archival
identity pin. An initial attempt pointed that original identity assertion at
the new artifact; it correctly rejected the different source SHA. The failed
XML is retained. That archive-specific check is not retargeted or deleted:
it is run unchanged on its original artifact, while new inverse/pin checks cover
the derived files. The final83-check new-source suite excludes only that old
artifact identity check; its behavior/quality tests still run. These two suites
reuse scenarios and are not130independent experiments.

## Evidence

- [Offline qualification and source/test pins](../../artifacts/orin/20261023-107/offline-source-readiness-001.json).
- [Exact full-file derivations](../../artifacts/orin/20261023-107/prepared-001/derivation.json).
- [Actual160tensor pilot-quality schema](../../artifacts/orin/20261023-107/typed-pilot-quality-001.json).
- [Actual archive/file-binding qualification](../../artifacts/orin/20261023-107/source-package-qualification-001.json).
- [Non-runnable target template](../../artifacts/orin/20261023-107/target-spec-template.json).
- [83new-source checks](../../artifacts/orin/20261023-107/combined-source-regression-005.xml).
- [47original identity/behavior checks](../../artifacts/orin/20261023-107/original-identity-and-regression-003.xml).

The [ordinary repeat](orin_cogact_ordinary_repeat_20261024.md) and earlier actual
paired pilot remain the model evidence. This source package does not create a
CDF, throughput row, new fidelity observation, active permission or whole-Goal
completion.
