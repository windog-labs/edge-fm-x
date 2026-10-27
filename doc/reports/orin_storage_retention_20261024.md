# Orin Paper Evidence Retention And Model Cleanup

Date: 2026-10-24. The user explicitly released completed five-model reproduction
payloads to recover disk space: the objective is real paper measurements with
documented provenance, not retaining every model/engine indefinitely. This
changes retention policy, not the original measured results or quality gates.
Qwen assets and execution environments are outside the cleanup roots.

## Retention Policy

Retained: original latency CSV/JSON, complete action/reference/observation/noise
data, quality and resource audits, CDF numerical inputs and PNG/PDF, launch
protocols, precision/config/source identities, build/failure logs, scripts,
small graphs, native provider sources/DSOs and the unchanged paper PDF.
No slow sample, negative result or failed quality record is removed.

Released: individually listed cold neural model/engine payloads at least64MiB
under the two experiment roots below: `.onnx.data`, weight-bearing `.onnx`,
`.engine`, `.plan`, `.pt`, `.pt2`, `.pt2e`, `.safetensors` or `.hbm`.
Not every matching type exists in each root. Raw `.bin`, `.npy`, `.npz`,
`.csv`, `.json`, reports and source are not deletion candidates. Symlinks,
foreign-owned files, nested mounts and observed open payloads are excluded.
No whole worktree, original repository, cache environment, H20/J6M evidence
root or NAS Qwen artifact is deleted.

Deleted payloads have pre-unlink complete SHA256, size, inode metadata and
write-ahead journal entries. Large payload backups are intentionally not
created. Old source/engine closure receipts remain historical evidence of
the actual experiment, not a promise that every recorded binary still exists.
New inference, full payload rehash or bit-identical rebuild cannot be assumed:
models/engines must first be prepared again. Host-only paper data analysis
remains available from the retained measurements.

## Space Recovery

| Location | Operation | Released Allocated GiB | Result |
|---|---|---:|---|
| Local workspace | Initial exact-duplicate consolidation,14paths/6groups |29.193352| Full bytes/pathnames retained at this intermediate step |
| Local `artifacts/orin` | Retire108released cold payload paths |194.347324| Paper evidence retained; initial consolidated payloads may also be removed under the expanded permission |
| Orin `/data/datalog/edgefm-orin` | Retire208released cold payload paths |190.121849| Target identity/idle preflight, complete hashes and journals retained |

Local available space changes from approximately26GiBbefore both steps to
approximately249GiBafterwards. The two local operations reclaim approximately
223.54GiBcombined; do not add duplicate logical pathname sizes to this number.
Orin's experiment data is on `/data/datalog` NVMe, not the eMMC root filesystem.
Final free-space observations and exact byte counters are in the
[post-cleanup verification](../../artifacts/storage-maintenance-20261024/post-cleanup-verification-001.json).
Background writes can affect free-space deltas; plan inode allocation and
filesystem availability are separately recorded.

## Verification

The local post-cleanup five-model exporter rechecks all69120requests,
61440measured rows,12configurations/60processes and18original CDF files.
Its main/quality/process CSVs and Markdown must be byte-identical to the
accepted pre-cleanup export. This is reanalysis of real retained Orin data,
not synthetic data or fresh inference after engine removal.

[Retained five-model tables](../../artifacts/orin/20261024-07/tables-001/tables.md),
[post-cleanup table export](../../artifacts/storage-maintenance-20261024/post-cleanup-five-tables-001/summary.json),
[original publication review](orin_core_acceptance_20261024.md),
[data-only current requirement audit](../../artifacts/orin/20261024-08/core-requirement-audit-003.json).
The audit reviews accepted historical measurements and remaining data; it
does not establish current engine availability. Original001/002receipts are
not rewritten to conceal the retention change.

Cleanup safety checks cover changed files, retained raw formats, open handles,
symlinks, hardlink counting and content hashes. Seventeen checks pass.
The first target preflight used a Python API unavailable on Python3.8and
failed before any unlink/journal. The separate compatible source retains the
same descendant/path checks; that failure is preserved and no environment,
service, clock, power or model run is used to fix it. Visibility is limited to
accessible current-UID handles, not proof of global process exclusivity.

## Receipts

- [Initial duplicate plan](../../artifacts/storage-maintenance-20261024/weight-dedup-plan-001.json)
- [Initial duplicate result](../../artifacts/storage-maintenance-20261024/weight-dedup-result-001.json)
- [Local payload plan](../../artifacts/storage-maintenance-20261024/local-payload-plan-001.json)
- [Local payload result](../../artifacts/storage-maintenance-20261024/local-payload-cleanup-001.json)
- [Target payload plan](../../artifacts/storage-maintenance-20261024/target-payload-plan-001.json)
- [Target payload result](../../artifacts/storage-maintenance-20261024/target-payload-cleanup-002.json)
- [Preserved first target preflight failure](../../artifacts/storage-maintenance-20261024/target-preflight-failure-001.json)
- [Seventeen safety checks](../../artifacts/storage-maintenance-20261024/cleanup-safety-tests-002.xml)

Qwen3.5 remains in progress, not complete from this cleanup. Continue its
complete vision/prefill/decode/state deployment using separately retained host
assets and TensorRT-only target execution. The five-model paper conclusions,
private-memory/alternate-mode limitations and unchanged PDF remain as reported.
