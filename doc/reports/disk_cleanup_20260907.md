# VLA Artifact Disk Cleanup

Executed on 2026-09-07 with explicit user authorization to reclaim local disk.
This is storage maintenance, not a new model validation or paper result.

## Result

- Status: `completed_and_verified`.
- 88 operations, approximately 54.61 GiB of allocated file blocks reclaimed.
- Available space: 3.20 GiB immediately before application, 57.62 GiB afterward.
- No processes were stopped; selected files had no observed open handles or
  mappings for the current user before execution.
- Model/runtime source and existing experiment reports were not edited.

| Operation | Files | Reclaimed GiB |
|---|---:|---:|
| Hardlink byte-identical frozen Bundle package copies | 35 | 36.37 |
| Remove downloaded ranges reproducible from verified complete weights | 51 | 11.77 |
| Remove compile-cache libraries identical to retained package ZIP members | 2 | 6.47 |

## Evidence

Evidence directory:
`artifacts/edgefm-vla-goal/20260906-044509/disk-maintenance-20260907/`.

- `plan.json`: explicit paths, original identities, SHA256, retained sources,
  range offsets/lengths, and ZIP member names. Plan SHA256:
  `2200df3fb2871335b18877861dac87e4c2c45a30dbd5766c401c40264a7716c1`.
- `actions.jsonl`: completed operation journal, SHA256:
  `6855518172ac3eb10845e08c116330424f1d1ef8f665de41e9e35c388eea08cf`.
- `report.json`: final disk-space and retained-content verification.
- `cleanup.py`: source of the reviewed plan and guarded execution.

All 45 verified package paths and 36 protected files were checked again after
cleanup. Retained content hashes are unchanged. Formal raw actions, timing
samples, CDF data, failure logs, unique compiled packages, complete weights,
input datasets, and source snapshots were retained. The pending pi05 formal
audit was not replaced with a storage-maintenance claim.

## Reuse And Recovery

Identical frozen package paths now share inodes. Do not modify them in place:
build into a fresh output directory, or make an independent copy before any
intentional modification. Removing one alias does not remove the others.

The complete pi0/pi05 `model.safetensors` files and conversion records remain
in their original cache directories. Deleted transfer ranges can be recreated
by reading `length` bytes from the retained weight at `offset`; verify against
the recorded range SHA256. Transfer logs and metadata were retained.

The two removed `.so` cache files can be restored from `retained` package and
`archive_member` recorded in the plan, then checked against their original
SHA256. Their surrounding cache source/metadata and the final AOTI packages
were retained. No unique intermediate was deleted based only on its age.
