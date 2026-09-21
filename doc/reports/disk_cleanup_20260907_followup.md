# Local Disk Cleanup Follow-up

Date: 2026-09-07. Scope: the user's explicit authorization to clean the four
evaluation worktrees, xs-perception waste, and concluded EdgeFM experiments.
No remote files, running processes, model implementation, or active main-thread
experiments were modified by this cleanup.

## Result

The measured cleanup interval increased available disk space from 55.559 GiB
to 203.017 GiB, a net increase of approximately 147.458 GiB. Final `df -hT /`
reported 916G total, 666G used, 204G available, 77% usage. Other local activity
can slightly affect filesystem-wide free-space deltas.

| Category | Result |
| --- | --- |
| Git temporary objects | 50 files removed; 63.345688 GiB released |
| Completed evaluation derived replay inputs | 28 MCAPs removed; 33.251381 GiB released |
| Concluded SmolVLA AOTI candidates | 60 package paths removed; 9.160324 GiB actually released after hardlink accounting |
| Four evaluation worktrees | Removed via `git worktree remove --force` only after verified recovery archival |
| Worktree recovery | One 5.986042 GiB archive; all 19,568 members verified; original shared-object hardlinks preserved |
| Worktree stage net space gain | Approximately 41.702324 GiB, including the retained recovery archive |

The xs-perception checkout now occupies about 59G. The evaluation-worktrees
parent is empty except for its directory entry. The existing EdgeFM Goal
artifact tree occupies about 135G; recovery/audit files are a separate 6G.

## Safety And Evidence

- Both active main repositories retained their original HEAD, index checksum,
  and tracked diff checksum throughout cleanup. Existing modifications remain.
- Main Git connectivity checks passed before/after file deletion and after
  worktree removal. All initialized xs-perception submodules subsequently
  passed recursive connectivity checks. Git now reports zero garbage objects.
- Process cwd, command line, open-file, and mapped-file checks found no visible
  owners of the deletion targets. Access to fd/maps for one `sd-pam` and two
  `sshd` processes was restricted and is explicitly recorded in the ledger.
- No process was terminated. The unrelated Stage1 transfer was excluded.
- Only explicitly listed old SmolVLA `.pt2` packages were removed from EdgeFM.
  Capture exports, source snapshots, raw outputs, inputs, reports, v6 candidate,
  current deployment packages, weights, and other model directories remain.
- The original evaluation recordings and measured outputs remain. Only derived
  replay inputs were removed. Historical reports were not rewritten as passes.
- Cleanup notices were added to both affected experiment roots. Historical
  integrity checks requiring removed files now need regeneration; their old
  successful execution records are not proof of current local completeness.
- No model, performance, board, or full-paper acceptance status is promoted by
  this maintenance. Other hardware and ongoing Goal tasks remain out of scope.

## Audit And Recovery

Audit root:
`/home/zhangzimo/Repos/private/edge-fm-x/artifacts/disk-cleanup-20260907T112748Z/`

- `plan.json`: fixed removal scope, identities, and SHA256 for experiment files.
  Plan SHA256: `b6b7c714c77e291190675f2496a1d8dbfa1aeb16d261bd93419cb8cacb1800a5`.
- `actions.jsonl`: per-file deletion, verified backup, and worktree removal log.
- `files-report.json`, `worktrees-report.json`, `recovery-report.json`: results.
- `*-backup-manifest.json`: exact repository HEADs, staged/unstaged patches,
  untracked/ignored file lists, and per-file recovery checksums.
- `worktrees-recovery.tar`: retained recovery archive, 6,427,463,680 bytes.
  SHA256: `ac465e7a47604b38a530e36ee7607fb94d0a94f80149a95c6d5c2f00b6db4b71`.

The four earlier per-worktree TAR files in the action log were replaced by this
verified combined archive; they are intentionally no longer present. The
archive contains modified/untracked/ignored files and private Git metadata,
not a full copy of unchanged tracked source. Use the recorded commits and the
retained Git object stores to reconstruct those files. Inspect recovery in a
separate staging directory; do not overwrite a live repository's Git metadata.

`cleanup.py` is a one-shot execution record, not a scheduled maintenance tool.
The early planning attempts encountered Python 3.10 API and `/proc` permission
issues before any deletion; the corrected fixed plan and all execution stages
completed. Do not rerun its destructive modes against already cleaned paths.
