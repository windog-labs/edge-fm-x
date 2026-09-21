# CUDA Platform Formal Coverage

2026-09-09 08:25 UTC reconciliation. This is a coverage and traceability index
for the available H100/H20 formal CUDA evidence. It is not an Orin/BPU result,
and the H20 rows remain model-tensor boundary measurements rather than a
sensor-to-action end-to-end claim.

2026-09-09 correction: subsequent user direction restricts new GPU results to
H20. H100/RTX rows below are retained historical experiments, not the current
target matrix. The normalized pi0/pi0.5 rows were previously transposed or
copied from the dual-output row; the corrected numbers below follow each
model's archived `*-h100-prepared/latency-table.csv`.

## H100 rows (resident model-tensor)

| Model | Output scope | off mean ms | required mean ms | required p99 ms |
|---|---|---:|---:|---:|
| SmolVLA | normalized + official scaled | 118.171 | 45.972 | 57.069 |
| pi0 | normalized | 109.108 | 59.052 | 63.362 |
| pi0 | normalized + native F64 | 107.460 | 59.861 | 63.795 |
| pi0.5 | normalized | 133.087 | 65.300 | 71.753 |
| pi0.5 | normalized + native F64 | 126.277 | 64.436 | 69.198 |

H100 pi0/pi0.5 dual-output values are from the 16-frame dual-output formal
replays. H100 Smol row is from `smolvla-h100x4-20260908-001`.

## H20 rows (resident model-tensor)

Latest closure: the new pi0.5 H20 formal campaign passed all 15 processes and
the full independent audit. `h20_vla_formal_table_20260909.md` now describes
the five-model table. The rechecked five-model CSV is
`artifacts/recovery-audit-20260909/h20-five-summary/cuda-session-table.csv`,
SHA256 `c3ea57c739e5b910acc4d68c6ac7a0ad69fd043ab226ec9d2561fca03e5852f4`.
The earlier four-model recheck below remains a separate preserved artifact.

Continuation recheck: the public `summarize_session_campaigns.py` re-read all
four completed archives and produced twelve policy rows in current-worktree
`artifacts/recovery-audit-20260909/h20-existing-four-summary-002/`. It checked
raw complete output bytes, frozen reference hashes, per-worker/aggregate timings,
CDF ranks and actual H20 UUID/type/driver telemetry. No inference was repeated.
The first local attempt lacked pi0's additional-output reference binaries; a new
archive copy received the NAS originals, which then matched the frozen hashes.
The original experiment files were not edited. pi0.5 is not in that historical
four-model summary; its new completed campaign is included in the five-model summary.
The rechecked set contains 61,440 measured calls and 190,080 complete output
tensors. Summary report SHA256:
`5ec3a99cc81c3d5f90fd8bf410f8da32e40afe4ea32ff0ec1ea4c5aa581e9d57`;
CSV SHA256: `19b2da7e6dd66ab8afec9835944e448e3285bb09a90545af1b051feb771e86e5`.

| Model | Output scope | off mean ms | required mean ms | required p99 ms |
|---|---|---:|---:|---:|
| SmolVLA | normalized + official scaled | 117.632 | 47.113 | 49.798 |
| RDT-1B | normalized + robot outputs | 176.711 | 164.398 | 168.415 |
| pi0 | normalized + native F64 | 139.629 | 84.651 | 86.143 |
| pi0.5 | normalized + native F64 | 180.263 | 95.344 | 113.096 |
| CogACT | five complete outputs | 84.870 | 76.231 | 78.861 |

## Missing cells

- H100: RDT-1B and CogACT formal rows were not measured; new H100 experiments
  are no longer requested.
- H20 pi0.5 is now complete at the resident-tensor boundary. Its new result
  is from `runs/pi05-dual-recovery-20260909/prepared/`. The earlier correction
  remains important:
  the 362.055789/360.010064/350.053228 ms dual-output campaign ran on RTX 3060,
  UUID `GPU-ce878329-6b58-5666-292d-94185f5e5585`, as shown by its original
  `prepared-002/protocol.json` and `runs/00-off/telemetry.jsonl`. It must not
  populate the H20 matrix. The older interrupted H20 attempts and short pilots
  are not included in the new completed formal campaign.
- Both platforms still lack official/vendor baseline comparison and Orin/BPU
  main-table cells.

## Source of values

- H100 pi0 normalized:
  `runs/pi05-h100-aten-recovery-20260909/pi0-h100-prepared/`.
- H100 pi0 dual:
  `runs/pi05-h100-aten-recovery-20260909/pi0-h100-dual-prepared/`.
- H100 pi0.5 normalized:
  `runs/pi05-h100-aten-recovery-20260909/pi05-h100-prepared/`.
- H100 pi0.5 dual:
  `runs/pi05-h100-aten-recovery-20260909/pi05-h100-dual-prepared/`.
- H100 Smol:
  `runs/smolvla-h100x4-20260908-001/complete-002/prepared/`.
- H20 reports are under `runs/smolvla-h20-20260908-001/complete-001/prepared/`,
  `runs/rdt-replay-20260908-003/prepared/`,
  `runs/pi0-replay-20260908-002/prepared/`,
  `runs/pi05-dual-recovery-20260909/prepared/` and
  `runs/cogact-numerical-h20-20260908-002/prepared/`.

These rows are model-tensor boundary numbers and cannot be placed into an
Orin/BPU end-to-end table.
