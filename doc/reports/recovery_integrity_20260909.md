# Recovery Integrity Corrections

This record corrects mistakes made during the continuation. It does not change
the original experiment outputs, controller logs or audit results.

## pi0.5 Platform

The `local-pi05-replay-20260908-001` 350.053228 ms required result is **RTX 3060**,
not H20. The original protocol selects UUID
`GPU-ce878329-6b58-5666-292d-94185f5e5585`; the first formal telemetry record
names NVIDIA GeForce RTX 3060. Both files remain under
`/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/openpi-inventory/local-pi05-replay-20260908-001/prepared-002/`.
The erroneous H20 row and five-model H20 completion statement were withdrawn.
H20 pi0.5 full dual-output formal coverage remains pending.

## Frozen Source

During the first RoPE retries, a new `numerical_probe.py` was incorrectly copied
over the old NAS `source-operator-benchmark-0905` file. The later successful RoPE
run uses the separate `source-rope-h20-20260909` source root, not this old root.

- Original SHA256: `f2cb22ab2f390e2951d7fe9911070b9cb6daed041ee89dad66504819153d7e13`.
- Accidental replacement SHA256: `deb2ea42ba38bddf05f37f43a9616884653704521b7587ed8ad1f13824d55732`.
- The replacement was saved at current-worktree
  `artifacts/recovery-audit-20260909/numerical_probe_accidental_update.py`.
- The exact original file was restored from the retained local 0905 snapshot;
  remote SHA was checked after restoration. No model outputs were rerun or
  re-certified, and no historical measurement is claimed to have used this
  replacement. Additional new helper files in the old root are not part of its
  historical frozen manifest.

## Qwen Launcher

Controllers 4070833 and 4070834 failed before launching a worker because their
output directories were pre-created. The first proposed 2B launcher also used
an x8-1 UUID on x8-2 and an incorrect relative checkpoint path. Both failed
directories and logs remain on NAS. No formal Qwen results came from them.

The replacement benchmark verifies the UUID on the actual host, accepts only
existing absolute model/image paths, runs monitored workers sequentially and
reuses the register/reset/register ownership handshake. CPU-available token
streaming supplies TTFT; decode throughput excludes the first token. It checks
and saves every warmup/measured token against an untimed reference, without
pickle. This is a generic official Python baseline, not a VLAForge/no-Python
deployment or an optimized-output equivalence claim.
