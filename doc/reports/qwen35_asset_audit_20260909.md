# Qwen3.5 Asset Audit

2026-09-10 status: both real-image official Python formal baselines are complete
in `qwen35_natural_profile_20260909.md`. The asset-only milestones and missing-cache
snapshot below are historical. Native deployment acceptance remains separate.

## Current Verification

2026-09-09: the user authorized downloads onto the shared H20 NAS. Both
snapshots are now complete and independently rehashed without model execution:

| Snapshot | Revision | Weight Bytes | Tensor Count |
| --- | --- | ---: | ---: |
| `/xs-train-nas/zzm/models/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 1746942600 | 488 |
| `/xs-train-nas/zzm/models/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 4548221488 | 632 |

Each snapshot's 13 files match pinned revision metadata fetched via
`hf-mirror.com`: LFS files use SHA-256, other files use Git blob SHA-1, and all
files also have local SHA-256 records. Safetensors shard keys, byte ranges and
index total size match. Both report Apache-2.0 licensing. The metadata service
was the mirror, not a separately authenticated signature or live upstream check.

Retrieved evidence in the current worktree:
`artifacts/recovery-audit-20260909/qwen-assets-002/report.json`, SHA-256
`fed0a8c669974cf0df96903cccd167998eaa0519c99df3e08f5351e788bc53fa`.
The earlier urllib/403 audit attempt is preserved as `qwen-assets-001` on NAS;
the successful audit used curl to fetch the same metadata. No model loaded or
GPU inference ran during this asset check. The later vision + prefill + decode
baseline results are linked above; weights are no longer a blocker.

## Historical Snapshot

2026-09-09 03:00 UTC. G6 cannot start from an empty cache. The local HF hub
contains only `models--Qwen--Qwen3.5-0.8B` and `models--Qwen--Qwen3.5-2B`
directory skeletons with a `refs/main` file and no snapshot/blob weights.

Remote scans of H20-x8-2, H100-x4 and H100-x8 under the available HF hub roots
and `/xs-train-nas/zzm` did not return Qwen3.5 weight directories.

## Historical Consequence

G6 remains pending on real 0.8B/2B assets and an environment with the matching
model implementation. Downloading unverified/gated weights, relabeling
Qwen2.5-VL data, or running text-only fixtures would not satisfy the paper's
vision + prefill + decode requirement.

## Historical Start Requirements

- Authorized Qwen3.5 0.8B and 2B snapshots with complete SHA/size manifests.
- Local or remote CUDA environment with the exact model source/version and
  multimodal processor.
- A documented image+text task and generation profile for TTFT, tokens/s,
  total E2E, memory and greedy output alignment.
