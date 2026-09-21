# OpenPI Board Data Handoffs

Status: offline data/reference handoff verified. Orin/BPU execution is pending.

Pack:
`/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/pi0-dual-data-002/pack`

Manifest SHA256:
`5f455275e12602ac36685b6749b51cf3eb2c4d55eb9a270edb3df7f3bd2911c4`.

The pack has 322 payload files totaling 64,678,337 bytes, excluding its manifest.
It contains 16 different frames from one real ALOHA episode, ten complete typed
inputs per frame including the original saved noise, and both full output ports:
normalized F32 `[1,50,32]` and official native-transform F64 `[1,50,14]`.

All inputs were checked against the SHA-bound official preprocessed NPZ arrays.
Noise bytes additionally match the earlier independent original-dataset audit.
All 32 complete output pairs are byte exact; MSE/max-abs are zero and the smallest
computed cosine is 0.9999999999999998. The F64 state and native output were not
downcast for packaging. A target backend must support these types or explicitly
declare a host/device partition; no target numerical equivalence is inferred.

The public `board_handoff.prepare` and `validate` APIs were reused without model
name dispatch or runtime changes. The data-only protocol records an explicit
path-relocation ledger and adds the already verified converted checkpoint SHA
`07e8a2ef8438e3cf839bc0992b2f2e07a95dac6049b156696bd47bbe79cb1518`.
It does not modify the frozen CUDA experiment or rehash checkpoint weight files
during this copy. Source records were checked unchanged after preparation.

A full temporary-directory copy passed validation with the same manifest hash.
All eight actual Orin/BPU preflight/build/run/collect dispatches returned exit 2
and `pending` because SKU/SDK/provider/driver/device details are unknown. No
board driver was launched. This does not supply a runnable board deployment,
weights, target engines, physical calibration, latency, power or thermal data.

Evidence is alongside the pack: `report.json`, `source-audit.json`,
`data-protocol.json`, `selection.json`, and `pending/`. The first attempt
`pi0-dual-data-001/failure.json` records an omitted local input-index retrieval;
it failed before creating a pack. The original index was then retrieved and
checked against the frozen hash before preparing the new 002 directory.

## Revalidation

```sh
REPO=/home/zhangzimo/Repos/private/edge-fm-x
PACK=$REPO/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/pi0-dual-data-002/pack
env CUDA_VISIBLE_DEVICES= python "$REPO/vlaforge/tools/board_handoff.py" validate \
  --pack "$PACK" --report /absolute/new/pi0-pack-validation.json
```

Use a new report path. Independently pin the manifest SHA before transfer, then
repeat validation at the destination. See `doc/specs/board_target_handoff_v1.md`
for the target descriptor and real driver obligations. Source CUDA bundle paths
inside the provenance are historical references, not Orin/BPU build inputs.

## pi0.5 Additional Pack

The same public preparation/validation API also completed a separate pi0.5 pack:
`/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/pi05-dual-data-001/pack`.
Its manifest SHA256 is
`5257c4c07c0b758eb19c5cbeef4c2aeeb813a211c47ae0e9fbb9813061af6d63`.
It has 328 files totaling 64,715,108 bytes, excluding the manifest, and the same
two complete output shapes/dtypes as the pi0 pack. All 160 input tensors match
the pi0.5 official prepared NPZ bytes, noise matches the original saved-noise
audit, and 32 full reference/direct output pairs are byte exact.

This pack derives from the completed pi0.5 ordinary CUDA formal-003 protocol,
not pi0 references or pi0.5's short replay experiment. The only added protocol
field is the already verified converted checkpoint SHA256
`7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6`.
All original protocol fields remain unchanged. Source tensor files were checked
unchanged after copying, and a separate temporary-directory copy validated with
the same manifest hash. See its adjacent `report.json` and `source-audit.json`.

No new model inference or board execution occurred. The source `/dev/shm` loader
root and x86 bundle paths are provenance only. Both target templates remain
pending. Revalidate with the command above using this pack's own path and a new
report filename; do not use pi0 weights or its manifest pin for pi0.5.
