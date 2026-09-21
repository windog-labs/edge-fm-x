# VLA Complete-Output Data Handoffs

Status: five model data/reference packages exist and validate offline. **No
Orin/BPU model has been executed or accepted.** These are not runtime bundles.

Root: `/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/`.
Each row identifies a directory under this root; its immutable package is `pack/`.

| Directory | Real observations | Complete public outputs | Files / bytes, excluding manifest |
| --- | --- | --- | --- |
| `smol-dual-data-001` | 16 episodes, one frame each | normalized + official inverse-normalized F32 `[1,50,6]` | 315 / 139950196 |
| `rdt-dual-data-001` | 16 histories, one episode | unified BF16 `[1,64,128]` + robot BF16 `[1,64,14]` | 265 / 88012821 |
| `pi0-dual-data-002` | 16 frames, one episode | normalized F32 `[1,50,32]` + native F64 `[1,50,14]` | 322 / 64678337 |
| `pi05-dual-data-001` | 16 frames, one episode | normalized F32 `[1,50,32]` + native F64 `[1,50,14]` | 328 / 64715108 |
| `cogact-data-002` | all 115 frames, one episode | raw/normalized F32 `[16,7]`, native F64 `[16,7]`, RNG U8 `[16]`, draws I64 `[1]` | 3351 / 283085128 |

## Manifest Pins

```text
smol-dual-data-001  1554ec215a3a02967ff56cec334918892ded6e0e507288689edff0fa6026eede
rdt-dual-data-001   2fbf37103e22a49026fefe59a81fce14fb777dfe6d2ace615eab923a6dde2514
pi0-dual-data-002   5f455275e12602ac36685b6749b51cf3eb2c4d55eb9a270edb3df7f3bd2911c4
pi05-dual-data-001  5257c4c07c0b758eb19c5cbeef4c2aeeb813a211c47ae0e9fbb9813061af6d63
cogact-data-002     c39053b6e02b351f3790326356b40b7bb522853993db770aef7431ea4b1c6b47
```

Each package was copied to a temporary directory and revalidated with the same
manifest SHA. Source inputs and saved noise retain their original complete bytes;
no inference, random draws, weight conversion or precision change occurs during
packaging. The manifests retain all offline acceptance flags as JSON `false`.
Both targets' preflight/build/run/collect dispatches return exit 2 / pending
without launching a driver. Adjacent `report.json`, `source-audit.json`, and
`selection.json` retain source evidence. All five packs' 40 actual shell-wrapper
dispatches are recorded under `five-pack-pending-dispatch-001/`; report SHA256
`51363bcee12803246f11468fe9a3fac9ed6816d1d3b7fdf2e5e44f6bf104ecb4`.
This adds the previously absent pi0.5 stage invocations without rewriting its
original preparation report. Driver launches and board successes remain zero.

## New Evidence

Smol rechecks all 34560 complete tensors from source formal002 against the
original normalized and official processor outputs. It retains the recovered
so100 statistics profile. The old normalized-only packages and their action-space
erratum are unchanged; physical calibration remains unverified.

RDT uses the completed ordinary dual-output-formal-002 source, with all eight
Region numerical bindings as provenance. Its unified and robot outputs are both
retained; the current three-policy experiment is a separate run. Numerical
provider enforcement on CUDA is not silently inherited by a board target.

CogACT independently rechecks all 5750 raw native tensors from ten processes,
including the two exact RNG receipt ports, against pinned NPZ inputs and complete
reference arrays. This remains the public-dependency configuration candidate:
original gated Meta configuration equivalence is unverified. RNG is an external
producer's explicit saved tape, not autonomous C++ RNG. Full native F64 storage
is retained without downcasting. No task success or physical calibration claim.

The first CogACT preparation failed before publishing a manifest because the
historical `independent-audit-002.json` references previous-audit SHA `255fd4...`,
while the available previous file hashes to `9fa971...`. The remote directory
contains raw process outputs but not that prior audit. This historical hash link
is **not accepted**. Both original files and `cogact-data-001/failure.json` remain
unchanged. New `cogact-data-002/source-audit.json` records the mismatch and fresh
full byte comparisons; SHA256:
`1a4f1ac1905df8c6846309777ff4ffa41a2503ceb07c4e110140301391fbae18`.

## Common Interface

The new `vlaforge.board_data_protocol/1` is independent of a formal benchmark
schedule. It allows all 115 real frames without inventing warmup, measured-call,
process-count or quality-gate fields. Existing benchmark v1/v2 behavior remains
unchanged. Models supply Module + typed outputs/references + provenance, with no
model-name branches in the public packer. See
`doc/specs/board_target_handoff_v1.md` for the exact schema and driver contract.

Targeted tests: 103 passed. Current-source isolated CPU regression013: **2074
passed, 83 skipped, zero failures**, source hashes unchanged. Regression report
SHA256 `a4402430bb65ca3564fdcf547b043a41fd3fe180e855e764bc162ebe4fd990a5`.
Skipped optional dependencies/CUDA/model tests are not hardware acceptance.
The existing Smol environment separately ran the statistics-migration and input
pack unit tests with safetensors installed: 20 passed, including the ten cases
skipped for that missing dependency in regression013. This supplementary CPU
result does not rewrite the original full-suite counts or establish new model
inference. XML SHA256 `f1d50395a28d017d2defdd89269ae83072369f146c059af8c4d4d9bdf20e4f28`.

`board-handoff/independent-five-packs-001/report.json` additionally rehashes all
4581 delivered files (640441590 bytes), checks all 179 sample contracts and 703
complete output pairs, independently reproduces NPY/raw storage including BF16,
and recomputes floating metrics without converting integer receipts. Ten actual
temporary-copy negative checks reject one changed output bit and a false board
acceptance flag for each pack. Original packs are not modified. Audit SHA256:
`e6ab8ebfd4051f035b361e4e34e635be11304e78e03aa1b1c5d3217585f39a57`.

## Next Commands

```sh
REPO=/home/zhangzimo/Repos/private/edge-fm-x
PY=/home/zhangzimo/.venvs/vlaforge-validation-py313-20260906/bin/python
PACK=$REPO/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/cogact-data-002/pack
env CUDA_VISIBLE_DEVICES= "$PY" "$REPO/vlaforge/tools/board_handoff.py" validate \
  --pack "$PACK" --report /absolute/new/offline-validation.json
```

Pin the selected manifest before copying. Use new report paths. Real D1/D2 still
require confirmed SKU/SDK/memory/owners, a supported provider and installed
SHA-bound driver, new target builds, full target numerical output validation,
formal latency/CDF, thermal/power records and same-platform ablations. Historical
x86 bundle paths are provenance only and cannot build a board executable.
