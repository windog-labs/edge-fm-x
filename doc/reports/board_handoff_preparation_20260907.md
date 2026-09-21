# D1/D2 Real-Input Handoff Preparation

## Action Space Erratum

The original v1 pack incorrectly called the SmolVLA output "native-scale".
Its original manifest SHA and payloads are preserved, not rewritten. The frozen
Module explicitly says `normalized_tensors_to_fresh_normalized_action_chunk`;
the actual saved finish graph only slices dimension 2 to `[0:6]`, makes it
contiguous and checks finiteness. The pack copied `reference.npy` unchanged.
Thus all 300 values per sample are **normalized model actions**, not inverse-
normalized robot commands. Static ZIP/JSON graph inspection and independent
checks of all 16 original/copy byte streams are recorded in
`board-handoff/smol-action-space-erratum.json`, SHA256
`be84580ece2254f19bc0de9de9e252417e9b5eed131999b1ddadd7e1f326166a`.

This exposes a functional G2/G3 gap, not just a label: a pure-Tensor Adapter
output Region must apply the verified `smolvla_processing` inverse normalization,
retain normalized output as an auxiliary port, then undergo new capture, compile
and complete native-output C++ checks. Public IR/runtime need no model-name
branch. The old formal measurements exclude that operation and cannot be
retroactively upgraded. Physical units and robot calibration remain separate.

## Delivered Scope

This is an offline **data/reference** delivery, not a board runtime or experiment.
Orin and BPU remain unexecuted and unverified. Neither an existing x86 bundle nor
the historical RTX formal latency results are promoted to board evidence.

The public interface and driver obligations are in
`doc/specs/board_target_handoff_v1.md`. New model integrations supply a Module,
the existing typed Session protocol and provenance. New hardware integrations
supply a target descriptor/provider/driver; the common module has no model-name
branches. Existing `benchmark_session.py` and its C++ runner were not modified.

## Actual Pack

Root: `/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/`

| Item | Verified result |
| --- | --- |
| Current Pack | `smolvla-real16-data-v2-normalized/` |
| Manifest SHA256 | `61f9ce28f0481fadc34c50776261f94b8383ce66adb8ad4f265fab2f07ad505c` |
| Copied files | 241 files, 147,179,332 bytes excluding manifest |
| Real samples | 16 distinct dataset episodes, one spaced observation per episode |
| Input integrity | 16 x 8 complete typed inputs equal their original recovered NPZ bytes |
| Saved noise | Original per-sample hashes match; no noise regenerated |
| Complete output | 16 x FP32 `[1,50,6]`, all 4,800 values preserved |
| Source reference/direct | All 16 complete outputs bitwise equal; MSE/max-abs zero |
| Action meaning | Recovered so100 profile, normalized active six action dimensions; inverse normalization absent |
| Target execution | `board_executed=false`, `board_verified=false` for both targets |

Source protocol:
`execution-context/torchscript-native/formal-context-v2/protocol.json`, SHA256
`a057c42f955dbcce46ede20ad33e7f8dcd884cc73bb48d5b0951af0247e0f2c4`.
The pack retains its historical `provider_enforcement=false` limitation. It does
not acquire a new 22-flag numerical-context guarantee by being copied. No G5
quantized candidate or invented denormalized output was substituted.

The v2 handoff's entire `samples/` file-record map is identical to v1: the same
inputs, original NPY files and raw complete outputs retain their original hashes.
Only the delivery semantics/provenance were corrected, with four appended proof
files. `normalized-v2-prepare-report.json` independently verifies the new pack.
The historical v1 pack remains at `smolvla-real16-data-v1/`, manifest SHA256
`30192dd7524e7cff23da134654b59d003b12184103ec8e548601309889d14cf0`.
Its native-scale labels and the old audit key `complete_300_native_values_bitwise`
are superseded by this erratum; the byte-integrity measurements remain valid.

`real-source-audit.json` SHA256
`b5d780d232b366faaee2092a586ebe0e86478e43b92534fd97f5fa68fec76acd`
independently joins the exact capture source records, original observation
manifest/checkpoint identities, all input bytes and complete official references.
This historical audit checks byte integrity, not action-scale semantics.
These are processor-boundary tensors, not the original camera transport stream;
preprocessing, physical calibration and robot execution remain outside scope.

`delivery-verification.json` SHA256
`f183304a1a1e5fb0bca9b1b384d2540ad19318a27b9d9a815d6791854b17a278`
records the historical v1 real copied/moved pack validation from `/tmp` and eight actual public
shell entrypoint calls. All eight returned exit 2 and explicit `pending`, with
no target child PID or driver launch. This validates the missing-driver gate,
not the absent driver. The relocated pack is under `relocated/`; every file and
the manifest retained the same bytes. No verification reads old absolute source
paths. `final-validator-report.json` repeats the relocated check after the final
manifest-symlink and atomic-publication hardening.

## Commands

Run commands from the local source repository. Use a new output/report path each
time; the tooling will not overwrite prior evidence.

```bash
REPO=/home/zhangzimo/Repos/private/edge-fm-x
DATA=$REPO/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/smolvla-real16-data-v2-normalized
PYTHON=/home/zhangzimo/.venvs/vlaforge-validation-py313-20260906/bin/python
"$PYTHON" "$REPO/vlaforge/tools/board_handoff.py" validate \
  --pack "$DATA" --report /absolute/new/offline-validation.json
```

Before copying the pack to the board, compare `sha256sum "$DATA/manifest.json"`
against the pinned SHA above. Transfer the immutable directory and the local
source snapshot via an isolated dry-run-first rsync, without `--delete`, then
repeat validation at the destination. NumPy plus the repo's Python validation
modules are needed for the offline audit; this does not put Python inside the
eventual native model process.

For a real target, create a new external descriptor based on the selected
template. Fill exact measured SKU, SDK versions, memory, installed SHA-bound
provider/driver paths, device nodes, execution partition and thermal policy.
Currently **no real board driver is supplied**. Leaving any unknown value null
correctly blocks the following entrypoints:

```bash
TARGET=/absolute/new/confirmed-target.json
PYTHON="$PYTHON" bash "$REPO/vlaforge/scripts/board/preflight.sh" \
  --pack "$DATA" --target-descriptor "$TARGET" --output /absolute/new/preflight
PYTHON="$PYTHON" bash "$REPO/vlaforge/scripts/board/build.sh" \
  --pack "$DATA" --target-descriptor "$TARGET" --output /absolute/new/build
PYTHON="$PYTHON" bash "$REPO/vlaforge/scripts/board/run.sh" \
  --pack "$DATA" --target-descriptor "$TARGET" --output /absolute/new/run
PYTHON="$PYTHON" bash "$REPO/vlaforge/scripts/board/collect.sh" \
  --pack "$DATA" --target-descriptor "$TARGET" --output /absolute/new/collect
```

Report integration reuses `vlaforge/tools/report_deployment_metrics.py` and the
existing typed benchmark report code. Actual board collectors must supply fresh
board raw timings, complete output bytes, execution identities and environment
records. An old protocol or an offline metrics report alone cannot fill a board
main-table cell. BPU timing and any CPU fallback require a real BPU driver.

## Verification and Pending Work

- Latest CPU suite: 112 passed, consisting of 47 new handoff tests and 65 existing
  typed Session tests. `tests-integration-004.xml` SHA256
  `93fe91c4e95904f2a1e75e77e98f51cee82024ef78830ccab88b98667d767e99`.
- Negative tests cover missing/unknown schema fields, fake board flags including
  integer/string impersonations, bad hashes/sizes, incomplete outputs, unknown
  dtypes/contracts, path traversal and manifest/input symlinks, create-only
  behavior, and failed atomic manifest publication. Multi-output integer tests
  preserve values above `2**53` without a floating-point round trip.
- The first test invocation found incorrectly omitted input/output IDs in the
  new synthetic test builder. After assigning the mandatory declaration IDs,
  45 tests passed; later symlink and atomic-publication regressions raised the
  new-test count to 47. No real model run was implicated by the fixture error.
- Latest validator source SHA256:
  `5c9e9f6a17ebf30cd8efcdfd28ac31abe3568d9061d4641cfe71c9991d8fc09f`.
  The pack retains its original creation-source snapshot; later stricter
  validation is a separate check, not a rewrite of the old manifest.
- Exact Orin SKU/JetPack/CUDA/TensorRT and BPU SKU/OE/compiler/runtime, board memory
  and owners, target provider/driver, native model artifacts, every actual board
  output, throughput/CDF, thermal/power evidence and same-platform ablations
  remain pending. Four other VLA model data packs have not been created here.
- End-of-task one-shot resource snapshot `resources-final.json`: H20 GPUs became
  idle, H100 remained occupied by foreign jobs. This enabled a separate π0 H20
  task to resume; it does not change any Orin/BPU status in this report.
