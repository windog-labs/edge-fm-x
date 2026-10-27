# OpenPI Parallel Host Export Preparation

Date: 2026-10-22. Status: **eight real neural/auxiliary ONNX graphs published;
host source preparation only; both models remain target-unqualified**.

All new artifacts are in `artifacts/orin/20261022-06/`. No production exporter,
runtime, board JSON, pre-existing artifact, NAS source or global environment was
modified. No Orin connection, target build, target inference or GPU export was
performed. The parent's RDT experiment retained exclusive Orin ownership.

## Recovered And Verified Sources

Both complete local checkpoint files were rehashed and their safetensors headers
counted, independently of the inherited assessment. The NAS copies were also
rehash-verified through authorized read-only `ssh zzm-4090-x8-1` access.

| Profile | BF16 Checkpoint Elements | Checkpoint SHA256 |
|---|---:|---|
| `pi0_aloha` | 3501372176 | `07e8a2ef8438e3cf839bc0992b2f2e07a95dac6049b156696bd47bbe79cb1518` |
| `pi05_aloha` | 3616757520 | `7ed2fb2f91b084efc387022383035f6908aa620f6bfa0da52204a4e52e7851b6` |

Local checkpoint directories:

```text
/home/zhangzimo/.cache/edgefm-vla-goal/openpi/pi0_base_strict_pytorch_chunked_001
/home/zhangzimo/.cache/edgefm-vla-goal/openpi/pi05_base_strict_pytorch_chunked_001
```

NAS root: `/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906`.
Its complete checkpoint directories are
`assets/openpi/weights/pi0_base_strict_pytorch_torch210_cpu` and
`assets/openpi/weights/pi05_base_strict_pytorch`.

The original OpenPI revision remains
`215abfb217dbac7d5f1273282331b9b1866c0479`. All ten pinned local upstream files,
normalization assets and the actual tokenizer were verified. The tokenizer's
SHA256 is `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`.
The inherited manifest and raw-source-provenance pins were independently checked.
Each checkpoint still includes all 263323648 zeroed unused parameter elements.
`source-capture-binding-001.json` also connects the historical capture reports'
checkpoint, revision, complete model config and ten-step declarations to the
current NAS archive hashes; historical numerical/performance results are not
imported into this task's qualification.

Actual source exports are native `torch.export` archives, not TorchScript and
not executable TensorRT engines. pi0's original four archives and examples were
downloaded, without source changes, from:

```text
NAS/runs/openpi-pi0-v2-phased-001/reference/exported_regions/
  -> artifacts/orin/20261022-06/sources/pi0/
```

All four downloaded archive SHA256 values match the current NAS originals.
pi0.5 reuses **unpruned reference** exports in:

```text
/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/
  openpi-inventory/local-pi05-v2-pruned-001/reference/exported_regions/
```

Despite the ancestor directory name, these reference archives retain the full
captured state; no `derived-capture` or pruned compiled package was selected.
Compared with NAS
`runs/pi05-h20-20260908-001/pi05-h20-v2-capture-20260908/exported_regions`, all
weight and constant members match **SHA256 exactly**. Functional serialized graphs
match exactly after excluding debugging metadata. Stack traces and some saved
example values differ, so the complete archives are explicitly distinct pins.
This comparison does not merge H100/H20 numerical references or establish
inference equivalence. The original H20 raw reference roots remain authoritative.

## Actual Published Exports

Existing local base Python has Torch `2.10.0+cu128`, ONNX `1.21.0`, ORT `1.24.4`
and ONNXScript. The suggested `vlaforge-openvla/bin/python` does not exist on
this machine. No Torch or dependency installation was performed.

`host_export.py` is request-driven and has no model-name branches. It reuses
Torch's native archive reader/deserializer and public `torch.onnx.export` API.
The pinned Torch version is required because its archive-loading APIs are
internal. Source hash checks precede loading. The only requested relocation is
CUDA to CPU: serialized tensor/device metadata and example tensor placement.
Every changed metadata location is recorded. Raw tensor payloads, dtype fields,
shapes, scalar constants and source files are not rewritten.

Exports use ONNX opset 18, `optimize=False`, external weights and no runtime
verification. CUDA is hidden and checked uninitialized. Neural reference forwards
were not executed. OMP/MKL/OpenBLAS threads are each bounded to two. Every export
directory has an `owner.json` containing PID, start time, argv and thread settings.
The parent's later resource observation supersedes the initial low-utilization
GPU query: all eight remote GPUs have foreign owners; none were used here.

Each row below contains `model.onnx`, optional `model.onnx.data`, an exporter
report and an immutable `receipt.json` under `exports/<directory>/`.

| Directory | Nodes | Graph Bytes | External Weight Bytes |
|---|---:|---:|---:|
| `pi0-prefix-001` | 10323 | 33422798 | 5636685824 |
| `pi0-step-001` | 3492 | 7875223 | 636026880 |
| `pi0-time-001` | 5 | 4412 | 0 |
| `pi0-finite-001` | 12 | 9136 | 0 |
| `pi05-prefix-001` | 10323 | 33178251 | 5636685824 |
| `pi05-step-001` | 3843 | 8660667 | 1097596928 |
| `pi05-time-001` | 5 | 4667 | 0 |
| `pi05-finite-001` | 12 | 9748 | 0 |

Total graph/weight publication: **13090160358 bytes**. These are real neural
prefix and iterative policy graphs, not shape-only ONNX or cached encoder
outputs. The historical `vf_cached_openpi_prefix_59` symbol names the actual
online three-image/text prefix computation; it must execute on every observation.

The ONNX exporter emits only referenced state. It does not change or prune the
complete checkpoint/source archive. Smaller per-region initializer counts and
aliased source-state element totals are not new whole-model parameter counts.
BF16 checkpoint storage also does not mean all captured computation is BF16:
the original captured graphs already contain mixed F32/BF16 parameters and Cast
operations, which were retained as source semantics rather than replaced by a
whole-model precision conversion. Numerical equivalence remains untested.

## Preserved Contracts And Checks

`publication-audit-001.json` independently reopens every published graph after
export, reruns the filename-based checker and verifies graph/sidecar hashes.
It also verifies the original archive identities and the pi0.5 member comparison.
All eight graphs pass. Audit SHA256:
`fef0aeaff570ac26f4a46cbbcec7bff428ad4101e7da3344e5ea976dddc219be`.

`typed-port-audit-002.json` compares all input/output counts, ordered shapes and
dtypes directly against the original serialized source signatures. All eight
pass. Only the finite graph output name changes (`reshape` to `view`) through
standard exporter decomposition; both positional mappings are explicit.

- pi0 retains I64 `[1,48]` token IDs, prefix length 816 and F64 `[1,32]` state.
- pi0.5 retains I64 `[1,200]` token IDs, prefix length 968 and its distinct
  discrete-state conditioning contract. Both its prefix and step expose F64 state.
- Both steps retain F32 `[1,50,32]` samples and F32 scalar time. Prefixes return
  the padding mask plus 36 BF16 KV tensors, rather than stored encoder features.
- Both original invocation IRs retain a `vla.for` loop with lower 0, upper 10,
  step 1. The loop was neither shortened nor absorbed into a different scheduler.
- Saved F32 noise remains authoritative. Examples are compile assets, not a
  substitute for the frozen 16-observation noise/reference sets.
- Native actions remain F64 `[1,50,14]` in the frozen full contract. These four
  phases stop at normalized actions; a native postprocessing graph was not
  exported or implicitly replaced with an F32 path.

Six focused CPU tests pass (`cpu-tests-002.xml`): device-only relocation,
dtype/shape/scalar preservation, changed-source rejection, immutable receipts,
explicit CUDA hiding and an F64 neural archive roundtrip. These tests and ONNX
checker passes are not full-model fidelity or target qualification.

Preliminary environment/path errors are retained in
`preparation-failures-001.json`. The first typed audit's overstrict name check
and its log are retained separately from the later positional ABI audit.
There were no failed neural export attempts. No source gate was relaxed.

## Handoff And Next Gate

`requests/` pins all eight inputs; `export-jobs.json` maps them to output
directories. `source-runtime-contract-001.json` binds the upstream/tokenizer and
the separate original raw-host profiles. Complete published graph and external
weight hashes, initializer types and ordered ONNX port names are in
`publication-audit-001.json`; the final `delivery-manifest-002.json` binds the
task scripts, requests, receipts and this report.
The first delivery manifest is retained as superseded because it hashed an
active finalization log before that log's final line was written. Its graph and
weight pins are unaffected; the second manifest is created without an active
log in its inventory and is independently checked afterward.

Exact CPU command for an additional immutable export attempt, from this worktree:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python artifacts/orin/20261022-06/host_export.py \
  --request artifacts/orin/20261022-06/requests/pi0-openpi_step.json \
  --output artifacts/orin/20261022-06/exports/pi0-step-002
```

The next deployment blocker is a declared, source-faithful precision/partition
and processor implementation for these mixed graphs: native raw image/text/state
preprocessing, I64 token handling, F64 state and original Cast boundaries,
BF16 KV/weight arithmetic, the unchanged ten-step chain and F64 native output
postprocessing. The parent's new F64/I64 byte-preserving host input work is
compatible with these assets; it is not evidence of F64 TensorRT arithmetic.
Some available historical native-output `.pt2` files are AOTInductor CUDA
packages rather than portable neural export archives and were not relabeled.

No full-model reference comparison, TensorRT parse/build, target provider,
native E2E, pilot, formal sampling, CDF or resource/performance result is added
by this task. Export publication does not increase formal-model completion.
