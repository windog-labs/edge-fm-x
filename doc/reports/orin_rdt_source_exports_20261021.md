# RDT Orin Source Recovery and Export

Date: 2026-10-21. Status: **source preparation verified, target model unqualified**.
This checkpoint does not add an RDT row to the Orin E2E performance table.
Subsequent target solver qualification and a non-exact conditions pilot are
recorded in [RDT solver and conditions](orin_rdt_bfloat16_solver_20261021.md).
The direct-BF16 parser failure below remains valid historical evidence, not a
claim that the later explicit carrier route was never executed.

## Source and Execution Boundary

The user directed this task to `ssh zzm-4090-x8-1`, whose shared NAS contains:

```text
/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/
  runs/rdt-torchscript-v1/artifacts/
  runs/rdt-gpu1-torch210-capture-0923/capture/exports/
  runs/rdt-torchscript-series16-v2/
```

The seven TorchScript archive hashes and seven example-input hashes match the
existing local historical compile/capture records. Source revision remains
`cd79363a1387e8f81c7724d070ef7e45fd23150f`. No checkpoint was substituted.
The historical `vf_cached_*` archive names contain the actual text/vision
encoders; they do not authorize replacing online encoders with cached features.

Preparation ran on CPU with `CUDA_VISIBLE_DEVICES=""`, using the existing
`envs/rdt-aoti-cu128-py311` environment, Torch 2.10.0+cu128. ONNX 1.21.0 and
ORT 1.24.4 imports use task-local `host-deps`, not a global environment change.
No 4090/H20/H100 model performance experiment was run. Orin did not install or
execute Torch/torchvision. CPU export durations are not inference benchmarks.

NAS preparation/output root:

```text
/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/orin-trt-prep-20261021/
```

The abandoned web download and its temporary localhost SSH forward were
identity-checked and stopped after the replacement access path was verified.
Completed small inputs and original failure logs were preserved.

## Published Exports

ONNX opset 17, original BF16 source profile. Each archive executes its example
forward on CPU before export. ONNX checker validation passes; ORT runtime
parity is explicitly **skipped**, not passed, for these BF16 exports.

| Region | Nodes | ONNX Bytes | External Files | External Bytes |
|---|---:|---:|---:|---:|
| `rdt_initialize` | 4 | 33,181 | 0 | 0 |
| `rdt_solver` | 60 | 5,129 | 0 | 0 |
| `rdt_finish` | 16 | 1,146 | 0 | 0 |
| `vf_cached_rdt_conditions_25` | 67 | 56,132,721 | 0 | 0 |
| `vf_cached_rdt_vision_20` | 1,312 | 826,130,890 | 0 | 0 |
| `rdt_denoiser` | 6,887 | 662,500 | 497 | 4,261,425,152 |
| `vf_cached_rdt_text_17` | 1,639 | 159,793 | 219 | 9,524,621,312 |

The independent publication audit rehashes each original archive/input, reopens
each final ONNX after temporary staging cleanup, checks the exact external-file
inventory, verifies every external-file size/hash and reruns the checker by
filename. Result: `published_sources_verified`, all seven regions; 716 external
files, 13,786,046,464 external bytes. The complete published graphs/weights total
14,669,171,824 bytes. Graph hashes alone would not bind those external weights.

The first CPU attempts for T5 and the denoiser failed because their traced
factory calls explicitly used `cuda:0`, despite `map_location="cpu"`.
Separate retries use opt-in device-constant relocation. Each changes exactly
one Device constant to `cpu`, not dtype, weights, scalar constants, action
dimension, scheduler or iteration count. The original archives are unchanged.
The first campaign's `partial_export` receipt remains intact; the independent
final audit covers the successful retries rather than rewriting that history.

## Actual Target Boundary

Target: `ssh orin`, AGX Orin serial `1421524312594`, system TRT 8.5.2.2.
Boot ID at this check: `3c801f5d-6897-4d97-b021-3c52db38a750`. The board's civil
clock is wrong; recorded durations are monotonic. MemAvailable was approximately
29,154 MiB at inspection, not a guarantee for a later large engine build.
Unrelated services were preserved.

The original solver graph, SHA256
`f570e039a044ebba30dbbdf43c74f7ccb77e9b8bd9a4b52770349d9d8663f997`,
was transferred to Orin and passed to native `trtexec` with `--fp16 --noTF32
--buildOnly`, workspace 256 MiB. The child **exited 1** before engine creation:

```text
Unsupported ONNX data type: <UNKNOWN> (16)
ERROR: input_0 ... Failed to convert ONNX date type to TensorRT data type.
```

The collector exits 0 after recording the child result; the authoritative
receipt says `compile-failed`, not successful engine creation. `--fp16` does
not make a BF16 graph input supported. There is no RDT target inference result,
complete-chain fidelity result or target performance measurement at this point.

## Remaining Gates

1. Declare one supported deployment precision/port contract, with generic graph
   lowering or plugins where required. Preserve the original BF16 graph and
   reference; explicit BF16 rounding in the solver cannot be silently removed.
   No whole-model FP32 fallback or quantization search was attempted here.
2. Validate all stages and the complete fresh chain: online text and six-view
   vision, conditions, all five original DPM solver steps and final output.
   The real pipeline includes approximately 4.762B T5 and 0.428B vision active
   parameters in addition to the 1.228B policy; nominal 1.2B is not total capacity.
3. Compare complete frozen-input actions on all 16 recorded observations.
   Use active/native `[1,64,14]` dimensions, not padding-diluted `[1,64,128]`
   errors. Keep native units and source-profile changes separately identified.
   Gate MSE < 1e-5 and cosine > 0.9999 before formal sampling; do not relax it.
4. Complete native Session and raw-observation E2E, then the required five
   independent processes, each 128 warmup + 1024 measured calls, full outputs
   and CDF. Export time and isolated-stage time cannot substitute for E2E.

SmolVLA's prior qualified results are unchanged. RDT, pi0, pi0.5 and CogACT
remain unqualified on Orin; no model was removed from the five-model Goal.

## Code and Evidence

The model-agnostic exporter now preserves external files with per-file hashes,
uses path-based checking for large graphs, rejects unsafe external paths,
publishes without overwriting existing files, binds surviving inputs by name,
records opt-in traced-device relocation and reports early reference failures.
Focused regression: **133 passed**; 13 cover the new export behavior directly.

- [Exporter](../../vlaforge/tools/export_torchscript_region_onnx.py)
- [Export regression tests](../../vlaforge/tests/unit/test_export_torchscript_external_data.py)
- Local evidence root: `artifacts/orin/20261021-01/rdt-recovered/`.
- `source-export-audit.json`: independent final source/publication audit.
- `host-source-exports/`: exact export reports, initial failures and retry logs.
- `solver-bf16-build-001/`: actual target parser failure, child identity and SDK logs.
- `export-regression.xml`: the 133-test host regression receipt.
- `host_export_campaign.py`, `audit_host_exports.py`: reproducible host steps.

The export/audit and target build processes have exited. This report is a
progress checkpoint, not completed experiment delivery.
