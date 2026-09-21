# Explicit Immutable Constant Precompute

## Scope

`vlaforge.analysis.constant_precompute` implements a model-independent,
compile-time static-precompute ablation. It changes a copy of an exported
TensorRegion, not the surrounding Invocation IR, KV ports, cache keys, state
transactions, scheduler, or C++ runtime. It is not a selected Agent microkernel.

The caller explicitly names immutable lifted parameters and persistent buffers.
No user input is made constant because it matched an example. A snapshot is a
contract for the artifact lifetime, not a mechanism that makes shared Tensor
storage read-only. Concurrent state mutation violates that contract.

## API

```python
from vlaforge.analysis.constant_precompute import (
    declare_immutable_snapshot, file_sha256, precompute_constants,
    save_precompute_bundle,
)

snapshot = declare_immutable_snapshot(
    program, immutable_state_targets,
    snapshot_id=explicit_snapshot_identity,
    source_artifact_sha256=file_sha256(source_archive),
)
result = precompute_constants(
    program, output_nodes=selected_fx_node_names, snapshot=snapshot,
)
ledger = save_precompute_bundle(result, new_output_directory)
```

`result.control` is an unchanged graph copy. `result.folded` preserves the
original external input/output signature and all original state tensors, adding
only separately owned persistent constant buffers. The source EP is not edited.
This first version deliberately does not remove unused weights or share the two
new buffers; those would be separate memory optimizations.

The pass requires a static exported profile and an explicit pure-ATen allowlist.
It refuses dynamic input leaves, unapproved/nonpersistent state, unknown calls,
hidden RNG or mutation, unsafe/view output aliases, externally exposed aliases,
downstream writes, nonfinite tensors, and unsupported tensor layouts. Arithmetic
uses the existing tensor dtype and device; BF16 raw hashing does not cast to FP32.

The ledger binds the original archive SHA, canonical graph/topology/signature
SHAs, dependency metadata and raw SHAs, literal arguments, output SHAs, inserted
buffers, removed nodes, implementation SHA, and newly serialized EP SHAs.
Archive identity and graph identity are distinct. Current save validation checks
the snapshot, both graphs, and new buffers before and after serialization;
failure does not publish a successful ledger. These checks do not establish
full-model quality or runtime performance.

`tools/precompute_exported_constants.py inspect` reads only the Torch serde-v8
JSON record and hashes the archive. Its report explicitly marks tensor hashes,
alias/effect checks, and model validation as not performed. The `precompute`
mode additionally requires an expected source SHA, snapshot identity, and
explicit immutable state target names. Failed attempts retain `failure.json`.

## Real Three-Way Validation

Evidence root:
`artifacts/edgefm-vla-goal/20260906-044509/operator-profiles/constant-precompute-vision-position-v1/`.

The authoritative source is the original `fresh-smolvla/recovered-spaced16`
prefix EP, SHA `e23f0ca316410b1e7e466704601eda970d3c1184bcdf4083910d8ddaaecedeb8`.
The earlier H20 microbenchmark's derived EP SHA `627b1687...` is separate evidence,
not the source for this integration. Its roughly 12% Embedding kernel result is
not attributed to static precompute.

The two selected operators are vision positional Embeddings, not text-token
Embedding. In the actual graph, both depend only on the same frozen position
weight and persistent fixed-position IDs through `unsqueeze` and static
`expand([1, -1])`; each output feeds a nonmutating `aten.add.Tensor`.

Three new prefixes were compiled using the public device-preserving,
unoptimized TorchScript API: original, unchanged clone-control, and folded.
The control excludes retracing alone as an explanation for a candidate change.
All three retain the other three original TorchScript Regions and the original
ten-step Invocation/scheduler. No Region split or Python native bridge is used.

- Each variant: all 16 real observations, all 33 prefix outputs bitwise equal
  to the original EP; input hashes unchanged before/after each prefix call.
- Each variant: all 16 complete `[1, 50, 6]` FP32 action chunks bitwise equal to
  the recorded official reference and previously verified original TS output.
- Each variant: one native C++ Session over 32 fresh calls, 16 observations twice,
  bitwise equal to that variant's direct artifact and the official reference.
- Native linkage and dynamic process maps contain no `libpython`.
- A separate CPU-only ZIP/raw audit reread all 48 direct and 96 native outputs.
  All 501 original state tensors, totaling 906,647,648 storage bytes, and their
  metadata are identical in original/control/folded archives. The folded archive
  adds exactly the two declared buffers, totaling 3,145,728 bytes.

The successful GPU artifacts use frozen `source-constant-precompute-v1`.
The later save-window guard is separately CPU-tested and does not retroactively
change their implementation identity. `independent-raw-audit.json` independently
checks their serialized state and new constants without another model forward.

Formal ordinary-only three-way timing completed under `formal/`, using frozen
`source-constant-precompute-benchmark-v1`. Its runtime remains the same stable
TorchScript baseline; only the three previously verified typed benchmark files
are overlaid from `source-rdt-benchmark-v2`. The campaign rotates variant order
across five independent processes per variant, with 128 warmup and 1024 measured
calls per process. Each variant contributes 5,120 measured calls. All 17,280
formal complete outputs, including warmup, were independently reread and found
bitwise equal to the official reference and each variant's direct artifact.
The 144 pilot outputs are excluded from formal counts and latency statistics.

RTX 3060, ordinary synchronous `torchscript-aten/1`, milliseconds:

| Variant | Mean | P50 | P95 | P99 | Calls/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original TS | 94.813 | 94.748 | 95.808 | 96.360 | 10.547 |
| Unchanged clone-control TS | 94.610 | 94.554 | 95.562 | 96.101 | 10.570 |
| Static-precompute TS | 94.872 | 94.680 | 96.584 | 97.070 | 10.541 |

The folded mean is 0.262 ms (0.277%) slower than clone-control. The exploratory
paired-process bootstrap interval for latency reduction is [-1.191, 0.568] ms;
there are only five independent process repeats, not 5,120 independent trials.
This campaign does not establish a stable speedup or a statistically secure
regression. The pass removes six FX nodes, including two Embeddings, but adds
3 MiB of logical constant storage while retaining all original state.
`selected_for_deployment` remains false. The earlier kernel microbenchmark is
not a selected or reintegrated Agent skill as a consequence of this experiment.

`formal-comparison.json` and `.csv` retain independent statistics, process means,
all raw evidence hashes, and worst-latency sample byte offsets.
`formal-cdf.png` shows the complete tail, an explicitly labeled central zoom,
and all five process-level paired traces. Individual generic reports retain
cosine/MSE/max-abs and complete action bytes. Independent auditing also checks
sample/revision ordering, process maps without Python, and every recorded GPU
owner PID. Approximately one-second owner sampling cannot exclude a transient
foreign process entirely between samples.

The boundary is recorded model tensors on CUDA, not sensor-to-actuator, physical
robot execution, Orin/BPU, or full-paper acceptance.

## Remaining Generic Partition Work

Dynamic RoPE/Q/K computation cannot use this state-only pass. Generic FX
partitioning still needs a frontend helper that clones the graph, partitions
selected node sets, binds only used immutable attributes into children, maps
all live-in/out ports, and replaces the original `vla.invoke` with child calls.
Existing TensorRegion/Plan/artifact APIs already support mixed TS/AOTI children;
public compiler/runtime model-name branches are unnecessary.

Important first-version gates are contiguous boundary tensors, no alias escape,
flat complete KV output maps, conservative preservation of original cache
invalidation dependencies, and stable parent-to-child provenance. The C ABI
currently has no arbitrary stride/storage-offset fields, and ordinary Region
execution performs synchronization/output copies. A microsecond-level kernel
gain may be lost at these new boundaries. Compare original monolithic TS,
same-partition all-TS, and same-partition selected-AOTI before selecting a skill.

## Tests

```sh
python -m pytest vlaforge/tests/unit/test_constant_precompute.py \
  vlaforge/tests/unit/test_operator_inventory.py \
  vlaforge/tests/unit/test_torchscript_export.py -q
```

Latest focused result: 42 passed in the CPU-only validation environment,
including separate before/during-save tamper checks; Ruff passed for all three
new Python files. `cpu-final-v3.xml` and `ruff-final.stdout` retain the results.
GPU validation and formal measurements are separate evidence stages.

## Literal-Only Capture Preparation (2026-09-08)

The same module now supports explicitly declared literal-only expressions:
`declare_immutable_snapshot(program, (), literal_only=True, ...)` or the CLI
`--literal-only` switch, mutually exclusive with `--immutable-state-target`.
No state or example-input dependency is approved in this mode. The original
archive and graph identities remain mandatory; an implicit empty declaration
is still rejected.

The allowlist additionally covers explicit `arange`, division, exponential and
tensor conversion. Factories require explicit dtype and device. An aliasing
`to` is eligible as a fresh selected result only when metadata proves a dtype
or device change, or `copy=True` is explicit. Metadata assertions are retained;
they are not erased to make constant producers disappear. Conversion is done
once using the original device/dtype/math, not by rebuilding CPU constants in
CUDA arithmetic.

This closes a real capture-preparation gap found in CogACT: its fixed 128-value
frequency vector was recomputed on CPU and synchronously moved to CUDA inside
every denoising step. Only the fixed 512-byte vector is precomputed. Actual
timestep, CFG, scheduler, noise tape and loop carry remain dynamic. The new
step was freshly captured and compiled; complete N10 carry outputs and three
GPU graph replays matched the original byte-for-byte. The complete updated
model's 16-frame Python IR reference also matches all five original outputs.
Fresh run 003 additionally passes a no-Python C++ required pilot: 48 calls /
240 full output tensors, 10 captured steps / 48 replay / 0 ordinary fallback.
The shared five-port independent audit and 12 real-evidence tests pass. These
short calls establish the complete native recovery, not formal performance.

Evidence lives in `cogact-replay-20260908/` locally and remote run
`cogact-replay-20260908-003/`, with earlier failures retained. Run 002's
`fresh-capture.json` held only a support-status report and could not certify a
bundle; run 003 uses the existing public `save_exported_region` API and fresh
complete-model references. See `doc/reports/cogact_execution_variants_20260908.md`.
Focused precompute/export tests: 41 passed. Full CPU regression 014: 2079 passed,
83 skipped, zero failed; neither result substitutes for CUDA or board evidence.
