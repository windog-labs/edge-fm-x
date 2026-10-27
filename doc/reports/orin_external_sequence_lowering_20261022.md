# External-Data Structural Sequence Lowering, 2026-10-22

The existing generic lowerer now rewrites external-data ONNX graphs without
loading neural weights into TensorProto or NumPy. The final revision passed
41 focused tests and filename-based checking of the real pi05 prefix and full
step exports. Source graph/sidecar hashes and typed input/output protobufs
were unchanged. This is host structural evidence only: real-model numerical
fidelity, TensorRT acceptance, Orin execution, and target performance remain
unverified.

## Implementation

Changed [lower_onnx_split_sequences.py](../../vlaforge/tools/lower_onnx_split_sequences.py)
and its [focused tests](../../vlaforge/tests/unit/test_lower_onnx_split_sequences.py).
All experiment output is under [20261022-09](../../artifacts/orin/20261022-09/).
Original exports, partitions, their manifests/reports, board state, and the
other agents' code were not edited.

- Load ONNX metadata with `load_external_data=False`. Index initializer and
  Constant tensor metadata without decoding every payload. Decode only scalar
  INT32/INT64 controls directly required by SplitToSequence/SequenceAt;
  external controls use bounded reads at their declared byte offsets.
- Shape inference uses a metadata copy with only those scalar controls
  temporarily inlined, with data propagation disabled. Restore the original
  external descriptors afterward. Remove the previous NumPy contiguity sweep;
  protobuf tensor payloads already store contiguous bytes.
- Reuse `external_files` and `publish_files` from
  `export_torchscript_region_onnx.py` without editing that file. Discovery
  includes nested graph attributes, sparse tensors, and local functions.
  Reject unsafe source paths/symlinks, duplicate metadata keys, malformed or
  out-of-file byte ranges, external tensors mixed with inline data, graph/file
  aliases, destination symlinks, and existing outputs/sidecars.
- Copy complete sidecars, including unused padding/trailing bytes, in 8 MiB
  chunks into a private staging directory. Hash the stream and independently
  rehash the staged copy; reject source changes during copying. Check the
  staged graph by filename before publishing weights and then graph using the
  existing no-overwrite publisher. Stage cleanup and publication rollback
  preserve a racing writer's graph. Output sidecars are separate copies,
  with no hard links to original weights.
- Keep relative locations, offsets, lengths, tensor descriptors, and default
  typed ports unchanged. Preserve F64 and BF16 by default. All four existing
  dtype-changing flags explicitly reject external-data models before
  publication: `--demote-f64-island`, `--promote-bfloat16-to-f32`,
  `--demote-spurious-complex-casts`, and `--normalize-cumsum-bool`. Existing
  inline F64/BF16 flag tests still pass. No scheduler or loop was changed.
- Generate collision-free names and key replacements by SSA output names.
  Reject vector split specifications, out-of-range accesses, exposed sequence
  outputs, nested sequence captures, unknown split extents, and runtime scalar
  control ports (including overridable initializers). An absent shape is not
  treated as a scalar shape. CLI reports use exclusive creation.

The supported sequence form remains a positive scalar split size,
`keepdims=1`, and nonnegative constant accesses with a known split-axis
extent. Nested graphs are preserved and checked for safe external references;
this pass does not recursively rewrite their sequence operators. Sidecar
collisions are rejected even when existing contents would match, so callers
should choose a fresh output directory.

## Focused Verification

Command:

```sh
python -m pytest -q vlaforge/tests/unit/test_lower_onnx_split_sequences.py
```

Result: **41 passed**. `git diff --check` passed for the two changed code
files. Exact command, output, and code identities are recorded in
[focused-tests.json](../../artifacts/orin/20261022-09/focused-tests.json).

Coverage includes bit-exact ONNX Runtime CPU equivalence for a small external
fixture with offset controls, shared nested sidecar paths, padding/trailing
bytes, preserved unused DOUBLE input, and independently loadable publication
after deleting the fixture's source. ORT cannot infer the original
SplitToSequence with external scalar controls, so only the tiny reference
fixture is loaded inline for that comparison. The lowerer itself remains
guarded against external loads and neural-weight NumPy conversion.

Other cases cover nested external Constant attributes and their unsafe paths,
external Constant scalar controls restored after inference, invalid byte
ranges, duplicate metadata, collisions, source/destination symlinks, racing
publication rollback, failed-check cleanup, explicit precision rejection,
runtime control ports, generated names, nested sequence captures, and the
original lowering/idempotence/precision tests.

The sparse 3 GiB initializer regression guards against payload decoding and
oversized protobuf serialization; it substitutes sparse file creation for
the bulk copy to keep the unit test small. Actual streaming, hashing, and
bounded-memory operation are exercised separately by the real 5.636 GB
prefix run below.

## Real Graph Evidence

Final revision runs use
[validate_real_graph.py](../../artifacts/orin/20261022-09/validate_real_graph.py).
The harness forbids external tensor loading and non-scalar NumPy decoding,
requires a filename checker, hashes source graphs and sidecars before/after,
and compares exact typed ports and external tensor descriptors. All default
precision flags were used. No real model was executed.

| Final Run | External Bytes | Structural Result | Process VmHWM |
| --- | ---: | --- | ---: |
| [pi05-prefix-structural-002](../../artifacts/orin/20261022-09/pi05-prefix-structural-002/validation.json) | 5,636,685,824 | 162 CastLike, 81 Shape, 227 same-type Cast rewrites; filename check passed | 421,680 KiB |
| [pi05-step-structural-002](../../artifacts/orin/20261022-09/pi05-step-structural-002/validation.json) | 1,097,596,928 | 37 SplitToSequence and 110 SequenceAt replaced by Slice; 239 same-type Cast rewrites; filename check passed | 154,116 KiB |
| [pi05-consumer-structural-002](../../artifacts/orin/20261022-09/pi05-consumer-structural-002/validation.json) | 1,097,596,928 source | Rejected before publication: sequence index `val_9` is a runtime graph input | 71,700 KiB |

The successful runs retained 598 and 207 external tensor descriptors,
respectively. Prefix resolved zero scalar controls; full step resolved four
inline scalar controls. Neither decoded external weight payloads. All three
runs confirmed unchanged source graph and sidecar hashes. Each successful
run preserved input/output protobufs exactly, including BF16 and DOUBLE.

Memory numbers use Linux `/proc/self/status` VmHWM for the Python process.
The `getrusage` field in receipts retained a 762,760 KiB launcher high
watermark even for an empty Python process, so it is not used in the table.
These are host utility memory observations, not Orin/model performance.
Memory scales with graph metadata and fixed copy/hash buffers rather than
the total external weight size.

The consumer partition exposes `val_9`, `val_11`, and `val_31` as INT64 scalar
ports rather than retaining their Constant producers. Assuming their values
would specialize its source-preserving contract. The static lowerer therefore
rejects it without publishing an output or changing its unused F64 state
port. Parent-owned partition/IR work must retain static control producers or
explicitly provide a different runtime-capable lowering. Earlier `001`
receipts, including the first consumer failure, are retained as prior-revision
evidence; `002` identifies the final code revision.

Reproduction, each with a fresh run name:

```sh
python artifacts/orin/20261022-09/validate_real_graph.py --input artifacts/orin/20261022-06/exports/pi05-prefix-001/model.onnx --run-name prefix-new
python artifacts/orin/20261022-09/validate_real_graph.py --input artifacts/orin/20261022-06/exports/pi05-step-001/model.onnx --run-name step-new
python artifacts/orin/20261022-09/validate_real_graph.py --input artifacts/orin/20261022-08/pi05-step-source-partition-001/consumer.onnx --run-name consumer-new
```

## SHA256 Identities

```text
lower_onnx_split_sequences.py
b52b5ba77f2963a33d43611cb75dbbc51ec8ddc4fc999bf11a2b1482f33fa30d
test_lower_onnx_split_sequences.py
cd18f5e16d71a268fe3b52652014a3b53244b96409fe739ca0084f49d98fe28a
reused external helpers, read only
83f019970a4bd0af5bacd21d2dc70b58b66d487646387645dd65eed3bbf56908

prefix source ONNX
72e2c30551fa4066112460bd54458a78eb3e0629ba23838585956a716dbddab7
prefix lowered ONNX (001 and 002 identical)
8a9f96ca3d4cd2ecfc649ea9c88914a3a67b42fb0dd74519fb14711186137c1f
prefix source and output model.onnx.data
5d40b7a6804eeb059d3e846abbe4bf6ec7cec2d3b6234a90dc7b566de5a2230e

step source ONNX
1dc37febee4e48331820e8db85e5bb92a10b4682295eaea62aa187d159d6c350
step lowered ONNX (001 and 002 identical)
6b9f78ae123add757b66ae7bb2a71e175e4e96306f8ee553586d3de45f8b0f86
step source and output model.onnx.data
bf3f91c4cdb3267abd1ee738e9e619bd59c33c1a8eea1b8285511f5bd87a5640
```

The legacy `requires_numerical_verification` report field only tracks explicit
F64/BF16 conversions. Its default `false` does not certify structural output
fidelity; `full_model_output_verified` remains `false`. No Orin/GPU access,
installs, other-platform model benchmarks, target dtype partition changes,
or formal campaign main-table updates were performed.
