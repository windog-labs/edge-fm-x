# OpenPI Source Partitions

Checked 2026-10-22 14:28 UTC. **Static source/weight validation only; no native
provider, TensorRT compilation, numerical comparison or target inference.**
The original eight published exports and their immutable manifest are untouched.

The existing model-neutral `partition_onnx_values.py` separates dependency-closed
F64 producers at their original FLOAT Cast outputs. It neither demotes DOUBLE
arithmetic nor moves a cast across another operation. Both complete iterative
graphs, including the original solver outputs, remain represented.

| Source | Original Nodes | Producer Nodes | Neural Consumer Nodes | Producer Live Inputs |
|---|---:|---:|---:|---|
| pi0 step | 3492 | 55 | 3437 | F64 state, F32 time |
| pi0.5 step | 3843 | 54 | 3789 | F32 time |

The producer has no learned initializers. It includes the original DOUBLE
frequency/time trigonometry and final FLOAT conversion; pi0 also includes the
original F64 state-to-F32 Cast. Both frontiers expose eight tensors, including
shared I64/FLOAT constants. Those routes must be represented explicitly, not
discarded when constructing a native/target chain. All source dtypes are kept.

`artifacts/orin/20261022-08/openpi-source-partition-audit-001.json` records an
independent static check of both results:

- Serialized source nodes occur exactly once across producer and consumer;
  there are no modified, missing or duplicated source operators.
- Every source initializer is retained with identical metadata; every external
  weight sidecar matches the original source bytes and recorded hash.
- Original input types and producer-to-consumer frontier types match.
- Consumer operators have no internal F64 outputs. Their unused F64 `state`
  input remains in the conservative partition interface and is **not** falsely
  counted as a supported TensorRT binding.

The generic partitioner intentionally keeps all original arguments. A later
explicit child-interface projection still needs to remove unused F64/BF16
producer arguments and unused consumer state while preserving the outer model
contract and source routes. The existing `project_region_inputs` pass can
adapt Region calls only after that artifact projection is proved.

Reproduction, using fresh output suffixes:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/tools python \
  vlaforge/tools/partition_onnx_values.py \
  --input artifacts/orin/20261022-06/exports/pi0-step-001/model.onnx \
  --value-name _to_copy --value-name _to_copy_1 \
  --output artifacts/orin/20261022-08/pi0-step-source-partition-NEW

CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/tools python \
  vlaforge/tools/partition_onnx_values.py \
  --input artifacts/orin/20261022-06/exports/pi05-step-001/model.onnx \
  --value-name _to_copy \
  --output artifacts/orin/20261022-08/pi05-step-source-partition-NEW
```

The checked `*-001/` directories contain both real ONNX pieces, exact sidecars
and explicit partition routing reports. This prepares a source-faithful path
for native DOUBLE math or a generic numerical plugin; it does not implement
either, certify low-precision neural arithmetic, solve static Sequence support,
or replace complete-action validation. All target model gates remain open.
