# Orin Parser Diagnosis And Static Control Audit

2026-10-23. Final bounded handoff under `artifacts/orin/20261023-03/`.
No SSH, Orin/GPU job, build job, learned inference, SDK change or precision-policy
change was performed by this worker. All prior artifacts and precision-plugin
CU/header/smoke files remain unchanged. Parent owns target execution and later
engine format/precision investigation.

## I64 Warning Audit: Actual V4

The actual graph is
`artifacts/orin/20261022-08/pi0-step-carrier-sincos-001/model-trt-opset17-v4.onnx`,
SHA256 `368e13f00bf9b729b5a8f3096043a48fd5a9214ed996638d791d9ed5ca66ad00`.
The read-only audit scans **1318 INT64 Constant tensors**, including dead cones.
There are **zero INT64 initializers**, zero sparse initializers, no other I64
tensor-attribute owner types, and no I64 public graph inputs/outputs. No integer
literal was skipped or left unscanned.

Exactly **38 literals contain out-of-INT32 values**. Every oversized value is
INT64_MAX, exactly decimal `9223372036854775807`, hex `0x7fffffffffffffff`.
The saturation value is INT32_MAX, exactly `2147483647`, hex `0x7fffffff`.
No graph or weight was modified to obtain these results.

| Classification | Count | Actual Uses / Proof |
| --- | --- | --- |
| Dead control cone | 1 | `node_Constant_114` / `val_114` reaches 37 retained Cast nodes, none reaching a declared graph output |
| Live Slice-end saturation equivalent | 36 | `node_slice_2`, `_4`, ... `_72`: axis 3, dimension 256, start 128, step +1; both original and saturated ends normalize to 256, selecting indices 128..255 |
| Live Slice-end saturation equivalent | 1 | `node_slice_73`: input `_to_copy_98` shape `[1,51,1024]`, axis 1, start -50, step +1; both ends normalize to 51 and start to 1, selecting 50 entries |
| Unproved oversized literals | 0 | No live use outside the proved Slice-bound scope |

The machine report `i64-audit-001.json` includes all literal ranges, original
and saturation payloads, immediate uses, **complete downstream node-index
closures**, liveness, reachable public outputs, and each Slice's original and
saturated bounds, normalized indices and selected count. Its node-index table
maps every index back to an exact node/SSA. JSON integer payloads are exact;
use Python or another integer-preserving reader for 64-bit values rather than
JavaScript Number. The summary also records oversized decimal values as strings.

### Independent Shape Check

Original internal `value_info` annotations are removed in a private in-memory
shape-proof copy. The copy adapts 634 BF16Round operations to Identity **for
shape inference only**, using their verified shape-preserving plugin contract.
The scalar sin/cos output shape is seeded from its immutable creator field word
count: `[1,1024]`. No learned result is synthesized or executed. Strict ONNX
shape inference runs with `data_prop=False`; all 73 Slice data-input shapes are
recovered. For every affected live Slice, the result agrees with its original
static source annotation. No inference shadow is saved as a candidate graph.

The proof compares original and saturated controls under ONNX Slice bound
normalization. It checks axes/steps remain unchanged, axes are unique and in
range, data dimensions are independently known, and normalized start/end/step
triples match. The current oversized live uses all have **positive step +1**;
there are no negative-step oversized uses to qualify in this graph.

This establishes value/shape equivalence of the enumerated literal saturation
within this exact static graph. It does not approve arbitrary I64 narrowing,
runtime IDs, state conversion, a different graph/profile, TensorRT kernel
precision, model outputs, or a formal gate. Runtime IDs/state retain their
separate checked provider contract from root14. Original failed attempts remain
failures; later engine format failures are outside this audit.

Reproduce without target access or a graph write:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python3 artifacts/orin/20261023-03/audit_i64_narrowing.py \
  --output artifacts/orin/20261023-03/i64-audit-review.json
```

## First Crash And Minimal Constant Pair

The first failure receipt reports SIGSEGV (`-11`) after 361.383 monotonic seconds,
with no engine. Its log shows 36 successful cache-input BF16Round imports and
no scalar sin/cos import. The subsequent debugger trace ends at
`node_unsqueeze_2`, with `memcpy` and `libnvonnxparser.so.8` stack frames.
It records the earliest semantic divergence:

1. Real source `node_Constant_1` uses `value_ints=[1]`, meaning INT64 vector `[1]`.
2. TensorRT 8.5.2 reports that result as an empty INT32 vector of shape `[0]`.
3. Concat stays empty, so Expand of rank-zero `time` remains rank zero.
4. Unsqueeze with the correctly shared axes `[1]` then encounters an invalid
   axis for a scalar. In the original graph Expand produces `[1]`, making the
   exact same Unsqueeze legal and its output `[1,1]`.

No time/input rank change or axes substitution is justified. Tensor-valued
Constant canonicalization preserves the original contract. Parent reports v3
cleared that crash; v3 then exposed genuine constant-dataflow Slice axes that
TensorRT treated dynamically. Parent v4 clears parsing after the static-control
folder below. These checkpoints do not mean an engine or model gate passed.

The extracted five-node, plugin-free pair is published at:

| File Under `minimal-001/` | SHA256 |
| --- | --- |
| `scalar-expand-unsqueeze-source.onnx` | `7e3c8dd02abf2c8286e7fae249c0473617e2c6afb70949fda6c075b77860d625` |
| `scalar-expand-unsqueeze-canonical.onnx` | `da96c8cd63f69d8665f4175206c1d091f35acb08615a17e4068064f4b78619e5` |
| `time.f32` and `expected.f32` | `e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c` |

Only `node_Constant_1` changes encoding. Axes are already tensor `[1]` in both
graphs. `time.f32` is one FLOAT scalar, value 1.0, four bytes; output is FLOAT
`unsqueeze_2` of shape `[1,1]`, four bytes. Both graphs pass ONNX checking and
six complete CPU byte comparisons including signed zero and a subnormal.
The receipt pins the actual source graph and each extracted source node.
These `.onnx` files exist even if default `rg --files` ignores them.
No target import of this pair was run by this worker.

## Canonicalizer Read-Only Review

The parent tool `vlaforge/tools/canonicalize_onnx_constants.py` was reviewed
read-only at SHA256 `d97af506182d0e967f633d4eca8f99354cb63ccf466591b4c92f0d4b54bf7d2b`.
No actionable semantic defect was found for its declared finite numeric-literal
scope. Scalar/vector rank and original INT64/FLOAT dtype are preserved; empty
vectors stay shape `[0]`; signed zero and subnormals are preserved. An additional
independent 4080 finite binary32 literal values matched complete original bytes.
Nonfinite FLOAT attributes are explicitly rejected; string/sparse/tensor forms
are unchanged. Their unchanged status is not a TensorRT support claim.
Initializers and external-weight descriptors are copied without materializing
or rewriting learned weights. The parent retains its own canonicalizer tests
and target qualification. No helper/plugin source edit was made for this crash.

## Static Control Folder Handoff

The only new production paths authorized during this task are:

- `vlaforge/tools/fold_onnx_static_controls.py`
- `vlaforge/tests/unit/test_fold_onnx_static_controls.py`

The tool evaluates only requested control inputs of standard Slice, Reshape,
Expand, Unsqueeze and Squeeze. Slice starts/ends/axes/steps are inspected;
missing axes are reported without inventing them. Dataflow is restricted to
Constant, Identity, Cast-to-I32/I64, Reshape, Concat, Unsqueeze and Squeeze,
with integer inputs/results only, 4096 elements per value, and bounded rank/depth.
Reshape is checked before reference execution/allocation: zeros, -1 inference,
element counts, shape product, rank and dimensions must be safe. The existing
validated bounded tensor reader is reused from the scalar sin/cos helper;
its broader float-math evaluator is not used because it lacks Reshape and
exceeds this task's integer-only scope.

Runtime/overridable inputs, floating producers, mean or other arithmetic,
custom-domain nodes, oversized dataflow and out-of-range I32 casts remain
unresolved. Proven final producers become tensor Constants with unchanged
name, SSA, dtype and complete payload; other nodes, learned weights, public I/O
and shared observers remain unchanged. The flat-graph CLI requires fresh output
beside source weights and serializes metadata without saving external data.

**27 unit tests pass**, including the real Cast/Reshape Slice pattern, shared
observers, I32/I64 identity, INT64 extremes, empty vectors, all permitted forms,
unsupported/runtime/overridable/float dataflow, preallocation rejection, cycles,
literal budgets, and fresh CLI publication with unchanged external bytes.
The first test attempt had 26 passes and a temporary-directory setup error;
creating the new diagnostic root resolved setup without changing an assertion.
Final XML is `static-controls-tests-002.xml`.

Tool SHA256: `c0c5e961feb08d1d7dad930c32fc93ba269b6b85dea792077931eaf822914ef7`.
Test SHA256: `d6fa44a2b8dbad3c2567027b57031cc55ca8588ad8269026dbbb3e239709aba2`.
Both remain unchanged after the parent's pin/stage003 request.

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python3 vlaforge/tools/fold_onnx_static_controls.py \
  --input /path/beside-weights/source-v3.onnx \
  --output /path/beside-weights/fresh-v4.onnx
```

## Evidence Pins

The first build and debugger evidence is under root00 `pi0-step-build-001/`
and `pi0-step-debug-001/`; the real Slice-axes rejection is retained in
`pi0-step-build-002/build.log`. The new diagnostic scripts, receipts, minimal
graphs, proof inventory and source hashes are all under root03. `source.sha256`
and `delivery-001.json` identify this delivery. None of this marks any full-model
fidelity, numerical source gate, target performance or final deployment complete.
