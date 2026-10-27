# OpenPI TensorRT Preparation

Checkpoint: 2026-10-22 15:45 UTC. **Host preparation, not Orin model support or
formal acceptance.** RDT retains the target. No target runtime was replaced.

## Verified Pieces

| Piece | Actual Evidence | Still Not Proved |
|---|---|---|
| Eight original ONNX exports | Original checkpoint/source pins; parent independently rehashed all 114 delivery files | Numerical/target fidelity |
| Source F64 native finishing | Parent rebuilt the strict-FP C++ core and independently reproduced 128 complete chunks / 89600 native F64 values byte-exact | Native Region/target integration |
| External-data structural lowering | Revision003: 51 focused tests, full 5.636 GB prefix and static 37-split/110-access step checks; weights/types unchanged | Real neural outputs or TRT support |
| Scalar DOUBLE sin/cos rewrite | 13 generic tests; two real step candidates preserve binary64 frequency bits, with 512 frequencies each | Actual CUDA/target numerical behavior |
| Scalar plugin implementation | CPU NVCC compilation/guarded metadata checks using real TRT8.5 headers; DOUBLE kernel/PTX, FLOAT interface | Orin CUDA11.x build, numerical/replay smoke |
| Source-derived timestep shadows | Original initialization and time-update graphs drive ten steps per model; 20480 feature values byte-exact on CPU | Full neural actions or target inference |

Combined current host software regression: **185 passed**, archived as
`artifacts/orin/20261022-08/native-types-sequence-sincos-regression-001.xml`.
This count is software verification, not experimental-model completion.

## No F64 Demotion

The model-neutral `vlaforge/tools/lower_onnx_scalar_sincos_f64.py` matches the
original scalar FLOAT-to-DOUBLE Cast, DOUBLE multiply, sin/cos, ordered Concat
and terminal FLOAT Cast. Only bounded constant dataflow may supply frequencies;
runtime/overridable parameters, shared observed intermediates, non-scalar time,
different precision/order or oversized constant evaluation are rejected.
Unrelated learned nodes and weights remain unchanged.

The same 512-value binary64 LUT appears in both source graphs, SHA256
`8c97769281adf81d75a55cd56f1de76bcd2122f4c75a0b66d7d0feab408bd3cc`.
Signed INT32 low/high words encode exact binary64 bits as plugin attributes;
they are not numerical frequency casts. `VLAForgeScalarSinCosF64/1` accepts a
FLOAT scalar view and computes actual DOUBLE multiplication/sin/cos before
FLOAT RNE output. There is no model-name dispatch or FLOAT sine approximation.

`pi0-step-sincos-001/` and `pi05-step-sincos-001/` under root08 are
**source-level candidates**, not the final target recipe. External weights are
copied and rehashed, not loaded as huge protobuf tensors. The CPU shadow
preserves original frequency operations/results, but CUDA libdevice and full
target actions are deliberately not certified by that shadow.

The actual final recipe must apply structural Sequence lowering before creating
producer interfaces, and apply the BF16-carrier pass before inserting the new
sin/cos plugin. The old consumer partition exposes constant indices as runtime
ports and is correctly rejected by static Sequence lowering. The original
rejection is not erased by using the correct pass order.

## Source Checks

- [Postprocessing source equations and original CPU validation](../../artifacts/orin/20261022-06/postprocess-report.md).
- Parent rebuilt library and 128-pair validation:
  `artifacts/orin/20261022-08/postprocess-parent-review-001.json` and
  `postprocess-review-validation-001/receipt.json`.
- [External-data structural lowerer](orin_external_sequence_lowering_20261022.md),
  with fixed source/control snapshot races in
  `artifacts/orin/20261022-09/revision-003.json`.
- [Scalar plugin compilation/ABI handoff](orin_scalar_sincos_plugin_preparation_20261022.md).
- `artifacts/orin/20261022-08/scalar-sincos-source-shadow-004/audit.json` records
  all twenty complete 1024-value feature comparisons and exact FLOAT timestep
  bytes from the original source update, not an independently invented schedule.
- Earlier shadow precheck failures are retained by
  `scalar-shadow-precheck-failures-001.json`: valid empty internal shapes and
  incomplete constant annotations required a bounded-dataflow proof in the
  reference tool. Source math and numeric gates were not changed.

## Remaining Integration

By 17:25 UTC `openpi-target-graph-catalog-001.json` lists **eight** syntax-checked
opset17 graphs: prefix, step, initial time and full finite predicate for each
model. Exposed neural graph ports are FLOAT/BOOL/INT32 only. The original
pi0 state Cast is explicitly externalized, pi0.5's unused F64 ports are
projected, and token narrowing is permitted only at standard Gather index
uses with mandatory runtime int32 checks. The source F64 state remains
separately available to native finishing. None of these graph-only edits
implements that checked runtime boundary or proves model/target support.

The generic input-preparation tool has eight focused tests. Opset lowering now
resolves literal integer Reshape/shape-only axes without freezing runtime
inputs; an additional regression prevents NumPy decoding from causing writes
to immutable external axes. Earlier `node_mean` rejection and the private
external-weight negative test remain in `opset-initial-boundaries-001.json` and
the retained red XML. These are converter limits/fixes, not neural failures.

Precision profile `002` requests FP16 for dense operators and FP32 for original
normalization, masks, non-dense arithmetic and time updates. In particular,
all 165 native LayerNormalization nodes per prefix and FLOAT mask selection
are explicitly covered. Finite predicates are separate non-neural FP32 control
graphs so full-range FLOAT classification bounds are not implicitly narrowed.
The engine builder must obey constraints and actual target precision still
needs inspection. There is no full-model FP32 fallback.

The current combined host regression is **245 passed** under
`openpi-target-preparation-regression-001.xml`, including the new input,
opset, precision and predicate coverage. Native postprocessing Region and
checked-cast CPU ABIs are separately prepared; CUDA runtime validation and
source-compatible finishing/control integration remain open.

By 16:10 UTC all four prefix/step carrier graphs were prepared on the host,
under `artifacts/orin/20261022-08/trt-carrier-001/`, with 3176/598 rounding
boundaries for pi0 prefix/step and 3176/670 for pi0.5. This uses explicit
FLOAT storage plus BF16 result boundaries; it is not proof of hardware compute
precision or permission for full-model FP32 fallback. The conversion ran
sequentially under a 24 GiB address-space ceiling with hidden GPUs.

The carrier-first step variants then received the scalar plugin under
`pi0-step-carrier-sincos-001/` and `pi05-step-carrier-sincos-001/`. Both retain
the exact same source LUT and portable-shadow hashes as the earlier
source-only candidates. `openpi-carrier-preparation-001.json` binds all graphs,
sidecars and reports. These still require explicit checked I64/I32 index
interfaces, unused F64 input projection and the original F64 state Cast
producer. No TensorRT parser or engine has run on these candidates.

Native decoded image/text/state/noise preparation, generic Region wrapping of
the F64 finishing core, and explicit checked F64/F32 and I64/I32 boundaries are
being prepared independently. These are necessary non-neural preprocessing,
conversion or finishing; all learned neural computation remains TensorRT.
Any host staging/synchronization must stay in measured E2E, not be hidden as
zero-copy or omitted from latency.

After RDT completes and service restoration is actually audited: create a fresh
integrated target source/runtime snapshot, run native plugin/Region checks,
build declared FP16-dense/precision-island engines, assemble the original full
prefix and ten-step loop, then validate every complete normalized/native action
before pilot/formal sampling. No full-model FP32 fallback, token shortening,
action-horizon reduction or quality-threshold relaxation is permitted.
