# Readonly Exp Repair

## Latest Closed Diagnostic Checkpoint

2026-10-25 11:58:09 UTC: independent17audit verifies all15actual calls,
735full outputs/112657920elements finite and every state carry/control edge.
Same-actual-prefix portable token comparison still fails. First recurrent
state is restored, but complete source logits are not equivalent; no model
or primary/formal acceptance is claimed. All owned Orin jobs are terminal,
no service lease remains and original manager/program states are unchanged.

The next query-scale hypothesis is actually excluded: original3node
Pow/Reciprocal/Mul and readonly-folded1node real-query outputs are both
byteexact to CPU on Orin, MSE0/cos1. The residual error is not demonstrated
to come from that isolated scale expression.

Three intrusive first-mixer observation variants parsed with all requested
names matched but cannot build under the unchanged original precision:
TensorRT reports an FP16Shuffle versus public FP32 output conflict in the
Slice-to-Cast13region. This is an observation-layout boundary, not a claim
that the original model is unsupported; its full decoder already builds and
runs. No precision constraints were relaxed to make observers compile.
The failed sources, graphs, builds, logs and terminal receipts are archived.

Next useful checks: compare the repaired decoder using the already validated
CPU-generated prefill state, to separate the old corrupted native-prefix
state from ordinary numerical error; use only the original three mixer
outputs or smaller precision-compatible node probes instead of forcing many
FP32observation outputs. Both corrected prefix stages have full CPU parity
and remain ready for target build after selecting the next qualified path.
These source diagnostics do not redefine the fixed same-Orin/same-precision
ordinary/optimized gate. Fresh raw chain, resident Session,16-observation
primary quality, TTFT/full16/CDF/memory and2Btarget measurements remain open.

## Full-Decoder Checkpoint

Window17successfully built engine4030963431B/SHA12b873c6 with zero unmatched
dynamic requests. Independent closure verifies all pins/empty cgroups/no
service lease, minimum6894680KiB/no guard. Its one real decoder call has
all49outputs/7510528elements finite. First recurrent state improves from
MSE0.00084036/cos0.42178 to MSE2.65044e-9/cos0.99999869. However final logits
still have MSE2.46307/cos0.92747 against the same-input CPU reference. This
partial fix is NOT complete numerical equivalence or primary acceptance.
All15recurrent calls completed finite with exact declared state routing, but
the source token discrepancy remains. No formal sampling is claimed.

Both original prefix stages now have the same repair prepared(8and10readonly
Exp sites), old dynamic precision preserved plus previously missing FLOAT
Exp/Softplus constraints. Before/after CPU chains validate all50outputs of
each stage byteexact for the real original inputs. They have not been rebuilt
or measured on target yet. A source-derived first-layer query-scale branch,
readonly Pow/Reciprocal plus real-query Mul, is next under actual microprobe;
its CPU readonly fold is exact. Further source differences remain supplemental
and do not replace the user's same-Orin/same-precision ordinary/optimized gate.

## Actual Localization

Source-derived9node gating graphs reproduce the full decoder's numerical
problem. All four intermediate-output probes confirm byte-exact runtime input
echo and original parameter Cast output. Softplus error is at most4.77e-7.
The earliest bad value is the readonly parameter Exp result, sometimes with
very large invalid values, before it affects recurrent state. The precise
internal importer/builder stage is not identified and no external issue is
used to claim this is a published SDK defect.

The official pinned TensorRT8.5trtexec runner reproduces the same wrong output
from both existing first-boundary engines. Its exported values agree with
the raw runner within SDK JSON decimal precision. Printed trtexec timing is
not used as a paper metric. This rules out the custom raw runner as the sole
cause of this particular constant-expression failure.

## Qualified Mitigation

The already installed NVIDIA GraphSurgeon prototype folds only readonly FLOAT
Cast/Exp expressions. Runtime inputs, dynamic Exp, model parameters and dense
precision are not changed. CPU outputs remain byte-exact. All four actual
Orin branch cases are then finite and pass the unchanged source operator gate:

| Branch | Policy | Before | Repaired MSE | Repaired Max Abs |
| --- | --- | --- | ---: | ---: |
| First recurrent | 13 | MSE0.2021835742 | 6.8001e-16 | 5.9605e-8 |
| First recurrent | 14 | MSE0.3147366510 | 6.8001e-16 | 5.9605e-8 |
| First nonfinite | 13 | 4of16NaNs | 4.9960e-16 | 5.9605e-8 |
| First nonfinite | 14 | MSE0.1278241013 | 4.9960e-16 | 5.9605e-8 |

This is a compiler-time immutable-expression repair, not model-specific
constants, runtime CPU fallback, whole-model FP32 or quantization search.
The standard mechanism is described in [NVIDIA's constant-folding example](https://github.com/NVIDIA/TensorRT/blob/main/tools/onnx-graphsurgeon/examples/05_folding_constants/README.md).

The common tool `vlaforge/tools/fold_onnx_readonly_unary.py` uses the existing
CPU ONNX Runtime for bounded readonly FLOAT Exp evaluation. It leaves other
precisions, overridable parameters, runtime data and all original parameter
TensorProtos untouched; identity FLOAT aliases only establish provenance.
External extents are bounded and checked; derived nonfinite results fail
closed.111related software tests pass. This is not full-model target quality.

## Complete Decoder Integration

All18readonly parameter Exp sites are replaced by FLOAT literals;18dynamic
Exp sites remain. Complete51input/49output ABI, original parameters,
187nativeBF16 MatMul,18FP16dense and13sourceFLOATdense sites are unchanged.
Complete before/after portable15step chains match all49outputs each step
byte-for-byte;16tokens match the fixed-cache CPU route. Existing source logit
differences remain supplemental, not mathematical-lossless or primary quality.

Window16correctly rejects18old source-layer storage selectors because literal
results no longer have those runtime layer names. No engine/inference, no
guard or service stop; independent closure rehashes all assets and verifies
empty cgroups and unchanged original programs/configuration. The projection
now explicitly discharges only those18compute/storage layer obligations into
typed FLOAT literal descriptors. Every dynamic request is retained unchanged;
the strict native unmatched-name check is not relaxed. Graph/weights/parity
are reused unchanged for window17. Main same-Orin paired quality, fresh raw
chain, resident Session and formal metrics remain open; five VLAs unchanged.

## Evidence

- `artifacts/orin/20261025-15/operator-audit-001.json`
- `artifacts/orin/20261025-15/readonly-expression-target-audit-001.json`
- `artifacts/orin/20261025-15/folded-preparation-001/report.json`
- `artifacts/orin/20261025-17/readonly-unary-neighbor-green-001.xml`
- `artifacts/orin/20261025-16/decoder-assets-001/preparation.json`
- `artifacts/orin/20261025-16/readonly-full-parity-001/report.json`
- `artifacts/orin/20261025-16/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-17/decoder-assets-001/preparation.json`
- `artifacts/orin/20261025-17/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-17/single-step-all-output-audit-001.json`
- `artifacts/orin/20261025-17/repaired-step-source-comparison-001.json`
- `artifacts/orin/20261025-17/native-recurrent-20261025-17-001/terminal-audit-001.json`
- `artifacts/orin/20261025-18/prefix-assets-002/preparation.json`
- `artifacts/orin/20261025-18/prefix-complete-parity-001/report.json`
- `artifacts/orin/20261025-15/query-scale-preparation-001/report.json`
- `artifacts/orin/20261025-17/recurrent-source-boundary-001/report.json`
- `artifacts/orin/20261025-15/query-scale-actual-audit-001.json`
- `artifacts/orin/20261025-15/mixer-observer-failures-001/report.json`
