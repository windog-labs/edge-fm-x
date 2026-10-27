# Remaining Decoder Boundary

Latest user scope:2Bcancelled; only0.8Bmust close before Goal completion.
Existing quality/measurement gates remain unchanged. Historical2Bplans below
are superseded, not a reason to continue2Bwork.

## Latest Checkpoint

The subsequent repaired-prefix campaign has now executed actual CPP pixels,
both complete prefix engines and all15decoder calls. Final16tokens match
the original frozen full-call reference. Source state/logit differences and
the nonmatching portable continuation from actual native prefix are preserved;
do not infer source losslessness. See
[fresh-prefix joint audit and current remaining work](orin_qwen_fresh_prefix_20261025.md).
Only0.8Bresident/primary/formal closure remains;2Bwas cancelled by the user.

2026-10-25T12:51:05Z: all15actual decoder calls are independently audited:
735outputs/112657920elements finite, all48state carries and token/counter
edges verified. Complete16tokens match the CPU portable continuation from
the exact same archived native-prefix inputs:
`[32,11952,369,3753,4613,264,3565,3618,11,440,264,2526,13358,1576,383,279]`.
This is not a fresh vision/prefill call, source byte-exact logits or resident
E2E acceptance. Same-actual-input per-step logits MSE spans0.000490790274 to
0.001442569180; minimum cosine0.999960687261. These remain supplemental source
differences, not a new primary gate. All owned units are terminal, original
services restored;119related host and13board tests pass. Five accepted VLA
paper campaigns are unchanged. Next: corrected prefix target/fresh full chain,
resident Session,16-observation same-Orin primary pair, formal TTFT/16token/
CDF/memory and final evidence, then close the Goal.2Bis cancelled.
No Qwen formal timing is accepted yet.

## New Evidence

### Reduction Schema Boundary

Both complete single-call target probes have now closed:49outputs each,
7510528elements finite, same first token11952. Against the same archived
native-prefix inputs logits improve from MSE2.46306794/cos0.92747366 to
MSE0.000907548091/maxabs0.171875/cos0.999974543218. With validated CPU-prefill
states the unchanged19engine gives MSE0.000987579156/maxabs0.140625/cos
0.999977046332, first recurrent MSE1.84878733e-9/cos0.999999099274. Both
unit closures preserve the manager/program snapshot. Source errors remain
explicit supplemental differences, not a redefined primary gate or lossless
claim. The full15call state diagnostic is now audited as summarized above;
it is not fresh prefix execution or resident C++ E2E timing.

The19build has since completed and independently closed: engine SHA256
99317ab33986478cae43566276d7fca7e7d5c91f4713986a1111065aa118c754,
4031068039bytes, native311.730028064s, wrapper321.338925216s, outer
324.41666416s.5695network layers/unmatched0, minimum8811088KiB/no guard.
All source/profile/weights/SDK helper pins match; both cgroups are empty.
The only data-collection lease343.93201248s was restored through the original
manager, PID3985484/start41405109. The other three known programs retain
their original births; other programs/configuration are unchanged. All49
outputs of each real decoder call are now checked; complete VLM still pending.
119related converter/precision tests and13board tests pass.

The original130node,3input/3output mixer now also completes on Orin with only
schema compatibility changed. Projection source cosine improves from
0.841550016 to0.999996082, MSE0.00217461218 to1.01469435e-8; conv remains
byteexact and recurrent cosine0.999999564/MSE8.95382881e-10. This confirms the
schema repair eliminates the large mismatch in the intact first mixer, not
only in an isolated norm. It remains a local numerical result, not full VLM.

The complete decoder build was launched in a new guarded19window. Its24GiB
admission needed only conditional temporary stop of data-collection.jar;
post-stop available28061568KiB, other three known services retained their
original births. The controller has a4GiB floor and original-Supervisor
restoration on success/failure/interruption. That build has since closed and
restoration is independently verified above, not a current open lease.

The19node gated-norm probe exposed an actual engine ABI mismatch before any
inference: source opset18ReduceMean uses constant axes input[-1], expecting
FLOAT[16,1], while the TensorRT8.5engine reports FLOAT[1,1]. The following
Add/Sqrt/Div outputs also become[1,1]. The strict source/engine shape check
correctly rejects this candidate, rather than reshaping or synthesizing data.

Applying the existing generic opset18-to17tool represents the same static axes
as an attribute. All19complete CPU subcut outputs remain byteexact. On Orin,
all19actual outputs have the correct shapes and finite values; none violates
the diagnostic MSE/cosine thresholds. The final gated-norm result is byteexact
to its exact-input CPU reference. This is a schema-compatibility diagnosis,
not a precision change or full-model acceptance.

The same suite's attention-core case fails the SDK FP16Shuffle/publicFLOAT
observer tactic, so the suite remains explicitly incomplete; only its
independent completed gated-norm case is audited. A first attempt to stage
the complete130node mixer failed before parsing weights because the diagnostic
stager prefixed a relative external filename incorrectly. Its exact source
and failed logs are retained; corrected staging uses one directory per case.
Neither failure is counted as a successful or numerically failed model run.

The complete decoder schema candidate is now host-validated:79static
ReduceMean axes lower to attributes,18explicitSplit and12ScatterElements retain
their opset17semantics; all original parameters,51inputs/49outputs and precision
maps are unchanged. All49outputs at each of15recurrent CPU steps and all16tokens
are byteexact to the previous portable reference. A missing diagnostic helper
import in host attempt001is preserved; worker002passes with an explicit helper
path. The generic converter's ScatterElements none/add/mul compatibility is
covered by new complete-output regressions; min/max remain rejected.35tests
pass, after the three expected red tests before the implementation.

At2026-10-25T12:26:03Z the independent projection has completed on Orin.
The exact CPU-observed LHS and frozen weight feed the original4node cut:
Transpose, BF16MatMul, BF16Round and the actual following residual Add.
The last is an original operation with its nonempty original FLOAT precision
contract, not a dummy node or a relaxed empty-profile builder. Both complete
outputs are finite, MSE3.46944695e-18/maxabs5.96046448e-8/cos approximately1.
No service was stopped; the owned unit is terminal and Supervisor/programs
unchanged. This excludes projection arithmetic on this exact input as the
large first-mixer error. It does not prove all MatMul cases or full model.

Two declarative source cuts were prepared and launched: gated-norm
19nodes/19outputs, attention-core66nodes/68outputs. All original operations,
weights and precision obligations are retained; complete49parent CPU outputs
and each extracted cut are byteexact against the frozen same-input CPU
reference. Observer boundaries deliberately exclude the original FP16Conv
island. Gated-norm is now audited; the intrusive core observer fails build as
noted above. Neither is formal timing evidence.

The unchanged repaired17engine was tested with valid CPU-generated prefill
states, not the earlier native prefix. All51inputs are bound to the original
source proof and BF16-to-FLOAT carrier conversion is round-trip exact. Actual
native49outputs are finite, first token11952matches CPU. However final logits
still have MSE2.4140321751/maxabs7.578125/cos0.928801443. First recurrent state
has MSE1.8487873e-9/cos0.999999099. Therefore the residual error is not explained
solely by the old incorrect native-prefix states. This is supplemental source
diagnosis, not a new same-Orin primary gate or formal measurement.

The original first linear-attention mixer was then extracted with three
real inputs and its three original outputs, retaining all130source operations
and42172416bytes of original parameter ranges. The parent observer preserves
all49complete CPU outputs; no model, precision or iteration change occurs.
This non-intrusive case actually builds and runs under the original policy.
The conv/recurrent outputs are close to reference, but the mixer projection
output FLOAT[1,1,1024]has MSE0.00217461218/maxabs0.47265625/cos0.841550016.
This localizes a remaining problem within the first mixer path before residual
and MLP. It does NOT establish that MatMul itself is wrong: its input, gated
normalization or upstream query path may differ. The isolated query-scale
original/folded probe already matches CPU byteexact and is not the cause.

## Next Focus

Build the already CPU-validated corrected prefix, join fresh vision/prefill
with the repaired decoder, and then move to native resident execution. The
complete mixer and15decoder steps now pass their declared diagnostics; the
separate intrusive core observer remains a build boundary, not original-model
unsupported.
Avoid marking many FP32public outputs
across the declared FP16Conv/view island: three such intrusive attempts fail
the SDK tactic type check, though their original full model remains runnable.
Keep those failures and do not relax compute/output precision to make probes
compile. Do not rebuild the unchanged full model to seek a different answer.

The corrected two-stage prefix remains host-prepared with8+10readonlyExp
folds and all50stage outputs byteexact before/after. Fresh prefix/vision,
resident C++ Session,16-observation same-Orin paired quality, formal TTFT/
16token/CDF/memory, and2Btarget measurements remain incomplete. The required
five VLA campaigns remain accepted and unchanged. Source differences stay
supplemental; do not substitute these probes for the fixed primary gate.

## Evidence

- `artifacts/orin/20261025-19/recurrent-source-boundary-001/report.json`
- `artifacts/orin/20261025-19/recurrent-source-audit-worker-001/attempt.json`
- `artifacts/orin/20261025-19/native-recurrent-20261025-19-001/terminal-audit-001.json`
- `artifacts/orin/20261025-19/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-19/schema-regression-tests-001.xml`
- `artifacts/orin/20261025-19/single-step-all-output-audit-001.json`
- `artifacts/orin/20261025-19/repaired-step-source-comparison-001.json`
- `artifacts/orin/20261025-19/valid-cpu-state-native-step-audit-001.json`
- `artifacts/orin/20261025-15/opset17-mixer-actual-audit-002.json`
- `artifacts/orin/20261025-15/native-opset17-mixer-20261025-15-002/terminal-audit-001.json`
- `artifacts/orin/20261025-19/build-window-001/job.json`
- `artifacts/orin/20261025-15/native-mixer-subcuts-20261025-15-001/terminal-audit-001.json`
- `artifacts/orin/20261025-15/native-mixer-subcuts-20261025-15-001/native-output/case-0/inspection.json`
- `artifacts/orin/20261025-15/opset17-subcuts-actual-audit-001.json`
- `artifacts/orin/20261025-15/native-opset17-subcuts-20261025-15-001/terminal-audit-001.json`
- `artifacts/orin/20261025-19/decoder-assets-001/preparation.json`
- `artifacts/orin/20261025-19/decoder-preparation-worker-002/attempt.json`
- `artifacts/orin/20261025-15/projection-preparation-002/report.json`
- `artifacts/orin/20261025-15/projection-actual-audit-001.json`
- `artifacts/orin/20261025-15/native-projection-20261025-15-001/terminal-audit-001.json`
- `artifacts/orin/20261025-15/mixer-subcuts-preparation-001/report.json`
- `artifacts/orin/20261025-17/valid-cpu-state-native-step-audit-001.json`
- `artifacts/orin/20261025-17/cpu-state-probe-assets-001/preparation.json`
- `artifacts/orin/20261025-17/native-cpu-state-probe-20261025-17-001/terminal-audit-001.json`
- `artifacts/orin/20261025-15/first-mixer-preparation-004/report.json`
- `artifacts/orin/20261025-15/first-mixer-minimal-actual-audit-004.json`
- `artifacts/orin/20261025-15/native-first-mixer-20261025-15-004/terminal-audit-001.json`
- `artifacts/orin/20261025-15/query-scale-actual-audit-001.json`
- `artifacts/orin/20261025-18/prefix-complete-parity-001/report.json`
