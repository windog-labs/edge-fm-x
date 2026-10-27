# Orin Qwen Reusable Decoder

Latest: source readonly Exp miscompilation is isolated by actual intermediate
outputs and official-runner corroboration. Selective constant folding restores
all4real branch cases(maxabs5.96e-8), without changing runtime inputs/weights
or dense precision. Generic tool111tests/complete15step49output CPU parity
pass.16strict-selector failure is closed;17projects only folded literal layer
obligations, same repaired graph/weights/dynamic constraints. No full target
or primary/formal acceptance yet. [Repair evidence](orin_qwen_readonly_exp_repair_20261025.md).

Latest operator minimization isolates two9node real-source gating branches;
the first actual policy13case reproduces the large error without the rest of
the model. Complete49parent outputs and extracted CPU branch outputs stay
byteexact, original weights/nodes preserved. Initial wrong CUDA ordinal was
a coordinator bug now fixed/tested, not unavailable hardware. Corrected15suite
is verified live3/4cases after reusing two engines; no service control or
formal metric. [Focused diagnosis](orin_qwen_gate_operator_diagnosis_20261025.md).

## Latest Terminal Checkpoint

2026-10-25 08:31:04 UTC: controlled14engine4030995216B/SHA106be7bb was
actually built (wrapper322.638877792seconds), then its first native inference
wrote all49raw outputs. The numerical post-check correctly rejected it:
32outputs contain3219968NaNs, including all248320logits. First affected
state is layer8recurrent,16384NaNs. No15step rerun or formal sampling is
started with this candidate. The outer error says `nonfinite raw input`
because the shared frame validator is also used on outputs; actual inputs
were validated before inference, and the failure is in raw OUTPUT checking.
No NaN is replaced, filtered or omitted from whole-tensor error metrics.

The controlled change does improve the earlier layer0recurrent comparison:
MSE0.00084035578/cos0.42178277 becomes MSE0.000002584508345/cos0.99871638.
This is partial numerical improvement, not a complete fix or primary quality
pass. Nonfinite candidate comparisons are recorded as null metrics with
explicit counts, never finite-only aggregate substitutes. The first strict
comparison worker failed closed; its source/output remain. A separate complete
failure audit succeeds only as an audit, with candidate `nonfinite-rejected`.

All14build/diagnostic units are terminal and empty. Data-collection was the
sole build lease, restored by original Supervisor after344.026676224seconds,
PID3647862; starter/router/compotask and unrelated programs/configuration
unchanged. The no-service diagnostic also verifies manager/program states
unchanged. No live experiment remains from these phases. Goal stays active
and incomplete, not paused/blocked/complete; five accepted VLAs are unchanged.

Next: isolate the first divergent recurrent boundary and the first nonfinite
boundary using real inputs and generic SSA/operator tools, rather than another
unchanged whole-model build or unbounded precision search. CPU proxies are
not evidence of target internal precision. Source comparisons remain separate
supplemental evidence and do NOT redefine the user's fixed same-Orin/same-
precision ordinary/optimized gate. Finite13 is only a functional pilot;14 is
numerically rejected. Fresh raw-input chain, resident C++ Session, complete
16-observation paired quality and formal TTFT/16token/CDF/memory remain open.
Qwen2B remains queued with prior host assets, no new target claim.

## Controlled Contract Trial

Window14actually parses5695SDKlayers/zero unresolved policy names and is
building with the same graph/weights/BF16/dense policy; only54missing FLOAT
nonlinear constraints are added. Conditional original-manager recovery is
guarded. The source policy repair has98neighbor tests, but is not a measured
numerical fix yet. CPU counterfactuals at the first actual recurrent boundary
do NOT reproduce the native error: original cosine0.421782765881, nonlinear-
half proxy0.421795367767 and convolution-half proxy0.421782765881 against
the native state. Thus neither proxy confirms the cause; no blind escalation
to wholeFP32, no threshold change. Only a direct new-engine comparison can
establish the effect of this controlled policy change. A repeated ordinary
first step using the old same engine produced all49outputs byteexact, which
is repeatability evidence, not ordinary-versus-optimized primary acceptance.

## Complete Recurrence And Source Boundary

All15actual decoder calls completed,735full outputs/112657920elements
independently rehashed and finite. Every carry references the preceding actual
output; token/counter edges and first-max selection were independently checked.
The diagnostic is terminal/empty, original manager/program states unchanged.
This still starts from archived native prefix, not a fresh raw-input E2E.

The native16tokens differ from both the frozen CPU source and a portable
decoder run from the same actual native prefix. Teacher-forcing each exact
native step input also exposes a local decoder difference: first logits
MSE1.8403668537838396/maxabs6.71875/cos0.9481594362924571; first layer recurrent
state cosine0.42178276588137487, while convolution cache is near byteexact.
Thus there is both prefix/trajectory difference and an additional local
decoder difference; finiteness is not numerical equivalence or primary quality.

A concrete generic precision-contract omission is found:36Exp+18Softplus
FLOAT sites had no explicit arithmetic/output constraint under globalFP16.
Two new tests fail before repair;98neighbor tests pass after adding these
operators to the existing FLOAT-opmath policy. The old implementation/profile,
engine and all negative outputs remain. The leading hypothesis is unrequested
nonlinear precision; other possibilities include declared FP16 convolution
error. This is not yet a proven root cause or a numerical fix.

Window14 stages a controlled build with the exact same4215node graph/weights,
187nativeBF16 providers,18FP16dense and13sourceFLOAT dense sites. Only the54
missing FLOAT nonlinear precision/output requests are added; no existing
request is changed/deleted.13-versus14is precision diagnosis, NOT a same-
precision ordinary/optimized speedup comparison. Primary same-Orin quality
remains open and all thresholds/generation counts are unchanged.

## Actual Step Outputs

The first reusable-engine call has completed on Orin. Independent raw-output
audit rehashes all49outputs,7510528elements, all finite; logits are
FLOAT[1,1,248320]and argmax11952. Actual typed engine metadata matches the
declared source contract through explicit checked integer storage codecs.
The unit is terminal/empty and manager/managed-program states are unchanged.
This is one actual step from archived real prefix, not whole-model quality.

A model-neutral declared carry diagnostic is now running all15engine calls,
with explicit input/output state edges, tensor step counter and first-maximum
argmax semantics.15 control tests pass. Each call will save all49outputs;
no process/file-I/O duration is reported as resident inference or E2E timing.
Visual/prefill is still archived in this diagnostic; fresh chain/resident
integration, all observations, same-Orin primary quality and formal remain open.

## Engine Emitted, Not Yet Inference-Validated

2026-10-25 07:50:10 UTC independent audit: window13actually emitted a
4031019296byte engine,SHA256
`cc4594e90c0d760f3572aedc247cf396fecdd1cb5e42475a22dccdd3fe94b348`.
Native build306.837512576seconds; wrapper314.981898688seconds, outer compute
318.422608672seconds, minimum12833856KiB/no guard. Both cgroups empty, all
source/profile/weights/tool/authority pins rechecked; no service lease was
needed, all original program births/configuration unchanged. This is not
inference/quality acceptance or a performance metric. SDK warnings for12
infinity,21subnormal and8below-FP16-subnormal weight conversions remain.
All49raw outputs must be examined before claiming any model result.

Latest actual target checkpoint,2026-10-25 07:46:48 UTC: corrected4215node
graph passes TensorRT parsing,5695imported layers/zero unresolved policy
names, native receipt building. Same187nativeBF16 MatMul providers plus
18source-low dense FP16 and13source-FLOAT accumulation sites. No whole-FP32
fallback, no new engine or inference yet. No service stop was needed in13.
60wide INT64 literals are independently proved to be positive Slice-end
sentinels only; all affected dimensions fit INT32 and both ends clip equally.
Runtime INT64 control ports still require separate range-checked codecs.
Explicit archived actual-prefix-to-decoder51input bridge is verified against
all50 raw native outputs and original semantic/SSA source states; only12KV
tensors receive15zero slots. This prepares a diagnostic, not fresh E2E.

## Source Preparation

The route still requires the original visual/prefill chain and15 decode calls
for16tokens. Single-step reuse does not shorten generation or change precision.
No Torch/package installation on Orin; five accepted VLA campaigns unchanged.
Strict477tensor reconstruction reproduces dynamic-cache CPU tokens/logits
byte-exact. Fixed342cache matches16tokens but not source logits. Reloaded
decoder matches all49 eager outputs at all15steps byte-exact. ONNX6633nodes
passes structure checking; rawBF16runtime parity explicitly skipped.
Generic4251node deployment has51inputs/49outputs. Both portable shadows
match all49outputs x15steps exactly;16tokens match fixed-cache CPU, but
logit MSE0.0009795830875004427/maxabs0.15625/cos0.9999520224328118 remains.
This is not source-lossless, target-plugin or same-Orin primary quality.

## Native Parser Boundary

Window12 fails actual TensorRT8.5 parsing at `/decode/Expand`: its integer
constant ScatterElements shape producer cannot compute a shape tensor.
No engine, no resource guard or numerical test. Compute8.010698112seconds,
minimum23111032KiB. The sole data-collection lease was restored by original
Supervisor after27.4718616seconds,PID3582918/start39561793. Other declared
program births, unrelated programs/configuration and asset/tool/authority pins
are unchanged. Both cgroups empty; independent07:35:49UTC audit retained.

## Generic Correction

The integer folder now accepts bounded constant ScatterElements only with
reduction=none, valid indices and unique normalized destinations per slice.
Dynamic indices, duplicate/negative-alias destinations, reducing operations
and floating data are excluded. Red tests fail before the fix;94 focused and
neighbor cases pass after it. Old implementation is archived at its old hash.
Actual12constant shape scatters disappear (4251to4215nodes);12dynamic KV
updates remain byteexact, as do51inputs/49outputs and floating precision
requests. Complete49outputs x15steps before/after portable chains match
byteexact;16tokens still match fixed-cache CPU, source errors retained.

Window13stages the corrected graph under the existing ongoing bench grant,
not a new human reply or expired-window reuse. Same24GiBadmission/4GiBfloor,
3600second task/7200second phase with900second restoration reserve. Only
the four declared recoverable programs may be conditionally stopped and
restored via the original manager. No new target engine/inference/quality
or formal timing is claimed yet.

## Evidence

- `artifacts/orin/20261025-11/reusable-cache-source-006/report.json`
- `artifacts/orin/20261025-11/reusable-decode-export-001/export-report.json`
- `artifacts/orin/20261025-11/reusable-carrier-parity-001/report.json`
- `artifacts/orin/20261025-12/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-12/integer-scatter-red-001.xml`
- `artifacts/orin/20261025-12/integer-scatter-neighbor-green-002.xml`
- `artifacts/orin/20261025-12/static-scatter-rewrite-001/report.json`
- `artifacts/orin/20261025-12/static-scatter-full-parity-001/report.json`
- `artifacts/orin/20261025-13/build-window-001/checkpoint-003.json`
- `artifacts/orin/20261025-13/wide-slice-bound-audit-001.json`
- `artifacts/orin/20261025-13/actual-prefix-decode-inputs-001/report.json`
- `artifacts/orin/20261025-13/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-13/reusable-decode-build-001/build.json`
- `artifacts/orin/20261025-13/single-step-all-output-audit-001.json`
- `artifacts/orin/20261025-13/native-reusable-step-20261025-13-001/terminal-audit-001.json`
- `artifacts/orin/20261025-13/recurrent-control-tests-001.xml`
- `artifacts/orin/20261025-13/recurrent-source-boundary-001/report.json`
- `artifacts/orin/20261025-13/native-recurrent-20261025-13-001/terminal-audit-001.json`
- `artifacts/orin/20261025-13/nonlinear-precision-red-001.xml`
- `artifacts/orin/20261025-13/nonlinear-precision-neighbor-green-001.xml`
- `artifacts/orin/20261025-14/build-window-001/checkpoint-002.json`
- `artifacts/orin/20261025-14/first-recurrent-counterfactual-001.json`
- `artifacts/orin/20261025-14/build-window-001/terminal-audit-001.json`
- `artifacts/orin/20261025-14/native-reusable-step-20261025-14-001/terminal-audit-001.json`
- `artifacts/orin/20261025-14/nonlinear-contract-trial-worker-001/attempt.json`
- `artifacts/orin/20261025-14/nonlinear-contract-trial-002/report.json`
