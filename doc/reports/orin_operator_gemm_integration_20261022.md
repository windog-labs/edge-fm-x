# Orin GEMM Whole-Model Integration

Date: 2026-10-22, Asia/Shanghai. Goal and D4 remain incomplete.

## Fixed-Gate Failure

The one measured MatMul-to-Gemm recipe was integrated into the complete adopted
SmolVLA prefix. Both variants execute seven native TensorRT engines and all ten
solver iterations, with identical input/noise records, semantic IR, Plan and
other six engine hashes. They each complete one warmup plus sixteen measured
tensor-input calls. All 34 normalized/native action pairs are independently
audited; no horizon or dimensions are dropped.

| Complete Output [1,50,6] | Minimum Cosine | Maximum MSE | Maximum Absolute Error | Gate |
|---|---:|---:|---:|---|
| Normalized action | 0.999997063102 | 1.32434496e-5 | 0.0166993141 | Fail MSE |
| Native action | 0.999998502809 | 0.0124503239 | 0.440723419 | Fail MSE |

Worst MSE is observation 10 in both spaces. These are maxima/minima across all
seventeen calls, including warmup; all per-dimension metrics and complete raw
outputs remain available. Baseline is byte-exact to the original adopted pilot
for both outputs. Candidate is not. Cosine passing does not override the fixed
MSE < 1e-5 requirement, even when normalized MSE is only slightly above it.

**No formal raw-observation E2E sampling or integrated speedup is authorized by
this result.** The standalone resident 1.448x / IO event 1.008x measurements stay
valid only within their operator boundary, not as accepted model performance.
The failure also does not depend on the pending normalized/native interpretation:
normalized action itself fails here. No tolerance, precision, iteration count,
noise or model semantics are changed to obtain a pass.

## Source and Build Checks

The typed source node, fragment boundaries and immutable constant values are
matched to the full frozen source. Global value/node name conflicts are avoided;
all original 840 initializers, including shared constants, are retained verbatim.
One MatMul is replaced by six qualified nodes; source node count goes from 2088
to 2093. Six initializers for transposed weight/shape bookkeeping are added.
Complete source IO and all compute/output precision-map entries are unchanged.
Independent verification re-reads both complete graphs rather than trusting a
boolean emitted by the splicer. The public runtime gains no model-name branch.

Source SHA256 is
`567ef82105c2ebbb7f7afde0faa7c3a960cf80c0711e31bc5a4c10dbef58a147`;
candidate source is
`264e85dc20365864285570262f0d91b39a3fbf86f224be04666cd2898d46b869`.
Candidate engine is
`2a88ae417e13ff59fb52eba7ded9c5e269e105c8e8a9aebde28627eda427d750`,
657625444 bytes. Matching FP16/noTF32/2048 MiB workspace and full precision-map
CLI settings are checked. The controller source revisions differ from the
historical build and remain recorded. Their identical numerical CLI settings
do not establish identical hidden vendor tactics or internal accumulation.

All 43 binding names/order and exposed descriptors match. The same three absent
inspector descriptors are explicitly retained; complete native runs pass the
frozen backend's type/layout/location/shape guards. The target is serial
1421524312594, boot 3c801f5d-6897-4d97-b021-3c52db38a750, TRT 8.5.2.2/CUDA 11.4,
with original MAXN and services unchanged. No Torch model compute runs on Orin.

Build-controller walltime is 161.949165 seconds, not inference time. Its max
child RSS is 7193776 KiB. The guard admits 15859892 KiB available system memory,
requires at least 12 GiB before building and samples 328 resource points; minimum
availability is 8981964 KiB, above the 2 GiB owned-job safety line. This is not
isolated GPU memory or a summed CPU/GPU footprint. Current-user-visible ownership
and inaccessible-process limitations remain as in the other target reports.

## Rebuild Control

The additional control rebuilt the **unchanged original full ONNX and unchanged
precision-profile bytes** with the current native build controller. It is not a
second graph candidate, quantization attempt or relaxed gate. Source SHA and
profile SHA exactly match the historical deployment. Source/IO/CLI checks pass.
Its engine hash is
`b0b77654c8b4312e6fe355709a12afa54005c83e3e2ddec1935b2fcbffd8c42e`.

The control completes the same seven-engine/ten-step protocol and another 34
action pairs. Historical baseline remains byte-exact in this second campaign.
The matched comparison reuses these audited captures, not new observations.

| Comparison | Normalized Max MSE | Native Max MSE | Normalized Gate |
|---|---:|---:|---|
| Original adopted baseline vs GEMM trial | 1.32434496e-5 | 0.0124503239 | Fail |
| Original adopted baseline vs same-source rebuild | 4.51921502e-7 | 0.00104099844 | Pass |
| Same-source rebuild vs GEMM trial | 1.37592294e-5 | 0.0127836698 | Fail |

Rebuilding unchanged source can change output bytes in this deployment; source
identity alone is not bit-exact reproducibility. The matched fresh-build
comparison still fails normalized MSE. The recipe is therefore **not accepted
for formal integrated E2E**, including when tested against this new control.
Neither the original failed receipt nor its baseline is retrospectively replaced.
This bounds the attribution but does not identify one hidden vendor kernel or
prove internal accumulation parity. The pending native-unit interpretation
cannot remove this recipe's normalized failure.

Control build walltime is 158.989255 seconds, max child RSS 7161692 KiB, minimum
observed system availability 8353700 KiB; no 2 GiB safety stop occurs. Only the
one graph candidate plus one unchanged-source build control were evaluated.
No quantization, solver, noise, precision-map or tolerance changes followed.

## Evidence

Root: `artifacts/orin/20261022-00/`.

| Evidence | Path |
|---|---|
| Full-source composition and candidate graph/profile | `gemm-integrated-prefix-001/` |
| Native stage build, engine and resource telemetry | `gemm-prefix-build-retrieval-001/` |
| Independent source/precision/ABI audit | `gemm-integrated-prefix-001/source-pair-audit-001.json` |
| Identical IR/Plan and exact changed-artifact declarations | `gemm-full-model-pair-001/pair.json`, baseline/candidate preparations |
| Complete target captures, phases, PID/start and output hashes | `gemm-full-model-pair-001/pilot-001/` |
| Complete action failure and per-dimension metrics | `gemm-full-model-pair-001/quality-001.json` |
| Initial splicing/composition/admission/full-action regressions | `gemm-full-integration-regression-001.xml`, 38 passed |
| Unchanged-source target control and proof | `gemm-rebuild-control-retrieval-001/`, `gemm-integrated-prefix-001/rebuild-source-audit-001.json` |
| Complete control actions and original-baseline comparison | `gemm-rebuild-full-model-pair-001/quality-001.json` and `pilot-001/` |
| Matched rebuilt-control vs trial comparison | `gemm-full-model-pair-001/matched-rebuild-control-001.json` |
| Final full-integration/control regressions | `gemm-full-integration-control-regression-002.xml`, 43 passed |

Source-composition, native C++ Session and complete-action auditing helpers are
model-neutral and artifact-driven. Exact target argv is retained in capture and
campaign JSON. Reruns use fresh output/build directories; original artifacts and
failed results are not overwritten. Existing SmolVLA formal same-engine results
remain unchanged. RDT is not restarted, and the other required VLA models remain
in the Goal.

Next required work is other target operator families or another independently
qualified effective recipe, plus the open LayerNorm acceptance-space question.
This negative integration experiment is complete, but the required successful
VLA integration/E2E gate remains open; it is not replaced by this report.
