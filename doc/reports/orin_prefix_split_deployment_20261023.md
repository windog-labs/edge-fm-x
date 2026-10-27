# Orin Prefix Split Deployment

2026-10-23 local date. This candidate addresses the full-prefix optimizer's
sampled-memory abort without stopping business services. It does not reduce
views, tokens, checkpoint weights, ten-step solver work or complete outputs.
Formal model coverage remains 2/5; no complete pi0 action/E2E result is claimed.
Latest evidence checkpoint: 2026-10-22 22:45:49 UTC. All four prefix engines
are built/ABI-inspected; Session003 linked natively, but its first raw pilot
failed initialization before neural enqueue. Corrected Session004 is also
natively linked; raw pilot002 passed initialization but exited 86 when source
acceptance rejected the first request at native finishing. No complete action
was published; no baseline is frozen. A real synchronized 744-tensor diagnostic
now localizes the first observed nonfinite boundary to part01 image tokens,
not missing workspace/bindings. The first local MLP probe now has 17 finite
actual Orin outputs. The deeper single-view tail now has 697 finite output
statistics with successful CUDA stream synchronization, not original whole
part01 equivalence. Next is the original part01 engine with all 18 real inputs
outside Session. Neither D1 nor D2 is accepted for pi0.

## Actual Publication

Source v5 SHA256:
`4d0eede245e56282630801993cc824b7aa0b239e2f2ba7733dae26a5b892c847`.
The generic partition API accepts only explicitly reviewed BF16-round/1 and
LayerNorm/1 shape contracts. Shape-only Identity substitutes never become
deployment nodes. Strict inference clears old internal annotations. Literal
Tensor Constants move to same-SSA initializers without changing tensor bytes;
the remaining source computations are extracted, not approximated.

| Part | Referenced Weight Bytes | Inputs / Outputs | Actual Target State |
|---|---:|---|---|
| 000 | 2802293696 | 8 / 10 | built-not-validated; actual ABI inspected |
| 001 | 2762950656 | 18 / 51 | built-not-validated; actual ABI inspected |
| 002 | 2776727552 | 69 / 26 | built-not-validated; actual ABI inspected |
| 003 | 2915147776 | 95 / 37 | built-not-validated; actual ABI inspected |

The total **11257119680 bytes** equals the original referenced byte union;
there is no duplicated initializer span. The streaming compactor verifies
source/copy/destination ranges before and after copying, with unchanged
non-address tensor metadata. It decodes no neural tensor and uses no hardlink,
symlink or source-payload write. Every directory is self-contained.

[Actual publication](../../artifacts/orin/20261023-15/published-002/publication.json)
references successfully published parts 000/001 under `published-001`, and
002/003 under `published-002`. The first aggregate publication stopped because
later cut lists contained already-inherited inputs; those logs remain. The
second run excludes only known inherited inputs from produced-value selectors,
and reuses earlier parts only after recomputing source metadata and full graph/
sidecar hashes. Actual partition ledgers, not previews, govern composition.

The [independent review](../../artifacts/orin/20261023-13/actual-publication-review001/review.json)
finds all **9589 used source NodeProtos exactly once**, including 810 dense
operators and all arithmetic/plugins. Only 35 exact ledger-declared passthrough
Identities are added. All 190 routes resolve; all 314 ports are static
FLOAT/BOOL/INT32 with zero I64 stage bindings. This proves structural/range
preservation, not neural numerical equivalence. The 77 focused software tests
and the review's ten tests are not model experiments.

## Target And Composition

All four actual Orin build receipts are `built-not-validated`, exit 0, using
`--buildOnly`. The engine identities below agree across build and actual
deserialization inspection receipts; this documentation audit rehashed the
local build receipts/logs, not the engine payloads retained on Orin.

| Part / Build Receipt | Engine Bytes | Engine SHA256 | Inspection / Required Context Bytes |
|---|---:|---|---|
| [000](../../artifacts/orin/20261023-15/stage000-build-001/build.json) | 2461615407 | `86ffeda4d78ffe6ad355d7195d8dc536f37ac7f235b51d4ec4694942c548a800` | [ABI](../../artifacts/orin/20261023-15/stage000-inspection-001.json) / 43588096 |
| [001](../../artifacts/orin/20261023-15/stage001-build-001/build.json) | 1389525332 | `d9ed520163f9f3da387a754e9f0b619160c15c5d9c91291070a781465edc0c58` | [ABI](../../artifacts/orin/20261023-15/stage001-inspection-001.json) / 374020608 |
| [002](../../artifacts/orin/20261023-15/stage002-build-001/build.json) | 1389692573 | `6e75f62c371df54b0fbd1777bb4a21fb89dcb293dc20986171166e2ea611cf21` | [ABI](../../artifacts/orin/20261023-15/stage002-inspection-001.json) / 226452992 |
| [003](../../artifacts/orin/20261023-15/stage003-build-001/build.json) | 1458965093 | `fa7757733df0fb006407093539463d7f70dfc459e1776d35e0c66a9e4eecf439` | [ABI](../../artifacts/orin/20261023-15/stage003-inspection-001.json) / 226452480 |

Inspections identify serial `1421524312594`, sm_87 and TensorRT 8.5.2.2.
All 314 ports are device LINEAR with the declared static shapes/dtypes. Each
inspection explicitly has `execution_context_created=false` and
`inference_executed=false`. Required context bytes are engine metadata, not
allocations, resident-chain capacity, process peaks or observed precision of
hidden arithmetic. Part000's 232.665867456-second build time is not inference
latency. Its 937 memory samples/minimum 12367120 KiB are sampled safety
observations, not continuous isolation or a matched memory-saving ratio.

The [derived ten-region Plan](../../artifacts/orin/20261023-14/actual-ledger-replay-002/composed/physical-plan.json)
uses the existing `split_region_invocation` implementation and actual ledgers.
Public schema remains `8751fa709441f09883da726d2cc2d6b52f7a5894bf5876ee36b6803e2dff9575`;
native physical IDs 0/1/3, original F64 state, fresh inputs, ten iterations,
normalized/native output contracts and acceptance/atomic commit remain.
The six existing contracts stay real. The
[attachment request](../../artifacts/orin/20261023-15/attached-all-stages-request-001.json)
adds the four actual stage engine/inspection identities; the
[attachment result](../../artifacts/orin/20261023-15/attach-all-stages-001.log)
reports `bindings-complete`, `actual_bound_regions=10`, `missing=[]`,
`target_acceptance=false`. Its four binding entries augment the original six;
they are not ten newly added entries. No synthetic prefix provider is used.

## Source Generation And Native Link

The old [generation001](../../artifacts/orin/20261023-15/generate-ordinary-session-001.log)
missing-request error and [generation002](../../artifacts/orin/20261023-15/generate-ordinary-session-002.log)
alignment-guard failure remain retained. The
[actual diagnostic](../../artifacts/orin/20261023-15/alignment-generation-diagnostic-001.log)
finds **334 old failures, all `old_count=0`**, at four/six-space nested
indentation; this was **not duplicate code**. Indent-aware task/direction/index
matching has zero new failures. The new wrapper does not modify old sealed
root08/root14 scripts. Its [12 software tests](../../artifacts/orin/20261023-15/guarded-generation-v2-tests-002.xml)
pass; these are not model runs.

[Session003 assessment](../../artifacts/orin/20261023-15/ordinary-session-003/binding-assessment.json)
is `candidate-session-source-generated-not-built`: all ten actual regions,
334 guarded descriptors each matching exactly once, unchanged public schema,
`native_outside_capture=true`, `model_inference=false`,
`full_model_quality_verified=false`, `target_acceptance=false`.
Assessment SHA256 is
`96b564d85c7d0c3d1fa076a50bce20f0a361175ec2819b35a659ef01b95875b7`.

The later [native build receipt](../../artifacts/orin/20261023-15/ordinary-session-003/native-shared-build-001/build-receipt.json)
and [controller log](../../artifacts/orin/20261023-15/native-session-build-001-controller.log)
record `native-shared-session-built-not-loaded`, library **234344 bytes**,
SHA256 `8bde931890361c1633608785eb510f9b45614817ca4cc09fba4620dc30de2579`.
All pinned Session003 source/metadata files match the local bundle. The build
receipt has GPU execution, Session loading/factory, model inference and target
acceptance all false; it proves native linkage only. A subsequent load attempt
failed as described below; the build receipt's scope is not extended.

## First Raw Pilot Failure And Correction

[Pilot001 controller](../../artifacts/orin/20261023-15/ordinary-raw-pilot-001-controller.log)
and [terminal guard](../../artifacts/orin/20261023-15/ordinary-raw-pilot-001-guard/guard.json)
agree on child PID/start **3633695 / 18905371**, exit **86**. The requested
16 warmup + 16 measured requests did not become completed model samples.
[Actual stderr](../../artifacts/orin/20261023-15/ordinary-raw-pilot-001-guard/stderr.log)
reports `shared plugin Region workspace contract mismatch`, with
`failed-completion-or-lifetime-unproven` and `retained_to_process_exit=true`;
stdout is empty. All three guard log hashes were rechecked. This is a Session
initialization/workspace integration failure **before neural enqueue**, not a
compilation failure, source-operator unsupported result or numerical gate
failure. It yields no valid action/E2E result.

The parent diagnosed ordinary-invoke codegen passing workspace zero despite
existing physical Plan `workspace_buffer` declarations: native cast regions
0/1 require **384/448 bytes**, **256-byte alignment**, CUDA. The correction binds
each task's already physicalized workspace, rather than allocating an extra
StaticArena; `LoadRegion` strictly checks queried size/alignment/device. Zero
workspace behavior is preserved. Requirements/checks/precision are not relaxed.

[Session004 source assessment](../../artifacts/orin/20261023-15/ordinary-session-004/binding-assessment.json)
again has ten bindings, `missing=[]`, `candidate-session-source-generated-not-built`
and no target/model/quality acceptance. Its semantic module, physical Plan and
artifact-contracts files are independently byte-identical to Session003; only
correct workspace binding is generated. The
[codegen/Plan regression](../../artifacts/orin/20261023-15/workspace-codegen-regression-001.xml)
has **28 passed**; [replay regression](../../artifacts/orin/20261023-15/workspace-replay-regression-001.xml)
has **19 passed, 2 skipped** (opt-in real CUDA). These software checks do not
prove the corrected target runtime. The explicit
[source-invariance receipt](../../artifacts/orin/20261023-15/workspace-source-invariance-001.json)
records unchanged Module/Plan/ten artifact contracts, no engine rebuild and no
model/precision change; independent byte comparison confirms the three files.

The [Session004 native link receipt](../../artifacts/orin/20261023-15/ordinary-session-004/native-shared-build-001/build-receipt.json)
is `native-shared-session-built-not-loaded`, **234344 bytes**, SHA256
`e7765d450813b5639b774604a81d0abd299035b3dc4d03dd250f87a6e393ef97`.
The receipt's GPU/model/target flags remain false. The later
[pilot002 launch log](../../artifacts/orin/20261023-15/ordinary-raw-pilot-002-controller.log)
records child **3641035 / start18940517**, now `child-exited`, **exit86**.
All native/TRT initialization passed, then actual run reached
native finishing and returned **`source acceptance rejected both output ports`**
on the first request. No complete normalized/native action is published; this
is not 32 completed requests or a usable frozen baseline. It is distinct from
pilot001's earlier initialization/workspace failure. No numerical-gate pass or
accepted complete-action/E2E result follows from reaching native finishing.

The [full pilot002 guard](../../artifacts/orin/20261023-15/ordinary-raw-pilot-002-guard/guard.json)
and [actual stderr](../../artifacts/orin/20261023-15/ordinary-raw-pilot-002-guard/stderr.log)
are now local and verified, including all three log hashes. Stderr confirms the
source-acceptance rejection and `retained_to_process_exit=true`. The subsequent
**untimed diagnostic** copies actual task I/O at synchronized boundaries; its
[guard](../../artifacts/orin/20261023-15/diagnostic-raw-002-guard/guard.json)
also exits86 under the unchanged acceptance check. Diagnostic synchronization/
copies are not formal latency samples. No completed action is fabricated from
partial stage outputs, and all old failures remain.

## Numerical Diagnostic

The [744-tensor audit](../../artifacts/orin/20261023-15/diagnostic-tensor-audit-002.json),
SHA256 `727c26374157c0b6f58bad55adfd1101c348a1824917c605ce265f63d1813616`,
has `diagnostic_only=true`, `formal_performance_eligible=false` and
`model_acceptance=false`. This documentation audit independently rechecked
**all 744 dump file sizes/SHA256s, 767531682 bytes**. Part00's ten recorded
outputs are all finite. The first observed nonfinite tensor is **task14,
call0, part01 output0**, named **`transpose_330`** in the actual engine ABI:
F32 `[1,1,816,256]`, 835584 bytes, buffer35. It contains **196608 NaNs** in
the **768 image tokens**; the remaining **48 text tokens / 12288 values** are
all finite. This split was independently checked directly from the raw dump,
whose SHA256 is `ab2269c87515f833b559f538e7f561ed2940decaf16cd0bdb4f68a30a3efa521`.
The audit records 503 nonfinite tensor snapshots downstream; these are not
503 independent model observations or accepted outputs.

`region_ssa_snapshot_mismatches=[]`: every recorded same-buffer SSA
producer/consumer boundary has matching SHA, **zero byte mismatches**. This
supports the observed boundary copies/routing only. It does not prove all
routing/lifetime cases correct or identify the first failing arithmetic node
inside part01. The current remaining gate is **numerical NaN localization**,
not missing workspace, bindings or Session initialization. Source acceptance
continues to reject complete publication; no baseline can be frozen.

The parent's [minimal vision MLP preparation](../../artifacts/orin/20261023-15/vision-mlp-probe-001/preparation.json)
preserves a selected diagnostic computation: **28 nodes, 17 outputs,
39687488 referenced weight bytes**. Its
[actual Orin build002](../../artifacts/orin/20261023-15/vision-mlp-probe-build-002/build.json)
is `built-not-validated`, exit0; engine **19991941 bytes**, SHA256
`42f30b32eab03899f0f3d63fd096906bea565f01678993f2256b47f68ed109ce`.
That immutable build receipt proves build feasibility only. A later
[actual Orin inference summary](../../artifacts/orin/20261023-15/vision-mlp-probe-infer-001-summary.json),
SHA256 `3fe3a3bd9ce004850060cd15abcae8a1b16bd49f5ce153db409ade6d8b79f381`,
records **all 17 observed outputs finite**. Its
[target guard](../../artifacts/orin/20261023-15/vision-mlp-probe-infer-001-guard/guard.json)
has child **3663165 / start19047940**, exit0, serial `1421524312594`,
`formal_sampling=false`; all three log pins were rechecked. The actual
`trtexec --loadEngine` request uses `add_23` and `layer_norm_23` from the same
synchronized diagnostic dump inputs, one iteration, no warmup. This is real
selected-subgraph execution, not a source-only or CPU result; its timing is
not promoted to E2E.

The selected source nodes and declared per-node precision are retained, but
the probe adds observation outputs. Those outputs can affect TensorRT
fusion/tactic selection. **Local finite outputs do not exclude a failure in
the original engine's corresponding region**, establish a numerical root
cause, or validate the complete prefix/model. The original NaN/acceptance
failure remains unchanged and no baseline is frozen.

The [root18 CPU shadow result](../../artifacts/orin/20261023-18/cpu-reference-002/result.json)
for the selected part01 graph has **695 finite probe statistics and 51 finite
public outputs**, using `CPUExecutionProvider`, `half_emulation=false`.
The separate [FP16 IO cast simulation](../../artifacts/orin/20261023-18/half-cast-001/result.json),
SHA256 `a1616b851dcdba24382980b734ac4b4f333067bf23413bc340b27b04e60f1a78`,
also has **695 finite probe statistics and 51 finite public outputs**, with
`half_emulation=true`. Both results explicitly retain
`target_accumulator_precision=unobserved` and `target_acceptance=false`.
These are CPU diagnostics, not measured TensorRT intermediate accumulation,
source-BF16/TRT equivalence, fallback deployment, target timing or complete
neural-action results. Finite CPU simulations do not substitute for actual
Orin outputs or prove/exclude FP16 overflow or a kernel cause.

The [full single-view tail probe preparation](../../artifacts/orin/20261023-15/vision-tail-probe-001/preparation.json)
selects `linear_162` with the same `add_23`/`layer_norm_23` inputs and
**697 observation outputs**, preserving the declared per-node precision. It
targets later-layer numerical localization, not a reduced-model acceptance
substitute. The [build001 controller](../../artifacts/orin/20261023-15/vision-tail-probe-build-001-controller.log)
records child **3665131 / start19057246**, exit0. That earlier controller-only
checkpoint did not establish a successful numerical probe. The subsequent
[actual native GPU result](../../artifacts/orin/20261023-15/vision-tail-native-infer-001/results.json)
is now local, SHA256
`429ad001e1c235375e1fed293b97a6dc5247cbc51b799a89820299d4ba420ad7`.
It records **697 outputs / 1624113332 bytes**, each with
`finite == elements`, `nan=0`, `inf=0`, and explicit completion
**`cudaStreamSynchronize-success`**. The full per-output statistics and their
byte totals were independently checked locally. The result explicitly has
`diagnostic_only=true`, `formal_model_coverage=false`, `full_vla_pass=false`
and `timing_reported=false`.

The parent retains all raw output bins **on Orin only**, without fetching them
because of host space constraints. Local raw-bin count is zero. This audit
therefore verifies the complete local statistics artifact, **not local raw-bin
hashes, elementwise equality or cross-engine parity**. A single-view tail
with additional observation outputs can affect fusion/tactics and is not the
original whole part01 engine. Its finite result neither erases pilot002 nor
proves original part01 correctness/equivalence, a full-model pass or a frozen
baseline.

The parent is preparing a separate run of the **original part01 engine with
the same 18 actual recorded inputs**, to distinguish a Session-dependent
issue from original-engine fusion/numerics. No independent whole-part result
is claimed yet. SDK, declared precision and acceptance are not relaxed; this
documentation task launches no target experiment.

The separate [root17 software ABI receipt](../../artifacts/orin/20261023-17/result-001.json)
and [15-passing-test XML](../../artifacts/orin/20261023-17/cpu-tests-002.xml)
cover real compiled **NONMODEL CPU shared-plugin** workspace behavior. They
do not add model passes. The earlier diagnostic header compilation failure
is retained in [its original log](../../artifacts/orin/20261023-15/diagnostic-native-build-001.log);
it is historical diagnostic preparation, not the current model blocker.

The [preflight](../../artifacts/orin/20261023-15/full-session-target-preflight-001.json)
has the same serial and both Java processes **3291273/start17225825** and
**3291324/start17225946**, still present. No business-service stop or extension
of the RDT-only authorization is inferred. This snapshot does not prove
continuous device exclusivity, application health or exhaustive resource use.

## Next Gate

Obtain/audit the parent's original part01 engine run with all 18 identical
real inputs, then distinguish Session-dependent behavior from the original
engine's fusion/numerical behavior using the existing probe/reference evidence.
Do not infer original whole-part correctness from the single-view 697-output
finite statistics. Preserve SDK, precision/source contracts and acceptance;
neither local probe is a complete-prefix gate. Workspace/binding is no
longer the remaining gate. After the numerical issue is resolved, rerun all sixteen frozen raw
observations through complete ten-step actions and qualify baseline/optimized
paths, resident capacity and lifetime. No accepted complete pi0 action,
D1/D2 acceptance or formal sampling is established here. The complete normalized F32 `[1,50,32]` gate is
fixed **MSE <1e-5 and cosine >0.9999**, all outputs finite; native F64
`[1,50,14]` errors and original H20 differences are supplemental. Byte-exact
regression is separate, not a replacement or relaxed gate. Only after complete
actions pass may the formal protocol advance. This documentation task executes
no SSH, GPU or model work and changes no code or sealed artifacts.
