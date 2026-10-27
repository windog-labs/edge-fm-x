# Qwen3.5 2B Orin Preparation

This is host-only preparation for the Orin supplement, not an Orin deployment
or performance result. It does not reopen the accepted H20 campaigns.
Current checkpoint: 2026-10-24 23:05 UTC. Both native representations and fixed
precision declarations are prepared. On one actual observation, the complete
host-reference prefix/tail chain runs and matches all sixteen source tokens;
source logits differ, so source losslessness is not certified. All fifty crop
rewrite outputs are byte-exact in the same reference representation. No host
worker remains active. Orin is below admission for a third consecutive Goal
turn without a fresh service grant; Goal is `blocked`, not complete.

[Resource/authorization block audit](orin_qwen_resource_block_20261025.md).

Prefix002 has 190400 nodes, 624 initializers, fifty outputs, thirty observed
and thirty safe-softmax sites; bounded export finishes in 196.064 seconds.
Metadata SHA256 is `973973723d061c2e25c65fc9a4ddb85ccf62c8b803871ed250214a7b729ebd19`.
Its 429 external-file content hashes and sizes match the earlier prefix as a
multiset, although graph-generated filenames differ. This is a payload check,
not a proof of unchanged operator bindings or outputs. Source archive/inputs
and the qualified exporter remain unchanged. Integer folding gives 32776
prefix nodes and 47261 tail nodes, with retained computations/public I/O/
floating literals preserved and no floating arithmetic evaluated.

The tail BF16 carrier is explicitly a storage representation: 2805 native
BF16 MatMul providers and 11562 BF16 rounding boundaries remain, including
48 input boundaries. It is not an all-FP32 model fallback or precision
certification. Its data payload is 7527288832 bytes, SHA256
`d92e899b5afda9af58b85c6804d739e7a5a1e5027e0911222e62fd70cfdc0a7d`.
Source equivalence and complete model quality are still unverified.
[New prefix export](../../artifacts/orin/20261025-01/whole-prefix-export-002/export-report.json),
[new prefix integer controls](../../artifacts/orin/20261025-01/prefix-external-002/model-integer-shapes.report.json),
[tail integer controls](../../artifacts/orin/20261025-01/whole-tail-export-large-host-001/model-integer-shapes.report.json),
[tail BF16 preparation](../../artifacts/orin/20261025-01/tail-native-001/model.report.json).

## Source Recovery

The old checkpoint directory `/xs-train-nas/zzm/models/Qwen3.5-2B` is not
currently visible through `zzm-4090-x8-1`. The previously accepted full
16-token TorchScript source remains available under
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/`.
No checkpoint download, weight substitution or new GPU experiment is needed
for this initial source assessment.

Recovered into `artifacts/orin/20261025-01/source-full-2b/`:

- `qwen_generate_exact.pt`: 4431187611 bytes, SHA256
  `818881968f0068abcda0d3cfc0e01e9d4fd04765da52560dfa548876a418a744`.
- The matching original native bundle and trace-validation report.
- All eight input/reference files for actual image sample zero, with complete
  byte-count and SHA verification against the original validation report.

The source validation report retains sixteen historical validation cases.
The initial preparation retrieved one case. A subsequent hash-only inventory
recovers all sixteen observations and verifies all 128 input/reference files
against the unchanged historical report. It does not rerun historical model
inference or reclassify H20 validation as Orin validation. The existing generic
`prepare_source_inputs.py` verifies source/archive/bundle agreement and creates
typed CPU inputs with GPUs hidden and two CPU threads. The bounded worker
finishes successfully in 4.002 seconds, with no memory-guard event.

[Source and input receipt](../../artifacts/orin/20261025-01/prepared-full-2b-001/preparation.json)
and [bounded preparation worker](../../artifacts/orin/20261025-01/input-preparation-worker-001/attempt.json).

[Complete sixteen-observation inventory](../../artifacts/orin/20261025-01/complete-asset-inventory-001.json)
and [terminal inventory worker](../../artifacts/orin/20261025-01/complete-asset-inventory-worker-001/attempt.json).
Seven focused checks cover a valid inventory and rejection of duplicate case
indices, unsafe paths, missing/corrupt payloads and changed archive/metadata.
None of the six negative cases publishes a success receipt.
[Inventory test results](../../artifacts/orin/20261025-01/trace-inventory-tests-001.xml).

## State Boundary

Read-only inspection of the actual saved graph finds 112580 nodes and sixteen
ArgMax sites. Cutting after the first site, at index 37735, yields fifty
nonconstant runtime values, all tensors. Other live immutable attributes and
constants are not incorrectly counted as mutable cache tensors. This boundary
comes from SSA consumers, not an assumed correspondence with the 0.8B model.

[Graph-boundary inventory](../../artifacts/orin/20261025-01/full-jit-cut-001.json)
and [inspection worker](../../artifacts/orin/20261025-01/jit-boundary-worker-001/attempt.json).

The complete same-CPU split-composition proof finishes successfully in
126.031 seconds under the existing 24 GiB RSS / 3 GiB available-memory /
900-second host guard, with no guard event. All sixteen output tokens and
all 248320 final BF16 logits are byte-exact to the same CPU source invocation;
caller inputs remain unchanged. This establishes the source split, not ONNX,
TensorRT or a reusable single-step decode engine.

| Runtime tensor dtype | Shape | Count |
| --- | --- | ---: |
| BF16 | `[1,6144,4]` | 18 |
| BF16 | `[1,16,128,128]` | 18 |
| BF16 | `[1,2,327,256]` | 12 |
| I64 | `[1,1]` | 2 |

These fifty actual tensors occupy 14340112 bytes. The raw split has 37736
prefix nodes and 77822 tail nodes, including independently cloned immutable
attributes/constants; no neural formula is substituted. The exact shapes
come from source execution and are not inferred from the 0.8B configuration.

The retained H20 comparison is separate: sixteen tokens are byte-exact, while
final logits have MSE 0.02411069841003053, max-abs 0.90625 and cosine
0.997723672589007. The source-device difference is not discarded or treated
as a same-Orin normal/optimized pass. This host proof supplies no Orin timing.

[Complete split proof](../../artifacts/orin/20261025-01/full-jit-cut-proof-001/report.json)
and [terminal bounded worker](../../artifacts/orin/20261025-01/full-jit-cut-proof-worker-001/attempt.json).

## Serialized Interfaces

The unchanged generic registered-parameter serializer finishes in
198.062 seconds, exit zero, guard null. Peak sampled host RSS is
11030347776 bytes and minimum host available memory is 21072060416 bytes.
The source weights remain registered parameters rather than frozen graph
constants. Both whole stages are saved, independently reloaded and checked
against every complete output of the source proof:

| Stage | Inputs | Outputs | Complete reloaded outputs |
| --- | ---: | ---: | --- |
| Prefix | 3 | 50 | Byte-exact |
| All remaining fifteen decode steps | 50 | 2 | Byte-exact |

The prefix archive SHA256 is
`647458c12db64145f0ceae8cf6f9b0d00d98a743742c2902d127daa8d5f02ab5`;
tail SHA256 is
`0d2b92a516537cf7becb6aa84f1def89d6b5cffd6f4ffbc07192622d14ccc00e`.
This does not establish a reusable single-step decode API, arbitrary prompt
length, ONNX fidelity or TensorRT support. It is the verified source interface
for subsequent complete-stage export.

[Serialization and output receipt](../../artifacts/orin/20261025-01/full-state-serialized-001/report.json)
and [terminal serialization worker](../../artifacts/orin/20261025-01/state-serialization-worker-001/attempt.json).

## Export Preparation

Independent shape observation on the actual two serialized stages preserves
all complete CPU outputs and produces source/input-bound export profiles:

| Stage | SDPA sites | Q/K/V heads |
| --- | ---: | --- |
| Prefix vision | 24 | `16/16/16` |
| Prefix language | 6 | `8/2/2` |
| Fifteen-step tail | 90 | `8/2/2` |

The 2B vision profile is not copied from the 0.8B model. Prefix observation
finishes in 104.036 seconds; tail observation in 36.013 seconds. Both bounded
workers exit zero with no guard event.
[Prefix shapes](../../artifacts/orin/20261025-01/whole-prefix-prepared-001/attention-shapes.json),
[tail shapes](../../artifacts/orin/20261025-01/whole-tail-prepared-001/attention-shapes.json).

The unchanged public exporter SHA256
`79d3b6db81a225cbf98781213637232aa35a8271204ab4a5bff495aa635af267`
matches the earlier qualified frozen exporter. Each stage job uses its own
observed profile, source archive and typed inputs, opset 17, static GQA
lowering and the declared composite-opmath policy. Exporter constant folding
and runtime validation are explicitly disabled. Checker success alone will
not prove numerical equivalence; subsequent complete-output checks remain
required. Neither precision rules nor generation steps are removed.

Complete prefix export finishes in 195.064 seconds, exit zero, guard null,
peak host RSS 10744700928 bytes. ONNX checker passes: 190040 nodes,
624 initializers, fifty outputs. The actual exported inputs are token IDs
and image patches; the unused attention-mask input is eliminated by export,
not replaced with a synthetic tensor. There are 429 external files totaling
4429672704 bytes. Model metadata SHA256 is
`dbdb1321f29477a484f09c1d8c5e29c0b5e3d191941414fffefb3309715a80ee`.
The initial complete tail export was launched separately with its own ninety-site
profile. Forty-four existing GQA/composite-opmath regression tests pass; their
expected negative-case diagnostics are not failed deployment attempts.

[Prefix export report](../../artifacts/orin/20261025-01/whole-prefix-export-001/export-report.json),
[terminal prefix export worker](../../artifacts/orin/20261025-01/prefix-export-worker-001/attempt.json),
[exporter regression results](../../artifacts/orin/20261025-01/exporter-regression-001.xml).

The prefix is consolidated with the existing generic externalizer. All 71266
tensor payloads, including node attributes, preserve their exact metadata and
4430639608 payload bytes; before/after aggregate fingerprint is
`9db395a2e2c28308d2c9d43474c5d4d3a538bc686febe32b83f0277e23d238c6`.
All 190040 nodes remain. This is storage consolidation, not a dtype or
numerical rewrite. Its worker finishes in 28.011 seconds without a guard event.
[Prefix tensor-preservation receipt](../../artifacts/orin/20261025-01/prefix-external-001/report.json).

### Tail Export Memory Diagnosis

Attempt001 stops after 115.858 seconds, exit -15, `host-memory-guard`, before
publishing a complete tail ONNX. At the final sample, its RSS is 24789962752
bytes, below the unchanged 24 GiB RSS ceiling; whole-host available memory
falls to 2615320576 bytes, below the unchanged 3 GiB floor. Prefix weight
externalization overlaps this late phase and adds host memory pressure. Both
original workers are terminal. Do not call this a kernel OOM, unsupported
TensorRT operator or an Orin resource failure.

One exact full-tail retry002 removes only concurrent large host work. Source,
inputs, profile, exporter, all fifteen steps, two CPU threads and the
24/3 GiB/900-second limits remain unchanged. It also stops: 124.711 seconds,
exit -15, peak RSS 26047975424 bytes exceeds the 24 GiB ceiling while host
available memory remains 6073430016 bytes, above the 3 GiB floor. Thus overlap
is not the sole cause: the isolated export exceeds this host's fixed RSS budget.
There is no further unchanged local retry and no increase to its guard.
[Failed tail worker](../../artifacts/orin/20261025-01/tail-export-worker-001/attempt.json),
[isolated tail worker](../../artifacts/orin/20261025-01/tail-export-worker-002/attempt.json),
[externalization worker](../../artifacts/orin/20261025-01/prefix-externalization-worker-001/attempt.json).

### Large-Host Compatibility

Read-only inspection of `zzm-4090-x8-1` finds about 684 GiB physical memory,
660 GiB available, and 437 GiB free in `/dev/shm`. The root overlay is nearly
full, so it is not a model-output destination. The existing base Python 3.10
has Torch 2.7.1; read-only additional package lookup supplies ONNX 1.19.1 and
ORT 1.22.0 from the existing shared `e2e_zk` package payload. A task-private
pure-Python pluggy copy supplies pytest compatibility. No global environment,
SDK, Torch installation or GPU model execution is changed.

Original exporter regression on that host is 39 passed / 5 failed. Two tests
assume the newer Torch registration-module path; they now test the same policy
restoration/non-overwrite contract through either supported import path. Three
failures expose all-masked-row NaNs. An independent ONNX ReferenceEvaluator
regression reproduces four floating-mask failures even on the original local
environment. The generic fix delegates SDPA to the original Torch symbolic,
guarding only its softmax so all-negative-infinity rows produce zero weights.
It retains dtype/opmath, GQA ordering and ordinary-row behavior. No model-name
branch, changed tolerance or skipped failing test is introduced.

The updated exporter SHA256 is
`8a25286fb6ef4ebda6b3f7c46fa90b7e83c257a8c15ad79a256d8842b0008014`.
Its byte-identical local snapshot is
`artifacts/orin/20261025-01/exporter-safe-softmax-004.py`; the source copy used
by the remote worker is also retained under that run's `vlaforge/tools/`.
The earlier prefix remains tied to frozen exporter SHA79d3b6..., which still
exists unchanged as `20261024-09/exporter-gqa-opmath-003.py`; no prior graph or
result is overwritten. Local 52 focused tests pass, then 88 neighboring
export/external-data/BF16 tests pass. Remote retry002 has 51 passes and one
NAS temporary-directory cleanup failure after export/checker pass. Moving
test scratch off NAS yields all 52 passing in remote retry003. All negative
reports remain; none is relabeled as a successful run.

[Local red cases](../../artifacts/orin/20261025-01/safe-softmax-red-001.xml),
[local 88-case regression](../../artifacts/orin/20261025-01/safe-softmax-neighbor-regression-001.xml),
[original remote failures](../../artifacts/orin/20261025-01/remote-exporter-regression-001.xml),
[remote NAS cleanup failure](../../artifacts/orin/20261025-01/remote-exporter-regression-002.xml),
[remote 52-case pass](../../artifacts/orin/20261025-01/remote-exporter-regression-003.xml).

A separate large-host-only preparation profile requires the expected hostname,
at least 512 GiB physical memory, at least 128 GiB available at admission and
runtime, at most 64 GiB child RSS, 900 seconds and two CPU threads with GPUs
hidden. Five host/profile rejection tests pass. These limits cannot be used
on the 31 GiB local machine and do not alter its 24/3 GiB profile or Orin's
16/4 GiB model gates. Complete source/archive/input/profile hashes are checked
again before the large-host full-tail job. Scratch model output goes to a
task-owned `/dev/shm` path and must be archived before cleanup.
[Large-host profile tests](../../artifacts/orin/20261025-01/large-host-profile-tests-001.xml).

The source archive, typed inputs and observed-profile transfer hashes match
the original local files exactly: `0d2b92a5...`, `5b06df99...` and
`3abbc78d...`. The large-host job starts at approximately 21:33 UTC, still
keeps all fifteen continuation steps, uses the new qualified generic exporter
and explicitly skips ONNX runtime validation. The guard additionally requires
32 GiB free scratch at admission and a 16 GiB scratch runtime reserve. Its
updated hostname/physical-memory/available-memory/CPU-only checks again pass
five tests. An exported graph is not a numerical or target-performance pass.

The historical Orin read at 21:34 UTC is 16310080 KiB, below 16777216 KiB;
both Java birth identities remain running, trtexec absent, no new control or
target model launch.
[Updated large-host checks](../../artifacts/orin/20261025-01/large-host-profile-tests-002.xml),
[Orin admission](../../artifacts/orin/20261025-00/partition-admission-012.json).

### Complete Tail Result And Archive

The large-host job finishes in 602.031 seconds, exit zero, no guard event.
Peak sampled child RSS is 38746980352 bytes; minimum available host memory
is 604682506240 bytes. This is CPU export preparation cost, not VLM latency
or another platform's performance experiment.

The tail has 86886 nodes, 321 initializers, fifty inputs and two outputs:
I64 `[1,16]` tokens and BF16 `[1,1,248320]` logits. Its ninety observed SDPA
sites are all consumed, with ninety safe-softmax sites recorded. Checker
passes; runtime parity is explicitly skipped. The exported source uses the
reported Torch 2.7.1 / ONNX 1.19.1 / ORT 1.22.0 environment, not the local
Torch 2.10 environment. Do not infer cross-version numerical equivalence.

Metadata SHA256 is
`e458b93fe33d811f6ada39ca3ffd905e54ec87e1e75ce40bf6afed8da92cb23f`.
There are 254 external files totaling 3763638272 bytes. The exact source
tensor-contract list of all fifty prefix outputs matches the entire tail
input-contract list; this proves the declared source interface, not runtime
state handoff, reset or semantic equality after ONNX lowering.

The persistent NAS copy is under
`runs/orin-qwen2b-cpu-export-preparation-20261025-001/whole-tail-export-001/`.
All 256 files (model, external payloads and report) are fsynced and rehashed
against the published producer report and original scratch copy. A local
copy, `artifacts/orin/20261025-01/whole-tail-export-large-host-001/`, then
passes `sha256sum --check --quiet` for every file in the verified NAS receipt.
Six tiny positive/corruption tests validate the archive checker without model
inference. Parent/worker PIDs 626517/626518 are confirmed absent after exit.
No scratch cleanup or unrelated service manipulation is performed.

[Full tail export report](../../artifacts/orin/20261025-01/large-tail-export-report-001.json),
[terminal large-host worker](../../artifacts/orin/20261025-01/large-tail-export-worker-001/attempt.json),
[all-file persistent archive receipt](../../artifacts/orin/20261025-01/tail-archive-verification-001.json),
[archive checker tests](../../artifacts/orin/20261025-01/archive-verifier-tests-001.xml).

### Prefix Integer Controls

The earlier prefix is structurally simplified by the unchanged generic integer
folder: 190040 to 32566 nodes over four passes, 156.049 seconds, no guard event.
Public I/O and retained computations are unchanged; floating literals are
unchanged and floating arithmetic is not evaluated. Original external payload
hashes are checked before/after. This is structural preparation of the earlier
candidate, not full fifty-state numerical validation or safe-softmax policy
synchronization.
[Integer folding receipt](../../artifacts/orin/20261025-01/prefix-external-001/model-integer-shapes.report.json),
[terminal worker](../../artifacts/orin/20261025-01/prefix-integer-fold-worker-001/attempt.json).

## Remaining Orin Work

### Complete Host Numerical Evidence

All following runs are CPU-reference preparation, not Orin performance or
same-Orin quality acceptance. Each uses the single recovered sample-zero
observation, not sixteen independent observations or a formal CDF campaign.
Both stages now use the qualified safe-softmax policy. The generic typed-stage
executor has no model-name or fixed port-count branches, rejects missing/
unknown inputs, bad shapes/dtypes and nonfinite values, compares integer
counters exactly, and records source diagnostics separately from byte-exact
same-carrier comparisons. Together with generic precision-evidence checks,
22 focused tests pass.

| Check | Complete data checked | Result |
| --- | --- | --- |
| Prefix source diagnostic | 50 tensors / 7170050 values | Finite; two integer states exact; worst float MSE 0.26210504355025055, worst cosine 0.9413985020429413 |
| Isolated tail from original source state | All 16 tokens and 248320 final logits | Tokens exact; logits MSE 0.0006836813294606944, max-abs 0.12890625, cosine 0.9999346458626419 |
| Full prefix output fed into tail | All 16 tokens and 248320 final logits | Tokens exact; logits MSE 0.012132943794328745, max-abs 0.5, cosine 0.9988499292735197 |
| Before/after crop-control rewrite | All 50 tensors / 7170050 values | All finite and byte-exact; not just first-token equality |

The source diagnostics compare the saved same-host Torch source with the
portable ONNX implementation. They are not simply H20 cross-device error,
not source equivalence, and do not replace the approved same-Orin ordinary/
optimized primary gate. Both complete output arrays and negative errors remain.
The crop rewrite proof covers the same representation on identical inputs,
not precision conversion or target kernels.

[Prefix source diagnostic](../../artifacts/orin/20261025-01/prefix-oracle-source-002/report.json),
[tail source diagnostic](../../artifacts/orin/20261025-01/tail-oracle-source-001/report.json),
[complete host chain](../../artifacts/orin/20261025-01/full-chain-oracle-002/report.json),
[all-state crop parity](../../artifacts/orin/20261025-01/prefix-crop-parity-002/report.json),
[typed-stage and precision tests](../../artifacts/orin/20261025-01/typed-stage-and-precision-tests-002.xml).

### Fixed Precision Declarations

| Stage | Native BF16 MatMul | Source-low dense mapped to FP16 | Source-FLOAT dense retained FP32 | Largest legacy CLI argument |
| --- | ---: | ---: | ---: | ---: |
| Prefix | 187 | 117 | 655 | 114841 bytes, fits |
| Fifteen-step tail | 2805 | 270 | 195 | 332130 bytes, exceeds single-argument limit |

These source-derived declarations enable FP16 dense kernels, retain necessary
FP32 arithmetic and native BF16 providers, disable TF32/INT8 and explicitly
declare extra FP16 internal-result rounding. They are not a whole-model FP32
fallback, quantization search or target precision certification. Full actual
target output revalidation remains required. Tail needs the separately
qualified file-profile builder; CLI overflow is not an unsupported-model
finding. No target model is built by these declarations.
[Prefix precision](../../artifacts/orin/20261025-01/prefix-precision-002.json),
[tail precision](../../artifacts/orin/20261025-01/tail-precision-001.json).

### Native Work Still Open

Next is actual Orin build/pilot of the already staged 0.8B route, followed by
the independent 2B path if supported. Real engine I/O is required to complete
binding/type adapters and resident native-chain integration; do not guess it
from ONNX ports or promote host-only proof to deployment acceptance.
Real resident generation, reset/isolation, same-Orin complete-output quality,
TTFT, fixed-sixteen-token latency, throughput, memory and formal CDF are all
still open. Do not infer support from the 0.8B route or from this source check.

No Orin service is stopped for this preparation. The full model admission
remains 16 GiB, with a 4 GiB runtime floor; the earlier two-hour service window
is closed and cannot be reused. This report does not accept either Qwen model
or complete the overall Goal.

Historical read-only target admission at 20:55 UTC is 16158152 KiB, below the
16777216 KiB full-model gate. Both Java processes retain their original birth
identities, no trtexec is active, and no new service/model operation occurred.
[Admission snapshot](../../artifacts/orin/20261025-00/partition-admission-009.json).

Latest target admission at 22:57 UTC is 15499516 KiB, still below 16777216 KiB.
The original Java births persist, trtexec absent, no new service/model operation.
The separate new service window remains pending; the consumed old reply is
not reused. Neither Qwen model has accepted Orin metrics; Goal is blocked
pending safe admission or a new bounded grant, with the full objective intact.
[Latest admission](../../artifacts/orin/20261025-00/partition-admission-016.json).
