# SmolVLA TensorRT Precision Diagnosis

Date: 2026-10-20. Historical diagnosis, updated 2026-10-21: the original frozen
baseline and the failed candidates below remain unchanged. A later combination
of the generic LayerNorm plugin rewrite and original FLOAT solver-tail partition
passed the complete normalized FP32-source gate on all 16 observations. See the
[source qualification report](orin_smolvla_trt_source_qualification_20261021.md).
The original BF16 H20 supplemental MSE comparison still fails. The full
five-model Goal remains active; source qualification is not full delivery.

## Reproduction

The deterministic red-capable check recomputes normalized complete-action error
from real captured Orin outputs and the frozen all-16 FP32 ONNX reference:

```bash
python3 artifacts/orin/20261020-00/gate_source_action_fidelity.py \
  --reference artifacts/orin/20261020-00/onnx-complete-reference-003 \
  --candidate artifacts/orin/20261020-00/native-session-001/pilot-001 \
  --sample 10 --report artifacts/orin/20261020-00/audit/new-red-report.json
```

Use a new report path; the tool rejects overwriting evidence. The observed exit
code is 2, with MSE `2.6492215645769304e-5`, cosine `0.9999955295595925` and
max-abs `0.014011383056640625`. The unchanged gate is MSE `<1e-5`, cosine
`>0.9999`. Full validation still uses all 16 observations, not only this case.

## Hypotheses and Outcomes

| Hypothesis | Controlled Test | Actual Outcome |
|---|---|---|
| Explicit dense FLOAT output declarations improve emitted precision | Same source/checkpoint, compute constraints, inputs/noise and solver; add FLOAT outputs only | Prefix build fails at `/Conv`; step builds but all 34 full-chain action tensors are byte exact with old outputs, so error is unchanged |
| Fused iteration-state arithmetic is the dominant loss | Extract the original FLOAT tail and execute it in its own actual FLOAT engine | Formula and ten-step CPU shadow are byte exact; real complete-model gate still fails, with 5/16 failed unique samples |
| Prefix numerical error already causes the failed full-model gate | Feed one actual Orin prefix result into unchanged CPU FP32 source steps | On diagnostic sample 10, this path already has MSE `2.4471504819410944e-5`, close to the full-board `2.6492215645769304e-5` |

These tests do not identify a single faulty Conv, GEMM, norm or cast. No input,
checkpoint, iteration count, action dimension, scheduler formula or tolerance
was changed; no quantization/calibration search was run. The observations narrow
the next investigation to prefix numerical fidelity instead of repeatedly
changing the iteration tail.

## Output Declaration Trial

The inherited precision generator requested FP16 compute for dense operators
and FP32 arithmetic, but left dense output types implicit. The new common tool
`prepare_trt_precision_profile.py` makes selected FLOAT outputs explicit and
rejects duplicate/ambiguous names; INT/BOOL outputs are not retyped. The unit
regression first failed on missing dense output declaration and then passed.
That verifies declaration generation, not backend fidelity.

For the real trial, all compute constraints stayed exactly equal to the old
profiles. Only 334 prefix and 148 step dense output declarations were added.
This was one frozen declaration-policy trial, not a score-driven precision grid.

- `prefix-build-006`: trtexec child exit 1, first failure is no implementation
  for `/Conv` under the requested compute/output pairing. No engine was made.
- `step-build-002`: compiled, 200604691 bytes, SHA256
  `8980eb5d4e76b324b9d082c53226917479e3a7235caebeb476441b3c5aad8945`.
- Full model with only the new step: real native Session completed all ten
  iterations per request; both complete action files are byte exact with the
  old pilot. Source sample-10 MSE remains `2.6492215645769304e-5`.

The old and new step inspectors both expose a Myelin output in HALF, followed
by reformat to the public FLOAT tensor. Opaque fusion internals are not inferred
from that metadata. Additional declarations alone did not change the observed
complete outputs; this is not claimed as a numerical fix.

## Explicit FLOAT Arithmetic Module

The model-neutral ONNX partitioner identifies a flat standard FLOAT arithmetic
tail with an original input and a produced frontier. It extracts original nodes,
initializers and dataflow using ONNX extraction, not a rewritten solver. It
rejects effects/custom domains, dynamic interfaces and duplicated arithmetic.

For this graph the cut contains the existing `Mul` and `Add` at the output:
1109 original nodes become 1107 core nodes and three tail nodes, including one
INT passthrough Identity. All 148 matrix operations retain requested FP16 compute;
the two extracted original arithmetic operations retain FP32 compute. The union
of compute constraints is exactly the old step profile, with unchanged weights.

`split_region_invocation` composes two pure, uncached typed Region interfaces
inside the existing loop, preserves original result names and consumers, and
keeps the external IO schema digest unchanged. It does not inherit the old step
artifact contract or assert numerical equivalence by itself. This works through
the existing Adapter/IR/Plan/native Session path without model-name dispatch.

The ten-step CPU shadow on real observation 10 is byte exact for every original
versus composed output. Real target build receipts and inspectors show:

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Neural core | 200597648 | bb2ebd05e926ac037daeb87f4862303670273ffe3049d1439599a9cf97c12953 |
| Arithmetic tail | 17517 | 457423a165a4d8ac2c455df46998ee43b41e894453bebabad4b2e22f070c7c4d |

The tail has an actual FP32 PointWise output and INT32 passthrough. The complete
six-engine native Session was executed, still with ten solver iterations.
Its normalized gate failed 5/16 unique samples: 1, 5, 9, 10 and 11. Worst MSE is
`2.4558961563690607e-5` on sample 10, versus four failed samples in the baseline.
This candidate is **not selected**, despite its smaller worst-case MSE. Its
timing is not substituted for the original-input E2E baseline or paper rows.

An initial wiring attempt was rejected by the existing exact artifact type
checks because the starting native IR already used physical binding orders.
The separate second attempt restores all existing interfaces to logical order
before partitioning and reapplies actual engine binding order during preparation.
No type check was relaxed. Initial setup/partial-preparation evidence remains.

## Prefix Cross-Feed

A static native probe executed the unchanged real prefix engine on observation
10 and saved every BOOL/FLOAT cache output. Those actual bytes were then used
with the unchanged CPU source initialization, ten steps, finish and native-output
stages, through the same typed Invocation IR interpreter. This is numeric
attribution only; an injected-cache CPU path is not board E2E or a timing result.

| Normalized Diagnostic | MSE vs Frozen Full FP32 Source | Cosine |
|---|---:|---:|
| Cached CPU prefix + CPU steps (control) | 4.0298139825090117e-13 | 0.9999999999999198 |
| Real board prefix + CPU steps | 2.4471504819410944e-5 | 0.9999956806271454 |
| Real board prefix + real board steps (old full pilot) | 2.6492215645769304e-5 | 0.9999955295595923 |

The board-prefix/CPU-step result versus the full board result has MSE
`8.148341334433818e-7`. Thus, for this diagnostic observation, prefix discrepancy
is already sufficient to fail the source-relative gate even with FP32 source
steps. The cached CPU prefix came from the earlier rewrite check, not a claim of
byte equality with the optimized source; the measured control difference above
is retained and passes the stricter diagnostic control bound. Native-unit errors
are reported separately in the full analysis JSON.

MSE is not additive without its cross term, so no percentage contribution is
claimed. This single observation does not establish prefix dominance on all
16 samples, nor isolate vision versus language or a particular operator.

## Build Outcome Safety

The first trial observer found an important bookkeeping failure: the collector
process can exit 0 after successfully recording a failed trtexec child. Its
original top-level `built-not-model-validated` label is not authoritative for
the prefix build. Actual build records prove `compile-failed`, child exit 1
and no engine. Original evidence is not rewritten; the independent outcome
file records the correction.

The common receipt validator now requires actual child status/exit, exact source
identity and a positive-size complete engine receipt. Ten tests cover successful
collector/failed child and other incomplete metadata. The corrected observer
uses those checks and was used for the core/tail trial. A successful receipt
still does not replace actual engine hash checks, inference or complete fidelity.

## Source-Extracted Position Module

The next hypothesis was separated into observable checks. No K=1 outer-product
MatMul exists in the prefix position computation; that initial proposed cause
was rejected without changing the dense-compute policy. Source positions feed
a FLOAT division by the original timescale constant, followed by Sin/Cos.
The old inspector already shows FLOAT trigonometric inputs and outputs. It
does, however, explicitly show their subsequent conversion to HALF on entering
the main Myelin module. Metadata does not reveal every internal computation.

Offline counterfactuals on observation 10 show that rounding a correct CPU cache
to HALF alone gives normalized action MSE `1.0269368582264497e-8`, far below the
actual prefix cross-feed discrepancy. A HALF phase argument makes the K error
worse. HALF-rounded source frequency constants improve the observed first-K
error signature but do not reproduce it; no single RoPE cause is established.

The common `partition_onnx_values.py` extracts a dependency-closed producer at
declared tensor values. It preserves original nodes and constants, records all
crossing values and original-input routes, and handles passthrough-output
aliases without duplicating arithmetic. The existing generic typed Region
transform wires it into the native Session. It does not assert numerical
equivalence or model support on its own.

For this source the producer is 14 non-neural nodes: masks, positions, division
and trigonometry. It contains no Conv/Gemm/MatMul. The consumer has 2488 nodes,
including one passthrough Identity. Their union of requested computation
precisions exactly matches the old prefix profile; all 334 dense operations
remain requested HALF. Only the isolated non-neural producer disables global
HALF. Two real-observation CPU shadows give **66/66 byte-exact full cache
outputs**. Expanded host regression gives **583 passed, 40 opt-in skips**.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| Position producer | 86641 | bbc5b499190190eab4e762213a9bb37b0c6ce13b3cc9b2206ec7e5b90c730ead |
| Unchanged-policy neural consumer | 599907972 | 769eceafd1cc73add22bbe07e26eb32299c48f307935526a252c196805fc3e72 |

Actual producer execution on observation 10 checks all three outputs: the mask
is byte exact; Sin/Cos MSEs are `3.065937e-12` and `2.245594e-12`, with maximum
absolute errors `1.567602e-5` and `1.615286e-5`. This is not action acceptance.

The complete six-engine Session retains the original initialize/step/finish/
native-output artifacts, all ten iterations, and all 16 observations/noise.
PID `649933`, start ticks `3134779`, exited 0 after 1 warmup + 16 measured
calls. Both complete normalized/native action files are finite and audited.
The unchanged normalized source gate still **fails 6/16 unique samples**:
1, 2, 5, 7, 9 and 11. Worst source MSE is `2.1000131315416967e-5` on sample 11;
sample 10 improves to `4.958225451095396e-6`. This candidate is **not adopted**.
Against original H20, worst normalized MSE is `9.028566408263734e-5`; native-unit
source/H20 errors and per-dimension checks are retained separately. Neither
the improved diagnostic sample nor high cosine overrides the full failed gate.

## Extracted First K/V Diagnostic

To distinguish preceding vision error from downstream arithmetic, 24 original
nodes were extracted: the first RMSNorm, two matrix projections, RoPE and cache
materialization. Inputs are actual CPU source activations and the original
FLOAT Sin/Cos values; outputs are two complete `[1,177,5,64]` K/V tensors.
CPU extraction/composition is byte exact on real observations 10 and 11. These
injected-activation runs are **not full-board inference or performance data**.

| Source Observation | K MSE, Original Policy | V MSE, Original Policy | K MSE, One K Projection Requested FLOAT | V MSE, V Still Requested HALF |
|---|---:|---:|---:|---:|
| 10 | 5.853668e-6 | 3.844206e-9 | 2.494253e-13 | 1.055987e-10 |
| 11 | 6.263128e-6 | 3.938158e-9 | 2.567148e-13 | 1.111891e-10 |

Original-policy engine SHA256:
`15936d16434a997efb50b7f85d86ee91a786faa588e4257aea692dc27a731fe4`.
One-K-FLOAT engine SHA256:
`856f1167fad7db9efb67d34ee55126cdbb228a39003347818587c71070a9a840`.
The only requested compute change is `/MatMul_195` HALF -> FLOAT, with its
output explicitly FLOAT. V projection remains HALF. Both inspectors expose an
opaque Myelin module with FLOAT public inputs/outputs; V output values are
100% HALF-representable in both cases. Its error also improves, so the outcome
cannot be attributed solely to the K GEMM independently of shared norm/fusion.

This evidence motivates one declared category trial: find the closest unique
constant-weight MatMul producer of each of the 16 declared cache-key outputs
and request it FLOAT. All other 318 prefix dense operations remain requested
HALF, source nodes/weights remain unchanged and the ordinary step engine is
retained. The full-model result must be checked separately before any adoption;
the minimal diagnostic is not a model pass or an all-FP32 fallback.

That full-prefix trial has now executed, not merely compiled. The new consumer
engine is 609777479 bytes, SHA256
`372d2f854940d5b65e48db7953e250f72eab641a6d8f26952a35b293abad1609`;
its build took `107.420852224 s`. A fresh native six-engine Session retains the
same source partition, positions, four nonprefix artifacts, inputs/noise,
external IO contract and all ten solver steps. Pilot PID `667965`, start ticks
`3221396`, exited 0; 17 complete normalized and 17 native tensors are finite.

| Complete Model Candidate | Source-Gate Failed Unique Samples | Worst Source Normalized MSE | Worst H20 Normalized MSE |
|---|---|---:|---:|
| Original five-engine baseline | 1, 5, 9, 10 | 2.649222e-5 | 6.393668e-5 |
| Position-only split | 1, 2, 5, 7, 9, 11 | 2.100013e-5 | 9.028566e-5 |
| Position split + 16 explicit FLOAT K projections | 1, 8, 9, 11 | 1.512044e-5 | 7.370220e-5 |

The last candidate is still **not adopted**. Source-worst observation 11 has
MSE `1.5120439304874278e-5`; minimum normalized source cosine is
`0.9999961339329697` on observation 9. Complete native-unit source MSE reaches
`0.0330194205439462`, reported separately from the normalized acceptance gate.
The actual inspector still shows HALF Sin/Cos inputs to the main Myelin and
HALF cache materializations widened to public FLOAT outputs. Opaque internals
are not inferred solely from requested precisions or the larger engine size.

The localized numerical result did not establish a complete-model fix. Earlier
sample-10 attribution is not proof of the remaining failure's location. No formal
sampling or CDF is started from this failed candidate, and no gate is relaxed.

## Current-Candidate Cross-Feed and Image Path

The exact position/key-FLOAT candidate was captured on observation 11 without
rebuilding either engine. A separate indexed-IO native diagnostic probe binds
inputs by actual engine order and records every declared route; all original
input hashes and both engine hashes are checked. Its two file-mediated calls
save all three position outputs and all 33 final prefix outputs. This probe
is not resident E2E execution or a latency measurement.

| Observation-11 Normalized Diagnostic | MSE vs Frozen Full FP32 Source |
|---|---:|
| CPU-cache / CPU-step control | 7.167304e-14 |
| Actual current prefix / CPU steps | 1.587114e-5 |
| Actual current prefix / actual board steps | 1.512044e-5 |
| Actual-prefix CPU-step result vs full-board result | 7.435584e-7 |

Both injected-prefix CPU paths execute the original initialize, all ten steps,
finish and native-output stages through the same Invocation IR interpreter.
The CPU control's max-abs is `7.748604e-7`. Thus the current prefix is already
sufficient to fail this observation even with unchanged FP32 CPU steps. This
does not establish a percentage attribution or prove every other sample's cause.

All 32 cache tensors were also compared in the pinned export's token groups:
64 camera1, 64 camera2, 48 language and one state token. First-key MSE is
`5.724218e-5` for camera1, `1.744718e-4` for camera2, `1.738078e-7` for language
and `6.294045e-8` for state. This localizes a substantial first-layer visual
discrepancy; it does not identify every subsequent Transformer operation.

The actual full-engine inspector shows FLOAT Resize, Pad and pixel normalization,
followed by two HALF patch Convs and HALF inputs to the main Myelin. To test
that observation directly, ten original nodes were extracted: image preparation
and the two patch projections, without any Transformer MatMul or norm. The
same real observation 11 was used, with all four complete tensor outputs saved.

| Complete Tensor | Original HALF-Conv MSE | FLOAT-Conv Diagnostic MSE | Original Max-Abs | FLOAT Max-Abs |
|---|---:|---:|---:|---:|
| Camera1 prepared image | 2.151405e-15 | 2.151405e-15 | 2.384186e-7 | 2.384186e-7 |
| Camera2 prepared image | 1.644165e-15 | 1.644165e-15 | 2.384186e-7 | 2.384186e-7 |
| Camera1 patch projection | 3.233954e-7 | 2.373200e-15 | 0.0167622 | 1.907349e-6 |
| Camera2 patch projection | 4.527657e-7 | 2.888275e-15 | 0.0622635 | 3.203750e-6 |

The prepared-image output bytes are unchanged between the two real target
engines. Only the two patch Conv compute/output declarations change to FLOAT.
This is an extracted primitive diagnostic, not a full FP32 model or a claim
that all remaining visual discrepancy is fixed. Engine identities are:

- Original-policy patch engine:
  `b20cc7c3185213b30d9c90291a1105be9b349a0b8974b36cacc53e933ca1a165`.
- FLOAT-patch diagnostic engine:
  `2be6e4a6fd599c4ce1a26d27d6034b6b2e0c3e66495ede44d2e6dea68baea05c`.

The resulting bounded full-model trial changes only these two Conv declarations
relative to the position/key-FLOAT candidate. All 316 remaining prefix dense
operations still request HALF; all original source nodes, weights, inputs/noise,
ten solver iterations and nonprefix artifacts remain unchanged. It must pass
the original full-16 action gate before any qualification or formal sampling.

The complete trial has now run with actual FLOAT Conv IO confirmed by the
inspector. Consumer engine SHA256 is
`fca6c023ad5a72a381f3c2fb83d0f535df369c7279bf27334b7b2c93d30f3927`,
610531988 bytes, build duration `105.851667840 s`. Native pilot PID `690402`,
start ticks `3330561`, exited 0. All 34 complete action tensors are finite.
The frozen normalized source gate now passes **15/16 unique observations**,
but observation 6 still fails: MSE `1.2165750625023526e-5`, minimum cosine
`0.9999975136386287`. Worst native-unit source MSE is `0.02973805789242019`.
The original H20-relative normalized MSE still reaches `1.1063809040964564e-4`;
it is retained as supplemental cross-profile evidence, not silently accepted.
This candidate is **not adopted** and does not enter formal sampling.

Observation 6 was subsequently captured from this exact two-Conv candidate.
Actual prefix plus CPU source steps still fails, MSE `1.1546298205742749e-5`;
CPU-prefix/CPU-step control is `8.905598480879462e-14`. The mixed CPU-step result
versus the full-board result has MSE `5.482245110503363e-7`. Prefix discrepancy
therefore remains sufficient to fail this observation. Adding the earlier
arithmetic-tail candidate solely to chase a passing score is not justified by
this attribution.

## Installed LayerNorm Plugin Probe

The actual target registry exposes 68 creators, including `LayerNorm`, version
1, empty namespace, with an `epsilon` field. It does **not** expose an `axis`
field on this installed release. This is not assumed identical to later 8.5
sources. The original prefix contains 50 lowered LayerNorm groups, all with
input shape `[1,1024,768]` and original reduction axes `[2]`.

The current plugin was tested directly before any whole-model rewrite. Nine
original normalization nodes provide the CPU reference; their actual upstream
FLOAT source activation from observation 6, gamma and beta are saved in full.
Original epsilon is preserved exactly as FLOAT, `9.999999974752427e-7`. The
rank-three/last-axis equivalence is explicitly checked; other ranks or axis
semantics are not silently assumed supported.

An ONNX custom-node probe compiles using the installed creator. A separate native
C++ wrapper calls `initLibNvInferPlugins` before engine deserialization, without
Torch or SDK changes. The actual inspector shows one `PluginV2` with all three
inputs and its output in linear FLOAT format. Engine SHA256:
`6a3bc879843fedb286ef31cb7ff035beec258b10e61221ce9ccb4ec7d759bd63`,
2856 bytes. All 786432 original output values were compared:

| Primitive Metric | Actual Result |
|---|---:|
| MSE | 2.3182053757621255e-15 |
| Max absolute error | 2.86102294921875e-6 |
| Cosine | 0.9999999999999929 |

The prospectively fixed primitive gate (finite, MSE `<1e-10`, max-abs `<1e-4`,
cosine `>0.999999`) passes. This is **one real-input primitive**, not a full
model pass, benchmark or proof that LayerNorm alone caused the remaining error.
The current production backend has not yet been changed to register standard
plugins, and no full prefix graph has yet been rewritten to use this plugin.
Next integration needs conservative model-neutral source-pattern/axis/epsilon
checks, explicit registration and real complete-model requalification.

NVIDIA's [LayerNorm plugin description](https://github.com/NVIDIA/TensorRT/blob/release/8.5/plugin/layerNormPlugin/README.md)
provides the mathematical/input contract and documents the older rank-dependent
normalization behavior. The installed registry and actual original-node probe
above, not a newer interface assumption, establish this target's tested scope.

## Evidence and Remaining Work

All paths below are relative to `artifacts/orin/20261020-00/`:

- `audit/source-gate-red-sample10-001.json`: original deterministic red case.
- `float-output-trial-001/independent-outcomes.json`: actual compile outcomes.
- `build/{prefix-build-006,step-build-002}/`: complete build metadata/logs.
- `native-step-float-output-001/pilot-001/`: unchanged full-chain outputs.
- `step-fp32-tail-partition-001/`: source/core/tail, original-node ledger,
  ten-step shadow and typed IR wiring; successful wiring is `wiring-002/`.
- `fp32-island-build-001/` and `build/step-{core,tail}-build-001/`: actual builds.
- `native-fp32-island-002/pilot-001/`: complete candidate actions and snapshots.
- `audit/source-gate-island-full16-001.json`: unchanged strict gate, failed.
- `prefix-crossfeed-001/{capture,analysis}.json`: real prefix and CPU cross-feed.
- `audit/fp32-island-regression-001.xml`: 575 passed, 40 explicit GPU/LibTorch
  opt-in skips; separate real target inference evidence is listed above.
- `prefix-rounding-diagnostic-001/analysis.json`: endpoint-rounding control.
- `prefix-phase-partition-001/`: original-node position partition, CPU shadows,
  binding ledger, position counterfactuals and typed Session wiring.
- `build/prefix-{phase,neural}-build-001/`: target build receipts/inspectors.
- `prefix-phase-capture-001/comparison.json`: all actual producer outputs.
- `native-prefix-phase-001/pilot-001/`: real complete candidate, not adopted.
- `audit/source-gate-prefix-phase-full16-001.json`: failed fixed source gate.
- `audit/prefix-phase-complete-output-001/{source,h20}.json`: all 34 action
  tensors per reference, native-unit errors and per-dimension evidence.
- `prefix-kv-core-probe-001/`: original-node subgraph, frozen inputs, exact CPU
  shadows and original/one-K-FLOAT profiles.
- `prefix-kv-{core,key-float}-capture-s{10,11}-001/comparison.json`: actual
  complete K/V tensors under both requested computation policies.
- `prefix-key-float-build-001.json`: declared 16-node category trial, preserving
  318 HALF dense operations; not by itself evidence of successful inference.
- `build/prefix-neural-key-float-build-001/` and `prefix-key-float-build-001/`:
  actual target build/inspector and independently verified child outcome.
- `native-prefix-key-float-001/pilot-001/`: actual six-engine full action pilot.
- `audit/source-gate-prefix-key-float-full16-001.json`: failed fixed source gate.
- `audit/prefix-key-float-complete-output-001/{source,h20}.json`: both complete
  action outputs, all 17 calls, native-unit and per-dimension errors.
- `prefix-key-crossfeed-s11-001/{capture,analysis,cache-token-groups}.json`:
  exact current prefix, original CPU steps and all cache token groups.
- `patch-embedding-probe-001/`: original ten-node image/patch subgraph, frozen
  observation-11 inputs, CPU outputs and both requested precision policies.
- `build/patch-embedding-{,float-}build-001/`: both real target builds/inspectors.
- `patch-embedding-{,float-}capture-001/comparison.json`: all four complete
  preprocessing/projection tensors under the controlled policy change.
- `prefix-key-patch-float-build-001.json`: bounded two-Conv full-model trial,
  with 316 other prefix dense operators still requested HALF.
- `build/prefix-neural-key-patch-float-build-001/` and
  `native-prefix-key-patch-float-001/pilot-001/`: real full candidate and all
  complete action bytes; failed fixed gate is
  `audit/source-gate-prefix-key-patch-float-full16-001.json`.
- `audit/prefix-key-patch-float-complete-output-001/{source,h20}.json`: both
  references, all normalized/native actions and per-dimension errors.
- `prefix-key-patch-crossfeed-s6-001/{capture,analysis}.json`: current exact
  remaining failure, CPU control and all ten source steps.
- `trt-plugin-registry-001.csv`: actual 68-creator target registry snapshot.
- `layernorm-plugin-probe-001/`: original nine-node reference, complete real
  activation/gamma/beta, exact epsilon and custom-node deployment probe.
- `build/layernorm-plugin-build-001/` and
  `layernorm-plugin-capture-001/{capture,comparison}.json`: actual registered
  FLOAT plugin execution and all output values, primitive gate passed.
- `patch-plugin-setup-attempts.json`: retained harness build/API/axis/transfer
  setup failures, explicitly separate from model results.
- `audit/prefix-phase-partition-regression-001.xml`: 583 passed, 40 opt-in skips.
- `audit/prefix-key-diagnosis-focused-001.xml`: 32 focused host tests passed.
- `audit/prefix-patch-plugin-focused-001.xml`: 32 focused host tests passed.
- `host-parity/prefix-001/external-weight-regularization.json`: our historical
  sidecar hardlink was replaced with a verified regular copy after ONNX's
  multiple-hardlink security rejection. Both sidecars now have nlink 1 and
  unchanged SHA256 `ba0efb7fba4c8de08fd054e6bf712af9584c6cdbf607d73c12f0cedafc53a921`.

No numerical fix, new E2E speedup, formal CDF or Goal completion is claimed.
Next work should conservatively integrate the tested installed LayerNorm plugin
through generic graph lowering and backend registration, then recheck every
complete action while retaining the old baseline and all negative candidates.
The primitive pass and earlier observations do not authorize an unbounded
precision search or relaxing the full-model gate.
Optimized Session fault/lifecycle,
memory/thermal/operator experiments and the four other required VLA models
remain unfinished.
