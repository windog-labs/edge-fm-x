# Orin Operator Ablation Preparation

Historical source-preparation checkpoint: 2026-10-21. The table and source-only
test totals below describe that checkpoint; current target amendments follow
the table. LayerNorm/GEMM/Attention have exposed-format pairs and state projection has
a negative configured-policy pair with an explicit fused-precision limitation.
The complete five-model Orin goal and D4 remain incomplete. These target pairs
run only after RDT formal sampling stopped, not concurrently with it.

Latest source checkpoint: six real-input families are host-prepared: LayerNorm,
RoPE/cache, GEMM, Attention Core, language Embedding and VLA state
materialization/projection. RoPE and state packing reuse the same conservative
generic static-Scatter lowering. At that checkpoint all were host-prepared, not
target-built or timed. Full source regression remains the earlier 2639 passed / 74 skipped /
six subtests. The six real case families/static-index checks have 110 passing
tests; the independent controller has 33, **143 combined**. These additions
are artifact-only and do not rerun or replace the full source regression.

| Family | Real Source Coverage | Bounded Candidate | New Target Pair |
|---|---|---|---|
| LayerNorm | One activation, 786432 complete values | Installed standard plugin | Not built/timed |
| RoPE/cache | Two observations, 113280 complete values | Static Scatter layout rewrite | Not built/timed |
| GEMM | Two V projections, 113280 complete values | Constant transpose/Gemm representation | Not built/timed |
| Attention Core | One observation, 786432 complete context values | Proven zero-mask Add elision | Not built/timed |
| Language Embedding | One distinct instruction across 16 observations, 46080 output values | Immutable full-table scalar fold | Not built/timed |
| VLA state feature | 16 distinct states, 15360 complete token values | Static state-pack layout rewrite | Not built/timed |

Each case has one fixed candidate, not a proven positive gain. Six prepared
operator families do not mean the five required model experiments are complete.

Later target progress on 2026-10-22: the new LayerNorm baseline/candidate both
build under matched 256 MiB workspace and original FP32 normalization contract.
All 786432 real output values pass the same-Orin gate (MSE 2.290928e-15,
cosine 0.9999999999999932), not byte-exact. Two five-pair timing boundaries now
pass independent audit, with 20480 raw records kept: resident compute
186.408 -> 43.062 us (4.329x), IO event latency 489.500 -> 297.729 us (1.644x).
Candidate resident std and IO max are not uniformly better and are retained.
Full-model performance effect remains unmeasured. The source-preparation table above is historical; target evidence
is in [LayerNorm target pair](orin_operator_layernorm_pair_20261022.md).
Subsequent GEMM target progress covers both real observations / 113280 values:
same-Orin worst MSE 3.836792e-9 and minimum cosine 0.999998009876 pass. Both
five-pair timing boundaries are independently audited with all 20480 records:
resident compute 38.142144 -> 26.338613 us (1.448146x); IO event latency
59.466626 -> 58.993376 us (1.008022x). IO-enabled compute slightly regresses,
and CPU-reference diagnostic error is higher. These negatives remain. Full
evidence is in [GEMM target pair](orin_operator_gemm_pair_20261022.md).
The later [complete GEMM integration and same-source rebuild control](orin_operator_gemm_integration_20261022.md)
rejects this trial for formal integrated E2E: normalized MSE exceeds the fixed
gate against both the adopted model and fresh control. Isolated speed is not
promoted to accepted model performance.
The later [state configured-policy pair](orin_operator_state_pair_20261022.md)
covers all 16 states / 15360 values and 20480 timing rows, but baseline Myelin
internals remain opaque. Matching FP16/OBEY configuration is checked without
claiming observed effective precision. Resident/IO event ratios 0.980551x /
0.979880x are negative; this recipe is not selected for integration.
The later [Attention target pair](orin_operator_attention_pair_20261022.md)
has full byte-exact context output and accepted 20800-row timing evidence;
resident/IO event ratios are 1.458263x/1.239668x. Its first short-count attempt
remains rejected. Embedding and RoPE remain host-only. The user confirmed
normalized complete action as the primary Norm MSE/cosine gate, with native
errors supplemental. Full raw-observation Norm pilots pass; the
[five-process-pair formal E2E](orin_operator_layernorm_raw_e2e_20261022.md)
is now audited: 125.900306 -> 133.994095 ms (0.939596x, negative), with all
23040 output tensors checked. No integrated E2E gain is certified.
See [current operator summary](orin_operator_ablation_results_20261022.md).
No RDT formal restart or substitute for its missing fifteen-process acceptance
is implied.

## Bound Source Inventory

The adopted seven-engine SmolVLA profile is bound through its actual successful
build receipts, source graph hashes, model preparation and completed formal
audit. Inventory evidence:
`artifacts/orin/20261021-08/smolvla-qualified-operators-001/inventory.json`.

| Region | Static ONNX Nodes | Logical Signature Groups | Nominal Invokes per Fresh Chunk |
|---|---:|---:|---:|
| initialize | 1 | 1 | 1 |
| finish | 8 | 8 | 1 |
| native output | 17 | 12 | 1 |
| prefix phase | 14 | 12 | 1 |
| prefix neural | 2088 | 98 | 1 |
| step numeric core | 1107 | 102 | 10 |
| step numeric tail | 3 | 3 | 10 |

Total: 3238 source nodes and 236 per-region signature groups. Invoke
multiplicities are static plan interpretation with fresh inputs, not measured
kernel launches, cache hit rates or hotspot rankings. ONNX logical FLOAT IO
does not establish target FP32 compute or physical strides. Decomposed
Attention/RoPE/norm, Embedding and VLA roles are not marked N/A because a
generic raw-node inventory has not assigned their semantics.

## First Real-Input Pair

One original LayerNorm activation from recorded observation 6 is reused in
full, together with unchanged gamma/beta, FLOAT epsilon and frozen output.
The activation is an original **CPU source upstream activation**, not a newly
captured activation of the current mixed-precision Orin engine. The case binds
to the adopted prefix's `vf_layer_norm_0` through exact input/output names,
initializer bytes, epsilon, graph/build hashes and the formal engine identity.
It does not infer binding from a similar node label.

| Property | Prepared Contract |
|---|---|
| Shape | `[1,1024,768]` |
| Complete output values | 786432 |
| Numeric IO | FLOAT, the original required normalization island |
| Epsilon | `9.999999974752427e-7`, unchanged |
| Baseline | Nine original standard ONNX primitives |
| Candidate | One installed standard TensorRT LayerNorm node |
| Runtime ports | Identical one-input/one-output contract |
| Gamma/beta | Identical constant-weight roles in both variants |
| Initializers | All initializer protobufs identical |
| Target pair build/quality/timing | Not performed |

This isolated FLOAT normalization contract is not a whole-model FP32 fallback
or a new precision-search campaign. The adopted full model's FP16-dense profile
and declared numerical islands are untouched. Source node counts 9 versus 1
are **not** a 9x gain or nine versus one actual CUDA launches: TensorRT may fuse
the baseline and actual target kernels must be inspected.

The older target plugin probe used three dynamic inputs (x/gamma/beta), whereas
this matched pair has x as its sole runtime input and both affine vectors as
constants. The old probe engine is therefore **not reused** as the new candidate
engine, and its successful quality result does not qualify this pair.

New artifacts: `artifacts/orin/20261021-08/layernorm-pair-001/`.

- `baseline.onnx`, SHA256
  `53a49a5c4cce48ec2f9471614df495707c758d4daa4b73ba65a60bd44595f89a`.
- `candidate.onnx`, SHA256
  `d88a1964dff1477ffe208d9f262180d7e74df8dff6f3fd8ffe0ade7931dbad2f`.
- `standard-shadow.onnx`: portable semantic reference, not target-plugin execution.
- `inputs/0.bin`, complete expected output, exact port declarations and hashes.
- Matching precision requests, typed rewrite proof and detailed `case.json`.

## Host Numerical Evidence

CPU ORT 1.24.4 runs original primitives and their portable standard-ONNX shadow
against all frozen output values; graph optimizations are disabled. It does
not run the TensorRT plugin or a GPU performance experiment.

| Comparison | MSE | Max Abs | Cosine | Byte Exact |
|---|---:|---:|---:|---|
| Original primitives vs frozen | 0 | 0 | 1 | yes |
| Portable standard shadow vs frozen | 2.928778e-14 | 8.583069e-6 | 0.9999999999999285 | no |

Both satisfy the pre-existing primitive gate: finite, MSE < 1e-10,
max-abs < 1e-4, cosine > 0.999999. This is one real observation/case, not
all-layer/all-model equivalence. The shadow check is not new target quality.
All source inputs, constants and expected values remain unchanged.

## Candidate and Measurement Boundaries

The bounded candidate set is one previously functionally motivated, generic
typed graph-rewrite recipe to an installed NVIDIA standard plugin. The recipe
source hash is recorded. It is not autonomous custom-kernel synthesis or a
new search over quantization/precision. Target build failures, search/compile
costs and timing gains remain unknown, not zero.

After RDT formal collection and without taking priority from the next core
model's target work, use a genuinely idle board window to:

1. Build both variants under matched declared norm precision; inspect actual
   target IO, placement, plugin identity, formats and build failures.
2. Independently verify every complete target output before timing.
3. Pilot the two variants and freeze repetitions before five independent
   paired-process comparisons. Separate operator GPU/host boundaries from
   raw-model E2E, include actual GPU completion and retain non-gains.
4. Revalidate any effective candidate inside a real VLA and repeat complete
   output/E2E comparison. Existing integration alone does not close a new
   before/after operator or whole-model gain claim.

Other required operator families still need semantic boundaries, actual input
cases, target comparisons and appropriate precision contracts. The existing
small K/V diagnostic contains real RMSNorm, matrix and RoPE inputs, but is not
reclassified as a complete Attention, VLA-specialized kernel or timing result.

## Regression

`artifacts/orin/20261021-08/layernorm-case-regression.xml`: **68 passed** across
real-case checks, generic rewrite and ONNX inventory. Negative fixtures reject
rehashed altered gamma, altered complete reference, dynamic ports, wrong output
binding and a silent input-precision change. The fixtures never overwrite the
original source/probe files. The pair's complete original inputs and constant
roles are checked separately from the earlier dynamic-weight engine contract.

At the initial LayerNorm-only checkpoint, the prior full host suite was 2630
passed / 74 skipped / six subtests; artifact-only preparation did not rerun it.
The subsequent static-Scatter change and new full regression are described
below. No C++ runtime/backend or frozen running model/engine was changed.

## Real RoPE and Cache Pair

Canonical evidence: `artifacts/orin/20261021-08/rope-pair-004/`. Earlier
preparations are retained with their recipe source snapshots; the canonical
case additionally binds all sidecar/profile hashes and the final conservative
dependency checker. Its graph and numerical outputs are unchanged from the
earlier real-input preparations.

The declaration in `rope-pair-request-001.json` selects a real source boundary
from the already frozen 24-node K/V diagnostic. Preparation is model-neutral:
ports and semantic roles are explicit declarations, while source binding
checks every selected operator, attributes, renamed tensor boundary, initializer
bytes, build receipt and the adopted full-model engine identity.

| Property | Prepared Evidence |
|---|---|
| Main input | Original CPU pre-RoPE K, `[1,177,5,64]` FLOAT |
| Other inputs | Original FLOAT Sin/Cos, each `[1,177,1,32]` |
| Source observations | 10 and 11, not independently generated trajectories |
| Output | Complete rotated K/cache tensor `[1,177,5,64]` |
| Complete values checked | 56640 per observation, 113280 total |
| Baseline | 12 source nodes, including two ScatterND |
| Candidate | 14 source nodes, zero ScatterND, Slice/Concat layout replacement |
| Non-Scatter arithmetic | Definitions and tensor boundaries unchanged |
| CPU result | Byte-exact with original frozen full K outputs for both observations |
| Target build/quality/timing | Not performed |

Two index tensors are independently bound to immutable INT64 initializer
contents. Source K/V outputs remain byte-exact when upstream observation points
are added. The extracted baseline composes byte-exactly back into those
original outputs; the candidate also matches every byte. Original normalized
activation and both projection outputs are saved for later GEMM/RMSNorm input
work, explicitly labeled captures rather than completed benchmark cases.

The baseline and candidate retain the same three-input/one-output FLOAT RoPE
island contract. No dense precision policy, scheduler or model action iteration
is changed. This is not a whole-model FP32 fallback. Logical input precision
still does not certify actual target compute/layout, which remain future gates.
More source nodes after lowering do not imply more CUDA kernels or a slowdown;
zero ScatterND does not by itself establish a speedup.

The candidate graph SHA256 is
`c3897c380aa9943ba6881d6fa79ee6c2466caa69a24e40e63499390fc14920e5`.
Its 2039040-byte `weights.bin` payload SHA256 is
`c3831b16d6ed077b44173972767558f1c0aa0d876d4cdea748ad6bda17ecdd1e`.
Both ONNX graph hashes, precision-profile hashes, all raw samples/references,
external payload, request, rewrite report and helper/recipe source identities
are recorded. No target engine or timing is borrowed from a diagnostic probe.

## Conservative Static-Index Lowering

The old generic pass used three equal-result probe feeds as evidence of static
indices. Regression demonstrates a counterexample: `x*(x-1)*(x+0.5)` vanishes
at all three old probe values but changes at another valid input. It must not
be frozen into a static write. New checks require index value independence
from immutable initializers/deterministic constant dataflow or fixed-profile
graph-input shapes. Runtime-overridable initializers, random/custom operations
and unproved intermediate shape annotations do not establish constantness.

Literal index tensors are read directly. They do not instantiate a complete
CPU model or try to execute unrelated backend plugins. Other proven constant
dataflow gets additional numerical probes, not a probe-only independence claim.
Per-node records use unique SSA outputs, not optional/repeated debug names;
generated replacement names are unique. Probe debug-label normalization does
not change original source tensor dataflow. Successful rewriting still retains
`requires_numerical_verification`, not a whole-model acceptance claim.

Red/green evidence and frozen earlier recipe sources are preserved under
`artifacts/orin/20261021-08/`. These changes do not modify or retroactively
reclassify running RDT engines or previous BPU measurements.

Final checks:

- `rope-final-regression.xml`: 21 passed, including the two actual observations,
  complete original outputs, sidecar/profile hashes, rehashed bad input/reference
  and same-name-but-wrong-operation rejection, plus all static-index boundaries.
- `scatter-final-regression.xml`: 15 passed for generic lowering, including
  shape-only positive cases, runtime overrides, random/custom operations,
  unknown rank, intermediate annotations and optional/duplicate debug labels.
- `full-host-regression-scatter-final.xml`: 2639 passed, 74 skipped, six subtests.

The next target step remains matched build/inspection and complete output
verification before paired timing in an idle board window. Attention,
Embedding, VLA-specialized cases and meaningful integrated timing still remain
required; raw decomposition or a host-only match cannot close those tasks.
The later Attention Core preparation below does not provide its target result.

## Real Constant-Weight GEMM Pair

Evidence: `artifacts/orin/20261021-08/matmul-pair-001/case.json`. The declaration
selects the V projection from the original frozen K/V diagnostic, with the
activation and complete output captured during the earlier source CPU run.
Exact node definition, weight values, source/build receipts and the formal
SmolVLA engine hash bind it to the adopted model, not merely a similar shape.

| Property | Prepared Evidence |
|---|---|
| Source operation | `/MatMul_196`, constant-weight V projection |
| Activation | FLOAT `[1,177,960]`, original CPU source activation |
| Constant weight | FLOAT `[960,320]`, logically unchanged |
| Output | FLOAT `[1,177,320]`, complete original frozen V-cache bytes |
| Observations | 10 and 11; 113280 output values across the two |
| Baseline | One original MatMul |
| Candidate | Flatten + Gemm with transposed constant/transB + runtime shape restoration |
| Candidate source nodes | Six, not an actual kernel count |
| Requested dense compute | Same adopted FP16 request in both variants, TF32 disabled |
| Output constraint | No new precision override, preserving the adopted declaration |
| CPU source/candidate result | Byte-exact with both original complete V outputs |
| Target effective precision/build/quality/timing | Not verified |

The generic recipe handles known-rank one-dimensional and batched floating
activations with one shared immutable rank-two matrix. Batch dimensions come
from runtime Shape, not a guessed intermediate annotation; dynamic batch
prefixes have unit coverage. The stored matrix is contiguous `B.T` and Gemm
uses `transB=1`: inverse transposition reproduces the original values byte for
byte. No Cast, model-name dispatch, quantization search or new FP32-dense
fallback is introduced. This is one bounded candidate, not autonomous kernel
synthesis. TensorRT may select identical tactics, and no gain is presumed.

The original source request is FP16; the adjacent K projection's different
precision request is not copied. FLOAT IO alone does not imply FP32 compute.
The CPU FLOAT comparison checks extraction/recipe semantics only. Future target
acceptance compares **same Orin, same effective precision**, conventional
versus candidate, MSE < 1e-5 and cosine > 0.9999. It must not impose an extra
strict FP32-source gate on the FP16 target or infer actual precision from a
builder request. Complete target outputs and effective precision/format parity
must be inspected before timing or full-model integration.

Baseline graph SHA256:
`4598e127c464d801897065085405eb3b64e686bca8ce2b2afb28c91b43ffd4f0`.
Candidate graph SHA256:
`2ae9c905e7e48ea43b6997fe03ba7afa218f64033db4d08fdaf7a8973f1bbf72`.
The case records both precision-profile hashes, source/capture/recipe hashes
and every complete input/reference hash. Both graphs use inline weights.

## Combined Case Regression

`artifacts/orin/20261021-08/real-operator-cases-regression-r2.xml`: **48 passed**
(20 GEMM, six RoPE, seven LayerNorm and 15 generic static-index checks).
GEMM coverage includes FLOAT/FLOAT16 ranks 1--4, dynamic batch prefixes,
unchanged logical weights, real FP16 declarations, signed-32-bit bounds of
actual shape controls, complete original frozen outputs, and rejection of
wrong precision requests or rehashed corrupted captures. No target timing is
generated by these tests.

The first combined run's failure is preserved in
`real-operator-cases-regression.xml`: a negative fixture hard-linked an external
payload, which ONNX 1.21 correctly refuses to load with multiple links. The
fixture now makes independent copies. Only the two identified task-owned
temporary links were replaced with independent byte-identical copies; the
original payload inode/hash are unchanged and its link count is again one.
Receipt: `fixture-hardlink-repair-001.json`. No loader safety check was disabled,
and no original model/reference or running target engine was modified.

## Real Attention Core Pair

Evidence: `artifacts/orin/20261021-08/attention-pair-001/`. This case selects the
first real vision attention core, not a synthetic matrix product or full-model
E2E substitute. Original Q/K/V projection and scaling operations are retained
in a source-upstream graph. All their node definitions and initializer bytes
are bound to the adopted prefix, its successful build receipt and formal
engine identity. Upstream input is the original CPU normalization output for
recorded observation 6, hash-bound to the earlier real LayerNorm case.

| Property | Prepared Evidence |
|---|---|
| Q, already source-scaled | FLOAT `[1,12,1024,64]` |
| Transposed K, already source-scaled | FLOAT `[1,12,64,1024]` |
| V | FLOAT `[1,12,1024,64]` |
| Complete context output | FLOAT `[1,12,1024,64]`, 786432 values |
| Source mask | Immutable positive-zero FLOAT `[1,1,1024,1024]` |
| Baseline | QK MatMul, zero-mask Add, last-axis Softmax, PV MatMul |
| Candidate | Same core with only the proven-neutral mask Add removed |
| QK/PV request | Original FP16, no new output-precision constraint |
| Softmax request/output | Original FP32 numerical island retained |
| CPU source/baseline/candidate | Every complete context value byte-exact |
| Target build/precision/quality/timing | Not performed |

This source context is newly recomputed on CPU from a previously frozen real
activation, not an older H20 full-model context or an actual mixed-precision
Orin activation. Projection/scaling are outside the isolated core's three-input
timing boundary; they must remain inside any full fresh-model E2E boundary.
No whole-model result or reduced E2E workload is inferred from this extraction.

The bounded generic recipe requires matched positive static matrix dimensions,
standard QK/Add/Softmax/PV dataflow, last-axis Softmax and an immutable finite
positive-zero mask whose broadcast does not expand scores. Nonzero/causal,
NaN, negative-zero, runtime-overridable and shape-expanding masks are rejected;
dynamic or unmatched precision/shape contracts are not silently specialized.
No Cast, scale movement, custom kernel, quantization search or runtime model-name
branch is added. Removal of a source node does not establish a kernel-launch
reduction: TensorRT may already elide/fuse the mask and select identical tactics.
An identical-tactic/no-gain or negative-gain result must be retained.

Graph SHA256 values:

- Baseline: `00832e940a9239a1e7e3fd6c310ec9ac3e550f5a3facdf5f29e2c73713d718fd`.
- Candidate: `41c60ccb67b1e0da9087e08a7bff010bb2b9644a8e050dd8006ad2fd7d7051d2`.
- Source upstream: `2a3f523cbd193d36b9eddbd5a3700b1accc10abecac948c7a5c55f97ddc37032`.

The case binds raw ports/references, original activation, request, recipe and
precision-profile hashes. Effective target precision/format parity is still
unverified; removing the FP32 zero Add cannot be assumed to preserve every
TensorRT intermediate conversion. Same-Orin target MSE < 1e-5 / cosine > 0.9999
must pass before paired timing. CPU FLOAT agreement is diagnostic only.

Reproduction on the existing host environment, using a new output directory:

```bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python \
  python artifacts/orin/20261021-08/prepare_attention_case.py \
  --request artifacts/orin/20261021-08/attention-pair-request-001.json \
  --output artifacts/orin/20261021-08/attention-pair-new
```

`four-operator-cases-regression.xml`: **67 passed**, including 19 Attention,
20 GEMM, six RoPE, seven LayerNorm and 15 generic static-index checks. Attention
tests cover FLOAT/FLOAT16 and either Add operand order, unsupported mask/shape/
precision rejection, exact retained-node precision requests, complete context
recomposition and all three real Q/K/V inputs recomputed from the original
activation. These artifact-only changes do not rerun target models or alter
frozen running RDT engines. The later Embedding preparation below is also
host-only; VLA cases, all target paired timings and meaningful integrated
operator effects remain open.

## Real Language Embedding Pair

Later host-only [native case/request preparation](orin_embedding_rope_native_preparation_20261022.md)
explicitly binds INT32 Gather indices and inlines RoPE payloads. Original
materials below remain unchanged. No target build/quality/timing is added;
the dense/FLOAT-only auditor cannot yet qualify these non-dense/INT32 cases.

Evidence: `artifacts/orin/20261021-08/embedding-pair-001/`. This is the actual
language token table, not a position/shape Gather relabeled as an Embedding.
The original `input_5` indices remain a live runtime input in both variants.
The exact source graph, successful build/engine, IO receipt and formal audit
bind the operator to the adopted SmolVLA profile.

| Property | Prepared Evidence |
|---|---|
| Immutable source table | FLOAT `[49280,960]`, 47308800 values |
| Original source scalar | FLOAT `31.0`, preserved exactly, not recomputed as sqrt(960) |
| Runtime token indices | `[1,48]`; source ONNX INT64, adopted native ABI INT32 |
| Complete output | FLOAT `[1,48,960]`, 46080 values |
| Baseline | Axis-zero Gather followed by original scalar Mul |
| Candidate | Same live Gather into a once-pre-scaled immutable table |
| CPU complete output | Byte-exact with the original source fragment |
| Real input coverage | 16 recorded observations, **one distinct instruction tensor** |
| Full-table transformation | Every source table value recomputes exactly into the candidate table |
| Target build/precision/quality/timing | Not performed |

The generic constant fold requires immutable same-dtype floating weights and a
finite scalar, known matched input/output shapes and finite folded values.
Runtime table/scalar inputs, dtype changes, nonfinite/overflowing folds and
unsupported shapes are rejected. This is offline preparation of all constant
weights, **not precomputed per-prompt features**: arbitrary valid token indices
remain runtime data. It does not change checkpoint semantics through a new
quantization/precision search or run a whole-model FP32 fallback.

All 16 original index payload hashes, shapes and ranges are independently
checked. Actual source tokens range from 2 to 31413 and fit both vocabulary
`0..49279` and INT32. Narrowing/expanding roundtrip checks preserve all values.
The adopted engine's hash-bound IO metadata confirms a linear INT32 input;
the future pair's actual IO is not yet certified. Invalid/negative/out-of-range
indices cannot be padded, truncated or silently narrowed. Multiple distinct
recorded instructions are supported by the preparer, but this pack contains
only one and is not advertised as 16 prompts or all-vocabulary target coverage.

The original Gather has **no explicit compute/output precision request**;
the following Mul has an FP32 compute/output request. The baseline preserves
those declarations. After folding, the candidate adds no Gather compute
request; the terminal FP32 output request moves from Mul to Gather because
the latter now produces the same complete external output. This is explicit
boundary rebinding, not evidence that internal storage/compute precision is
unchanged. Actual target storage/compute parity and same-Orin complete-output
MSE < 1e-5 / cosine > 0.9999 remain required before timing. A failed precision
or fidelity gate is retained, not repaired through an unbounded precision search.

Graph SHA256 values:

- Baseline: `049fcb834714b8ecbb8b67c1f3a08e43101a5d91cc791a45fe3e491f129a1d16`.
- Candidate: `ad1f088871cc9ca419ecded805fcaf357b70036201d6183cdbbd6913988918d9`.

The table bytes intentionally change under the declared scalar fold; original
and transformed value hashes are both retained. The original source graph is
untouched. Host preparation took 5.160 seconds including input/hash checks,
artifact IO and CPU references; this is not target compilation, tuning cost or
inference latency. Target failures/costs/timings remain unknown, not zero.

`embedding-case-regression.xml`: **37 passed**; includes FLOAT/FLOAT16,
INT32/INT64, positive/zero/negative scalar cases, valid boundary/repeated IDs,
invalid fold/index contracts, original physical-ABI evidence, complete real
output and recomputation of all 47308800 transformed weight values.
`five-operator-cases-regression.xml`: **104 passed** with the earlier four
families and static-index protection.

## Independent Build Controller

The frozen `20261020-00/run_trt_build_v2.py` requires both precision maps to be
nonempty. That is unsuitable for the new GEMM profile (no output override) or
the folded Embedding profile (no compute override); adding fake constraints
would change the declared experiment. A separate stdlib-only controller,
`20261021-08/run_trt_build_v3.py`, emits each map independently and rejects
ambiguous/wildcard selectors, invalid types, source mismatches and a completely
empty purported constraint profile. It preserves `--fp16 --noTF32 --buildOnly`
and records exact emitted flags, original declarations, empty maps, source,
DSO-environment probes, monotonic duration, child identity/status and engine hash.
Build success remains `built-not-validated`, never execution/fidelity pass.
Nonzero child exit also propagates as a nonzero controller exit.

The original controller SHA256 remains
`757ad6a5b3ed4b1fecb0356ee4644dd0893b112df46a3a436049e365da1f9b8b`.
The new controller SHA256 is
`e7ee0ee1621407750137fde0599ec35b09c0879fadcb5f117d6d61cdc8a1d680`.
No running RDT engine, controller, plugin or model archive was changed.

Controller regression: **31 passed**, including all ten actual operator
profiles, absent-map preservation, invalid-profile rejection, mocked success/
failure receipts and compatibility with the existing receipt-only build gate.
Mocks are isolated host temporary fixtures, not real TensorRT engines or board
evidence. `operator-and-builder-regression.xml`: **135 passed** combined.
The new controller has not yet been transferred to or executed on Orin.

Example for a genuinely idle target window, after transferring/hash-checking
the pair and the new controller (not a command already executed):

```bash
python3 run_trt_build_v3.py \
  --onnx embedding-pair-001/candidate.onnx \
  --expected-sha256 ad1f088871cc9ca419ecded805fcaf357b70036201d6183cdbbd6913988918d9 \
  --precision-profile embedding-pair-001/candidate-precision.json \
  --output embedding-candidate-build-new
```

Both variants still need real matched target builds/precision and full-output
checks before paired timing. This host preparation does not close D4 or take
priority over completion of RDT and the remaining core VLA models.

## Real VLA State Feature Pair

Evidence: `artifacts/orin/20261021-08/state-feature-pair-001/`. The original
state-token boundary includes state packing and the checkpoint projection,
not only an isolated toy Scatter or an action-head/E2E substitute. This is a
VLA-specific workload declaration using generic graph lowering, not a
model-name branch in the runtime. No measured hotspot ranking is claimed.

| Property | Prepared Evidence |
|---|---|
| Input | Real prepared FLOAT state `[1,6]` in the original model ABI |
| Packing | Original zero background `[1,32]`, immutable contiguous INT64 indices |
| Projection | Original Gemm weight/bias, transpose/alpha/beta unchanged |
| Complete output | State token `[1,1,960]` FLOAT |
| Coverage | 16 distinct recorded states, 15360 complete token values |
| Baseline | Expand, Reshape, ScatterND, Gemm, Unsqueeze |
| Candidate | Same arithmetic with static Scatter replaced by Slice/Concat layout |
| Source nodes | Five versus six, not actual CUDA-launch counts |
| Dense request | Original `/Gemm` FP16; no output override added |
| CPU complete output | Byte-exact source/candidate for every real state |
| New target build/precision/quality/timing | Not performed |

Input payloads are rehashed against the original preparation. Additional
verification follows the independently audited formal baseline hash through
the original native pilot, native-build receipt and preparation SHA256 before
checking every input. Exact original node/constant/build/engine binding is
preserved. All non-Scatter definitions and both projection weight/bias hashes
are unchanged; the new candidate is not an alternative checkpoint.

An independent CPU calculation recomposes zero padding and the original affine
projection for every state. Its worst source-relative MSE is
`1.9946735958555523e-15`, max-abs `2.384185791015625e-7`. Source/candidate
outputs are byte-exact, while this separately evaluated NumPy arithmetic is
within the existing CPU primitive gate rather than advertised as byte-exact.
These are newly generated CPU source-fragment references, not actual mixed-
precision Orin activations or complete policy actions.

The candidate reuses the already value-independent, literal-index layout pass;
no new public lowering/backend/runtime code is changed. Canonical baseline and
candidate graphs embed all weights, so each graph hash binds its complete
canonical payload. Intermediate lowering graphs/sidecars and their hashes are
also retained for provenance. Source graph SHA256 values:

- Baseline: `810d454c1ebeeed35df56736e3f81452cc340af4bca7ed6a6fea89e92e13840f`.
- Candidate: `c19db36d8c442306edd33719b19b8db429c8a3c3627436ff7ca2186087b94180`.

Host preparation took 2.892 seconds including source/hash checks, artifact IO
and references. This is not target compilation, tuning cost or inference
latency. Actual target IO/layout/precision, complete-output gate and paired
timings remain open; more source nodes can still compile to the same or fewer
kernels, and no benefit is presumed. Any useful candidate must additionally
be integrated and revalidated inside a complete real VLA.

`state-feature-and-builder-regression.xml`: **39 passed** (six real state-case
checks plus 33 controller tests including all twelve current precision
profiles). `six-operator-and-builder-regression.xml`: **143 passed** combined
across all six families, the generic static-index protection and controller.
Earlier XMLs retain their original checkpoints and are not overwritten.
