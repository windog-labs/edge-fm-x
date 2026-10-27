# Orin LayerNorm Operator Pair

Status: new target engines, complete-output pair quality and both five-pair
operator timing protocols are independently checked. This is one real normalization signature,
not complete-model E2E or an all-layer/kernel-synthesis claim. The interrupted
RDT formal campaign is not restarted or accepted by this work.

## Contract And Admission

The source is `20261021-08/layernorm-pair-001/case.json`, already bound to the
adopted SmolVLA prefix and observation 6. Original input/output FLOAT shape
`[1,1024,768]`, gamma/beta constants and epsilon remain unchanged. All 786432
output values are checked. This is the original necessary FP32 normalization
island; the full VLA's FP16-dense contract is not replaced by FP32.

Each target operation is admitted only with more than 8 GiB MemAvailable,
matching Orin serial, default Torch absent and no competing visible device
users except desktop processes. Recorded headroom is about 10 GiB. Device-FD
coverage is still current-user limited, not exclusive-machine attestation.
Java/platform services and power/clock settings are unchanged.

Only one tiny engine is built or probed at a time. Both builds use identical
`--fp16 --noTF32`, explicit FP32 layer/output constraints and workspace 256 MiB.
That workspace limit is **not total process memory**: recorded builder max RSS
is 2248316/2198244 KiB for baseline/candidate. Root:
`artifacts/orin/20261022-00/operator-layernorm-pair-001/`.

## Actual Builds

| Variant | Source | Build Controller Seconds | Engine Bytes | Engine SHA256 |
|---|---|---:|---:|---|
| Baseline | Nine original primitives | 16.055 | 37646 | `c486b811460a8e0e3a5ff996d3a545832e99d21fef3ab11b1523c1fb62626591` |
| Candidate | Installed standard LayerNorm plugin | 3.639 | 11076 | `a3f8c6f73d12db38c87b40a4fb0892792b64dbe36278cf987fbc060d9db0276a` |

Both child exit codes are zero; records remain `built-not-validated` until the
separate quality audit. Layer inspector metadata shows linear device FP32 on
all observed dataflow ports. Baseline includes two Reduce, two ElementWise,
one PointWiseV2, two Constant and one NoOp records; candidate has one PluginV2
and two Constant records. These are inspector layers, **not CUDA launch counts**.
Opaque vendor kernel internals are not independently instrumented.

## Complete Output Check

The generic native indexed-IO C++ probe uses the installed plugin registry,
same real input bytes and both newly built engines. The generic capture tool
checks admission, identities/hashes and actual complete output files. Its v2
adds owned-child cleanup on timeout; both canonical captures use the same v2
and native probe. Earlier v1 successful capture/source are retained separately.
Six host capture/admission/timeout tests pass; mocks are not board experiments.

Same-Orin candidate versus baseline:

| Checked Values | MSE | RMSE | Max Abs | Cosine | Byte Exact |
|---:|---:|---:|---:|---:|---|
| 786432 | 2.2909282567946777e-15 | 4.7863642326871427e-8 | 1.9073486328125e-6 | 0.9999999999999932 | no |

This passes the frozen pair gate MSE < 1e-5 / cosine > 0.9999. It is tolerance-
bounded equality, not mathematical losslessness. CPU source comparisons are
diagnostic only, not an extra FP32-reference gate. Evidence:
`same-target-quality-001/quality.json`, both `*-quality-002/capture.json`, full
raw output tensors, source/build/engine/probe hashes and before/after snapshots.

## Timing Gates

The frozen plan was executed completely: two boundaries, five independent
process pairs each, 1024 exported measured records per process/variant, twenty
native timing processes and **20480 retained records**. Order is AB/BA/AB/BA/AB;
every process uses 1000 ms warmup, one stream, no CUDA graph/spin-wait/threads,
same actual real input file and unchanged engine hashes. All native PIDs/start
ticks are distinct. Minimum admitted system memory across runs is 10393452 KiB.
The collection took 149.764 seconds including setup/environment/output logging;
that controller duration is not inference latency.

| Declared Boundary / Metric | Original Mean (us) | Candidate Mean (us) | Ratio of Pooled Means |
|---|---:|---:|---:|
| Resident, vendor GPU compute | 186.408 | 43.062 | 4.329x |
| IO enabled, vendor GPU compute | 204.564 | 40.103 | 5.101x |
| IO enabled, H2D + compute + D2H event latency | 489.500 | 297.729 | 1.644x |

The primary predeclared compute boundary is the resident row, not whichever
row yields the largest ratio. Resident uses the shipped NVIDIA sample's managed
IO selection on integrated devices, performs initial input staging and final
output fetch outside timed per-query DMA, and keeps every vendor-exported
post-warmup record. Source hashes are in `frozen-timing-plan-001.json`; the
installed `sampleInference.cpp`/`sampleReporting.cpp` confirm these boundaries
and export warmup exclusion. No custom hand-written timer replaces the vendor
benchmark implementation.

IO event latency excludes CPU enqueue/preprocessing and is **not host-synchronized
or raw-observation VLA E2E**. Absolute exported timestamps are rounded by the
vendor JSON writer, so no microsecond CPU latency is reconstructed by subtracting
those rounded endpoints. Complete last-output JSON values are also decimal text;
their difference from the same-engine raw tensor is checked within tolerance,
never called byte-exact. All twenty runs' full output values pass that check.

| Metric | Original (us) | Candidate (us) |
|---|---:|---:|
| Resident compute p99 | 196.899 | 59.376 |
| Resident compute max | 217.041 | 83.008 |
| Resident compute std | 3.921 | 6.133 |
| IO event latency p99 | 527.083 | 318.581 |
| IO event latency max | 2051.210 | 2127.320 |

The higher candidate resident std and higher IO maximum are retained. Lower
mean does not prove uniform jitter improvement. Full-range ECDF PNG/PDF and all
raw CSV/JSON preserve these tails, with no trimming, mean-centering or hidden
x-axis clipping. Resident per-pair mean ratios range from 4.213x to 4.525x;
the independent experimental units are five process pairs, not 5120 unique
robot trajectories or independently sampled activations.

Primary evidence: `paired-timing-audit-001/timing-audit.json`,
`all-operator-records.csv`, `cdf/layernorm-operator-cdf.{png,pdf}`, frozen plan,
twenty captures/exports and original target engines. Twenty host regression
checks cover admission/owned-child timeout cleanup, vendor decimal boundaries,
bad traces, all-row retention, actual paired data and unfavorable tails.
Existing platform services and limited owner visibility remain documented;
there is no dedicated/exclusive-machine or continuously locked-clock claim.

## Remaining Gates

The whole-prefix source-pair audit now reproduces the adopted candidate ONNX
SHA exactly by re-running the generic lowering on the frozen original graph.
It replaces 450 FLOAT primitive nodes with 50 FLOAT Norm plugins; all remaining
nodes, weights and precision/output declarations are unchanged. Both builds use
the same builder, FP16/noTF32 and 2048 MiB workspace. Their 43 binding names/order
and all reported physical descriptors match. Both inspectors omit descriptors
for input_1, input_3 and input_6; full target ABI validation is still required.
The first auditor correctly exposed that coverage gap; its source/failure note
is retained, and revision 2 explicitly reports the missing descriptors rather
than inventing them. Eight auditor regression checks pass.

Evidence: `artifacts/orin/20261022-00/norm-integration-source-audit-002.json`.
Inspector layer counts (46 versus 646) are not kernel counts or a performance
prediction. Both whole-model variants will use one common seven-engine IR/Plan
and all ten solver iterations, replacing only the prefix engine. Complete
action quality and raw-observation E2E are not yet revalidated.

Update: both common-IR seven-engine variants subsequently completed target
full-action pilots. Normalized actions pass the fixed gate, while native-unit
MSE does not pass the same numeric threshold. All errors and the conservative
failure are retained; no integrated speedup is published. See the independent
[whole-model integration report](orin_operator_layernorm_integration_20261022.md).

The standalone timing protocols above are complete. Their GPU compute and
H2D/compute/D2H boundaries remain distinct from raw-observation VLA E2E. No build
controller or correctness-capture walltime is an inference benchmark. Preserve
all records and fixed timing boundaries when extending observation coverage.

Effective full-VLA integration and revalidated complete outputs/E2E remain
required. Other five prepared families
still need target paired build/quality/timing. No speedup is inferred from
source-node, inspector-layer or engine-byte reductions.
