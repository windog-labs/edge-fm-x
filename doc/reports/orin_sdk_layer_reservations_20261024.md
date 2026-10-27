# Orin Native SDK Layer Diagnostics

Verified: 2026-10-24 00:57 UTC. **The ordinary TensorRT IProfiler path works
under the existing account without requesting the denied CUPTI APIs.** This
is a frozen SmolVLA prefix-stage diagnostic, not a new complete-model E2E
experiment, primary quality gate, physical-memory peak or fifth-model result.
The existing Norm full-model negative and all four accepted main-table rows
remain unchanged.

## Scope And Complete Outputs

Reuse the original primitive-expression prefix engine and its already measured
50Norm-plugin counterpart, with unchanged engine/build hashes and precision
declarations. No new engine build, precision search, source formula or dtype
conversion is introduced. Seven actual materialized inputs from recorded
observation0match the frozen native preparation hashes. Previously checked
I64-to-I32tokens are reused unchanged, min2/max31413. The producer stage
actually runs on Orin to provide the mask/sine/cosine tensors; no host feature
or reference-only substitute is supplied to the neural prefix.

The generic probe derives from the frozen typed SDK engine probe with an
exact whole-file inverse. Its changes add standard plugin initialization,
IProfiler records, context-requirement queries and complete same-engine output
checks. It dispatches by declared engine inputs/routes, not model-name kernels.
The reference direction remains primitive expression ->50plugins, never
silently reversed to turn the old negative ablation into a positive result.

Each neural engine runs once unprofiled, then3warmup+16profiled calls. During
all19profiled/warmup calls every one of33complete prefix outputs is compared
bytewise to that engine's own unprofiled reference:627comparisons/variant,
1254total. Runtime per-call check records and the exact executed source are
retained. Independent host re-reading checks66saved complete prefix output
pairs/132tensors, finite FLOAT and canonical BOOL values. The intervening
per-call output bytes are not all archived, so this is **not**1254independent
raw-byte re-reads. None of these outputs is a complete action chunk.

## Reported SDK Layers

| Quantity | Primitive Expression |50Norm Plugins |
|---|---:|---:|
| SDK-reported layers |46|646|
| Measured layer records |736|10336|
| Sum of per-layer reported mean times (ms) |61.999058|69.608742|
| Reformat layers |37|205|
| Sum of Reformat mean times (ms) |0.363006|10.075274|
| Myelin groups |1|13|
| Sum of Myelin mean times (ms) |60.186780|42.536360|
| Norm PluginV2 layers |0|50|
| Sum of Norm-plugin mean times (ms) |N/A|2.090496|

All16measured iterations retain every reported layer and its original name/
index; zero-time Constant/NoOp records are kept. SDK layer counts are **not**
CUDA kernel-launch counts. Layer times can represent fused work and are not
added to stages or substituted for E2E latency. In particular, the opaque
baseline Myelin group cannot be decomposed into exact source-operator times
or hidden compute/weight-storage dtypes using IProfiler.

The larger Reformat contribution and changed reported fusion layout are
consistent with the earlier integrated Norm slowdown, but do not establish
complete causal attribution. No new whole-model speedup is claimed. Standalone
4.329xNorm micro speedup remains separate from the preserved6.4287%whole-model
slowdown. [Original complete E2E ablation](orin_operator_layernorm_raw_e2e_20261022.md).

## Separate Memory Quantities

| Metric | Primitive Expression |50Norm Plugins |
|---|---:|---:|
| Serialized engine file bytes |610531988|657852105|
| SDK context device-memory requirement bytes |438248960|155889152|
| Same requirement (MiB) |417.946777|148.667480|
| Diagnostic probe's unique IO requested bytes |14668652|14668652|

`ICudaEngine::getDeviceMemorySize()`is an SDK context requirement, not a
measured physical peak. Serialized plan size is not resident GPU-weight size.
Probe IO buffers are separate explicit allocations, not the optimized Session
arena. These quantities must not be summed into total Jetson memory or added
to RSS; CPU/GPU physical memory is shared. Unknown private allocations, caches
and physical peaks remain unknown, not0. The queried context requirement is
identical with/without profiling for each same engine.

## Device Warning And Closure

The SDK emits its device-model compatibility recommendation in allfive engine
calls. These stderr records are retained. A separate installed `trtexec`
load-only run with `--buildOnly` explicitly skips inference and reports Orin,
SM8.7/16SMs,1.3GHzcompute/memory clocks,62800MiBglobal memory,164KiBshared
memory/SM and256bitbus. The two original build logs agree at this printed
resolution, and build/current board serials match1421524312594. This does
not prove every device property equal or establish the warning's underlying
cause. No warning is suppressed, no SDK/clock setting changed and no engine
rebuilt to erase it. These remain loaded-engine diagnostic observations.

Both owned units finish success/inactive/MainPID0and empty post cgroups.
Independent closure rehashes217target paths and11tracked controller/compiler/
probe/metadata/post birth identities terminal. Java942765/942805and original
Supervisor parent/birth/argv/cwd remain unchanged. No application stop, root
diagnostic, CUPTI subscription or driver-permission change occurs. The original
8GiBsmall-stage admission/2GiBruntime floor stay in force, not replacing
CogACT24/4GiBgates. Fifteen actual/source/corruption/figure/export tests pass.

The first figure is retained after visual review found legend/tick overlap;
the accepted second render uses distinct grid tracks for plots, legend and
scope notes. Its PNG/PDF and source hashes are pinned, pixels are nonblank and
visual review is complete. No diagnostic numeric value changes between renders.

## Evidence

- [Independent output/profile/scope and target closure](../../artifacts/orin/20261024-00/audit-001.json).
- [Frozen stage request and real-input routes](../../artifacts/orin/20261024-00/plan-001.json).
- [Exact source derivation](../../artifacts/orin/20261024-00/source-derivation-001.json).
- [692reported-layer means CSV](../../artifacts/orin/20261024-00/tables-001/sdk-layer-means.csv).
- [Non-additive memory roles CSV](../../artifacts/orin/20261024-00/tables-001/separate-memory-quantities.csv).
- [Accepted figure/source receipt](../../artifacts/orin/20261024-00/figures-002/receipt.json).
- [PNG figure](../../artifacts/orin/20261024-00/figures-002/sdk-stage-diagnostic.png).
- [Fifteen checks](../../artifacts/orin/20261024-00/diagnostic-tests-002.xml).

The installed methods match NVIDIA's versioned
[IProfiler attachment/reporting example](https://raw.githubusercontent.com/NVIDIA/TensorRT/release/8.5/samples/common/sampleInference.cpp).
The local source, compiler/SDK identities and actual artifacts are authoritative
for this run; external documentation is not platform execution evidence.
No change is made to the paper PDF. Overall formal model coverage remains4/5;
CogACT, unresolved operator eligibility and private-memory/alternate-mode gates
remain open. Do not rerun completed one-shot units or promote stage times into
the main table.
