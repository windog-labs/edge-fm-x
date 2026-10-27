# Orin Delivery Reconciliation

Five required VLA formal D1-D3cohorts are accepted. This update consolidates
existing evidence only: no new target inference, precision search, application
control or global runtime change. The full Goal remains active: final
requirement-by-requirement acceptance and optional Qwen assessment are pending.

## Five-Model Export

The new export rechecks all69120requests/61440measured rows across12configurations
and60independent processes, and rehashes18original CDF assets against their
producing receipts. The first four models' statistics are exactly equal to the
sealed112export; that old export is not overwritten. CogACT adds two fully
accepted rows, not its older pilots/failed partial attempts.

| Model | Conventional Mean ms | Optimized Mean ms | Runtime Ratio | Optimized Mode |
|---|---:|---:|---:|---|
| SmolVLA |134.409630|113.139929|1.187995x|full-loop replay|
| RDT-1B |3065.625218|2975.587343|1.030259x|full-loop replay|
| pi0 |1129.869939|1125.379221|1.003990x|arena only, no replay|
| pi0.5 |1258.792712|1256.955605|1.001462x|arena only, no replay|
| CogACT actual~7.63B public-dependency configuration |694.393388|693.790506|1.000869x|arena only, no replay|

Tiny differences, negative paired rounds, worse tails and failure to meet the
draft jitter target remain. These are same-Orin, matched-profile ratios, not
H20/PyTorch/BPU/robot-task or significant-gain claims. Finite16observation cycles
are not independent trajectories. Native/H20 errors stay supplemental.

Public output count is a manifest contract rather than a model-name branch.
The first four models have two numeric outputs. CogACT has three numeric
outputs plus exact-only U8RNG/I64count ports:57600total tensors including
warmup,34560numeric and23040exact-only auxiliary tensors. Normalized/native
quality is distinct from opaque bytes, which have no cosine/MSE. Full raw
tensor quality is inherited from the pinned independent full-output audits;
this export does not claim a second read of every raw tensor.

- [Main CSV](../../artifacts/orin/20261024-07/tables-001/main-table.csv).
- [Quality CSV](../../artifacts/orin/20261024-07/tables-001/quality-table.csv).
- [Sixty-process CSV](../../artifacts/orin/20261024-07/tables-001/process-table.csv).
- [Full p50/p95/p99/tail table](../../artifacts/orin/20261024-07/tables-001/tables.md).
- [Input/source/export pins](../../artifacts/orin/20261024-07/tables-001/summary.json).
- [Twenty checks](../../artifacts/orin/20261024-07/table-tests-001.xml).

## Component Integration

Two small immutable target contract files were copied and checked against the
actual accepted216/230file closure pins. Every declared resource SHA/size is
present in its accepted full-model closure. Three checked-cast DSOs match their
native CUDA11.4build registrations, schema, source dtype pairs and actual
workspaces384/448/1056bytes. All require synchronization; none claims external
capture or shared execution-context support. Public F64state is not a fake
FLOAT carrier. Source/descriptor/range/poison contracts and the earlier native
functional smoke remain separate from full-model acceptance.

Both accepted step engines match their actual build source graph SHA and
successfully imported`VLAForgeScalarSinCosF64`. The actual source node's signed
INT32word payload reconstructs512finite F64frequencies, with SHA equal to the
two original ten-step source-shadow audits. Existing real scalar numerical/
two-stream capture smoke remains a component check; complete model acceptance
comes from the later independent joint receipts, not promotion of that smoke.
The original128chunk/97772assertion native smoke lacks initial target PID/start
and retains that limitation. No full OpenPI pipeline replay, arbitrary new
shape or CUDA hardware-fault certification follows.

[Component linkage audit](../../artifacts/orin/20261024-07/component-reconciliation-001.json).
The older preparation reports remain historical, not current unbound status.

## Memory Envelopes

All five accepted cohorts'60whole-worker resource envelopes are now indexed,
covering78667telemetry samples. RDT covers all15workers, not the older14worker
checkpoint. Missing private allocation/physical GPU peak fields are null, not0.
RSS/high-water/shared-system availability are not added to each other or to
weights, KV, context requirements or CUDA requested bytes.

| Model | Mode | Maximum Sampled Process VmHWM MiB |
|---|---|---:|
| SmolVLA |ordinary / shared / replay|3111.270 /3114.137 /3113.816|
| RDT-1B |ordinary / shared / replay|6811.164 /6810.781 /6811.098|
| pi0 |ordinary / arena|4161.227 /4161.277|
| pi0.5 |ordinary / arena|3228.504 /3228.793|
| CogACT actual~7.63B |ordinary / arena|4352.203 /4351.051|

These are sampled observations of each process's OS high-water value, including
initialization/warmup, not isolated GPU physical peaks or per-request memory.
No temporal causal attribution or general leak freedom is inferred. Older
SmolVLA35minute single-worker soak and covered CUDART allocation/lifecycle
diagnostics remain separate. They do not prove private driver/SDK coverage;
the actual CUPTI access refusals remain. Blank GPU-frequency/guard fields in
the two older flat summaries are not reconstructed as zeros or copied from
another window. Original raw telemetry/temperature/frequency files stay intact.

[Per-process resources](../../artifacts/orin/20261024-07/resources-001/per-process-resources.csv),
[per-mode resources](../../artifacts/orin/20261024-07/resources-001/per-mode-resources.csv),
[source relationships and limitations](../../artifacts/orin/20261024-07/resources-001/summary.json).

## Operator Budgets And Costs

Six real source signature contracts and twelve accepted engine build receipts
are indexed together. Each family uses one declared fixed graph recipe, not
autonomous kernel synthesis or a broad search. The selected build-controller
elapsed sum is116.830536192seconds; it is not inference time or total historical
campaign/search cost. Unrecorded Agent reasoning time is null, not0. One
initial host lookup used a missing original-path metadata copy; the actual
preserved retrieval copies were then matched by engine SHA to the sealed plans.
No target build was retried or raw result changed for that host-path correction.

Old four-family vendor-event/1000mswarmup and new two-family native CUDA-event/
128callwarmup contracts remain distinct. Attention retains all20800accepted
retry rows; its earlier20479failure is not rounded up. Across the six rows,
all123200operator measured records are retained. State configured-policy-only,
Embedding negative pairs, RoPE worse resident max and old104/110precision
limits remain. This table does not rank cross-family absolute latency or infer
hidden accumulation/storage precision.

The representative full-model Norm integration remains complete and negative:
all10240measured/23040full tensors, normalized primary gate passes, native error
supplemental,125.900306->133.994095ms (6.4287%slower). Positive gain is not an
added completion gate. GEMM/Attention full-action failures and the later new
Norm short-count non-adoption remain failures, not accepted integrated gains.

[Six contracts, budgets and build provenance](../../artifacts/orin/20261024-07/operator-costs-001/summary.json),
[build-cost CSV](../../artifacts/orin/20261024-07/operator-costs-001/build-costs.csv),
[actual integration reconciliation](../../artifacts/orin/20261024-02/integration-reconciliation-001.json).

## Verification And Remaining Work

Publication follow-up, 2026-10-24: the [core acceptance review](orin_core_acceptance_20261024.md)
and [post-publication audit](../../artifacts/orin/20261024-08/core-requirement-audit-002.json)
now provide the explicit requirement review, same-request timing boundaries,
memory limits and updated claim-gap table. These close the core publication
within its scope, not private GPU attribution or optional Qwen assessment.

The host-only combined regression covers real accepted data, unchanged original
statistics, shortened/wrong-boundary CSV rejection, typed port omission,
invented integer metrics, resource unknowns and changed cast contracts. It
does not create new model observations. Rechecks need fresh output names;
completed one-shot target launchers are not rerun.

```bash
python3 -B -m pytest artifacts/orin/20261024-07/test_five_tables.py \
  artifacts/orin/20261024-07/test_components.py \
  artifacts/orin/20261024-07/test_resource_envelopes.py \
  artifacts/orin/20261024-07/test_operator_costs.py -q
```

At this report's original checkpoint, final acceptance still needed the
requirement-by-requirement review. That review is now linked above; optional
Qwen full vision/prefill/decode assessment has not run. No PDF is modified and
no full Goal completion is claimed here.
