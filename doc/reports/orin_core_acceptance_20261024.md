# Orin Core Acceptance And Reproduction

Retention update,2026-10-24: the user released completed model/engine payloads
to recover disk space. Accepted raw measurements/quality/CDF/source identities
and documents remain; historical payload closures do not imply the binaries
are still present. [Scope and post-cleanup data verification](orin_storage_retention_20261024.md).
The commands below reanalyze retained data. Fresh inference or complete payload
closure checks require rebuilding/retrieving the retired models first.

Date: 2026-10-24. This closes publication of the required five-VLA measurement
package, with explicit limits below. It does not accept every manuscript claim
or complete the active Goal: optional Qwen support assessment remains pending.
The [actual active objective](../../artifacts/orin/20261024-08/active-goal-objective.md)
is authoritative; the old 20261020 Torch-oriented plan is superseded.

## Accepted Evidence

- Five full native Orin chains, 12 configurations, 60 independent processes,
  69120 raw requests and 61440 measured requests; each configuration has five
  processes with 128 warmup and 1024 measured calls. All tails are retained.
- Complete finite normalized/native outputs remain bound to the original
  independent byte audits. Same-Orin normalized MSE0/cosine1; native and H20
  comparisons are separate. CogACT auxiliary U8/I64 outputs are exact-only,
  not assigned numeric cosine. Sixteen recorded observations cycle, not
  independent robot trajectories or proof of task success.
- Six real operator signature families and twelve selected build receipts;
  the representative full Norm integration passes the fixed normalized gate
  but is 6.4287% slower. Negative results are completed experiments, not gains.
- Representative ordinary/shared/replay, lifecycle, covered allocation,
  more than 30 minutes continuous model work and matched power observations;
  private allocation coverage and second-mode performance remain unmeasured.

[Main/quality/percentile tables](../../artifacts/orin/20261024-07/tables-001/tables.md),
[operator results](orin_operator_ablation_results_20261022.md),
[integration](orin_integration_requirement_reconciliation_20261024.md),
[delivery index](edgefm_orin_experiment_delivery.md),
[current claim-gap table](orin_claim_gap_20261021.md).

## Requirement Review

The [post-publication audit](../../artifacts/orin/20261024-08/core-requirement-audit-002.json)
contains all 31 original numbered requirements, exact objective/source hashes,
evidence paths and review notes. Its 29 core verdicts are scope-bounded, not
claims of private-driver visibility or stronger paper assertions. It rechecks
archived CSV/source/identity relationships and accepted measurements; it does
not reread every historical action file or run new inference. Earlier audit001
is preserved as a pre-publication snapshot, not a receipt for these new docs.
The later [data-only review003](../../artifacts/orin/20261024-08/core-requirement-audit-003.json)
records updated retention notices. Old receipts/verification hashes remain
historical; they are not silently regenerated as if no cleanup had occurred.

| Goal Item | Acceptance And Scope | Evidence |
|---|---|---|
| route.1 | Target native/TRT only; host exports separate, no target Torch baseline | Input manifest, target observation |
| route.2 | Actual target TRT builds and complete native Sessions | Five accepted full-model audits |
| route.3 | Old smokes/cancelled Torch build retained, not model acceptance | Delivery historical sections, original audits |
| route.4 | Same-Orin deployment-profile comparison; H20 supplemental | Main/quality tables, claim-gap |
| route.5 | Frozen SDK, original-manager restoration, no new privileged changes | Cohort restoration, target observation, power permission |
| D0.1 | PDF claims mapped to measurements and unsupported boundaries | Claim-gap table, cross-platform scope report |
| D0.2 | All five models; CogACT actual~7.63B public-dependency identity | Five-model input manifest |
| D0.3 | Frozen source, shape, horizon, solver/profile and actual noise | Original cohort contracts and source assessment |
| D0.4 | Actual build/failure/source records; build-only not acceptance | Source assessment, component reconciliation, reports |
| D1.1 | Five separate accepted cohorts, target compute serialized | Original full-model audits |
| D1.2 | Fresh online vision/language and every solver iteration | Accepted native-chain data, raw timing |
| D1.3 | Declared mixed/high-precision islands, no whole-model FP32 fallback | Frozen precision profiles, operator contracts |
| D1.4 | Checked casts, source/plugin lineage, complete outputs | Integrated component reconciliation |
| D1.5 | Adapter/typed IR/Plan/native Session; semantic declarations | Component/source contracts, public runtime implementation |
| D2.1 | Same weights/observation/noise/profile; source errors separate | Input manifest, primary quality and H20 reports |
| D2.2 | Complete finite typed outputs, hashes, dimensions and errors | Original full-byte audits and quality table |
| D2.3 | Normalized MSE<1e-5/cosine>0.9999 unchanged; failures retained | Original quality policies and integration audit |
| D3.1 | Runtime resource identity and graph-engine differences explicit | Main tables and integration report |
| D3.2 | Decoded CPU input to complete CPU chunk with GPU completion | Raw clocks/source checks, complete-chain audits |
| D3.3 | Three recorded same-request boundaries, not independent reruns | Boundary summary and per-request CSV |
| D3.4 | Five processes/configuration, 128+1024, no merged partials | All 69120 original CSV rows and launch contracts |
| D3.5 | Full statistics/raw/CDF and finite16-observation disclosure | Main export, all18original CDF pins |
| D4.1 | Six real paired signatures with numerical/method limits | Six-family contracts/cost index and pair audits |
| D4.2 | Fixed recipe/candidate budgets and real negative integration | Cost index, original failed candidates, Norm integration |
| D4.3 | Runtime/reset/isolation/lifecycle, process peaks, covered steady APIs | Allocation audit, resource envelopes, limits below |
| D4.4 | Resource time series, continuous measured work>1800s, matched power | Soak, all60process envelopes, matched-power audit |
| D4.5 | Actual alternate-mode permission assessment after main cohorts | MAXN query/direct-permission evidence; no switch/result |
| D5.1 | Optional pending: complete Qwen vision/prefill/decode/state assessment | No new Orin Qwen acceptance |
| D5.2 | Optional pending: TTFT/16-token/quality/resource benchmark | Depends on supported complete chain |
| D5.3 | Named index/tables/CDF/resource/claim-gap/reproduction published | This report, delivery index and post-publication audit |
| D5.4 | Original evidence/failures/PDF preserved; target observation recorded | Original closures and fresh read-only observation |

Complete numeric byte equality implies zero per-dimension error and tied zero
worst samples for these same-Orin comparisons. This is a consequence of the
existing full-byte audits, not a new raw-tensor reread or source-BF16 proof.
Original nonzero source/native integration errors remain in their reports.

## Timing Boundaries

All 69120 original requests, including warmup, satisfy the same-request
timestamp partition. The 61440 measured requests yield 36 distributions over
12 configurations; counting three correlated intervals does not triple the
number of observations. All 5120 samples/configuration remain intact.

1. `decoded_raw`: original decoded-input to consumable full CPU chunk E2E.
2. `prepared_host_subinterval`: `end - prepared_end` in that same call;
   includes H2D, binding, Session and D2H, not an isolated precomputed-input run.
3. `resident_IO_Session_subinterval`: `model_end - bound_end`, after actual
   completion. Includes internal native finishing/copies where applicable;
   it is not pure GPU-kernel time or the sum of stage means.

| Optimized Configuration | Raw Mean ms | Host-Ready Mean ms | Resident-IO Session Mean ms |
|---|---:|---:|---:|
| SmolVLA complete-loop replay | 113.139929 | 111.726282 | 110.029825 |
| RDT complete-loop replay | 2975.587343 | 2928.063389 | 2925.875908 |
| pi0 verified Arena, no replay | 1125.379221 | 1048.048160 | 1047.321348 |
| pi0.5 verified Arena, no replay | 1256.955605 | 1179.477155 | 1178.742875 |
| CogACT actual~7.63B Arena, no replay | 693.790506 | 634.245952 | 633.621366 |

[All modes/percentiles](../../artifacts/orin/20261024-08/boundaries-001/boundary-statistics.csv),
[per-request intervals](../../artifacts/orin/20261024-08/boundaries-001/same-request-intervals.csv),
[actual clock sources and scope](../../artifacts/orin/20261024-08/boundaries-001/summary.json).
The original E2E table and its CDFs are unchanged.

## Memory And Power Limits

All 60 formal processes / 78667 telemetry records have resource envelopes.
The following are maxima of sampled OS process `VmHWM` observations across
the accepted configurations, including initialization/warmup. They are not
isolated GPU physical peaks, per-request allocations or model-only memory.

| Model | Largest Sampled Process VmHWM MiB | Private SDK / GPU Physical Peak |
|---|---:|---|
| SmolVLA | 3114.136719 | Unknown |
| RDT-1B | 6811.164062 | Unknown |
| pi0 | 4161.277344 | Unknown |
| pi0.5 | 3228.792969 | Unknown |
| CogACT actual~7.63B | 4352.203125 | Unknown |

Full [per-process](../../artifacts/orin/20261024-07/resources-001/per-process-resources.csv)
and [per-mode](../../artifacts/orin/20261024-07/resources-001/per-mode-resources.csv)
records retain RSS, system availability, valid temperature and unknown fields.
No CPU/GPU shared RAM, engine weight size, context requirement or requested
CUDA bytes are added to infer a physical peak. Missing sensors are not zero.

The separate 35.016-minute worker includes 2022.466140544 seconds measured
model work, 18432 measured requests and 37120 complete tensors exact. Covered
CUDART diagnostics show 217 acquisitions / 217 releases and zero live device
bytes after process cleanup, plus balanced graph resources over 12 paired
cycles / 24 Sessions. In 1024 steady ModelRuns all covered counters are zero.
Two shared stream leases intentionally live to process exit. None of this
certifies all CPU heap/private SDK/driver allocation, universal leak freedom
or a global zero-malloc/zero-copy pipeline. CUPTI Activity and Callback
permission failures remain actual error35, not empty successful traces.

Matched ordinary/shared/replay power uses 2423 raw / 1854 eligible samples,
38.438594/38.552769/42.267198W respectively; replay power is higher. These are
client-observed input-rail windows, not isolated model energy/request. Existing
MAXN/ID0 is not60W. The post-five-model direct-permission assessment records
no validated switching route for UID1000 and EMC EACCES13; no second-mode
performance was measured. These documented limits satisfy the permitted
assessment outcome, not the stronger manuscript claims.

## Reproduction

Host-only rechecks below use the existing Python environment (`numpy`,
`pytest`). Run from the workspace root. Use a new, nonexistent output path
each time; completed receipts are immutable and one-shot target launchers
must not be rerun as host audits. No target GPU, Torch, service/power operation
or new reference output is required by these commands.

```bash
python3 -B artifacts/orin/20261024-07/export_five_tables.py export --output /tmp/orin-five-tables-recheck-NEW
python3 -B artifacts/orin/20261024-07/audit_components.py --output /tmp/orin-components-recheck-NEW.json
python3 -B artifacts/orin/20261024-07/export_resources.py --output /tmp/orin-resources-recheck-NEW
python3 -B artifacts/orin/20261024-07/export_operator_costs.py --output /tmp/orin-costs-recheck-NEW
python3 -B artifacts/orin/20261024-08/export_timing_boundaries.py --output /tmp/orin-boundaries-recheck-NEW
python3 -B artifacts/orin/20261024-08/audit_core_requirements.py --output /tmp/orin-core-recheck-NEW.json
python3 -B -m pytest artifacts/orin/20261024-07/test_five_tables.py artifacts/orin/20261024-07/test_components.py artifacts/orin/20261024-07/test_resource_envelopes.py artifacts/orin/20261024-07/test_operator_costs.py artifacts/orin/20261024-08/test_timing_boundaries.py artifacts/orin/20261024-08/test_core_publication.py -q
python3 -B artifacts/orin/20261024-08/verify_core_publication.py --output /tmp/orin-publication-recheck-NEW.json
node --test tools/orin-task-board/test/board.test.mjs
```

Fresh recheck receipts are linked from the
[post-publication verification receipt](../../artifacts/orin/20261024-08/publication-verification-001.json);
original tables/curves are not overwritten. This includes nine byte-exact
CSV/Markdown reproductions, the component recheck and48passing host tests.
The original PDF remains
SHA256 `92a3af4618a08c7ea66e62b928c3bb32d326e0445a491eee079ecccf70adf997`.
Whole-Goal completion remains false until optional support assessment is
actually performed and its result or concrete platform limitation is recorded.
