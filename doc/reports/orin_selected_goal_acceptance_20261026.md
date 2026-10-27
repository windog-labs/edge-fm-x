# Orin Selected Goal Acceptance

Date: 2026-10-26, Asia/Shanghai. **Selected-scope acceptance complete:**
measurement, resource, target closure, actual CDF visual review and all31
Goal items are independently verified within the explicit limits below.
Five required VLA campaigns are
accepted; Qwen3.5-0.8B formal006 has completed its full cohort. Qwen3.5-2B
is explicitly user-cancelled, not tested, passed, unsupported or a remaining
completion prerequisite. Original numerical thresholds and sample counts
are unchanged.

[Independent31-item acceptance](../../artifacts/orin/20261025-21/selected-goal-review-001.json).
This is completion of the selected measurement deliverables, not proof of
every stronger manuscript claim; the latter remain in the claim-gap table.

## Scope And Evidence

This reviews the [original 31-item Goal](../../artifacts/orin/20261024-08/active-goal-objective.md)
under the [explicit selected-model amendment](../../artifacts/orin/20261025-20/user-scope-amendment-001.json).
The [core acceptance](orin_core_acceptance_20261024.md) and
[retained-data review](../../artifacts/orin/20261024-08/core-requirement-audit-003.json)
remain valid within their recorded scope. Completed VLA payloads were
[deleted with permission](orin_storage_retention_20261024.md); retained raw
measurements, full-output audit evidence, quality/CDF/configuration/source
records support data reanalysis, not currently executable retired engines.

Qwen data/resource/target closure is recorded in the actual [complete export](../../artifacts/orin/20261025-21/final-export-006-retry-001/summary.json)
and [hash-bound visual observation](../../artifacts/orin/20261025-21/cdf-visual-review-006.json).
Both actual CDF PNGs were displayed and inspected; original PDF/rank hashes
are checked independently. The original manuscript PDF is not modified.

## Main Measurements

All entries are real Orin, batch1, decoded CPU observations to complete
consumable CPU outputs. Five independent processes per configuration retain
128 warmup and 1024 measured requests each. Sixteen varying recorded
observations cycle; these are not independent robot trajectories.

| VLA | Ordinary Mean ms | Selected Mean ms | Selected chunks/s | Comparison |
|---|---:|---:|---:|---|
| SmolVLA | 134.409630 | 113.139929 | 8.838613 | Complete-loop replay |
| RDT-1B | 3065.625218 | 2975.587343 | 0.336068 | Complete-loop replay |
| pi0 | 1129.869939 | 1125.379221 | 0.888589 | Arena only; no replay |
| pi0.5 | 1258.792712 | 1256.955605 | 0.795573 | Arena only; no replay |
| CogACT actual approximately7.63B | 694.393388 | 693.790506 | 1.441357 | Public-dependency configuration; Arena only |

The [original full tables](../../artifacts/orin/20261024-07/tables-001/tables.md)
retain all percentiles, standard deviations, per-process statistics and
negative rounds. Core totals are12configurations/60processes,
69120raw/61440measured requests and18original CDF files. Small Arena mean
differences do not establish significant gains. No H20 latency substitutes
for these measurements.

### Qwen3.5-0.8B Supplement

| Policy | n | TTFT Mean / P95 / P99 / Max / Std ms | Full16 Mean / P95 / P99 / Max / Std ms | E2E tokens/s | Native VmHWM MiB |
|---|---:|---|---|---:|---:|
| Ordinary off | 5120 | 334.328554 / 339.407525 / 343.079333 / 357.044384 / 2.586327 | 1593.061317 / 1600.865789 / 1605.455708 / 1618.577408 / 4.441399 | 10.043556 | 6729.117188 |
| Verified Arena | 5120 | 333.349658 / 338.417126 / 341.265658 / 357.665152 / 2.655338 | 1586.510069 / 1593.629232 / 1598.201941 / 1628.726880 / 4.247181 | 10.085029 | 6706.246094 |

This is the complete10process/5pair/11520raw/10240measured cohort, not pilot
timing or merged partials. The average full16 reduction is approximately0.41%,
not established significant acceleration. Both observed maxima are higher
under Arena; TTFT standard deviation also increases. All tails and all five
paired rounds remain. Sampled process VmHWM is not private GPU memory, a
matched request peak, or evidence of a memory-saving benefit.

[Full metrics and scope](../../artifacts/orin/20261025-21/final-export-006-retry-001/measurements.md),
[performance CSV/CDF summary](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/summary.json),
[sampled resource audit](../../artifacts/orin/20261025-21/final-export-006-retry-001/resources/summary.json).
Across the five VLA plus selected VLM packages, totals are14configurations,
70processes,80640raw and71680measured requests. Their action/token throughput
units remain distinct; no combined throughput is computed.

## Numerical And Timing Contract

The primary comparison is ordinary versus optimized TRT on the same Orin,
same precision/engines/inputs/noise. Complete normalized actions retain the
fixed MSE<1e-5/cosine>0.9999 gate; native-unit and H20 differences are
supplemental. Integer auxiliary outputs use exact equality, not cosine.
The five formal VLA campaigns are byte-exact under their qualified profiles.

All184320Qwen output token values are exact to the accepted same-Orin
reference.320complete final-logit/ID witnesses cover every input in every
process's first/last cycle, including160direct paired logit comparisons:
all finite and byte-exact, MSE0/cosine1. Intermediate logits from every
request are not claimed persisted. Varying-input reset and independent
process repetition are verified; concurrent Qwen Sessions are not tested.

Qwen uses the same3neural and2parameter-free control engines in both
profiles, fresh online native Rust tokenization and RGB normalization/
patchification every request, explicit state reset, and fixed16new tokens.
The complete path has17neural and16control calls; it does not replay the
whole loop or memoize vision/prefill. TTFT ends at CPU-ready first token;
full16 includes all token/final-logit D2H. Initialization, file decoding,
reset bookkeeping and output logging are outside the measured interval.

Qwen source/H20 comparisons remain negative supplemental evidence:
only7/16complete token arrays match, worst final-logit MSE8.65417583235,
minimum cosine0.6973061539. Same-Orin optimization equivalence is not
original-source equivalence, mathematical losslessness or robot-task success.

## Requirement Review

Core rows below retain the original accepted scope. The independent final
review verifies all31rows, complete Qwen data/resources/CDF visual evidence,
target/service closure and retained core evidence. No pending selected item
is silently converted to a paper-level assertion.

| Goal ID | Evidence-Bounded Result |
|---|---|
| route.1 | Native C++/CUDA/TRT target inference; host export/reference separate; no Orin Torch model |
| route.2 | Five actual native full-model deployments accepted; complete native Qwen0.8B data/closure verified |
| route.3 | Historical smokes and cancelled Torch build retained, not model acceptance |
| route.4 | Same-Orin matched-profile comparison; H20 supplemental only |
| route.5 | Qwen original Supervisor unchanged; two authorized Java programs restored, all selected owned units closed |
| D0.1 | Original PDF claims mapped to actual evidence and remaining unsupported claims |
| D0.2 | All five required VLA models; actual CogACT public-dependency identity retained |
| D0.3 | Frozen source/checkpoints/profile/shape/horizon/solver and actual noise |
| D0.4 | Actual builds, source bindings, failures and capacity boundaries retained |
| D1.1 | Five separate accepted model cohorts, serialized target compute |
| D1.2 | Complete fresh vision/language and all action iterations |
| D1.3 | Declared FP16/mixed profiles and high-precision islands; no whole-model FP32 fallback or quantization search |
| D1.4 | Generic graph/cast/plugin source lineage and complete-output qualification |
| D1.5 | Adapter/typed IR/Plan/native Session, with explicit model semantics |
| D2.1 | Matched inputs/weights/noise/precision, source/H20 error separate |
| D2.2 | Complete finite typed VLA outputs, shapes/hashes and original byte audits |
| D2.3 | Fixed normalized primary gates, supplemental native errors, failed candidates retained |
| D3.1 | Engine/runtime differences explicit; Arena-only is not replay |
| D3.2 | Batch1 decoded CPU input to complete CPU action chunk with actual GPU completion |
| D3.3 | Three correlated same-request timing boundaries, not summed stage means or independent reruns |
| D3.4 | Five independent processes/configuration,128warmup+1024measured; no partial-cohort merge |
| D3.5 | All raw samples/tails, percentiles, throughput and original CDF artifacts retained |
| D4.1 | Six real paired operator signature families, with precision/method limits |
| D4.2 | Fixed recipes/build costs/failures; full Norm integration qualified but6.43%slower |
| D4.3 | Representative reset/isolation/lifecycle/replay and covered allocations; private SDK/GPU peaks unknown |
| D4.4 | Recorded resource series,>30min continuous measured model work and matched power |
| D4.5 | MAXN verified; alternate-mode permission limit recorded, no second-mode performance |
| D5.1 | Complete native0.8B vision/prefill/15decode, explicit state/reset and quality evidence;2Buser-cancelled |
| D5.2 | Complete0.8B TTFT/full16/output/CDF/resources verified; gaps and source/H20 errors explicit |
| D5.3 | Core and selected-scope tables/reproduction published; all31requirements independently reviewed |
| D5.4 | Historical evidence/PDF preserved; all11model/controller births closed and actual original-manager restoration independently verified |

## Resource And Claim Limits

Native process RSS/VmHWM is separate from system MemAvailable and not an
isolated physical GPU peak. Qwen's independent RSS observers003/004/005
ended with retained metadata errors; their valid prefixes plus naturally
terminal006are reviewed chronologically. They contain3598/5083/7216/1734
valid rows, covering all10model births. Gaps between valid segments are
574.391277/317.417639/163.190024seconds, not imputed. Observation gaps, first-process late
attachment and unknown unobserved peaks remain explicit. No gap filling,
zero inference or continuous1Hz coverage claim is allowed. Kernel VmHWM
is lifetime high-water at read time, including initialization/warmup.

MAXN is not relabeled60W. Power-unit uncertainty and alternate-mode
permission limits remain. Core pooled jitter does not meet the manuscript's
1.2ms target; the separate single-worker soak is not a substitute. Isolated
operator gains do not imply full-model gains, zero-copy/zero-malloc, universal
allocation freedom or closed-loop robot performance. These are reporting
boundaries, not experimental values invented to close the Goal.

The [actual target closure](../../artifacts/orin/20261025-21/final-export-006-retry-001/target-closure.json)
checks125frozen asset pins,6mapped SDK files,11recorded controller/model
births and successful empty compute/final-observer units. The resource
window also closed successfully. The original Supervisor PID4237 is
unchanged; data-collection.jar PID643146 and starter PID643080 are Running.
Restored program PIDs changed, so `managed_programs_unchanged` remains false;
the separately audited `authorized_service_scope_closed` is true. This
does not certify application functional health or restored in-memory state.

Post-terminal collection initially failed on an oversized SSH argument and
a consumer assumption that preflight records contain `unit`. The new
stdin transport and actual owner/guard/native-cgroup contract repair are
[recorded separately](../../artifacts/orin/20261025-21/terminal-archive-repair-001.json).
The empty failed export destination remains preserved; the fresh retry is
the published export.405CPUtests+138subtests pass. No numerical worker,
engine, protocol, raw result, service setting or model run changed.

## Reanalysis

Use fresh output paths; the completed receipts are immutable. Host-only
commands reanalyze retained data and do not start inference or control services.

```bash
python3 -B artifacts/orin/20261025-21/audit_resident_formal.py --root artifacts/orin/20261025-21/native-resident-formal-20261025-21-006 --plan artifacts/orin/20261025-21/resident-formal-assets-006/plan.json --output /tmp/qwen-data-review-NEW.json
python3 -B artifacts/orin/20261025-21/review_selected_goal.py --export artifacts/orin/20261025-21/final-export-006-retry-001 --visual artifacts/orin/20261025-21/cdf-visual-review-006.json --report doc/reports/orin_selected_goal_acceptance_20261026.md --output /tmp/orin-selected-review-NEW.json
node --test tools/orin-task-board/test/board.test.mjs
```

Performance export additionally requires the existing report environment
`/home/zhangzimo/.venvs/edgefm-vla-report-py313-20260906/bin/python` and the
audit's exact bytes/SHA256 via `publish_resident_formal.py --audit ...
--audit-bytes ... --audit-sha256 ... --output NEW_DIRECTORY`.
See [delivery index](edgefm_orin_experiment_delivery.md),
[managed-run history](orin_qwen_managed_sampling_20261026.md) and
[claim-gap table](orin_claim_gap_20261021.md) for original evidence and limits.
