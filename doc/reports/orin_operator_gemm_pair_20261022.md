# Orin GEMM Operator Pair

Date: 2026-10-22, Asia/Shanghai. This is one operator signature, not a VLA
main-table or autonomous-kernel result. The five-model Goal remains incomplete.

## Result

The real SmolVLA V projection has FLOAT external ports `[1,177,960]` to
`[1,177,320]`, with a constant `[960,320]` weight. One fixed graph candidate
represents it as Gemm with transposed constant storage and restored batch shape.
Weights, dimensions and logical arithmetic remain unchanged. Both builds retain
the original FP16 dense request and an **empty output-precision override map**.

| Timing Boundary | Original Mean (us) | Candidate Mean (us) | Ratio of Pooled Means |
|---|---:|---:|---:|
| Resident, vendor GPU compute, primary | 38.142144 | 26.338613 | 1.448146x |
| IO enabled, vendor GPU compute, secondary | 22.513342 | 22.558700 | 0.997989x |
| IO enabled, H2D + GPU + D2H event latency | 59.466626 | 58.993376 | 1.008022x |

The IO-enabled compute mean is slightly worse. Two of five IO event-latency
process pairs also have ratio below one (0.972137 and 0.999275). The 1.008x
pooled event-latency ratio is a small observation, not evidence of a substantial
end-to-end improvement. All these results are retained, rather than selecting
only the favorable resident result. Resident per-pair ratios span
1.359492--1.578011.

| Statistic | Resident Original / Candidate (us) | IO Event Original / Candidate (us) |
|---|---:|---:|
| p50 | 36.9873 / 26.1230 | 59.4482 / 59.0210 |
| p95 | 42.7246 / 27.6489 | 62.2559 / 60.7910 |
| p99 | 45.8752 / 30.5176 | 66.0925 / 62.3663 |
| Max | 65.9180 / 56.7017 | 139.8930 / 92.7124 |
| Std | 3.232729 / 1.296002 | 2.646458 / 1.842195 |

Each boundary/variant has 5120 measured rows from five independent processes.
Twenty native timing processes produce **20480 retained records**. The fixed
order is AB/BA/AB/BA/AB per boundary, one stream, 1000 ms vendor warmup and at
least 1024 measured rows per process. There is no graph/spin-wait/thread flag,
trimming or mean-centering. Four earlier 128-record pilots (512 rows total) are
independently checked but excluded from these primary results.

## Quality and Precision

Two original real observations, samples 10/11, supply all 113280 output values.
They are recorded CPU reference activations, not fresh mixed-precision Orin
intermediate captures. All complete outputs are finite and pass same-target
MSE < 1e-5 and cosine > 0.9999, but are not byte-exact.

| Observation | Same-Orin MSE | Cosine | Max Absolute Error |
|---|---:|---:|---:|
| 10 | 3.78315245e-9 | 0.999998009876 | 0.0029296875 |
| 11 | 3.83679218e-9 | 0.999998039690 | 0.0029296875 |

Candidate errors against the CPU FLOAT diagnostic are **higher**, approximately
3.84--3.94e-9 MSE versus 1.06--1.11e-10 for baseline. They remain reported, not
converted into an additional post-hoc gate or described as lossless math.

Both actual dense inspector layers expose FP16 input/output data and 307200
Half weights, with identical external linear FP32 binding shapes/types. The
baseline uses `CublasConvolution`, tactic 2. Candidate uses
`CaskGemmConvolution`, named tactic `ampere_h16816gemm_64x64_ldg8_stages_64x5_tn_v1`.
This is a declared/data/weight precision comparison; internal accumulation is
not independently instrumented. Their five/seven inspector layers are **not**
CUDA-kernel counts. No casts or precision constraints are added to make one
candidate win.

Full final-output JSON from all twenty timing runs is additionally checked
against its own frozen native engine/input output within tolerance. Vendor
decimal output is not a raw byte-exact dump. The revised auditor binds the
reference's engine, input, boot ID and producer to the timing context and
rejects mismatches. It also rechecks all original LayerNorm timing rows without
changing the original LayerNorm receipt.

## Boundaries and Cost

Resident follows the installed NVIDIA sample's managed IO choice on integrated
devices; initial input staging and final output retrieval are outside per-query
DMA timing. IO event latency includes H2D/GPU/D2H, excludes CPU enqueue and
preprocessing, and is **not** host-synchronized raw-observation VLA E2E. No host
latency is reconstructed from rounded absolute exported timestamps.

One declared candidate was evaluated, not an autonomous kernel search. Builder
controller walltimes are 9.020168 / 7.191311 seconds, including its version and
process probes; they are not inference times. Child process-lifetime max RSS is
3027352 / 3008868 KiB, not isolated CUDA memory or summed CPU/GPU physical memory.
Serialized engines are 621864 / 626090 bytes; smaller storage is not claimed.

Target serial `1421524312594`, boot `3c801f5d-6897-4d97-b021-3c52db38a750`, TRT
8.5.2.2/CUDA 11.4/L4T 35.4.1 and existing MAXN remain unchanged. Minimum timing
admission is 15496212 KiB available system memory, above the fixed 8 GiB guard.
Current-user-visible GPU owners and SDK/temperature/frequency snapshots are
retained, with permission-denied process visibility documented. No exclusive
machine, new power mode or 30-minute operator soak is claimed. No other service
is stopped. The 140.075391-second collection-controller duration includes
setup, environment capture and logging and is not an inference metric.

## Evidence

Local root: `artifacts/orin/20261022-00/`.

| Evidence | Path |
|---|---|
| Original real case / one-candidate budget | `../20261021-08/matmul-pair-001/case.json` |
| Generic staged request | `operator-matmul-pair-001/request.json` |
| Complete builds and four raw quality probes | `operator-matmul-retrieval-002/` |
| Same-target quality audit | `operator-matmul-pair-001/quality-001/quality.json` |
| Four pilot checks and frozen plan | `operator-matmul-pair-001/timing-pilot-audit-001.json`, `frozen-timing-plan-001.json` |
| Terminal collection, exact inputs, engines, raw timings/outputs | `operator-matmul-retrieval-004/` |
| Independent timing audit after completed retrieval | `operator-matmul-pair-001/paired-timing-audit-002/timing-audit.json` |
| All measured records and full-range PNG/PDF | Same audit directory, `all-operator-records.csv`, `cdf/gemm-operator-cdf.*` |
| Focused request / quality / precision / raw-timing regressions | `gemm-target-regression-002.xml`, 47 passed |

Earlier retrieval 001 is a nonfinal snapshot, not the authoritative complete
quality receipt. Timing audit 001 is retained; audit 002 was repeated after the
retrieval process exited successfully and is the reporting receipt. No original
case, engine, output, failure or prior result is overwritten.

The generic request runner is model-neutral, uses the existing native builder
and indexed probe, and has admission/owned-process-group timeout cleanup. It
does not add per-model branches to the public runtime. Re-audit locally with
`audit_record_operator_pair.py --root <retrieval-002> --output <fresh-directory>`
and `audit_operator_timings_v2.py --root <retrieval-004/paired-timing-001>
--plan <frozen-plan> --quality <quality.json> --operator-root <retrieval-004>
--output <fresh-directory>`, using paths above. Actual target argv remains in
the respective campaign JSON; target reruns require new directories.

## Still Open

Follow-up: the real full-model integration pilot has now run. Its normalized
action MSE exceeds 1e-5, so the fixed whole-model gate fails and formal E2E is
not started. The completed unchanged-source/profile control also leaves the
trial's normalized MSE above threshold (1.375923e-5 relative to that control);
the isolated operator ratio is not promoted to a model gain. See the
[complete integration report](orin_operator_gemm_integration_20261022.md).

This is not an integrated VLA effect. Effective-candidate full-model output/E2E
validation, Attention/Embedding/RoPE/state target pairs, RDT formal resource
recovery and the remaining required models remain open. The LayerNorm integrated
normalized/native acceptance-space question is separate and unchanged. This
operator result does not complete D4 or the five-model Goal.
