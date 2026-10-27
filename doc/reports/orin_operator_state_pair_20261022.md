# Orin VLA State Operator Pair

Date: 2026-10-22, Asia/Shanghai. This is a completed **configured-policy
comparison**, not a VLA E2E result or an observed-internal-precision claim.
The five-model Goal and successful integration gate remain incomplete.

## Result and Scope

The source is SmolVLA's real state materialization/projection: FLOAT `[1,6]`
input, zero packing to `[1,32]`, original Gemm/bias, FLOAT `[1,1,960]` output.
One candidate replaces static ScatterND packing with Slice/Concat. Projection
definition, weights/bias, logical dimensions and source precision requests
remain unchanged. No actions, horizon or solver are benchmarked here.

| Configured-Policy Boundary | Original Mean (us) | Candidate Mean (us) | Original / Candidate |
|---|---:|---:|---:|
| Resident vendor GPU compute | 29.811383 | 30.402696 | 0.980551x |
| IO enabled, vendor GPU compute | 29.488672 | 30.961976 | 0.952416x |
| IO enabled, H2D + GPU + D2H event latency | 39.361035 | 40.169228 | 0.979880x |

There is **no measured mean speedup** in either primary boundary. Resident
per-pair mean ratios span 0.947148--1.032989, with three below one; all five
IO-enabled compute ratios are below one. These are sample-window observations,
not a significance claim from thousands of independent robot trajectories.
The candidate is not selected for whole-model integration.

| Statistic | Resident Original / Candidate (us) | IO Event Original / Candidate (us) |
|---|---:|---:|
| p50 | 29.1138 / 30.1514 | 38.6353 / 39.7949 |
| p95 | 32.8979 / 31.8604 | 42.9688 / 42.1143 |
| p99 | 37.4408 / 36.3422 | 61.4306 / 51.8799 |
| Max | 85.5713 / 103.1490 | 79.7119 / 69.9463 |
| Std | 2.331106 / 2.168916 | 3.750847 / 2.186749 |

Lower tail statistics or std do not erase slower means or the worse candidate
resident maximum. Full-range ECDF PNG/PDF retain all slow records.

## Numerical and Precision Evidence

All sixteen distinct real states are used for complete-output correctness:
**15360 values**, 32 native probes. Same-target worst MSE is
2.8772600378e-8 and lowest cosine 0.999999952857; all sixteen pass the unchanged
MSE < 1e-5 / cosine > 0.9999 gate. Outputs are not byte-exact. Complete native
outputs and CPU-reference diagnostic errors remain, not just selected values.
This validates a state token under the stated tolerance, not robot action
quality or a lossless mathematical transformation.

Both builds explicitly use `/Gemm:fp16`, `--fp16`, `--noTF32` and
`--precisionConstraints=obey`, with no output-precision override. The actual
installed `NvInfer.h` documents that OBEY requires the specified computational
precision or fails the build. This supports the **configuration contract**.

However, baseline Inspector reports a single opaque Myelin stage; its internal
data/weight formats are not expanded. Candidate exposes a CaskGemmConvolution
with FP16 data and 30720 Half weights. External ports on both are linear FP32
and match the original shape contract. Baseline weight storage and internal
accumulation are not independently measured. One/eight Inspector rows are not
CUDA-kernel counts. No cause for the latency difference is assigned to one
hidden kernel or inferred from these row counts.

The quality receipt and timing protocol explicitly retain
`precision_evidence_mode=configured-obey-with-opaque-fusion` and
`same_effective_precision_main_table_qualified=false`. This dataset must not be
relabeled as an independently observed identical-effective-precision kernel
comparison. An explicit audit mode is required for opaque fusion; strict
Inspector mode still rejects it. Identical dense definitions and initializer
bytes, exact CLI maps/OBEY flags, physical IO and every input/output are checked.
No FP32 fallback, output override or quantization experiment is introduced to
force a match. This observability limit remains attached to the measured result.

## Protocol and Cost

Four 128-record CLI pilots and their complete last outputs are independently
audited before freezing five process pairs per boundary. Primary collection has
twenty native processes and **20480 records**, 5120 per boundary/variant. Order
is AB/BA/AB/BA/AB, one stream, 1000 ms warmup, at least 1024 measured rows per
process, no CUDA graph/spin-wait/thread option, trimming or mean-centering.
Correctness covers 16 states; timing reuses the first real state, not 16 fresh
trajectories. Vendor decimal output is checked against each own-engine raw
reference within tolerance, not called byte-exact.

Resident uses the installed NVIDIA sample's integrated-device managed IO path,
with initial staging/final retrieval outside per-query DMA. IO event latency is
H2D + GPU compute + D2H and excludes CPU enqueue/preprocessing; it is neither
host-synchronized state latency nor decoded-observation VLA E2E. Rounded absolute
exported endpoints are not subtracted to manufacture microsecond host timing.

Target serial 1421524312594, boot 3c801f5d-6897-4d97-b021-3c52db38a750, TRT
8.5.2.2/CUDA 11.4/L4T 35.4.1, original MAXN and existing services are unchanged.
Minimum timing admission is 13846640 KiB system-available memory, above the
fixed 8 GiB guard. Current-user-visible owners, restricted-process coverage,
SDK, temperature and frequency snapshots are retained. No exclusivity, new
power mode or full physical-memory attribution is claimed.

One candidate is evaluated. Build-controller walltimes are 5.810065 / 6.846257
seconds, with process-lifetime child max RSS 2236272 / 3010272 KiB. These are not
inference or isolated CUDA-memory measurements. Engine sizes are 96254 / 95833
bytes; neither size nor smaller source-node count proves a speedup. Collection
controller duration 139.483123 seconds includes setup/snapshots/output logging.

## Evidence

Root: `artifacts/orin/20261022-00/`.

| Evidence | Path |
|---|---|
| Original source case / one-candidate budget | `../20261021-08/state-feature-pair-001/case.json` |
| Typed request / 16 distinct real inputs | `operator-state-pair-001/request.json` |
| Two builds / 32 complete native probes | `operator-state-retrieval-001/` |
| Configured-contract quality and explicit precision limit | `operator-state-pair-001/quality-001/quality.json` |
| Actual installed SDK contract | `state-nvinfer-8.5.h`, hash bound in quality receipt |
| Four pilots and frozen scope/protocol | `operator-state-pair-001/timing-pilot-audit-001.json`, `frozen-timing-plan-001.json` |
| Terminal collection / raw inputs / outputs / timings | `operator-state-retrieval-003/` |
| Independent all-row audit and CSV | `operator-state-pair-001/paired-timing-audit-001/` |
| Complete CDF PNG/PDF | Same audit directory, `cdf/state-operator-cdf.*` |
| Request/quality/fusion/timing-scope regressions | `state-target-scoped-regression-002.xml`, 50 passed |

The request parser supports both documented leaf-local and sample-root-relative
paths and explicitly declared unambiguous single-port records. It rejects
ambiguous names, wrong-sample paths, traversal and external symlinks. Old case
files, original helpers and earlier frozen receipts are unchanged. New generic
helpers do not branch on model names in the public runtime.

Re-audit with `audit_record_operator_pair_v2.py --root <retrieval-001>
--allow-fused-configured-contract --sdk-contract <installed-header-copy>
--output <fresh-directory>` and `audit_operator_timings_v3.py` with the archived
plan, quality, terminal collection and matching raw-reference root. Exact target
argv remains in capture/campaign JSON; reruns require fresh directories.

Attention, Embedding and RoPE still need target pairs. A successful effective
full-VLA integration/E2E, RDT recovery and other required models remain open.
This negative configured-policy result does not close D4 or the five-model Goal.
