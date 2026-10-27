# Orin Attention Core Pair

Date: 2026-10-22, Asia/Shanghai. This is a real operator microbenchmark, not
complete VLA E2E or a claim that all five models are finished.

## Contract and Quality

The case uses one real SmolVLA observation (sample 6), with all three FP32 ABI
inputs explicitly loaded: Q [1,12,1024,64], transposed K [1,12,64,1024] and
V [1,12,1024,64]. These are original CPU-recomputed activations, not a capture
of all intermediate values inside the mixed-precision Orin model. Upstream
Q/K scaling is unchanged. The candidate removes only a statically proved,
non-overridable finite positive-zero mask Add before the final-axis Softmax.
This case does not establish causal/masked attention generalization.

The same-target complete context contains 786432 FP32 output values. All are
**byte-exact**, MSE 0 and cosine 1. Both sides retain requested and inspector-
observed FP16 QK/PV data and FP32 Softmax data. Unlike constant-weight GEMM,
these MatMuls have two runtime matrix inputs; no constant-weight precision is
invented. Internal accumulation precision is not independently instrumented.
The original Add's FP32 precision declarations disappear only with that Add;
retained operators' compute/output declarations are unchanged. CPU FLOAT
diagnostic errors are identical (MSE 5.30532650e-8, cosine 0.999997010145),
not new PyTorch FP16-equivalence or cross-device acceptance evidence.

The original graph has four source nodes and the candidate three. Actual
TensorRT inspector rows are eleven and nine, including reformats; this is not
a measured GPU kernel-launch count. Native build-controller elapsed times are
8.838 and 4.694 seconds, not inference latencies. Candidate budget is one
source-derived generic graph recipe, not autonomous kernel synthesis/search.

## Timing

Each boundary has five independent process pairs, ordered AB/BA/AB/BA/AB.
Warmup is 1000 ms per process. The accepted retry requested 1040 queries and
retained all 1040 exported measured rows per process: **20 processes, 20800
rows**, 5200 per variant/boundary. The minimum remains 1024 exported rows.
No outlier filtering, process centering or truncation was applied. Every
exported final complete output is also checked against its frozen native probe.

| Boundary | Before Mean (ms) | After Mean (ms) | Ratio of Means |
|---|---:|---:|---:|
| Resident GPU compute | 2.746298 | 1.883267 | 1.458263x |
| IO-enabled GPU compute, secondary | 2.893363 | 2.031240 | 1.424432x |
| DMA + GPU event latency | 3.852482 | 3.107672 | 1.239668x |

Resident p99/max/std before are 3.023080/3.138920/0.074646 ms, after
2.089360/2.234740/0.059288 ms. IO event p99/max/std before are
4.272992/5.716800/0.123051 ms, after 3.464979/4.345210/0.115679 ms.
All five resident and all five IO compute pair ratios exceed one for this
bounded input/profile. IO event latency is the vendor sum of H2D, GPU compute
and D2H events, not host wall-clock E2E. Resident timing excludes transfers.

The first collection is **rejected**, not merged into the accepted retry: one
process exported 1023 rows and nineteen exported 1024, total 20479. All rows,
logs and the original protocol are preserved. Retained TensorRT 8.5 sample
sources show separate inference warmup/skip and reporting cut logic; this is
consistent with a counting-boundary issue but does not isolate its exact
off-by-one cause. The retry adds a request margin while keeping the original
acceptance minimum, all observations, precision, engines and full tails.

## Evidence

Root: `artifacts/orin/20261022-00/`.

| Item | Location |
|---|---|
| Real-input source and precision request | `operator-attention-pair-001/request.json` |
| Complete native quality/inspector audit | `operator-attention-pair-001/quality-001/quality.json` |
| Four 128-query pilots and full final outputs | `operator-attention-pair-001/timing-pilot-audit-001.json` |
| Original rejected count audit | `operator-attention-pair-001/export-count-failure-001.json` |
| Frozen accepted-retry protocol | `operator-attention-pair-001/frozen-timing-plan-002.json` |
| Accepted raw target archive | `operator-attention-retrieval-004/paired-timing-002/` |
| Independent complete timing audit | `operator-attention-pair-001/paired-timing-audit-002/timing-audit.json` |
| All timing rows | `operator-attention-pair-001/paired-timing-audit-002/all-operator-records.csv` |
| Unclipped PNG/PDF CDF, hashes | `operator-attention-pair-001/cdf-002/` |
| Source/quality/multi-input regressions | `attention-multi-input-regression-002.xml`, 38 passed |
| Primary policy and rejected-count regressions | `primary-policy-and-export-counts-001.xml`, 6 passed |

Engine SHA256 before:
`6b0e12f587c858c51a20bfb3cef88895c74f5fe7bd4a2905ddeca80383870237`;
after: `23641c3e47eac12dddf14911fa10604b76cffabcca40200e98ccad10dc621508`.
Target serial `1421524312594`, boot `3c801f5d-6897-4d97-b021-3c52db38a750`,
TRT 8.5.2.2, CUDA 11.4, existing MAXN unchanged. Each process retains environment,
current-user-visible owners, memory, native argv/PID/start ticks and raw outputs.
Permission-denied owner visibility is not proof of machine exclusivity.

Re-audit without target computation, choosing a fresh output directory:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-00/audit_operator_timings_v5.py \
  --root artifacts/orin/20261022-00/operator-attention-retrieval-004/paired-timing-002 \
  --plan artifacts/orin/20261022-00/operator-attention-pair-001/frozen-timing-plan-002.json \
  --quality artifacts/orin/20261022-00/operator-attention-pair-001/quality-001/quality.json \
  --operator-root artifacts/orin/20261022-00/operator-attention-retrieval-004 \
  --output artifacts/orin/20261022-00/operator-attention-pair-001/recheck-new
```

Full-model integration of this Attention recipe is not measured here. Later
[full-prefix source preparation](orin_attention_integration_preparation_20261022.md)
is complete but has no target build or complete-action/E2E result. LayerNorm
integration is a separate negative E2E experiment. RDT's old resource-blocked
cohort remains preserved; its new authorized-service-window campaign is running.
Core VLA formal completion remains 1/5.
