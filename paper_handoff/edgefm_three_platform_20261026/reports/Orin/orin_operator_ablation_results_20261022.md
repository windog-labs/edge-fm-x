> 可移植报告副本：保留原报告时点；最新结论以根目录总汇总为准。未随包携带的历史链接已转为文字说明；本包不提供原机执行命令或连接信息。

# Orin Operator Ablation Results

Updated checkpoint: 2026-10-24. All six operator families have actual paired
same-Orin timing evidence, with explicitly different numerical contracts and
measurement boundaries. Embedding/RoPE now use newly built FLOAT-only SDK
policy engines; the older FP16-permitted internal precision failures remain
unqualified and are not relabeled. This is not the five-model E2E main table.
Every case uses one fixed source-derived candidate, not an autonomous kernel
search. Global FP32 fallback and quantization search are not used.

| Family | Resident Compute Before / After (us) | Ratio | IO Event Before / After (us) | Ratio | Numerical / Precision Scope |
|---|---:|---:|---:|---:|---|
| LayerNorm | 186.408479 / 43.062342 | 4.328805x | 489.500161 / 297.729034 | 1.644113x | Original FP32 norm island, actual exposed formats checked; tolerance pass, not byte-exact |
| GEMM | 38.142144 / 26.338613 | 1.448146x | 59.466626 / 58.993376 | 1.008022x | FP16 data / Half weights observed; tolerance pass, not byte-exact |
| Attention Core | 2746.298173 / 1883.267162 | 1.458263x | 3852.482448 / 3107.671844 | 1.239668x | FP16 QK/PV and FP32 Softmax observed; full output byte-exact |
| VLA state projection | 29.811383 / 30.402696 | 0.980551x | 39.361035 / 40.169228 | 0.979880x | **Configured-policy only**: baseline Myelin internals opaque, identical effective precision not established |
| Language Embedding |16.460231 /16.269644|1.011714x|28.954250 /28.883931|1.002435x| New standalone FLOAT-only SDK numerical policy; target output byte-exact, actual INT32/FLOAT ABI; hidden storage not observed; three negative pairs in both boundaries |
| RoPE/cache |41.206087 /30.265394|1.361492x|58.652144 /51.242712|1.144595x| New standalone FLOAT-only SDK numerical policy; target outputs byte-exact, actual FLOAT ABI; hidden precision not observed, resident max worse |

The state row is separated conceptually from an observed-effective-format
comparison and must retain that limit in any exported paper table. Exposed
formats do not independently instrument internal accumulation precision.

Each family has five independent process pairs per boundary in
AB/BA/AB/BA/AB order and one stream. The original four use1000mswarmup;
the new Embedding/RoPE use exactly128warmup+1024measured per process, with
native CUDA-event brackets and pinned-host IO copies when declared. Their
40processes retain all46080raw/40960measured records. Do not silently equate
that event bracket or128calls to the older vendor-event implementation.
Norm, GEMM
and state retain 20480 measured rows each. Attention retains all 20800 rows
from its accepted retry; its earlier 20479-row attempt is rejected because one
process exported only 1023. The retry keeps the minimum 1024 gate and requests
1040, with no truncation, mixing attempts, trimming or process centering.
Thousands of repeated timing records are not independent robot trajectories.

Resident timing excludes per-query data transfers. IO event latency sums vendor
H2D, GPU compute and D2H events, excluding host preprocessing/enqueue overhead.
The new two use actual start/end CUDA events around enqueue and, for IO,
the declared transfers; they are not the older vendor-event sum. Neither
implementation is complete decoded-observation E2E. Full tails, raw CSV and complete
outputs remain in each report, including worse tails/std and negative means.

## Historical Precision Boundary

The remaining two operator families have now actually built and executed in
the target follow-up（原记录未随包携带，见未打包引用清单）, using the frozen
native cases（原记录未随包携带，见未打包引用清单）. Complete
same-target outputs are byte-exact and all four native SDK ABI queries pass.
RoPE's internal slice alias cannot be used as a public binding descriptor;
the new auditor changes only that lookup, preserving numerical/source/precision
gates and a whole-function inverse. Both vendor-fused cases still lack observed
internal data/storage precision, so neither enters timing as identical-effective-
precision. All66source/typed/SDK tests pass;160target files and26terminated
process identities are checked. The original typed auditor（原记录未随包携带，见未打包引用清单）,
old dense/FLOAT auditor and prior configured-only limits remain unchanged.

This paragraph describes the original104/110engines, not the new06contract.
The [new FLOAT-island report](orin_embedding_rope_float_islands_20261024.md)
adds actual new builds, same-input full output/ABI checks, exact-count timing,
budget/cost/negative pairs,33checks and521target hashes/71terminal births.
It is not whole-model FP32fallback, a precision search or proof of hidden
instruction/storage bytes. New recipes are not integrated into a VLA here;
the required representative full-model Norm integration remains below.

## Whole-Model Status

- **Norm:** full seven-engine/ten-step tensor-input pilots pass the normalized
  complete-action primary gate (max MSE 5.30081810e-6, min cosine
  0.999998499934). Native max MSE 0.0121000582 is supplemental under the user's
  explicit decision, not discarded or reclassified as zero error. Raw-input
  pilots pass all 80 output tensors against their own frozen references, but
  show a slower candidate (132.387 vs 125.310 ms). The full five-process-pair
  E2E protocol is now audited: all 10240 measured rows / 23040 full tensors
  retained, 125.900306 -> 133.994095 ms (0.939596x, candidate 6.4287% slower).
  All five pair ratios are below one; no integrated gain is inferred from
  4.329x micro. [Formal raw E2E report](orin_operator_layernorm_raw_e2e_20261022.md).
- **GEMM:** full integration fails normalized MSE against both the adopted
  model (1.324344964e-5) and an unchanged-source/profile rebuild control
  (1.375922937e-5). It is not admitted to formal E2E despite micro gains.
- **State:** negative means and limited precision observability; not selected
  for integration.
- **Attention:** operator quality/timing is complete. The adopted full-prefix
  [candidate now actually builds and runs](orin_operator_attention_integration_20261024.md):
  one zero-mask Add removed, all840original initializers and retained precision
  declarations preserved. Both complete seven-stage/ten-step paths execute;
  34complete normalized/native output pairs are audited. Normalized worstMSE
  1.2796944331e-5fails against the adopted baseline, and1.2985515613e-5still fails
  against an existing unchanged-source rebuild. Native errors are supplemental.
  High cosine does not override the fixed MSE gate. No raw integrated E2Etiming
  starts for this candidate; the micro ratio is not a whole-model speedup.

Existing same-engine SmolVLA ordinary/shared/replay results are separate runtime
experiments. They must not be multiplied by these micro ratios or retrospectively
treated as graph-ablation measurements. A high cosine does not certify physical
robot safety, task success or mathematical losslessness.

## Evidence

| Family | Detailed Report |
|---|---|
| Norm | Operator（原记录未随包携带，见未打包引用清单）, integration（原记录未随包携带，见未打包引用清单） |
| GEMM | Operator（原记录未随包携带，见未打包引用清单）, [integration and rebuild control](orin_operator_gemm_integration_20261022.md) |
| Attention | Operator quality, timing, CDF and rejected attempt（原记录未随包携带，见未打包引用清单）, [complete-model numerical failure/control](orin_operator_attention_integration_20261024.md) |
| State | Configured-policy comparison and negative results（原记录未随包携带，见未打包引用清单） |
| Embedding/RoPE | [New FLOAT-policy complete timing/cost/CDF](orin_embedding_rope_float_islands_20261024.md), original precision-unqualified target result（原记录未随包携带，见未打包引用清单）, historical source preparation（原记录未随包携带，见未打包引用清单） |

The web board（原机本地服务，未随包提供） and its
structured source（原记录未随包携带，见未打包引用清单） remain the current
progress view. Core VLA formal completion is4/5, with complete quality pilots5/5.
CogACT formal sampling is not activated pending separate authority and the
remaining admission/repeat gates. These two operator probes do not add a formal
model, positive full-model integration or optional Qwen3.5 result. Historical
resource-stopped datasets and failures stay separate from accepted cohorts.
