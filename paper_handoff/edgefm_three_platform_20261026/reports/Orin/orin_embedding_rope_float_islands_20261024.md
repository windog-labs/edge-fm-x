> 可移植报告副本：保留原报告时点；最新结论以根目录总汇总为准。未随包携带的历史链接已转为文字说明；本包不提供原机执行命令或连接信息。

# Orin Embedding / RoPE FLOAT Islands

The four newly built engines implement a separately declared standalone FLOAT
SDK numerical policy: original ONNX/profile bytes, no FP16/INT8 permission,
TF32 disabled and original exact OBEY maps. This is not whole-model FP32
fallback, a precision search, or retrospective acceptance of104/110results.
Those prior internal precision-observability failures remain unchanged.

## Verified Target Results

- Four actual Orin builds exit0. Embedding engines189243786/189243788bytes;
  RoPE engines1152250/268368bytes. Build success alone is not correctness.
- Actual native SDK queries validate every public binding's type, shape,
  device location and linear format against the original contracts.
- All six original native executions complete; all three baseline/candidate
  output pairs,159360values, are independently byte-exact. Primary MSE0,
  cosine1. The original full CPU-reference diagnostics are retained too.
- No application stop/start or SDK/power/clock modification. All managed
  program identities/state, original manager and config remain unchanged.
- Independent terminal closure rehashes120target paths including all four
  engines, checks20recorded births terminal and both owned units empty.
- Thirty-three CPU source/guard/numerical/count/order checks pass. These are not extra
  model observations or evidence of physical GPU allocation precision.

Hidden SDK physical weight/instruction bytes are not claimed observable.
This evidence supports the new declared FLOAT-policy microbenchmark contract,
not the original FP16-permitted effective-precision claim, full action quality,
source model equivalence, an automatic Agent search or VLA E2E speedup.

## Completed Timing

Eight pilot processes pass exact128warmup+128measured and complete final
output checks. The unchanged native C++ counter/event implementation then
executes40independent formal processes: five per variant/family/boundary,
each128warmup+1024measured, all46080raw/40960measured rows retained. Independent
audits verify original alternating order, actual argv/counts, every final
complete output byte, startup8GiB/runtime2GiBguards and all tails.

| Family | CUDA-Event Boundary | Before Mean ms | After Mean ms | Ratio | Negative Pairs |
|---|---|---:|---:|---:|---:|
| Embedding | resident |0.016460231|0.016269644|1.011714x|3/5|
| Embedding | pinned-host IO |0.028954250|0.028883931|1.002435x|3/5|
| RoPE/cache | resident |0.041206087|0.030265394|1.361492x|0/5|
| RoPE/cache | pinned-host IO |0.058652144|0.051242712|1.144595x|0/5|

Embedding does not show a robust gain. RoPE resident max is worse
(0.062816002before/0.071680002after), despite improved mean. All tails remain.
These are explicit CUDA start/end events around enqueue, with IO copies when
declared and actual completion; pinned host buffers are prepared beforehand.
They are not whole-model, decoded-CPU-observation latency or automatic Agent
kernel synthesis. Older operator rows use a different vendor-event timing
implementation/1000mswarmup, not silently equated to these128calls.

One fixed source recipe per family; four actual builds, zero compile failures
in this attempt. Builder-controller elapsed is5.550014/4.858371seconds for
Embedding and3.678220/40.650733seconds for RoPE, not inference or Agent reasoning
time. Older failures remain. Final closure rehashes521target paths and checks
71recorded births terminal, all four owned units empty, all managed program
identities/configuration unchanged. No application outage occurred.
The new CDF PNG is inspected: nonblank, all ranks and labels visible. PDF is
a raster export, not claimed vector artwork. The host uses installed Pillow;
missing matplotlib does not trigger an environment rebuild.

## Reproduction And Evidence

The one-shot build/quality launchers already ran and must not be relaunched.
The independent host audit can use a fresh output name:

> 原机执行命令不属于论文交接内容，已省略。

An initial host invocation omitted that PYTHONPATH and failed to import
`vlaforge`; no new inference or tolerance change was needed to resolve this
host setup error.

- Frozen original inputs/new policy（原记录未随包携带，见未打包引用清单）.
- Build capture（原记录未随包携带，见未打包引用清单）.
- Full quality audit（原记录未随包携带，见未打包引用清单）.
- Target closure（原记录未随包携带，见未打包引用清单）.
- 27checks（原记录未随包携带，见未打包引用清单）.
- Full timing audit（原记录未随包携带，见未打包引用清单）.
- Order/guard/negative-pair audit（原记录未随包携带，见未打包引用清单）.
- Final target closure（原记录未随包携带，见未打包引用清单）.
- 33checks（原记录未随包携带，见未打包引用清单）.
- [CSV statistics](../../data/Orin/orin/20261024-06/publication-001/operator-statistics.csv).
- [Build costs](../../data/Orin/orin/20261024-06/publication-001/build-cost.csv).
- [CDF PNG](../../figures/Orin/orin__20261024-06__publication-001__operator-cdf.png),
  [CDF PDF](../../figures/Orin/orin__20261024-06__publication-001__operator-cdf.pdf).

Five core VLA D1-D3cohorts remain accepted; D4 and the whole Goal remain open.
