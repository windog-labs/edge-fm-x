# H20 Experiment Delivery

This is the H20 closeout snapshot of 2026-09-10. Its deferred J6M statements
below describe that date. Subsequent SmolVLA J6M stage measurements are in
[the 2026-09-14 stage delivery](smolvla_j6m_performance_20260914.md);
see [the combined summary](edgefm_h20_j6m_experiment_summary_20260914.md)
for current platform coverage. The original H20 measurement values are retained.

The H20 experiment deliverables requested for the current Goal are complete,
including native Qwen3.5 deployment on the fixed H20 profile. This is scoped
completion of the available-hardware work, not acceptance of the whole edge
deployment paper. Orin/J6M/BPU execution remains deferred, and the qualified
CogACT profile must retain its actual model identity. The original PDF and
pre-existing dirty worktree are preserved.

## Coverage and Remaining Work

| Requested Item | Current Evidence | Remaining Work / Conditional Estimate |
|---|---|---|
| Five VLA H20 throughput and latency | Complete: SmolVLA, RDT, pi0, pi0.5 and qualified CogACT; official/Session original-input comparison plus separate resident and host-model-tensor tables | No additional H20 formal run required for these profiles |
| Continuous inference CDF and jitter | Complete: raw CSVs, all outliers, per-model and combined PNG/PDF | No additional run required |
| Complete action quality | Complete: normalized/native typed outputs, MSE, max absolute error, cosine and byte equality | Equality is established on the recorded inputs; no physical success-rate claim |
| Six operator families | Complete: five process pairs per family, complete tensor and ownership audits; negative results retained | Autonomous kernel synthesis and integration of these candidates into full models are not established |
| Qwen3.5 0.8B and 2B official multimodal baselines | Complete: real image+text, TTFT, full generation latency, decode throughput, peak memory and full tokens | Python reference only; native compatibility is recorded separately |
| Qwen3.5 native deployment | Complete for the fixed H20 profile: 0.8B/2B, first-token and 16-token, 20 workers, 20480 measurements, 46080 complete output tensors byte exact, four independent audits, CDFs and 80 lifecycle checks | No additional work for this fixed profile; Orin/J6M/BPU and arbitrary streaming profiles remain out of scope |
| Final tables, evidence index and regression | Complete: 2935 evidence identities rechecked, Qwen closeout 2275 passes/62 skips, CogACT closeout 2276 passes/62 skips, and focused C++ checks | No additional experimental work for this delivery |
| Orin and J6M/BPU main-table experiments | Deferred by user because hardware is unavailable | Cannot give a dependable completion date before hardware, SDK and runtime access are available |
| Incorporation into manuscript | Evidence tables and claim review supplied; PDF unchanged | About 2-4 hours for table/figure insertion and wording reconciliation once editable manuscript sources are available; excludes new experiments and author review |

Estimates are engineering judgments about additional work, not running-job
countdowns. The Qwen result uses explicit typed state and a generated native C++
Session on H20. CogACT's autonomous C++ RNG path is now separately verified and
boundary-aligned with the official implementation; equivalence to the original
gated Meta dependency configuration remains unverified and must stay qualified.

## Original-Input H20 Table

Each row contains five independent processes with 128 warmups and 1024 measured
calls per process. Across ten rows there are 50 workers and 51200 measured calls.
The timer starts with decoded observations in CPU RAM and ends with complete CPU
action outputs. It includes preprocessing, H2D, inference, postprocessing and D2H;
file/video decoding, initialization, logging, validation and robot transport are
excluded. Two raw timing segments sum to each complete-call latency. Initialization
and sampled worker memory are separately available in the machine-readable table.
This full host workflow uses Python around a generated native Session; it is not
a certificate of a completely Python-free sensor-to-actuator pipeline.

| Model | Configuration | Mean ms | p99 ms | Std ms | Fresh Chunks/s |
|---|---|---:|---:|---:|---:|
| SmolVLA | official-pytorch | 206.161639 | 220.579781 | 9.333303 | 4.850563 |
| SmolVLA | session-required | 50.726747 | 53.682128 | 1.808891 | 19.713466 |
| RDT-1B | official-pytorch | 239.685959 | 301.828993 | 11.895341 | 4.172126 |
| RDT-1B | session-required | 221.804152 | 230.311815 | 3.366552 | 4.508482 |
| pi0 | official-pytorch | 221.305113 | 350.640813 | 24.208986 | 4.518648 |
| pi0 | session-required | 108.152217 | 113.976963 | 1.901058 | 9.246227 |
| pi0.5 | official-pytorch | 256.840307 | 274.030360 | 10.815259 | 3.893470 |
| pi0.5 | session-required | 116.853524 | 122.893219 | 2.031607 | 8.557722 |
| CogACT (public dependencies) | official-pytorch | 120.286270 | 127.398996 | 2.700216 | 8.313501 |
| CogACT (public dependencies) | session-required | 79.971651 | 82.199409 | 0.982177 | 12.504431 |

Throughput counts freshly computed action chunks, not robot servo cycles.
All 25 required workers record 1152 replay invocations each with zero ordinary
fallback; replay executes the captured computation rather than returning a saved
action. RDT uses its original five-step scheduler; the other profiles use ten
steps. Warmups are excluded from latency and retained in fidelity storage.

All 149760 complete output tensors, including warmups, match their frozen
references byte for byte. Floating action MSE and max absolute error are zero;
the minimum reported cosine is 0.9999999999999998. Full F64/BF16 outputs retain
their dtypes. CogACT's integer RNG receipts are compared exactly without an
artificial cosine score. Repeated calls cycle through finite observation/noise
sets; they are not 51200 distinct trajectories.

The combined CDF PNG/PDF and per-model plots have been inspected. The full table,
quality table, report and combined figures are in the absolute directory:

`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-five-summary-001/`.

| Artifact | SHA256 |
|---|---|
| `report.json` | `077362fc3ec81e74c8440374c677f525dd1ae6271b5d196e753849b7d2b9856c` |
| `host-pipeline-table.csv` | `108e60979891bd7e43b9a93d8eef36b3fd8dbd4bbb22f3cadddefc976ff6af02` |
| `complete-output-quality.csv` | `723f752ecda899537d0fab4bc9af6988f64a1d4fa531231bbc6af061354e64ff` |
| `host-pipeline-cdf.png` | `4fe5c243c8ab3995bf598dab428036dc7d53b19580759c59502f507a7b67ab66` |
| `host-pipeline-cdf.pdf` | `5101cbe062d5d66ec6dcdc2b7885c1299a19697444e52313927dd1bc570b5f16` |

The report records all local/remote roots, all 50 worker/container and NVML PIDs,
controller PIDs, GPU UUIDs, protocol hashes, input/reference identities, frozen
source provenance, environment bindings, replay receipts and remote/local audit
hashes. Full archives are in the adjacent `raw-host-formal/{smolvla,rdt,pi0,pi05,cogact}/`
directories, including the per-model raw latency/CDF CSVs and typed output bytes.

## CogACT Autonomous Same-Boundary Comparison

The [boundary-aligned CogACT report](cogact_timing_boundary_formal_20260910.md)
replaces the earlier `1.401x` figure. Five official and five autonomous native
processes were serialized on H20-2 GPU 0 through GPU 4 with 128 warmups and 1024
measured calls per process. Both paths include their RNG preparation, model or
original VLM/DDIM sampler, and CUDA completion; both exclude static input H2D,
binding, preprocessing and output conversion.

| Configuration | Mean ms | p50 ms | p95 ms | p99 ms | Calls/s |
|---|---:|---:|---:|---:|---:|
| Official PyTorch | 129.258 | 124.006 | 177.383 | 232.080 | 7.736 |
| Autonomous native Session | 88.751 | 89.199 | 90.683 | 95.811 | 11.268 |

The all-sample official/native mean ratio is `1.456x` with a 95% bootstrap
interval of `[1.450, 1.463]`; native mean latency is 31.34% lower. Official
worker 4 has a retained heavy tail, so the median paired-GPU mean ratio `1.393x`
is reported alongside the aggregate rather than removing the worker. All 5120
measured calls per configuration produced complete raw, normalized, native and
RNG outputs byte-exact against the same reference.

| Model | Terminal Controller PID | H20 Host | GPU UUID |
|---|---:|---|---|
| SmolVLA | 113987 | zzm-h20-x8-2 | GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5 |
| RDT | 119430 | zzm-h20-x8-2 | GPU-ae321531-504c-f416-e693-94fa02779130 |
| pi0 | 85392 | zzm-h20-x8-2 | GPU-3596876b-e679-6289-f784-c8d98294ed23 |
| pi0.5 | 80976 | zzm-h20-x8-2 | GPU-29fb9c6e-2a11-e303-5f26-ad079fa86fa5 |
| CogACT | 143713 | zzm-h20-x8-2 | GPU-3596876b-e679-6289-f784-c8d98294ed23 |

Both authorized H20 hosts were checked dynamically. SmolVLA's H20-1 pilot passed;
its formal attempt was rejected when a foreign GPU owner appeared, before any
formal worker launched. That failed attempt is retained and excluded. The accepted
replacement has its own H20-2 protocol and audited pilot. No foreign job was
displaced. `huoshan-private` was used only for transfer from shared NAS.

## Other Formal Tables

The [CUDA-resident table](h20_vla_formal_table_20260909.md) and
[host-model-tensor table](h20_host_tensor_formal_table_20260909.md) each contain
15 rows, 76800 measured calls and 224640 complete output tensors. They distinguish
off, batch-only and required policies and retain their own timing boundaries.
They must not be merged into or relabeled as the original-input measurements.

The [six-family operator report](h20_operator_table_20260909.md) contains:

| Family | Baseline us | Candidate us | Latency Reduction |
|---|---:|---:|---:|
| Attention | 103.301055 | 103.313151 | -0.0117% |
| GEMM / Linear | 11.540214 | 12.870522 | -11.5276% |
| LayerNorm | 13.816088 | 13.716502 | 0.7208% |
| Embedding | 2.142171 | 1.865898 | 12.8969% |
| RoPE | 21.332679 | 8.486607 | 60.2178% |
| VLA solver/update | 4.289051 | 2.203624 | 48.6221% |

Values are means of five per-process medians of graph-batch device timings.
This is agent-assisted selection of fixed compiler recipes on archived real-model
signatures. Attention is unchanged, Linear is slower, and no candidate is promoted
to full-model deployment by this table. LayerNorm covers the normalization family;
separate RMSNorm results and autonomous kernel invention are not claimed.

## Qwen Evidence

The native Qwen3.5 formal result is in
[Qwen3.5 Native H20 Formal Deployment](qwen35_native_formal_20260910.md).
Both model sizes completed first-token and fixed-16-token native C++ campaigns:

| Model | Native profile | Mean ms | p99 ms | Calls/s | 16-token tokens/s |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B | first token | 96.905373 | 114.008338 | 10.319345 | - |
| Qwen3.5-2B | first token | 90.657933 | 137.604724 | 11.030474 | - |
| Qwen3.5-0.8B | 16 tokens | 369.289954 | 625.620183 | 2.707899 | 43.326388 |
| Qwen3.5-2B | 16 tokens | 370.497338 | 629.293585 | 2.699075 | 43.185196 |

The four campaigns contain 20 independent processes, 20480 measured calls,
46080 complete output tensors and 5721488640 values. All outputs match complete
direct/eager references byte for byte; each generated C ABI also rejects seven
invalid shape probes, rejects `accepted=false`, and preserves committed output
hashes after failure. A separate generic lifecycle probe passed for all four
bundles: repeated calls, two-Session interleaving, rejected reset, successful
reset invalidation, post-reset reproduction and destroy/recreate were covered,
then independently rechecked by 80 assertions. The timer includes H2D, binding,
Session run, completion and complete typed D2H, but excludes image preprocessing,
initialization, validation/log I/O and detokenization.

The [official multimodal report](qwen35_natural_profile_20260909.md) provides five
independent processes and 5120 measured calls per model. Each uses a real 640x480
image and text, 327 input tokens including 300 image tokens, and 16 generated
tokens. All 5760 full token outputs per model, including warmups, match the
same-backend reference exactly.

| Model | TTFT Mean ms | TTFT p99 ms | Full Generation Mean ms | Full Generation p99 ms | Decode tokens/s | Peak Allocated bytes |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | 152.937227 | 164.270234 | 558.793883 | 595.405434 | 36.958862 | 1805916160 |
| Qwen3.5-2B | 156.301746 | 171.681141 | 545.668130 | 592.475612 | 38.524127 | 4532012032 |

Both TTFT and total latency start at encoded image bytes in host RAM and include
image preprocessing/H2D, vision, prefill and generation; file IO, initialization
and detokenization are excluded. Diagnostic vision/forward timings are stored
separately. This is a fixed-image/prompt profile, not a multi-image formal test.

The earlier [cache-copy export failures](qwen35_native_blockers_20260910.md)
remain historical route evidence. They were superseded for this fixed profile by
the explicit-state adapter and accepted native bundles. The result does not
claim arbitrary prompt/image/generation support or a streaming per-token trace;
Orin/J6M/BPU remain deferred.

## Validation and Claim Limits

[Final validation](h20_final_validation_20260910.md) records the earlier 160
focused passes and 2259 full CPU passes, followed by the Qwen closeout regression
with 2275 passes, 62 skips and no failures after excluding one unavailable
external `openpi` fixture. It also records four collection-integrity rejection
tests. Focused cases overlap the full suite. All 305 production files match the
final frozen source provenance; report helpers remain separate from core runtime
code. The final integrity refresh rehashed 2935 evidence identities without GPU
reruns. The Qwen lifecycle audit independently rechecked all four report schemas,
bundle/library hashes, reset semantics and cross-Session output identities.

The [PDF claim review](h20_paper_scope_review_20260910.md) maps these measurements
to the draft. In particular, CogACT is the actual 7,630,224,071-parameter checkpoint
with public dependency configuration and sixteen frames from one Fractal episode.
Its raw-input formal campaign uses an archived external RNG tape, while the
separate autonomous native campaign uses the C++ `CudaRngProvider`; the two
boundaries must not be merged. SmolVLA uses recovered SO100 statistics; RDT retains the
qualified original camera profile and Torch 2.10 component environment; pi0 and
pi0.5 use sixteen frames from one ALOHA episode. Native action scales do not
establish physical robot calibration. Full-pipeline no-Python, zero-copy,
hard-real-time, universal low-precision fidelity, robot success and vendor/board
speedups are not inferred from these H20 results.

The current machine-readable Qwen closeout index is
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-evidence-index-002.json`,
SHA256
`c13c977de929a0cfadc403093e2aae1c45b74d3830186ed51e2c6657c1b2c4ae`.
It binds the Qwen campaigns, lifecycle audit, CDFs and document identity by
SHA256. The earlier global `final-evidence-index-001.json` predates the Qwen
native closeout and is retained as historical evidence, not as the current Qwen
index. CogACT's same-boundary closeout is indexed by
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/cogact-timing-boundary-evidence-index-001.json`.
Historical pilots, failed resource attempts and rejected exports remain separate
from formal results.

The requirement-by-requirement closeout for the current Goal is
`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/goal-completion-audit-002.json`.
It separates passed requirements from the explicit CogACT Meta-configuration,
fixed-profile Qwen, and deferred board limitations.
