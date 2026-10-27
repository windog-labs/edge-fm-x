> 可移植报告副本：保留原报告时点；最新结论以根目录总汇总为准。未随包携带的历史链接已转为文字说明；本包不提供原机执行命令或连接信息。

# Orin LayerNorm Complete-Model E2E Ablation

Date: 2026-10-22, Asia/Shanghai. **The complete experiment is audited; the
performance result is negative.** Quality qualification is not a speedup claim.
The five-model Goal remains active and incomplete.

2026-10-24reconciliation: this complete representative integration experiment
fulfills the integration/remeasurement subtask despite negative performance.
Positive E2E gain is unsupported, not an additional prerequisite for experiment
completion. Fresh raw/source/resource re-audits preserve all results; the current
rewrite-tool identity differs explicitly while the same frozen graph reproduces.
See[reconciliation](orin_integration_requirement_reconciliation_20261024.md).
Other operator/resource requirements and the five-model Goal remain incomplete.

## Result

| Metric | Original Prefix Normalization | FP32 Norm Plugin Prefix |
|---|---:|---:|
| Measured records | 5120 | 5120 |
| Mean latency (ms) | 125.900306 | 133.994095 |
| p50 (ms) | 125.233728 | 132.956416 |
| p95 (ms) | 128.758525 | 144.546314 |
| p99 (ms) | 142.700808 | 150.914833 |
| Maximum (ms) | 157.419936 | 169.750976 |
| Pooled std (ms) | 3.255075 | 4.100485 |
| Complete chunks/s | 7.942792 | 7.463015 |

Ratio of pooled latency means is **0.93959593x**, or **6.4287% slower** for the
candidate. The five pair ratios are 0.940075, 0.937722, 0.940043, 0.941316 and
0.938824: all below one. No significance or independent-trajectory claim is
made from repeated timing samples. Means, tails and std are all retained.
The standalone Norm resident 4.329x result therefore does **not** persist as
an E2E gain in this complete graph. Fusion/tactic differences may matter, but
this experiment alone does not isolate their causal contributions.

This is a controlled graph-ablation result, not a new runtime replay benchmark.
The existing frozen main SmolVLA profile and its prior ordinary/shared/replay
results are unchanged. No default engine was silently replaced based on this
negative result and the micro ratio is not multiplied into the main table.

## Complete Outputs and Accepted Policy

The user explicitly confirmed: use complete **normalized action** for the
unchanged MSE < 1e-5 / cosine > 0.9999 gate; report native-unit errors separately.

| Complete [1,50,6] Output | Minimum Cosine | Maximum MSE | Maximum Absolute Error | Role |
|---|---:|---:|---:|---|
| Normalized action | 0.999998499934 | 5.30081810e-6 | 0.00802004337 | Primary gate passes |
| Native action | 0.999998119753 | 0.0121000582 | 0.353607178 | Supplemental; same numerical MSE threshold not met |

Every one of the **23040 output tensors** from **11520 complete calls**,
including warmup, is independently byte-exact to its own variant's frozen
tensor-input reference. The two graph variants are **not** byte-exact to one
another. Cross-variant errors above cover all 16 distinct observations; repeated
calls cannot add new trajectory diversity. Complete per-sample/per-dimension
metrics remain in the quality receipts, including any dimension-level threshold
failures. The primary gate is the full normalized chunk, not a new assertion
that every per-dimension cosine exceeds that threshold.

The old conservative `same-target-complete-actions-failed` receipt that applied
the same numerical threshold to both spaces remains untouched. A separate
hash-bound acceptance receipt records the user's decision. No threshold, raw
native error, precision, solver iteration or noise was changed. High cosine
does not prove mathematical losslessness, physical action safety or robot
task success. GEMM still fails its normalized gate and is not requalified.

## Frozen Controls and Boundary

- Both variants use ordinary scheduling (`off`), identical full semantic IR,
  scheduled Plan, input/noise records, host preprocessing and output contracts.
  Seven engines and all ten solver iterations execute on Orin for every call.
- Only the neural prefix engine differs. Generic source rewriting replaces
  450 original FLOAT primitive nodes with 50 FLOAT Norm plugins; other weights,
  nodes and precision declarations stay unchanged. This is the original FP32
  numerical island, not whole-model FP32 fallback or quantization search.
- Matching builds use FP16/global, noTF32 and 2048 MiB workspace. Forty-three
  prefix binding names/order and all exposed descriptors agree. Three missing
  inspector descriptors remain documented; actual native binding guards pass.
- Boundary: CPU already-decoded RGB/native state/text/noise through online native
  preparation, H2D/binding, complete vision/language/action computation, native
  output conversion, actual completion and full CPU action. Initialization,
  file decoding, output writing and validation are outside timing. Segment
  conservation is checked; no stage-mean sum substitutes for E2E.
- Each configuration has five independent processes, each 128 warmup + 1024
  measured, with AB/BA/AB/BA/AB order. All 10240 measured rows and 1280 warmup
  rows remain. No trimming, process centering or protocol reduction occurs.
- All 128 preprocessed input tensors are bound to the prior raw-input audit and
  exactly match the paired tensor-input model contract, including actual noise.
  The repeated cycle is 16 recorded observations, not thousands of new trials.

## Device and Resources

Serial `1421524312594`, boot `3c801f5d-6897-4d97-b021-3c52db38a750`;
TRT 8.5.2.2, CUDA 11.4, L4T 35.4.1, existing MAXN unchanged. Orin civil clock
is wrong; host UTC checkpoints and native monotonic intervals are used.
All ten native processes exit zero. Terminal checks find the owned collector
and workers no longer present; no inference process is left running.

Minimum admission MemAvailable is 12140180 KiB. Minimum sampled system
MemAvailable is 9767304 KiB (9.315 GiB), above the fixed 2 GiB floor. Maximum
sampled native process RSS is 3151144 KiB; observed process swap is zero.
Linux process RSS is not isolated physical GPU memory, and shared CPU/GPU
memory is not added twice. Snapshot-visible GPU owners are not proof of full
machine exclusivity; permission-denied process visibility remains documented.

Across 1380 telemetry samples per thermal channel, CPU reads 46--56.25 C and
GPU 40.656--54.718 C. CV0/1/2 each report raw -256 C; those readings are retained
as unqualified, not interpreted as valid temperatures or a proven SDK sentinel.
The original raw-summary receipt also remains. Existing Java services, power,
SDK and clocks were not changed. No new matched power-efficiency, leak-free,
unobserved-peak or causal resource claim is made. Controller duration
1597.550 s includes setup/monitoring/logging and is not model latency.

## Evidence and Reproduction

Root: `artifacts/orin/20261022-00/`.

| Evidence | Location |
|---|---|
| Original complete-action quality and unchanged conservative failure | `norm-full-model-pair-001/quality-001.json` |
| User-confirmed primary acceptance | `norm-full-model-pair-001/primary-acceptance-001.json` |
| Source/precision/ABI pairing | `norm-integration-source-audit-002.json` |
| Raw pilots and complete-output audit | `norm-raw-e2e-pair-003/pilot-001/`, `pilot-audit-002/audit.json` |
| Exact formal request, native binaries and safety-helper hashes | `norm-raw-e2e-pair-003/formal-request-002.json` |
| All target files, CSV, full action bytes, maps, telemetry | `norm-raw-e2e-pair-003/formal-001/` |
| Independent final audit and all raw timing rows | `norm-raw-e2e-pair-003/formal-audit-001/` |
| Full-tail CDF PNG/PDF and hashes | `norm-raw-e2e-pair-003/formal-cdf-001/` |
| Qualified resource summary, invalid readings retained separately | `norm-raw-e2e-pair-003/formal-resources-002.json` |
| Collector/worker termination checkpoint | `norm-raw-e2e-pair-003/live-snapshot-terminal-001.json` |
| All operator/integration regression | `all-operator-and-integration-regression-002.xml` |

Verification: 141 focused operator/integration tests, six board state/server
tests and nine browser tests pass. This is not a freshly rerun full-source
test-suite claim. The final browser receipt is
`artifacts/orin-task-board/norm-raw-final-board-green-006.xml`.

Original prefix engine SHA256:
`fca6c023ad5a72a381f3c2fb83d0f535df369c7279bf27334b7b2c93d30f3927`.
Candidate SHA256:
`e3eaeaf3988b0a0ff919173275213c24a5fa9b3e27e156329135ddda72ae8ab3`.
The formal request SHA256 is
`ebc0aab6f3495621efdbb9fac3d112195e230d69addd80159821a05451cda9a6`.

Re-audit retained files without target computation, using a fresh output:

> 原机执行命令不属于论文交接内容，已省略。

The earlier incorrect-source-IR preparation and template/header compile failures
are retained in `norm-raw-e2e-pair-001/` and `norm-raw-e2e-pair-002/`. The
successful path reuses a matching, previously qualified generic native runner,
not a model-name-specialized execution path. RDT is not restarted; Embedding,
RoPE, effective positive-gain integration and the remaining required models
are still open.
