> 可移植报告副本：保留原报告时点；最新结论以根目录总汇总为准。未随包携带的历史链接已转为文字说明；本包不提供原机执行命令或连接信息。

# SmolVLA Orin Formal E2E

Date: 2026-10-21, Asia/Shanghai.

**Status: formal same-engine E2E/CDF and bounded functional fault/lifecycle checks passed. Memory reclamation remains unqualified.**
This is one deployment profile, not the complete five-model paper task.

## Protocol

- Real Jetson AGX Orin, serial `1421524312594`, system TensorRT 8.5.2.2, CUDA 11.4 and L4T 35.4.1.
- Existing MAXN and CPU governor policy, no global SDK/clock/power changes; MAXN is not labeled 60W.
- Batch 1, five independent sequential processes per mode, 128 warmup + 1024 measured per process.
- Direct per-request timing from decoded CPU RGB/native state/text through online native preprocessing, transfers, complete fresh context/all ten solver steps, postprocessing and complete CPU actions.
- Exact same seven engine hashes, weights, observations, saved noise and precision across modes; no target Torch/Python model execution.
- File/image decoding, initialization, output checking and evidence serialization are outside timing.
- Sixteen recorded observations repeat. This is not 15360 independent trajectories, action-queue throughput or physical control frequency.
- No slow rows removed. Mode order rotates across rounds; sysfs, process maps, visible ownership and tegrastats are preserved.

## Formal Results

| Mode | Requests | Mean (ms) | p50 | p95 | p99 | Max | Std | Chunks/s | Runtime Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Conventional TRT | 5120 | 134.409630 | 134.349024 | 136.952802 | 138.496310 | 142.241472 | 1.509459 | 7.439943 | 1.000000x |
| Shared-stream batch | 5120 | 133.067807 | 133.085408 | 135.528658 | 136.765336 | 138.972224 | 1.492815 | 7.514966 | 1.010084x |
| Full-loop replay | 5120 | 113.139929 | 113.600224 | 115.987411 | 117.187907 | 121.009632 | 2.034434 | 8.838613 | 1.187995x |

Speedup uses same-platform conventional TRT mean divided by the mode mean. It is not a comparison to TensorRT-Edge-LLM, native PyTorch, H20 or a different engine/precision profile.

The pooled replay standard deviation is 2.034434 ms, above the draft target of 1.2 ms and above the conventional pooled standard deviation of 1.509459 ms. Its lower p99/max is not a proof that every jitter metric improves. Process means are retained below without centering, trimming or discarding the slower rounds.

CDF files: `smolvla-e2e-cdf.png` and `smolvla-e2e-cdf.pdf` beside the formal audit.

A later renderer validation reuses these exact audited request rows, adding
pooled and independent-process CDF PNG/PDF under
`artifacts/orin/20261021-07/smolvla-formal-cdf-check-001/`. Its plot receipt binds
the original audit and CSV hashes, all 5120 measured requests per mode, original
min/max and rendering versions. No new inference or changed statistics are
introduced. The independent process plots retain the replay process shifts
visible below; they do not center each process to reduce pooled variability.

### Process Means

| Mode | Independent Process Means (ms) |
|---|---|
| Conventional TRT | 133.849938, 134.619833, 134.861904, 134.581958, 134.134515 |
| Shared-stream batch | 133.817164, 133.382013, 132.028149, 132.331076, 133.780636 |
| Full-loop replay | 114.224336, 114.863861, 114.383047, 110.511248, 111.717151 |

## Complete Action Fidelity

- All 34560 complete normalized/native action tensors, including warmups, are byte exact with the source-qualified target baseline.
- Formal measured calls: 15360 total (5120 per mode); two complete action outputs per call, each `[1,50,6]` FLOAT.
- Same-target optimized-vs-conventional cosine 1, MSE 0, max-abs 0; every saved tensor is finite.
- Source qualification has 16 distinct observations; the normalized aggregate gate stays MSE <1e-5 and cosine >0.9999.
- Worst source-relative normalized MSE 8.7428591201881816e-06; minimum cosine 0.99999748121527865.
- Original BF16 H20 supplemental MSE remains failed. The frozen FP32-interface deployment reference is not relabeled as that original profile, and native-unit/per-dimension errors remain separate.

## Runtime Checks

- Required replay captures the complete ten-step loop and reports zero ordinary fallback on every formal call.
- Three additional paired-Session cycles per mode: 18 Sessions created/destroyed, 135 full-model calls, 54 rejected NaN/Inf transactions.
- Failed transactions retain complete previous outputs, publish nothing and recover exactly; separate Session publications/counters remain isolated.
- Reset invalidates publication and rejects unbound reuse; rebinding and recapture recover byte-exact actions.
- Three-cycle CUDA ambient free-memory readings are preserved, not interpreted as a leak certificate or precise per-Session accounting.

Last two lifecycle cycles show ambient free-memory deltas of Conventional TRT: -115.349 MiB, Shared-stream batch: -79.563 MiB, Full-loop replay: -81.779 MiB. These negative observations are not hidden, but do not isolate an allocator leak from host/runtime caching or unrelated shared-memory users; memory reclamation remains unqualified.

## Resources and Limits

- Largest sampled process VmHWM: 3114.137 MiB; largest valid thermal reading: 57.187 C.
- Shared-memory Jetson CPU/GPU usages are not added together. Sampled process HWM/RSS is not an instrumented device/host allocation peak.
- Visible process ownership is monitored, but inaccessible FD directories remain a coverage limitation. Unrelated background services are preserved.
- Board civil clock is incorrect; duration accounting uses monotonic clocks and recorded boot ID.
- Raw power-rail readings and tegrastats are retained without inventing a power aggregate or energy estimate.
- This sequential multi-process campaign does not replace a continuous >=30-minute same-worker soak.
- Operator/Agent tuning ablations, exhaustive private-SDK/driver allocation accounting, alternate power mode and the remaining required models are still open.

The subsequent 35-minute soak and resource follow-up（原记录未随包携带，见未打包引用清单）
has now completed. It reports flat observed steady RSS, balanced covered CUDART
resources and a partial-window input-rail power estimate. Its one-worker std
1.060070 ms is a different statistic from the 2.034434 ms formal pooled std;
neither replaces the other. Private driver/SDK attribution and other models
remain open. The original main-table numbers above are unchanged.

## Evidence

- Formal audit: `artifacts/orin/20261021-00/smolvla-raw-formal-001/audit-output/audit.json`; SHA256 `7c7d5d103232a2b6b0871abc8e081be18f8ac0b2120b07152563c9fe3dddb3c7`.
- Runtime audit: `artifacts/orin/20261021-00/qualified-session-probes-001/audit.json`; SHA256 `355e8bfd8733d246de77287ee21fdcdd050e2a25d9c77c2569fcca5e35273bd0`.
- Source gate: `artifacts/orin/20261020-00/audit/source-gate-layernorm-solver-full16-001.json`; SHA256 `a4b7b05bfc28e7b7cee339bc37f0ea44a6e3480a228ddcfaa147dfbc898eb49d`.
- Every request is retained in `all-requests.csv`; original process timing/action/telemetry files and their hashes remain in the campaign.
- Source qualification and manuscript limitations: `orin_smolvla_trt_source_qualification_20261021.md` and `orin_claim_gap_20261021.md`.
