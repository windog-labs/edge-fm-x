# SmolVLA Orin Source Qualification

Updated: 2026-10-21, Asia/Shanghai. Status: **source-qualified complete-model
profile with independently audited formal E2E/CDF**. See the
[formal results](orin_smolvla_formal_e2e_20261021.md).
Four other required VLA models and operator/runtime/power tasks remain open.

## Boundaries

- Device: actual Jetson AGX Orin, serial `1421524312594`, system TensorRT 8.5.2.2,
  CUDA 11.4 and L4T 35.4.1. No target Torch/torchvision or Python model execution.
- Complete fresh prefix plus initialization, all ten original solver steps,
  finish and native-unit output. Original checkpoint, 16 observations, saved
  noise, horizon and external IO are unchanged.
- The source reference is the frozen FP32-interface ONNX deployment profile.
  It is not relabeled as the original BF16 H20 profile. High cosine does not
  establish mathematical losslessness, robot task success or original-profile
  equivalence.
- Normalized complete action uses the frozen aggregate gate: MSE `<1e-5` and
  cosine `>0.9999`. Native-unit and per-dimension errors remain separate.
  No tolerance was relaxed and no quantization search was performed.

## Generic Implementation

`vlaforge/tools/fuse_onnx_layernorm_trt_plugin.py` matches static last-axis
FLOAT LayerNorm dataflow and replaces it with the existing TensorRT LayerNorm
plugin. It does not select model or node names. Axes, epsilon, dtype, scale,
offset, leaked intermediates and static shapes are checked conservatively.
Other ranks are flattened/restored explicitly; unmatched nodes and weights are
preserved. A standard-ONNX shadow path enables CPU semantic checks.

The complete real prefix contains 50 matching groups with shape
`[1,1024,768]`. All initializers, unmatched nodes and external interfaces are
byte exact. CPU shadow validation of all 66 cache tensors on observations 6/11
has maximum absolute error `3.123283386230469e-5` and worst MSE
`6.828624314180184e-12`; BOOL tensors are exact. This shadow alone is not board
qualification.

The opt-in `VLAFORGE_TENSORRT_STANDARD_PLUGINS=ON` backend option links the
system plugin library and registers it once before deserializing engines.
Default behavior is unchanged. On-device plugin tests build in a fresh child
process and deserialize in the parent through the actual backend, avoiding
inherited registration as a false positive. Actual target CTests: **20/20**.
Failed build/test snapshots 001--004 remain preserved; the successful runtime
is `native-standard-plugins-build-005`.

The adopted profile composes this prefix with the previously extracted
original FLOAT solver tail through typed IR/Plan/Session partitioning. Dense
constraints remain unchanged: the prefix's 316 remaining dense requests stay
HALF, its declared key/patch islands stay FLOAT, and the step core's 148 dense
requests stay HALF. The original two FLOAT solver arithmetic operations execute
in a separate tail engine. This is not whole-model FP32 fallback.

## Frozen Engine Identity

All three execution modes use these exact seven serialized engines:

| Engine | SHA256 |
|---|---|
| Initialize | `613351ddc8fd7b64d548e39d6a940562c93b1440ba8c8ef1b49bfdd7e5728c3c` |
| Finish | `6ee9409be2217958b61eb7fb400cd1f062fd4db243be527153c8c9e673582951` |
| Native output | `6d37341f66e14991c09aadf16e5368821701860a45d40a9d8877a80c7bb7264f` |
| Prefix phase | `bbc5b499190190eab4e762213a9bb37b0c6ce13b3cc9b2206ec7e5b90c730ead` |
| Prefix neural | `e3eaeaf3988b0a0ff919173275213c24a5fa9b3e27e156329135ddda72ae8ab3` |
| Step numeric core | `bb2ebd05e926ac037daeb87f4862303670273ffe3049d1439599a9cf97c12953` |
| Step original FLOAT tail | `457423a165a4d8ac2c455df46998ee43b41e894453bebabad4b2e22f070c7c4d` |

The new prefix engine is 657852105 bytes. Its inspector shows all 50 plugin
layers have actual linear FP32 IO. Build elapsed time is 156.367401824 seconds.
The system plugin DSO SHA256 is
`5a30c8e46ba47c3a93eb825ad0a27cbe969fb6f174ea963e4d22dfe33630097a`.

## Full Action Qualification

Evidence root: `artifacts/orin/20261020-00/`.

| Comparison | Worst Normalized MSE | Minimum Cosine | Outcome |
|---|---:|---:|---|
| Seven-engine native pilot vs frozen FP32 source, all 16 observations | 8.742859120188182e-6 | 0.9999974812152786 | Passed frozen aggregate gate |
| Same seven-engine modes vs qualified target pilot | 0 | 1 | All 120 complete action tensors byte exact per campaign |
| Seven-engine pilot vs original BF16 H20 reference | 9.907372960652268e-5 | 0.9999815097392623 | Supplemental MSE failed; retained |

Source-relative worst normalized sample is 5; maximum absolute error is
`0.012404084205627441`. Sample 6 MSE is `8.732485197547323e-6`.
Worst native-unit source MSE is `0.02188867886544287`, not compared against a
normalized-unit threshold. All full outputs and per-dimension metrics remain
in the independent complete-output reports.

The earlier plugin-only six-engine trial still failed 1/16 samples (sample 6,
MSE `1.0504458316824191e-5`). Cross-feed with its actual board prefix and CPU
steps passed that diagnostic sample, providing evidence to compose the original
solver-tail partition. The failed trial is not overwritten or promoted.

Key proof files:

- `audit/source-gate-layernorm-solver-full16-001.json`
- `native-layernorm-solver-001/pilot-001/{pilot.json,samples.csv,outputs.f32,output-1.f32}`
- `audit/layernorm-solver-complete-output-001/{source,h20}.json`
- `qualified-execution-modes-001/{campaign,audit}.json`
- `qualified-raw-e2e-pilots-001/{campaign,audit}.json`
- `native-standard-plugins-build-005/{build.json,ctest.xml}`

## Original-Observation Pilot

All three modes ran in separate processes, each 4 warmup + 16 measured requests.
They start from CPU decoded RGB/native state/text plus saved noise and include
online native preprocessing, H2D, complete fresh model execution, GPU completion,
postprocessing and complete CPU normalized/native actions. No precomputed
vision/language features are substituted. Each timing row conserves its E2E
boundary. All 120 full output tensors are byte exact with the source-qualified
target pilot, and all ten solver steps replay with zero ordinary fallback.

| Mode | Pilot E2E Mean (ms) | Measured Requests |
|---|---:|---:|
| Conventional TRT | 131.778306 | 16 |
| Shared-stream batch | 131.321208 | 16 |
| Required full-loop replay | 111.328476 | 16 |

These are short pilots, not final paper estimates or a CDF. Do not compare
them to older five-/six-engine candidates to calculate acceleration.

## Formal Protocol

Resumed after power-off on 2026-10-21. Rebooted GPU bench clients were stopped
under the existing explicit authorization, after checking PID/start time/argv.
Unrelated services were preserved. Board civil time remains incorrect; use
monotonic durations and the recorded boot ID.

New root: `artifacts/orin/20261021-00/smolvla-raw-formal-001/`.
Three configurations each use five independent sequential processes with
128 warmup + 1024 measured requests, batch 1. Mode order rotates between rounds;
the fixed 16 real observations repeat, not independent trajectories. The exact
qualified binaries, source gate, input/preprocessor/tokenizer identities and
seven engine hashes are frozen before execution. No slow rows are removed.
Preserve all complete action bytes, every timing row, replay counters, maps,
environment before/after, ownership checks, sysfs telemetry and tegrastats.

Formal sampling and its independent audit are complete: all 15 processes and
34560 complete action tensors pass. Means: 134.409630 ms ordinary, 133.067807 ms
shared batch and 113.139929 ms replay, a 1.187995x same-engine runtime speedup.
Pooled replay std 2.034434 ms exceeds the draft 1.2 ms target; all rows remain.
The separate bounded fault/reset/isolation and three-cycle lifecycle checks also
pass, but observed ambient free-memory drops remain unqualified. The subsequent
[35-minute soak and resource follow-up](orin_smolvla_soak_memory_20261021.md)
now supplies a flat sampled RSS envelope, covered CUDART accounting and a
partial-window power estimate. Exhaustive private driver/SDK attribution,
operator/Agent ablation, alternate power modes and other models remain open.

CPU regression from the qualified implementation snapshot: 2503 passed,
74 explicitly skipped and six subtests passed; evidence
`audit/layernorm-plugin-regression-expanded-002.xml`. These are host tests,
not target PyTorch installation or additional performance experiments.
