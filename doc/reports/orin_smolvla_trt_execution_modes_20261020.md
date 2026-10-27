# SmolVLA Same-Engine TensorRT Execution Modes

Date: 2026-10-20. Status: complete tensor-boundary output equivalence passed
on real Orin; formal E2E/CDF and the complete five-model Goal remain unfinished.

## Experiment Boundary

Target is the same Jetson AGX Orin, serial `1421524312594`, L4T 35.4.1,
CUDA 11.4, TensorRT 8.5.2.2. No Torch/torchvision or Python model runtime
was installed or used on the board. All three executions use the same five
engines, 16 original frozen observation/noise records and ten solver steps.
Weights, numerical precision, full action horizon and output postprocessing
are unchanged. The per-engine SHA256 identities are checked against the
[original complete pilot](orin_smolvla_trt_native_pilot_20261020.md).

The baseline is the common native Session with ordinary per-Region stream/run/
synchronize, not an official PyTorch baseline or bare standalone `trtexec`
measurement. The other modes retain the same complete computation, validation
and transactional publication, changing only the runtime loop execution:

- `off`: ordinary per-Region execution; shared-context opt-in remains false.
- `batch-only`: runtime-owned shared CUDA stream and one bounded-loop drain,
  with stable carried/staging storage; no CUDA graph capture.
- `required`: a native CUDA Graph captures all ten solver steps and their
  device copies, using the same shared-stream contract. Failure cannot silently
  fall back to ordinary execution.

The original images/state/text have already been preprocessed in these runs.
Host-model-tensor timing includes H2D, binding, full computation/completion,
publication and complete normalized/native output D2H. Original-input CPU
preprocessing was not part of this tensor-boundary checkpoint; it is now
implemented in the later raw-input pilot. These tensor-only rows still must not
be relabeled as the requested paper E2E rows.

## Results

Each mode ran in its own process with 4 warmups + 16 measured invocations,
over all 16 saved observation/noise records. Warmups are also checked for full
output correctness. These are pilots, not 5 x (128 + 1024) formal sampling.

| Mode | Mean ms | p50 ms | p95 ms | Max ms | Complete output equivalence |
|---|---:|---:|---:|---:|---|
| Ordinary | 121.644062 | 121.583328 | 122.784696 | 122.788320 | Both actions byte exact |
| Shared-stream batch | 121.192102 | 121.119264 | 122.434224 | 123.166080 | Both actions byte exact |
| Full 10-step replay | 102.090680 | 101.984672 | 103.303120 | 103.853344 | Both actions byte exact |

All 120 complete FLOAT `[1,50,6]` tensors (3 modes x 20 calls x 2 outputs)
match the frozen Orin conventional pilot byte exactly, including every
normalized and native coordinate. Same-target MSE/RMSE/max-abs are zero;
cosine is one. This is equivalence to the same-engine Orin baseline, not proof
of equality to the original H20/BF16 computation or physical robot quality.

The independent audit checks raw binary sizes/hashes, dtype/shape, complete
sample mapping, every warmup/measured output and all timing segments. No slow
sample is removed. Mixed-schema `REPLAY` records in the raw stdout CSV are
separately parsed and validated; they are telemetry, not discarded latency
samples. All 20 timing rows remain included per mode.

Required replay telemetry is `REPLAY_FINAL,11,1,10,20,0`: loop task 11,
ready state, all ten steps captured, 20 replays and zero ordinary invocations.
Every per-call telemetry record also matches its monotonically increasing
count, including warmups. Batch mode reports zero replays and 20 ordinary
bounded-loop invocations. This is real target capture evidence, not only
compiler eligibility or a single-step engine benchmark.

The frozen H20 reference comparison is unchanged: worst normalized MSE is
6.393668376929344e-5, which still fails `<1e-5`; minimum cosine remains
0.9999852940845904. Native-unit errors are retained separately in the audit.
No quantization search, precision change or relaxed tolerance produced these
execution-mode results.

## Generic Implementation

The TensorRT backend exposes the existing optional execution-context sidecar;
the v1/v2 Region ABI tables remain unchanged. A Region borrows a runtime-owned
stream, rejects context changes/reload while pending, clears pending state only
after synchronization, and never destroys a borrowed stream. Synchronizing
during capture is rejected without invalidating a valid capture. Failed drain
quarantines engine resources instead of releasing storage still in use.

The new native CUDA Graph provider uses CUDA runtime graph APIs and no LibTorch
allocator/RNG integration. CPU-only builds return no provider. Unproven capture/
launch/completion quarantines graph resources and is not an ordinary fallback.
Code generation selects provider headers by backend, while TensorRT shared
context and graph support are explicit artifact opt-ins. Graph support requires
shared context, an audited pure loop, fixed staging and actual runtime capture.
No model-name dispatch or model-specific CUDA kernel was introduced.

For this concrete step graph, the preparation checks the frozen source ONNX
SHA256 and audits every node against an explicit conservative set of functional
standard ONNX operations. Custom domains, unreviewed operators, nested functions/
graphs and training programs are rejected. This is a source-effect audit, not
a universal claim that all ONNX or all TensorRT plugins are capture safe.

## Verification

- Actual target runtime CTest: 17/17 passed. Covers ordinary/static/dynamic TRT,
  stream binding, wrong-device/ABI rejection, pending detach/reload rejection,
  real CUDA graph capture/launch/abort and return to a private stream.
- Host native runtime CTest: 12/12 passed; CPU build does not simulate CUDA graphs.
- Code-generation focused regression: 29 passed, 2 explicit GPU opt-ins skipped.
- Expanded host regression: 523 passed, 40 explicit CUDA/LibTorch opt-ins skipped.
  Those host skips do not replace the separate actual Orin TRT tests.
- Native model build logs and process maps bind AArch64 TensorRT/CUDA execution;
  inspected maps/dependencies contain no Torch or Python model DSO.

The earlier ordinary-mode NaN/Inf, transactional retention, Session isolation
and reset probes remain valid for that frozen ordinary build. They have not
yet been repeated against the new shared/replay builds. Repeated lifecycle,
long-run memory/allocation behavior and a 30-minute thermal soak remain open.

## Evidence and Reproduction

All local paths below are beneath `artifacts/orin/20261020-00/`:

| Evidence | Location |
|---|---|
| Frozen native runtime archive | source/trt-context-v2-source.tar |
| Target build/test record and JUnit | trt-context-runtime-001/build.json, ctest.xml, logs and snapshots |
| Exact frozen archive SHA256 | 471605a369fb17000986138ae7e2a49e53210c64c10f2dd01bfc7f9a6dc314f0 |
| Three-mode controller/argv/PIDs | execution-modes-001/campaign.json |
| Independent full audit and all sample errors | execution-modes-001/audit.json |
| Ordinary model sources/build/pilot | native-mode-off-002/ |
| Batch model sources/build/pilot | native-mode-batch-002/ |
| Replay model sources/build/pilot | native-mode-replay-002/ |
| Expanded host JUnit | audit/trt-context-regression-expanded-001.xml |

Remote runtime/source prefix is
`/data/datalog/edgefm-orin/20261020-00/trt-context-runtime-001/`.
Remote model prefixes are beneath
`/data/datalog/edgefm-orin/20261020-00/trt-smolvla-001/` with the same mode
directory names. All controller/build/model child processes exited; stale PID
fields identify historical terminal processes, not active jobs.

Preparation is model-neutral and source-driven:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python3 \
  artifacts/orin/20261020-00/prepare_trt_execution_mode.py \
  --base artifacts/orin/20261020-00/native-session-001 \
  --source-spec artifacts/orin/20261020-00/native-session-001.spec.json \
  --pilot artifacts/orin/20261020-00/native-session-001/pilot-001 \
  --reference artifacts/orin/20261020-00/references/smolvla-h20-001/manifest.json \
  --mode required --output artifacts/orin/20261020-00/native-mode-replay-new
```

Use a new directory for every reproduction. `build_trt_context_runtime.py`,
`run_trt_execution_modes.py` and `audit_trt_execution_modes.py` preserve build,
runner and audit commands/hashes. Earlier `native-mode-*-001` preparation
attempts failed before a metadata digest method was called correctly; no board
model run occurred for them. The later auditor initially treated stdout replay
telemetry as latency rows; the corrected audit validates both complete schemas
without changing any original CSV or model output.

The subsequent [original-observation E2E pilot](orin_smolvla_native_raw_e2e_20261020.md)
implements and verifies native preprocessing and complete action equivalence.
Before paper sampling: repeat runtime fault/lifecycle tests for these modes,
finish attribution/qualification of the cross-device
precision discrepancy, then freeze formal sampling, ownership/clock policy,
telemetry and power/thermal measurements. The other four required VLA models
and optional Qwen have not been silently removed or marked complete.
