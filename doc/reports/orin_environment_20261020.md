# Orin Target Backend Compatibility

Date: 2026-10-20. This closes the first native backend smoke step, not the
five-model VLA performance objective. The full Goal remains active.

The user subsequently selected TensorRT-only Orin execution. The Torch build
and its dependency download were stopped with process-identity evidence;
their caches and failures are preserved below as history, not prerequisites.
Orin will run native C++/TensorRT, with model export/reference preparation on
existing host environments. Same-platform comparisons are conventional TRT
versus EdgeFM + TRT, not Orin PyTorch baselines.

## Hardware and Resource Release

`ssh orin` reaches a Jetson AGX Orin Developer Kit, serial `1421524312594`,
running L4T 35.4.1, CUDA 11.4, cuDNN 8.6 and TensorRT 8.5.2.2. The board has
approximately 62,800 MiB shared RAM and a writable NVMe at `/data/datalog`.

The user expressly authorized clearing the bench. The exact application
process identities and their process manager were recorded before SIGTERM;
the managed GPU viewer was stopped through its supervisor. The terminal
check and actions are in `environment/bench-release.json`. Visible GPU device
users after release were two GNOME processes. Root-owned FD directories were
not fully readable, so this is not an exclusive-owner attestation.

## Actual Baseline Failure

The frozen `aa5558f` source archive was copied to an independent target
directory. CMake configuration succeeded; compilation exited 2 because the
installed TensorRT 8.5 headers lack `kBF16`, `kINT64`, `kFP8` and `kINT4`.
The original errors and source bytes are retained rather than overwritten.

The generic backend adaptation now:

- gates BF16/INT64 type mappings on TensorRT >=9 and rejects unsupported types;
- validates dimensions against the compiled TensorRT extent type before narrowing;
- uses `inferShapes` for named input bindings before resolving outputs;
- builds explicit-batch smoke engines on TensorRT 8 and derives runtime metadata
  from actual SDK version macros;
- tests static/dynamic named shapes, wrong dtype rejection and TRT 8 dimension
  overflow rejection through the existing on-device smoke.

This does not silently convert model inputs, add model-name branches or claim
TensorRT 8 can execute all types available in newer releases.

## Executed Target Validation

| Step | Actual result |
|---|---|
| Original source native configure/build | Configure passed; build failed on missing dtype enums |
| Patched source native configure/build | Passed with the actual target headers and libraries |
| Target CTest | 15/15 passed, 7.42 s reported by CTest |
| CUDA device copy smoke | Passed |
| TensorRT value ABI/invalid create arguments | Passed |
| TensorRT static identity and metadata rejection | Passed |
| TensorRT dynamic identity and named shape inference | Passed |
| Native ELF/library identity | AArch64, `libnvinfer.so.8`, `libcudart.so.11.0` and target `libcuda.so.1`; no Python dependency in inspected ldd output |
| Local related artifact/board contract tests | 110 passed |

The identity engine is a backend test fixture. Its execution is not a model
latency measurement or a full VLA action-quality check. TensorRT 10 regression
has not been executed in this step.

## Clock and Build Tools

The target clock reports March 2023 while the collecting host reports
2026-10-20. The clock was left unchanged. Initial build logs expose this skew;
only the extracted patched source copy had file mtimes normalized. Durations
use monotonic clocks; source and target wall timestamps must be kept distinct.

The default `ctest` command was missing from PATH. The existing CMake 4.3.1
installation was found beneath the board's Python 3.8 environment. Tests ran
using its actual binary, without installing a replacement system tool:

```sh
/home/xingshen001/workspace/edge-fm-x/.venv/lib/python3.8/site-packages/cmake/data/bin/ctest \
  --test-dir /data/datalog/edgefm-orin/20261020-00/build-trt85-v1 \
  --output-on-failure --parallel 1
```

## Evidence and Remaining Work

Local evidence root: `artifacts/orin/20261020-00/`.
Remote evidence root: `/data/datalog/edgefm-orin/20261020-00/`.

- `source/vlaforge-aa5558f.tar`: baseline SHA256
  `2a1ce91d809993f34984c31822b0e9fac18379eab66603b8951f76745c70d3cb`.
- `source/trt85-compat-files.tar`: patch SHA256
  `50b31ba4745f392e88d405fb6c2e51df31ce0b5d957ba0780de89689a25d0c8f`.
- `build/baseline-build*.log`, `build/trt85-v1-build*.log`: original and patched builds.
- `build/trt85-v1-ctest-002.{log,json}`: actual native test command/exit/duration;
  log SHA256 `9d89a866873d1a7ce396b354cc4d87ac44a2b14cad581f0edfa2d9f8246248e1`.
- `environment/target-snapshot*.json`: hardware, versions, clock discrepancy,
  visible GPU owners, permissions and collection timestamps.
- `environment/smolvla-asset-verification.json` and
  `environment/smolvla-raw-input-verification.json`: staged model assets and
  independent target checks of the new 16-sample original-observation pack.

Historical PyTorch preparation (no longer an active gate): pinned LeRobot requires
Python >=3.10 and a newer Torch/torchvision pair than the default SDK's wheel.
Isolated Python 3.11.14 was built on the NVMe and passed SSL/ctypes/bz2/lzma/
sqlite3/zlib/venv checks; the original Python was not replaced. Build commands
and their hashes are in `build/python311-build.json`. A real
SmolVLA conventional/optimized TRT pilot, all five-model formal metrics and subsequent
operator/runtime/power experiments remain pending.

## TRT-Only Model Preparation

Real SmolVLA prefix trials use native system `trtexec --fp16 --noTF32 --buildOnly`.
The first preserved failure is missing LayerNormalization support. Reusing
the generic decomposition and a conservative opset conversion exposed the next
real failure: dynamic Slice axes. Host ORT basic static-shape folding removes
those graph constructs, and `prefix-build-004` built a 596080746-byte engine.
Real-input execution returned all outputs but failed prefix fidelity severely
(max-abs 15.33856201171875, minimum cosine 0.10738510871175629). Its timing is
not accepted as a performance result.

`prefix-build-005` uses FP16 matrix/conv with explicit FP32 arithmetic and
reductions, not a full FP32 model fallback. Its 599997225-byte engine restores
cache nonzero coverage on the same input; remaining max-abs 0.20618629455566406
and minimum cosine 0.999856929760148 against the declared FP32 ONNX source
remain unqualified. Full action fidelity and baseline/optimized comparison
are still required. At this earlier checkpoint the step engine had built (200603483 bytes,
SHA256 `7c3389406c52e6724efa726c3408f57b41aeddd1f59603a585eaee21b40867fe`),
without inference acceptance. The later complete native Session execution and
independent action audit are recorded below; the earlier build alone remains
distinct from inference qualification.

All 16 saved observations were used for host source-versus-rewrite comparison,
including all 33 outputs per observation. All 528 complete tensors passed the
declared graph-only tolerance (max-abs 1e-3, RMSE 1e-4, exact bool/integer).
Observed max-abs is 3.966689109802246e-5 and worst RMSE 2.3245896541825294e-6.
The reference is the explicitly declared old FP32-interface source ONNX,
not the original BF16 model. Full Orin action qualification is still required.

The new host-side opset conversion tool does not change tensor precision or
external weight bytes, rejects unsupported/dynamic schema forms, and has been
checked together with the existing lowering passes: 37 tests passed.
The expanded CPU regression including deployment/board contracts has 147 passes.
The host report/output archive is `host-parity/prefix-001/`; target build
records are `trt-smolvla-001/prefix-build-001` through `prefix-build-004`.

## Complete Native SmolVLA Session

Five real engines now run in a generated C++ Session on the target, including
fresh prefix encoding, all ten solver steps and both complete action outputs.
The collect-only pilot exited 0, with 1 warmup + 16 measured calls. All 34
FLOAT `[1,50,6]` output tensors were independently checked against frozen H20
evidence. Minimum normalized cosine is 0.9999852940845904, but worst MSE is
6.393668376929344e-5, above the 1e-5 gate. This is not fidelity acceptance.

Actual finish/native engine guards passed 28/28 NaN/Inf/large-finite/overflow/
incoming-false probes. A separate native two-Session probe traced 210 real
Region calls: six injected NaN/Inf faults abort without replacing either
complete published action; recovery, reset invalidation, binding invalidation
and Session isolation pass. Baseline/reset outputs match the original Orin
pilot byte exactly. No Torch or Python model DSO is linked.

These results do not cover shared-stream/replay, original-input preprocessing,
formal CDF, repeated lifecycle, long-run memory or thermal/power modes. See
[the complete native pilot report](orin_smolvla_trt_native_pilot_20261020.md)
and the goal status for hash-bound evidence and remaining requirements.

A later independently frozen runtime adds the generic TRT shared-stream
sidecar and native CUDA Graph provider. It passes 17/17 real target CTests.
Three full SmolVLA mode pilots preserve both actions byte exactly (120 complete
tensors); required replay captures all ten steps with 20 replays and zero
ordinary fallback. See the
[same-engine mode report](orin_smolvla_trt_execution_modes_20261020.md).
This does not qualify original-input E2E, formal CDF or the failed H20 MSE gate.

The next independent native build adds task-local Hugging Face Rust tokenization
and general C++ host preprocessing; it passes 19/19 target CTests. All 128
prepared input tensors are byte exact, and three original-observation E2E pilots
preserve all 120 complete action tensors. See the
[raw E2E report](orin_smolvla_native_raw_e2e_20261020.md). Formal CDF, thermal/
lifecycle ablations and the H20 precision gap remain open.

## Historical Isolated CUDA and Torch Preparation

CUDA 11.8 was fetched from the official NVIDIA arm64 Ubuntu 20.04 repository,
with all 29 package SHA256 identities independently checked after transfer.
Packages were extracted into the NVMe task prefix without installing system
packages. `cuda118-extracted-v1/usr/local/cuda-11.8` contains the compiler,
runtime/math libraries and the Jetson upgrade compatibility libraries.

A native SM87 probe built with that compiler executed a GPU kernel, H2D/D2H
and completion successfully, reporting runtime=11080 and driver=11080.
`build/cuda118-extract.json` and `build/cuda118-smoke-*.log` retain the commands
and results. The kernel probe is not a VLA performance measurement.

The isolated Torch 2.6/vision 0.21 build environment has been provisioned with
offline arm64 Python 3.11 dependencies. The Torch source was staged as 73
commit-pinned trees using local Git objects and official archives, with complete
archive hashes in `environment/pytorch-source-staging.json`. The initial recursive
Git fetch failures remain in their original log; they were not called successful.

The initial controller PID 258176 ended with exit 1: the minimal SDK prefix
lacked `nvml.h`, required by `CUDAAllocatorConfig.cpp`. Its original state/log
are retained unchanged. The omitted official `cuda-nvml-dev-11-8` arm64 package
11.8.86-1 (SHA256
`dc2386fbea53b4712a12d940bff2b3d8c59546492de3d4b163aea628c062c834`)
was verified and extracted into the task-owned prefix only. The preprocessor
probe and exact failed object compile now pass, without editing Torch sources
or changing system CUDA. See `environment/cuda118-nvml-supplement.json`.

Controller PID 275262, start ticks 1284791, resumed the existing object cache
under a distinct script/state/log identity: `resume_torch260_r1.py`,
`environment/torch260-controller-r1.json` and `torch260-build-r1.log`.
CUDA/cuDNN were enabled, targeting SM87 and three compiler jobs. The task-owned
controller group was then stopped under the new route; no wheel/runtime/vision
pass is claimed. `environment/torch260-r1-direction-stop.json` records the
verified identities and terminal group check. Preserve the evidence and do not
restart the build or downloader automatically.
