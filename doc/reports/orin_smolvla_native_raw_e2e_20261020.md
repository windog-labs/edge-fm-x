# SmolVLA Native Original-Observation E2E Pilot

Date: 2026-10-20. Status: original-input native pipeline and same-engine pilot
passed on real Orin. This is not formal 5-process sampling, a completed CDF,
or completion of the five-model Goal. Cross-device precision remains unqualified.

## Actual Boundary

The start boundary is a CPU-resident, already decoded two-view RGB observation,
native state, task text and the exact saved noise. The end boundary is complete
CPU-resident normalized and native action chunks ready to consume. Each request
includes native image scaling, checkpoint state normalization, real tokenization,
padding/masks, H2D, binding, fresh vision/language encoding, all ten solver steps,
finite guards, affine postprocessing, transactional publication and complete D2H.

File/video decoding, raw-record loading, tokenizer/model initialization, reference
loading, output validation and evidence writes are outside the per-request clock.
The runner reads raw bytes/text, not precomputed image features or processed input
tensors, during its actual preparation path. Inputs are reprocessed on every call.
All model computation runs through the same real TensorRT engines on Orin.

Target is Jetson AGX Orin serial `1421524312594`, L4T 35.4.1, CUDA 11.4,
TensorRT 8.5.2.2. The target uses C++/CUDA and a native Rust tokenizer library,
not Torch/torchvision or a Python model runtime. No global SDK/toolchain was
installed or replaced. Target civil time remains incorrect; durations use
monotonic clocks. MAXN is not labeled 60W and exclusive ownership is not claimed.

## Input Qualification

The original 16 LeRobot observations are from
`lerobot/svla_so100_pickplace@728583b5eaf9e739a7f119e2def466fa1d552402`.
Raw manifest SHA256 is
`1724d43d2b694b7065b525c6dfcdf937a9a9f7149fb92015c9c235ab009c1c88`.
Images/state vary across the 16 recorded frames; the dataset task text is the
same across these samples. Repeating the finite input set is not an independent
trajectory or a physical robot evaluation.

The data-driven declaration selects:

- two UINT8 CHW RGB arrays `[3,480,640]`, divided by 255 into FLOAT model inputs;
- exact recovered-checkpoint state mean/std, with the declared epsilon `1e-8`;
- the declared suffix, special-token policy, maximum length 48, right padding
  with ID 2 and tokenizer-configured left truncation;
- actual available-camera masks, and unchanged caller-provided noise `[1,50,32]`.

There is no instruction-to-token lookup table or model-name dispatch in the
common processing/runner code. The native provider invokes Hugging Face
Tokenizers for every text operation. This dataset's 12 nonpadding tokens and
all padding/masks match the frozen processed input contract.

`native-observations-001/results/audit.json` independently verifies all 128
complete input tensors (16 frames x 8 ports), including dtype/shape, file hashes,
finite values, valid BOOL storage and byte equality with the frozen model-input
pack. The scalar/statistics declaration is derived from the original processor
files, not new dataset statistics or an action-to-state substitution.

Profile SHA256: `bd881a7c50f4388009dbee668d5d7ac6f806950a511a78496983e278f613b64f`.
Tokenizer JSON SHA256:
`5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a`.
State statistics SHA256:
`dea286b621a7dc320a6f8ba6163391c9cc8881fd1eb1ea7c57667c3172c111a4`.
These numerical-scale checks do not certify physical robot calibration/units.

## Complete-Model Pilot

Each mode ran in a separate native process with 4 warmups + 16 measured requests.
All use identical checkpoint/engine hashes, precision, observed frames/noise,
ten-step solver, output dimensions, validation and publication. The baseline
is ordinary native Session execution, not an official PyTorch baseline or a
bare `trtexec` subgraph measurement.

| Mode | E2E Mean ms | p50 ms | p95 ms | Maximum ms | Full Actions |
|---|---:|---:|---:|---:|---|
| Ordinary TRT | 123.062324 | 123.103008 | 124.116304 | 124.533760 | Byte exact |
| Shared-stream batch | 121.994428 | 122.031904 | 123.214936 | 123.409504 | Byte exact |
| Complete 10-step replay | 102.747002 | 102.723344 | 103.609584 | 104.544768 | Byte exact |

Across all three modes, all 120 complete FLOAT `[1,50,6]` output tensors
(20 requests x 2 outputs x 3 modes), including warmups, match the frozen Orin
conventional output byte exactly. Same-target MSE/RMSE/max/mean-abs are zero;
cosine is one. Every normalized/native action coordinate is preserved.

Required replay reports 20 actual replays, ten captured solver steps and zero
ordinary fallback; every per-call telemetry row is independently checked.
Shared batch reports 20 ordinary bounded-loop executions and zero replay.
Fresh prefix encoding is not replaced by saved features.

The CSV preserves `preprocess_ns`, `h2d_ns`, `bind_ns`, `model_ns`, `d2h_ns`
and the directly measured start-to-end `latency_ns`. All timestamps are adjacent
within the same request and every row conserves the complete duration exactly.
E2E is not reconstructed from averages of independently timed stages. Native
preprocessing averages about 1.40 ms in these pilots. All slow/warmup samples
remain in their original files with explicit measured flags.

These are short, single-process-per-mode pilots. They do not establish a paper
speedup, tail/CDF stability, thermal equilibrium, or five-process confidence.
The older host-model-tensor pilot remains a separately labeled boundary and is
not substituted for this original-observation measurement.

## Precision Gap

The same-target byte-exact result does not erase the original H20 comparison.
The worst normalized Orin-versus-H20 MSE remains `6.393668376929344e-5`, above
`1e-5`, with minimum cosine `0.9999852940845904`. No tolerance was relaxed,
no scheduler/iteration count was changed and no quantization search was run.

A separate host CPU reference executes all five frozen source ONNX stages
through the Invocation IR, including the ten-step loop. It is reference
preparation, not an added platform performance experiment. Checked INT32/INT64
index conversions preserve values and are logged; floating dtype mismatch is
rejected rather than silently cast.

Initial diagnostic normalized MSEs:

| Sample | FP32 ONNX vs H20 | Orin vs FP32 ONNX | Orin vs H20 |
|---|---:|---:|---:|
| 0 | 1.9592896268962023e-5 | 2.780667006937063e-6 | 2.1266888167588533e-5 |
| 9 | 2.455723602525217e-5 | 1.7643159866790003e-5 | 6.393668376929344e-5 |

Thus the observed discrepancy is not attributed solely to export or solely to
TensorRT. The source gap includes export and numerical-profile differences
relative to the old H20 execution, not one proven cast/operator. The audit
retains the MSE cross term: total MSE equals source-gap MSE + engine-gap MSE +
twice their mean product. Source and engine MSEs cannot simply be added.
The subsequent all-16 reference run completed 224 source Region invocations
and saved all 32 complete normalized/native outputs. Its independent audit is
`onnx-complete-reference-003/drift-audit.json`:

| Normalized Comparison | Passing Samples / 16 | Worst MSE | Worst Sample | Minimum Cosine |
|---|---:|---:|---:|---:|
| Orin vs H20 | 2 / 16 | 6.393668376929344e-5 | 9 | 0.9999852940845902 |
| FP32 source ONNX vs H20 | 3 / 16 | 5.467356835394047e-5 | 6 | 0.9999889157728636 |
| Orin vs FP32 source ONNX | 12 / 16 | 2.6492215645769304e-5 | 10 | 0.9999951141597108 |

All 16 comparisons clear the cosine threshold, but source-relative Orin MSE
fails on samples 1, 5, 9 and 10. This is not a passing FP32-source fidelity
certificate. The same-engine ordinary/optimized actions remain byte exact;
that narrower statement does not make either source-relative or H20-relative
quality pass. Formal acceptance remains open, and no precision policy was
changed to obtain these diagnostics.

An initial CPU reference attempt failed before complete inference because the
local optimized ONNX directory lacked its external-weight sidecar. The existing
lowering reports and actual target file independently confirm sidecar SHA256
`ba0efb7fba4c8de08fd054e6bf712af9584c6cdbf607d73c12f0cedafc53a921`.
A new hardlink restores that exact host asset without changing graph or weight
bytes; the original failed record is preserved.

## Generic Code and Tests

`host_preprocessing.h/.cpp` implements bounded U8 scaling, explicit FLOAT
standardization, token padding/truncation and suffix handling. The declared
pipeline uses typed host storage and explicit operations; it does not infer a
model's normalization statistics, channel ordering or tokenizer policy.

The optional native tokenizer bridge uses [Hugging Face Tokenizers 0.21.2](https://github.com/huggingface/tokenizers/tree/v0.21.2)
with a thin ownership/error boundary. Invalid JSON, UTF-8 and size errors do not
unwind across the C ABI; token IDs are range-checked before INT32 publication.
There is no Python wrapper or hand-written BPE engine on the board.

Rust 1.90.0 component hashes were checked against its official release manifest.
The target build is entirely offline, under the task-owned prefix, using a
checksum-verified vendor corpus. The resolved 76-package lock is retained at
`vlaforge/backends/hf_tokenizer/Cargo.lock`, SHA256
`268f1321458fe8b0b685ff57ad047a926033bf15e785cd7e40da81d612f5bfa7`.
The compiled native Rust archive SHA256 is
`dc69eff52c40ca218bf287c6239e6dc4a25aa7d77563ed42f5fad2a99c81244c`.
The larger vendor archive retains upstream optional/development packages; those
are not all part of the linked provider.

Verified coverage:

- Target native runtime/tokenizer CTest: 19/19 passed, including invalid JSON,
  invalid UTF-8, empty/oversized text, scalar boundaries, stream and graph tests.
- Host native runtime CTest: 13/13 passed, including in-place token padding.
- Native profile/measurement/runner focused regression: 54 passed.
- Expanded existing host regression: 545 passed, 40 explicit GPU/LibTorch opt-ins
  skipped. These skips do not substitute for the actual target tests.
- Independent raw-input audit: 128/128 complete tensors byte exact.
- Independent original-input model audit: 120/120 complete actions byte exact;
  exact timing conservation and per-call replay counts checked.

## Evidence

Local root: `artifacts/orin/20261020-00/`. Remote task root:
`/data/datalog/edgefm-orin/20261020-00/`.

| Evidence | Relative Location |
|---|---|
| Model-specific semantic declaration | smolvla-native-host-profile-001.json |
| Native build, resolved lock, tests and snapshots | native-preprocessing-build-001/ |
| Input preparation, raw bytes and independent audit | native-observations-001/ |
| Original-input pilot controller, exact argv/PIDs/hashes | raw-e2e-pilots-001/campaign.json |
| Complete independent E2E/output/replay audit | raw-e2e-pilots-001/audit.json |
| Three compiled native raw runners and raw CSV/output evidence | raw-runner-{off,batch,replay}-002/pilot-001/ |
| Two-sample source/engine error decomposition | onnx-complete-reference-002/drift-audit.json |
| Full-corpus reference outputs and error decomposition | onnx-complete-reference-003/{reference,drift-audit}.json |
| Expanded host JUnit | audit/native-raw-e2e-regression-001.xml |

Exact reproduction commands are preserved in build/controller records. The
materializer accepts `--profile`, `--raw-manifest`, `--model-ir` and a new output
directory. Raw runner preparation verifies the complete qualified input-port
contract, while preserving both full IR digests: execution-mode annotations
legitimately differ and input compatibility is not full-model equivalence.
The separate same-engine action audits establish the latter for these cases.
Never overwrite an existing run to reproduce it.

Remaining before formal acceptance: finish precision attribution/qualification,
freeze the 5 x (128 warmup + 1024 measured) protocol, qualify shared/replay
fault/lifecycle/allocation behavior, collect telemetry and a 30-minute soak,
then complete operator/Agent ablations. RDT, pi0, pi0.5 and qualified CogACT
remain required; Qwen remains lower priority. No full Goal completion is claimed.
