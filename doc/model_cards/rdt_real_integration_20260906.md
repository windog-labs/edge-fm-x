# RDT Real-Weight Integration

## Scope

This is an explicit RDT Adapter and official-reference implementation, not a
claim that `robot_matrix.py` fixtures support the real model. Public IR, Plan,
memory and C++ runtime modules contain no new model-name branches.

The current entry point is `vlaforge/tools/run_real_rdt_reference.py`.
It requires all three complete assets, an explicit numerical environment, a
verified upstream source tree, and a recorded AgileX input pack. It runs the
official T5 text encoder, SigLIP vision encoder and complete RDT action chunk.
`--partition` independently evaluates condition adaptation, denoiser and the
official scheduler from the exact saved initial noise. Reference execution and
partition parity are separate from IR capture, compilation and C++ deployment.

## Fixed Sources

| Asset | Official Repository | Fixed Revision | Required Weight Bytes |
| --- | --- | --- | ---: |
| RDT source | `thu-ml/RoboticsDiffusionTransformer` | `cd79363a1387e8f81c7724d070ef7e45fd23150f` | N/A |
| RDT policy | `robotics-diffusion-transformer/rdt-1b` | `eb09036cc64ca4945051acbd1bd581d30a1d7711` | 2456755578 |
| T5 | `google/t5-v1_1-xxl` | `3db67ab1af984cf10548a73467f0e5bca2aaaeb2` | 44541587809 |
| SigLIP | `google/siglip-so400m-patch14-384` | `9fdffc58afc957d1a03a25b10dba0329ab15c2a3` | 3511950624 |
| Recorded data | `robotics-diffusion-transformer/rdt-ft-data` | `a3f4b6624d4fc3a09b538f65081b37401fd8bd84` | First shard: 8589934592 |

Official model inventories, processor files, complete-file LFS SHA256 and Git
blob verification are archived under
`artifacts/edgefm-vla-goal/20260906-044509/rdt/`.
The selected three-model assets total 50511104233 bytes including processors
and model cards. The T5 checkpoint includes an unused decoder and SigLIP includes
an unused text tower. Do not equate complete checkpoint size with online
parameter count; strict-load reports list both active subnetworks and explicitly
unused keys. Every `.bin` is loaded with `torch.load(weights_only=True)`.

## Environment

H20-2 isolated root:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906`.

- Assets: `assets/RDT`.
- Official inference environment: `envs/rdt-official-cu121-py311`.
- Fixed source: `toolchains/RoboticsDiffusionTransformer-cd79363`.
- Raw recorded sample: `assets/RDT/recorded-episode5-step64`.
- Local installation selection: `rdt/environment/official-inference-py311.txt`
  under the artifact root above, with actual pip reports in `rdt/remote-audit`.

The baseline follows Torch 2.1.0+cu121, torchvision 0.16.0+cu121, diffusers
0.27.2, transformers 4.41.0, timm 1.0.3 and accelerate 0.30.1. Training-only
DeepSpeed/WandB and robot hardware drivers are not needed for this inference
entry point. Protobuf, NumPy and Hugging Face Hub compatibility pins are explicit.
A Torch 2.10/CUDA 12.8 execution must use a separately named environment and
`--numerical-profile torch210-cu128`; it is not the official Torch baseline.
That isolated environment is now installed at `envs/rdt-aoti-cu128-py311`,
with its complete pip selection report retained in `rdt/remote-audit`.

## Input and Denoising Contracts

- The verified first archive shard contains complete
  `close_glasses_box/episode_5.hdf5` and its instruction JSON. Only these regular
  members are extracted, with path/link rejection and exact member-size checks.
  Neither the whole archive nor the full dataset is claimed as downloaded.
- The actual episode has `sim=false`, 146 observations, native FP32 14-dimensional
  proprioception, and three compressed camera streams. Step 64 uses real frames
  63 and 64 in history-major high/right-wrist/left-wrist order, without invented
  camera views or padded history.
- The pack records the chosen channel decoding. `upstream-opencv-array` preserves
  published HDF5 OpenCV BGR arrays passed unchanged to PIL. `pil-rgb` is a distinct
  explicit input profile. Camera color calibration is not asserted by either.
- Each input pack is checked against independently redecoded, hash-pinned HDF5
  members and source processing files. Editing a manifest boolean or replacing
  its NPZ hash is insufficient to admit fabricated observations.
- State and action gripper scales are the official AgileX wrapper transforms,
  not SO100 statistics or arbitrary dataset normalization. Native robot-space
  outputs are not labeled calibrated physical units.
- Inference uses `DPMSolverMultistepScheduler`, not the training DDPM configuration
  label: `prediction_type=sample`, five steps, and the pinned library defaults.
  Timesteps, sigmas, `model_outputs`, `lower_order_nums`, and `step_index` are
  preserved. There is no Euler/flow-matching substitution or CFG invention.
- The official full `[1,64,128]` unified chunk and `[1,64,14]` robot-space result
  are retained. Exact initial noise and before/after RNG state are saved and
  checked against the noise actually consumed by the official model.

## Acceptance Remaining

The asset gate is complete. CPU strict-load passed in
`rdt/cpu-strict-load/report.json` under the local artifact root. Required state
tensor coverage is 618 policy tensors, 220 text-encoder tensors, and 448
vision-encoder tensors. Active online parameters are 1228319872 for RDT,
4762310656 for the T5 encoder, and 428225600 for the SigLIP vision encoder.
The full online pipeline is therefore much larger than the RDT action model
alone. T5 and SigLIP use eager attention; RDT uses the published timm fused
attention selection. The official CPU environment passed 48 focused tests.

Initial attempts failed on an older vendor Git ownership check, missing
Protobuf for tokenizer conversion, and a meta-device nonpersistent SigLIP
position buffer. Each failed attempt remains in its own report. The successful
load uses a process-local Git config, the explicit Protobuf dependency, and the
official CPU SigLIP constructor to create nonpersistent position IDs before
strictly loading every required weight. No global configuration or pretrained
weight was rewritten.

The H20-2 GPU 1 official reference is complete in
`rdt/gpu1-reference/{report.json,full_reference.npz,fidelity.json}`. All five
denoiser outputs and the full 8192-element unified chunk match the independent
same-environment partition exactly (MSE, RMSE and max-abs are zero). The official
AgileX BF16 action scaling was independently reproduced exactly. The complete
online T5, six-view SigLIP and RDT pipeline ran with timesteps
`[999,799,599,400,200]`, saved noise and checked RNG consumption. A 2.85-second
hook-instrumented reference call is not a benchmark. This is tensor/reference
parity, not calibrated physical-output or no-Python deployment acceptance.

Full-model AOTI compilation and the no-Python C++ unified-chunk path are now
executed for one recorded observation, as detailed below. The primary active
14-dimensional compiled numerical gate fails. Different-observation parity,
end-to-end throughput/latency/CDF and operator-tuning reintegration remain work.
The hook-instrumented first reference and three repeated native calls are not
latency benchmarks.
Orin/BPU execution remains deferred and cannot be replaced by the H20 reference.

## Tensor-Carry Implementation

`adapters/rdt/rdt_fresh.py` declares seven Regions through the shared
`InvocationBuilder`: online T5, online six-view SigLIP, condition projection,
history initialization, denoiser, scheduler step and complete-chunk output.
The five-step loop explicitly carries the sample, two scheduler model-output
slots, valid-history count and device step index. Prefix cache dependencies
include every relevant token, mask, image and state input. Noise is an explicit
input; the compiled graph does not perform a hidden random draw. Solver and
initialization Regions have zero model parameters. Tokenization, image decoding
and robot mapping are separately audited boundaries, not part of a no-Python
claim yet.

`adapters/rdt/rdt_scheduler.py` does not substitute an Euler update or reimplement
DPMSolver equations. It FX-traces the pinned diffusers first/second-order
primitives and replaces their finite CPU scalar constants with fixed device
tables indexed by a tensor. Unsupported scheduler profiles fail closed.
The initial CUDA lowering exposed a real BF16 rounding difference: CPU FP32
scalar multiplication uses FP32 opmath on CUDA, while an equivalent CUDA scalar
can be rounded to BF16 before multiplication. The corrected graph explicitly
preserves FP32 multiplication and the original BF16 result conversion.
The four-element negative CUDA test fails on the old lowering, and all five
real saved denoising updates are exact with the corrected implementation.
Before/after reports are `rdt/remote-audit/solver-real-{before,after}.json`.

The separate Torch 2.10 official full pipeline differs slightly from the
Torch 2.1 baseline despite identical saved noise and processed observations:
8192-element MSE `9.277514436689672e-7`, cosine `0.9999824402840329`, max-abs
`0.01953125`, 602 differing elements. This passes the stated numerical thresholds
but is explicitly a cross-environment comparison, not exact same-version or
compiled-deployment acceptance. Raw values and per-dimension metrics are in
`rdt/gpu1-torch210-capture/cross-torch-fidelity.json`.

The first real capture retained six successful exports, including full online
T5, vision and denoiser, plus a failed conditions export caused by inference-mode
examples. The adapter now clones ordinary detached capture examples and has a
targeted export regression. The original partial report and incorrect solver
outputs remain archived; they are not relabeled as passed.

The corrected real run is complete in
`rdt/gpu1-torch210-capture-fixed/capture/capture.json`: seven of seven Regions
captured with exact eager/export comparisons and all seven static operator
inventories recorded. The complete tensor-carry model matches the same-Torch
official pipeline exactly, including online T5, online vision, all five denoiser
outputs and the full unified chunk (MSE/max-abs zero). Full raw samples and
per-dimension fidelity are retained. The actual large exports remain on the
isolated NAS at
`runs/rdt-gpu1-torch210-capture-0923/capture/exports`; local reports bind every
export, example and inventory by SHA256. This is capture and same-version
tensor-carry verification, not full-model compiled/C++ deployment acceptance.
The latest full CPU suite is 66 passed with one explicit CUDA test skipped;
the separate CUDA-inclusive focused suite passed 18 tests, including the
negative-capable scalar regression.

The corrected real solver export also compiled on H20 through the shared
`compile-artifact` CLI with `eager-numerics`. Its 495103-byte AOTI package has
SHA256 `9d0b7c9119cbd09dde6784496ecbd02e8de8f8378ae52d774a04548c85de25fa`.
Python-loaded execution of that actual package, driven by saved real denoiser
outputs, matched every sample across five accumulated scheduler steps exactly.
Both history slots, the valid-history count and device step index also matched
the official scheduler at every step, with complete raw state tensors retained.
See `rdt/aoti-solver/rdt_solver.compile.json` and
`full-solver-reference-v3.{json,npz}`. This verifies a compiled solver Region,
not by itself the complete online AOTI model or a no-Python C++ Session.
The initial loader-only failure (Torch's missing eager import of
`torch._inductor.codecache`) remains in `direct.log`; the successful loader
explicitly imports that same installed module and does not alter the package.

## Complete AOTI and Native Session

`tools/build_real_rdt_fresh.py` now supports `compile`, `exports`, `direct` and
`session` stages using shared capture, compiler, Interpreter and bundle APIs.
Seven actual `eager-numerics` AOTI candidates compiled serially on H20-2 GPU 1.
Only the previously verified solver package was reused; online T5 weights were
not attached to every Region. The serialized seven-Region IR, executing loaded
ExportedPrograms, matches the same-Torch official complete output exactly.

The full compiled chain executes online T5, six-view SigLIP, condition projection
and all five denoiser/scheduler iterations. Its complete output is retained, but
the primary paper numerical gate **fails**:

| Comparison Space | Elements | MSE | Cosine | Max-Abs | Paper Numeric Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Unified internal vector | 8192 | 5.1059248083e-6 | 0.9999119807 | 0.025390625 | Not the acceptance space |
| Active AgileX action chunk | 896 | 4.6682741104e-5 | 0.9999119807 | 0.025390625 | Fail |
| Official wrapper native scale | 896 | 0.0018753021531 | 0.9999209914 | 0.3125 | Fail; not calibrated physical units |

Inactive zero padding in the unified 128-dimensional vector must not dilute the
MSE used for paper acceptance. The tool now derives the primary gate from the
explicit binary action mask and includes a regression where the padded-space
gate passes but the active-space gate fails. Historical `direct/report.json`
and `session/report.json` were generated before this reporting correction and
retain their original internal-space metrics; their paper-gate interpretation
is superseded by `full-aoti-0940/active-output-acceptance.json`. Raw outputs and
all failed attempts are unchanged. Active and robot-space full arrays are in
`active-and-robot-outputs.npz`. The official BF16 AgileX scaling formula is
independently exact on the reference before applying it to compiled outputs.

The generic generated C++ Session completed three fresh calls, incrementing
every input revision so online text and vision are not replaced by stale cached
embeddings. All three complete BF16 chunks are bitwise equal to the same AOTI
direct execution. Their raw SHA256 is
`9ea249bb8d75475e66f2d68929968f8f46d294ec53c57d464141779d11b30179`.
The runner exits zero with invalid `PYTHONHOME`/`PYTHONPATH`; archived `ldd` and
actual process maps contain no `libpython`. This establishes a no-Python
tensor-input to unified-action-output path, not C++ image decoding/tokenization,
calibrated robot control, different-observation generalization or paper quality.

Torch 2.10's package loader ignores `TMPDIR` and extracts into hardcoded system
`/tmp`; the 9.53 GB T5 package exceeded the 7.4 GB free overlay. Its failed partial
was moved intact to the isolated NAS, restoring system capacity. A private
mount-namespace probe was denied. The explicit `--aoti-loader extracted` route
instead extracts every regular archive member on NAS, rejects unsafe paths and
links, binds each file to the original package bytes, and uses the repository's
existing raw `.so` AOTI loader. Embedded mmap weights, cubins and metadata remain
together and are covered by the bundle manifest. No system library or global
mount was changed. A second failed attempt exposed raw-runner list versus
single-Tensor output structure; the tool adapter now restores declared Region
arity and has focused regression coverage.

Local evidence root: `artifacts/edgefm-vla-goal/20260906-044509/rdt/full-aoti-0940`.
Complete large packages and runnable bundle remain on the isolated NAS:
`/xs-train-nas/zzm/repos/edgefm-vla-goal-20260906/runs/rdt-full-aoti-0940`.
The native bundle is `session/bundle`; the saved seven-input binary pack is
`session/inputs`. The ordinary-loop binary accepts:

```sh
CUDA_VISIBLE_DEVICES=1 session/bundle/bin/vlaforge_generated_runner \
  session/bundle session/inputs NEW_EMPTY_OUTPUT_DIRECTORY 3
```

Run only after checking GPU ownership, and create a separate output directory.
The next numerical candidate must be separately labeled and compared against
the same-Torch official full active action chunk before timing acceptance.
The final RDT regression in the locked Torch 2.10 environment is 84 passed and
one intentionally skipped CUDA opt-in test (`remote-audit/rdt-full-tests-1036.*`).
The new orchestration tests account for 18 passes, including archive integrity,
raw output arity and inactive-dimension MSE dilution cases.

## Independent Native ATen Candidate

The new `tools/build_real_rdt_torchscript.py` uses the public
`compile-torchscript` entry point and the stable `torchscript-aten/1` backend.
It does not replace or relabel the failed AOTI numerical result above. The
complete stable C++/Python baseline comes from the local
`source-torchscript-benchmark-v1` snapshot; only RDT orchestration/adapter files
are overlaid in independent `source-rdt-torchscript-v1` and `-v2` snapshots.
All 575 source files in each snapshot passed remote SHA256 verification.

All seven saved archives passed per-Region example bitwise validation on H20.
The actual online T5 archive is 9,525,087,244 bytes, SHA256
`f395f1208cd743c508bcd9d07aafa635da25f9f46d809fc251c2d171d78d0c42`.
No offline text embedding replaces it. CPU scalar placement is preserved and
JIT optimization is explicitly disabled. Recorded trace warnings remain in
each manifest; this result does not prove dynamic shapes or arbitrary new
data-dependent branches are supported.

The same recorded AgileX observation and saved noise completed three fresh
full Invocation IR calls. Every input revision changes for each call. Online
text, all six history-major camera views, RDT and the original five-step
DPMSolver state chain execute each time. All three complete 8192-element outputs
are bitwise equal to the same-Torch official reference. In the primary active
14-dimensional chunk (896 values), MSE/RMSE/max-abs are all zero and cosine is
one. The full output NPY SHA256 is
`30581a6c2fcbac3cbd20f15dcceae2fe74ae69ece476cc6937b657830babfb2a`.
This is one observation repeated three times, not three independent samples.

Evidence: local `rdt/torchscript-v1/compile.json`, per-Region compile manifests,
and `direct/{report.json,fidelity-0.json,full-actions-0.npy}` beneath the shared
Goal artifact root. Large archives and intermediate raw Region outputs remain
at the isolated NAS `runs/rdt-torchscript-v1` directory.

The independent generic C++ Session also completed three fresh calls with
every input revision incremented. Each full BF16 output is bitwise equal to
the same archive's direct output and to the same-Torch official reference;
the active 896-value MSE and max-abs are zero. All three 16,384-byte raw outputs
have SHA256
`9e61ec7b6779f5dd2eea06626d5a967ab7b3cda42ba6fced35c267773fdd690f`.
The runner exits zero under invalid `PYTHONHOME` and `PYTHONPATH`; `ldd` and
actual process maps contain neither `libpython` nor `libtorch_python`.
`session/report.json` and complete per-run fidelity reports retain this evidence.
The full runnable NAS bundle is `runs/rdt-torchscript-v1/session/bundle`;
the saved binary input pack is its sibling `session/inputs`.

These are model-tensor-boundary numerical checks. No throughput, latency CDF,
multi-observation quality, C++ image decoding/tokenization, robot-calibrated
physical output or full paper acceptance is claimed. The original AOTI
numerical failure remains unchanged and is a separate backend candidate.

## Sixteen Recorded Histories

The independent `rdt-torchscript-series16-v2` run now validates sixteen distinct
legal two-frame histories from the same fully verified episode, at steps
1, 10, 20, 29, 39, 49, 58, 68, 77, 87, 97, 106, 116, 125, 135 and 145.
Each observation has its own saved noise and seed (2026090600 through
2026090615). This is not a multi-episode or robot-control success evaluation.

The same-Torch official full online pipeline ran for every observation. All
sixteen complete direct outputs and all sixteen outputs from one persistent
generic C++ Session match the official 8192-element BF16 storage bit for bit.
The primary active 14-dimensional, 64-step action chunk has zero MSE, RMSE and
max-abs for every observation. The Session loads no Python library and each
input revision changes, so cached text/vision do not replace online inference.
An independent local check decodes each 16,384-byte raw output and verifies its
SHA256 and complete output bits against both saved references.

Evidence under the shared local Goal root:
`rdt/torchscript-series16-v2/{reference.json,direct/report.json,native/report.json,native-local-byte-audit.json}`.
The complete new runnable bundle and typed input packs remain on the NAS at
`runs/rdt-torchscript-series16-v2/{native-build/session/bundle,native-inputs}`.
The original metadata-only failure is retained separately: official scheduler
configuration contains `lambda_min_clipped=-inf`, which strict JSON initially
rejected before any sample. Evidence serialization now explicitly tags such
configuration numbers; it does not alter the scheduler or permit nonfinite
inputs/actions. Frozen `source-rdt-series-v3` and `-v4` bind the successful
reference and direct/native tools respectively.

Formal timing/CDF, calibrated physical units, sensor preprocessing deployment
and Orin/BPU execution remain outside this numerical acceptance.

## Formal H20 Tensor-Boundary Benchmark

The separately frozen `source-rdt-benchmark-v2` ordinary-only campaign is now
complete. Each of five independent no-Python processes performs 128 warmup and
1024 measured fresh calls over the same sixteen recorded histories, with all
input revisions changing. The online text encoder, six-view vision encoder,
RDT and original five-step DPMSolver execute on every call. All 5760 complete
BF16 outputs, including warmup, are bitwise equal to the same-archive direct
output and the same-Torch official reference. The 896-value active action
chunk has zero MSE, RMSE and max-abs throughout; inactive dimensions do not
enter the primary numerical gate. The separate pilot has 48 exact calls and
is excluded from these statistics.

| Boundary / Backend | Measured Calls | Mean | P50 | P95 | P99 | Std | Calls/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Resident model tensors, H20, native ATen BF16, ordinary | 5120 | 174.163 ms | 173.619 ms | 176.363 ms | 178.194 ms | 1.376 ms | 5.742 |

Independent-process mean standard deviation is 1.280 ms. The complete worst
measured sample is retained: 194.190 ms, process 0, measured index 27.
Full steady-loop wall throughput, including D2H/verification/serialization,
averages 5.738 calls/s. This is not a matched Python baseline speedup, an Agent
ablation, sensor-to-action latency or an Orin/BPU result.

The H20 container exposes a different PID namespace from NVML. The original
pilot in `benchmark-ordinary-v1-failed` was stopped before model execution;
it is not a numerical failure or a benchmark sample. The explicit replacement
uses a pre-Session context register/reset/re-register handshake to causally
identify its own NVML PID. It never resets the device or an existing model
context. Every subsequent sampled owner matches that mapping. Telemetry is
sampled approximately once per second plus query overhead, so it cannot rule
out transient activity between samples. Each process records its complete
handshake, raw telemetry and loaded-library maps.

All five 18,874,368-byte raw BF16 output files have SHA256
`3cf741cc02b228d54295c10587d9579f42fbf666af79c93b3cc706c41d22bb7f`.
An independent local audit re-reads every byte against both hash-verified
reference sets, validates every call/sample/revision and warmup boundary, and
checks no-Python maps and observed owner identities. The CDF image was rendered
from all 5120 formal samples and visually checked; pale lines retain individual
process distributions, and the tail panel keeps the complete latency range.

Local evidence under the Goal artifact root:
`rdt/benchmark-ordinary-v2/{report.json,latency-table.csv,off-cdf.csv,latency-cdf.png,independent-local-audit.json}`.
Per-process raw output, CSV, fidelity, telemetry and maps are in `runs/00-off`
through `runs/04-off`. Runnable models and frozen generated benchmark source
remain on the isolated NAS at `runs/rdt-benchmark-ordinary-v2`.
Generic typed benchmark and focused RDT tool regressions passed 109 CPU tests;
the typed benchmark accounts for 64, including actual host C++ exhaustive
BF16/FP16 decoder checks and namespace handshake failures.

Full paper acceptance, physical calibration, deployed sensor preprocessing,
low-bit deployment, Agent-selected operator reintegration and Orin/BPU remain
unverified by this campaign.
