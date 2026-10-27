# RDT Denoiser and Native Session Checkpoint

Status: in progress, not full-model acceptance. Evidence root:
`artifacts/orin/20261021-05/`. The five-model Orin goal remains active.
Only `ssh orin` performs TensorRT model execution. `zzm-4090-x8-1` accesses
the existing shared NAS for CPU export/reference preparation, with
`CUDA_VISIBLE_DEVICES=""`; no host model performance experiment is added.

**User clarification, 2026-10-21:** the strict primary quality gate compares
conventional TRT and EdgeFM on the same Orin, engines, precision and inputs.
H20 is supplemental. Cross-device failures below remain valid diagnostics but
are not an additional blocker for formal sampling. Earlier wording calling
them a mandatory source gate reflects the former stricter interpretation.
Same-Orin mode equivalence and raw-input E2E remain necessary and distinct.

## Export Diagnosis

The first GELU/SiLU-corrected denoiser graph retained 66 COMPLEX128 Casts.
Its portable graph cannot load: ONNX Runtime reports invalid complex128 input
to `/Mul_1068`. A two-by-eight real BF16 `torch.var(..., correction=1)` trace
reproduces the issue in under a second. This is not evidence that RDT requires
complex arithmetic. The source archive and its inputs remain unchanged.

The generic export policy now computes native 16-bit `var`, `var_mean` and
`std` composites in FLOAT and casts each result back to its original dtype,
in addition to the existing GELU/SiLU policy. It does not promote hand-written
arithmetic patterns or dispatch on model names. Symbolic registration is
restored, including on failure, and existing custom policies are rejected.
Tests cover BF16/FP16, corrections 0/1/2, tuple outputs and complete numeric
values. The regression was observed failing before the fix, then passing.

New denoiser source export records 33 native BF16 variance promotions, two
GELU promotions, two SiLU promotions and one declared CUDA-to-CPU factory
device relocation. The old invalid graph is retained separately.

- Source ONNX SHA256:
  `4059e401d8acf8f495a0f7b2bf1ab1123d02b162cf361e22a097c0210b98f75e`.
- Carrier ONNX SHA256:
  `0f736d35354ec48e071cdebcfa1241fae8a619e84ea25ee2f3ebdb2076a17d56`.
- External payload: 4,828,332,032 bytes, SHA256
  `af605f980a4081e5675445e3812befd3b3c47860df8385a4ec00389ac37e518e`.
  It matches the earlier carrier payload exactly; only graph metadata changed.
- Precision profile: 313 MatMul and four Gemm selected for FP16 compute/output,
  FLOAT arithmetic, TF32 disabled, no full-model FP32 fallback.
- CPU portable graph versus original CPU denoiser fixture, full `[1,64,128]`:
  MSE `3.617852927351863e-6`, cosine `0.9999308339099398`,
  max-abs `0.027857080101966858`, not value-exact.

**Precision limitation:** both the older and corrected source ONNX graphs
declare a FLOAT output, while the original archive returns BF16. Some internal
operations also have FLOAT types. Carrier lowering only rounds proven typed
BF16 boundaries; it is not proof of complete native BF16 operator equivalence.
The portable comparison is not the FP16 TensorRT comparison, and the full
action gate remains unevaluated. No reference or tolerance is changed.

## Ranges and Integer ABI

`denoiser-opmath-002/dense-ranges-v1.json` examines all 317 dense operators'
inputs, outputs and weights on one frozen CPU portable case. No observed
boundary exceeds FP16 range. This is not a global bound over all observations
or all five iterations.

`denoiser-integer-audit-001.json` classifies all 57 out-of-INT32 constants as
positive-step Slice end sentinels. Both INT64_MAX and INT32_MAX exceed their
static slice extents, so this clamping does not change these Slice results.
The timestep initializer is in range. Frozen input narrowing separately checks
step index 0, frequency 25 and binary text mask; the complete five-step loop
must preserve indices 0..4. Runtime warnings are not silently discarded.

## Conditions and Output Contracts

Corrected conditions engine is built and executes on Orin:
`/data/datalog/edgefm-orin/20261021-05/conditions-build-001/model.engine`.
SHA256 `d39cc7dd0a9f0ccd139cae4793eee8f9b4934933734f8e15e997156fc9668f4c`.
All four inputs and all three complete outputs were audited; 22 FLOAT plugin
implementations are inspected. These remain feature results, not action gates.

| Complete output | Shape | MSE vs original CPU | Cosine |
|---|---|---:|---:|
| Language | `[1,6,2048]` | 1.1995101464e-6 | 0.9999953383 |
| Images | `[1,4374,2048]` | 3.1674593250e-5 | 0.9999901181 |
| State | `[1,1,2048]` | 1.1964263502e-5 | 0.9999945384 |

All sixteen frozen model-input packs have been restored and checked against
the existing reference hashes, including original BF16 noise bytes. No new
noise is generated. `rdt-frozen-tensors-001/pack.json` records the exact
FLOAT carriers and original dtype/byte identities. This is a tensor-boundary
pack, not decoded-observation preprocessing.

The original AgileX output method and index declaration were extracted from
the pinned source without loading the full model. Applying this exact method
to all sixteen frozen unified chunks matches all sixteen native references
value-for-value. The mapping is **not the first fourteen coordinates**:
`[50,51,52,53,54,55,60,0,1,2,3,4,5,10]`. Original BF16 gripper scale constants
and multiplication rounding are preserved. Physical units are not independently
verified. `output-source-001/source-proof.json` locks the source and mappings.

`complete-native-ir-001.json` declares the extra output Region and verifies the
full eleven-Region online IR, retaining both native14 and unified128 chunks,
all five action iterations, and final finite-output validation. The generic
Session preparation now binds all eleven real build receipts, ONNX contracts
and physical engine port orders; no dummy engine hashes are supplied.

## Complete Resident Orin Execution

Denoiser build succeeds in `denoiser-build-001/`; engine SHA256
`0327f1933653e0bb20c26a57483764543aa76ef35e6dee3f5b388faf2ba05531`,
2,438,692,823 bytes. Build/payload attestation passes. All eight inputs and its
complete `[1,64,128]` output execute natively. Against the original CPU stage
fixture, MSE is `4.199721607269942e-6`, cosine `0.9999202198954474`, max-abs
`0.030635833740234375`. All 303 inspected plugins have FLOAT IO. This is a
single denoiser call, not a full action gate. The observed integer/slice
classification above is retained with its warnings.

The native output conversion engine is also built: SHA256
`97f100621f11312c91e10d6353cadd2701fd83af1ef30a2a07cff0703fa8da5b`,
25,191 bytes. It is a declared small FP32 numerical island with the BF16 result
rounding plugin, not a full-model FP32 fallback. Its complete native14 tensor
and finite predicate match the original source fixture value-for-value.

The generated eleven-engine C++ Session compiles and executes on Orin at
`/data/datalog/edgefm-orin/rdt-session-20261021-05-001/`.
Its preparation source digest is
`54248c48300faadc306f2a459777eabdc8937212647c576f7e83a461c0cf958e`,
IO schema digest
`b0fa784728624383c3e18500d4780d8d0d620abc90a678365b4528e57517dabe`.
It uses the existing generic Adapter/IR/Plan/Session route, not a model-name
branch in the public runner/backend. All encoders execute online; exact cache
is disabled and all five action iterations remain in the IR. Intermediates
stay resident; this is no longer a file-mediated engine chain.

The first preflight attempt was rejected before inference: globally inheriting
LD_PRELOAD made the Python collector and environment probe GPU owners. That
failure is retained as `pilot-001/`. The replacement collector preloads the
precision DSO **only in the native C++ worker**; the original ownership gate is
unchanged. Use `collect_native_session_pilot.py`, not the rejected
`run_precision_session_pilot.py` wrapper, for future runs.

`native-session-target-001/pilot-002/` contains one warmup and sixteen measured
tensor-boundary calls, seventeen complete tensors for each output, telemetry
snapshots, native process maps and raw CSVs. The worker exits zero, all outputs
are finite, and repeated sample zero produces identical values for both
outputs. Process maps contain the frozen precision plugin and no LibTorch,
libc10 or torchvision. All 112 target input files match the frozen preparation
hashes in a post-run check; this is not claimed as a preflight hash attestation.

### Supplemental H20 Comparison: Failed

`complete-action-audit-001.json` checks every output hash, source-reference
hash, complete horizon and sample pairing. Both native14 and **active14 only**
normalized outputs are gated. Inactive unified coordinates are zero and do
not dilute the quality metric. Full unified128 diagnostics are retained
separately. Thresholds remain strictly MSE < 1e-5 and cosine > 0.9999.

| Original frozen H20 BF16 reference | Passed / 16 | Worst MSE | Minimum cosine | Worst MSE sample |
|---|---:|---:|---:|---:|
| Native14 | 0 / 16 | 0.04446649863166385 | 0.9984520880792738 | 12 |
| Normalized active14 | 0 / 16 | 0.0004377936456876341 | 0.9990622380952998 | 12 |

The native worst-sample errors are dominated by joint-output coordinates 6
and 13, which include the original gripper rescaling. This identifies where
the error is visible; it does not establish which upstream stage caused it.
The earlier vision/text feature gaps and source ONNX dtype drift remain open.
These failed complete-source comparisons cannot be replaced by the passing
single-call denoiser fixture or repeated-output determinism. RDT still needs
the same-Orin complete-output and raw-input E2E qualifications before formal
E2E/CDF sampling. The H20 comparison alone is supplemental under the subsequent
explicit user clarification, not the primary no-change claim.

An additional **untimed original-archive CPU BF16 quality** reference campaign
has completed on the NAS host. It uses exactly the original archives,
observations and saved noise, all five original solver calls and the pinned
output method. All archive hashes were checked before/after, input bytes
checked for all sixteen samples, and source outputs remain BF16 before
lossless serialization as FLOAT. There are eighty denoiser calls and eighty
solver calls. The process has exited. It does not authorize replacing the
original H20 reference or relaxing gates and is not another platform
performance experiment.

The complete three-way comparison is
`three-way-action-attribution-001.json`, backed by
`original-cpu-actions-001/reference.json` and all thirty-two complete NPYs.
All comparisons below fail the original strict gate on all sixteen samples.
Only active14 normalized coordinates enter these MSEs.

| Comparison | Worst normalized-active14 MSE | Worst native14 MSE |
|---|---:|---:|
| Original archive CPU vs frozen original H20 | 0.0009384180608874106 | 0.15612410342089333 |
| Orin TRT vs original archive CPU | 0.0011009523001673318 | 0.1797175875354599 |
| Orin TRT vs frozen original H20 | 0.0004377936456876341 | 0.04446649863166385 |

Thus CPU source-archive portability is itself nonexact. This does not isolate
an individual stage or explain away the TensorRT gap, and switching references
does not rescue the gate. The original H20 archive execution was previously
byte-exact with official outputs, as recorded in
[RDT execution variants](rdt_execution_variants_20260908.md); that historical
H20 result is not an Orin result.

## Native BF16 GEMM Capability

The next numerical route is supported by an actual Orin cuBLAS probe, not an
assumption about TensorRT support. `probe_cublas_bfloat16.cu` compiles using
the installed CUDA toolchain and runs with cuBLAS 11606 on SM87. The probe
uses bounded binary-rational inputs whose products and sums are exactly
representable in FLOAT, a complete 32x32x64 GEMM, and an independent exact
CPU result followed by BF16 round-to-nearest-even.

| Native library path | Status | BF16 output mismatches / 1024 |
|---|---|---:|
| BF16 operands, FLOAT accumulation, BF16 output | success | 0 |
| BF16 operands, FLOAT accumulation/output, then BF16 rounding | success | 0 |
| FP16 operands, FLOAT accumulation, FP16 output, then BF16 rounding | success | 69 |

This establishes a concrete extra-rounding mechanism and basic native BF16
GEMM availability. It does **not** establish the amount of model error caused
by that mechanism, a TensorRT plugin pass, batched/broadcast/transpose coverage,
full-action source fidelity or performance. Original BF16 operands in this
probe are exactly representable as FP16; operand conversion alone therefore
does not explain these 69 differences. No full-model FP32 fallback or
quantization search was used.

Raw evidence: `bf16-gemm-capability-001.{build.log,stdout,stderr}`. Reproduce
without profiling or timing as follows (use fresh output files):

```sh
/usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_87 \
  /data/datalog/edgefm-orin/20261021-05/probe_cublas_bfloat16.cu \
  -lcublas -o /data/datalog/edgefm-orin/20261021-05/probe_cublas_bfloat16
/data/datalog/edgefm-orin/20261021-05/probe_cublas_bfloat16
```

Next: implement and independently validate a model-agnostic optional precision
provider/graph lowering for the original BF16 dense boundaries, with no
allocations during enqueue and no model-name branches. Keep the existing
engines and complete negative gate as the baseline. Prove source-op semantics,
actual library execution and complete sixteen-sample outputs before any formal
RDT E2E/CDF campaign. The probe alone is not delivery of this provider.

## Native Text and Raw Input Assets

`t5-tokenizer-source-001/` contains the fast tokenizer JSON generated from the
original SentencePiece assets and all sixteen raw instructions/token fixtures.
JSON SHA256 `fed9dca2339821f6075bd5fda1005673046406947d7ff50e3f7ce01b03fe1c18`.
The existing frozen native Rust/C++ tokenizer provider executes all sixteen
cases on Orin with identical IDs (`t5-native-tokenizer-probe-001/receipt.json`).
The provider static library hash is
`dc69eff52c40ca218bf287c6239e6dc4a25aa7d77563ed42f5fad2a99c81244c`.
This verifies these fixed texts, not arbitrary instructions or raw E2E.

`rdt-raw-observations-001/` restores all sixteen raw observation packs and their
manifests plus the original SigLIP processor configuration. All manifest/input
hashes and decoded tensor shapes were checked: six history-major `[480,640,3]`
uint8 views and `[1,14]` FLOAT native proprioception per sample. OpenCV BGR
arrays are passed unchanged as in the original source; no color-calibration
claim is added. Native image padding/bicubic resize/normalization, BF16 carrier
rounding, native state preparation and integration into a raw-observation E2E
runner are still required.

## Verification and Remaining Gates

- Export/activation/variance/external-payload regression: 38 passed.
- Complete host suite: 2592 passed, 74 skipped, six subtests passed.
- `git diff --check`: passed.
- Existing native CTests are not rerun by these host-only changes.
- Denoiser/output engines and the complete resident ordinary-mode Session now
  execute; the sixteen complete normalized/native comparisons fail the frozen
  source gate. Numerical attribution, source qualification, native raw image/
  state preprocessing, optimized-mode equivalence and formal E2E/CDF remain
  required. No new RDT result is qualified for the paper main table.
