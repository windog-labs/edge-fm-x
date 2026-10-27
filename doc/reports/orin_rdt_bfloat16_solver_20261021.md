# RDT BF16 Solver and Conditions Pilot

Date: 2026-10-21. **Solver stage qualified on saved intermediates; complete RDT
is unqualified. No RDT E2E performance result is reported.**

## Representation

The original BF16 input is unsupported by system TRT 8.5.2.2. The optional,
model-neutral `VLAForgeBFloat16Round/1` plugin instead carries BF16 values in
FLOAT storage and implements BF16 round-to-nearest-even at input/arithmetic
boundaries. It neither adds native BF16 bindings nor implements matrix kernels.
Checkpoint, saved noise, coefficients, dimensions and five timesteps are unchanged.

Generic lowering preserves integers and exact-copy operations; unproven
DOUBLE-to-BF16 casts, random/quantized graphs and nested control flow reject.
Shape inference precedes external weight loading. Source/output external files
are hashed. The default-OFF provider registers before native engine loading;
standalone `trtexec --plugins` loading also passes. Core runtime remains passive
and kernel-free. Architecture audit exceptions cover only this optional
provider's exact source/target and dedicated test, with new negative tests.

## Hardware and Tests

- AGX Orin serial `1421524312594`, L4T 35.4.1, CUDA 11.4, TRT 8.5.2.2.
- Boot ID `3c801f5d-6897-4d97-b021-3c52db38a750`; monotonic durations, wrong
  board civil clock. No global SDK/power change or unrelated job termination.
- Visible GPU owners were desktop processes. Restricted directories remain a
  visibility limitation, not an exclusive-ownership attestation.
- No target Torch/torchvision. Host references use CPU, not 4090 benchmarks.
- Native CTests 20/20; default host CPU CTests 13/13.
- Plugin unit: 327680 values compared to CUDA BF16 RNE, including ties,
  signed zero, subnormals, overflow, infinities and NaN class, plus malformed
  serialization rejection and CUDA replay.
- Final host suite: 2532 passed, 74 skipped, six subtests passed. Initial
  architecture failure and all reruns remain in `full-host-regression*.xml`.
  The boundary was updated and negative-tested, not bypassed by skipping tests.

Initial native configure failures (compiler path, missing test fixture) and the
PATH-only `ctest` failure are retained. Successful receipt:
`bf16-plugin-002/ctest-r1.xml`; source snapshot `native-bf16-plugin-source-002.tar`.

## Solver

| Item | Verified Result |
|---|---|
| Samples | 16 original recorded samples x five original solver steps |
| State feedback | Each preceding real target output feeds the next step |
| Full outputs | All 400 complete stage tensors numerically equal to source |
| Final active unified action | Worst MSE 0; minimum cosine 0.9999999999999998 |
| Engine graph | Three input-rounding plugins and seven arithmetic-rounding plugins |

The tiny solver is an explicit FP32 arithmetic island, not full-model FP32.
Original INT64 count/index values use the target's INT32 ABI: input indices
0..4, output 1..5, valid-count 0..2. Starting at zero, five increments and
saturation cannot overflow INT32. All real states are independently checked.
Numerical equality is not byte equality across BF16/FLOAT or INT64/INT32 storage.

The portable carrier reference also matches all 400 source TorchScript tensors;
source final active actions match frozen H20 actions. However, original saved
denoiser outputs are injected. Online text/vision/denoiser, native-unit output
conversion, complete-model fidelity and latency are not verified by this test.

Frozen identities:

```text
Original solver archive:
273bdd0b68d3c5f4321054ec6db51ab436966e954f64b869d08fc69b7d4c1446
Solver engine:
b160c29d945ab57ad863e900d06eeaeb4d060fcd37e07abc4c1723bc7cbb1243
Required plugin DSO:
7c362ab37bfe96d009fa8ea8d2c978dbe975f1d40a7e3986f07ed5dc73fbde68
```

## Conditions Pilot

Candidate 001 parses but cannot build `/MatMul` under FP16 compute/FLOAT-result
constraints. Separate candidate 002 explicitly requests FP16 dense results.
It builds and runs: seven dense layers inspect as FP16 input/output; all 54
rounding plugins inspect as FLOAT input/output. This adds FP16 result rounding
before BF16 rounding, not an equivalent numerical profile.

The observed difference is consistent with implementation-dependent precision
constraints. The guide provides context, while actual 8.5.2 build/inspection
receipts remain authoritative.
[NVIDIA TensorRT 8.5 guide](https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-tensorrt/developer-guide/index.html).

One original captured tuple is tested. Four inputs/14 weights show maximum
observed absolute value 102, zero FP16 overflow and 48 underflows to zero under
an FP16 cast. These observations do not prove all intermediate runtime ranges.

| Intermediate | Shape | MSE vs BF16 Source | Max Abs | Cosine |
|---|---|---:|---:|---:|
| Language | `[1,6,2048]` | 1.710270e-6 | 0.0078125 | 0.9999936389 |
| Image | `[1,4374,2048]` | 3.550573e-5 | 0.125 | 0.9999889258 |
| State | `[1,1,2048]` | 1.686836e-5 | 0.0625 | 0.9999922714 |

Complete outputs are finite canonical BF16 values in FLOAT storage, but are not
exact. Two intermediate MSEs exceed 1e-5. These are not actions: the complete-
action gate has not been evaluated. Status is `pilot`, not complete RDT `passed`
or lossless evidence. No quantization search or full-model FP32 fallback.

Conditions engine SHA256:
`5d77fe66152064d3a1a4dffbb217f301c7afbdcc297c54a35b93ae3876b94768`.
External source weights were additionally hash-checked after building. The
collector's graph-only hash is not called a prior external-weight attestation.

## Evidence and Next Gate

Root: `artifacts/orin/20261021-02/`.

- `solver-cases-002/manifest.json`, `solver-stage-chain-001/`: complete inputs,
  references, target states/outputs and process receipts for all 80 steps.
- `solver-stage-audit-001.json`: independent full-tensor/state-routing audit.
- `conditions-case-001/`, `conditions-stage-001/`: complete single-case tensors.
- `conditions-stage-audit-001.json`: numerical errors and actual precision audit.
- `solver-carrier-build-001/`, `conditions-carrier-build-001/002/`: successful
  and failed builds. Collector exit 0 alone is not successful engine creation.
- `bf16-plugin-002/`: native build/test receipts and exact DSO.
- `full-host-regression-r2.xml`, `cpu-default-ctest.xml`: final host checks.

Next: integrate T5, vision, denoiser, initialize and finish; run complete fresh
actions on 16 frozen observations and the native raw-observation Session.
Formal five-process E2E/CDF follows only after full-action acceptance. SmolVLA's
existing frozen measurements are unchanged; RDT, pi0, pi0.5 and CogACT remain
incomplete Orin results. All current jobs exited; the full Goal remains active.
