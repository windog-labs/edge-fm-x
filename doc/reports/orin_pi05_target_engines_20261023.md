# Pi0.5 Target Engine Checkpoint

Eight actual Orin TRT engines are built and typed-ABI inspected; complete
VLA inference, D2 and formal E2E/CDF remain **unverified**. Two original native
providers retain public schema/native physical IDs0/2. The ten-region source
composition keeps200tokens/cache968/ten steps/F64 state and native output.
Required-model formal coverage remains3/5, not4/5.

## Engine Builds

All build results are `built-not-validated`, exit0. Seconds below are **engine
construction time**, not inference latency, and must not enter the E2E table.
Same serial1421524312594, boot3c801f5d-6897-4d97-b021-3c52db38a750,
TRT8.5.2.2/CUDA11.4/L4T35.4.1 and unchanged MAXN. Each guarded build has16GiB
admission and2GiB safety floor; no service was stopped for these builds.

| Region | Bytes | Build Seconds | Typed Ports | Source/Receipt Root |
|---|---:|---:|---:|---|
| Prefix part00 | 1407746956 | 228.997609 | 18 | root44/stage000-build-001 |
| Shared visual callable | 484582813 | 139.064520 | 3 | root45/shared-build-001 |
| Original visual/language consumer | 901004425 | 513.254509 | 72 | root45/consumer-build-001 |
| Prefix part02 | 1389687337 | 577.228270 | 95 | root45/stage002-build-001 |
| Prefix part03 | 1458999614 | 585.227090 | 132 | root45/stage003-build-001 |
| Complete action step | 1101543612 | 281.102962 | 41 | root45/step-build-001 |
| Initial time, retained input | 2949 | 3.444535 | 2 | root45/time-build-001 |
| Complete finite predicate v2 | 14877 | 3.734121 | 2 | root50/finite-v2-build-001 |

Roots mean `artifacts/orin/20261023-44`, `-45`, and `-50`. Each stage has the
actual `build.json`, `layers.json`, native inspector JSON and final guard logs.
Context sizes from inspectors are engine requirements, not measured peaks.
Neural dense FP16 and source-declared FP32 islands are unchanged. Six
`*-dense-formats-001.json` reports under root45 check exposed dense I/O and
available weight storage, not hidden accumulation or complete action quality.

## Finite ABI Boundary And Actual Control Outputs

The first uncanonicalized finite graph declared BOOL[1], but the actual engine
inspector returned BOOL scalar[]. That engine is preserved at
`root45/finite-build-001` and is **not** accepted by changing the contract.
This is an observed ABI failure, not a diagnosed neural precision failure;
the individual effect of folding versus reduction lowering is not isolated.
Reusing the existing generic canonicalizer, static
control folder and proved BOOL0/1 INT64-to-FLOAT minimum lowering produces
the already frozen `fp32-control.onnx` bytes. No new model-specific operator,
predicate formula, tolerance or public output shape was introduced.

The parent independently reran
`artifacts/orin/20261023-10/prepare_control_stages.py` into
`prepared-parent-20261023-50`:570 full CPU output comparisons pass across
the original/lowered/control variants. Pi0.5 finite graph SHA256 remains
`807221d25febca0d9f1b1f891358cf99db9228a49ade37d367697a6afbae2117`.
Its newly built engine SHA256 is
`79df1fc55922ea07ae9107e0dc1576524d8cfd558818e8be03cea5efeaa9adb6`,
with actual BOOL[1] output. The time engine retains its1600-FLOAT noise input
and scalar FLOAT output; the old pi0 zero-input assumption is not reused.

`root50/control-output-qualification-001.json` joins120 actual native cases:
60 per control engine, including all1600 input values and first/middle/last
abnormal positions. NaN/Inf/signaling-NaN, finite maxima, subnormals, signed
zero and values beyond FP16 range are included. Entire output bytes match:
time `0000803f`, and correct finite-predicate bytes. Both controls explicitly
disable FP16 and TF32; this is not a whole neural-model FP32 fallback.
The first time probe used an incorrect metadata-SHA argument and exited before
engine loading; its separate setup failure is retained at
`root50/time-native-probe-001`, not merged with the successful002 batch.

## Materialization And Next Gate

All actual engine payloads have host byte/extent checks against their successful
build and actual inspector receipts:

- `root47/cache-001/materialization.json`: regions5-9,5642021145bytes.
- `root47/cache-002/materialization.json`: step/time and the OLD scalar finite.
- `root47/cache-003/materialization.json`: replacement finite region1,14877bytes.

The valid source catalog must select finite from cache003, never cache002's
ABI-rejected scalar. Native providers0/2 use their existing real GPU smoke and
source-bound declarations, not invented capabilities or workspace queries.
The independently checked source proof remains root39/prepared-001, public
schema2cca61bd2132c82d053daba6ea951fa0e1f2d0125dd66db8bbb13a3fece2bd74.

Actual contract/catalog preparation and generic source-compatible Session
generation are under roots49/48. No completed control test or metadata-only
Session preparation establishes a finite neural action, source-BF16 parity,
same-Orin paired action gate, E2E throughput or robot success. Next is the
actual generated Session build, complete fresh raw-observation pilot, then
frozen same-Orin reference/optimized comparison before any formal sampling.
