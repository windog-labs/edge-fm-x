# Scalar F64 Sin/Cos TensorRT 8.5 Plugin Preparation

Prepared 2026-10-22. This is a bounded plugin handoff for the parent five-model
Orin deployment task. CPU compilation and guarded host metadata tests pass.
No Orin connection, GPU execution, runtime installation, frozen runtime change,
numerical qualification, performance measurement, or target replay was performed.
RDT retains ownership of the target. Target validation remains with the parent.

## Changes

- `vlaforge/backends/tensorrt_precision_plugins.cu`: add scalar sin/cos kernel,
  immutable frequency ownership, TensorRT 8 dynamic plugin, creator, validation,
  serialization, cleanup, and optional registration.
- `vlaforge/include/vlaforge/backends/tensorrt_precision_plugins.h`: document
  the public creator and lifecycle contract.
- `vlaforge/tests/cpp/tensorrt_precision_plugin_smoke.cu`: add host metadata
  cases, `--scalar-host-only`, and target-only independent two-stream numerical
  and graph capture/replay cases. Existing BF16 tests remain in the default run.
- `artifacts/orin/20261022-13/cpu_prepare.sh` and `host_registry_guard.cpp`:
  reproducible CPU checks with a test-only registry and linker guards.

The pre-existing optional library/registration entry point is reused, including
its existing CMake option `VLAFORGE_TENSORRT_BFLOAT16_PLUGIN=ON`.
No CMake file changes are needed. Round/MatMul implementations are preserved;
removing only the new includes, scalar section and scalar registration from the
result reproduces the original CU hash exactly. The ONNX matcher and frequency
extraction remain parent-owned. Native numeric output and raw input workers'
files were not modified.

## Creator And Tensor Contract

| Item | Exact contract |
| --- | --- |
| Plugin name | `VLAForgeScalarSinCosF64` |
| Version / registered namespace | `1` / empty string |
| Creator fields | Exactly one field, `frequency_words` |
| Field type / length | `PluginFieldType::kINT32`, exactly `2*N` words |
| Bound | `1 <= N <= 16384` (at most 128 KiB device constants) |
| Word encoding | Signed INT32 low word, signed INT32 high word for each IEEE754 binary64 frequency |
| Input | One FLOAT/LINEAR scalar tensor, static volume one, rank 0 or all-one dimensions through rank 8 |
| Output | One FLOAT/LINEAR tensor, exactly `[1, 2*N]` |
| Output order | First `N` sine values, last `N` cosine values |
| Workspace | Zero bytes |

For each pair of signed words, preserve their 32-bit representations, combine
`uint64_t(high) << 32 | uint64_t(low)`, then `memcpy` the resulting bits into a
double. Do not numerically cast the signed words into frequencies. The creator
copies the data immediately and accepts unaligned parser storage. No DOUBLE
plugin attribute or F64 TensorRT binding is used.

For each `j`, the kernel computes `angle = __dmul_rn(double(time[0]), frequency[j])`,
then `__double2float_rn(sin(angle))` and `__double2float_rn(cos(angle))`, with actual
double overloads. Frequencies are finite immutable constants, including signed
zero and finite subnormals. Arbitrary finite binary64 frequencies are accepted.
Finite frequencies do not guarantee a finite runtime product: nonfinite time
or overflowing `double(time)*frequency` follows CUDA double math semantics;
enqueue does not inspect device input values or synchronize to validate them.

Unknown, missing, duplicate, null, wrong-type, odd, nonpositive, and oversized
creator fields are rejected. NaN/Inf frequencies, malformed ranks/extents,
non-FLOAT/non-LINEAR carriers, wrong output shapes, and dynamic profiles are
rejected. An invalid configure call poisons that instance and its clones.
The input/output buffers cannot be the same address.

## Parent Matcher Handoff

The parent's `vlaforge/tools/lower_onnx_scalar_sincos_f64.py` now emits the
agreed plugin in ONNX domain `vlaforge.trt`, version `1`, with empty plugin
namespace and signed INT32 low/high `frequency_words`. Literal source checks
confirm its `MAX_FREQUENCIES = 16384`; the plugin retains the same bound, with
no stricter admission limit. Its existing static scalar validation accepts
the real FLOAT `[1,1]` input and FLOAT `[1,1024]` output at `N=512`.

The parent reports 13 passing CPU matcher tests and two transformed source
candidates under root08: `pi0-step-sincos-001` and `pi05-step-sincos-001`.
Both have 512 frequencies and identical F64 LUT SHA256
`8c97769281adf81d75a55cd56f1de76bcd2122f4c75a0b66d7d0feab408bd3cc`.
Those frequencies were evaluated from the original source DOUBLE constant
dataflow, rather than recalculated in FLOAT. The test count, transformed graph
results and LUT identity are parent-reported evidence; this worker did not
rerun the matcher tests or reference shadow validation.

The final target ONNX recipe must preserve this order:

1. Apply the BF16-carrier pass.
2. Insert the scalar F64 sin/cos plugin using the parent matcher.
3. Parse/build and validate on the target after the RDT resource reservation
   ends, retaining the agreed plugin field and tensor contracts.

These remain source-level candidates. This handoff adds no TensorRT admission,
target numerical or replay pass. Plugin code, its checked source hashes and the
existing CPU evidence remain unchanged. Updated integration notes and receipt
are also packaged in `scalar-plugin-matcher-aligned-source.tar.gz`; the earlier
prepared archive is retained unchanged.

## Serialization And Lifetime

The version-1 serialization is explicitly little endian, `16 + 8*N` bytes:

| Byte offset | Content |
| --- | --- |
| 0 | UINT32 magic `0x56465331` |
| 4 | UINT32 `N` |
| 8 / 12 | Low / high UINT32 words of FNV-1a-64 checksum |
| 16 onward | Each frequency's low / high UINT32 words, in order |

The checksum uses offset basis `14695981039346656037`, prime `1099511628211`,
and covers the little-endian magic, count and frequency payload, excluding the
checksum itself. It detects accidental corruption; it is not authentication.
Deserialization checks bounded count before size arithmetic, exact length,
finite payload, magic and checksum. Namespace is supplied by the creator.

Each instance/clone has its own CPU vector and CUDA allocation. `initialize()`
and `attachToContext()` allocate/upload once per attachment; a completed default
stream fence follows H2D upload before any nonblocking consumer can use it.
Repeated initialization while attached reuses the allocation. Clones inherit
constants, namespace and validation state; they allocate their own storage.

Enqueue performs descriptor/device checks and a kernel launch on the supplied
stream. It does not allocate, upload, synchronize, or mutate frequency buffers.
Terminate/detach/destruction check the current device and fence all pending
device work before freeing constants. On a wrong device or failed completion
check, the instance is poisoned and its allocation is intentionally leaked
until process exit. It is not reused or freed after unproven completion.
The caller must retire captured graphs before destroying/detaching the plugin;
a device fence cannot prevent a caller from launching an old graph later.

## Completed CPU Checks

The real TensorRT headers were supplied by the parent, remain read-only, and
declare TensorRT `8.5.2.2`. No SDK was installed or modified.

1. Local CUDA 12.8.61 NVCC compiled both CUDA units for `sm_87` against those
   headers, without launching kernels. Expected TensorRT 8 deprecated interface
   and overload-hiding diagnostics appeared. Initial object files are retained.
2. Remote `zzm-4090-x8-1`, private directory
   `/tmp/vlaforge-scalar-sincos-cpu-20261022-o4waKo`, CUDA 12.8.93 NVCC, x86_64:
   `CUDA_VISIBLE_DEVICES` was empty. Both units compiled for `sm_87` with
   `-Wall -Wextra -Werror`, excluding deprecated TensorRT interface and NVCC
   legacy overload-hiding diagnostics. Vendor headers were system includes.
3. NVCC emitted `backend.ptx`; the scalar kernel contains `cvt.f64.f32`,
   `mul.rn.f64` and `cvt.rn.f32.f64`. The sin/cos device implementations operate
   in double; there is no `sin.approx`/`cos.approx` FLOAT substitution.
4. The actual compiled plugin and smoke objects linked with a test-only host
   registry, without libnvinfer. Seven linker guards abort on CUDA device lookup,
   allocation, copy, synchronization, free or kernel launch. Running only
   `--scalar-host-only` passed without reaching those guards.
5. Host cases cover schema/count/null/type validation; finite signed-word and
   unaligned storage roundtrips; borrowed buffer mutation; signed zero,
   subnormal and maximum finite constants; count boundaries 1 and 16384; every
   truncation and every single-bit corruption of the small serialized fixture;
   trailing/overflow lengths; clone state; scalar ranks 0..8; malformed shapes,
   formats/types and profile bounds; and enqueue before initialization.
6. `git diff --check` passed. The unchanged BF16 CU section was checked against
   its pre-task complete-file SHA256 by reconstructing the original text. The
   same reconstruction check also reproduced the original smoke file hash.
   Explicit added-file whitespace checks emitted no errors, and `sh -n` passed
   for the CPU preparation script.

Remote logs and object/PTX outputs are in
`artifacts/orin/20261022-13/cpu-attempt002/`, including the exact checked source
hashes in `source.sha256`. `cpu-attempt001/backend.log` preserves the first
strict-check failure: the vendor header's unused `pluginFactory` parameter.
The second attempt fixes include classification, without changing that header.
The initial `scalar-plugin-cpu-source.tar.gz` records the first-attempt package;
use `scalar-plugin-prepared-source.tar.gz` for the completed handoff.

These checks are CPU compilation and host validation only. The stub registry
does not validate TensorRT registry/runtime behavior. CUDA 12.8 x86_64 object/PTX
generation is not a native Orin CUDA 11.x build or a target numerical pass.
`getOutputDimensions` integration with a real builder remains target work.

The target-only smoke is implemented but has NOT run: independent frequency
vectors of sizes 257 and 3, separate nonblocking streams, ranks 0..8, initialized
clone survival after source destruction, deserialize/attach, CPU double sin/cos
references (at most one FLOAT ULP, signed zeros exact), two graph captures, and
replay after changing time inputs. The default smoke also retains the existing
BF16 MatMul and rounding checks. Smoke tolerance is not a full-model source gate.

## Exact Eventual Target Build And Smoke

After the parent releases the RDT resource reservation, stage the prepared
sources into a fresh snapshot of the parent's integrated tree on Orin and run
these commands from that snapshot's root. Use the installed native CUDA/TensorRT SDK and a fresh build
directory; do not point this at a frozen runtime build or install prefix.
These commands were prepared and have NOT run:

```sh
cmake -S vlaforge -B artifacts/orin/20261022-13/target-sm87-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DVLAFORGE_BUILD_TENSORRT_BACKEND=ON \
  -DVLAFORGE_TENSORRT_BFLOAT16_PLUGIN=ON \
  -DVLAFORGE_TENSORRT_DRIVERLESS_COMPILE_ONLY=OFF \
  -DVLAFORGE_TENSORRT_INCLUDE_DIR=/usr/include/aarch64-linux-gnu \
  -DVLAFORGE_TENSORRT_LIBRARY=/usr/lib/aarch64-linux-gnu/libnvinfer.so
cmake --build artifacts/orin/20261022-13/target-sm87-build \
  --target vlaforge_tensorrt_precision_plugin_smoke -j2
ctest --test-dir artifacts/orin/20261022-13/target-sm87-build \
  -R '^vlaforge_tensorrt_precision_plugin_smoke$' --output-on-failure
```

Equivalent direct smoke command after the build:
`artifacts/orin/20261022-13/target-sm87-build/tests/cpp/vlaforge_tensorrt_precision_plugin_smoke`.
The normal CMake DSO registration is used; do not link `host_registry_guard.cpp`
into the target runtime. After this smoke, parent-owned ONNX parser/build,
serialized-engine reload, real solver replay and fixed full-model source gates
are still required before any deployment or fidelity claim.

## Source Identity

| File | Pre-task SHA256 | Prepared SHA256 |
| --- | --- | --- |
| `vlaforge/backends/tensorrt_precision_plugins.cu` | `447121e451f6cbbc2e97744a5b191009804d08ce07ff8c63f0b4dd57bf01c433` | `b1b18efa89550cf2e28266f66d9299e1d4541bfaf7ea24ae6519419b87c7fa43` |
| `vlaforge/include/vlaforge/backends/tensorrt_precision_plugins.h` | `f1106233b38224d49743446a7cef90c1a46187d138f79801ff75292b9c395108` | `8fe04725c40cfb42668184e176222169b3dbc7fd5d666157d8c006c9197e9e5d` |
| `vlaforge/tests/cpp/tensorrt_precision_plugin_smoke.cu` | `37733ec0375a443aef888fde447f89b994898b69ff7ef15e5fa7ebe1bcb467a6` | `9c23a23448f6b37ec05d715c6f34da0c757d3fd995224897a474deb66dc9b28c` |

The parent's unchanged `trt85-headers/NvInfer.h` SHA256 is
`b8916a5596dc42058d0b71add9055ab07e20711c56f830e2f07e6ea325893d19`.
CPU-check script SHA256:
`ff866fc41c40965ad0282f49751487846009f8112360fdb8dcc70ff43c69c8d4`.
Test-only registry/guard SHA256:
`04a2103c158a2bfc76d12ddebdab1605a76e8d518171cb2f1b659c0bbfda0975`.
