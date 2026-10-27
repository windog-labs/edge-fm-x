# Checked Narrowing Cast Region Preparation

2026-10-22. Status: native CPU reference ABI checked; CUDA source and host wrapper
compiled/linked with hidden GPUs; CUDA/Orin runtime and parent integration
UNVERIFIED. All new implementation and evidence is under
`artifacts/orin/20261022-14/`. Shared production sources and the entire scalar
plugin delivery under `20261022-13` were not changed. No GPU execution, Orin
connection, SDK install, frozen-runtime modification or benchmark occurred.

## Provider And Profiles

`checked_cast_region.cpp` exports the existing
`vlaforge_region_executable_value_api`, returning the stable
`VLAForgeRegionExecutableValueApi` ABI 2. The standard external-plugin loader
is reused. The provider implements only two numeric conversions, with no neural
operations, model-name dispatch, fallback neural inference, or dtype/shape
carrier reinterpretation:

| Operation | Input 0 | Output 0 | Numeric acceptance |
| --- | --- | --- | --- |
| `f64_f32_rne` | Tensor F64 | Tensor F32 | Finite input and finite RNE result |
| `i64_i32_checked` | Tensor I64 | Tensor I32 | Every value in `[-2147483648,2147483647]` |

Each compiled profile declares exactly one input and one output, identical
arbitrary static shapes, one exact region ID, device/ordinal, schema and variant.
Tensor rank 0 is supported; it remains a Tensor value, not a Scalar ABI value.
Zero-element shapes are supported using aligned non-null sentinel pointers.
Dynamic/negative extents, noninteger dimensions, product/byte/address overflows,
and mismatched shapes are rejected. Shape entries are nonnegative int64;
rank must fit the ABI's uint32, and total byte counts must fit size_t/uint64.
There is no fixed model dimension, vocabulary, token length or horizon.

Six examples declare F64 `[2,4]`, I64 `[2,1,3]`, F64 rank 0, empty I64 `[2,0,3]`,
and the first two shapes on CUDA. They are fixtures, not parent model contracts.
`build_checked_cast.py` accepts a new five-field JSON profile:

```json
{"operation":"f64_f32_rne","shape":[2,4],"device":"cuda","ordinal":0,"region_id":0}
```

The builder uses the existing `io_schema_digest` implementation for a fixture
Module. For final registration, pass the actual parent semantic Module digest
through `--io-schema-digest`. Shape/type guards remain compiled into the DSO.
Generated configuration has internal linkage; builds use hidden visibility and
`-fno-gnu-unique`. Symbol inspection of CPU/CUDA DSOs found only the exported
stable provider and no GNU-unique configuration symbols.

## Guards And Workspace

Create/load validate ABI structure sizes/version, region ID, device/ordinal,
exact schema digest, target (`cpu` or `sm_87`), exact variant, nonempty bounded
artifact path with no embedded NUL, artifact hash presence and nonzero byte size.
The verified bundle caller owns actual artifact file size/hash verification;
the provider cannot embed its own self-referential DSO hash. The CPU test adapter
checks the actual file before using the real existing external loader.

Bindings validate Tensor/Scalar kind, nested structure size, input/output index,
dtype, contiguous layout, static dimensions, exact byte count, device/ordinal,
declared power-of-two alignment and actual pointer alignment. Input alignment
is at least 8 bytes; output alignment is at least 4. Run/synchronize revalidate
borrowed descriptors against their bound snapshots, including descriptor
identity, dimension pointer and current extent values. Input, output and the
used workspace must be disjoint. CUDA also checks device-allocation pointer
attributes and the current device before accessing data. Raw allocation
capacity/valid CPU storage remain the caller's responsibility under the ABI.

Workspace is borrowed and static: exactly `256 + 4*element_count` bytes minimum,
alignment 256, on the declared device. Bytes `[0,4)` hold the CUDA validation
flag, bytes `[256,256+4*N)` hold typed F32/I32 scratch. No steady-run provider
allocation occurs. Create allocates one host handle. CUDA load creates one
private nonblocking stream and one 4-byte pinned host status slot; no CUDA device
buffer is privately allocated. The caller supplies all device scratch.

Run validates and stages the entire conversion. Public output remains unchanged
even after a successful run. Synchronize performs the checked commit. Invalid
input, integer range failure, nonfinite/overflowing float result, descriptor
drift or a rejected state transition poisons that instance; later calls cannot
publish old scratch or recover it. A successful commit releases input/output
bindings; each invocation must bind both again. Workspace persists across calls.
One run must be paired with one synchronize. Rebind/reload/run while pending is
rejected. Destruction discards uncommitted scratch.

Operations on an instance are serial. The caller must keep input contents,
descriptors, dimensions, output storage, and exclusive workspace unchanged and
alive through synchronize; mutating scratch or source while pending violates
the borrowed-storage contract. No output or data-content concurrency is claimed.

## Numeric And CUDA Completion Semantics

The CPU reference uses a genuine double-to-float numeric conversion under
`FE_TONEAREST`, with fast math disabled, FP contraction disabled and
`-frounding-math`. It rejects flush-to-zero/denormal-input environments without
changing caller settings. The exact overflow midpoint is `0x1.ffffffp+127`:
its magnitude or larger is rejected. Finite values between FLT_MAX and that
midpoint produce signed FLT_MAX, avoiding an out-of-range C++ conversion. Ties,
signed zeros and subnormals have explicit bitwise reference cases. Rounding
mode is rechecked before commit. I64 narrowing checks the full integer value
before any numeric cast and never wraps, truncates, masks or bitcasts it.

CUDA F64 conversion uses `__double2float_rn`; both source double and result float
must be finite. CUDA I64 conversion checks the same exact int32 limits before
casting. A kernel writes only scratch and atomically flags any invalid element.
The full scratch is admitted only after the flag is copied to the persistent
pinned host slot and the private stream completes successfully.

Before any kernel input read, run executes checked `cudaDeviceSynchronize()` on
the declared current device. This explicitly establishes completion of producers
on nonblocking streams; it does not depend on default-stream copy ordering.
Producer and all validation-stage completion must succeed before a public
output copy can be submitted. The commit is a device-to-device copy followed
by private stream synchronization, so a successful synchronize returns a fully
completed output. No CPU transport of the tensor or CPU neural operation is
hidden in this CUDA profile; only the four-byte validation result crosses D2H.
All fences/transfers/conversion work must be included in parent E2E timing.

On an unproven CUDA completion failure, the instance is poisoned. Destroy does
not free its host handle, pinned status slot or private stream. Caller-owned
buffers, descriptors, workspace and the DSO must also remain pinned until
process exit; the caller must not unload the plugin while work may still refer
to it. This lifetime requirement applies to the parent failure path too.
Ordinary validated rejection has a completed stream and can be destroyed safely.
Normal destruction fences the private stream before releasing its resources.

Public output is untouched for known invalid/overflow/nonfinite input. CUDA
hardware/runtime failure during the final copy may leave an unusable partially
written output: no failure-atomic CUDA commit is claimed. Any such result is
rejected, poisoned and must not be consumed. That failure path has not run on GPU.

This is a **syncing profile outside bounded solver replay**. No execution-context
sidecar is exported. `supports_execution_context=false` and
`supports_external_cuda_graph=false`; no graph capture or replay capability is
claimed. A capture-related CUDA error poisons the invocation; the provider is
not intended to be called from capture.

## Chunk Placement Proof Required From Parent

The original state Cast already present in a source graph can be assigned to
this region once per fresh action chunk only after the parent establishes:

1. The exact source Cast is F64-to-F32, and its complete input is determined by
   fresh-chunk input state, with no timestep, noise, loop-carried value or state
   mutation dependency inside the bounded loop.
2. The source producer dominates every use, and the state remains immutable for
   every loop iteration using that converted value. A fresh chunk invalidates
   the old value even if a memory address is reused.
3. The original F64 state remains separately owned and available to native
   finishing; the cast output has distinct F32 storage. No F32-to-F64 roundtrip
   substitutes for the original state.
4. Parent source-shadow/complete-output checks establish equivalence at the
   final declared shape and source values before this placement is admitted.

The provider performs a cast on every explicit call; it does not cache by pointer,
recognize model names, infer loop invariance, or silently hoist operations.
I64 token projection and int32 acceptance likewise remain explicit parent graph
contracts. Raw input and numeric output workers' sources were not modified.

## Verified Evidence

| Prepared build under `20261022-14/` | Evidence | DSO SHA256 |
| --- | --- | --- |
| `build-f64-cpu-002` | Real CPU ABI, 1024 assertions | `4e7682cc6bf4d9d41ead47a26dcd8d6ba5e58231569329b96493d8219db469c8` |
| `build-i64-cpu-002` | Real CPU ABI, 741 assertions | `c409887362b2eebc95a9bfb05d399bd18ad1afa2898f4ea3f8cd22d4192bb25e` |
| `build-scalar-cpu-002` | Real CPU ABI, 922 assertions | `49b5367594e9361969e14f4924186df4f342e6a44dc2044ea63e9289461f1dc4` |
| `build-empty-cpu-002` | Real CPU ABI, 658 assertions | `720cc247700aef7e5531893123160d3409383c104ba0c7a21d813fdec9a2b480` |
| `build-f64-cuda-compile-003` | CUDA/host compilation and linking only | `1b1c63b84e22d14504cd906080d195d8b2336334c7850f838d07d9bb3ca097e6` |
| `build-i64-cuda-compile-003` | CUDA/host compilation and linking only | `b241d6fd1232988d5c556f88142238f1fbafb83d2879a0707c57f857d167948b` |

There are four passing CPU executable runs with **3345 assertions**, not 3345
independent tests or observations. They verify artifact file hash/size before
the real external loader, stable ABI, descriptor/shape/device/type/layout/
alignment guards, schema/target/variant rejection, byte/address/workspace
overflows, pending/poison state, aliasing, instance isolation, untouched source,
run-without-publication, and preservation of previous output on rejection.
F64 tests cover ties, signed zeros, tiny/subnormal/underflow values, the overflow
midpoint, NaN/Inf, non-RNE modes and FTZ/DAZ. I64 tests cover exact int32 limits
and rejection of both int64 extremes. Profile validation has five passing
unittest methods with dimension/byte-overflow/enum/unknown-field subcases.

Final receipts contain every build command, frozen source/header hashes,
configuration hash, schema, registration and library identity. Compile-only
CUDA used hidden GPUs, GCC 11.4.0 and NVCC 12.8.61 on x86_64 with `-arch=sm_87`.
Those DSOs are **not AArch64 Orin binaries** and are not target admission or a
CUDA numerical, nonblocking-producer, poison/failure, or timing result.

Earlier build directories are preserved. CUDA compile attempt `001` compiled
and linked but failed registration validation because its target was `cuda`;
the existing contract requires an SM target. Attempt `002` corrected that to
`sm_87`; final `003` additionally uses the load-time pinned host validation slot.
CPU `001` results remain intact; `002` adds guard/rounding checks and snapshots
the final wrapper. No expected numeric result was relaxed.

## Exact Build And Registration

From the integrated repository root, a reproducible CPU reference build/test is:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
  python artifacts/orin/20261022-14/build_checked_cast.py \
  --profile artifacts/orin/20261022-14/profile_f64_cpu.json \
  --output artifacts/orin/20261022-14/review-f64-cpu --cpu-tests
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python artifacts/orin/20261022-14/test_profile_contract.py
```

The exact **eventual native Orin build**, after the parent releases RDT resources,
uses the installed CUDA SDK without a runtime install. Run this on Orin with the
parent's reviewed CUDA profile and actual semantic schema digest:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
  python artifacts/orin/20261022-14/build_checked_cast.py \
  --profile artifacts/orin/20261022-14/profile_f64_cuda.json \
  --io-schema-digest "$PARENT_IO_SCHEMA_SHA256" --region-name checked_state_cast \
  --cuda-root /usr/local/cuda \
  --output artifacts/orin/20261022-14/parent-f64-cuda-native
```

The example profile must be replaced with the actual declared state shape/region
ID. Select `i64_i32_checked` and the actual token shape for the token bridge.
Omit `--cpu-tests` for CUDA: this builder deliberately never executes GPU code.
Native target load/functional tests are parent-owned and must exercise actual
nonblocking producers, invalid final elements, producer/validation/commit failure
handling and unchanged full outputs against source references. They have not run.

Each successful build emits `registration.json`, validated by the existing
`CppArtifactRegionDefinition`, and a bundle-relative
`artifacts/checked_cast_region.so`. Use the existing codegen adapter:

```python
registration = CppArtifactRegionDefinition(**registration_json)
artifact_regions[registration.region_name] = registration
sources = generate_cpp_session(
    plan, semantic_module, artifact_regions=artifact_regions, validators=validators)
```

CUDA variants are `shared-plugin/checked-cast-cuda-sync-f64-f32-rne/1` and
`shared-plugin/checked-cast-cuda-sync-i64-i32-checked/1`, target `sm_87`, device
`cuda:<ordinal>`, Session residency, callable ABI 2. CPU reference variants use
`cpu-ref` instead of `cuda-sync`, target/device `cpu`. Both capability flags
remain false. Parent owns Plan placement, I/O projection, original F64 lifetime,
actual schema and registration, and all target/source qualification.

## Source Hashes

| Source | SHA256 |
| --- | --- |
| `checked_cast_region.cpp` | `3c0181d253a4ec9358e261ef6124979aebfd0cf9e59cecb03222ffbe13a5de4c` |
| `checked_cast_cuda.cu` | `0e91eb95f2a1c73a339ca4c5c3d6cb2215810e3ba440b8127f2c012439c0463a` |
| `checked_cast_cuda.h` | `2a71b1c486bcfe4f1a94a41a881a8c1d8d6a5fcb7aeb0bd33e335cf8a8a186fc` |
| `build_checked_cast.py` | `82976ba7824f5f608ba144f489b07fbd417281db7b5081e7ae2e4bfca49d2eaf` |
| `checked_cast_cpu_smoke.cpp` | `20ea3c27a88cb2282820d070f4c11a44a884c677e3e8bb10b5ba01b39a8cacae` |

The prior scalar CU/header/smoke hashes remain, respectively,
`b1b18efa89550cf2e28266f66d9299e1d4541bfaf7ea24ae6519419b87c7fa43`,
`8fe04725c40cfb42668184e176222169b3dbc7fd5d666157d8c006c9197e9e5d`,
and `9c23a23448f6b37ec05d715c6f34da0c757d3fd995224897a474deb66dc9b28c`.
