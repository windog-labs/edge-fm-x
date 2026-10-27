# Native Numeric Region Preparation

Date: 2026-10-22. Status: **real external-plugin CPU ABI verified; target/Plan
integration prepared, not run**. All new implementation/evidence is under
`artifacts/orin/20261022-10/`. No prior `06` sources, reports, manifests or
exports were modified. No shared production edit, GPU execution, Orin connection,
installation, global environment change or performance experiment occurred.

## Actual Plugin Contract

`numeric_region_plugin.cpp` exports the existing
`vlaforge_region_executable_value_api` symbol and implements
`VLAForgeRegionExecutableValueApi` ABI version 2. The verified numeric C++ core
and generated coefficient headers are copied byte-exact from the immutable
`06` delivery into each build's frozen source snapshot. There is no model-name
dispatch; profile differences are compiled numeric data.

| Direction / ID | Contract |
|---|---|
| Input 0 | prepared normalized state, CPU F64 `[1,32]`, 256 bytes, alignment >= 8 |
| Input 1 | complete normalized actions, CPU F32 `[1,50,32]`, 6400 bytes, alignment >= 4 |
| Input 2, tensor variant | CPU Tensor BOOL `[1]`, exactly 1 byte, canonical 0 or 1 |
| Input 2, scalar variant | Scalar BOOL, exactly the declared scalar ABI; canonical 0 or 1 |
| Output 0 | complete normalized actions, CPU F32 `[1,50,32]`, 6400 bytes |
| Output 1 | complete native actions, CPU F64 `[1,50,14]`, 5600 bytes |

There are exactly **three inputs and two tensor outputs**, with zero scalar
outputs. Tensor/scalar acceptance forms have separate schemas and variants;
neither is silently converted into the other. Tensor storage must be contiguous,
properly aligned, fully sized, nonoverlapping and valid through `synchronize`.
No F64/F32 conversion is introduced at these ports. Output 0 is a complete
byte-preserving F32 copy, not a neural operation or FP32 neural fallback.

`load` checks callable ABI, declared schema digest, target `cpu`, exact variant,
artifact descriptor/path/hash presence and nonzero size. The standard bundle
caller remains responsible for verifying the actual `.so` size/SHA256 before
`dlopen`; the test adapter performs that check before calling the **real existing
external loader**. The plugin does not invent self-hash enforcement.

## Lifecycle And Numeric Guard

`run` validates all borrowed descriptors, dimensions, byte counts, dtype,
device/ordinal, layout, alignment and aliasing. It snapshots complete normalized
actions into CPU staging and calls the unchanged source-faithful arithmetic core.
Both public output buffers remain untouched, even after successful `run`.

`synchronize` rechecks descriptor identity, source acceptance and `FE_TONEAREST`,
then commits both completed CPU buffers. This is atomic at the serial Region
invocation boundary: all possible validation/numeric failures precede the two
non-failing CPU memory copies. It is not a concurrent hardware-atomic multi-buffer
store. The normal stable-ABI serial ownership rule applies.

Load/binding/run/commit violations and numeric/source rejection poison that
executable. Later calls cannot recover it or publish stale staging; destruction
and recreation are required. A pending invocation cannot run/rebind again.
Destroying a pending invocation discards both staged outputs. After successful
synchronization, all borrowed descriptors are released; every invocation must
bind all three inputs and both outputs again. Independent instances do not share
poison state or staging.

The production plugin checks `std::fegetround() == FE_TONEAREST` at create,
load, run and before commit. It never calls `fesetround` or changes a process
numeric setting. Negative tests alter and restore only the test caller's rounding
mode. Builds use `-fno-fast-math -ffp-contract=off -frounding-math` for both the
wrapper and the numeric core. Workspace query returns zero external workspace;
wrapper staging is allocated at load. The preserved core still allocates its
own temporary numeric vectors, so zero workspace is not an allocation-free claim.

## Verified Files And Tests

The authoritative build is `build-003/receipt.json`. Its four real shared
libraries and ready-to-review `CppArtifactRegionDefinition` JSON records are:

| Profile / Acceptance | Path Under `build-003/` | Library SHA256 |
|---|---|---|
| pi0 / Tensor BOOL | `profile-0-tensor/artifacts/numeric_region.so` | `994169c0820b7272ac6999692e537a508137582db0068d76fe1d9769fad2de68` |
| pi0 / Scalar BOOL | `profile-0-scalar/artifacts/numeric_region.so` | `28f9123eaaf48ca0bbc8fb3b3c517b2f741628f5d6c5cc2f6128a650901d4df9` |
| pi0.5 / Tensor BOOL | `profile-1-tensor/artifacts/numeric_region.so` | `d42f238c0f46668f2b09d9ec4d47a628bb299568cd3c145a03cf4bb0ab601328` |
| pi0.5 / Scalar BOOL | `profile-1-scalar/artifacts/numeric_region.so` | `08c26ff7fbd6920bf75052656a93440c3b127ef24392a9d778c604db750e618d` |

All four libraries are loaded through the frozen existing
`vlaforge_external_region_plugin_open`, not by bypassing the external plugin ABI.
The standalone test loader also probes actual C++ ABI structure sizes against
the ctypes test adapter. That loader includes test-only rounding controls and
is not a production runtime delivery.

`validation-002/receipt.json` records **256 complete dual-output comparisons**:
the previously verified 128 source reference pairs through both acceptance
contracts. All 409600 normalized F32 values and 179200 native F64 values match
their own complete frozen references byte-for-byte; native MSE/maxabs are zero.
This is duplicated contract coverage, not new independent observations or model
inference. Every case verifies that successful `run` alone publishes nothing.

`cpu-abi-tests-003.xml` records **213 passing CPU tests** covering create/load
contracts, tensor/scalar distinctions, noncanonical booleans, metadata/shape/size/
layout/alignment/device guards, missing/extra outputs, overlap, full-width finite
checks, poisoned reuse, pending destruction, per-instance isolation, borrowed
descriptor drift, late predicate/rounding drift and retention of both previously
committed outputs on a subsequent rejection. All three non-nearest rounding modes
are rejected without changing the caller's chosen mode.

The earlier `build-002`, `validation-001` and `cpu-abi-tests-001` results remain
intact. That validation stopped after 64 cases and the first suite reported 64
failures because GNU-unique inline schema/variant symbols were interposed between
libraries. Configuration data now has internal linkage; plugin builds use hidden
visibility and `-fno-gnu-unique`, with the stable provider explicitly exported.
The final symbol audit confirms no GNU-unique configuration symbols remain.
No gate or expected output was relaxed to obtain the passing results.

## CUDA Boundary

**CUDA transport is not implemented or claimed.** CUDA creates/bindings are
rejected before data access. No CUDA runtime dependency, execution-context
sidecar, shared-context capability or capture capability is advertised.
`supports_execution_context` and `supports_external_cuda_graph` are both false.
The stage belongs **after the bounded solver loop**, never inside captured replay.
All neural operations remain the parent's TensorRT responsibility.

A later CUDA-public-port transport must use preallocated CPU staging and explicitly
establish producer completion before its first host read. In legacy mode, an
explicit correct-device completion barrier such as `cudaDeviceSynchronize` is
needed to cover nonblocking shared/replay producer streams. `CopyBytes` completion
of the default-stream copy alone does **not** establish completion of those
producers. Default-stream ordering must never be assumed.

For Tensor BOOL with all three inputs on CUDA, such a wrapper would add 6657
bytes D2H and 12000 bytes H2D per invocation, plus explicit producer/copy completion
and the CPU finishing work. Actual placement may retain state or outputs on CPU;
that must be declared rather than inferred. Every added transfer and barrier
belongs in E2E timing. There is no zero-copy, transport parity, CUDA failure
atomicity, target compilation or target-support result in this delivery.

## Exact Registration Step

The current standalone tensor/scalar schema digests were generated with existing
`vlaforge.ir.serializer.io_schema_digest`, rather than a custom schema algorithm:

```text
Tensor BOOL: 1a845cf0687679ee3882fde3ea33d6eae8b89aff307fc8e594482cddff44a090
Scalar BOOL: 755368f3894b40d6e393668bf264ca5f637b4d8d4bf2ce0d1cd8aa40cac0bd22
```

These are finishing-fixture schemas, not the parent's integrated whole-model
schema. After review and final parent Plan construction, rebuild using the
parent's **actual** `io_schema_digest(semantic_module)` and chosen acceptance:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
  python artifacts/orin/20261022-10/build_numeric_region.py \
  --request artifacts/orin/20261022-06/postprocess-request-pi0.json \
  --acceptance tensor --io-schema-digest "$PARENT_IO_SCHEMA_SHA256" \
  --output artifacts/orin/20261022-10/parent-schema-review-build
```

For pi0.5 select its immutable request file; the wrapper source stays identical.
Each build writes a validated `registration.json` with backend `shared_plugin`,
callable ABI 2, target/device `cpu`, session residency and exact artifact hashes.
The variant is `shared-plugin/numeric-cpu-tensor-bool/1` or
`shared-plugin/numeric-cpu-scalar-bool/1`. Place the pinned `.so` at the record's
bundle-relative `artifacts/numeric_region.so`, then use the existing adapter:

```python
registration = CppArtifactRegionDefinition(**registration_json)
artifact_regions["native_numeric_finish"] = registration
sources = generate_cpp_session(
    plan, semantic_module, artifact_regions=artifact_regions, validators=validators)
```

Parent owns that Plan region, CPU/CUDA copy placement, actual schema and production
registration. No such integration was silently applied here. `delivery-001.json`
binds this report, final libraries, source snapshots, registration data and tests.
