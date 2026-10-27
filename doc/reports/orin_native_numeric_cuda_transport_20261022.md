# Native Numeric CUDA Transport

Date: 2026-10-23. Status: **CUDA 12.8 compiled and CPU metadata/linker guards
verified; CUDA runtime and native CUDA 11.x/Orin behavior unverified**.
All new artifacts are under `artifacts/orin/20261022-15/`. Prior `06`, `10` and
checked-cast14 evidence was read or copied, never modified. No real CUDA plugin
was loaded, no GPU/Orin execution or target build occurred, and no package or
global configuration was installed or changed. Parent owns ModelIR/Plan integration.

## Variant And Profiles

The new model-neutral variant is
`shared-plugin/numeric-cuda-transport-tensor-bool/1`, callable ABI 2, backend
`shared_plugin`, target `sm_87`, device `cuda:0`. It exports the existing
`vlaforge_region_executable_value_api` provider. Numeric profile differences
are compiled data, with no model-name branches in the wrapper or numeric core.

| Port | Complete Public Contract |
|---|---|
| Input 0 | CUDA F64 prepared normalized state `[1,32]`, 256 bytes |
| Input 1 | CUDA F32 normalized actions `[1,50,32]`, 6400 bytes |
| Input 2 | CUDA Tensor BOOL `[1]`, 1 canonical zero/one byte |
| Output 0 | CUDA F32 normalized actions `[1,50,32]`, 6400 bytes |
| Output 1 | CUDA F64 native actions `[1,50,14]`, 5600 bytes |

There are three tensor inputs, two tensor outputs and zero scalar outputs.
Scalar BOOL is deliberately not accepted by this CUDA-public-port profile.
All buffers must be nonoverlapping, contiguous, naturally aligned and actual
non-managed device allocations on cuda:0. Metadata and descriptor identity are
checked again before publication. Output/source aliasing is rejected.

The same independently verified CPU numeric core is copied byte-exact:
`postprocess-native.cpp` SHA256
`0baee0192186271e990f6ed1bb6ea87bf1fb32746f0350fc26e7e959562b2be6`.
The pi0 and pi0.5 numeric coefficient headers are likewise unchanged from `06`.
No tensor downcast, neural operation, scheduler change or reference substitution
was introduced. All neural computation remains the parent's TensorRT responsibility.

## Execution And Transfers

`load` validates schema, target, variant and artifact declaration, requires
`FE_TONEAREST`, the caller's current device cuda:0 and actual compute capability
8.7, then allocates one private nonblocking stream and **12264 bytes of pinned
CPU staging**. External workspace is zero. The source numeric core still allocates
its original temporary CPU vectors; no allocation-free claim is made.

`run` checks bindings and rounding, then verifies the correct current device and
calls **`cudaDeviceSynchronize` before reading any producer tensor**. This covers
nonblocking shared/replay producer streams; default-stream ordering is not assumed.
It verifies device pointer attributes, transfers the complete state/actions/BOOL
to pinned staging on the private stream, and establishes copy completion with
`cudaStreamSynchronize` before any host payload read. Canonical acceptance and
the unchanged numeric core are checked on CPU. No output transfer occurs in `run`.

`synchronize` rechecks metadata and FENV, establishes the correct-device producer
barrier again, recopies the BOOL byte and waits before reading it. Only after
acceptance and final metadata/FENV checks does it copy **both full outputs** H2D
and explicitly synchronize the private stream. Successful completion releases the
borrowed descriptors; the next invocation must bind all ports again.

Per successful invocation, the declared transfer volume is:

| Stage | Bytes |
|---|---:|
| `run`: state/actions/acceptance D2H | 6657 |
| `synchronize`: acceptance recheck D2H | 1 |
| `synchronize`: both full outputs H2D | 12000 |
| Total | 18658 |

This also introduces two device-wide producer barriers and three private-stream
completion waits. All transfers, waits, CPU finishing and publication belong
inside raw E2E timing. There is no zero-copy or FP32-neural-fallback claim.

## Failure And Lifetime Contract

Known bad metadata, noncanonical/false acceptance or rejected numeric output
branches occur before H2D publication, leaving both public output buffers untouched.
Such instances are poisoned and cannot be reused. Runtime tests of these CUDA
branches remain pending; the current evidence is source/control-flow preparation
plus CPU metadata guard coverage.

Any failed CUDA API call conservatively returns `BACKEND_ERROR` with explicit
unproven-completion/retention text. A failure during the two-buffer commit may
have modified one or both outputs. **CUDA commit is not hardware-atomic**, no
rollback is attempted, and neither output may be accepted after that failure.

Following checked-cast14, tainted instances retain their stream, pinned staging
and metadata; `destroy` intentionally does not free unproven resources. The caller
must retain the owning Session/arenas, all borrowed buffers and descriptors, and
the DSO until process exit whenever completion is unproven. `destroy(void)` is
not completion proof and cannot report cleanup failure; teardown must be owned
by a caller that preserves this lifetime contract, rather than unconditionally
unloading the DSO or releasing borrowed arenas after an error.
The caller's taint bit must remain sticky after `BACKEND_ERROR`; later failed
precondition returns do not downgrade the lifetime requirement.

There is **no execution-context or capture sidecar**. Both support flags remain
false. This stage belongs **after the bounded solver loop**, never inside its
captured replay. Taint-aware caller lifetime handling and that placement are
integration requirements, not features silently added to the shared runtime.

## Actual Build And Tests

Authoritative build: `build-003/receipt.json`. The production wrapper was compiled
with NVCC **12.8.61**, `-arch=sm_87`, hidden GPUs and strict host FP flags. The
unchanged numeric core was compiled with GCC 11.4.0 using
`-fno-fast-math -ffp-contract=off -frounding-math`. Configuration has internal
linkage; hidden visibility and `-fno-gnu-unique` prevent exported config symbols.
The only intended public provider is the stable value API; no test hook or
execution-context sidecar is exported by the production DSOs.

These are **x86_64 host compilation artifacts linked to `libcudart.so.12`**, not
deployable aarch64/CUDA 11.x Orin libraries. NVCC emits CUDA fatbinary registration
stubs, so the real DSOs were not even `dlopen`-loaded during CPU tests.

| Profile | Path Under `build-003/` | Production DSO SHA256 |
|---|---|---|
| pi0 numeric data | `profile-0/artifacts/numeric_cuda_transport.so` | `ffc7e16c2189e9fa214bc39f828e945606edb930e623364c664dbaf73e47503f` |
| pi0.5 numeric data | `profile-1/artifacts/numeric_cuda_transport.so` | `485ac8f1afbcb8dd4d0099af962867f082c8b7154ef1a56db43dd42e02632830` |

`metadata-tests-001.xml` records **182 passing CPU metadata/linker-guard tests**.
The test-only metadata DSO includes the same production source but links against
a fail-closed CPU guard library: every CUDA entry returns an error, and no CUDA
operation is emulated or forwarded. Tests cover dtype/device/ordinal/rank/shape/
bytes/layout/alignment, scalar and extra-count rejection, overlap, borrowed
descriptor drift, schema/variant/target declarations, FENV preservation and ELF
visibility. Metadata paths require zero CUDA guard calls.

Two isolated child probes additionally verify the fail-closed load path: exactly
one guarded `cudaGetDevice` error poisons/taints the instance, tainted destruction
attempts no CUDA cleanup, and the test caller retains the loader/DSO to process
exit. These are **linker-failure controls, not CUDA lifetime fault-injection tests**.
Real producer ordering, copy parity, acceptance rejection, multi-buffer commit
faults, cleanup faults and native CUDA 11.x target behavior are still unverified.

The failed `build-001` metadata-harness compilation and logs are retained. GCC
rejected an anonymous-namespace subobject type under `-Werror`; the binding type
now has named hidden linkage. No compiler check was disabled. `build-002` remains
an earlier successful compilation; only `build-003` has the final harness tests.

## Registration And Next Step

Each profile has a validated existing `CppArtifactRegionDefinition` in its
`registration.json`, with `supports_execution_context=false` and
`supports_external_cuda_graph=false`. The standalone finishing schema digest is
`0fa111f69aa39e371397eb88739ae1732637e13817eb2e15559b78c58da6dda8`.
It was generated with the existing public `io_schema_digest` API and is not the
parent's final whole-model schema.

Exact host preparation command after parent review, substituting the actual
parent schema and selecting the desired immutable numeric request:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 \
  python artifacts/orin/20261022-15/build_cuda_transport.py \
  --request artifacts/orin/20261022-06/postprocess-request-pi0.json \
  --io-schema-digest "$PARENT_IO_SCHEMA_SHA256" \
  --output artifacts/orin/20261022-15/parent-schema-review-build
```

Parent must rebuild the frozen wrapper/core/profile on the matching native
aarch64 CUDA 11.x toolchain, update actual binary pins, and register the resulting
artifact with backend `shared_plugin`, target `sm_87`, device `cuda:0` and the
variant above. Place it after the bounded loop, supply all public CUDA ports and
implement taint-aware DSO/buffer retention before running target tests. No ModelIR
or Plan integration was applied here. `delivery-001.json` binds the new sources,
compiled registrations, CPU-only tests, prior immutable pins and this report.
