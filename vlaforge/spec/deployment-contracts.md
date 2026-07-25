# VLAForge Deployment Contracts v0.2

## Session boundary

The generic C ABI version is `VLAFORGE_SESSION_ABI_VERSION == 2`:

```text
bind_tensor(session, input_id, BoundTensor, InputStamp?)
bind_scalar(session, input_id, ScalarValue, InputStamp?)
run(session)
read_output_tensor(session, output_id)
read_output_scalar(session, output_id)
reset_episode(session, new_episode)
```

Every generated model also exposes `InputId`, `OutputId`, `ModelInputs`,
`ModelOutputs`, and `ModelSession::Run(const Inputs&, Outputs*)`.
`io_schema_digest` and stable IDs are embedded in source and bundle; mismatch is
a hard initialization error.

Artifact-backed models additionally expose
`vlaforge_model_session_create_from_bundle(bundle_root, ...)`. Initialization
resolves only normalized bundle-relative paths and verifies regular-file
containment, byte size, SHA-256, callable ABI, I/O schema, backend variant, and
target compute capability before the first `Run()`.

Bindings are push-only and borrowed until `Run()` returns. The Session never
pulls sensors or retains/frees host buffers. Compatible CPU/CUDA buffers can be
zero-copy; mismatches require an explicit preprocessing Region or fail.

## RegionExecutable

The Tensor-only compatibility ABI remains version 1. The current Tensor/Scalar
value ABI is:

```text
VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION == 2
vlaforge.region_executable/2
```

Both use create/load/query-workspace/bind/run/synchronize/destroy function
tables with `struct_size` and `abi_version`. ABI v2 binds `VLAForgeValueView`
whose kind is statically Tensor or Scalar. Host objects such as protobuf,
ROS/Cyber messages, `std::any`, and arbitrary `void*` payloads are forbidden.

External preprocessing such as NV12/RGB conversion, resize/normalize, point
cloud/BEV encoding, CAN packing, and customer features must end at a static
Tensor/Scalar Region boundary.

## Artifact and bundle

- Region artifact schema: `vlaforge.region_artifact/3`.
- Input/output schema: `vlaforge.io_schema/2`.
- Compilation certificate: `vlaforge.compilation_certificate/2`.
- Compile bundle: `vlaforge.compile_bundle/4`.

The bundle records:

- Semantic IR and Scheduled Plan digests;
- exact input/output schemas, stable IDs, groups, and `io_schema_digest`;
- state reset/retention schema and four-class memory plan;
- immutable Region artifacts with SHA-256, byte size, backend, variant,
  callable ABI, model/upstream/checkpoint/graph identity, Region input/output
  signature digests, target, and workspace;
- generated sources/binaries and toolchain provenance;
- exact-cache dependency certificates and loop-invariance dispositions.

A compiled bundle cannot accept unknown inputs. New model inputs require
Adapter/schema change and recompilation unless a fixed optional extension port
was declared at compile time.

## Compiler profiles

- `off`/`conservative`: no cross-Run derived cache, no verified loop
  invariance rewrite, conservative temporary allocation.
- `verified`/`auto`: exact cache only with complete
  InputRevision/StateSnapshot identity, bounded loop analysis, and static-arena
  reuse.
- `force-on`: test-only and explicitly marked in the certificate.

Guarded approximate reuse is a different contract and is not enabled by exact
memoization.

## Clean deployment acceptance

A release artifact must configure/build in a clean tree, run with invalid
`PYTHONHOME/PYTHONPATH`, and show no Python library in `ldd`. Host tests cover
schema mismatch, required/optional/default input, Tensor+Scalar binding,
shape/dtype/device/layout failure, borrowed binding consumption, revision
hit/miss, typed/generic output parity, external Region ABI, state commit/abort,
and episode reset. Orin execution is separate hardware evidence.

The production AOTI path may bind CUDA input tensors zero-copy and copy a CUDA
artifact result into a statically declared CPU output contract. This device
transition is explicit in the Region artifact value contracts; it is never
silently inferred from an arbitrary host object.
