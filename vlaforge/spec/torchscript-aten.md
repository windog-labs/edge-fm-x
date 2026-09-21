# Device-Preserving ATen Regions

`torchscript-aten/1` is an optional backend variant for already exported, static,
tensor-only Regions. It uses the generic TensorRegion contract, Value ABI v2,
Invocation IR, Plan and generated C++ Session. Models do not need a C++ adapter.
It is not a replacement for the legacy CPU-only TorchScript ABI v1.

## Offline Compilation

```bash
vlaforge compile-torchscript region.pt2 --output region.pt --manifest compile.json
```

The Python API `deployment.torchscript_export.export_torchscript_region` also
accepts additional `validation_cases` as positional tensor tuples. All inputs
must have static shape, dtype, device and contiguous layout. Outputs must be a
tensor or flat tensor tuple. Input/persistent-state mutation and implicit RNG
are rejected before tracing; explicit state and noise belong in the IR inputs
and loop carries. Pure invocation-local workspace mutation is audited.

The helper traces an exported Region under inference mode with TorchScript
optimization disabled. It reloads the archive without device remapping and
compares the complete output bytes for each supplied case before publishing a
new artifact. It records warnings, effect audits, compiler and artifact hashes.
Existing output paths are never replaced. Local Region cases are not proof of
full-model parity, unseen-input behavior, latency or no-Python deployment.

For a newly observed, supported v2 numerical context, opt in to a same-precision
compile record:

```bash
vlaforge compile-torchscript region.pt2 --output region.pt --manifest compile.json \
  --numerical-context observed-context.json --target sm_90
```

The corresponding Python API is `compile_torchscript_region(exported_path,
output, reference_context=context, validation_cases=(), target=None)`. It measures
the source, graph and saved archive identities, infers the target from actual
tensor devices, and observes the current numerical context before and after
compilation. CPU plus one CUDA device is allowed; mixed CUDA devices and other
accelerators are rejected. An optional target must match the observed target.
The helper never restores flags, upgrades legacy contexts, or authenticates the
reference run. The caller must supply that provenance and explicitly establish
the reference policy before calling it.

Policy drift or a changed source rejects publication. Existing outputs,
including a concurrent writer's output, are preserved. The returned
`numerical_compile_record` can be consumed by the existing
`RegionNumericalBinding` API. It does not certify a runtime provider or full-model
parity. The CLI places this result under `numerical_compilation`; its default
without `--numerical-context` retains the original Region-only behavior.

The compile record's graph identity describes the program loaded from the
hashed export, using `frontend.exported_graph_digest`. Export serialization can
rename values, so this identity need not equal an in-memory capture digest.
Keep the original capture evidence and export-file hash as provenance; the new
artifact contract must use the actual compile record's graph identity. Do not
rewrite old capture records to make the two digests appear equal.

Device preservation is numerical semantics: changing a CPU zero-dimensional
scalar to a CUDA scalar can change mixed-precision arithmetic. A deployment
contract must describe the original archive devices, not relocate the archive
implicitly. A CUDA archive must be regenerated for a different logical device
mapping unless that mapping has been explicitly revalidated.

## Native Session

Set `ArtifactKind.TORCHSCRIPT_ARCHIVE`, variant `torchscript-aten/1`, and the
capability returned by `torchscript_backend_capability` in the Region contract.
Use the existing `build_artifact_compile_bundle` or `build-bundle` path.
Generated builds enable `VLAFORGE_BUILD_TORCHSCRIPT_BACKEND`; CUDA Regions also
enable `VLAFORGE_TORCHSCRIPT_ENABLE_CUDA`. LibTorch and, for CUDA, a matching
CUDA toolkit/runtime are required. The executable does not require Python.

Each native Region owns its loaded module. Successful reload clears bindings;
failed reload leaves execution unavailable. The backend rejects foreign device
constants, Python execution, asynchronous fork and unmanaged CUDA operations.
CUDA execution uses the selected device's default stream and drains it before
returning. It restores the caller's thread-local JIT optimization setting.
Output count, shape, dtype and device are checked before output copies.

## Optional Shared Context

CUDA deployment may explicitly select variant `torchscript-aten-context/1`
with `torchscript_backend_capability(target, dtypes, shared_context=True)`.
The same device-preserving archive can be used, but the deployment contract
and generated bundle must record the different runtime variant. The offline
exporter's ordinary Region checks do not certify this execution mode.

The generated Session binds its owned execution context before loading each
Region. Loading, execution and output copies use that context's stream. A
bound Region enqueues work without waiting; the Session owns completion before
publishing outputs or state. Pending work forbids context/tensor rebinding and
reload. The caller retains all borrowed buffers and the stream until completion.
An unbound context variant executes synchronously on the default stream.

`off`, `batch-only`, `prefer` and `required` use the existing loop policy API.
The shared LibTorch graph provider is also used by AOTI; its old AOTI entrypoint
remains available. Capture candidates require pure audited effects, fixed
memory/schedule, explicit loop carries and successful warmup/capture. The
provider capability alone is not artifact-specific capture evidence. Valid
partial capture can fall back under `prefer`; an invalidated capture poisons
execution and retains unsafe resources instead of silently running again.

Full-model numerical checks and timing must be repeated for this variant.
Shared context and replay do not establish a numerical-policy provider,
zero allocation, board performance or sensor-to-action timing.

For an existing verified CUDA bundle, `tools/build_session_variants.py` accepts
`--torchscript-shared-context`. It checks the static Session-resident profile,
preserves archive hashes, effect audits, I/O and reference data, records the
old/new contracts, and builds new executables in a fresh destination. This
option never edits the source bundle or certifies capture from capability
flags. All policies use the same selected contracts; run the new bundle's
full-output pilot before formal measurements.

## Explicit Limits

- Session residency only; static contiguous tensor ABI, no scalar Value ABI.
- The base variant rejects shared execution context and external CUDA graph
  replay; the opt-in context variant is CUDA-only and Session-resident.
- Internal ATen allocations and weights remain backend-owned. A zero external
  workspace requirement does not mean zero allocation or full arena ownership.
- Disabling JIT optimization does not enforce process-global TF32, cuDNN,
  attention, threading or other numerical policies. An independently verified
  `RegionNumericalBinding` can require the generic LibTorch numerical provider
  in the generated Session. The worker must explicitly initialize its policy
  under the declared ownership contract before Session creation. Enabling the
  TorchScript context variant neither creates such a binding nor relaxes its
  runtime checks; unbound artifacts do not inherit numerical enforcement.
- Real full-action checks must be rerun for the complete model, target device,
  environment, archived inputs and precision. A passing Region sample is only
  a compilation gate, not a lossless or paper-level claim.
