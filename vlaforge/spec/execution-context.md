# Shared Execution Context

Status: generated artifact Sessions automatically bind supported Regions to
one runtime-owned context per execution device. Ordinary execution retains
per-Region synchronization and synchronous Session copies. Eligible opt-in
bounded loops additionally use generated shared-stream or whole-loop replay
callbacks described in [bounded replay](bounded-replay.md); the context
primitive alone is not graph replay.

## Ownership and Compatibility

`execution_context.h` provides a runtime-owned CPU or single-device CUDA
context. A CUDA context exclusively leases one runtime-owned nonblocking stream; the public C header
does not expose CUDA headers. `get_view` requires an output descriptor whose
`struct_size` is initialized. Views copy descriptors and borrow the underlying
stream, not its ownership.

Existing Region executable ABI v1 and value ABI v2 are unchanged. The optional
`VLAForgeRegionExecutionExtensionApi` is a separate versioned provider with a
`bind_context` method. Absence of this provider means the caller must use the
legacy path. `SHARED_CONTEXT` does not imply graph-capture safety, externally
managed workspace, or asynchronous output publication.

The external-plugin loader queries the optional sidecar symbol at open time.
Missing symbols preserve legacy behavior. A present provider that throws,
returns null, or advertises an invalid ABI is rejected, not silently ignored.

Generated Sessions use the AOTI provider or the external-plugin provider when
present. TensorRT currently follows its legacy path. Contexts are allocated
lazily at the first supported Region load, shared by exact device identity
(including ordinal), and retained across invocation-resident Region reloads.
CPU contexts follow the same contract with no native stream. Load failures
release the partially created Region; Session initialization failures release
all Regions before contexts and plugin handles. Normal destruction uses the
same order, including a partially initialized Session. Output-storage allocation
exceptions after Region initialization also trigger this cleanup before the
constructor propagates the exception.

A caller may bind one context to multiple compatible Regions. It must keep
the context alive until those Regions have been destroyed or synchronized
and detached. A null view detaches a context. Context and Region lifecycle
operations are serial, not thread-safe. The runtime restores the caller's
CUDA device after context creation, synchronization, and destruction.

## Stream Lease Lifetime

CUDA context destruction drains the stream before returning its exclusive lease
to a process-lifetime idle pool. A subsequent context on the same device may use
that stream. Concurrent live contexts never share a lease, and device ordinals
are isolated. The pool is thread-safe, but operations on one context remain
serial. All previously borrowed views become invalid when their context is
destroyed even if the physical stream remains cached.

Idle stream identities are retained up to the successful lease-concurrency high
watermark per device. This avoids accumulating backend allocator and BLAS caches
keyed by a never-reused external stream after each Session. It does not promise
zero live allocator bytes: backend workspaces can remain cached. Backend handle
caches may additionally depend on worker-thread identity, so unlimited creation
of threads is not a fixed-resource lifecycle claim. The runtime neither calls
`emptyCache` nor clears global LibTorch workspaces belonging to other users.

Poisoned contexts, device-selection failures, active captures, or failed drains
do not return a lease to the pool or destroy a potentially borrowed stream.
These exceptional resources are retained until process termination. CUDA device
reset and unloading/reinitializing the CUDA context while runtime leases or
cached streams exist are outside this process-lifetime contract.

A CUDA device-selection or stream-drain error from `synchronize` also poisons
the context persistently. A later successful CUDA result cannot make that lease
reusable. Its first call returns the backend error; subsequent status, view,
copy, and synchronize calls reject the poisoned context. Ordinary invalid input
rejection does not poison a healthy context. Previously copied views cannot
check health themselves and must no longer be used after an owner failure.

`test_execution_context_pool.py` compiles the actual runtime with a CPU CUDA
fault-injection shim. It checks reuse, simultaneous lease isolation, device
restoration, cross-device reuse, eight-worker contention, failed creation,
poisoning, failed drain, active capture, and device-selection failure. These
tests are control-flow coverage, not actual GPU memory or timing evidence.

## AOTI Contract

`vlaforge_aoti_region_execution_extension_api` exposes the optional provider.
Only a matching device kind and ordinal are accepted. The descriptor is
copied, so the caller may release or overwrite the descriptor itself after
binding; tensor storage remains borrowed through Region synchronization.

For a bound CUDA context, AOTI package/raw runners and sequence nodes receive
the explicit stream handle. A CUDA stream guard also covers tensor boundary
copies and sequence contiguous materialization, then restores the caller's
current stream. Region `synchronize` waits for the bound stream. Without a
bound context, the legacy device-wide synchronization path is preserved.

An invocation that may have submitted CUDA work marks the Region pending,
including calls that return an error. Context rebinding is rejected until
Region `synchronize` succeeds. Synchronizing the owner context alone does not
clear a Region's pending bookkeeping. Destroying a pending AOTI Region drains
its work before releasing the executable. A caller must establish ordering
for inputs produced on a different stream before invoking a Region.

## Verification

`execution_context_smoke.c` tests C ABI construction, validation, copied
descriptors, detachment, unsupported devices, and CPU-only CUDA rejection.
The existing Region ABI tests continue to exercise unchanged v1/v2 tables.

`aoti_execution_context_smoke.cpp` chains two AOTI Region instances on a
runtime-owned non-default stream without an intermediate Region fence. It
checks repeated numerical results, device mismatch, caller stream restoration,
pending-work rebind rejection, and the legacy path after detachment. It uses
the audit computation `(sin(x) + x.square()) * gain`, not model performance
data. The opt-in pytest entrypoint also checks the executable has no libpython
dependency and runs it with invalid Python environment paths.

`test_execution_context_codegen.py` compiles generated CPU Sessions with both
legacy and sidecar plugins. Instrumented failure injection covers context
creation, Region creation/binding/loading, malformed providers, repeated runs,
output-storage allocation failure, and both residency modes. Linker wrappers verify that context destruction
occurs once and after every Region destruction. Structural tests also verify
distinct CUDA ordinals receive distinct context slots.

## Remaining Work

TensorRT does not yet provide this extension. Ordinary-path Session copies and
per-Region synchronization retain their existing completion contract. Generated
eligible loops select the separate bounded replay runtime, with stable staging
and same-context copies; unsupported loops still fail or follow an explicitly
recorded fallback policy. Backend workspace ownership beyond the supported
LibTorch paths and broader dynamic-loop eligibility require separate work.

## Lifecycle Diagnostics

`tools/diagnose_session_lifecycle.py` rebuilds a diagnostic runner around an
existing frozen benchmark, retaining its model artifacts and runtime source.
For numerical-policy-bearing bundles it reuses the public worker initializer
with the protocol's explicit exclusive-process/calling-thread acknowledgements.
Initialization occurs once after GPU ownership registration and before inputs
or Session lifetimes; it is not a setter inside each Session or inference call.
Legacy unbound protocols retain their unconfigured status.

The prepared diagnostic freezes numerical-worker metadata, per-policy loop
schedules and both original and newly linked runtime-library identities. Each
Session cycle checks complete outputs and actual runtime mappings. The worker
marker and replay counters must cover the declared policy and all cycles.
Allocator observations separate before/create/run/destroy phases. Reusing a
stream or observing a finite plateau does not prove zero allocations, ownership
of backend workspaces or an unbounded no-leak property. Diagnostic timings are
not the formal latency/CDF benchmark.

`--graph-memory-policy source|retain|scoped-reclaim` defaults to the policy in
the source bundle's build metadata. Missing source metadata requires an explicit
choice; it is not silently treated as retain. An override is recorded separately
from the original policy. The diagnostic reuses the bundle builder's SDK gates,
passes ON/OFF explicitly to CMake, checks the resulting cache and freezes the
selection. Scoped reclaim remains opt-in and limited to the audited Torch 2.10
provider. Comparing its memory does not inherit retain-mode timing evidence.
