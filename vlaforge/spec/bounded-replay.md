# Bounded Replay Foundation

Status: runtime API and opt-in Python/IR/Plan/generated Session integration.
Real AOTI CUDA audits cover two model-independent Tensor loop patterns, the
independent runtime harness, and a real SmolVLA checkpoint's full ten-step
generation loop. These correctness audits are not performance experiments
or full-paper acceptance.

## Compiler Selection

`InvocationBuilder.iterate(..., replay="off" | "batch-only" | "prefer" | "required")` records
an explicit per-loop policy. The default `off` does not add an IR attribute,
reserve reset buffers, change prior schema digests, or alter generated source.
The reference interpreter always executes the mathematical ordinary loop.

Deployment may select `compile_module(..., loop_execution=...)` or
`build_artifact_compile_bundle(..., loop_execution=...)`. Its default `source`
leaves source declarations unchanged; `off`, `batch-only`, `prefer` and
`required` override only `vla.for` policy attributes. Bounds, body operations,
carry and scheduler values are unchanged. Overrides apply to all bounded
loops, including unsupported ones; required modes fail rather than silently
skipping a host-dependent or nested loop. Other host control is not rewritten.
`batch-only` and `required` also reject a module with no bounded loop.

An explicit override adds a pass certificate while retaining its original
input semantic digest. The bundle records `metadata/input_semantic_ir.json`,
the existing compiled semantic IR and `metadata/loop_execution.json`, with
all files covered by manifest hashes. Region artifact and exported-graph
identities are unchanged; the execution policy is not an AOTI compile option.
The default `source` adds no override metadata and preserves prior behavior.

`prefer` retains the ordinary path with a diagnostic if compile-time eligibility
fails. `required` rejects compilation for those cases. A candidate still needs
successful runtime warmup/capture; a passed analysis is not graph execution
evidence. `replay_analysis.json` reports that distinction. Generated C++ exposes
`GetReplayInfo(task_id, info)` without changing existing Session C ABI tables.
Compile-time fallbacks report their reason with zero runtime graph counters;
runtime candidates use the bounded replay counters described below.
Generated headers also declare the optional free C function
`vlaforge_model_session_get_replay_info(session, task_id, info)` when any loop
selects a non-off policy; the original Session API function table is unchanged.

`batch-only` requires the same Tensor/effect/device/residency eligibility and
runtime shared-context extension, but does not require graph-provider support.
The generated callbacks, stable seed/staging storage and same-stream copies
match replay. The runtime receives `ORDINARY`, zero warmups and a null graph
provider; every invocation queues all N ordinary steps, then drains once.
Info stays `UNPREPARED`, with zero captured steps/replays and an increasing
ordinary count. That state means graph capture was not attempted, not failure.
This mode isolates queue batching from graph replay in a controlled ablation.

The initial compiler contract is deliberately restricted:

- A nonempty fixed bound of at most 65536 steps and straight-line TensorRegion
  calls followed by simultaneous Tensor yields; no nested/host control.
- Fixed contiguous Tensor shapes for every input, output, live-in and carry.
  The host induction variable cannot feed the body. A Tensor timestep, token,
  index or scheduler carry is allowed with its original Tensor semantics.
- Each participating artifact binding is exclusive to this loop and uses the
  same CUDA ordinal with session residency. CPU scalars, mismatched devices,
  invocation-resident reloads and shared mutable Region bindings are rejected.
- The semantic Region is pure and has no memoization decision in the body.
  Its actual exported `EffectAudit` must also pass, with no explicit RNG or
  lifted mutable state. Declaring mathematical purity alone is insufficient.
- For graph modes, the artifact advertises external graph support and a
  compatible backend graph provider exists. `batch-only` instead requires
  shared execution-context support (external graph support implies it).
  `aoti_backend_capability(...)` supplies both generic CUDA AOTI capabilities;
  old contracts default to unsupported. A provider candidate is not a proof
  that every contained kernel is capture-compatible.

Plan lowering reserves separately owned reset seeds and live-in staging in the
ordinary arena. Their lifetimes span the whole loop and cannot alias initial
values, carried results or simultaneous-yield scratch while active. Generated
code copies current inputs/state snapshots into these buffers once per logical
call, then resets carry contents before warmup/capture/replay. Input addresses
and authoritative state-ring slots may change between invocations without
changing captured addresses. The generated step callback has only Region runs
and same-context copies; multi-carry updates use typed scratch before updating
any carry destination. Adapters do not implement a C++ callback.

Cache lookup, state snapshot/staging, validation, transaction commit/abort and
output publication stay outside the graph. Logical Region trace events are
emitted after completion, not during warmup/capture or from replayed device
work. Episode reset destroys graphs before resetting state; a later call
recaptures. Destruction releases graphs before Regions, weights and context.
On fatal capture errors the generated Session rejects Run, reset, state
initialization and output reads, and quarantines arena/storage and Regions
until process exit. This is intentional fatal isolation, not normal cleanup.

## Public Contract

`runtime/bounded_replay.h` defines a backend-neutral, versioned C interface.
It borrows a runtime-owned execution context, an explicit Region registration
list, fixed boundary storage, and two trusted-host callbacks. No public API
branches on model identity. Existing Region ABI v1/v2 and execution-extension
ABI v1 stay unchanged; graph lifecycle is a separate provider table.

`steps` is a fixed bound from 1 to 65536. Graph modes require at least one
complete warmup. `ORDINARY` runs the callbacks without graph creation.
`PREFER` allows a recorded unsupported-capture fallback. `REQUIRE` returns a
failed-precondition status instead of silently choosing ordinary execution.
Info exposes state, captured step count, replay count, ordinary count and reason.
Warmup calls are not counted as published invocations.

`prepare(user_data)` runs outside capture before every complete warmup, before
capture and before every published invocation. It resets input and carry
**contents** while preserving storage. Queued writes must use the same context
stream. A producer on another stream must establish dependency first. A host
buffer used for an asynchronous write must live until completion.

`step(user_data, step_index, phase)` runs exactly `steps` times during one full
capture, not once for a single-step graph. In the first implementation it may
enqueue registered Region runs and same-context device copies. Backend-private
temporary allocation is allowed only through a capture-aware backend allocator.
It must not synchronize, allocate or rebind boundary storage, mutate persistent
session transactions, read device values on the host, use runtime host branching,
or perform RNG operations. Host logging and counters in tests observe callback
execution only; they do not become replayed graph operations. Phase-specific
branches must not change the successful computation.

Every Region binding, shape, dtype, layout, device, weight allocation and input,
output or carry address stays fixed from creation through graph destruction.
Use fixed device tensors for timestep and scheduler state, with copies/updates
captured as device work. Swapping host pointers between iterations is not a
captured carry update. Input-dependent loop counts, CFG branches, unresolved
RNG state and dynamic outputs must be rejected with a preflight reason until
the compiler/backend has an explicit equivalent contract. Shape bounds alone
do not prove eligibility. These are trusted compiler/host obligations, not a
runtime sandbox or an automatic graph-safety proof.

## Completion and Ownership

Creation validates and binds every registered Region to the same execution
context. A failed create may have already rebound earlier Regions to that same
valid context; the caller retains ownership and must preserve its lifetime.
Repeated Region calls on the same stream may remain pending between steps.
Context rebinding remains forbidden while pending. AOTI Region `synchronize`
detects active capture and rejects it before calling CUDA synchronize, allowing
an explicit unsupported-operation callback to abort a still-valid partial graph.

After full warmup, capture end or ordinary execution, the runtime drains the
context and then calls each registered Region's normal `synchronize`. This
clears pending bookkeeping without changing the ordinary Region contract.
Each published `run` returns only after the stream and Regions complete; output
is usable only after OK. A callback error also drains submitted work when the
capture/context remains valid. Reentrant runs are rejected; concurrent calls
are forbidden, not made thread-safe by this check.

Destroy in this order: replay graph, registered Regions/weights, boundary
storage, execution context, provider/plugin. No Region may be reloaded while
its graph survives. A graph can otherwise reference released AOTI weights or
private output temporaries even though the external binding addresses remain
stable. Shared descriptors borrow resources and do not extend their lifetime.

## AOTI Pool and Failure Policy

The AOTI graph provider uses LibTorch `at::cuda::CUDAGraph`, not bare CUDA
capture calls. It owns the graph and its capture-aware caching allocator pool
until graph destruction. The external stream guard covers begin, end,
instantiate, replay and valid abort. AOTI still forwards the explicit stream
and guards output copying, including when called inside capture.

In Torch 2.10 and newer, both package and raw CUDA `.so` loading explicitly
select the single-threaded AOTI runner. Session ownership already serializes a
callable; the default runner pool can query completion events between calls,
which is invalid inside an external capture. Increasing the runner count does
not establish this contract. The older raw constructor remains source-compatible,
but raw replay on older Torch versions has not been validated.

Preflight rejection or unavailable graph support records a reason without
capture. An explicit failed-precondition callback during a still-valid capture
may fall back only after successful capture end/reset and Region drain. Other
callback errors propagate; they are not reported as successful fallback.

Failed begin/end/abort, graph launch or completion drain poisons the replay and
context. Further owner view export, copy, synchronize and replay are rejected.
Poisoning invalidates all already-borrowed views too; callers must stop direct
Region calls because legacy copied views cannot independently query owner health.
The graph provider receives a poison flag even when only the runtime drain
failed. It quarantines graph/pool objects instead of reclaiming possibly-live
storage. A poisoned context similarly retains its CUDA stream until process exit.
Destroying these runtime wrappers therefore does not mean fatal native resources
were cleaned up.

In pinned Torch 2.10, an error at `cudaStreamEndCapture` can occur before
allocator-pool cleanup, and `CUDAGraph::reset` explicitly does not guarantee
recovery of incomplete capture state. See the
[Torch 2.10 CUDAGraph implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.10.0/aten/src/ATen/cuda/CUDAGraph.cpp).
There is no supported in-place recovery in this implementation. A new context
is necessary for a poisoned context, but it is **not sufficient** if LibTorch
process-global allocator/generator state was corrupted. Exit and recreate the
worker process in that case. Do not attempt an ordinary fallback after fatal
capture invalidation or claim this deliberate quarantine is leak-free recovery.

## Verification

`bounded_replay_smoke.cpp` is a CPU state-machine and copy unit test. Its fake
graph provider tests lifecycle only, not simulated GPU evidence. It covers
ordinary execution, missing backend, explicit rejection, required capture,
full-loop warmup/capture callback counts, reentrancy, prepare/step failures,
valid abort, fatal begin/end/abort/launch/drain and poison-aware destruction.

`aoti_bounded_replay_smoke.cpp` loads a real compiled AOTI package or raw `.so` implementing
`(sin(x) + x.square()) * gain` into two Regions. Four loop iterations execute
eight total Region calls and copy the final output into fixed carry storage.
It checks three changed seeds against eight CPU reference applications, fixed
addresses, completed carry, restored caller stream and pending rebind rejection.
Successful graph execution uses eight warmup step calls, four capture step
calls and zero ordinary step calls for three replays. A preflight fallback
performs no capture; a rejected synchronize during a valid partial capture is
followed by correct ordinary output and a new successful graph on the same
context. Ordinary execution also works after graph destruction.

The raw `.so` audit also runs the exact same artifact against a frozen older
runtime with the default runner-pool setting. That separate worker must reject
capture with zero ordinary fallbacks and exit explicitly after poisoning. Its
failure is regression evidence, not a successful inference or a performance row.

The separate `fatal` process deliberately invalidates CUDA capture with raw
stream synchronization. It verifies poison, zero ordinary invocations and
rejected follow-up calls, destroys runtime wrappers without unsafe native
cleanup, then uses `_Exit(78)`. Exit 78 is an expected fatal-isolation test,
not successful inference. Never run this case inside a shared model worker.

The opt-in pytest entrypoint is
`VLAFORGE_RUN_CUDA_AOTI=1 pytest tests/codegen/test_aoti_execution_context.py`.
It builds the package and executable, checks `ldd` for no libpython dependency,
and runs success/fatal smoke processes with unusable Python environment paths.
CUDA 12.8 / Torch 2.10 / sm86 is the locally validated configuration; additional
devices and backend versions need their own evidence.

`tests/codegen/test_replay_codegen.py` additionally compiles real AOTI packages
from captured Tensor modules and automatically generated Sessions for:

- Continuous generation: `(2, 3)` floating carry and Tensor index, two Regions,
  four steps, explicit changing seed and condition inputs.
- Autoregressive-style generation: `(1, 4, 8)` cache, Tensor token and index,
  two Regions, three steps, retained authoritative state with rotating slots.

Each runs ordinary and required-replay bundles with changed input addresses,
rejected validation preserving the previous publication/state, and episode
reset. The CUDA audit uses no Python runtime for the generated runner, checks
`ldd`, and executes fatal capture tests in separate processes. Parameters are
real compiled synthetic test parameters, not VLA checkpoint evidence. Compared
with eager, locally observed worst max-abs is 7.46e-9 for continuous generation
and 5.97e-8 for autoregressive-style generation. These are same-precision
numerical checks, not a strict losslessness claim. The final audit archives
every full output and confirms ordinary/replay bitwise equality for these
eight published-output reads, while retaining eager cosine/MSE/max-abs.

The real checkpoint audit uses an already captured strict-statistics SmolVLA
fresh-chunk program, four fixed eager-numerics AOTI packages, sixteen episode
samples and two repetitions per policy. It builds generated no-Python Sessions
with compiler overrides `off`, `batch-only` and `required`, without re-exporting
or recompiling AOTI. All 32 complete FP32 `[1,50,6]` outputs per policy are
bitwise equal to each sample's saved same-artifact direct output, and to each
other. Runtime counters confirm ten captured steps and 32 replays for required,
versus zero captures/replays and 32 ordinary runs for batch-only. Each observed
revision changes, so the prefix cache cannot substitute an earlier observation.

Evidence is under the repository artifact directory
`artifacts/edgefm-vla-goal/20260906-044509/execution-context/compiler-loop-policy/`;
the raw Sessions remain under `fresh-smolvla/recovered-spaced16/session-attempts/`.
The fresh tool binds the original captured IR, input pack, capture records,
compile manifests and package hashes to an isolated `--session-label` and
rejects reuse under a changed policy. It records fatal capture failure and exits
the worker instead of pretending ordinary fallback succeeded.

Same-artifact equivalence does not prove strict equivalence to the official
eager model. The upstream eager comparison, preprocessing/calibration caveats
and paper tolerances remain separate evidence. First required invocation
includes two complete warmups and capture; these small correctness-run timings
must not be presented as a fair steady-state throughput or CDF experiment.

## Remaining Validation

Additional checkpoint/model replay coverage, same-platform latency distributions,
peak allocator-pool memory and end-to-end benefit still need measurement.
Scheduler, timestep, CFG and random-state contracts must come from generic
IR/Adapter semantics, not model-name branches. Unsupported explicit RNG,
dynamic control, multi-device or invocation-resident loops remain ordinary or
fail required selection. Quantization is a separate validation lane.

## Experimental Scoped Graph Reclaim

`VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM` is a CMake option, default `OFF`.
Enabling it requires the LibTorch CUDA graph provider and the audited Torch
2.10 SDK. Both CMake and the private C++ adapter reject other minor versions.
The public graph ABI, generated Session contract, and model interfaces do not
change. The option applies equally to eligible TorchScript and AOTI Regions.

The public Python bundle builder accepts
`libtorch_graph_memory_policy="retain"` (default) or `"scoped-reclaim"`.
Scoped selection requires CUDA LibTorch Region contracts and qualified
`backend_versions` entries for every participating `aoti`/`torchscript`
backend. Unknown versions are rejected before compilation; CMake and the C++
header independently check the actual SDK. The selection is recorded in
`metadata/build_configuration.json`, included in the bundle file hashes, and
passed explicitly as ON/OFF in both actual and recorded CMake commands.
Choosing a memory policy does not itself select replay; loop policy and replay
eligibility remain independent.

Each graph owns a distinct `at::cuda::MemPool` keeper. Simultaneous live graphs
never intentionally share a pool token. After a successful stream drain,
checked destruction first destroys the executable graph, then the raw graph,
then releases the graph's allocator reference. Ownership flags are cleared only
after each operation succeeds. The graph object next unregisters generator
references; only then is the private keeper destroyed. That final MemPool
destructor can issue scoped device frees. This is not global `emptyCache`, a
counter reset, or a zero-free policy. It has a real destruction cost and must
not inherit timing evidence from the default policy.

This implementation uses the protected ownership fields of the pinned
[Torch 2.10 CUDAGraph implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CUDAGraph.cpp)
because its ordinary `reset()` only warns on CUDA destroy errors. The keeper
uses the SDK's [MemPool lifecycle](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/MemPool.cpp).
No global warning handler is changed. Every new Torch minor needs a fresh
ownership audit and actual failure/success verification before enabling this
option.

Active or failed capture, an explicitly poisoned execution, a failed drain, or
a checked cleanup exception quarantines the remaining graph and keeper until
process termination. The provider logs `[VLAFORGE-GRAPH-QUARANTINE]` to stderr.
The existing destroy ABI returns `void`, so it cannot report a destructor error
to the caller or poison the owner through its borrowed view. A graph-destroy
failure after a confirmed drain therefore retains its private resources but
does not itself poison the stream owner. That already drained stream may later
be reused; the quarantined graph cannot launch or lend its pool. Context-level
synchronization failures instead have the persistent poisoning contract in
`execution-context.md`. Healthy bounded retention and fault quarantine are
different outcomes.

CPU fault tests compile the actual checked adapter against injected SDK status
results, and separately check the version gates. The optional native test
`test_libtorch_graph_cleanup_cuda.py` covers two live graphs over five cycles,
valid partial abort, complete output bytes, scoped free counters, and separate
injected graph-destroy failure processes. Real fixed-artifact Session evidence
is documented in `doc/session_lifecycle_20260907.md`; it is not a formal latency
or CDF measurement.
