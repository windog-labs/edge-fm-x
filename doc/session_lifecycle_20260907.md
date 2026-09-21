# Same-Process Session Lifecycle Diagnostic

## Result

**The diagnostic completed; bounded post-destroy memory retention did not pass.**
On the same RTX 3060 used by the original allocator pilot, each of three
independent native processes created and destroyed five Sessions. Every Session
processed all 16 real recovered-statistics SmolVLA observations once. All 240
complete action chunks were byte-identical to both the same-artifact reference
and the official full model. No Python library was loaded into these processes.

After an explicit CUDA drain following each destruction, LibTorch native
allocator `allocated_bytes`, `active_bytes`, and `requested_bytes` had the same
sequence for all three policies:

| Session Cycle | Retained Bytes | Live Allocations |
| --- | ---: | ---: |
| 1 | 9,568,256 | 2 |
| 2 | 19,136,512 | 4 |
| 3 | 28,704,768 | 6 |
| 4 | 38,273,024 | 8 |
| 5 | 47,841,280 | 10 |

The retained amount increased by **9,568,256 bytes and two allocations per
Session**, rather than reaching a plateau during these five lifetimes.
Reserved caching also grew; it is recorded separately from active allocations:

| Policy | Reserved Bytes After Cycle 1 | Reserved Bytes After Cycle 5 | Per-Cycle Increase |
| --- | ---: | ---: | ---: |
| off | 1,925,185,536 | 9,625,927,680 | 1,925,185,536 |
| batch-only | 1,925,185,536 | 9,625,927,680 | 1,925,185,536 |
| required | 1,929,379,840 | 9,646,899,200 | 1,929,379,840 |

This is a confirmed cross-Session retention observation, not yet an ownership
or leak-root-cause attribution. The common growth in all three policies does
not isolate replay as the cause. The generated Session/runtime, LibTorch
allocator stream lifetime, and tensor ownership require a separate controlled
diagnosis. No `emptyCache`, counter reset, or hidden cleanup was used to produce
these results. Do not extend the number of lifetimes blindly: cycle 5 already
retains approximately 9.6 GB of reserved allocator storage on this 12 GB GPU.

## Scope And Evidence

Evidence root, relative to this repository:

```text
artifacts/edgefm-vla-goal/20260906-044509/execution-context/session-lifecycle-v1/
```

- `prepared.json` binds `allocator-pilot-v1/`, not the evolving main runtime.
- `frozen-files.json` verifies 674 source, input, artifact, executable, and
  supporting files. All 44 copied runtime files and 160 input/reference files
  match the original pilot byte-for-byte; generated Session source is unchanged.
- `runs/{off,batch-only,required}/` retains native commands, PIDs, owner
  telemetry, stdout, stderr, five complete output archives, before/after-destroy
  process maps, and five allocator phases for every Session.
- `report.json` reports completed validated observations. Its `status=passed`
  does **not** mean bounded memory retention, zero allocation, or leak freedom.
- `independent_audit.py` and `independent-audit.json` reread all 240 complete
  outputs (288,000 bytes), all 75 allocator snapshots, source hashes, CUDA
  identity, and counter conservation without importing the lifecycle validator.

The physical GPU UUID was
`GPU-ce878329-6b58-5666-292d-94185f5e5585`. Each process started with an empty
compute-owner preflight and was monitored using its own PID. One-second owner
sampling cannot exclude a transient foreign process wholly between samples.
All three processes exited successfully and the GPU was empty after completion.

Caller CUDA input buffers total 118,074,528 bytes and are allocated once outside
the five Session lifetimes. They are not covered by LibTorch allocator counters.
Neither are runtime `cudaMalloc`, CPU heap storage, nor other CUDA library or
backend allocations. `cudaMemGetInfo` is device-wide context, not allocator
ownership evidence. Cumulative counters and peaks remain monotonic across
Sessions and are never reset.

The runner reuses the existing 1+15 call partition only to place observations
after the first call and after the remaining 15 calls. These are not warmup or
performance-measurement phases. Incidental timing fields in raw logs are not
summarized and are excluded from formal latency/CDF evidence. This is a model
tensor boundary diagnostic, not sensor-to-action or Orin/BPU evidence.

## Implementation And Reproduction

The new generic validator is
`vlaforge/python/vlaforge/validation/session_lifecycle.py`; the orchestration
tool is `vlaforge/tools/diagnose_session_lifecycle.py`. The shared benchmark
template has two explicit lifecycle hooks, with empty default replacements in
the ordinary benchmark. Session loop control is not hidden in allocator hooks.
The numerical gates validate complete typed primary and auxiliary outputs;
integer auxiliary values are compared without conversion through floating point.

CPU regression: **164 passed**, including missing cycles, reordered rows,
counter resets, cross-cycle conservation, failed native execution, foreign
owners, Python-loaded maps, and full integer output comparison. Ruff passed.
The JUnit record is
`execution-context/session-lifecycle-cpu-v1.xml` under the artifact run root.

Preparation and execution must use a new empty output label for another
experiment. Reuse the exact frozen pilot, verify physical GPU identity and owner
availability, and preserve this failed bounded-retention baseline:

```sh
PY=/home/zhangzimo/.venvs/vlaforge-validation-py313-20260906/bin/python
BASE=artifacts/edgefm-vla-goal/20260906-044509/execution-context
$PY vlaforge/tools/diagnose_session_lifecycle.py prepare \
  --prepared "$BASE/allocator-pilot-v1" --output "$BASE/NEW-LABEL" \
  --cycles 5 --gpu-uuid GPU-ce878329-6b58-5666-292d-94185f5e5585
$PY vlaforge/tools/diagnose_session_lifecycle.py run --output "$BASE/NEW-LABEL"
```

Hashes for this completed experiment:

```text
report.json
c1418156f96bfbb990dfd31032f25a880022d1edcff7034b4b01a6db00c8ffa6
independent-audit.json
7256bdb8bc7ca65e2d4729021d2b0789362b87572e4642110d8e83fed378beae
frozen-files.json
6d1af347375a214b47685ee061c8049d4e79c2df46adb4f2432bee965b57d11a
```

## Stream Retention Repair

The preceding results remain the unchanged failing baseline. A subsequent
single-file runtime repair was independently frozen and tested in
`execution-context/session-lifecycle-stream-pool-v3/` under the same artifact
run root. Only `runtime/execution_context.cpp` differs from the original 44
runtime files; generated Session code, all models, inputs, and reference bytes
remain unchanged.

Actual allocator block addresses after the original Session destruction all
matched the LibTorch cuBLAS and cuBLASLt workspace maps: 8,519,680 and 1,048,576
bytes respectively, with no unattributed active blocks. A minimal native
program without TS or model inference reproduced the same increment using
only execution-context creation and the official BLAS workspace getters.
Removing those getters gave zero retained allocator storage; retaining the
same context gave a fixed 9,568,256 bytes. This agrees with the fixed
[PyTorch 2.10 BLAS workspace implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp).

The generic runtime now exclusively leases streams from a device-specific idle
pool. Only successfully drained, non-poisoned streams are returned. Simultaneous
live contexts cannot lease the same stream. The pool survives late global
Session destructors, and cached stream identities remain until process exit.
No global workspace clear or allocator cache clear is performed. This fixes
unbounded identity churn at fixed worker and lease concurrency; it does not
claim zero cached storage or bounded resource use under unlimited concurrency
or creation of worker threads.

The new three-process experiment again validated **240 complete action chunks**
against both official and same-artifact references. Independent verification
checked **915 hashes, all 240 raw chunks, and all 75 allocator snapshots**:

| Policy | Active / Allocated / Requested After Every Destruction | Reserved After Cycles 1-5 |
| --- | ---: | --- |
| off | 9,568,256 B | constant 1,925,185,536 B |
| batch-only | 9,568,256 B | constant 1,925,185,536 B |
| required | 9,568,256 B | 1,929,379,840; 1,933,574,144; 1,937,768,448; 1,941,962,752; 1,946,157,056 B |

Thus the stream-keyed active-storage problem is fixed in this measured scope,
but **required-policy reserved memory still grows by 4,194,304 bytes per
Session**. A separate actual five-cycle observation identified five distinct
graph-private pool IDs. Each retained two inactive segments totaling 4 MiB,
with allocated and active bytes both zero. Inactivity is not accepted as a
bounded-retention result. A scoped graph-pool lifecycle repair remains open.

Focused regression results for the stream fix:

- 179 CPU tests passed; one CUDA opt-in test was skipped in that CPU invocation.
- That CUDA test was separately enabled and passed on the RTX. It executes
  real GEMM and both BLAS workspace paths over five cycles at each of one and
  three simultaneous context leases, in a native no-Python runner.
- Ten CPU fault-injection cases compile the actual runtime implementation.
  They cover reuse, simultaneous leases, devices, eight-thread contention,
  creation/drain/capture/device failures, poisoning, and late global destruction.
  The original implementation failed six of the first nine cases; the repaired
  implementation passes all ten. The late-global-destructor test is coverage,
  not a claim that the old undefined behavior reliably failed in this environment.

Evidence for address attribution, minimal controls, the extra 160 complete raw
action rechecks, and the remaining private pools is in
`execution-context/retention-diagnosis-v1/diagnosis-report.json`.

```text
runtime/execution_context.cpp (stream repair, process-lifetime holder)
95591c0edb4c1d82dca279d1be6693f882d7cbffb96bce646584a50a78da16ba
session-lifecycle-stream-pool-v3/report.json
ded3014e1587bbe341a4e41a0ecaeaa8063880ebff1cfe28a4816bd127ce21ad
session-lifecycle-stream-pool-v3/independent-audit.json
59e313aa6f6969582fb08683f14d47610e79046e8eb80d46898972a78fe68933
retention-diagnosis-v1/diagnosis-report.json
ed797ac8512251e88453e5f0097e81fd14bc1c48f2d1c1d65922a2a7f4fbec08
```

The intermediate stream-pool-v2 experiment is preserved separately. It passed
the measured tensor and allocator checks but preceded the late-destruction
holder change; do not assign v3 source identity to that earlier run.

G1 lifecycle memory acceptance remains open for required graph-private pools.
These diagnostics do not update any previous formal latency or CDF evidence,
do not establish a new startup/destruction cost, and do not imply full-paper
acceptance. The root agent's broader CPU suite has a different frozen source
identity and is not relabeled by these targeted tests.

A following scoped-reclaim prototype is recorded in
`retention-diagnosis-v1/graph-scoped-prototype-report.json`. Two simultaneously
live graphs over five actual CUDA cycles returned private reserved memory to
zero while explicitly performing two device frees per cycle. Fourteen complete
small-program outputs, including two injected CUDA-destroy failure processes,
were byte-verified. This is not the integrated VLA provider and does not close
the required-policy gap above. See `graph-pool-safety-proposal.md` in that
diagnostic directory for ownership/version constraints and remaining tests.

## Integrated Scoped Reclaim Variant

The historical failures and prototype above remain unchanged. The subsequent
`execution-context/session-lifecycle-scoped-reclaim-v4/` freezes the integrated
public provider with `VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=ON`. The public
default is still **OFF**. This is a separate memory-policy variant, not a
replacement of the earlier runtime or formal latency measurements.

The variant changes the context implementation, graph provider and one private
checked-cleanup header, plus only the CMake option/version/definition prefix.
The remaining 41 original runtime files are byte-identical; all generated
Session computation and all 160 input/reference files remain unchanged. The
context additionally makes actual synchronization failures persistently
poisoning. Two CPU negative cases first failed against the preceding `95591...`
source and then passed after that fix. Normal invalid-input rejection remains
non-poisoning. This source must not inherit the earlier source hash.

The same RTX, three policies, five Session lifetimes and sixteen real recovered
statistics observations were rerun. **All 240 complete action chunks are
bitwise equal to the official and same-artifact references.** Independent
verification checked 921 frozen hashes, all 240 raw outputs (288,000 bytes),
all 75 allocator snapshots, cumulative counter monotonicity and conservation,
and sampled process ownership.

| Policy | Worker PID | Active / Allocated After Each Destruction | Reserved After Each Destruction | Device Frees Per Destruction |
| --- | ---: | ---: | ---: | ---: |
| off | 1443554 | 9,568,256 B | 1,925,185,536 B | 0 |
| batch-only | 1443728 | 9,568,256 B | 1,925,185,536 B | 0 |
| required | 1443855 | 9,568,256 B | 1,925,185,536 B | 2 |

Every number is constant across the five cycles. The observed required-policy
4 MiB-per-lifetime reserved growth is therefore fixed in this variant and
scope. The 9,568,256 bytes of attributed BLAS workspace remain cached; this is
bounded retention, not zero allocator activity. The required variant explicitly
performs private-pool device frees. Destruction time has **not** been quantified,
and no startup, latency, CDF, board, or full-paper claim is upgraded.

The public provider was also tested in six separate native CUDA processes:
two simultaneously live graphs over five cycles, valid partial captures,
injected executable/raw graph destroy errors, explicit caller poisoning, and
the existing bounded replay success/fallback/invalidated-capture cases. All 27
complete small-program raw outputs were independently byte-verified. After an
injected failure the affected private pool was neither reclaimed nor reused;
a new graph received a distinct pool and produced exact output, as did the
surviving graph. These are injected CUDA API return statuses, not spontaneous
driver faults. The existing `void` destructor ABI cannot return failure to the
caller; stderr records isolation. Graph-destroy failure after a confirmed drain
does not itself poison the stream owner. The quarantine is deliberately not
claimed as healthy bounded memory.

Focused CPU regression: **193 passed, two opt-in CUDA tests skipped** in that
CPU invocation. Actual CUDA evidence above was run separately with the same
public provider and test computation; the opt-in pytest entrypoint itself was
not executed in that invocation. Ruff passed. Evidence:

```text
session-lifecycle-scoped-reclaim-v4/report.json
432aea28ba4a64d1592f0a24b13cf6834de712903bba00feb97a6ecd9b0481ab
session-lifecycle-scoped-reclaim-v4/independent-audit.json
a12861514032773a6ae4079cce0199e9feedf1316caeccec979a55122357f0a8
retention-diagnosis-v1/public-provider-report.json
b9efd7a5e722be2513fd218e5f538bced178f506042b33c717dfd25c1eb48dce
runtime/execution_context.cpp
ae52b5a82fcab7b73827e76ebebb8a042efc812f9bee9e9b2055a67d37c466e0
backends/libtorch_graph.cpp
df55a21b1cd1b642c472529401f4d50df8dd5ef7863204114c4de31c362a0a68
backends/libtorch_graph_cleanup.h
ef3fb5ed59e9738db48acd2ba6ab2c6bdf3fbe8eaeaf3d98035e033a1088a7a7
```

These results establish only the measured same-process lifecycle boundary.
Inputs allocated with caller `cudaMalloc`, other backend allocators, unlimited
worker/concurrency growth, reset-after-first-context, other Torch minors and
other hardware remain outside it. One-second owner sampling cannot observe
every transient foreign process. Existing formal runtime and global CPU-suite
identities are not reassigned to this new experimental variant.

## Public Build Parameter

`build_artifact_compile_bundle` now exposes the generic keyword
`libtorch_graph_memory_policy="retain" | "scoped-reclaim"`. Default `retain`
preserves compatibility; explicit scoped selection checks every participating
CUDA LibTorch backend's declared 2.10 version, followed by the actual CMake/SDK
checks. The bundle includes and hashes `metadata/build_configuration.json` and
records the exact ON/OFF CMake definition. This does not automatically enable
replay or change any model Adapter.

The independent public-build regression compiled a real, small device-neutral
TorchScript archive and generated CUDA-capable Session through this keyword,
with `CUDA_VISIBLE_DEVICES` empty and without initializing Torch CUDA. The
temporary CMake build was deleted by the public builder. The saved actual cache
and provider compilation flags both contained the scoped option; modifying the
bundle's configuration file was rejected by file verification. This is CPU-only
SDK compilation evidence, not a VLA fixture promoted to numerical support.

Twenty parameter/native-build tests passed in the CUDA SDK environment. A
separate CPU invocation passed 84 parameter, contract, DSO-collection,
external-plugin build and timer-validation tests, with the SDK-build opt-in
test skipped there. Evidence is in
`retention-diagnosis-v1/public-policy-build-test-v2/` and its adjacent JUnit XML.

```text
deployment/build.py
aeebe0d6dd1dd2ff8a6c9b98a540a7833e5fa679c355ae63c2f3927d110f7814
public-policy-build-test-v2/test_public_scoped_argument_re0/evidence.json
ea70148a39efc896683ce1b27b77b9f48b64dd4ab22fc449ec0d527fd73dea7a
small build-only bundle manifest
8e7bb9deee45c09aab85622ae6d033088849d0175f6f9668346de7e9fba7a8fb
```

## Destruction Timing Diagnostic

A new runner, rather than the original formal inference benchmark, measures
`api->destroy` and the following explicit CUDA drain with three steady-clock
timestamps. Snapshot collection, process maps and output-file writing are
outside this interval. API-only and post-drain raw nanoseconds are retained
separately; their sum is independently checked against the total.

`session-destruction-retain-v1/` and `session-destruction-scoped-v1/` use identical
runtime source, generated runners, models and data. Only the scoped-reclaim
CMake option differs. Each has three independent loop-policy processes and
five Session lifetimes per process, with sixteen complete real observations
per lifetime. Both independent audits passed: **480 full actions, 150 allocator
snapshots and 1,168 frozen entries per variant**. Required with `retain` again
grew 4 MiB per lifetime; scoped required stayed bounded and issued two device
frees per destruction. Both ordinary policies retained fixed allocator storage.

Total Session destruction plus completion-drain time, milliseconds:

| Loop Policy | Retain Median [Min, Max] | Scoped Median [Min, Max] |
| --- | --- | --- |
| off | 30.108 [26.367, 40.607] | 35.556 [25.378, 39.828] |
| batch-only | 34.344 [26.195, 36.407] | 34.132 [27.127, 37.183] |
| required | 37.610 [36.482, 44.837] | 38.140 [36.289, 46.075] |

These are five descriptive samples per process, one process per policy and
variant, executed retain then scoped. CPU resources were not exclusive. The
overlapping ranges and noise in non-reclaim controls mean the observed required
median difference of about 0.53 ms **does not isolate the cost of two CUDA frees**
and is not evidence of a significant slowdown or speedup. It does quantify the
observed complete destructor boundary. No confidence interval, startup time,
formal inference latency/CDF or default-policy change follows from these data.

```text
session-destruction-comparison-v1.json
84bb7265241c73a994b009096882cd1011056c26dc07220dc06febf6b873f723
session-destruction-retain-v1/report.json
d4159f372798ac6bb94c3da2427b7ccfab391dcfcc3b3969f82287c21954486f
session-destruction-retain-v1/independent-audit.json
c723ed270a1c8b82a8f809eccfb68763afe9d5048bc7a964dee8fef15bd7b916
session-destruction-scoped-v1/report.json
93cc791aab88e9ee07518450574d1c7c185e88af225e187b5bd0cb64f27b078b
session-destruction-scoped-v1/independent-audit.json
bf62f69876b285d13ec2e1f7134f75135836e2eb2b77e80241bf13e0ba6c0967
```

### Historical Helper Paths

The original lifecycle ledgers also named five main-tree Python helper files.
Their exact bytes were verified and copied before subsequent benchmark edits
to `retention-diagnosis-v1/historical-helper-source-v1/`. Original ledgers and
reports were not rewritten. The mapping digest is
`4c0b2a6788e73ce9797371ed383e8f9cd61834ba03c211f71743f3e7c0e1b2f9`.

An explicit historical-source relocation verifier now checks the unchanged
ledger against either its original path or the pinned byte-identical snapshot.
It does not treat arbitrary missing or changed artifacts as acceptable. After
the two main benchmark helpers actually changed, it successfully rechecked all
1,168 scoped ledger entries, resolving only those two exact historical source
identities. Wrong mapping pins, unmapped artifact mismatches and changed
snapshot bytes each failed a separate negative test. This is explicitly a
source-relocation audit, not a claim that the old ordinary same-path verifier
continues to pass against modified main files or that five helpers form a full
historical Python environment.

The new independent raw/counter/timing comparison reads the preserved outputs
directly and remains separate from mutable main-tree code. Commands and source
are retained in `retention-diagnosis-v1/{audit,compare}_destruction_timing_v1.py`
and `verify_historical_helpers.py`.
