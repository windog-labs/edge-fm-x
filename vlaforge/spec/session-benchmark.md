# Tensor Session Benchmark

`tools/benchmark_session.py` samples existing compiled Sessions without a Python
inference runtime. The current protocol is deliberately one CUDA device, fixed
Tensor input profiles. Protocol v1 has one complete FP32, FP16 or BF16 output;
v2 supports the typed complete-output contract in `session-benchmark-multi-output.md`.
The measured unit is
one Session invocation, not a token, an individual action or a robot control tick.

## Frozen Protocol

An explicit `vlaforge.session_latency_protocol/1` JSON file supplies:

- By default three built bundle paths: `off`, `batch-only`, `required`, all
  sharing identical original semantic IR. An explicit nonempty `policies`
  subset permits a backend without replay to measure only `off`; it does not
  imply evidence for omitted policies. Region artifacts are not recompiled.
- Paired input raw-file mappings and full direct/eager reference arrays
  for every sample. The model-specific sample preparation stays outside the
  generic sampler. Every sample participates equally in every process.
- Exactly 128 warmup calls, at least 1024 measured calls, five independent
  processes per policy, explicit quality status and retained source evidence.
- CUDA ordinal, compile architecture, toolchain location and a named boundary.

`prepare` copies runtime sources, generated Session source and sample data into
an isolated directory, builds only the benchmark executables, checks `ldd`,
and records hashes of the protocol, sources, inputs, references, evidence,
executables and original packages. Live sampling/report helper sources are
bound too. `pilot` is separately labeled and never pooled into formal results.
`run` refuses changed bindings and output-directory reuse. The process order
rotates selected policies within each repeat to reduce fixed-order thermal bias.

## Resident Timing Boundary

All sample input tensors are allocated and uploaded once before warmup. Each
call cycles through the fixed sample set, binds its resident tensors, and assigns
every input a strictly increasing explicit revision. Thus context-cache identity
cannot reuse a previous observation merely because an address was reused.

The `steady_clock` interval covers Session run and CUDA completion. It includes
all model work represented by the compiled invocation, such as vision, prefix,
the full generation loop, final validation and transaction commit. It excludes
preload/initialization, input binding, output D2H, metric checks and file logging.
The synchronized pre-call boundary is also outside the interval.

This is a **model-tensor boundary**, not camera/sensor-to-action or robot-system
end-to-end latency. Timed calls/second is the reciprocal of the mean at that
boundary. A separate steady wall interval includes binding, output transfer,
validation and logging. The supervisor's process wall interval additionally
includes initialization and up to one second of polling after child exit, and
is labeled accordingly. Those throughput figures must not be conflated.

Graph warmup/capture happens inside the initial warmup calls. The fixed 128
application warmups are excluded from every latency distribution. CUDA graph
providers may execute additional private warmup steps while preparing the graph;
these are not extra published Session invocations.

## Host Tensor Timing Boundary

Both protocol versions accept the explicit boundary string:

```text
host-model-tensor: H2D, input binding, Session run, completion and complete D2H
```

The default remains the resident boundary. With this opt-in boundary, every
warmup and measured call copies the selected sample's complete host tensors to
the preallocated CUDA buffers, binds a fresh revision, runs the Session, waits
for completion and copies every declared output to host storage. The host input
storage is pageable; any driver staging and synchronization cost is included.
Buffer allocation and initial file loading stay outside the per-call interval.

`host-timing.csv` contains adjacent, non-overlapping `h2d_ns`, `bind_ns`,
`model_ns` and `d2h_ns` segments, plus `run` and `host_call_ns`. The output segment
includes publication/metadata checks and complete copies. Floating/integer
reference comparisons, replay telemetry and raw-output logging follow the end
timestamp. All segments must sum exactly to `host_call_ns`, which must equal
the corresponding main `samples.csv` latency. The sidecar includes warmups.

Process reports bind that sidecar by SHA256. Formal aggregation rejects missing
or changed sidecars, mismatched boundaries, reordered calls and inconsistent
segment sums. It recomputes reported latency from the main CSV. The generic
typed diagnostic renderer exposes the same choice as `host_io=True`.

This boundary starts with already prepared model tensors. It does not include
raw image preprocessing, image file IO or robot transport and cannot establish
sensor-to-action latency. Model-specific postprocessing is included only when
it is already represented in the compiled Session. Host and resident campaigns
require separate frozen evidence and must not be pooled into the same CDF.

## Correctness and Ownership

Every warmup and measured invocation retains its complete output. C++ checks
nonfinite values, exact same-artifact bytes, MSE, max-abs and cosine against
direct and eager references. Zero-norm cosine has an explicit undefined flag,
not an invented perfect score. Required/batch-only runtime counters are checked
on every call; final counters are printed separately. A fatal capture exits the
worker without an unsafe ordinary fallback.

After process exit, an independent NumPy pass checks every archived raw chunk
against the paired direct reference and recomputes eager metrics. Paper numeric
gates and same-artifact equivalence are separate. A failed eager quality gate
remains failed in the performance row, even when all scheduling outputs are
bitwise identical. The tool never promotes a row into a lossless paper table.
An explicitly passed quality status must hold for every archived output or the
run is marked numerically failed. `eager_validation: "bitwise"` additionally
requires complete byte equality against the official reference on every call;
paper MSE/cosine thresholds alone do not satisfy this stronger gate.

Each process checks compute owners before launch and records 1 Hz GPU telemetry:
device UUID/name, driver, temperature, utilization, memory, power and clocks.
If another compute owner appears, the sampler terminates only its own child,
retains the incomplete run and stops. It neither terminates graphics owners nor
changes power, clock, fan or scheduler settings. A telemetry failure also stops
the child instead of continuing an unobserved run.

## Native Session Contract and Lifecycle Probe

`tools/probe_native_session_contract.py` is a model-neutral host-side diagnostic
for a generated Session C ABI. Its default mode checks typed input rejection,
`accepted=false` failure, and preservation of the last committed output. The
opt-in `--lifecycle` mode additionally requires a distinct alternate sample and
checks repeated calls, two-Session interleaving, rejected and accepted episode
resets, post-reset invalidation, post-reset reproduction, and destroy/recreate
stability.

The lifecycle mode compares complete output bytes through the generated C ABI.
It does not time the Session, prove a no-Python deployment, or replace the
formal campaign audit. `tools/audit_native_session_lifecycle.py` independently
rechecks the report's output relationships and binds the report to the bundle,
generated library, and metadata SHA256 values.

## Evidence

Each process emits `samples.csv` in original call order, dtype-specific raw output, independent
`fidelity.json`, `telemetry.jsonl`, execution status and latency report. Aggregation
keeps the five processes distinct and also provides exact pooled empirical CDFs,
nearest-rank quantiles and between-process mean variation. Raw data must remain
available alongside any table or plot.

The local real-model run is an RTX 3060 CUDA-stage result only. It cannot replace
Orin, BPU, H20 or H100 measurements. Sensor preprocessing, calibration, robot
behavior and low-bit deployment remain separate acceptance paths.

## Optional Native Allocator Observation

`allocator_observation: "libtorch-native/1"` explicitly opts a LibTorch AOTI or
TorchScript campaign into a separate diagnostic. The default `"off"` performs
no allocator queries. The initial supported domain is LibTorch 2.10.0's native
CUDA caching allocator; unknown releases and asynchronous allocators fail closed.

The runner records `allocator-snapshots.jsonl` at `before_session`, `after_load`,
`after_warmup`, `after_measured` and `after_destroy`. An uninitialized allocator
before Session creation remains unknown, not a zero-valued measurement. Queries
are outside individual call timing; no counters are reset and no cache is emptied.
Live, peak, cumulative allocated and cumulative freed counters are distinct.
Allocator requests are distinct from device allocation calls. The memory returned
by `cudaMemGetInfo` is a whole-device observation, not process ownership or peak.

Validation rejects duplicate JSON keys, missing/reordered snapshots, unknown
counters, counter regression and violations of adjacent-counter conservation:
`delta_current == delta_allocated - delta_freed`. Each process report binds both
the raw JSONL and recomputed allocator report by SHA256. The allocator report also
binds the execution record and pilot status. Formal aggregation rechecks these
bindings, frozen inputs and the complete raw-counter calculation; missing or
modified evidence is not accepted.

Coverage is limited to the process/device LibTorch native caching allocator. It
excludes the runtime arena, caller `cudaMalloc`, CPU heap and CUDA/library memory
outside that allocator. Therefore `zero_allocation_claim_verified` is always
false, even if a measured interval contains no observed requests. Post-destruction
residuals do not prove either a leak or leak freedom without lifecycle/ownership
evidence. An instrumented pilot is not pooled into a prior latency campaign.

## NAS-Resident AOTI Variants

`build_session_variants.py --materialize-aoti` reuses the public materialized
deployment contract. It extracts each verified input package once under the
new output directory, rebinds artifact and numerical lineage identities, and
packages those payloads for all execution policies. The original package and
its evidence remain unchanged. Existing non-AOTI Regions retain their original
contracts. This excludes `aoti_package_extraction_root` and requires fresh full
model correctness/benchmark checks; it does not inherit old performance data.

Use NAS paths for the output and compiler caches on container hosts with small
root filesystems. `TMPDIR` alone does not redirect all LibTorch native package
extraction. Materialized Regions load their verified NAS files directly.
