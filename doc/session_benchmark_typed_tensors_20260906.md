# Typed Session Benchmark Contract

The shared `tools/benchmark_session.py` measures a resident-CUDA-tensor
boundary. Sensor decoding, tokenization, input H2D, output D2H, fidelity
calculation and serialization are outside each `Session::run` latency.
The separate full steady-loop wall measurement includes output verification
and storage. Neither boundary is a board or robot-control experiment.

## Storage And Quality

- Input dtype and static shape come from the bound Module, not file size
  guesses. Supported input storage is FP32, FP16, BF16, INT64, INT32 and BOOL.
  Every binary has exact length; floating inputs must be finite and BOOL
  bytes must be zero or one.
- The single complete output may be FP32, FP16 or BF16. `tensor-contract.json`
  records dtype, shape, little-endian byte order and complete byte count.
  The C++ runner also verifies runtime dtype, shape, layout and device.
- `outputs.f32`, `outputs.f16` and `outputs.bf16` contain untouched storage,
  including every warmup and measured call. BF16 is exactly two bytes per
  element. It is never interpreted as an FP32 byte stream.
- FP32 and FP16 NPY references must have the declared dtype. BF16 references
  use FP32 NPY representation, accepted only when all low 16 bits are zero
  and BF16 encode/decode reproduces every FP32 storage bit, including signed
  zero. No tolerance or implicit rounding repairs a reference.
- C++ and Python independently decode storage to FP64 for metrics. The
  mandatory same-artifact gate remains equality of all original bytes.
- Optional `active_dimensions` selects unique last-axis indices for the
  primary MSE/cosine gate. Complete-storage metrics and equality remain
  separately recorded, so padded dimensions cannot dilute acceptance.
  Near-zero cosine is undefined, not automatically one. Raw arrays and
  per-call MSE, RMSE, max-abs, norms and gate outcomes remain available.

`bundle_metadata_mode: ordinary-source` supports bundles generated through
the public source-policy API without the CLI selection manifest. It requires
an explicit `off` label and checks that every scheduled loop is actually off.
It does not infer replay support or rewrite the bundle.

## GPU Identity

An isolated CUDA mapping binds `monitor_gpu` and `cuda_visible_devices` to the
same physical UUID, with logical `gpu_ordinal: 0`. The default process-PID
monitor remains compatible with non-container hosts.

Containers whose NVML PIDs use an invisible host namespace may explicitly
select `owner_identity_mode: cuda-registration-handshake`. After the idle
preflight and before any input tensor, allocator or Session is created:

1. The child registers its primary context and waits at a file barrier.
2. The supervisor requires exactly one GPU owner and records its NVML PID.
3. The child resets only its own unused primary context. The supervisor
   requires that owner to disappear.
4. The same child registers again. The same unique NVML PID must return before
   the supervisor permits tensor uploads and Session loading.

Every barrier identifies the actual child PID and compiled logical device.
Unexpected/multiple PIDs, timeout, a context that does not disappear, a
changed PID or child failure are fatal. Only the benchmark's own child is
terminated. The causal mapping and every observation are saved in
`owner-handshake.json`; it is explicitly not an NSpid-verified mapping.
No host PID namespace escape, unknown platform socket, system GPU reset or
global environment modification is used. Handshake cost is outside timing.
Owner/temperature/power telemetry is sampled approximately once per second,
plus query overhead. No foreign owner observed at those points is not proof
that a transient process could not exist between samples. The reservation,
idle preflight, handshake and raw telemetry are retained as distinct evidence.

## Verification

CPU tests exercise all 65,536 BF16 and FP16 bit patterns through the actual
C++ decoder, including NaN/Inf classification and signed zero. Additional
tests reject lossy BF16 references, malformed input storage, active-dimension
padding dilution, changed owner identities, missing handshake phases and
unlocked source-loop policies. Mock process tests exercise the independent
raw-output audit and primary numerical failure path without GPU execution.

Real RDT evidence is separate: the initial H20 pilot was stopped before model
execution because host and container PIDs differed. Its incomplete logs remain
at `rdt/benchmark-ordinary-v1-failed` under the Goal artifact root. The explicit
handshake candidate uses a new frozen source and output directory; no failed
or pilot call may be included in the formal five-process CDF.
