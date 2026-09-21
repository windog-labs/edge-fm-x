# Explicit INT8 Linear Deployment

This is a precision-changing backend, not a same-precision optimization. Its
construction records never certify deployed kernels, full-model fidelity,
held-out free-running behavior, performance, or a board result.

## Interfaces

`deployment.int8_linear.lower_int8_linear` consumes a typed `PrecisionPlan`, its
complete calibration report, the selected site and explicit step, the source
Region digest, the current full numerical context, and actual weight/bias
Tensors. It returns an owned module and a construction record.

`lower_scheduled_int8_linear` consumes the same declarations without a fixed
step. Its module accepts `(activation, step_index)`. `step_index` must be int64
`[1]` on the same device; it selects the immutable scale table using Tensor
operations. There is no implicit Python step counter, timestep approximation,
or scheduler replacement. Out-of-range indices produce all-NaN output, not a
silently clamped finite result. The surrounding invocation must reject nonfinite
outputs before publication. This local guard does not make unrelated indexing
operations in an original Region safe for invalid inputs.

`deployment.linear_precision.lower_exported_linear` accepts a static original
ExportedProgram and one explicit selection: calibrated producer, exact
`aten.linear.default` consumer, and user-input step Tensor. Weight and bias must
directly resolve through the original graph signature to immutable lifted
parameters or persistent buffers. It returns a new ExportedProgram and ledger;
original calculations other than that selected Linear and the external ABI are
preserved. It does not select nodes by model name or by mathematical similarity.
The caller verifies the actual archive SHA before loading and binds the step
input to its real scheduler. A supplied digest alone is not authentication.

New buffer placeholders are lifted into the graph. The small replacement's
metadata is propagated into the original export's FakeTensor domain using the
version-pinned [Torch 2.10 implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/fx/passes/fake_tensor_prop.py); the full original floating graph is
not retraced to perform this merge. This metadata-only operation is not runtime
validation. The replacement output must preserve dtype, shape, device, layout,
stride, and storage offset. No original artifact certificate is reusable.

## Arithmetic

- Verified backend API: Torch 2.10.0. Input/weight dtypes: FP16, BF16, FP32.
- Static input rank is at least two. Flattened GEMM M is at least 16; K and N
  are multiples of eight, N at least eight. K*127*127 must fit in int32.
- Activation scale is the fitted, representable FP32 scalar for this site/step.
  Its complete calibration group is independently reconstructed, excluding
  held-out samples. Global fitting may include other explicitly quantized sites.
- Activations: FP32 divide by scale, nearest-ties-even round, clamp to
  [-127,127], then signed INT8 conversion.
- Weights: owned original snapshot, per-output-channel FP32 absolute maximum
  divided by 127, floored at FP32's smallest positive normal, same round/clamp.
- Actual GEMM is `aten._int_mm.default`: INT8 x INT8 -> int32. Dequantization is
  sequential FP32(accumulator)*activation_scale*weight_scale + FP32(bias), then
  original input dtype. Fusing these operations needs its own numerical gate.
- Nonfinite activation rows stay nonfinite after the module; integer conversion
  must not hide invalid inputs from the invocation commit validator.

## Ownership

Quantized weights, scales, and bias are owned snapshots, not views of the source
weight. Construction requires the observed full numerical context to equal the
plan and the current process/thread state before and after construction.
`to_data()` verifies current module buffers' logical bytes, dtype, shape,
device, layout, stride, and storage offset. A Torch module remains mutable code;
this is not a runtime certificate or protection against arbitrary caller edits.

The graph transformation retains original state by reference. The caller must
keep it immutable through serialization and hash/freeze the actual resulting
artifact and its new contracts. Numerical-context enforcement belongs to the
deployed backend/Session, not to the exported Tensor graph. Training/autograd
equivalence is outside this inference-only precision change.

## Evidence And Limits

On 2026-09-06 the original SmolVLA trajectory observer retained 160 complete
FP32 `[1,50,720]` head inputs from 16 distinct episodes, with a predefined 8/8
calibration/held-out split. Weight/bias bytes were independently matched from
the source step archive to the fully hashed checkpoint. Four fitted strategies
were executed on RTX 3060 in two independent 40-module x 16-input campaigns.

The second campaign binds driver/controller before/after hashes, argv, exit
status, PID/NSpid and owner monitoring. Independent CPU audit reconstructed all
1920 output files, 640 integer accumulators and dequantized candidates; original
FP32 references also match the original complete-trajectory node observations.
Forty actual traces associate `aten::_int_mm` with an Ampere INT8 GEMM kernel.
This is real isolated-kernel evidence, not a compiled full-model deployment.

Held-out local-head worst MSE was about 1e-4 while all cosine values exceeded
0.99996. These are not complete-action errors and do not prove losslessness.
Global and per-site head scales happened to agree because their fitted absolute
maxima matched; their calibration groups were different, not a one-site plan.

Scheduled module/graph replacement plus calibration currently have 133 focused
CPU tests. Earlier graph-merge failures from deep-copying FakeTensor metadata
and mixing export domains are retained in the v1/v2 XML; v4 passes.

The next real campaign ran four strategies on all 16 original observations,
with quantized state actually fed into the next iteration. An independent audit
checked all 64 complete action chunks, 4544 raw Tensors, original state bytes,
and unchanged non-selected mathematical nodes. All four held-out sets had
0/8 passes at the predeclared MSE < 1e-5 threshold. Worst complete-action MSE
was 4.287605e-5 (global/site), 5.080927e-5 (step), and 5.060398e-5 (step-group),
despite cosine above 0.99999. These are normalized active-six-dimensional
outputs: the original finish only slices/copies and checks finiteness. No inverse
normalization or physical-unit action metric is implied. This is a failed
quality gate, not lossless deployment.

The subsequent `int8-native-v1/attempt-002/` campaign produced seven new public
TorchScript archives, checked 47 complete Region cases, and built four new
precision-changing/provider-required C++ bundles. All 64 complete normalized
actions in no-Python native Sessions match their corresponding quantized
free-running candidate byte-for-byte. Full version-2 numerical policies and
the bundled provider DSO were checked in the actual workers. The independent
audit SHA is `dfa1fdba9f2f91cc5d0c895bd56f0525f63f522dc4c90c510b1489ab06ddecd6`.
The same official quality gate still fails 8/8 held-out samples in every lane.
No native performance, native profiler, inverse-normalized output, or board
claim is inherited from the isolated-kernel campaign.

`int8-trajectory-errors-v2/` additionally binds 3211 existing input files and
recomputes 2560 per-step metrics in both full-32-storage and active-six spaces.
All 64 final metric dictionaries exactly match the original free-running
reports. The fixed held-out mean MSE for step scaling starts at 1.113001e-6
versus global/site 1.325157e-6, but ends at 2.068598e-5 versus 1.948666e-5.
Lower initial local error does not establish lower accumulated final error.
This is offline analysis of verified raw trajectories, not new inference,
refitting, or an intermediate acceptance threshold. Its CSVs, plotting source,
PNG/PDF and provenance are retained; the first analysis attempt's compact/full
sample-schema mismatch remains a recorded failure. Same-source native pilots
are a separate experiment and are not formal latency/CDF or selection results.

Evidence roots: `artifacts/edgefm-vla-goal/20260906-044509/precision-probe-v1/`,
`int8-kernel-v1/`, and `int8-kernel-independent-v1/` under the same campaign root.

## Native Pilot And Attribution

The subsequent same-runtime five-lane pilot independently verified 240 complete
normalized outputs and 160 measured host timestamps. Baseline/global/site/step/
step-group descriptive mean latencies were 94.386/93.959/94.853/96.117/94.440 ms.
Each lane used only one process with 32 measured calls: this is neither a formal
CDF nor evidence of a reliable speed ranking. Quantized quality still fails.
The quantized paths made 230 additional allocator requests per complete call,
increased allocated peak by 24576 bytes, and increased the four Region archives
by 27554 bytes. No end-to-end speedup, zero allocation or compression is claimed.

Separate native NSYS runs measured three complete calls each for baseline,
global and step scaling. All nine complete outputs match their candidates.
Within PID/TID-correlated complete-call ranges, baseline has zero INT8 kernels;
each quantized call has ten `ampere_igemm_int8_128x128_ldg4_nn` kernels, one per
generation step. Kernel counts rise from 12367 to 12597 per call. NSYS 2024.6.2
reported driver 13.2 outside its compatibility list and used CUDA 12.8 CUPTI:
this limitation is retained, not relabeled as calibrated or loss-free profiling.
Profile timestamps are not combined with pilot or formal latency distributions.

Independent audit SHAs are
`850848bec584b35f5299df7ca26a32750793ff1eee2fb1253c42e7794d40f4d3`
(`int8-native-pilot-v1/attempt-001/independent-pilot-audit.json`) and
`e2450e37d2667cd641f4c4e3ffe3cec71572400740b55a1477ee0a3b74d9b4d9`
(`int8-native-profiler-v1/`, independent profiler audit). The matched pilot
retains its older runtime source; later scoped-memory fixes are not inherited.
