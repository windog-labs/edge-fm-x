# Trace-Bound Precision Calibration

`analysis.precision_calibration` is the first offline calibration slice for paper
III-B. It does not alter IR, exported graphs, scheduler equations or runtime
precision, and it does not yet lower an INT8 artifact.

The Adapter declares actual profile and numerical-context hashes, an ordered
unique identity for every schedule entry, exact Region artifact/node identities,
and disjoint calibration/held-out sample manifests. A schedule identity must bind
the actual timestep, integration coefficients, CFG and scheduler history profile;
an index or a model name alone is not sufficient. Inputs and actual noise tensors
are bound separately. Partition keys identify the intended split unit, such as
an episode. Renaming an observation, changing its noise or selecting a different
frame of the same partition cannot bypass the split gate.

Declarations are digest-locked at construction and rechecked before observation,
reporting and fitting. Changing a split, site, artifact, schedule or context after
collection is rejected; create a new collector for a different experiment.

`PrecisionCalibration.observe` accepts real NumPy or PyTorch floating tensors,
including BF16. It records full storage-byte hashes and finite extrema without
casting the identity hash. This diagnostic copies data and may synchronize CUDA;
its execution is not timing evidence. Hashes and extrema are computed from the
same independent, owned CPU snapshot, not from two reads of mutable caller storage.
The snapshot uses a newly allocated canonical CPU tensor followed by a copy;
`is_contiguous()` alone does not guarantee a byte-view-compatible stride for
singleton dimensions. Actual strided singleton inputs exposed this case, and
FP16/BF16/FP32/FP64 regression cases now cover it.
The caller must finish producer writes and must not concurrently mutate the
observation or collector. Each context site is observed exactly once
per sample. Each iterative site must have every declared step for every calibration
sample, with stable shape and dtype. Held-out samples cannot contribute to fit.

`fit` supports `global`, `site`, `step` and explicit `step-group` scales, all using
the same complete calibration record. Step groups must partition the schedule
exactly once. A site can explicitly retain its existing floating precision; it is
not silently included in global quantization. The initial format is symmetric
INT8 with range [-127, 127], zero point zero, nearest-ties-to-even rounding and an
absolute-max fitted FP32 scale. A smallest-normal-FP32 scale floor is recorded,
including all-zero and subnormal ranges. Statistics pooled for a shared scale are
identified by the scale group, not misreported as a single-site sample count.
Group identities encode structured keys; legal site names cannot collide with a
step suffix. Scale records must round-trip through the stated FP32 absolute-max
fit, and their group identities must agree with the declared strategy/coverage.

`PrecisionPlan` binds the profile, numerical context, split and complete calibration
report. Its scale records cover every selected site/step exactly once. Serialization
always states `backend_lowered=false`, `real_low_precision_kernel_verified=false`,
`held_out_validation_complete=false` and `lossless_verified=false`. This object is
not an existing deployment capability or a permission to label an artifact INT8.

`PrecisionPlan.from_data` and `from_json` restore typed immutable declarations
and require canonical re-encoding to match the input exactly. Unknown/missing
fields, changed deployment flags, duplicate JSON keys and nonfinite constants
are rejected. Restoring a plan does not upgrade any verification flag.

The initial real SmolVLA observer campaign has completed the fixed original
trajectories and 8/8 split. Subsequent real INT8 kernels, independently verified
64 full free-running trajectories, and 64 precision-changing native actions are
documented in [INT8 deployment](int8-linear-deployment.md). Native actions match
their quantized candidates, but every strategy fails the fixed held-out final
MSE gate. The new per-step accumulated-error plots are offline reanalysis of
verified raw data, not additional inference or a new scale fit. All these Smol
outputs are normalized; inverse-normalized actions are not yet deployed.
Remaining acceptance includes free-running saturation, broader predeclared
error budgets, native latency/memory/artifact-size comparisons and native-output
verification. A fake-quant experiment remains separate from real kernel evidence.

## Half-Precision Replacement Path

`deployment.half_linear` provides a separate model-independent FP16/BF16 Linear
replacement. It does not fit INT8 scales: source FP32 weight/bias are owned and
cast to the selected half type, the explicit int64 schedule input remains in the
exported ABI for Region compatibility, invalid schedule indices yield nonfinite
output, and the lowered profile is restored to the original source dtype.
`deployment.linear_precision.lower_exported_linear` accepts
`precision_kind="float16"` or `"bfloat16"` and applies the same graph rewrite,
state ownership and output-profile checks as the INT8 path.

On the fixed SmolVLA split, full original-IR CUDA free-running trajectories for
both half precisions pass the predeclared held-out MSE/cosine gate while all
four INT8 scale strategies remain below the gate. These diagnostics still leave
`real_low_precision_kernel_verified`, `full_model_output_verified`,
`held_out_validation_complete`, `lossless_verified` and native deployment flags
false until no-Python artifact and runtime validation closes them.
