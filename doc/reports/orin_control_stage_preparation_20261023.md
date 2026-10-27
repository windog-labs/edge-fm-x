# Orin Non-Neural Control Preparation

## Scope and Status

Worker-owned writes are confined to `artifacts/orin/20261023-10/` and this
report. Published root03, production tools, plugins, ModuleIR, original ONNX
graphs and learned weight payloads are unchanged. No SSH, GPU execution,
installation, target build, or other-platform benchmark was performed by this
worker. Parent owns target execution and its authoritative receipts.

Host preparation is complete: 570 full-output source/candidate comparisons
passed across both models and ten graph variants. Existing control preparation
tests passed 25/25; independent prefix audit tests passed 12/12. Host C++ syntax
and CPU-only reference checks do not certify Orin compilation or GPU outputs.
The native probe's 19 host tests passed, including all four reference contracts,
strict malformed-metadata rejection and refusal of host target execution.

Parent-reported target state at handoff:

- Prefix003 was aborted by the outer MemAvailable <2GiB guard. Its owned child
  was removed. The unfinished inner build receipt is not a parsing or unsupported
  operator verdict. No prefix engine or inference pass is claimed.
- Pi0 zero-input time001 engine built, SHA256
  `24302840135788db41901f8541295931ebda84f513dd532a0bea98958aba43a4`.
  This is a build pass only, pending complete GPU output validation.
- Finite001 failed CLI admission with `Invalid DataType bool`, before GPU build.
  The original controller and its negative evidence remain unchanged. Parent's
  fresh root06 controller omits IO-format flags for BOOL bindings while retaining
  actual ONNX BOOL type; finite002 is parent-owned and pending.

## Prepared Controls

`prepare_control_stages.py` uses the existing canonical Constant and bounded
integer-control folding tools read-only. Artifacts are under `prepared-001/`.
Original source exports (root06, opset18), lowered graphs (root08, opset17), and
each candidate were compared with the CPU ONNX reference evaluator and explicit
numeric references, preserving complete output dtype, shape and bytes.

The 57 cases include full positive/negative max FLOAT32, signed zero, minimum
normal, signed subnormals, values beyond HALF range, NaN payloads, infinities,
first/middle/last-element invalid values, random finite data and raw bits.
These are source-level checks, not a whole-network floating fallback.

Finite's private BOOL -> INT64 -> ReduceMin -> BOOL cone is optionally projected
to BOOL -> FLOAT -> ReduceMin -> BOOL. Its proved domain is nonempty exact 0/1;
the final BOOL result is identical, and no source floating predicate or public
port changes. Shared/public INT64 observers are rejected. The source maximum
FLOAT32 comparison literal is retained exactly; global HALF would be invalid.

`control_fp32_builder.py` admits only bounded static non-neural operators/types,
pins graph bytes and a host certificate, uses fixed trtexec argv without FP16,
INT8 or TF32, and requires native AArch64 plus board identity for execution.
Target Python uses the standard library only. Certificates include source hashes,
operator/dtype counts and literal payload hashes. Neural/plugin operators,
external tensors and oversized or nonfinite literals are rejected. Original
BOOL `--outputIOFormats=bool:chw` admission is now a documented target failure;
use the parent's fresh correction, not the original batch build commands.

## Actual Interfaces

| Stage | Actual Input | Actual Output |
| --- | --- | --- |
| Time source | unused `noise`, FLOAT `[1,50,32]` | `ones`, FLOAT `[]`, exact 1.0 |
| Time zero-input variant | zero inputs | `ones`, FLOAT `[]`, exact 1.0 |
| Finite | `sample`, FLOAT `[1,50,32]` | `view`, BOOL `[1]` |

Both source time graphs have no consumer of `noise`. The explicit zero-input
variant removes only that graph input. Parent must project the `openpi_time`
region argument and regenerate its IR/schema; public Module noise remains the
solver's initial sample. FLOAT rank0 is a tensor, not a scalar-value ABI carrier.
BOOL `[1]` must remain BOOL, not FLOAT or an integer bitcast.

Both actual semantic modules route `openpi_time(vf_value_19)` to `vf_value_60`,
then ten loop iterations carry sample/time. Loop results `vf_value_69` sample
and `vf_value_70` time feed finite (`vf_value_69` -> `vf_value_71`) and native
output (`vf_value_17` original F64 state, sample, BOOL acceptance). Pi0 Step
sample/time input slots are 1/2, outputs `add_165`/`add_166`; pi05 slots are 0/1,
outputs `add_202`/`add_203`. Step sample is FLOAT `[1,50,32]`; time is FLOAT `[]`.
Per-step next-time computation remains in Step, not the initial-time engine.
Exact original module/graph hashes and consumer ports are in `handoff-001.json`.

## Native Target Probe

`control_native_probe.cpp` uses native TensorRT 8, CUDA and OpenSSL only. It
accepts typed metadata, engine/metadata SHA pins and board serial, with no
model-name branches. It enumerates actual IO and rejects any name, mode, dtype,
rank, dimension, storage width, packing or device-location mismatch. Zero inputs
are genuinely supported; no dummy is synthesized. A private nonblocking stream
orders complete H2D, output sentinel, enqueue and D2H; checked completion precedes
every complete-byte output comparison. Unproven CUDA completion poisons and
retains all affected host/device storage and TensorRT objects until exit.

Compile on Orin from the integrated source root:

```sh
sh artifacts/orin/20261023-10/build_native_probe.sh
```

Exact native compiler/link command, direct invocation and all four metadata pins
are in `artifacts/orin/20261023-10/NATIVE_PROBE.md`. The `.meta` files live in
`native-probe-metadata-001/`, with an independent `receipt.json`. Do not rename
or regenerate them for the parent's BOOL IO-format-only correction.

Time compares all four bytes against FLOAT 1.0. Finite uploads all 1600 elements
for 60 cases: signed zeros/max/subnormals, minimum normal, beyond-HALF finite
values, signed NaNs including signaling payload, and signed infinities, both
full tensor and first/middle/last positions. Complete BOOL output is compared
against IEEE754 finite classification, independently checked by native
`std::isfinite` compiled without fast-math. Unsafe NaN optimizer simplification
fails; no graph-capture, performance or model-level gate is claimed.

Host syntax passed against copied real TensorRT 8.5.2.2 headers and installed
CUDA headers, without linking/loading CUDA. A separately compiled
`CONTROL_PROBE_CPU_ONLY` executable exercised references without any engine or
GPU. The batch request `native-probe-target-request-001.json` names the original
finite001 directories and is not valid unchanged for finite002. Direct native
invocation accepts the successful corrected engine path with its actual SHA.

## Prefix INT64 Audit

Audited graph SHA256:
`4d0eede245e56282630801993cc824b7aa0b239e2f2ba7733dae26a5b892c847`.
The new root10 copy validates all 165 FLOAT rank3 LayerNorm contracts against
the source (gamma/beta channel descriptors, version1, empty namespace and exact
epsilon bits), then uses Identity(x) only in a private independent shape view.
All old internal value_info is cleared; 3176 shape-preserving round plugins
are also represented by private Identity nodes. No graph or weight is saved.

`prefix-i64-audit-001.json` is preserved: 35 proved, one dead, 162 then unproved
because omitted Slice axes were not admitted. Refined `002` explicitly proves
ONNX default axes and positive steps, yielding 197 proved static saturating
Slice bounds, one dead constant, zero unproved/unscanned oversized uses.
Both INT64 extrema are inventoried. Of the 197, 162 slice a length-4 Shape
result vector, not an assumed head-width dimension; 35 use independently
inferred dimension256. Runtime IDs/state are outside this literal-only proof
and still require checked providers. This is not blanket clamp approval.

Independent `data_prop=True` crosscheck `prefix-independent-data-prop-003.json`
also clears old internal metadata and agrees on all 197 affected input shapes,
with zero unknown/disagreed shapes. It reads no learned external payload,
executes no learned neural values and does not alter the audited graph.

## Pending Gates

Native target compilation, complete GPU control outputs, corrected finite
builds and parent integration remain unverified here. Root10's source/hash
manifest records worker evidence only; target parent receipts must be retained
separately. No full-model fidelity, inference, latency, memory or deployment
acceptance follows from CPU proof or engine construction.
