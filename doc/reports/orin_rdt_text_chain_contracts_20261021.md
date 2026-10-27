# RDT Text Chain and Native Contracts

Date: 2026-10-21. **Complete text encoder runs; full RDT remains unqualified.**
No complete RDT action gate, native resident Session, E2E throughput/latency or
formal CDF is claimed. The five-model Goal and earlier negative evidence remain.

## Complete Target Text Chain

All 24 T5 blocks execute on Orin as four real engines. Only original token IDs
and mask are supplied initially. The controller routes actual earlier outputs
to later stages and retains relative-position bias/other declared dependencies.
The independent audit rehashes every routed input/output; no CPU intermediate
is substituted between target stages. All output tensors are finite.

| Complete `[1,6,4096]` Feature Reference | MSE | Max Abs | Cosine |
|---|---:|---:|---:|
| Portable CPU | 5.599313e-6 | 0.015625 | 0.9996753914 |
| Original TorchScript CPU | 6.916295e-6 | 0.0173339844 | 0.9995993885 |
| Frozen H20 language | 6.425018e-6 | 0.03125 | 0.9996278948 |

These are language features, not actions. Cosines are below 0.9999 and the
complete-action criterion is not evaluated. Each native engine is initialized
in its own probe process with file-mediated handoff. This is numerical encoder
execution, not resident inference or a sum of stage timings advertised as E2E.

## Range-Required Exceptions

The first all-FP16 dense candidate for segment 1 constructs successfully but
produces 24576 NaNs. Instrumenting its portable reference at all 54 dense
boundaries identifies only two output-range violations:

| Node | Observed Max Abs | Elements Above 65504 | Adopted Compute/Result |
|---|---:|---:|---|
| `/MatMul_71` | 99914.4296875 | 6 | FP32/FP32 |
| `/MatMul_98` | 136408.78125 | 10 | FP32/FP32 |

The generic precision profile accepts exact named exceptions. The other 52
dense nodes in this segment retain FP16; the remaining three segments retain
their FP16 dense policies. Source/weight bytes, all layers and timestep behavior
are unchanged. No clipping, solver replacement, quantization search or global
FP32 model fallback is used. Other segments' observed dense boundaries fit
FP16 for the frozen reference input. Tokens/mask match the recorded 16-input
series, but observed ranges are not universal prompt guarantees. Final target
chain outputs are independently checked finite after the declared exceptions.

Frozen adopted engine identities:

```text
part 0: 9c39c004d285b963ff6534b423df2804195b0338124939cb3b45d50e861063a1
part 1: 7952ff0d3e9286bf6db41208fc92b4ccde5a83a3389dd3a68f7f9736a9920a1d
part 2: 50a8c4fbb14e2d640c6ff03854015f3d203c3ddef3e36a66e02be5a3e157be21
part 3: dfcd5f02537f9cef7454d5d1edf8278034ac09566827b23870f1c2d09fd54b84
```

The full chain manifest binds all four actual engine hashes and physical port
orders. No unbuilt engine is represented by a dummy hash or treated as passed.
Engine inspection confirms both named exception layers have FP32 interfaces.

## Whole-Model Contract Preparation

The original RDT Invocation IR is adapted to the declared FLOAT-carrier/INT32
artifact interfaces with explicit non-equivalence metadata. Tensor types change
only under this declaration; host loop indices, shapes, output consumers, source
observations/noise and all five DPM steps are retained. Pure exact memoization
is disabled for fresh encoding. The model still contains all online components:

- Four complete text-encoder Regions.
- Vision and conditions.
- Zero-input initialization, denoiser, five-input solver and finish.
- Original five-iteration state/history/count/index flow and unified output.

The ten-Region IR and scheduled Plan verify. The generic input projection pass
updates nested calls from explicit original indices, preserves all results and
external contracts, and rejects effectful/cached Regions and invalid mappings.
Zero inputs and reordered subsets are tested. It neither guesses unused opaque
arguments nor certifies dtype conversion or artifact numerical equivalence.

`complete-ir-001/preparation.json` says `complete-ir-prepared-not-target-backed`,
not an executed Session: contracts for all engines and native/raw 14-coordinate
output handling are still incomplete. No fake artifact bytes are generated.

## Small Regions and Predicate Boundary

| Check | Native Result | Scope |
|---|---|---|
| Initialization | Zero inputs, four full output tensors numerically exact | One original source fixture |
| Finish | Two full outputs numerically exact, including acceptance BOOL | One original finite source fixture |
| Ordered IsNaN/IsInf rewrites, mixed-precision builder | IsInf fails 143362/327680 patterns | Rejected deployment candidate |
| Same predicate graph, isolated FP32 engine | All five predicates exact on 327680 patterns | Required independently verified island |

Original INT64 counters become native INT32 under checked small bounds; source
BF16 values are carried as FLOAT. Equality is numerical, not a claim of equal
storage bytes across dtypes. Init/finish are tiny declared FP32 arithmetic
islands, not whole-model FP32 fallback.

The real SDK imports IsNaN but rejects IsInf. Deployment preserves the supported
native IsNaN. Full-range IsInf comparison bounds did not survive the tested
mixed-precision configuration even with layer FP32 constraints, so that variant
is not approved. CPU proof alone would have missed this failure. The verified
FP32 island retains finite extrema, subnormals, signs, NaN classes and Inf
detection flags. This does not prove all possible full-model non-finite paths.

The finish exporter widens FLOAT to DOUBLE before classification. The generic
rewrite bypasses only that exact widening for classification, removes it only
when dead, and preserves observable/other consumers. Arbitrary DOUBLE arithmetic
is rejected rather than silently cast to FLOAT; both positive and negative
cases are unit-tested.

## Remaining Work

The corrected denoiser export records two GELU/two SiLU FLOAT-opmath promotions;
its source/carrier files are prepared, not a target inference pass. A first
publication attempt collided with old external-weight filenames and failed
without overwriting; the isolated-directory retry succeeds. Native IsNaN will
be retained rather than adopting an unnecessary rewrite experiment.

Conditions still need rebuilding under the corrected native-composite policy.
Vision's earlier large feature-fidelity gap remains unresolved. Native resident
Session, exact original raw-input preprocessing/output transformation, all 16
complete normalized/native actions, resource checks and formal E2E/CDF remain.
No stage/feature result substitutes for full VLA acceptance. pi0, pi0.5, CogACT,
D4 ablations and final delivery remain in scope; Qwen stays lower priority.

## Evidence

Root: `artifacts/orin/20261021-04/`.

- `t5-chain-spec-001.json`, `t5-complete-chain-001/`,
  `t5-complete-chain-audit-001.json`: complete target computation/routes/tensors.
- `t5-part-001-dense-ranges.json`: frozen-input observed overflow boundaries.
- `t5-part-001-stage-001/`: retained NaN-producing candidate; build 002 is adopted.
- `t5-part-000/001/002/003-build-*`: actual native engine receipts and placement.
- `complete-ir-001/`: verified complete-model IR/Plan, explicitly unexecuted.
- `small-regions-audit-001.json`, `initialize-stage-001/`, `finish-stage-001/`.
- `predicate-conformance-summary.json`, `predicate-stage-001/`,
  `predicate-fp32-stage-001/`: retained failed and passing exact-classification runs.
- `denoiser-opmath-001/`, `source.report.json`: prepared source artifacts only.
- `full-host-regression-r3.xml`: **2574 passed, 74 skipped, six subtests passed**.

No production C++ or target DSO changes in this checkpoint; no target Torch or
new other-platform performance experiment. Prior frozen SmolVLA measurements
and solver-stage evidence are unchanged. All current jobs exited. Goal active.
