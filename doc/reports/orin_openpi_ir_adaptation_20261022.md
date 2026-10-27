# OpenPI Typed IR Adaptation

2026-10-22 campaign, completed 2026-10-23 local time. This is working,
data-driven host compiler code under `artifacts/orin/20261022-16/`, not a target
deployment or model numerical result. No Orin connection, GPU/model inference,
target build, installation, shared production edit or reset was performed.
CogACT root07, its report and manifest are preserved. Workers 11/14/15 and the
parent's RDT/runtime changes were not edited.

## Implemented Adapter

`ir_adapter.py::adapt_module` accepts an immutable typed Module and explicit
prepared child-graph contracts. It has no model-name dispatch. `prepare_ir.py`
loads hash-pinned metadata, applies the adapter, and invokes the existing
`compile_module(profile="verified", loop_execution="source")`. The result is
verified semantic IR, a verified physicalized plan, a structural compiler
certificate, and **unbound** artifact requirements. No placeholder engine or
DSO hash is manufactured.

Existing APIs used directly are `parse_canonical_json`, `verify`,
`configure_exact_cache`, `project_region_inputs`, `compile_module`,
`verify_plan`, `ValueContract.from_ir`, `io_schema_digest` and the standard
IR/Plan serializers. CodeGraph was used to inspect these APIs. Their actual
source-file hashes are recorded before/after compilation in `provenance.json`.

The adapter checks actual ONNX headers without loading external weight data or
executing a graph. Input reports must bind the source/prepared graph hashes;
the opset-v2 report binds the prepared/final graph; precision002 must bind that
exact final graph. Projection indices are explicit, unique and type checked.
Dropped inputs must actually be unused, narrowed IDs must have only standard
Gather-index consumers, and Casts must match original source dataflow/bytes.
Nested captures, hidden RNG, effectful Regions, ambiguous SSA types, derived
state conversions and private carriers escaping public/control outputs fail.

The source prefix's 36 BF16 cache outputs, their SSA definitions and consuming
step parameter types are all replaced with the proposed FLOAT-carrier types.
No public input/output type changes. Original prefix memoization is explicitly
disabled: private caches are per invocation, with no pointer or cross-chunk
reuse. `project_region_inputs` then applies the actual report routes, including
pi0.5's unused state removal. Parameter/binding names follow the prepared graph
order rather than an inferred model convention.

## Frozen Sources

Sources remain in the original repository's board-handoff packs:

```text
/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/pi0-dual-data-002/pack/source/module.json
/home/zhangzimo/Repos/private/edge-fm-x/artifacts/edgefm-vla-goal/20260906-044509/board-handoff/pi05-dual-data-001/pack/source/module.json
```

The independently pinned original pack manifests bind those Module bytes.
`request-pi0.json` and `request-pi05.json` include all source, report, final
graph and precision-profile paths/hashes. The original OpenPI revision remains
`215abfb217dbac7d5f1273282331b9b1866c0479`; checkpoint pins remain those in
root02. This preparation does not recount checkpoint parameters, prune unused
weights, rehash large external weight payloads or inherit target acceptance.

| Source Module | File SHA256 |
|---|---|
| pi0 | `72ce499c901d1bba8277b72ba319010353b035d017ccfbeb482cab6c48ed58cd` |
| pi0.5 | `9ac2cbfb757f5c2769e24dbb0635136431b832a5f105a75514daa58261229d40` |

The proposed neural graphs are the actual sibling
`*-trt-opset17-v2.onnx` files under root08, bound to precision002:

| Proposed Graph | SHA256 |
|---|---|
| pi0 prefix | `cec3843fab9bac6a8ce41f0425a6c288398a04c638db81eb5d7b5b926dc8f713` |
| pi0 step | `3da39d30d9cf66d66cdab329d07b59e28fb1d9b8b4416f3848002e5569222f55` |
| pi0.5 prefix | `52f78cfff7ec54d1d47459c697306ad468fb20c0dacba6e1dc6f6ab1a5ed754a` |
| pi0.5 step | `4c550a0bb610452792ba14b85874936ed47cc19494438cf9f5601247294bb277` |

These hashes identify ONNX sources, not compiled engine artifacts.

## State Cast Placement

The original pi0 ONNX at root06 `exports/pi0-step-001/model.onnx` contains:

```text
node__to_copy: Cast[to=1](state: DOUBLE[1,32]) -> _to_copy: FLOAT[1,32]
NodeProto SHA256: 9e161fcb112dde8dbeb0f237cf9774928a8bf87d66698b92c9ce50ce4ae6b486
```

Both this original node and the root08 pre-preparation node have identical
serialized bytes and exact signature. The original state graph input is not
overridable, has no other neural consumer, and is not a graph output. The
prepared report explicitly externalized this same Cast; its old
`hoisting_proved=false` is not rewritten or inherited as proof.

The new structural proof independently establishes:

1. `state` is read from the immutable external input by root operation 9 into
   F64 SSA `vf_value_17`, before the original root loop at operation 13.
2. Every source Region is declared pure, the Module has no persistent state,
   and the conversion consumes this direct input read, not time, noise,
   induction, a loop carry, derived computation or a state snapshot.
3. One uncached checked F64/F32 invocation is inserted after that input read.
   It dominates the loop and all ten consumers. Compiled task dependencies
   independently confirm preheader dominance. A fresh chunk executes it again,
   even when the same memory allocation is rebound.
4. Converted F32 SSA has a distinct logical buffer. Original F64
   `vf_value_17` remains the unchanged input-0 operand of native finishing.
   There is no F32-to-F64 roundtrip. `original_wide_input_retention` records
   both types, both SSA names and the preserved original observer.

The caller must keep bound external bytes immutable until invocation
completion; an IR purity declaration does not prove physical caller aliasing
or asynchronous-buffer lifetime. The actual checked provider remains unbound.
Its source/target numeric and complete-output admission remain parent work.

I64/I32 conversion is similarly an explicit uncached region before prefix,
with full-range checking required on every element. It is not a reinterpret,
unchecked truncation, vocabulary substitution or pack-only range assumption.
Conversion errors abort before commit; no dummy acceptance tensor is emitted.

## Preserved ABI

Both public schemas remain byte-identical to their source schemas:

| Port | pi0 | pi0.5 |
|---|---|---|
| Three images | F32 `[1,3,224,224]` each | Same |
| Three image masks | BOOL `[1]` each | Same |
| Language tokens | I64 `[1,48]` | I64 `[1,200]` |
| Language mask | BOOL `[1,48]` | BOOL `[1,200]` |
| State | F64 `[1,32]` | Same, still used by native finish |
| Noise | F32 `[1,50,32]` | Same, original saved noise |
| Normalized actions | F32 `[1,50,32]` | Same, not shortened to 14 |
| Native actions | F64 `[1,50,14]` | Same |

Private prefix: eight prepared inputs including checked I32 IDs; one BOOL
padding result plus 36 F32 cache results. Cache/padding length is 816 for pi0,
968 for pi0.5; each cache is `[1,1,length,256]`. Step retains two results:
full F32 actions and a rank-zero F32 time Tensor. It has 40 inputs for pi0,
including the converted state, and 39 for pi0.5, with unused state projected.
The public pi0.5 state port is not removed.

Original native finishing remains exactly:

```text
(state F64[1,32], normalized F32[1,50,32], incoming_accepted BOOL[1])
    -> (native F64[1,50,14], accepted BOOL[1])
```

The source finite region and time region remain unchanged. Native acceptance
still passes through the original `vla.validate`; both complete outputs are
staged into the same action group, then atomically committed by the original
transaction. All ten original loop iterations, carries, bounds and source
noise are preserved. No accepted flag or missing model tensor is fabricated.

## Final Outputs And Registration

Use `pi0-final-001/` and `pi05-final-001/` under root16. Each contains source,
adapted and compiler-canonical semantic Modules, a physical plan, baseline
plan, structural compiler certificate, complete unbound requirements, checked
cast profile JSONs, structural proof and source/API provenance.

| Identity | pi0 | pi0.5 |
|---|---|---|
| Semantic module digest | `01e23ea637ceb7ce1191448b56eccb485bf23a94a3b6b536a16a5f0fcc13900b` | `6db7a5bfd24fc5d841b3dfb50c17d3da05653c295bbb5cfff8b6bb339afed716` |
| Public schema digest | `8751fa709441f09883da726d2cc2d6b52f7a5894bf5876ee36b6803e2dff9575` | `2cca61bd2132c82d053daba6ea951fa0e1f2d0125dd66db8bbb13a3fece2bd74` |
| Logical arena bytes | 30,101,440 | 35,705,472 |

Semantic/plan digests use the existing canonical APIs; file-byte hashes are
reported separately. Arena sizes describe compiler buffer allocation only,
excluding unbound engine weights, engine workspaces and actual provider
workspace allocations. They are not measured target memory.

Exact canonical registration IDs:

| Region | pi0 ID | pi0.5 ID |
|---|---:|---:|
| `checked_input_openpi_step_0` | 0 | Absent |
| `checked_input_vf_cached_openpi_prefix_59_6` | 1 | 0 |
| `openpi_finite` | 2 | 1 |
| `openpi_native_output` | 3 | 2 |
| `openpi_step` | 4 | 3 |
| `openpi_time` | 5 | 4 |
| `vf_cached_openpi_prefix_59` | 6 | 5 |

`unbound-region-requirements.json` contains the **entire public schema and
every ordered Region input/output**, including exact NN binding names,
dimensions, dtypes, device, ABI 2 and these IDs. It is intentionally not a
loadable `RegionArtifactContract`: artifact path/hash/size are null. The plans
also retain `backend="uncompiled"` and no artifact path. Source compatibility
is not target artifact registration or numerical-provider enforcement.

The ready-to-use cast profiles select worker14's `f64_f32_rne` `[1,32]` or
`i64_i32_checked` `[1,48]`/`[1,200]`, with actual IDs and CUDA ordinal 0.
Their required aligned device scratch minima are respectively 384, 448 and
1056 bytes at alignment 256. They are outside solver replay, synchronizing,
and have no graph-capture/execution-context support. Real DSOs must be built
with the actual parent schema/IDs, not reused from fixture registrations.

NN requirements pin precision002 and require actual TensorRT engines and
plugin registration: `VLAForgeBFloat16Round` creator version `1`, namespace
empty, plus `VLAForgeScalarSinCosF64` version `1`, namespace empty for steps.
Actual artifact hashes, capability/effect audits, workspace, target numerical
binding and complete-output fidelity are explicitly unbound.

## Verification And Remaining Work

`tests-006.xml`: **20 passed**. This includes actual source Module/Plan
verification and roundtrip, final NN header agreement, 36 cache type changes,
public-schema equality and plan dependency proofs for both real profiles.
No real source Module is executed by a synthetic provider.

Small, clearly named NumPy unit fixtures separately execute ten solver steps,
compare full fixture outputs, verify original F64-state identity, rebind the
same allocations across fresh chunks, and reject overflow/nonfinite casts
without replacing previous committed outputs. Other tests reject source Cast
hash drift, live-input deletion, derived-state hoisting, effectful Regions,
public carrier exposure, duplicate/boolean indices, nested captures, hidden
RNG and changed file pins. These are software fixtures, not neural-model
fidelity evidence. Earlier failed fixture-construction/test receipts remain
as `tests-001.xml` through `tests-004.xml`; the real source structural cases
passed there too. No expected model output or numerical threshold was changed.

There are no remaining source typed-IR verification failures for these two
prepared contracts. Target ABI work remains unbound: actual engines and plugin
DSOs, native finish's exact two-output variant from worker15, original time/
finite providers, workspace/capability/effect/numerical contracts, native raw
inputs from worker11, and synchronization/lifetime/error integration. Public
input alignment remains unchanged at 1, while worker14 requires input
alignment at least 8 and output at least 4; the parent must validate or stage
aligned bound buffers rather than weakening either contract. Native F64/I64
buffer copying is not TensorRT DOUBLE arithmetic support.

Final `delivery-001.json` and `manifest.json` bind the files, real source pins,
tests and this report. The audit rechecks the root07 manifest/report hashes;
it does not rewrite root07 or rehash its 44 GiB payloads. The normal compiler
certificate is structural only and must not be promoted to target acceptance.

## Exact Next Commands

From the repository root, regenerate without overwriting the final delivery:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONPATH=vlaforge/python:artifacts/orin/20261022-16 \
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B artifacts/orin/20261022-16/prepare_ir.py \
  --request artifacts/orin/20261022-16/request-pi0.json \
  --output artifacts/orin/20261022-16/pi0-review-001
```

Use `request-pi05.json` and a fresh `pi05-review-001` directory for pi0.5.
To reproduce the focused structural/unit checks:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONPATH=vlaforge/python:artifacts/orin/20261022-16 \
  PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  python3 -B -m pytest -q artifacts/orin/20261022-16/test_ir_adapter.py \
  --basetemp=artifacts/orin/20261022-16/review-test-tmp \
  --junitxml=artifacts/orin/20261022-16/review-tests.xml -p no:cacheprovider
```

Before target execution, the parent must replace every unbound requirement
with a real compiled artifact contract and validate the complete outputs.
This task does not start target work or infer model support from host IR.
