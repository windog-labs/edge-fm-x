# Empty Dynamic Proxy Lists: Bounded AOTI Repair

## Scope and Result

The real captured pi0 time Region now compiles through the public
`compile-artifact --inductor-profile aten-preserving` command and returns the
original zero-dimensional CUDA FP32 `1.0`, bitwise equal on three calls.
This is a single-Region compiler regression closure, not complete pi0 or paper
acceptance. The original captured noise tensor is used and saved; the scalar
shape, timestep, scheduler and weights are not replaced.

Evidence root:
`artifacts/edgefm-vla-goal/20260906-044509/openpi-proxy-empty-list/`.
Final actual evidence is `openpi-proxy-public-005/compile.json` and
`openpi-proxy-public-005/verification/report.json`; raw inputs and all three
outputs are adjacent. `public-005-source/` freezes the relevant public helper,
CLI and verification code. H100 runs use Torch 2.10.0+cu128.

- Original EP SHA256: `d4167e15186da77f7ca7a37ca2dd05a222e61e32b81833910a8fb3dfe32561c9`.
- Final package SHA256: `ebc0a979c8e62253f84f777f7a1dfa4e6030c6359e2466de48c20a324c62ea8b`.
- Final `aoti_export.py` SHA256: `74279641cc1b499ba656513efc1736217883588248f6a037252277b3f2fcdc5f`.

## Cause and Repair

Torch 2.10's OSS proxy registers empty integer, symbolic-integer and tensor
lists as dynamic arguments of length zero, but skips their initialization
during calls. Required `[]` therefore remains an IValue `None`. The real
`aten.ones.default([])` package fails with
`Expected SymIntList or IntList but got None`.
The pinned upstream C++ source and SHA are retained under the evidence root:
`oss_proxy_executor-v2.10.0.cpp` (registration around lines 247/259, skip at 734).

A post-grad-only native-lowering marker is insufficient: `ones` has an official
decomposition but no direct native lowering, so that experiment failed with
`both a fallback and a decomp for same op: aten.ones.default`. Both failures
remain in the isolated diagnostic evidence directories.

The public preparation now uses two explicit stages:

1. `prepare_backend_program(program, configs)` clones only the FX graph and
   signature, with independent custom metadata and shared original weight
   storage. Schema-identified empty dynamic lists are marked before AOT
   selective decomposition, including omitted empty schema defaults. The
   original EP graph, metadata and tensors are not changed by this preparation.
2. `prepare_backend_options(configs)` retains its post-grad schema/default,
   scalar and nonfinite handling, and marks any remaining empty dynamic lists
   after AOT retracing. In the actual time case, the pre-AOT ledger contains
   `ones.size`; the post-grad ledger contains the official decomposition's
   `full.size`.

`None` is not changed into an empty list, and no list is changed to `[1]`.
Nonempty lists are untouched. Empty statically serialized float/bool lists are
outside this repair. Native lowering may still reject unsupported operators;
there is no silent eager/Python fallback or claim of universal operator support.

## Records and Integration

The compile result adds `backend_program_audit = {passes, rewrites}` for the
pre-AOT stage. `backend_graph_passes` and `backend_graph_rewrites` retain their
post-grad meaning. `backend_program_pass_records(configs)` and existing
`backend_pass_records(configs)` expose the expected stage/source identities.
Post-grad graph caching remains disabled to retain actual rewrite evidence.

Public CLI, operator compilation and SmolVLA artifact reuse are connected.
RDT's artifact gate also checks the new ledger; OpenPI's owning adapter checks
it independently. Prior ATen-preserving manifests without this pre-AOT stage
or with an old helper SHA fail the new provenance checks. Historical reports
and their frozen sources remain valid historical evidence, not new acceptance.
Legacy direct compilation tools that do not enable `fallback_by_default`
do not require or use this workaround.

The first public verification process had a missing lazy `codecache` import
and failed before loading or producing candidate outputs. That tooling failure
is retained separately in `public-004-verifier-failure.txt`; it is not counted
as a numerical failure. Final verification uses the corrected frozen tool and
a separately compiled package.

## Regression Command

```sh
env PYTHONPATH=vlaforge/python python -m pytest \
  vlaforge/tests/unit/test_aoti_export.py \
  vlaforge/tests/unit/test_compile_artifact_cli.py \
  vlaforge/tests/unit/test_smolvla_fresh_tool.py \
  vlaforge/tests/unit/test_operator_benchmark_tool.py \
  vlaforge/tests/models/test_rdt_build_tool.py \
  vlaforge/tests/deployment/test_aoti_empty_list_diagnostic.py \
  vlaforge/tests/unit/test_aoti_package.py \
  vlaforge/tests/models/test_openpi_aoti.py -q
```

Final CPU run: 164 passed. JUnit evidence:
`public-api-final-164-cpu-tests-corrected-fixture.xml`. The earlier 164-case
attempt retained an incorrectly named size field in the newly added RDT
metadata fixture and failed that fixture's positive case; its failed JUnit is
preserved. These tests do not claim GPU performance,
full-model numerical acceptance, CUDA replay support, or board validation.
