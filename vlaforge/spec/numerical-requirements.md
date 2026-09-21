# Numerical Requirements and Compile Records

Status: neutral runtime sidecar, generated Session transport and the actual
LibTorch 2.10.0 validation provider for AOTI/TorchScript are implemented. CPU
native getters and generated TorchScript Sessions have been tested; real VLA
policy-bound deployment is not yet verified. TensorRT is **unsupported**.
Existing record-only bindings remain unimplemented; they
are never promoted by reading the current environment. A supported provider
must accept an explicitly requested policy before any Region loads. None of
these records or checks is evidence that model outputs passed a fidelity gate.

## Separate Claims

- `NumericalPolicy` is a backend-owned, explicitly versioned namespace and a
  sorted immutable set of scalar settings. There is no Torch dependency or
  model-name dispatch. Python capture observations can be supplied through this
  interface, but no implicit conversion or default snapshot is performed.
- `NumericalCompileRecord` binds reference policy, requested compilation policy,
  observed compilation policy, EP/graph/artifact SHA, artifact size, backend,
  target, compiler version and canonical backend configuration. Configuration
  can include exact options, rewrite identifiers and library versions. Requested
  and observed compile policies must have equal typed canonical digests.
- `NumericalRequirement` binds the runtime policy, reference context SHA and
  compile-record SHA. Its only specified enforcement mode is `require-current`.
  The `same-precision` and `quantized` lanes produce different identities; neither
  means that outputs passed a gate.
- `RegionNumericalBinding` connects a Region, requirement and compile record.
  Version 1 retains `runtime_enforcement=unimplemented`. Explicit version 2 uses
  `provider-required/1`, a runtime acceptance obligation, not a success claim. Runtime
  observations and measured output evidence are intentionally not represented
  by free booleans in these records. Their future records must separately bind
  execution boundaries, artifacts, inputs, reference outputs and full metrics.

The first schema supports only an identity projection from the observed compile
policy to the runtime policy. Backend-specific compile-baked/runtime-required
projections need a future validated contract; no field is silently discarded.
Reference and compiler namespaces may differ, allowing backend-specific policies
without forcing Torch settings into TensorRT/BPU. No equivalence is inferred
between different reference and compiler vocabularies.

Canonical JSON uses sorted string keys, compact separators, ASCII escapes and
finite JSON values. SHA-256 identifies those exact bytes. Scalar types remain
distinct: `true`, `1`, `1.0` and `"1"` have different policy identities. Missing,
unknown and duplicate record keys, non-finite values, mismatched digests and
unsupported versions fail. Configurations are stored as immutable canonical
JSON text rather than retaining a mutable caller dictionary.

These records verify internal identity and consistency, not authenticity of a
user-supplied observation. EP archive verification, actual compiler execution,
runtime checks and model output comparisons remain separate operations. A
compile record cannot repair a historical artifact's missing numerical evidence.

## Versions and Binding

| Document | Legacy | Policy-Bearing |
| --- | --- | --- |
| Region artifact | `vlaforge.region_artifact/3` | `vlaforge.region_artifact/4` |
| Compilation certificate | `vlaforge.compilation_certificate/2` | `vlaforge.compilation_certificate/3` |
| Compile bundle | `vlaforge.compile_bundle/4` | `vlaforge.compile_bundle/5` |

Legacy serialization remains unchanged and contains no numerical fields. A
missing policy means unspecified, not verified. New readers accept legacy
documents; new policy-bearing documents deliberately use versions that older
readers reject instead of silently ignoring requirements. Explicitly selecting
a new version without requirements, or putting numerical fields under a legacy
version, fails. New-version document keys are strict.

`RegionArtifactContract.numerical_binding` must match the Region name, backend,
target, graph SHA and artifact SHA/size. A policy-bearing certificate contains
sorted unique `numerical_bindings`; the bundle requires them to match exactly
the policy-bearing artifacts and additionally serializes per-Region binding
digests. Mixed legacy/policy artifacts remain distinguishable and do not imply
whole-model numerical coverage. The bundle as a whole is not deployable while
any requirement lacks runtime enforcement.

The existing schema constants/defaults keep their legacy values. Use the explicit
`NUMERICAL_ARTIFACT_SCHEMA`, `NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA` and
`NUMERICAL_BUNDLE_SCHEMA` constants when constructing policy-bearing metadata.
For example, attach an already validated binding with `dataclasses.replace` and
the corresponding explicit schema. Schema changes never alter the mathematical
Invocation IR, scheduler, loop policy, Plan or old C ABI tables.

## Fail-Closed Boundaries

`require_runtime_deployable()` rejects record-only bindings and backends without
a numerical provider route. Implemented routes are `shared_plugin` and the
explicit supported LibTorch policy for `aoti`/`torchscript`, with Session-resident
Regions. Success means code-generation eligibility, not
that the actual plugin has accepted the policy. Legacy objects remain compatible;
this method returning normally for legacy objects is not a fidelity check.

`build_artifact_compile_bundle()` checks every contract before compiling or
creating output directories. `generate_cpp_session()` and
`generate_compiled_cpp_session()` require exact typed binding identities in both
the certificate and artifact descriptors; unverified certificate mappings are
rejected. They lower only explicit version-2 bindings to immutable C++ typed
tables, preserving namespace, scalar types, policy and requirement digests.
Integer transport is signed int64; values outside that range are rejected.
Artifact, certificate and bundle versions in the table above remain unchanged:
their strict `runtime_enforcement` marker must agree with the nested bindings.
Older readers reject the new marker/version instead of dropping it. Mixed
record-only/provider-required modes in one certificate are rejected.

`CompileBundleManifest.verify_files()` still verifies byte identity only. A
policy-bearing manifest may describe existing bytes for provenance review, with
`runtime_enforcement=unimplemented`; file verification does not make those bytes
safe to execute or numerically verified. `load_bundle_manifest()` rejects
duplicate keys, including nested policy/configuration entries.

The neutral module does not interpret backend fields or change any setting.
Providers read their actual process/current-thread state. There are no Session
setters, implicit environment restoration, precision upgrades, graph-policy
overrides or changes to existing artifact binaries.

## Native Provider And Session

`runtime/numerical.h` defines an optional ABI-v1 sidecar, independent of the
existing Region and Session ABI tables. A plugin may export
`vlaforge_numerical_provider_api`; absent or malformed sidecars fail only when
numerical enforcement is explicitly requested. Legacy loading is unchanged.

The provider validates its namespace, every key/type/value and lane in
`query_support`, obtains an immutable current-policy lease in `acquire_current`,
binds it before Region load, checks it with `validate_current`, and releases it
after Region destruction. All callbacks are nonthrowing, host-only and
validation-only. A provider must return unsupported for unknown requirements,
not substitute defaults. Callback acceptance is not a measured output record.

The runtime-owned `NumericalLeaseSet` stages all requirements before acquiring
any lease. It rejects incompatible namespace/policy/typed entries in a shared
provider process domain, even if the caller supplies matching digest strings
for contradictory entries. Provider domains are not keyed by CUDA ordinal.
The provider itself owns cross-Session arbitration in its actual process domain;
the runtime does not claim its per-Session object or a static-library mutex is a
process-wide broker across different DSOs.

Generated initialization order is:

1. Verify every artifact file, then resolve every plugin/API and policy support.
2. Acquire every required lease, then validate all policies before any Region
   create/load. Acquisition failure releases previously acquired leases.
3. Create each Region, bind its lease/context, load its artifact and workspace.
4. Drain loaded Regions/contexts and check policies again after all loads.

At every Run entry, policy validation precedes input preparation or cache reuse
on the actual calling thread. After staging full outputs, all loaded Regions
and contexts drain, then policies are checked before transaction commit or
output publication. Failure aborts the transaction, invalidates staged outputs
and caches, destroys reusable graphs after successful drain, and permanently
rejects further Run calls on that Session. Previously committed outputs remain
available. Restoring a global setting externally cannot reactivate the failed
Session or reuse its cache.

Normal destruction orders graphs, drained Regions/backend storage, contexts,
leases and plugin handles. A drain failure quarantines storage, contexts,
Regions, graphs and leases until worker-process exit; it must not be reported as
successful cleanup or ordinary fallback. Fatal graph-capture cleanup also
retains policy ownership. This slice tests CPU drain failure, not actual CUDA
capture failure with a numerical provider. AOTI graph-only policy enforcement,
standalone backend invocation enforcement and structured observed-value
reporting remain future backend work.

A lease does not intercept an unrelated external global-state setter. Entry
and exit checks cannot exclude mutation-and-restore between checks. Strict
isolation requires an explicitly exclusive/cooperative worker. No claim of
process-global isolation or hidden policy restoration is made here.

## LibTorch Domain

`deployment/libtorch_numerical.py` explicitly projects a newly observed
`NumericalContext` v2 into `libtorch.python_context_v2/1`. It requires the exact
Torch 2.10.0 reduction API, all 22 flags and two release/API identifiers. The full
Python package version remains compile provenance, not a guessed C++ getter.
Old v1 observations and bool-only reduction APIs cannot be upgraded into this
domain. FP16/BF16 reduction each retain all three distinct reduction/split-K
pairs. The matmul precision/TF32 aliases must be consistent, including `medium`.

`vlaforge_libtorch_numerical_backend` is one shared provider DSO, linked by both
AOTI and TorchScript. Its process registry arbitrates active policies and its
callbacks read the actual global and calling-thread ATen getters. One matching
LibTorch instance and one provider registry per worker are required; independently
loaded copies are not coordinated. Other LibTorch releases reject this domain
without evaluating unsupported getters or modifying settings.

This domain does not cover CPU/inter-op thread counts, InferenceMode, JIT
optimization, strides/layout, streams, SDPA priority order, newer FP32 overrides,
all MKLDNN/cuBLAS environment flags, allocators or hardware/library identity.
Those need separate execution contracts and provenance. No equality of 22
getters establishes complete floating-point execution equivalence.

## Explicit Worker Bootstrap

The validation-only provider and generated Session never change policy. A
separate `vlaforge_libtorch_numerical_initialize_worker` API can initialize a
dedicated worker before model/device work or its first successful numerical
lease. Its versioned options require both explicit exclusive-process and
calling-thread ownership acknowledgements. These declarations cannot prove
that third-party threads or setters are isolated.

All fields and options are checked before mutation. Only changed declared flags
are set, reduction setters always receive both arguments, and matmul precision
is set through its primary setter without overwriting `medium` through its TF32
alias. Full readback must match. A partial failure restores and checks all 22
original flags; an unverifiable rollback poisons the worker and requires exit.
The API does not restore unlisted execution settings or claim device cleanup.

Successful initialization fixes one worker policy. Same-policy, same-thread
repetition is only idempotent before the first lease and while all current
flags still match. Retuning, initialization after any successful acquisition
(even after release), and thread transfer are refused. A retained thread-local
identity prevents OS thread-ID reuse from authorizing another thread. Session
acquisition also checks the initialized policy after all prior leases release.

Python deployment code supplies all already verified LibTorch Region bindings
to the generic renderer, which rejects conflicting policies:

```python
from vlaforge.codegen.numerical import generate_libtorch_worker_initializer

bootstrap = generate_libtorch_worker_initializer(
    bindings,
    acknowledge_exclusive_process=True,
    acknowledge_calling_thread=True,
)
# runner_body explicitly calls the generated function before constructing Sessions.
runner_source = bootstrap + runner_body
```

The runner checks the returned `VLAForgeStatus` from
`vlaforge_initialize_numerical_worker()` before continuing. Adapters do not emit
model-specific C++ setters. This is explicit worker setup, not implicit Session
initialization, and it cannot create missing historical capture/compile records.
Actual CPU verification includes 14 fresh-process bootstrap cases and a
seven-mode generated TorchScript runner. A separate LD_PRELOAD ATen setter
interposer verifies partial-mutation rollback and rollback-failure poisoning;
production code has no test hook. The old thread-ID reuse error and the retained
identity fix were tested against the same fresh-thread counterexample. These
are native LibTorch contract tests, not real VLA numerical acceptance.

## Verification Scope

CPU tests cover canonical round trips, typed identities, nested duplicate keys,
configuration immutability, schema downgrade rejection, changed reference/compile/
artifact identity, wrong runtime projection, sorted certificate coverage,
certificate/bundle disagreements, unsupported enforcement claims, quantized lane
separation, legacy serialization, bundle file verification and both fail-closed
deployment entrypoints. Real compiled CPU plugin/Session tests cover same-policy
leases and Sessions, missing/invalid sidecars, conflicts before load, acquisition/
creation/binding/load failures, Run-thread and post-compute/drain drift, complete
output preservation, exact-cache reuse and failure invalidation, drain-failure
quarantine, manifest build/reload and no-Python linkage. These are contract tests,
not real-model numerical fidelity or GPU/board experiment results.
