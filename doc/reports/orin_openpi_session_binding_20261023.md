# OpenPI Session Binding: Host Checkpoint

Date: 2026-10-23. Scope: new
[`artifacts/orin/20261023-08/`](../../artifacts/orin/20261023-08/)
and this report only. No old root04/root16 artifacts, shared source, task board,
target processes or installs were changed. This worker used no SSH, GPU,
TensorRT deserialization or model inference. Goal acceptance remains active at
**SmolVLA + RDT, 2/5**, not complete.

## Usable Binder

[`bind_session.py`](../../artifacts/orin/20261023-08/bind_session.py) implements
model-neutral artifact inspection, typed Region binding and candidate Session
source generation. It uses the existing `RegionArtifactContract`,
`ArtifactVariant`, `compile_module`, `generate_compiled_cpp_session`,
`project_region_inputs` and `reorder_region_interface` APIs, also used by the
RDT/SmolVLA artifact path. No model-name dispatch, inline stand-in Regions,
synthetic providers or nominal replacement checkpoint is used.

The frozen current requests are
[`request-pi0-003.json`](../../artifacts/orin/20261023-08/request-pi0-003.json)
and [`request-pi05-003.json`](../../artifacts/orin/20261023-08/request-pi05-003.json).
Each pins the real adapted semantic Module, physical Plan, prepared Region
requirements, actual DSOs/build records, source/native evidence and numerical
declarations. Earlier 001/002 requests and audit/test receipts are retained
development checkpoints, not the current request or new model evidence.

`audit` verifies the actual source Module and Plan, exact source/schema identity,
pure uncached Regions, actual artifact bytes and source-backed signatures.
`generate` runs the same checks and rejects missing artifacts or inspection
before creating its output directory. With complete real bindings it emits
the native C++ Session, header/CMake, physical plan/certificate, artifact
contracts, full public/native mappings, and copies only verified engine/DSO
bytes into a new bundle. This generation branch has not been exercised with
a complete actual model artifact set: those artifacts are still missing.

Public image/mask/token/state/noise ports, IDs and F64/I64 storage are unchanged.
Complete normalized actions remain F32 `[1,50,32]`; native actions remain
F64 `[1,50,14]`. Source native inputs remain state, normalized actions and the
existing incoming acceptance tensor. Both native outputs and the subsequent
predicate validation/atomic group commit are preserved. Nothing inserts a
fabricated acceptance tensor or omits an output. All ten source iterations
remain; conversions are per invocation with no pointer cache/cross-chunk reuse.

## Actual Bound Evidence

Current assessments:
[`pi0-binding-audit-004.json`](../../artifacts/orin/20261023-08/pi0-binding-audit-004.json),
[`pi05-binding-audit-004.json`](../../artifacts/orin/20261023-08/pi05-binding-audit-004.json).
The full per-Region I/O and current missing artifact contracts are in
[`pi0-required-artifacts.json`](../../artifacts/orin/20261023-08/pi0-required-artifacts.json)
and [`pi05-required-artifacts.json`](../../artifacts/orin/20261023-08/pi05-required-artifacts.json).

| Public Schema | Full Digest |
|---|---|
| pi0 | `8751fa709441f09883da726d2cc2d6b52f7a5894bf5876ee36b6803e2dff9575` |
| pi0.5 | `2cca61bd2132c82d053daba6ea951fa0e1f2d0125dd66db8bbb13a3fece2bd74` |

Five actual CUDA native providers are checked against their complete
[native build receipt](../../artifacts/orin/20261023-00/native-regions-native-build-001/receipt.json),
registration/profile signatures, all frozen source dependencies and the real
[GPU functional audit](../../artifacts/orin/20261023-00/native-region-gpu-smoke-001-audit.json).
The source harness asserted the actual workspace query; the binder validates
the exact supported provider-family formula and rejects arbitrary DSO defaults.

| Profile | Source Region ID | External Workspace Bytes / Alignment |
|---|---:|---:|
| pi0-r0, F64->F32 RNE | 0 | 384 / 256 |
| pi0-r1, checked I64->I32 | 1 | 448 / 256 |
| pi0-r3, native F64 + accepted | 3 | 0 / 1 |
| pi05-r0, checked I64->I32 | 0 | 1056 / 256 |
| pi05-r2, native F64 + accepted | 2 | 0 / 1 |

All workspaces are CUDA:0. Native providers require synchronization and expose
neither shared execution-context nor external-capture capability. They remain
outside capture. Zero external workspace is not zero internal memory: native
DSOs own streams/pinned staging; TensorRT owns its context/tactic memory. These
are explicit ownership declarations, not exhaustive resource accounting.
The native audit is non-neural: 128 source chunks / 89600 F64 values byte-exact,
MSE 0 and 97772 assertions. Initial target PID/start identity was not recorded;
it is not formal sampling or complete neural-model evidence.

The real [step-006 receipt](../../artifacts/orin/20261023-00/pi0-step-build-006/build.json)
is checked by its `built-not-validated` status, exit 0 and actual local engine
bytes, not the enclosing controller exit. The 640218807-byte engine is
independently rehashed as
`b130aa7e403bc203f7d46db60c840a7a5a9e8d9d8941887e7ecc9e45a1a99d1d`.
Its graph/precision007 pins and all 42 reported bindings are checked. This does
not yet produce a full TensorRT Region contract: actual named tensor modes,
types, dimensions, formats, location, shape-I/O flags and context memory are
still required. They are not inferred from ONNX or fabricated from layer text.
The exact intake schema is
[`engine-inspection.schema.json`](../../artifacts/orin/20261023-08/engine-inspection.schema.json).

Numerical/effect declarations bind artifact/source hashes and pinned evidence.
They explicitly retain `complete_model_fidelity_verified=false` and
`artifact-local-only` numerical enforcement. They do not pretend that a typed
legacy artifact contract supplies a missing runtime numerical-policy provider
or proves FP16/BF16 equivalence. Full effect semantics remain source-reviewed
declarations, not a sandbox proof against an arbitrary malicious DSO.

## Prefix Lineage And Pruning

The new binder independently reruns the now-stable generic LayerNorm helper on
metadata for v4 -> v5. It compares every helper-produced report field and the
entire final protobuf, then checks exact public graph I/O and original/sparse
initializer messages. No success boolean is accepted as proof. Source/output/
tool/report pins are checked before and after replay; external-data directory
and initializer references remain unchanged. This worker does not rehash the
11.257 GB learned payload or import the rewrite as numerical equivalence.

The actual v5 prefix graph SHA256 is
`4d0eede245e56282630801993cc824b7aa0b239e2f2ba7733dae26a5b892c847`;
the helper SHA256 is
`d2e0ed7e22cd99a1271f218b56bad8481d882b2c8dba64dc4e4d348bc83dfd44`.
All 165 matches have `[1,256,1152]` FLOAT data and use the existing standard
`LayerNorm/1` plugin, empty namespace, FP32, unchanged epsilon/parameters.
No new custom LayerNorm library is invented. The source-derived
[v5 precision proposal](../../artifacts/orin/20261023-06/proposal-prefix-v5-001/precision-proposal.json)
requests four FP32 dense nodes, 806 FP16 MatMul and 165 FP32 LayerNorm plugins;
it remains a proposal, not accepted output fidelity.

Prefix-001's native LayerNormalization importer failure remains separate from
prefix-002's setup-only failure before parsing. Parent reports prefix-003 live
and importing standard LayerNorm plugins. No terminal build-003 engine receipt
is admitted by this checkpoint. The binder always requires a successful inner
`build.json`, a nonempty engine with matching bytes/hash, and actual inspection.

If TensorRT prunes an input such as unused noise on the source time Region,
the binder uses an explicit inspected name order and `project_region_inputs`.
It proves the removed input is not consumed anywhere in the pinned graph,
including nested subgraphs; otherwise it rejects the projection. Output
projection, dtype changes, fabricated inputs and graph-used inputs are rejected.
This is conservative: pruning that needs a stronger static-shape proof remains
a reproducible blocker until a real prepared graph/proof is provided.

## Driver And Runtime Boundary

[`driver-handoff-001.json`](../../artifacts/orin/20261023-08/driver-handoff-001.json)
binds worker09's existing interface and worker07's actual aarch64
[link manifest](../../artifacts/orin/20261023-07/result-003/link-manifest-native-002.json).
Both worker09 target requests have exactly the same full public ModuleIO as
these Modules, not just a matching digest. Root08 generates no raw API or
second driver header. Worker09 remains the single driver/header generator;
worker07 supplies its existing headers/libraries and dependencies, including
the native SentencePiece `atomic` dependency. This is integration metadata,
not execution of the raw API or driver by this worker.

The eventual generated shared Session uses ABI 2 symbols
`vlaforge_model_session_api` and
`vlaforge_model_session_create_from_bundle`. Existing CMake supports
`-DVLAFORGE_GENERATED_SHARED=ON` and
`-DVLAFORGE_TENSORRT_STANDARD_PLUGINS=ON`. The current handoff has null Session
DSO/bundle paths because no complete Session was generated. Ordinary-loop
mode has no replay task expectations; a shared/replay mode is not invented.

Two task-local, checked extensions are applied only to eventual generated
sources, never to shared code. Native descriptor alignment is advertised only
after checking the actual pointer address against each typed artifact's
required alignment; no bytes, public types or shapes change. CUDA/internal
backend failure in the isolated candidate exits 86 before stack unwinding,
preserving the process-lifetime ownership boundary for unproven completion.
This is an explicit fatal candidate policy, not a recoverable library policy.
Generator-template changes reject the extension rather than silently applying
a broad text rewrite. Complete target source/build/runtime verification of
these generated extensions remains pending with the full artifact set.

## Tests And Next Command

[`tests-004.xml`](../../artifacts/orin/20261023-08/tests-004.xml) records **29 CPU
tests passed**. Coverage includes both real source Module/Plan verifications,
all five real DSO contracts/workspaces, actual step integrity, real v5 replay,
missing-binding rejection with no output directory, wrong schema/device/target,
changed workspace, effects/numerical mislabeling, file mutation, unsupported
engine ABI/context metadata, checked unused-input projection, guarded alignment,
isolated failure guards and a real inner compile failure despite outer-success
conventions. Small graph/inspection/code-text fixtures are unit tests only;
there are no synthetic model providers or inference results.

First reproducible blockers are actual prefix/time/finite artifacts and actual
engine inspection; pi0.5 also lacks its actual step engine. No pending artifact
hash or workspace measurement is filled with a dummy. No complete Session
source/binary or full numerical/performance result is claimed.

From the repository root, this command must currently fail nonzero and create
no output directory:

```sh
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 PYTHONPATH=vlaforge/python:artifacts/orin/20261023-08 \
  python3 -B artifacts/orin/20261023-08/bind_session.py generate \
  --request artifacts/orin/20261023-08/request-pi0-003.json \
  --output artifacts/orin/20261023-08/pi0-session-after-real-bindings-001
```

When the parent supplies real receipts/inspection, make a new configuration
and freeze a fresh request with `freeze_requests.py --configuration PATH
--output NEW_REQUEST`, then run the same generator against a fresh directory.
Do not overwrite these published requests/delivery or the older root04/root16.
The [delivery receipt](../../artifacts/orin/20261023-08/delivery-001.json) records
the current hash/test/dependency checks and exact rejected generation command.
