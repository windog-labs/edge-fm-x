# Embedding and RoPE Native Preparation

Current target follow-up: [2026-10-24 real build/output/SDK ABI results](orin_embedding_rope_target_20261024.md).
Both pairs now execute and produce complete byte-exact outputs. Internal fusion
precision remains unqualified and neither pair is timed. The dated preparation
checkpoint below is retained as historical evidence, not current target status.

Checkpoint: 2026-10-22. **Host-only native case/request preparation, not target
build, precision qualification, operator timing or complete-model deployment.**
The RDT formal collector remains the only Orin GPU experiment. No target code,
engine, service, SDK, power mode or frozen experiment protocol is changed.

## Prepared Contracts

| Case | Native Inputs | Complete Output | Real Input Coverage | Materialization |
|---|---|---|---|---|
| Language Embedding + fixed scale | INT32 `[1,48]` IDs | FLOAT `[1,48,960]`, 46080 values | One distinct instruction, preserving aliases for all 16 recorded observations | Explicit bounded Gather-index ABI; source ONNX INT64 retained in lineage |
| RoPE/cache | FLOAT K `[1,177,5,64]`, FLOAT Sin/Cos each `[1,177,1,32]` | FLOAT `[1,177,5,64]`, 56640 values | Original observations 10 and 11 | Candidate's 2039040-byte external payload inlined without value changes |

Each baseline and candidate is executed only by the host CPU operator reference
checker, using its original actual input files. Every complete output remains
byte-exact to both its original source variant and frozen reference. The
Embedding check is 46080 values per variant; RoPE is 113280 across two inputs
per variant. These are not new Orin outputs, action chunks, independent prompts
or whole-model inference. No timing claim follows from this preparation.

## Explicit Changes, Not Silent Casts

The frozen Embedding ONNX input is INT64; its saved payload and adopted full
engine ABI are INT32. The new copy explicitly changes only that input type and
its matching input annotation. Both variants' runtime Gather consumes the IDs
directly; no precomputed prompt feature replaces it. The allowed domain is
`0..49279`, bounded by the original table row count and INT32. Original recorded
values range `2..31413`; each payload is range-checked and roundtrip-checked.
The source CPU comparison upcasts those same values for the original INT64
graph, then compares complete outputs with the native INT32 graph.

The general conversion is limited to declared nonnegative indices consumed by
axis-zero Gather, with known table bounds. It rejects wide/negative ranges,
unknown ports, duplicate rules, unsupported consumers/axes and index-initializer
overrides. It is not general INT64 state lowering or an arbitrary-input support
claim. A future public runner must validate incoming data; these microbenchmark
requests bind only the actual recorded inputs and their hashes.

RoPE's baseline graph bytes remain unchanged. The candidate external payload
has its original length/hash verified before standard ONNX external-data APIs
load and inline it. Node definitions, each variant's initializer value bits,
typed ports and raw input/reference files stay unchanged. Upstream diagnostic
captures remain in the original pack and are explicitly not benchmark ports;
all three declared inputs and the complete output are retained.

Embedding's source-vs-candidate table values differ by the previously declared
fixed scalar fold. The new materializer preserves **each variant's own** table;
it performs no new fold, precision change, quantization or search. Original
compute/output maps, including the candidate's empty compute map and explicit
FP32 terminal output rebinding, are preserved. The native input conversion is
separate from storage/compute precision, which still requires target inspection.
RoPE preserves its original FLOAT numerical island, not a full-model FP32
fallback. Actual target types/tactics/accumulation are not certified here.

## Graph Pins

| Case / Variant | Original Graph SHA256 | Native Graph SHA256 |
|---|---|---|
| Embedding baseline | `049fcb834714b8ecbb8b67c1f3a08e43101a5d91cc791a45fe3e491f129a1d16` | `994ff9238ddf0dea9881308c9bb13f65ef87718e53616b106194bccf860793a2` |
| Embedding candidate | `ad1f088871cc9ca419ecded805fcaf357b70036201d6183cdbbd6913988918d9` | `4ab1ef67c6d8405cb6203c38a2037269844f5afb707b34ac327a09dfbda79ae3` |
| RoPE baseline | `83c257c4889b874a4c24c8e5041ec549623483c86152874b554d54f11fe779d0` | Same |
| RoPE candidate | `c3897c380aa9943ba6881d6fa79ee6c2466caa69a24e40e63499390fc14920e5` | `8e7c3cdae8cf284d74a8e235ef7002dc093ff09ddb5162cd1e1215b000992ede` |

The inlined RoPE payload's source SHA256 remains
`c3831b16d6ed077b44173972767558f1c0aa0d876d4cdea748ad6bda17ecdd1e`.
All original case/graph/profile/payload files remain present; new graphs and
updated graph-source hashes live in separate native case copies.

## Remaining Target Gates

Later host-only [typed auditing](orin_typed_operator_audit_20261022.md) adds the
new INT32/non-dense code path and producer-side inspector pins, with 58 combined
tests. The old dense/FLOAT auditor limitation described below is preserved as
the reason for this extension. Actual attempts on these unexecuted roots still
return pending; no target build/quality/timing or observed storage parity is
certified by adding the new tool.

The existing generic build/probe runner can consume the new request schema;
this is not proof the target parser, memory or native ABI will pass. Only after
RDT releases the GPU may a scheduled build use the fixed native builder/probe,
FP16 enabled/noTF32, 256 MiB workspace and original precision declarations.
The 8 GiB resource admission and owned-job safety rules remain applicable.

The current `audit_record_operator_pair_v3.py` is **not yet an acceptance path**
for these families: it requires at least one MatMul/Gemm/Conv and its binding
checker accepts only FLOAT ports. These cases have no dense node, and Embedding
also has an INT32 port. A test confirms the existing checker rejects a Gather;
it is not relaxed to let an empty evidence set imply precision parity.

Before timing, obtain real inspector/native ABI evidence for INT32 IDs, FLOAT
outputs and non-dense data/storage/arithmetic types. Any fused/unexposed data
must retain a configured-only limitation rather than qualify observed precision.
The Embedding table-fold/output-constraint rebinding and RoPE layout rewrite
need their distinct declared semantic proofs, not an Attention zero-mask rule.
Then independently audit all actual outputs under MSE <1e-5 and cosine >0.9999.
No numeric or precision gate is bypassed by producing these requests. Compile,
quality, cost and timing results are unknown, not zero; failures remain evidence.

## Evidence and Reproduction

Root: `artifacts/orin/20261022-04/`.

- `prepare_native_operator_case.py`: model-neutral layout/role/ABI declarations,
  standard external-payload handling, CPU full-output comparisons and reuse of
  the frozen request preparer. No model-name runtime/timing branch is added.
- `embedding-native-spec-001.json`, `rope-native-spec-001.json`: original case
  pins, explicit native contracts, record/role mapping, bounded index rules and
  fixed request-preparer/builder/probe identities.
- `embedding-native-001/`, `rope-native-001/`: native `case/`, `request.json` and
  `preparation.json`. Requests bind 7/13 files respectively; target flags remain
  false. The original packs are not overwritten or relabeled as deployed.
- `native-operator-preparation-regression-001.xml`: **27 passing** native staging
  and existing request-layout tests, including private-copy payload corruption,
  missing output, unsafe sidecar and the no-vacuous-dense-acceptance regression.

The later `native-operator-preparation-regression-002.xml` has **29 passing**
tests. It additionally reopens the actual saved ONNX files, executes all complete
CPU outputs and rechecks original graph/profile/sidecar pins; the source
preparer and original evidence are unchanged. Eight state/server and nine
desktop/tablet/mobile board tests pass, recorded under `artifacts/orin-task-board/`
as `native-cases-node-tests-012.xml` and `native-cases-browser-tests-013.xml`.
None of these host tests establishes target compute/storage precision or speed.

Reproduce on the host using fresh output directories:

```bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-04/prepare_native_operator_case.py \
  --spec artifacts/orin/20261022-04/embedding-native-spec-001.json \
  --output artifacts/orin/20261022-04/NEW-embedding-native
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-04/prepare_native_operator_case.py \
  --spec artifacts/orin/20261022-04/rope-native-spec-001.json \
  --output artifacts/orin/20261022-04/NEW-rope-native
```

The [operator results](orin_operator_ablation_results_20261022.md) still contain
only four target-timed families. Neither new request fills a paper row or
complete-model gate. Required formal-model completion stays 1/5.
