# Attention Integration Preparation

Actual follow-up: [2026-10-24 full-model integration result](orin_operator_attention_integration_20261024.md).
The real prefix build and both complete native model paths now execute, but
normalized MSE fails the unchanged gate against both adopted and existing
unchanged-source controls. No raw E2E timing follows. The dated source-only
checkpoint below is preserved as historical preparation, not current target status.

Checkpoint: 2026-10-22. **Host source composition only: no new target build,
model inference, complete-action acceptance or integrated timing.** RDT remains
the only Orin GPU experiment. Its frozen collector and service guard are not
edited, restarted or bypassed by this work.

## Exact Source Change

The already same-Orin-qualified [Attention micro case](orin_operator_attention_pair_20261022.md)
is bound to the adopted complete prefix source, successful parent build and
SmolVLA formal audit. Its original recipe is reapplied to the exact extracted
four-node fragment and must reproduce the stored candidate and recipe proof.
This is one existing candidate, not a new recipe/precision/search budget.

| Property | Prepared Result |
|---|---|
| Full prefix nodes | 2088 -> 2087 |
| Sole removed node | `/Add_138`, immutable finite positive-zero attention mask |
| Retained core | Original QK MatMul, final-axis Softmax, PV MatMul |
| Q/K scaling and full-model projections | Unchanged |
| Original initializers, including shared/unused constants | All 840 retained verbatim |
| Outside nodes and full stage IO | Unchanged |
| Type/shape metadata | Only removed Add output entry deleted; all unrelated entries unchanged |
| Retained compute/output precision declarations | Unchanged |
| Removed precision declarations | Only `/Add_138: fp32` in each map |
| Target engine / full actions / E2E | Not produced or measured |

The source is the adopted prefix, not the standalone four-operation timing
graph. All complete-model vision/text/projection/solver work remains required
in the eventual experiment; the stored Q/K/V diagnostic activations cannot
replace online model work. Removal of a source node is not proof of one fewer
GPU launch or a complete-model speedup.

Source SHA256:
`567ef82105c2ebbb7f7afde0faa7c3a960cf80c0711e31bc5a4c10dbef58a147`.
Candidate SHA256:
`c3abe66a6a9eceff411ee3a0dde76ceadb8ac7496e2632ba448d35ca6ed5f152`.
Current candidate-profile SHA256:
`faf7eef69aba375085abb29d0163004830e7ca27cbf97bd3403be6a4e4dd3674`.
Parent adopted engine remains
`e3eaeaf3988b0a0ff919173275213c24a5fa9b3e27e156329135ddda72ae8ab3`;
the candidate has no engine identity yet.

## General Composition Contract

The existing single-operator GEMM splicer and its receipts remain unchanged.
The new model-neutral helper accepts only flat, named, inline-constant graphs
and a strict nonempty retained-node subset. It checks exact parent definitions,
typed boundaries, opset compatibility and immutable constants, and rejects
undeclared outside consumers of internal values. Retained operations preserve
their names, outputs, attributes and source order; only explicitly declared
input-edge rewiring is allowed. No new constants or operations can be injected.

Composition alone does not prove a rewrite's mathematics: the independently
bound recipe supplies that restriction. The Attention recipe still rejects
nonzero/negative-zero/nonfinite, causal, overridable and shape-expanding masks.
An independent comparison of the actual reloaded full candidate verifies all
outside nodes, initializer bytes, IO, metadata and retained type entries.
There is no model-name dispatch in runtime/timing, new plugin, quantization
search, dtype cast or silent fallback.

The candidate profile removes only the eliminated node's two declarations.
FP16 QK/PV and FLOAT Softmax requests, original scale and all remaining parent
constraints persist. This does **not** establish whole-stage actual precision,
hidden accumulation or vendor tactic parity after a future build. Previous
unchanged-source rebuild drift and the failed GEMM integration remain relevant;
neither micro equality nor graph identity substitutes for full target actions.

## Evidence

Root: `artifacts/orin/20261022-03/`.

- `compose_elided_fragment.py`: bounded general composition and separate
  actual-result verification; not a target runtime change.
- `prepare_elision_integration.py`: fixed source/request/build/recipe/formal
  bindings and source-only preparation. Parent seven-engine hashes are retained
  as future replacement constraints, not a new complete-model run.
- `attention-integration-request-001.json`: explicit paths and fixed hashes.
- `attention-integrated-prefix-002/{candidate.onnx,candidate-precision.json,preparation.json}`:
  current checked source candidate; every target/full-model claim is false.
- `elision-composition-regression-003.xml`: **43 passing** composition, lineage,
  no-overwrite and existing GEMM/admission tests. A non-Attention synthetic
  fragment tests shared constants and an interleaved outside branch. Bad
  boundaries/constants/opsets, hidden consumers, missing type metadata,
  undeclared inputs, false precision/quality and changed source lineage reject.

The first source-only composition and its 35-test receipt remain under `-001`;
the original helper is frozen as `compose_elided_fragment_v1.py`. The later
helper explicitly guards preservation of unrelated unused type information.
Regenerated `-002` has the **same candidate ONNX bytes**, not another graph
candidate or a hidden search attempt. Its 18.350-second host preparation time
is neither target build time nor inference latency. No original source bytes,
failed experiment or target artifact are overwritten.

Final host rehash checks bind both helper versions to their recorded receipts,
confirm equal candidate ONNX bytes, and confirm every pinned original source
unchanged. After the scoped progress update, eight board state/server tests and
nine desktop/tablet/mobile browser tests pass. Receipts under
`artifacts/orin-task-board/` are `attention-source-node-tests-011.xml` and
`attention-source-browser-tests-012.xml`; these are not target-model tests.

Reproduce on the host using a fresh output path:

```bash
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-03/prepare_elision_integration.py \
  --request artifacts/orin/20261022-03/attention-integration-request-001.json \
  --output artifacts/orin/20261022-03/NEW-attention-integrated-prefix
```

## Next Target Gates

Do not build or infer this candidate while the RDT campaign owns the GPU.
After that experiment closes and the next target work is scheduled, use the
existing owned-build/resource-admission workflow, preserving parent FP16,
noTF32, 2048 MiB workspace and OBEY settings. Inspect real IO/precision and
compare complete seven-engine, ten-step actions with identical raw observations
and saved noise. Only the one prefix artifact may change.

The user-confirmed complete normalized action gate remains MSE <1e-5 and
cosine >0.9999; native-unit and H20 errors are retained separately. Only after
that gate may a fresh matched raw-observation formal E2E experiment be sampled.
Retain unchanged-tactic/no-gain, negative and fidelity-failure results; do not
multiply micro speedup by runtime speedup or change the adopted main profile.
Positive complete-model graph benefit remains unproven and the Goal incomplete.
