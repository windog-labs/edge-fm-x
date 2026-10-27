# Typed Operator Audit

Current target follow-up: [2026-10-24 native results](orin_embedding_rope_target_20261024.md)
include real Embedding/RoPE builds, complete numerical outputs and actual SDK
bindings. The new binding lookup has a whole-function inverse; the old auditor
below is unchanged. Opaque internal precision still prevents timing admission.
The following2026-10-22checkpoint is historical host-tool preparation.

Checkpoint: 2026-10-22. **Host tooling and archived-data checks only. No new
Orin build/inference/timing or required-model acceptance.** RDT continues its
frozen formal campaign without another target GPU experiment.

## What Changed

The previous record auditor accepts only FLOAT bindings and requires a dense
MatMul/Gemm/Conv. It remains unchanged, with all existing receipts preserved.
Two model-neutral modules provide a separate typed path:

- `typed_operator_inspection.py`: a small interface for physical bindings and
  explicitly covered data/index/storage precision. INT32 is checked as INT32,
  not treated as FLOAT. Conflicting, missing, host or non-linear binding
  descriptions reject. Empty dense evidence is N/A, never evidence of parity.
- `audit_typed_operator_pair.py`: complete archived capture/file/input/output
  validation, frozen-source reconstruction and a distinct precision decision.
  A successful numerical check alone does not set precision qualification.

The typed inspector handles declared non-dense Gather, floating arithmetic and
RoPE layout signatures. Gather needs its actual INT32 index and a traced Constant
storage type/count, including the SDK's real lowercase `weights` field. It does
not infer weight storage from a source FLOAT tensor or FLOAT output. ScatterND
requires an exposed INT32 index port; integer shape/control ports are kept
separate from floating data. Out-of-range INT64 constants cannot silently qualify
as INT32. Unknown/opaque fusion, unsupported operations, missing storage evidence
or incompatible types do not become observed precision.

The existing exposed-placement dense checks are reused. An otherwise visible
MatMul cannot hide an unrequested Gather or other unproved arithmetic path.
Internal accumulation is not independently instrumented. This is an observable
typed-data/storage contract, not a claim about every vendor instruction. Existing
configured-only/fused-state results are not retrospectively rewritten; their
separate contract and limitations remain. No configured-only approval is added
automatically by this new tool.

## Full Evidence Before Timing

Every declared input is checked against its native binding order and full file
hash. Both outputs/variants and every declared source observation remain in
the comparison. The original MSE <1e-5 / cosine >0.9999 gate is unchanged;
CPU references stay supplemental. No transfer/latency/speedup is inferred from
these correctness captures.

For newly materialized native cases, the auditor reopens the fixed original
case, graphs, external payloads and profiles, reconstructs the declared native
graphs and regenerates the entire request. It checks sample IDs, typed ports,
all original input/reference files, index bounds/roundtrip and unchanged
precision maps. A request and preparation file that are mutually consistent
but omit an original observation still reject. The original source recipe's
qualification is inherited from its pinned case, not claimed as new inference.

Numeric passage with incomplete precision gets
`same-target-numerical-gate-passed-precision-unqualified-not-timed`, with
`target_quality_passed=false` and
`same_effective_precision_main_table_qualified=false`. The ordinary timing
gate cannot mistake that for a qualified result. A complete qualified result
still has no full-model/performance claim. Missing target captures return
`pending-no-target-acceptance` with exit 3; they do not produce a success row.

The future generic controller `20261022-00/run_static_operator_case_v2.py`
adds producer-side size/hash pins for `build.json`, `layers.json` and
`model.engine` immediately after building. Original builder/probe, workspace,
FP16/noTF32/OBEY, admission and owned-child cleanup stay unchanged. The new
controller has **not run on Orin**. New native captures require these producer
pins; old archives use their already frozen independent layer-info receipts,
not a retrofit of fields the old producer never wrote.

## Checks Performed

| Evidence | Verified Scope |
|---|---|
| Original adopted full-prefix inspector | Actual `input_5` INT32 descriptor and Constant Float/count schema, not a new Embedding engine |
| Archived Attention native captures | All three original Q/K/V inputs and complete 786432-value context; same-target byte equality and prior visible precision result reproduced |
| Current Embedding/RoPE source packs | Fixed native graphs/profiles and all original points/files rechecked; no native capture exists |
| Explicit synthetic inspector fixtures | Unknown fusion/storage, changed storage type, wrong index/format/shape and empty evidence reject qualification; these are not board outputs |
| Private-copy source/capture corruption | Missing observation, bad output/input/capture bytes and inconsistent inspector pins reject; originals unchanged |
| Combined regressions | **58 passed**, including native-case/request checks and actual INT32/multi-input timing-input contracts |

The original Attention result remains an existing operator result, not a newly
run experiment. The Embedding and RoPE pending receipts explicitly state that
target/source acceptance is not completed; direct host lineage tests are a
separate scope. No new precision choice, graph recipe, quantization search,
custom kernel or model-name runtime/timing dispatch is introduced.

## Sources and Receipts

Root: `artifacts/orin/20261022-05/`.

- `typed_operator_inspection.py`, `audit_typed_operator_pair.py` and
  `test_typed_operator_audit.py`: current modules and focused tests.
- `attention-existing-capture-audit-003.json`: current archived-data recheck,
  no new target execution or timing.
- `embedding-pending-003.json`, `rope-pending-003.json`: real no-capture attempts,
  exit 3 and no target quality/performance qualification.
- `typed-and-native-operator-regression-005.xml`: 58 combined passing checks.
- Earlier prototype receipts and `_v1.py`/`_v2.py` source snapshots remain;
  current `-003` receipts bind the current modules. These revisions are verifier
  changes, not additional model/precision candidates.

Recheck existing data without target computation, using a new output:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=vlaforge/python python \
  artifacts/orin/20261022-05/audit_typed_operator_pair.py \
  --root artifacts/orin/20261022-00/operator-attention-retrieval-004 \
  --prior-quality artifacts/orin/20261022-00/operator-attention-pair-001/quality-001/quality.json \
  --output artifacts/orin/20261022-05/NEW-attention-recheck.json
```

After RDT releases the GPU and target work is scheduled, transfer the unchanged
native case/request/preparation directory and the new generic controller to a
fresh target directory. Reuse the hash-matched builder/probe/capture helpers.
Keep the complete returned archive, then audit it with `--root`, the original
host `--native-spec`, and a fresh `--output`. Build success alone is not passage;
opaque or incompatible observed placement must retain its limitation rather
than enter timing as identical precision. No new target run is launched here.

The [native preparations](orin_embedding_rope_native_preparation_20261022.md)
still lack real build/output/precision/timing results. The operator table still
has four measured families and the mandatory VLA formal count remains 1/5.
