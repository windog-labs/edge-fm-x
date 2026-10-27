# Prefix Weight-Budget Cut Plan

## Handoff

An actual v5 metadata-derived **four-part** plan is feasible. Use
`artifacts/orin/20261023-13/cut-manifest-001.json`, SHA256
`d51c966f1e96fc5ae0fa9fe973f2bcd4ef11cb4600f1cf5890538f1462680e68`.
It contains exact sequential partition arguments, source node indices and
protobuf hashes, external initializer descriptors, typed ports, composed input
routes, and passthrough aliases. Stage 3 takes the final remaining consumer.

The first actual API preview passed with these source values:

```python
values = [
    "add_23", "layer_norm_23", "add_78", "layer_norm_78",
    "add_133", "layer_norm_133", "mul", "cat_1", "mul_1", "_to_copy_4",
]
first, remaining, receipt = partition(
    model, values, external_base_dir=source.parent,
    trusted_plugins=("bf16-round/1", "layernorm/1"),
    hoist_constants=True,
)
```

This part has **2,802,293,696 unique external initializer bytes**, or
2.609839 GiB. It includes the original text embedding lookup and matching
prefixes of all three view branches. The returned first frontier is exactly
the ten values above, not a name-based approximation.

| First Output | Type | Shape |
| --- | --- | --- |
| `add_23`, `layer_norm_23` | FLOAT | `[1,256,1152]` each |
| `add_78`, `layer_norm_78` | FLOAT | `[1,256,1152]` each |
| `add_133`, `layer_norm_133` | FLOAT | `[1,256,1152]` each |
| `mul` | FLOAT | `[1,48,2048]` |
| `cat_1` | BOOL | `[1,816]` |
| `mul_1` | BOOL | `[1,816,816]` |
| `_to_copy_4` | FLOAT | `[1,1,816]` |

These outputs total 8,141,040 bytes. This is interface storage, **not a bound
on TensorRT builder or runtime peak memory**.

## Four Parts

GiB means `2**30` bytes. Each stage counts a referenced initializer once,
regardless of how many source nodes or image views use it in that stage.

| Stage | Unique External Bytes | GiB | Used Source Compute Nodes | Actual API Inputs / Outputs |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2,802,293,696 | 2.609839 | 3,236 | 8 / 10 |
| 1 | 2,762,950,656 | 2.573198 | 4,851 | 18 / 50 |
| 2 | 2,776,727,552 | 2.586029 | 739 | 68 / 15 |
| 3 | 2,915,147,776 | 2.714943 | 763 | 83 / 37 |
| Sum | **11,257,119,680** | **10.484010** | **9,589** | |

The original external initializer span union is also 11,257,119,680 bytes.
Cross-stage duplicated external span bytes are **zero in this selected plan**.
This is not a general deduplication assumption: splitting shared-weight view
branches at different depths can repeat the same learned weights in multiple
parts. The planner accounts stage sums and the union separately, and a focused
fixture proves that a shared initializer reused across stages is counted twice
in the stage sum. This plan synchronizes free work across the three branches,
which avoids that repetition. There are 599 external initializers distributed
193 / 292 / 55 / 59 across these parts, with disjoint source byte ranges.

The largest single source initializer is 2,106,589,184 bytes (about 1.962 GiB).
A simple source-order cut cannot place the completed three-view vision prefix
and that embedding lookup together under 3 GiB: their cumulative union jumps
to 3.507 GiB. The chosen first part is dependency-closed but noncontiguous in
source node order. Selection uses initializer byte budgets, dependency closure,
and proved static frontiers, never model-name or layer-name dispatch.

The planner's cumulative live global frontiers are 8,141,040 / 75,412,272 /
112,909,104 / 30,081,840 bytes. These include retained original outputs and
are distinct from each newly extracted producer's outputs. The current API
retains prior private inputs, including unused ones; this explains the growing
actual input counts. The manifest routes every such port to an original input
or an earlier stage output without inventing tensors or changing dtypes.

## Source And Shape Proof

Pinned source:
`artifacts/orin/20261022-08/trt-carrier-001/pi0-prefix-trt-opset17-v5.onnx`.
SHA256:
`4d0eede245e56282630801993cc824b7aa0b239e2f2ba7733dae26a5b892c847`.
The graph has 13,499 source nodes and 599 initializers. It is loaded with
`load_external_data=False`; no external tensor payload is materialized or
hashed. Byte accounting comes from typed dimensions and checked external
location/offset/length metadata. Source graph hashes are checked before and
after exploration and preview.

Shape evidence reuses root10 `audit_prefix_i64_narrowing.independent_shapes`,
including its original-v4 gamma/beta descriptor and epsilon/rank validation
for all 165 LayerNorm replacements. Its inference call is captured with
`strict_mode=True, data_prop=True` and **zero internal value_info seeds**.
The resulting static contracts agree with the new production
`infer_reviewed_shapes(model, ('bf16-round/1','layernorm/1'))` helper.
Both custom contracts require exact version 1, empty namespace and FLOAT
types; LayerNorm additionally requires proved rank-three data and matching
FLOAT channel parameters. All 3,176 Round and 165 LayerNorm nodes are covered.

The production helper's Identity shadows are only an inference mechanism.
Every actual preview deployment node is checked against its original v5
NodeProto, normalizing only explicit input passthrough aliases. No shadow
Identity replaces a Round or LayerNorm in any proposed executable part.
No plugin numerical equivalence is established by these shape checks.

Production API pin:
`dd0a8e54e1ebca35b892d12cb92ef28964800140577279e2f59a2fd96a5404e5`.
Production shape helper pin:
`703510e1867b31d730a9be59a27d2ef8e75e004fd3922ef7c0a7b9c78ee782f1`.
Root10 auditor pin:
`9290143bee3f6c70d089d3a520de9308cf22ff5361ede732f96e8df216b6230a`.

## Actual API Preview And Ownership

`preview_api.py` invokes the actual production `partition` three times on the
successive remaining consumers, then audits the last remainder. Production
hoisting, reviewed shape inference, dependency closure, extraction and alias
creation run unmodified. The only intercepted operation is `_check_model`:
its filename checker normally creates a temporary model beside the root08
sidecar, outside this task's exclusive write scope. The preview instead checks
the **real original sidecar's paths and descriptor bounds read-only**. It does
not fabricate a sidecar, publish an ONNX graph, or claim filename-checker or
weight-content validation. Parent root15 owns actual extraction/publication,
compaction, filename checking and target compilation.

`api-plan-preview-002.json` verifies all of the following against actual API
results, not just the planner's predicted closures:

- All 9,589 output-reachable non-Constant source compute nodes occur exactly
  once. Their original NodeProtos, including plugin attributes, remain intact.
- All 3 original Conv nodes, 807 MatMul nodes, 4 Gather nodes, 3,176 Round nodes
  and 165 LayerNorm nodes are retained. Full three-view and text computation is
  preserved by complete used-node ownership, not by cached features or a
  smaller token/image path.
- Canonical tensor Constants become initializers with the same output SSA,
  dtype, dimensions and stored values. There are 3,665 declared literal hoists;
  only TensorProto.name changes to the SSA name. No I64 literal becomes a stage
  binding, and there is no integer narrowing or runtime-control specialization.
- The 2,298 source nodes outside original output reachability contain 2,053
  unused Constants, 210 unused Identity aliases and 35 unused literal Reshapes.
  The unused non-Constant nodes depend only on literals. No used learned computation
  is omitted. This distinction fixes an initial exploratory all-node completion
  check that incorrectly tried to schedule dead literal aliases as a fifth part.
- Every actual stage input/output is positive-static FLOAT, BOOL or INT32.
  There are **zero I64 stage bindings**. All eight original private input names
  remain present in every actual stage input set.
- The final consumer preserves all 37 original outputs in original order.
  It has 23 extra Identity passthroughs for outputs computed earlier, including
  the prefix mask and earlier K/V outputs. Their alias routes are explicit.

## Tests And Reproduction

Focused tests: **12 passed** (`tests-002.xml`). They cover canonical I64 literal
hoisting, unchanged source bytes, model/value renaming, unknown-shape rejection,
I64/F64 runtime-frontier rejection, used-node reachability, overlapping external
span rejection, within-stage shared-weight deduplication, cross-stage duplicate
accounting, actual budgets/plugin counts, all composed route types/shapes, and
an independent actual span-union/stage-sum recomputation.

Host-only reproduction requires new receipt filenames:

```sh
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python \
  artifacts/orin/20261023-13/plan_prefix_cuts.py --output /NEW/plan.json
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python \
  artifacts/orin/20261023-13/preview_api.py --plan /NEW/plan.json --output /NEW/preview.json
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python -m pytest -q \
  artifacts/orin/20261023-13/test_cut_planner.py \
  -o cache_dir=artifacts/orin/20261023-13/pytest-cache-new
```

For actual publication, read each manifest stage's `partition_values` on the
current consumer, with the reviewed kwargs. Do not reuse the cumulative global
frontier lists as later API selections: earlier original outputs are now
private aliases. The manifest records the exact actual per-call selections,
all input routes, and the final `take-final-consumer` operation.

This work wrote only new root13 files and this report. It used no SSH, GPU,
neural execution, package installation, shared-production edits or target
measurements, and did not touch root07/root11 evidence. Neural fidelity,
TensorRT target support, actual compiler/runtime peak memory and complete
normalized/native action gates remain unverified. The initializer budget alone
does not establish that any stage fits the parent's available target memory.
