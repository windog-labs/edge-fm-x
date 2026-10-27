# ONNX External Data Compaction

Status: **ready, 56 CPU tests passed; production tool/test source frozen** for the parent's actual part publication. This worker edits only the new tool, new unit file, `artifacts/orin/20261023-12/`, and this report. `partition_onnx_values.py`, existing exporter, source graphs/weights, old root09/root05 evidence, services and target state were not modified. No Orin/SSH/GPU, actual neural-model execution, installation or model-performance experiment was performed.

## Stable API

```python
from compact_onnx_external_data import publish

rewritten_model, report = publish(
    model, source_dir, output_dir,
    model_name="model.onnx",
    data_name="model.onnx.data",
    chunk_bytes=8 << 20,
)
```

`model` is an already extracted `ModelProto`, loaded with `load_external_data=False`. `source_dir` is the original external-data root, **not** a directory to which unrelocated part metadata was merely saved. `output_dir` must be absent, its parent must exist, and it must be outside the source tree. Names are distinct plain basenames. The returned model is a copy, with only its external-data location/offset/length entries retargeted. Default outputs are `model.onnx`, `model.onnx.data`, and `compaction.json`.

The CLI is equivalent:

```sh
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python vlaforge/tools/compact_onnx_external_data.py \
  --input /absolute/extracted-part.onnx --output-dir /absolute/fresh-part-publication
```

For CLI use the extracted graph must still be beside its referenced sidecar. API callers may instead supply the original source directory explicitly. No ONNX saver, NumPy tensor conversion, implicit external-data loading, dtype conversion, tensor pruning or graph transformation is performed. Graph publication uses `SerializeToString`; learned bias/gamma/beta, shapes, dtypes, names, inline values and other model metadata are preserved.

## Validation And Publication

The tool reuses the exporter's `external_files` recursive discovery/path checks and `digest` convention. It deliberately does **not** reuse its hard-link publisher. All TensorProtos are visited, including nested graph/attribute tensors and sparse initializer values/indices; every external tensor present in the extracted model is preserved, even if an upstream extractor retained an unused initializer.

Relative paths cannot escape the source root or use symlink components. Sources must be regular, single-link files opened read-only with no-follow checks. Duplicate/unsupported external metadata keys, materialized/conflicting payloads, negative/overflowing offsets/shapes, segmented tensors, truncated ranges and inconsistent dtype/shape byte extents reject. Fixed-width numeric/BOOL types and packed 4-/2-bit storage use exact bit-width/ceiling-byte extents. Variable-width STRING/undefined/unknown external types reject. `checksum`/`basepath` are explicitly unsupported: silently preserving a stale file-wide checksum or rereading an entire unreferenced source file is not an implicit policy.

Exact file/offset/length aliases are grouped once. Overlapping and adjacent ranges are merged per source file; only these physical unions are streamed into one newly created sidecar. Different files are not value-deduplicated. Source and destination ranges have independent SHA256 verification, including each distinct member range, not just a shared full-file hash. Metadata-body comparison, with only external-address entries excluded, proves no other protobuf field changed. Caller metadata is checked unchanged before return.

Source pins are referenced bytes plus stable device/inode/size/mtime/ctime/link-count checks before and after; **unused source payload is not rehashed**. Reads use bounded chunks, no tensor decoding, and no source temporary/checker files. The publisher reads each source union three times: pin, checked copy, independent post-copy verification. Output verification reads only the compact output. The sidecar and graph are fresh single-link regular files, never symlinks or hardlinks. Weights are written/verified before the graph. Existing output directories/files are never replaced. Failed attempts raise; where an output directory was created, partial weights and `failure.json` are preserved but an owned graph is removed. A failed directory is not a valid publication and must not be reused.

`compaction.json` records source before/after identities, tensor path/name/type/shape/address mappings, unique-range aliases, independently matching before/after/output SHA256 values, model/sidecar SHA256 and link counts. Accounting fields are explicit:

| Field | Meaning |
| --- | --- |
| `source_used_bytes` | Sum of all logical tensor-reference lengths, including aliases |
| `source_unique_range_bytes` | Sum of distinct file/offset/length spans |
| `source_union_bytes`, `copied_bytes` | Physical referenced unions actually copied |
| `exact_duplicate_bytes` | Logical references minus distinct spans |
| `overlap_duplicate_bytes` | Distinct spans minus physical unions |
| `total_duplicate_bytes` | Logical references minus physical unions |
| `source_unreferenced_bytes` | Source file sizes minus physical referenced unions |

For the parent's four-part planner with no alias duplication, stage sidecar sizes should sum to the planner's referenced payload, approximately 11.257 GB, not four copies of that file. Actual reports must establish those numbers. This worker did not duplicate or publish the actual large stages; parent owns the source-out15 publication and disk budget. Failed attempts retain partial bytes and therefore still consume disk space.

## Tests

[tests-001.xml](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-12/tests-001.xml) records **56 passed, zero failures/errors/skips**. Coverage includes exact/partial overlap aliases and independent range hashes; two backing files; scalar/zero/default-offset/default-length tensors; numeric/complex/BOOL/BF16/packed storage; every bias/gamma/beta fixture; nested/sparse tensors; inline data and source-metadata immutability; truncation, malformed extents, path escapes, links, duplicate/unsupported metadata, loaded payloads, existing outputs and source/output mutation. Instrumentation rejects calls to ONNX loading/saving, array conversion and hard-link creation inside publication and verifies only the referenced source unions were scanned.

The tiny synthetic ORT fixture is byte-exact on CPU, and the published model still loads and runs after the original synthetic source directory is deleted. This is a small storage-publication test, **not** execution of the actual neural prefix or source-equivalence validation of a partition plan. No full model is called valid: helper success is specifically `published-byte-verified-not-runtime-validated`. Actual neural execution of all parts, routed full outputs, fidelity, target support and resource admission remain separate parent-owned gates. Resource-guard aborts are not relabeled as kernel OOM.

Exact test command:

```sh
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q vlaforge/tests/unit/test_compact_onnx_external_data.py \
  --basetemp artifacts/orin/20261023-12/pytest-work-001 \
  -o cache_dir=artifacts/orin/20261023-12/pytest-cache-001 \
  --junitxml artifacts/orin/20261023-12/tests-001.xml
```

Use fresh artifact paths for any later rerun. [READY.json](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261023-12/READY.json) is the parent-facing ready/frozen signal. The production source and test will not be edited during the parent's publication.

| File | SHA256 |
| --- | --- |
| [compact_onnx_external_data.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/vlaforge/tools/compact_onnx_external_data.py) | `eac90aa9906bb8d5d5cdb331f9ea2e32498f61c27c64e6b0c96f819db6ebb046` |
| [test_compact_onnx_external_data.py](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/vlaforge/tests/unit/test_compact_onnx_external_data.py) | `d2f4cff3579eba7198207f77fa6c9e63b8501fba4c55dafe25c89f66be5a8165` |
| Test XML | `ed3356c673f84ed9ef01e3aed4ed442307d3afc8499e863abbd229e6a76a652f` |
