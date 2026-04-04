# Kernel Authoring

## Start Here

写 cuTile Python kernel 时，默认按这个顺序找资料：

1. `docs/source/quickstart.rst`
2. `docs/source/execution.rst`
3. `docs/source/data.rst`
4. `docs/source/operations.rst`
5. 与任务最接近的 `samples/*.py`
6. 对应的 `test/test_*.py` 或 `test/kernels/*.py`
7. 必要时再看 `src/cuda/tile/_execution.py`、`_stub.py`、`_datatype.py`、`_ir/ops.py`

## Most Useful Entry Points

### Minimal runnable examples

```text
third_party/cutile-python/samples/quickstart/VectorAdd_quickstart.py
third_party/cutile-python/samples/VectorAddition.py
third_party/cutile-python/test/test_frontpage_example.py
third_party/cutile-python/test/test_readme_example.py
```

### Common kernel patterns

```text
third_party/cutile-python/samples/MatMul.py
third_party/cutile-python/samples/BatchMatMul.py
third_party/cutile-python/samples/Transpose.py
third_party/cutile-python/samples/LayerNorm.py
third_party/cutile-python/samples/FFT.py
third_party/cutile-python/samples/AttentionFMHA.py
third_party/cutile-python/samples/MoE.py
third_party/cutile-python/samples/AllGatherMatmul.py
```

### Behavioral tests by topic

```text
third_party/cutile-python/test/test_load_store.py
third_party/cutile-python/test/test_gather_scatter.py
third_party/cutile-python/test/test_mma.py
third_party/cutile-python/test/test_mma_scaled.py
third_party/cutile-python/test/test_copy.py
third_party/cutile-python/test/test_reduction.py
third_party/cutile-python/test/test_scan.py
third_party/cutile-python/test/test_static_eval.py
third_party/cutile-python/test/test_static_assert.py
third_party/cutile-python/test/test_static_iter.py
third_party/cutile-python/test/test_control_flow.py
```

## Search Patterns

### Find public API usage

```bash
rg -n "@ct\.kernel|ct\.launch|ct\.load|ct\.store|ct\.gather|ct\.scatter|ct\.mma"   third_party/cutile-python/samples third_party/cutile-python/test
```

### Find compile-time helpers and target-specific knobs

```bash
rg -n "ByTarget|static_eval|static_assert|static_iter|allow_tma|latency"   third_party/cutile-python/docs/source   third_party/cutile-python/samples   third_party/cutile-python/test   third_party/cutile-python/src/cuda/tile
```

### Find dtype / shape / tile behavior

```bash
rg -n "DType|Tile|TiledView|PaddingMode|RoundingMode|MemoryOrder|MemoryScope"   third_party/cutile-python/docs/source   third_party/cutile-python/src/cuda/tile
```

## Practical Workflow

1. 从最接近的 sample 复制结构，不要从零开始猜 API。
2. 再找与该 sample 对应的 test，确认边界条件、dtype、layout、shape 假设。
3. 如果 sample 只展示 happy path，而用户问题是边界行为，就切到 `test/test_*.py`。
4. 只有当 docs 和 sample 不能解释当前行为时，才展开 `src/cuda/tile/_ir/ops.py` 或 `_execution.py`。

## Public API vs Internals

- `cuda.tile` public API：优先 docs + samples
- public API 的真实运行路径：`src/cuda/tile/__init__.py`、`_execution.py`
- type / tile / array semantics：`_datatype.py`、`_stub.py`、`docs/source/data.rst`
- load/store / gather/scatter / hints：`docs/source/performance.rst`、`src/cuda/tile/_ir/ops.py`

## When To Pull In Other Skills

- 遇到 TMA、驱动、CUDA toolkit、Nsight、底层性能定位：联动 `cuda-skill`
- 遇到 CUTLASS / CuTe 对照分析：再联动 `cutlass-skill`
