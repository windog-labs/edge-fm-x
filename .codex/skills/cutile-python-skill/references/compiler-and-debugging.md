# Compiler And Debugging

## Read These First

与编译、导出、ABI、异常、IR 相关的问题，优先读这些文件：

```text
third_party/cutile-python/docs/source/compilation.rst
third_party/cutile-python/docs/source/debugging.rst
third_party/cutile-python/src/cuda/tile/_compile.py
third_party/cutile-python/src/cuda/tile/_context.py
third_party/cutile-python/src/cuda/tile/_cache.py
third_party/cutile-python/src/cuda/tile/compilation/_export.py
third_party/cutile-python/src/cuda/tile/compilation/_signature.py
third_party/cutile-python/src/cuda/tile/compilation/_name_mangling.py
third_party/cutile-python/test/test_export_compat.py
third_party/cutile-python/test/test_cache.py
```

如果要进一步查编译器内部，再往下看：

```text
third_party/cutile-python/src/cuda/tile/_ir/ir.py
third_party/cutile-python/src/cuda/tile/_ir/ops.py
third_party/cutile-python/src/cuda/tile/_ir/load_store_impl.py
third_party/cutile-python/src/cuda/tile/_passes/ast2hir.py
third_party/cutile-python/src/cuda/tile/_passes/hir2ir.py
third_party/cutile-python/src/cuda/tile/_passes/token_order.py
third_party/cutile-python/src/cuda/tile/_passes/check_dtype_support.py
```

## Export / ABI Workflow

用户提到下面这些词时，优先走 export/ABI 线路：

- `export_kernel`
- `KernelSignature`
- `CallingConvention`
- `cutile_python_v1`
- cubin / bytecode
- name mangling
- driver API / `cuLaunchKernel`

最先看的三处：

1. `docs/source/compilation.rst`
2. `src/cuda/tile/compilation/_signature.py`
3. `test/test_export_compat.py`

注意：

- `KernelSignature.from_kernel_args(...)` 方便，但可能把样例参数里的对齐/shape 假设固化进去。
- 真要稳定导出，优先显式构造 `KernelSignature` 和约束对象。
- 讲 ABI 时不要只凭 README；要对照 `cutile_python_v1` 的文档和测试。

## Debugging Knobs

这些环境变量是排障首选：

- `CUDA_TILE_ENABLE_CRASH_DUMP=1`
- `CUDA_TILE_COMPILER_TIMEOUT_SEC=<sec>`
- `CUDA_TILE_LOGS=CUTILEIR`
- `CUDA_TILE_TEMP_DIR=<dir>`
- `CUDA_TILE_CACHE_DIR=<dir>`
- `CUDA_TILE_CACHE_SIZE=<bytes>`

用途：

- 编译器执行失败或超时：先开 `CUDA_TILE_ENABLE_CRASH_DUMP=1`
- 想看 cuTile IR：开 `CUDA_TILE_LOGS=CUTILEIR`
- 怀疑 cache 干扰：看 `CUDA_TILE_CACHE_DIR`、`test/test_cache.py`
- 怀疑 `tileiras` 卡住：调 `CUDA_TILE_COMPILER_TIMEOUT_SEC`

## Search Patterns

### Find compiler entry points

```bash
rg -n "export_kernel|KernelSignature|CallingConvention|from_kernel_args|mangle|demangle"   third_party/cutile-python/src/cuda/tile/compilation   third_party/cutile-python/test   third_party/cutile-python/docs/source/compilation.rst
```

### Find IR and lowering stages

```bash
rg -n "IRContext|ast2hir|hir2ir|Bytecode|token_order|code_motion|check_dtype_support"   third_party/cutile-python/src/cuda/tile
```

### Find load/store lowering and optimization hints

```bash
rg -n "allow_tma|latency|load_store_hints|tile_load|tile_store"   third_party/cutile-python/src/cuda/tile   third_party/cutile-python/test   third_party/cutile-python/docs/source/performance.rst
```

## Experimental Autotuning

用户提到 autotune 时，优先看：

```text
third_party/cutile-python/experimental/src/cuda/tile_experimental/__init__.py
third_party/cutile-python/experimental/src/cuda/tile_experimental/_autotuner.py
third_party/cutile-python/samples/AttentionFMHA.py
third_party/cutile-python/test/test_autotuner.py
```

关键信号词：

- `autotune_launch`
- `clear_autotune_cache`
- `force_retune`
- `compiler_time_limit_sec`
- search space / tuning record / cache hit

## Escalation Heuristics

- 问题已经上升到驱动、`tileiras`、`ptxas`、Nsight 或硬件特性时，联动 `cuda-skill`
- 问题在 docs 层面解释不清，但 tests 能复现，就以 tests + 源码当前行为为准
- 如果用户问“为什么和 CUDA C++ / CUTLASS 不一样”，再补查 `cutlass-skill`
