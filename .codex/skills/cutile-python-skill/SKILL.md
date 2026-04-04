---
name: cutile-python-skill
description: "Write, debug, compile, export, and autotune cuTile Python kernels using the local third_party/cutile-python repository in this workspace. Use when the user mentions cuTile Python, cuda.tile, cuda.tile_experimental, ct.kernel, ct.launch, ct.load, ct.store, ct.mma, ct.gather, ct.scatter, cuda.tile.compilation, export_kernel, KernelSignature, CallingConvention, tileiras, TileIR, or asks about authoring, understanding, testing, or optimizing Python GPU kernels in cuTile."
---

# cuTile Python Development

这个 skill 以当前 workspace 的 `third_party/cutile-python/` 为唯一默认事实源。

不要默认去查外部安装包、PyPI 副本或网上示例。只要当前仓库里有 `third_party/cutile-python/`，就优先以它的 docs、samples、tests 和源码为准。

同时，这个 skill 不是独立世界。凡是涉及 CUDA 驱动/Toolkit、`tileiras`、`ptxas`、Nsight、硬件能力、TMA 或更底层的 GPU 行为，必须联动 `cuda-skill`，优先查本地 CUDA 资料，而不是凭记忆回答。

## Workspace-First Path

先解析运行时路径，再开始查资料。

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CUTILE_REPO="$REPO_ROOT/third_party/cutile-python"

echo "CUTILE_REPO=$CUTILE_REPO"
```

硬规则：

- 只要 `third_party/cutile-python/` 存在，就以它为准。
- 不要默认把 `cuda.tile` 当成 Triton、CUTLASS 或裸 CUDA C++ 的薄封装；先看 cuTile 自己的 docs、samples、tests。
- 用户如果问的是 cuTile Python public API，优先看 `docs/source/` 和最接近的 sample/test。
- 用户如果问的是真实实现、编译路径或 bug 根因，再展开 `src/cuda/tile/` 与 `cext/`。

## Source Of Truth

按任务类型使用下面的优先级：

1. public API / 使用方式：`docs/source/*`
2. 最接近的 runnable 示例：`samples/*`、`samples/quickstart/*`
3. 行为边界与回归预期：`test/*`
4. 当前实现细节：`src/cuda/tile/*`、`experimental/src/cuda/tile_experimental/*`、`cext/*`
5. 通识记忆

如果文档、sample、test、源码看起来不一致：

1. 先以当前源码和 tests 的实际行为为准
2. 再说明文档可能滞后
3. 不要直接猜

## What To Read First

不要一上来就翻完整源码。先按任务选线路。

- 写第一个 kernel、改 `ct.load` / `ct.store` / `ct.mma` / 控制流 / tile 操作
  看 `references/kernel-authoring.md`
- 查 JIT、AOT 导出、ABI、name mangling、`KernelSignature`
  看 `references/compiler-and-debugging.md`
- 排查编译失败、cache、IR、`tileiras`、超时、autotune
  看 `references/compiler-and-debugging.md`
- 想知道怎么 build、装 editable、跑 samples / tests
  看 `references/validation-workflow.md`

## Repo Map

最常用入口如下。

### Docs

```text
$CUTILE_REPO/docs/source/
├── index.rst
├── quickstart.rst
├── execution.rst
├── data.rst
├── memory_model.rst
├── performance.rst
├── operations.rst
├── compilation.rst
├── debugging.rst
└── known_issues.rst
```

### Samples

```text
$CUTILE_REPO/samples/
├── quickstart/VectorAdd_quickstart.py
├── VectorAddition.py
├── MatMul.py
├── BatchMatMul.py
├── Transpose.py
├── LayerNorm.py
├── FFT.py
├── AttentionFMHA.py
├── MoE.py
└── AllGatherMatmul.py
```

### Runtime / Compiler

```text
$CUTILE_REPO/src/cuda/tile/
├── __init__.py
├── _execution.py
├── _compile.py
├── _context.py
├── _cache.py
├── _datatype.py
├── _stub.py
├── _memory_model.py
├── _ir/
├── _passes/
└── compilation/
    ├── _export.py
    ├── _signature.py
    └── _name_mangling.py
```

### Experimental

```text
$CUTILE_REPO/experimental/src/cuda/tile_experimental/
├── __init__.py
└── _autotuner.py
```

### Validation

```text
$CUTILE_REPO/test/
├── test_frontpage_example.py
├── test_load_store.py
├── test_mma.py
├── test_mma_scaled.py
├── test_copy.py
├── test_attention.py
├── test_export_compat.py
├── test_cache.py
├── test_cudagraph.py
└── kernels/
```

## Search Strategy

只加载必要片段。优先 `rg`、`find`、`sed -n`，不要整仓库全文拖进上下文。

### 查 public API / docs / samples

```bash
rg -n "ct\.kernel|ct\.launch|ct\.load|ct\.store|ct\.mma|static_eval|static_assert|static_iter"   "$CUTILE_REPO/docs/source" "$CUTILE_REPO/samples" "$CUTILE_REPO/test"

sed -n '1,220p' "$CUTILE_REPO/docs/source/quickstart.rst"
sed -n '1,220p' "$CUTILE_REPO/docs/source/execution.rst"
sed -n '1,220p' "$CUTILE_REPO/docs/source/compilation.rst"
```

### 查编译 / 导出 / ABI

```bash
rg -n "export_kernel|KernelSignature|CallingConvention|mangle|from_kernel_args"   "$CUTILE_REPO/src/cuda/tile/compilation" "$CUTILE_REPO/docs/source/compilation.rst" "$CUTILE_REPO/test"
```

### 查编译器内部 / IR / passes

```bash
rg -n "IRContext|TileCompiler|allow_tma|token_order|ast2hir|hir2ir|bytecode|cache"   "$CUTILE_REPO/src/cuda/tile"
```

### 查 autotune / experimental

```bash
rg -n "autotune_launch|clear_autotune_cache|force_retune|compiler_time_limit_sec"   "$CUTILE_REPO/experimental/src/cuda/tile_experimental" "$CUTILE_REPO/samples"
```

## Working Rules

- 先找离用户任务最近的 sample 或 test，再落到源码实现。
- 解释 public API 时，尽量给出 docs 路径或 sample 路径，不要只贴内部实现。
- 如果用户要的是“为什么这个 kernel 编不过/跑错了”，优先看异常类型、环境变量、`_compile.py`、`_context.py`、`_ir/*`、`_passes/*`。
- 如果用户要的是导出 cubin / bytecode 或 C ABI 对接，必须检查 `docs/source/compilation.rst`、`src/cuda/tile/compilation/*` 和 `test/test_export_compat.py`。
- 如果用户提到 autotune，先确认是否使用了 `cuda.tile_experimental`，再看 `_autotuner.py` 和 `samples/AttentionFMHA.py`。
- 如果用户问题明显需要 CUDA 硬件能力、TMA、Nsight 或驱动/toolchain 支撑，联动 `cuda-skill`。
