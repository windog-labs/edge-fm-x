# CuTe DSL Reading Guide

这份附录聚焦 `python/CuTeDSL/` 与 `examples/python/CuTeDSL/` 的开发路线，包括安装、架构选择、常用源码目录、example/test 入口、调试环境变量、AOT/FFI/JAX。

## Canonical Reading Order

默认主线：

1. `media/docs/pythonDSL/overview.rst`
2. `media/docs/pythonDSL/quick_start.rst`
3. `media/docs/pythonDSL/cute_dsl.rst`
4. `media/docs/pythonDSL/cute_dsl_api.rst`
5. `media/docs/pythonDSL/limitations.rst`
6. `media/docs/pythonDSL/faqs.rst`

排障和工程化时再补：

- `media/docs/pythonDSL/cute_dsl_general/debugging.rst`
- `media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.rst`
- `media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.rst`
- `media/docs/pythonDSL/cute_dsl_general/framework_integration.rst`
- `media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.rst`
- `media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.rst`
- `media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.rst`

## Source Map

### Core source tree

```text
$CUTLASS_REPO/python/CuTeDSL/cutlass/
├── base_dsl/
│   ├── runtime/
│   ├── export/
│   └── utils/
├── cute/
│   ├── arch/
│   ├── experimental/
│   ├── export/
│   └── nvgpu/
│       ├── cpasync/
│       ├── tcgen05/
│       ├── warp/
│       └── warpgroup/
├── cutlass_dsl/
├── pipeline/
├── utils/
│   └── gemm/
└── jax/
```

### Example tree

```text
$CUTLASS_REPO/examples/python/CuTeDSL/
├── ampere/
├── hopper/
├── blackwell/
├── blackwell_geforce/
├── experimental/
├── distributed/
├── jax/
├── cute/
│   ├── export/
│   ├── ffi/
│   └── tvm_ffi/
├── helpers/
├── utils/
└── notebooks/
```

### Test tree

```text
$CUTLASS_REPO/test/examples/CuTeDSL/
├── hopper/test_grouped_gemm.py
└── sm_100a/
    ├── test_dense_blockscaled_gemm_persistent_prefetch.py
    ├── test_dense_gemm_persistent_prefetch.py
    ├── test_rmsnorm.py
    └── test_tutorial_gemm.py
```

## Recommended Entry Paths

### 入门最小路径

先看：

```text
overview.rst
quick_start.rst
examples/python/CuTeDSL/ampere/elementwise_add.py
examples/python/CuTeDSL/ampere/sgemm.py
examples/python/CuTeDSL/ampere/tensorop_gemm.py
```

### Hopper 路线

先看：

```text
examples/python/CuTeDSL/hopper/dense_gemm.py
examples/python/CuTeDSL/hopper/dense_gemm_persistent.py
examples/python/CuTeDSL/hopper/grouped_gemm.py
examples/python/CuTeDSL/hopper/fmha.py
test/examples/CuTeDSL/hopper/test_grouped_gemm.py
```

### Blackwell 路线

先看：

```text
examples/python/CuTeDSL/blackwell/dense_gemm.py
examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py
examples/python/CuTeDSL/blackwell/dense_gemm_persistent_prefetch.py
examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py
examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent_prefetch.py
examples/python/CuTeDSL/blackwell/grouped_gemm.py
examples/python/CuTeDSL/blackwell/fmha.py
examples/python/CuTeDSL/blackwell/rmsnorm.py
test/examples/CuTeDSL/sm_100a/test_dense_gemm_persistent_prefetch.py
test/examples/CuTeDSL/sm_100a/test_dense_blockscaled_gemm_persistent_prefetch.py
test/examples/CuTeDSL/sm_100a/test_tutorial_gemm.py
```

### 工程化路线

需要这些能力时，再看：

- AOT export
  `media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.rst`
  `examples/python/CuTeDSL/cute/export/`
- TVM FFI
  `media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.rst`
  `examples/python/CuTeDSL/cute/tvm_ffi/`
- JAX
  `examples/python/CuTeDSL/jax/`
  `python/CuTeDSL/cutlass/jax/`
- experimental APIs
  `python/CuTeDSL/cutlass/cute/experimental/`
  `examples/python/CuTeDSL/experimental/`

## Environment Variables

以下变量最常用，也是排障时最值得先开的。

| Env | 作用 | 默认建议 |
|---|---|---|
| `CUTE_DSL_ARCH` | 指定目标架构，DSL 运行时也会检查它是否匹配当前 GPU | Hopper 用 `sm_90a`，Blackwell 用 `sm_100` / `sm_100a`，按你的 GPU 调整 |
| `CUTE_DSL_LINEINFO` | 编译时带 `--lineinfo`，方便 profiler / debugger / PTX-SASS 对照 | 调试和 profiling 时打开 |
| `CUTE_DSL_PRINT_IR` | 把生成的 IR 打到日志 | IR 构造出问题时打开 |
| `CUTE_DSL_KEEP_IR` | 把 IR 落文件 | 需要离线检查时打开 |
| `CUTE_DSL_KEEP_PTX` | 把 PTX 落文件 | 看生成代码或对照 PTX 时打开 |
| `CUTE_DSL_KEEP_CUBIN` | 把 CUBIN 落文件 | 看 SASS / `cuobjdump` / `nvdisasm` 时打开 |
| `CUTE_DSL_DUMP_DIR` | 指定 IR/PTX/CUBIN dump 目录 | 建议显式设到临时目录 |
| `CUTE_DSL_CACHE_DIR` | 指定文件缓存目录 | 复现实验时显式设置更稳 |

同样很常用的还有：

| Env | 作用 |
|---|---|
| `CUTE_DSL_LOG_TO_CONSOLE` | 打开控制台日志 |
| `CUTE_DSL_LOG_TO_FILE` | 把日志写文件 |
| `CUTE_DSL_LOG_LEVEL` | 调整日志级别 |
| `CUTE_DSL_DRYRUN` | 只生成 IR，不执行 |
| `CUTE_DSL_NO_CACHE` | 关闭 JIT cache |
| `CUTE_DSL_JIT_TIME_PROFILING` | 分析 IR 生成 / 编译 / 执行时间 |

最常用的调试组合：

```bash
export CUTE_DSL_ARCH=sm_100
export CUTE_DSL_LINEINFO=1
export CUTE_DSL_PRINT_IR=1
export CUTE_DSL_KEEP_IR=1
export CUTE_DSL_KEEP_PTX=1
export CUTE_DSL_KEEP_CUBIN=1
export CUTE_DSL_DUMP_DIR=/tmp/cute_dsl_dump
```

## Quick Search Recipes

### 查调试与环境变量

```bash
rg -n "CUTE_DSL_|debug|lineinfo|IR|PTX|CUBIN|cache" \
  "$CUTLASS_REPO/media/docs/pythonDSL" \
  "$CUTLASS_REPO/python/CuTeDSL"
```

### 查 pipeline / cpasync / tcgen05

```bash
rg -n "pipeline|cpasync|tcgen05|warpgroup|TMA|GMMA" \
  "$CUTLASS_REPO/python/CuTeDSL/cutlass" \
  "$CUTLASS_REPO/examples/python/CuTeDSL"
```

### 查 AOT / export / TVM FFI / JAX

```bash
find "$CUTLASS_REPO/examples/python/CuTeDSL/cute" -maxdepth 2 -type d | sort
find "$CUTLASS_REPO/examples/python/CuTeDSL/jax" -maxdepth 2 -type f | sort
rg -n "AOT|export|tvm_ffi|jax" "$CUTLASS_REPO/media/docs/pythonDSL"
```

### 查 Blackwell persistent / blockscaled

```bash
find "$CUTLASS_REPO/examples/python/CuTeDSL/blackwell" -maxdepth 2 -type f | sort
find "$CUTLASS_REPO/test/examples/CuTeDSL/sm_100a" -maxdepth 2 -type f | sort
```

## Workflow Recommendations

### 想写新 DSL kernel

1. 先选最近的架构 example
2. 再看 `python/CuTeDSL/cutlass/cute/` 与 `pipeline/`
3. 需要 host integration 时再看 `framework_integration.rst`
4. 需要导出时再看 AOT / export / TVM FFI

### 想理解编译链

1. `dsl_code_generation.rst`
2. `python/CuTeDSL/cutlass/base_dsl/`
3. 打开 `CUTE_DSL_PRINT_IR=1`
4. 再看 `__mlir__` / `__ptx__` / `__cubin__`

### 想验证真实用法

优先用 `test/examples/CuTeDSL/`，因为这些测试比 notebook 更接近可回归的最小样本。

## Important Caveats

- 这个仓库里的 `quick_start.rst` 是安装事实源，按本地文档写环境要求。
- example 覆盖面比 API 文档更快反映当前用法，尤其是 Blackwell persistent / blockscaled。
- 如果 DSL 的行为已经落到 PTX/TMA/barrier/GMMA 层，别停在 Python 语法表面，直接切 `cuda-skill`。

## When To Jump To cuda-skill

以下场景必须联动：

- 想确认 `tcgen05` / `wgmma` 的真实 PTX 语义
- 怀疑 TMA / barrier / memory ordering 问题
- 想把 DSL dump 出来的 PTX/CUBIN 和底层 ISA 对齐
- 准备用 Nsight / compute-sanitizer 深挖 kernel

优先查：

- PTX tensor core / async copy / barrier / Blackwell 文档
- CUDA guide `async barriers`、cluster / programming model
- Nsight Compute / Nsight Systems 指南
