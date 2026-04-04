---
name: cutlass-skill
description: "Develop, debug, and optimize CUTLASS C++, CuTe C++, and CuTe DSL kernels using the local third_party/cutlass repository in this workspace. Use when the user mentions CUTLASS, CuTe, CuTe DSL, cute::Layout, cute::Tensor, TiledMMA, TiledCopy, CollectiveBuilder, CollectiveMainloop, CollectiveEpilogue, GEMM kernel, grouped GEMM, sparse GEMM, FMHA, StreamK, warp specialization, TMA, WGMMA, tcgen05, blockscaled GEMM, or asks about writing or understanding high-performance CUDA kernels with CUTLASS/CuTe."
---

# CUTLASS Development In This Repo

这个 skill 以当前 workspace 里的 `third_party/cutlass/` 为唯一默认事实源。

不要默认使用 `~/.codex/skills/.../repos/cutlass/`。只有当当前仓库没有 `third_party/cutlass/` 时，才回退到 home 目录下安装的 CUTLASS skill 副本。

同时，这个 skill 不是独立世界。凡是涉及底层 CUDA/PTX/WGMMA/`tcgen05`/TMA/`mbarrier`/cluster/Nsight/compute-sanitizer，必须联动 `cuda-skill`，优先查本地 CUDA 参考资料，而不是凭记忆回答。

## Source Of Truth

优先级固定如下：

1. 当前 workspace 的 `third_party/cutlass/`
2. 当前 workspace 的 `.codex/skills/cuda-skill/references/`
3. home 目录下安装的 `cutlass-skill` / `cuda-skill`
4. 你当前对 CUTLASS/CUDA 的通识记忆

如果本地文档、example、header 三者不一致，优先级如下：

1. `media/docs/*`
2. `examples/*`
3. `include/*`
4. 猜测

## Resolve Paths

先解析运行时路径，再开始查文档。

```bash
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

CUTLASS_REPO="$REPO_ROOT/third_party/cutlass"
if [ ! -d "$CUTLASS_REPO" ]; then
  CUTLASS_REPO="$(find "$HOME/.codex/skills" "$HOME/.claude/skills" "$HOME/.cursor/skills" \
    -path '*/cutlass-skill/repos/cutlass' -type d 2>/dev/null | head -1)"
fi

CUDA_REFS="$REPO_ROOT/.codex/skills/cuda-skill/references"
if [ ! -d "$CUDA_REFS" ]; then
  CUDA_REFS="$(find "$HOME/.codex/skills" "$HOME/.claude/skills" "$HOME/.cursor/skills" \
    -path '*/cuda-skill/references' -type d 2>/dev/null | head -1)"
fi

echo "CUTLASS_REPO=$CUTLASS_REPO"
echo "CUDA_REFS=$CUDA_REFS"
```

硬规则：

- 只要 `third_party/cutlass/` 存在，就以它为准。
- 只要 `.codex/skills/cuda-skill/references/` 存在，就以它为准。
- 不要再把外部 clone 当成默认路径。

## Repo Map

当前仓库里最重要的入口如下。

### CUTLASS C++ / CuTe C++

```text
$CUTLASS_REPO/media/docs/cpp/
├── quickstart.md
├── code_organization.md
├── gemm_api_3x.md
├── pipeline.md
├── profiler.md
├── blackwell_functionality.md
├── blackwell_cluster_launch_control.md
└── cute/
    ├── 00_quickstart.md
    ├── 01_layout.md
    ├── 02_layout_algebra.md
    ├── 03_tensor.md
    ├── 04_algorithms.md
    ├── 0t_mma_atom.md
    ├── 0x_gemm_tutorial.md
    ├── 0y_predication.md
    └── 0z_tma_tensors.md
```

```text
$CUTLASS_REPO/include/cutlass/
├── gemm/
│   ├── collective/
│   ├── kernel/
│   └── device/
├── epilogue/
├── pipeline/
├── conv/
└── arch/

$CUTLASS_REPO/include/cute/
├── layout.hpp
├── tensor.hpp
├── stride.hpp
├── swizzle.hpp
├── algorithm/
├── arch/
├── atom/
└── util/
```

### Examples

```text
$CUTLASS_REPO/examples/
├── 48_hopper_warp_specialized_gemm
├── 49_hopper_gemm_with_collective_builder
├── 54_hopper_fp8_warp_specialized_gemm
├── 57_hopper_grouped_gemm
├── 62_hopper_sparse_gemm
├── 67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling
├── 70_blackwell_gemm
├── 71_blackwell_gemm_with_collective_builder
├── 72_blackwell_narrow_precision_gemm
├── 74_blackwell_gemm_streamk
├── 77_blackwell_fmha
├── 81_blackwell_gemm_blockwise
├── 83_blackwell_sparse_gemm
├── 92_blackwell_moe_gemm
├── 93_blackwell_low_latency_gqa
├── 94_ada_fp8_blockwise
├── 111_hopper_ssd
├── 112_blackwell_ssd
└── cute/tutorial/
```

### CuTe DSL

```text
$CUTLASS_REPO/media/docs/pythonDSL/
├── overview.rst
├── quick_start.rst
├── cute_dsl.rst
├── cute_dsl_api.rst
├── limitations.rst
├── faqs.rst
└── cute_dsl_general/
    ├── debugging.rst
    ├── autotuning_gemm.rst
    ├── dsl_ahead_of_time_compilation.rst
    ├── dsl_code_generation.rst
    ├── dsl_dynamic_layout.rst
    ├── dsl_jit_arg_generation.rst
    ├── dsl_jit_caching.rst
    └── framework_integration.rst
```

```text
$CUTLASS_REPO/python/CuTeDSL/cutlass/
├── base_dsl/
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
└── jax/
```

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
└── notebooks/
```

注意：

- 当前仓库里没有 `examples/python/CuTeDSL/notebooks-zh/`，不要再引用它。
- 当前仓库里确实有 `blackwell_geforce/`、`experimental/`、`jax/`、`cute/export/`、`cute/ffi/`、`cute/tvm_ffi/`。

### Profiling / Tests

```text
$CUTLASS_REPO/tools/profiler/
$CUTLASS_REPO/tools/library/
$CUTLASS_REPO/tools/util/

$CUTLASS_REPO/test/unit/
├── cute/
├── gemm/
├── pipeline/
├── cluster_launch/
└── epilogue/

$CUTLASS_REPO/test/examples/CuTeDSL/
├── hopper/
└── sm_100a/
```

## What To Read First

不要一上来就翻 `include/`。先按任务选线路。

- 改 CUTLASS C++ GEMM / grouped / sparse / blockscaled / FMHA / GQA
  看 `references/cutlass-cpp.md`
- 想先理解 `cute::Layout` / `cute::Tensor` / `TiledMma` / `TiledCopy`
  看 `references/cute-cpp.md`
- 写或改 CuTe DSL kernel
  看 `references/cute-dsl.md`
- 做 correctness / 性能排障
  看 `references/debugging-and-profiling.md`

## Search Strategy

只加载必要片段。优先 `rg`、`find`、`sed -n`，不要整文件拖进上下文。

### 搜 CUTLASS C++ 文档

```bash
rg -n "CollectiveBuilder|GemmUniversalAdapter|Pipeline|Profiler|Blackwell|StreamK" \
  "$CUTLASS_REPO/media/docs/cpp"

sed -n '1,220p' "$CUTLASS_REPO/media/docs/cpp/quickstart.md"
sed -n '1,220p' "$CUTLASS_REPO/media/docs/cpp/gemm_api_3x.md"
sed -n '1,220p' "$CUTLASS_REPO/media/docs/cpp/pipeline.md"
```

### 搜 CuTe C++ 文档与 header

```bash
sed -n '1,200p' "$CUTLASS_REPO/media/docs/cpp/cute/index.rst"
rg -n "Layout|Tensor|TiledMma|TiledCopy|GMMA|TMA" \
  "$CUTLASS_REPO/media/docs/cpp/cute" \
  "$CUTLASS_REPO/include/cute"

find "$CUTLASS_REPO/examples/cute/tutorial" -maxdepth 2 -type f | sort
```

### 搜 CuTe DSL 文档与源码

```bash
rg -n "CUTE_DSL_|debug|autotuning|AOT|TVM|JAX|cache|lineinfo" \
  "$CUTLASS_REPO/media/docs/pythonDSL" \
  "$CUTLASS_REPO/python/CuTeDSL"

find "$CUTLASS_REPO/examples/python/CuTeDSL" -maxdepth 2 -type f | sort
find "$CUTLASS_REPO/test/examples/CuTeDSL" -maxdepth 3 -type f | sort
```

### 搜 example 与 test

```bash
find "$CUTLASS_REPO/examples" -maxdepth 2 -type d | sort
find "$CUTLASS_REPO/test/unit" -maxdepth 3 -type d | sort

rg -n "pipeline_cluster_launch_control_async_warp_specialized_blackwell|CollectiveBuilder|tcgen05|wgmma" \
  "$CUTLASS_REPO/test/unit" \
  "$CUTLASS_REPO/include"
```

### 查本地支持的架构

不要背 `CUTLASS_NVCC_ARCHS`。以本地 `CMakeLists.txt` 为准。

```bash
rg -n "CUTLASS_NVCC_ARCHS_SUPPORTED|CUTLASS_NVCC_ARCHS_ENABLED" \
  "$CUTLASS_REPO/CMakeLists.txt"
```

## Development Playbooks

下面五条 playbook 是默认工作流。每次都先走文档，再落到 example，再落到 header/test/profiler，最后才跳到底层 CUDA 资料。

### 1. 改现有 CUTLASS GEMM

1. 先看 `media/docs/cpp/quickstart.md`、`gemm_api_3x.md`、相关架构文档。
2. 找最接近的 example。
   Hopper 优先看 `49`、`54`、`57`、`62`、`67`。
   Blackwell 优先看 `70`、`71`、`72`、`74`、`81`、`83`、`92`、`93`、`111`、`112`。
3. 再看对应 header 家族。
   `include/cutlass/gemm/collective/`
   `include/cutlass/gemm/kernel/`
   `include/cutlass/epilogue/collective/`
   `include/cutlass/pipeline/`
4. 用 `test/unit/gemm/device/`、`test/unit/pipeline/`、`test/unit/cluster_launch/` 找回归样例。
5. 如果涉及 `wgmma` / `tcgen05` / TMA / barrier / cluster，跳到 `cuda-skill` 查底层 CUDA/PTX 文档。

### 2. 从 CuTe tutorial 起步写内核

1. 先按 `cute/index.rst` 的顺序读 `00` 到 `0z`。
2. 按 `examples/cute/tutorial/` 的梯度抄最小样例。
   `sgemm_1.cu` -> `sgemm_2.cu` -> `sgemm_sm70.cu` -> `sgemm_sm80.cu`
   `tiled_copy.cu` / `tiled_copy_if.cu`
   Hopper: `hopper/wgmma_sm90.cu`、`hopper/wgmma_tma_sm90.cu`
   Blackwell: `blackwell/01-05_mma*_sm100.cu`
3. 再去 `include/cute/` 看 `layout.hpp`、`tensor.hpp`、`algorithm/`、`arch/`、`atom/`。
4. 用 `test/unit/cute/core/`、`test/unit/cute/hopper/`、`test/unit/cute/ampere/` 做概念核对。
5. 如果你开始关心 PTX 指令级语义，立刻切到 `cuda-skill`。

### 3. 写 / 改 CuTe DSL kernel

1. 先看 `media/docs/pythonDSL/overview.rst`、`quick_start.rst`、`cute_dsl.rst`、`cute_dsl_api.rst`。
2. 按架构选 example。
   入门：`ampere/elementwise_add.py`、`ampere/sgemm.py`、`ampere/tensorop_gemm.py`
   Hopper：`hopper/dense_gemm.py`、`hopper/dense_gemm_persistent.py`、`hopper/grouped_gemm.py`
   Blackwell：`blackwell/dense_gemm_persistent.py`、`dense_blockscaled_gemm_persistent.py`、`grouped_gemm.py`、`fmha.py`
3. 再看 `python/CuTeDSL/cutlass/` 下的 `base_dsl/`、`cute/`、`pipeline/`、`utils/`、`cutlass_dsl/`。
4. 用 `test/examples/CuTeDSL/` 验证真实用法，尤其是 `hopper/` 与 `sm_100a/`。
5. 遇到生成 IR/PTX/CUBIN、`CUTE_DSL_ARCH`、JIT 缓存、AOT/TVM/FFI/JAX，再看 `references/cute-dsl.md`。
6. 若 DSL 暴露出 `wgmma` / `tcgen05` / TMA 行为差异，再跳 `cuda-skill`。

### 4. 定位 correctness 问题

1. 先确认你看的 API 和 example 是否同架构、同数据类型、同 layout。
2. 再找最近的 unit test / example test。
3. 用 profiler 或最小样例复现，不要先改模板。
4. 对 CuTe DSL，优先开 `CUTE_DSL_LINEINFO`、`CUTE_DSL_PRINT_IR`、`CUTE_DSL_KEEP_IR`、`CUTE_DSL_KEEP_PTX`。
5. 对 C++ kernel，优先看 `test/unit/gemm/device/`、`test/unit/pipeline/`、`test/unit/cute/*`。
6. 一旦怀疑 barrier、TMA descriptor、memory ordering、alignment，直接切 `cuda-skill` 查底层文档。

### 5. 定位性能问题

1. 先确认 example / build flag / arch flag 是否对。
2. 用 `tools/profiler/cutlass_profiler` 找 baseline。
3. 对 C++ kernel 看 `media/docs/cpp/profiler.md` 和相关架构 example。
4. 对 CuTe DSL 打开 line info，dump PTX/CUBIN，然后用 Nsight / `cuobjdump` / `nvdisasm` 看。
5. 对 TMA/WGMMA/`tcgen05`/cluster 相关 kernel，不要只盯 template 参数，要联动 `cuda-skill` 的 PTX 与 Nsight 资料。

## CUTLASS <-> CUDA Skill Crosswalk

| CUTLASS / CuTe 概念 | 先看 CUTLASS 本地资料 | 再跳 `cuda-skill` 看什么 |
|---|---|---|
| Hopper GMMA / WGMMA | `media/docs/cpp/cute/0t_mma_atom.md`、Hopper examples、`include/cute/arch/mma_sm90_gmma*.hpp` | `ptx-simple/ptx-isa-tensor-cores.md`、`ptx-docs/9-instruction-set/` 里的 WGMMA |
| Blackwell `tcgen05` | `media/docs/cpp/blackwell_functionality.md`、`examples/70-83/`、`include/cute/arch/mma_sm100*.hpp` | `ptx-simple/ptx-isa-sm100-blackwell.md`、PTX `tcgen05` 文档 |
| TMA / async copy | `media/docs/cpp/cute/0z_tma_tensors.md`、`pipeline.md`、`include/cute/arch/copy_sm90_tma.hpp`、`copy_sm100_tma.hpp` | `ptx-simple/ptx-isa-async-copy.md`、CUDA guide `async-barriers.md` |
| `mbarrier` / pipeline / software pipeline | `pipeline.md`、`include/cutlass/pipeline/`、`test/unit/pipeline/` | `ptx-simple/ptx-isa-barriers.md`、PTX barrier 文档 |
| Cluster / CLC / persistent scheduling | `blackwell_cluster_launch_control.md`、`include/cutlass/pipeline/sm100_pipeline.hpp`、`test/unit/pipeline/pipeline_cluster_launch_control_async_warp_specialized_blackwell.cu` | CUDA guide 里的 cluster / programming model、PTX cluster 相关资料 |
| Profiling | `media/docs/cpp/profiler.md`、`tools/profiler/` | `ncu-guide.md`、`nsys-guide.md`、`ncu-docs/`、`nsys-docs/` |
| Correctness / race / invalid memory | CUTLASS unit tests、CuTe DSL debugging docs | `debugging-tools.md`、compute-sanitizer、cuda-gdb |

## Reference Docs In This Skill

- `references/cutlass-cpp.md`
  CUTLASS C++ 主线、关键 example、header 家族、测试入口
- `references/cute-cpp.md`
  CuTe C++ 学习路径、layout/tensor/atom/TMA/WGMMA 映射
- `references/cute-dsl.md`
  CuTe DSL 文档入口、源码目录、example/test、环境变量
- `references/debugging-and-profiling.md`
  build / profiler / test / IR/PTX/CUBIN / Nsight / compute-sanitizer 闭环

## Default Rules

- 优先查本地 `media/docs/*`，不要先猜模板参数。
- 改 C++ kernel 时，先找最近的 example，再找 header，再找 unit test。
- 改 DSL kernel 时，先找最近的 example/test，再看 `python/CuTeDSL/cutlass/` 源码。
- 凡是底层 CUDA 行为问题，必须跳 `cuda-skill`。
- 不要再引用不存在的 `notebooks-zh`。
- 不要默认外部 clone。
- 不要在没有 profiling 的前提下讨论 CUTLASS 性能。
