# CuTe C++ Reading Guide

这份附录聚焦 CuTe C++：`cute::Layout`、`cute::Tensor`、layout algebra、predication、TMA、MMA atom、`TiledMma`、`TiledCopy`。

## Canonical Reading Order

按这个顺序读，能最快建立 CuTe 心智模型：

1. `media/docs/cpp/cute/00_quickstart.md`
2. `media/docs/cpp/cute/01_layout.md`
3. `media/docs/cpp/cute/02_layout_algebra.md`
4. `media/docs/cpp/cute/03_tensor.md`
5. `media/docs/cpp/cute/04_algorithms.md`
6. `media/docs/cpp/cute/0t_mma_atom.md`
7. `media/docs/cpp/cute/0x_gemm_tutorial.md`
8. `media/docs/cpp/cute/0y_predication.md`
9. `media/docs/cpp/cute/0z_tma_tensors.md`

读法建议：

- `00-04` 建立 vocabulary
- `0t` 理解 atom / tiled 抽象
- `0x` 把概念串成 GEMM
- `0y` 解决边界和 predication
- `0z` 进入 Hopper/Blackwell TMA 世界

## Header Map

### 核心对象

```text
include/cute/layout.hpp
include/cute/tensor.hpp
include/cute/stride.hpp
include/cute/swizzle.hpp
```

### 算法

```text
include/cute/algorithm/
```

这里主要找：

- `copy`
- `copy_if`
- `gemm`
- tuple/tensor 上的基础算法

### 架构相关 PTX 包装

```text
include/cute/arch/copy_sm90.hpp
include/cute/arch/copy_sm90_tma.hpp
include/cute/arch/copy_sm100.hpp
include/cute/arch/copy_sm100_tma.hpp
include/cute/arch/mma_sm90.hpp
include/cute/arch/mma_sm90_gmma.hpp
include/cute/arch/mma_sm100.hpp
```

### Atom / Tiled 抽象

```text
include/cute/atom/copy_atom.hpp
include/cute/atom/mma_atom.hpp
include/cute/atom/mma_traits_sm90_gmma.hpp
```

## Tutorial -> Header -> Test Mapping

| 学习目标 | 文档 | 教程 / 示例 | 对应 header | 对应测试 |
|---|---|---|---|---|
| 理解 `Layout` | `01_layout.md` | `examples/cute/tutorial/sgemm_1.cu` | `layout.hpp`、`stride.hpp` | `test/unit/cute/core/complement.cpp`、`composition.cpp`、`logical_divide.cpp` |
| 理解 layout algebra | `02_layout_algebra.md` | `examples/cute/tutorial/sgemm_2.cu` | `layout.hpp`、`swizzle.hpp` | `test/unit/cute/core/coalesce.cpp`、`logical_product.cpp`、`swizzle_layout.cpp` |
| 理解 `Tensor` | `03_tensor.md` | `examples/cute/tutorial/sgemm_sm70.cu` | `tensor.hpp` | `test/unit/cute/core/tensor_algs.cpp`、`pointer.cpp` |
| 理解 `copy` / `gemm` 算法 | `04_algorithms.md` | `examples/cute/tutorial/tiled_copy.cu`、`tiled_copy_if.cu` | `algorithm/` | `test/unit/cute/ampere/tiled_cp_async.cu` |
| 理解 MMA atom | `0t_mma_atom.md` | `examples/cute/tutorial/sgemm_sm80.cu` | `atom/mma_atom.hpp`、`arch/mma_sm90_gmma.hpp` | `test/unit/cute/ampere/cooperative_gemm.cu`、`turing/cooperative_gemm.cu`、`volta/cooperative_gemm.cu` |
| 理解 Hopper WGMMA | `0x_gemm_tutorial.md`、`0z_tma_tensors.md` | `examples/cute/tutorial/hopper/wgmma_sm90.cu`、`wgmma_tma_sm90.cu` | `arch/mma_sm90_gmma.hpp`、`arch/copy_sm90_tma.hpp`、`atom/mma_traits_sm90_gmma.hpp` | `test/unit/cute/hopper/cooperative_gemm.cu`、`tma_load.cu`、`tma_store.cu` |
| 理解 Blackwell MMA / TMA | `0t_mma_atom.md`、`0z_tma_tensors.md` | `examples/cute/tutorial/blackwell/01_mma_sm100.cu` 到 `05_mma_tma_epi_sm100.cu` | `arch/mma_sm100.hpp`、`arch/copy_sm100_tma.hpp` | 优先用 Blackwell example 本身，再补 `test/unit/pipeline/` 与 `test/unit/gemm/device/sm100_*` |
| 理解 predication | `0y_predication.md` | `tiled_copy_if.cu` 与 GEMM tutorial | `tensor.hpp`、`algorithm/` | `test/unit/cute/core/compare.cpp`、`domain_distribute.cpp` |

## Practical Entry Points

### 只想先看最小 GEMM

顺序：

```text
sgemm_1.cu
sgemm_2.cu
sgemm_sm70.cu
sgemm_sm80.cu
```

### 只想先看 copy / tiling

顺序：

```text
tiled_copy.cu
tiled_copy_if.cu
```

### 只想看 Hopper

顺序：

```text
media/docs/cpp/cute/0t_mma_atom.md
media/docs/cpp/cute/0z_tma_tensors.md
examples/cute/tutorial/hopper/wgmma_sm90.cu
examples/cute/tutorial/hopper/wgmma_tma_sm90.cu
test/unit/cute/hopper/
```

### 只想看 Blackwell

顺序：

```text
media/docs/cpp/cute/0t_mma_atom.md
media/docs/cpp/cute/0z_tma_tensors.md
examples/cute/tutorial/blackwell/01_mma_sm100.cu
examples/cute/tutorial/blackwell/02_mma_tma_sm100.cu
examples/cute/tutorial/blackwell/03_mma_tma_multicast_sm100.cu
examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu
examples/cute/tutorial/blackwell/05_mma_tma_epi_sm100.cu
```

## Quick Search Recipes

### 查 layout/tensor 相关 API

```bash
rg -n "make_layout|composition|complement|logical_divide|zipped_divide|local_tile" \
  "$CUTLASS_REPO/media/docs/cpp/cute" \
  "$CUTLASS_REPO/include/cute"
```

### 查 TiledMma / TiledCopy / MMA atom

```bash
rg -n "TiledMma|TiledCopy|Mma_Atom|Copy_Atom" \
  "$CUTLASS_REPO/include/cute" \
  "$CUTLASS_REPO/examples/cute/tutorial"
```

### 查 Hopper GMMA / TMA

```bash
find "$CUTLASS_REPO/include/cute/arch" -type f | grep -E 'sm90|gmma|tma'
find "$CUTLASS_REPO/test/unit/cute/hopper" -type f | sort
```

### 查 Blackwell MMA / TMA

```bash
find "$CUTLASS_REPO/include/cute/arch" -type f | grep -E 'sm100'
find "$CUTLASS_REPO/examples/cute/tutorial/blackwell" -type f | sort
```

## How CuTe Relates To CUTLASS C++

默认做法：

1. 先用 CuTe 文档理解 `Layout` / `Tensor` / `Atom`
2. 再去 CUTLASS `gemm_api_3x.md` 看这些抽象怎样被嵌进 `CollectiveMainloop`
3. 再看具体 example

如果你在 CUTLASS header 里看不懂线程布局、tile 切分、copy path、mma path，先退回 CuTe 文档，不要硬啃 `include/cutlass/gemm/collective/*`。

## When To Jump To cuda-skill

出现这些问题时就不要只停留在 CuTe 层：

- `mma.sync` / `wgmma` / `tcgen05` 到底是什么指令
- TMA coordinate / descriptor 为什么这样排布
- `mbarrier` / async pipeline 为何这样同步
- 共享内存对齐 / bank conflict / cluster 行为怎样影响 kernel

这时请去 `cuda-skill` 查：

- `ptx-simple/ptx-isa-tensor-cores.md`
- `ptx-simple/ptx-isa-async-copy.md`
- `ptx-simple/ptx-isa-barriers.md`
- `ptx-simple/ptx-isa-sm100-blackwell.md`
- CUDA guide 里的 `async barriers`、cluster/programming model 章节
