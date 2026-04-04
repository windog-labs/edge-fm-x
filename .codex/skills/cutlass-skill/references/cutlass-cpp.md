# CUTLASS C++ Reading Guide

这份附录面向 CUTLASS C++ 开发：GEMM、grouped GEMM、sparse GEMM、blockscaled GEMM、FMHA、GQA、SSD、pipeline、cluster launch control。

## Canonical Reading Order

先按这条主线读，不要一开始直接钻 `include/`。

1. `media/docs/cpp/quickstart.md`
2. `media/docs/cpp/code_organization.md`
3. `media/docs/cpp/gemm_api_3x.md`
4. `media/docs/cpp/pipeline.md`
5. `media/docs/cpp/profiler.md`
6. `media/docs/cpp/blackwell_functionality.md`
7. `media/docs/cpp/blackwell_cluster_launch_control.md`

需要补背景时，再看：

- `media/docs/cpp/efficient_gemm.md`
- `media/docs/cpp/grouped_scheduler.md`
- `media/docs/cpp/cutlass_3x_design.md`
- `media/docs/cpp/cutlass_3x_backwards_compatibility.md`
- `media/docs/cpp/dependent_kernel_launch.md`

## Core API Mental Model

CUTLASS 3.x / 4.x 的 GEMM 主线可以先记成 4 层：

1. `cutlass::gemm::collective::CollectiveBuilder`
2. `cutlass::epilogue::collective::CollectiveBuilder` 或 `DefaultEpilogue`
3. `cutlass::gemm::kernel::GemmUniversal`
4. `cutlass::gemm::device::GemmUniversalAdapter`

最常用的 header 入口：

```text
include/cutlass/gemm/collective/collective_builder.hpp
include/cutlass/epilogue/collective/collective_builder.hpp
include/cutlass/gemm/kernel/
include/cutlass/gemm/device/
include/cutlass/pipeline/
```

## Key Header Families

### Hopper / SM90

重点查看：

```text
include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp
include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp
include/cutlass/gemm/collective/sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp
include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp
include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_pingpong.hpp
include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp
include/cutlass/pipeline/
```

### Blackwell / SM100+

重点查看：

```text
include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp
include/cutlass/gemm/collective/sm100_mma_warpspecialized_blockwise_scaling.hpp
include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp
include/cutlass/gemm/collective/sm100_sparse_mma_warpspecialized.hpp
include/cutlass/gemm/collective/sm100_blockscaled_sparse_mma_warpspecialized.hpp
include/cutlass/gemm/kernel/sm100_gemm_tma_warpspecialized.hpp
include/cutlass/gemm/kernel/sm100_sparse_gemm_tma_warpspecialized.hpp
include/cutlass/epilogue/collective/sm100_epilogue_tma_warpspecialized.hpp
include/cutlass/pipeline/sm100_pipeline.hpp
```

## Example -> Header -> Test Mapping

下面这张表是从本地仓库结构出发的实战入口。

| Example | 先看什么 | 再看哪些 header | 再看哪些 test |
|---|---|---|---|
| `49_hopper_gemm_with_collective_builder` | `gemm_api_3x.md`、`pipeline.md` | `gemm/collective/collective_builder.hpp`、`epilogue/collective/collective_builder.hpp`、`gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` | `test/unit/gemm/device/sm90_*`、`test/unit/pipeline/` |
| `54_hopper_fp8_warp_specialized_gemm` | `gemm_api_3x.md`、`efficient_gemm.md` | `sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp`、`sm90_epilogue_tma_warpspecialized.hpp` | `test/unit/gemm/device/sm90_gemm_f8_*` |
| `57_hopper_grouped_gemm` | `grouped_scheduler.md`、`profiler.md` | `collective_builder.hpp`、`gemm/kernel/` 下的 array / grouped 相关实现 | `test/unit/gemm/device/gemm_grouped_sm80.cu`、同目录下 grouped/cluster 相关用法 |
| `62_hopper_sparse_gemm` | `gemm_api_3x.md` | `sm90_sparse_mma_tma_gmma_ss_warpspecialized.hpp`、`sm90_gemm_tma_warpspecialized*.hpp` | `test/unit/gemm/device/*sparse*sm80*.cu`、`sm90_*` 稀疏相关测试 |
| `67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling` | `gemm_api_3x.md`、`profiler.md` | `sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp` | `test/unit/gemm/device/sm90_gemm_f8_*` 与 blockscaled 相关 profiler 用法 |
| `70_blackwell_gemm` | `blackwell_functionality.md` | `sm100_mma_warpspecialized.hpp`、`sm100_gemm_tma_warpspecialized.hpp` | `test/unit/gemm/device/sm100_tensorop_gemm/` |
| `71_blackwell_gemm_with_collective_builder` | `blackwell_functionality.md`、`gemm_api_3x.md` | `collective_builder.hpp`、`sm100_gemm_tma_warpspecialized.hpp`、`sm100_epilogue_tma_warpspecialized.hpp` | `test/unit/gemm/device/sm100_tensorop_gemm/`、`test/unit/pipeline/` |
| `72_blackwell_narrow_precision_gemm` | `blackwell_functionality.md` | `sm100_mma_warpspecialized_blockwise_scaling.hpp`、`sm100_blockscaled_mma_warpspecialized.hpp` | `test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/` |
| `74_blackwell_gemm_streamk` | `blackwell_functionality.md`、`profiler.md` | `sm100_gemm_tma_warpspecialized.hpp`、调度与 kernel 层实现 | `test/unit/gemm/device/sm100_*streamk*`、blockscaled sparse stream-k 测试 |
| `77_blackwell_fmha` | example 目录内 `collective/`、`kernel/`、`device/` | `gemm/collective/`、`epilogue/collective/`、`pipeline/sm100_pipeline.hpp` | `test/unit/pipeline/`、相关 gemm/device 测试 |
| `81_blackwell_gemm_blockwise` | `blackwell_functionality.md`、`profiler.md` | `sm100_mma_warpspecialized_blockwise_scaling.hpp`、`sm100_blockscaled_mma_warpspecialized.hpp` | `test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/` |
| `83_blackwell_sparse_gemm` | `blackwell_functionality.md` | `sm100_sparse_mma_warpspecialized.hpp`、`sm100_sparse_gemm_tma_warpspecialized.hpp` | `test/unit/gemm/device/sm100_sparse_tensorop_gemm/` |
| `92_blackwell_moe_gemm` | example 目录本身、`profiler.md` | `gemm/collective/`、`kernel/`、`epilogue/` | `test/unit/gemm/device/sm100_tensorop_gemm/`、grouped/blockscaled 相关测试 |
| `93_blackwell_low_latency_gqa` | example 目录本身、`blackwell_cluster_launch_control.md` | `pipeline/sm100_pipeline.hpp`、`sm100_gemm_tma_warpspecialized.hpp` | `test/unit/pipeline/pipeline_cluster_launch_control_async_warp_specialized_blackwell.cu`、`test/unit/cluster_launch/cluster_launch.cu` |
| `94_ada_fp8_blockwise` | example 目录本身、`profiler.md` | `gemm/collective/`、Ada/FP8 相关实现 | `test/unit/gemm/device/gemm_f8*sm89.cu` |
| `111_hopper_ssd` | example 目录中的 `collective/`、`kernel/`、`device/`、`reference/` | 先从 example 反查 `include/cutlass/gemm/collective/` 与 `kernel/` | `test/unit/pipeline/` 与相关 gemm/device 测试 |
| `112_blackwell_ssd` | example 目录中的 `collective/`、`kernel/`、`device/`、`reference/` | 先从 example 反查 `sm100_*` header 与 `pipeline/sm100_pipeline.hpp` | `test/unit/pipeline/`、`test/unit/cluster_launch/`、`sm100_*` 测试 |

## How To Use The Tests

默认先用测试目录找“支持矩阵”和“合法组合”。

### Hopper

优先看：

```text
test/unit/gemm/device/sm90_*
test/unit/pipeline/
```

### Blackwell

优先看：

```text
test/unit/gemm/device/sm100_tensorop_gemm/
test/unit/gemm/device/sm100_sparse_tensorop_gemm/
test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/
test/unit/gemm/device/sm100_blockscaled_sparse_tensorop_gemm/
test/unit/gemm/device/sm120_tensorop_gemm/
test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/
test/unit/gemm/device/sm120_blockscaled_sparse_tensorop_gemm/
test/unit/pipeline/
test/unit/cluster_launch/
```

这些目录尤其适合回答三类问题：

- 某种 data type / layout / alignment 是否受支持
- 某个 `CollectiveBuilder` 配法是否已有先例
- cluster / CLC / blockscaled / sparse 的最小正确组合是什么

## Quick Search Recipes

### 查 `CollectiveBuilder`

```bash
rg -n "CollectiveBuilder" \
  "$CUTLASS_REPO/media/docs/cpp" \
  "$CUTLASS_REPO/examples" \
  "$CUTLASS_REPO/include/cutlass"
```

### 查 Hopper warpspecialized GEMM

```bash
find "$CUTLASS_REPO/include/cutlass/gemm/collective" -type f | grep 'sm90.*warpspecialized'
find "$CUTLASS_REPO/include/cutlass/gemm/kernel" -type f | grep 'sm90.*warpspecialized'
```

### 查 Blackwell blockscaled / sparse / stream-k

```bash
find "$CUTLASS_REPO/include/cutlass/gemm/collective" -type f | grep 'sm100'
find "$CUTLASS_REPO/test/unit/gemm/device" -type f | grep -E 'sm100_blockscaled|sm120_blockscaled|streamk|preferred_cluster'
```

### 查 pipeline / cluster launch control

```bash
rg -n "Pipeline|Cluster Launch Control|PipelineCLCFetchAsync" \
  "$CUTLASS_REPO/media/docs/cpp" \
  "$CUTLASS_REPO/include/cutlass/pipeline" \
  "$CUTLASS_REPO/test/unit/pipeline"
```

## When To Jump To cuda-skill

出现以下任一关键词时，立刻联动 `cuda-skill`：

- `wgmma`
- `tcgen05`
- TMA descriptor
- `mbarrier`
- cluster / CLC
- PTX / SASS
- Nsight Compute / Nsight Systems

跳过去之后，优先查：

- PTX tensor core 文档
- PTX async copy / barrier 文档
- CUDA programming guide 的 cluster / async barriers
- Nsight Compute / Nsight Systems 指南
