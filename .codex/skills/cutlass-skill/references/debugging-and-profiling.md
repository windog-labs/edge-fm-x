# CUTLASS Debugging And Profiling

这份附录把 CUTLASS C++、CuTe C++、CuTe DSL 的 correctness 与性能排障串成一个统一闭环。

## 0. Ground Rules

- 先确认架构、数据类型、layout、alignment、example 路径都对。
- 不要一边猜 API，一边测性能。
- 正确性问题先最小复现，再追文档 / example / test。
- 性能问题先 profiler，再讨论 template 和 tile 参数。

## 1. Choose `CUTLASS_NVCC_ARCHS`

不要手背支持矩阵。以本地 `CMakeLists.txt` 为准：

```bash
rg -n "CUTLASS_NVCC_ARCHS_SUPPORTED|CUTLASS_NVCC_ARCHS_ENABLED" \
  "$CUTLASS_REPO/CMakeLists.txt"
```

最常用的实际值：

- Ampere: `80`
- Hopper: `90a`
- Blackwell SM100: `100a`
- Blackwell GeForce / SM120: 看本地 `CMakeLists.txt` 支持项，再决定 `120` / `120a` / `120f`

如果你要做 `wgmma` / `tcgen05` / cluster / blockscaled，优先用明确的架构值，不要偷懒用宽泛 PTX 目标。

## 2. Minimal Build Recipes

### 只构建 profiler，尽量缩短编译时间

```bash
CUTLASS_BUILD="${CUTLASS_REPO}/build-codex-profiler"

cmake -S "$CUTLASS_REPO" -B "$CUTLASS_BUILD" \
  -DCUTLASS_NVCC_ARCHS=90a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_UNITY_BUILD_ENABLED=ON

cmake --build "$CUTLASS_BUILD" --target cutlass_profiler -j
```

### 构建测试

```bash
CUTLASS_BUILD="${CUTLASS_REPO}/build-codex-tests"

cmake -S "$CUTLASS_REPO" -B "$CUTLASS_BUILD" \
  -DCUTLASS_NVCC_ARCHS=100a

cmake --build "$CUTLASS_BUILD" --target test_unit -j
```

如果要找更窄的测试目标，先列出它们：

```bash
cmake --build "$CUTLASS_BUILD" --target help | grep test_unit
```

已在本地文档里明确出现的典型目标：

- `test_unit`
- `test_unit_gemm_warp`

## 3. CUTLASS Profiler Workflow

先看本地文档：

```text
media/docs/cpp/quickstart.md
media/docs/cpp/profiler.md
tools/profiler/
```

### 最小 GEMM baseline

```bash
"$CUTLASS_BUILD/tools/profiler/cutlass_profiler" \
  --kernels=sgemm \
  --m=4352 --n=4096 --k=4096
```

### 按 kernel name 搜索

```bash
"$CUTLASS_BUILD/tools/profiler/cutlass_profiler" \
  --kernels='*gemm*' \
  --enable-kernel-performance-search \
  --sort-results-flops-per-sec
```

### 对固定 shape 找 best kernel

```bash
"$CUTLASS_BUILD/tools/profiler/cutlass_profiler" \
  --kernels='*gemm*' \
  --enable-best-kernel-for-fixed-shape \
  --m=6144 --n=6144 --k=6144 \
  --sort-results-flops-per-sec
```

### Blackwell / cluster / blockscaled

做 Blackwell blockscaled / flexible cluster 时，先读：

- `media/docs/cpp/blackwell_functionality.md`
- `media/docs/cpp/blackwell_cluster_launch_control.md`
- `media/docs/cpp/profiler.md`

再选对应 example：

- `72_blackwell_narrow_precision_gemm`
- `73_blackwell_gemm_preferred_cluster`
- `81_blackwell_gemm_blockwise`
- `83_blackwell_sparse_gemm`
- `93_blackwell_low_latency_gqa`

## 4. Correctness Workflow

### CUTLASS C++ / CuTe C++

1. 从最近的 example 复制最小配置
2. 查最近的 unit test 目录
3. 如果问题与 pipeline / cluster / barrier 相关，再查 `test/unit/pipeline/` 或 `test/unit/cluster_launch/`
4. 再考虑改 header 或模板参数

推荐测试入口：

```text
test/unit/cute/core/
test/unit/cute/hopper/
test/unit/gemm/device/
test/unit/pipeline/
test/unit/cluster_launch/
```

### CuTe DSL

1. 从最近的 example 起步
2. 用 `test/examples/CuTeDSL/` 验证真实调用方式
3. 打开 line info / IR / PTX / CUBIN dump
4. 再决定是 DSL 层问题还是底层 CUDA/PTX 问题

推荐环境变量：

```bash
export CUTE_DSL_ARCH=sm_100
export CUTE_DSL_LINEINFO=1
export CUTE_DSL_PRINT_IR=1
export CUTE_DSL_KEEP_IR=1
export CUTE_DSL_KEEP_PTX=1
export CUTE_DSL_KEEP_CUBIN=1
export CUTE_DSL_DUMP_DIR=/tmp/cute_dsl_dump
```

## 5. Dump IR / PTX / CUBIN For CuTe DSL

本地文档事实源：

- `media/docs/pythonDSL/cute_dsl_general/debugging.rst`
- `python/CuTeDSL/cutlass/base_dsl/env_manager.py`
- `python/CuTeDSL/cutlass/base_dsl/common.py`

### 文件 dump

```bash
export CUTE_DSL_KEEP_IR=1
export CUTE_DSL_KEEP_PTX=1
export CUTE_DSL_KEEP_CUBIN=1
export CUTE_DSL_DUMP_DIR=/tmp/cute_dsl_dump
```

### 运行 example

```bash
python "$CUTLASS_REPO/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py"
```

### 检查产物

```bash
find /tmp/cute_dsl_dump -maxdepth 2 -type f | sort
```

### 程序内取回产物

CuTe DSL 编译后的对象支持：

- `__mlir__`
- `__ptx__`
- `__cubin__`

如果你在调 DSL lowering 或 codegen，优先用这些接口，而不是猜编译器做了什么。

## 6. Inspect Generated Code

### PTX / SASS

```bash
cuobjdump -ptx <kernel.cubin>
cuobjdump -sass <kernel.cubin>
nvdisasm <kernel.cubin>
```

如果是普通 C++/CUDA 构建产物，也可以先看：

```bash
cuobjdump -ptx <binary_or_so>
cuobjdump -sass <binary_or_so>
```

### 什么时候必须看生成代码

- Hopper WGMMA 路径不符合预期
- Blackwell `tcgen05` / blockscaled / sparse 路径不符合预期
- 怀疑 TMA load/store 或 barrier 序列有问题
- profiler 显示数学吞吐异常低，但高层模板看不出问题

## 7. Nsight / Sanitizer Workflow

### 时间去哪了

先用 nsys：

```bash
nsys profile -o report ./your_program
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
```

### 为什么这个 kernel 慢

再用 ncu：

```bash
ncu --kernel-name "yourKernel" --set full -o report ./your_program
```

### 怀疑越界 / race / 未定义行为

```bash
compute-sanitizer --tool memcheck ./your_program
compute-sanitizer --tool racecheck ./your_program
```

这些工具的底层解释不要在 CUTLASS 文档里硬猜，直接跳 `cuda-skill`：

- `ncu-guide.md`
- `nsys-guide.md`
- `debugging-tools.md`
- `performance-traps.md`

## 8. Typical Diagnostic Paths

### Hopper FP8 GEMM 改造

1. `54_hopper_fp8_warp_specialized_gemm`
2. `gemm_api_3x.md`
3. `sm90_mma_tma_gmma_ss_warpspecialized_fp8.hpp`
4. `test/unit/gemm/device/sm90_gemm_f8_*`
5. dump PTX/SASS，看是不是走了预期的 GMMA 路径
6. 去 `cuda-skill` 查 WGMMA/PTX 语义

### Blackwell blockscaled GEMM 排障

1. `72_blackwell_narrow_precision_gemm` 或 `81_blackwell_gemm_blockwise`
2. `blackwell_functionality.md`
3. `sm100_mma_warpspecialized_blockwise_scaling.hpp`、`sm100_blockscaled_mma_warpspecialized.hpp`
4. `test/unit/gemm/device/sm100_blockscaled_tensorop_gemm/`
5. 如果是 sparse，再看 `sm100_blockscaled_sparse_tensorop_gemm/`
6. 若怀疑 `tcgen05` 或 scale layout，跳 `cuda-skill`

### CuTe DSL Blackwell persistent kernel 调试

1. `examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py`
2. `test/examples/CuTeDSL/sm_100a/test_dense_gemm_persistent_prefetch.py`
3. 打开 `CUTE_DSL_LINEINFO`、`KEEP_IR`、`KEEP_PTX`、`KEEP_CUBIN`
4. 检查 dump 产物
5. 用 `cuobjdump` / `nvdisasm` 看生成代码
6. 再去 `cuda-skill` 查 `tcgen05` / TMA / barrier / Nsight

## 9. When To Jump To cuda-skill

出现以下任一情况就切换：

- 不确定 PTX 指令语义
- 不确定 barrier / memory ordering / TMA 行为
- 不确定 cluster / CLC 的底层约束
- 需要解释 Nsight 指标
- 需要解释 `compute-sanitizer` 报错

默认跳转目标：

- PTX tensor cores
- PTX async copy / barriers
- PTX Blackwell `tcgen05`
- CUDA programming guide cluster / async barriers
- Nsight Compute / Nsight Systems / compute-sanitizer 资料
