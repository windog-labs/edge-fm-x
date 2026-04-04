---
name: triton-skill
description: "Write, debug, and optimize Triton and Gluon GPU kernels using local source code, tutorials, and kernel references. Use when the user mentions Triton, Gluon, tl.load, tl.store, tl.dot, triton.jit, gluon.jit, wgmma, tcgen05, TMA, tensor descriptor, persistent kernel, warp specialization, fused attention, matmul kernel, kernel fusion, tl.program_id, triton autotune, MXFP, FP8, FP4, block-scaled matmul, SwiGLU, top-k, or asks about writing GPU kernels in Python."
---

# Triton & Gluon Kernel Development

## Workspace-First Source Location

在 `edge-fm-x` 工作区内，**优先以当前仓库的 `third_party/triton/` 为准**。  
这个目录是本仓库里的 git submodule，比旧的 skill 安装副本更可信。

```bash
TRITON_REPO="/xs-train-nas/zzm/repos/edge-fm-x/third_party/triton"
```

如果 `third_party/triton` 还没初始化，先在仓库根目录执行：

```bash
git submodule update --init --recursive third_party/triton
```

只有当前工作区不存在 `third_party/triton` 时，才退回到 skill 安装目录下的 `repos/triton/`。

当前 skill 默认**聚焦 NVIDIA 后端**，优先看 CUDA / Hopper / Blackwell / Ampere 相关路径；除非用户明确要求，否则先忽略 AMD 路线。

## Repo Map

### Triton Tutorials

先看 `python/tutorials/README.rst`，里面有最小依赖说明：`pip install -r python/tutorials/requirements.txt`。

```
TRITON_REPO/python/tutorials/
├── 01-vector-add.py                  # Triton 基础: @triton.jit, program_id, load/store
├── 02-fused-softmax.py               # reduction, 内核融合, tl.max/tl.sum/tl.exp
├── 03-matrix-multiplication.py       # block GEMM, L2-friendly launch order, @triton.autotune
├── 04-low-memory-dropout.py          # tl.rand, seed-based dropout
├── 05-layer-norm.py                  # backward, atomic ops, tl.atomic_cas
├── 06-fused-attention.py             # fused attention / Flash Attention 风格实现
├── 07-extern-functions.py            # libdevice / extern function 调用
├── 08-grouped-gemm.py                # grouped GEMM, tensor descriptor, TMA
├── 09-persistent-matmul.py           # persistent kernel, TMA, warp specialize, proton
├── 10-block-scaled-matmul.py         # FP4 / FP8 / MXFP, tl.dot_scaled
└── 11-programmatic-dependent-launch.py  # PDL, gdc_wait, gdc_launch_dependents
```

### Gluon Tutorials

```
TRITON_REPO/python/tutorials/gluon/
├── 01-intro.py                       # Gluon 与 Triton 的关系, @gluon.jit, autotune
├── 02-layouts.py                     # BlockedLayout / LinearEncoding / layout 基础
├── 03-async-copy.py                  # async copy, pipeline, shared memory
├── 04-tma.py                         # TensorDescriptor, TMA, mbarrier
├── 05-wgmma.py                       # Hopper WGMMA
├── 06-tcgen05.py                     # Blackwell tcgen05 MMA / tensor memory
├── 07-persistence.py                 # 持久化 kernel, MMA abstraction, multi-stage pipeline
├── 08-warp-specialization.py         # warp specialization, producer/consumer
├── 09-tma-gather-scatter.py          # Blackwell TMA gather/scatter
├── 10-tcgen05-copy.py                # tcgen05_copy, shared -> tensor memory
├── 11-tcgen05-mma-scaled.py          # tcgen05_mma_scaled, nvfp4 / mxfp4 / mxfp8
├── 12-cluster-launch-control.py      # CLC, dynamic work distribution
├── 13-conv-im2col.py                 # TMA im2col, implicit GEMM convolution
└── 14-multicta.py                    # multi-CTA / CGA, distributed shared memory, CLC
```

### Gluon Examples

`python/examples/gluon/` 是**完整实现**，适合在看完教程后抄生产级结构。

```
TRITON_REPO/python/examples/gluon/
├── 01-attention-forward.py           # Blackwell attention forward, producer/consumer, TMA, tcgen05_mma
├── 02-convolution.py                 # 端到端 convolution, TensorDescriptorIm2Col, implicit GEMM
├── 03-matmul-multicta.py             # 2CTA / multi-CTA matmul, cga_layout, CLC
└── 04-2cta-block-scale-matmul.py     # 2CTA block-scaled matmul, tcgen05_mma_scaled, MXFP/NVFP4
```

### Production Kernels

`python/triton_kernels/triton_kernels/` 是 Triton 官方仓库里最值得参考的“可复用实现”。

```
TRITON_REPO/python/triton_kernels/triton_kernels/
├── matmul.py                         # 高层 matmul API
├── matmul_details/
│   ├── _matmul.py                    # dense GEMM, TMA, mxfp4/8
│   ├── _p_matmul.py                  # persistent GEMM, ragged TMA
│   ├── _common.py                    # 偏移/launch 共用逻辑
│   └── opt_flags*.py                 # NVIDIA 优化标志与相关 opt flags
├── reduce.py                         # reduction kernel
├── topk.py                           # top-k forward/backward
├── swiglu.py                         # SwiGLU
├── compaction.py                     # masked compaction
├── distributed.py                    # distributed / SymmetricMemory
├── numerics.py                       # FP8 / MXFP 数值入口
├── numerics_details/
│   ├── mxfp.py                       # MXFP 量化入口
│   ├── flexpoint.py                  # flexpoint
│   └── mxfp_details/                 # downcast / upcast 细节
├── tensor.py                         # tensor / layout 抽象
├── tensor_details/
│   ├── ragged_tensor.py              # ragged tensor
│   ├── layout.py                     # layout 入口
│   ├── bitmatrix.py                  # bitmatrix
│   └── layout_details/               # Blackwell / Hopper 等布局细节
├── testing.py                        # assert_close, compute_sanitizer helpers
├── roofline.py                       # roofline 分析
├── target_info.py                    # target feature 抽象
├── meta.py                           # kernel meta helpers
└── specialize.py                     # specialization helpers
```

测试与 benchmark 也在同目录下：

```
TRITON_REPO/python/triton_kernels/tests/
TRITON_REPO/python/triton_kernels/bench/
```

### Python Frontend / Runtime / Tools

```
TRITON_REPO/python/triton/
├── __init__.py                       # triton.jit / autotune / Config 导出
├── language/                         # tl.* 语义与实现
├── runtime/                          # jit / cache / interpreter / autotuner / driver
├── compiler/                         # codegen / compile pipeline
├── backends/                         # backend compiler / driver 接口
├── knobs.py                          # 环境变量与调试开关
├── testing.py                        # benchmark / perf_report 工具
├── tools/
│   ├── compile.py                    # 编译辅助
│   ├── disasm.py                     # 反汇编辅助
│   ├── link.py                       # 链接辅助
│   ├── tensor_descriptor.py          # host-side TensorDescriptor 工具
│   ├── ragged_tma.py                 # ragged TMA 工具
│   ├── mxfp.py                       # MXFP helper types
│   ├── gsan.py                       # GSan 工具入口
│   └── triton_to_gluon_translator/   # Triton -> Gluon translator
└── experimental/
    ├── gluon/                        # Gluon frontend / runtime / language
    └── gsan/                         # GSan concurrency sanitizer
```

### NVIDIA-Specific Gluon Entry Points

```
TRITON_REPO/python/triton/experimental/gluon/
├── nvidia/hopper.py                  # Hopper TensorDescriptor
├── nvidia/blackwell.py               # Blackwell TensorDescriptor / tensor memory
└── language/
    ├── nvidia/hopper/                # tma, mbarrier, cluster
    ├── nvidia/blackwell/             # tma, clc, float2, tcgen05 相关
    └── nvidia/ampere/                # async_copy, mbarrier
```

如果只是做当前项目里的 Triton / Gluon CUDA 内核，默认先不用展开 `experimental/gluon/amd/`。

### Compiler / IR / Passes

```
TRITON_REPO/include/triton/
├── Dialect/
│   ├── Triton/                       # Triton IR dialect
│   ├── TritonGPU/                    # layouts / encodings / GPU attrs
│   ├── TritonNvidiaGPU/              # WGMMA / TMA / tcgen05 / tensor memory
│   ├── Gluon/                        # Gluon dialect
│   └── TritonInstrument/             # instrumentation / sanitizer dialect
├── Conversion/
│   ├── TritonToTritonGPU/            # TTIR -> TTGIR
│   └── TritonGPUToLLVM/              # TTGIR -> LLVM
├── Analysis/                         # alias / allocation / axis info / membar
└── Tools/                            # layout / swizzle / plugin utils

TRITON_REPO/lib/
├── Dialect/
│   ├── Triton/                       # ops / canonicalize / transforms
│   ├── TritonGPU/                    # coalesce / prefetch / pipeliner / warp specialization
│   ├── TritonNvidiaGPU/              # TMA / MMALowering / TMem / ConSan NVIDIA
│   ├── Gluon/                        # infer encodings / inline / simplify CFG
│   └── TritonInstrument/             # GlobalSanitizer / ConcurrencySanitizer / FpSanitizer
├── Conversion/
│   ├── TritonToTritonGPU/
│   ├── TritonGPUToLLVM/
│   └── TritonInstrumentToLLVM/
└── Target/LLVMIR/                    # LLVM IR utilities / passes
```

相关可执行工具：

```
TRITON_REPO/bin/triton-opt
TRITON_REPO/bin/triton-llvm-opt
TRITON_REPO/bin/triton-lsp
TRITON_REPO/bin/triton-reduce
TRITON_REPO/bin/triton-tensor-layout
```

### Docs / Tests

```
TRITON_REPO/docs/python-api/triton.rst
TRITON_REPO/docs/python-api/triton.language.rst
TRITON_REPO/docs/python-api/triton-semantics.rst
TRITON_REPO/docs/python-api/triton.language.extra.cuda.rst
TRITON_REPO/docs/programming-guide/chapter-3/debugging.rst

TRITON_REPO/test/Conversion/          # lowering / backend conversion 样例
TRITON_REPO/test/Gluon/               # Gluon dialect / layout inference
TRITON_REPO/test/Hopper/WarpSpecialization/
TRITON_REPO/test/LLVMIR/
```

## Search Strategy

**优先用 `rg` 做定点搜索，不要整仓库全文加载。**  
先确定 `TRITON_REPO`，然后全部用绝对路径搜索。

```bash
TRITON_REPO="/xs-train-nas/zzm/repos/edge-fm-x/third_party/triton"
```

### Triton Frontend / Language

```bash
# 查 Triton 教程里的典型写法
rg '@triton.autotune|triton\.Config|tl\.dot|tl\.dot_scaled|tl\.make_tensor_descriptor'   $TRITON_REPO/python/tutorials

# 查某个 tl.* API 的定义与语义
rg 'def (load|store|dot|dot_scaled|make_tensor_descriptor|make_block_ptr)'   $TRITON_REPO/python/triton/language

# 查 host-side JIT / autotune / cache / interpreter
rg 'class Autotuner|def jit|class JITFunction|class CacheManager|class Interpreter'   $TRITON_REPO/python/triton/runtime   $TRITON_REPO/python/triton/compiler   $TRITON_REPO/python/triton
```

### Gluon / NVIDIA-Specific APIs

```bash
# 查 gluon.jit、constexpr、must_use_result
rg '@gluon\.jit|@gluon\.constexpr_function|@gluon\.must_use_result'   $TRITON_REPO/python/tutorials/gluon   $TRITON_REPO/python/examples/gluon

# 查 layout / cga_layout / num_ctas 模式
rg 'BlockedLayout|LinearEncoding|NVMMASharedLayout|cga_layout|num_ctas'   $TRITON_REPO/python/tutorials/gluon   $TRITON_REPO/python/examples/gluon

# 查 Hopper / Blackwell / Ampere 特性
rg 'wgmma|tcgen05|tma|mbarrier|clc|tensor_memory|TensorDescriptorIm2Col|async_copy'   $TRITON_REPO/python/tutorials/gluon   $TRITON_REPO/python/examples/gluon   $TRITON_REPO/python/triton/experimental/gluon
```

### Convolution / Multi-CTA / Block-Scaled MMA

```bash
# TMA im2col / implicit GEMM convolution
rg 'im2col|TensorDescriptorIm2Col|async_copy_global_to_shared_im2col'   $TRITON_REPO/python/tutorials/gluon/13-conv-im2col.py   $TRITON_REPO/python/examples/gluon/02-convolution.py

# Multi-CTA / CGA / CLC
rg 'num_ctas|cga_layout|distributed shared memory|clc|cluster'   $TRITON_REPO/python/tutorials/gluon/12-cluster-launch-control.py   $TRITON_REPO/python/tutorials/gluon/14-multicta.py   $TRITON_REPO/python/examples/gluon/03-matmul-multicta.py

# Block-scaled MMA / MXFP / NVFP4
rg 'dot_scaled|mxfp|nvfp4|tcgen05_mma_scaled|MXFP4Tensor|MXScaleTensor'   $TRITON_REPO/python/tutorials/10-block-scaled-matmul.py   $TRITON_REPO/python/tutorials/gluon/11-tcgen05-mma-scaled.py   $TRITON_REPO/python/examples/gluon/04-2cta-block-scale-matmul.py   $TRITON_REPO/python/triton_kernels/triton_kernels
```

### Production Kernels

```bash
# Dense / persistent / ragged TMA matmul
rg 'persistent|ragged|tma|pre_hook|TensorDescriptor'   $TRITON_REPO/python/triton_kernels/triton_kernels/matmul.py   $TRITON_REPO/python/triton_kernels/triton_kernels/matmul_details

# Top-k / SwiGLU / reduce / compaction
rg 'topk|swiglu|reduce|compaction|bitmatrix'   $TRITON_REPO/python/triton_kernels/triton_kernels

# Layout / swizzle / target-specific numerics
rg 'swizzle|blackwell|hopper|mxfp|flexpoint'   $TRITON_REPO/python/triton_kernels/triton_kernels/tensor_details   $TRITON_REPO/python/triton_kernels/triton_kernels/numerics_details
```

### Compiler / IR / Passes / Sanitizer

```bash
# Triton / TritonGPU / TritonNvidiaGPU / Gluon / TritonInstrument op 定义
rg 'def .*Op|def .*Attr|def .*Type'   $TRITON_REPO/include/triton/Dialect

# TritonGPU transform passes
rg 'Coalesce|Prefetch|RemoveLayoutConversions|AccelerateMatmul|WarpSpecialization|Pipeliner'   $TRITON_REPO/include/triton/Dialect/TritonGPU   $TRITON_REPO/lib/Dialect/TritonGPU/Transforms

# Gluon layout / CFG / inlining passes
rg 'InferCoalesced|ResolveAutoEncodings|SimplifyControlFlow|Inline'   $TRITON_REPO/include/triton/Dialect/Gluon   $TRITON_REPO/lib/Dialect/Gluon

# TTGIR -> LLVM lowering
rg 'DotOpToLLVM|MemoryOpToLLVM|ReduceOpToLLVM|ScanOpToLLVM|ConvertLayoutOpToLLVM'   $TRITON_REPO/include/triton/Conversion/TritonGPUToLLVM   $TRITON_REPO/lib/Conversion/TritonGPUToLLVM

# Instrumentation / GSan / ConSan
rg 'GSan|GlobalSanitizer|ConcurrencySanitizer|FpSanitizer|ConSan'   $TRITON_REPO/python/triton/experimental/gsan   $TRITON_REPO/python/triton/tools/gsan.py   $TRITON_REPO/include/triton/Dialect/TritonInstrument   $TRITON_REPO/lib/Dialect/TritonInstrument
```

### Docs / Tests / Translator

```bash
# Python API / semantics / debugging docs
rg 'autosummary|device_assert|TRITON_INTERPRET|compute-sanitizer'   $TRITON_REPO/docs/python-api   $TRITON_REPO/docs/programming-guide/chapter-3/debugging.rst   $TRITON_REPO/README.md

# Triton -> Gluon translator
rg 'translator|slice_kernel|inline_helpers'   $TRITON_REPO/python/triton/tools/triton_to_gluon_translator

# MLIR tests 里找 backend lowering 样例
rg 'wgmma|tma|tcgen05|warp_specialize|clc|tensor memory'   $TRITON_REPO/test/Conversion   $TRITON_REPO/test/Gluon   $TRITON_REPO/test/Hopper/WarpSpecialization
```

## When To Use Which Source

| Need | Source | Path |
|------|--------|------|
| Triton 基础语法 / grid / load-store | Tutorials 01-05 | `python/tutorials/01-*.py` ~ `05-*.py` |
| Triton GEMM / autotune / L2-friendly launch | Tutorial 03 | `python/tutorials/03-matrix-multiplication.py` |
| Persistent kernel / TMA / proton | Tutorial 09 | `python/tutorials/09-persistent-matmul.py` |
| Block-scaled matmul / FP4 / MXFP | Tutorial 10 | `python/tutorials/10-block-scaled-matmul.py` |
| Fused attention | Tutorial 06 | `python/tutorials/06-fused-attention.py` |
| Gluon 入门 / host launch / autotune | Gluon tutorial 01 | `python/tutorials/gluon/01-intro.py` |
| Layout / encoding / CTA 内 tile 映射 | Gluon tutorial 02 | `python/tutorials/gluon/02-layouts.py` |
| Async copy / TMA / mbarrier | Gluon tutorials 03-04 | `python/tutorials/gluon/03-async-copy.py`, `04-tma.py` |
| Hopper WGMMA | Gluon tutorial 05 | `python/tutorials/gluon/05-wgmma.py` |
| Blackwell tcgen05 / tensor memory | Gluon tutorials 06, 10, 11 | `python/tutorials/gluon/06-tcgen05.py`, `10-tcgen05-copy.py`, `11-tcgen05-mma-scaled.py` |
| Persistent / warp specialization | Gluon tutorials 07-08 | `python/tutorials/gluon/07-persistence.py`, `08-warp-specialization.py` |
| CLC / multi-CTA / CGA | Gluon tutorials 12, 14 | `python/tutorials/gluon/12-cluster-launch-control.py`, `14-multicta.py` |
| Convolution / TMA im2col | Gluon tutorial 13 + example 02 | `python/tutorials/gluon/13-conv-im2col.py`, `python/examples/gluon/02-convolution.py` |
| 完整 attention forward | Example 01 | `python/examples/gluon/01-attention-forward.py` |
| 完整 multi-CTA matmul | Example 03 | `python/examples/gluon/03-matmul-multicta.py` |
| 完整 2CTA block-scaled matmul | Example 04 | `python/examples/gluon/04-2cta-block-scale-matmul.py` |
| 生产级 GEMM / ragged / MXFP | `triton_kernels` | `python/triton_kernels/triton_kernels/matmul*.py`, `numerics_details/`, `tensor_details/` |
| Top-K / SwiGLU / compaction / reduce | `triton_kernels` | `python/triton_kernels/triton_kernels/topk.py`, `swiglu.py`, `compaction.py`, `reduce.py` |
| tl.* 语义 / 签名 / helper | language source + docs | `python/triton/language/`, `docs/python-api/triton.language.rst` |
| Tensor descriptor host 工具 | tools | `python/triton/tools/tensor_descriptor.py`, `ragged_tma.py` |
| Triton -> Gluon translator | tools | `python/triton/tools/triton_to_gluon_translator/` |
| Backend / JIT / cache / interpreter | runtime/compiler | `python/triton/runtime/`, `python/triton/compiler/`, `python/triton/knobs.py` |
| IR dialect / attrs / types | include | `include/triton/Dialect/` |
| GPU transform / pipeline / warp specialize | lib/include | `lib/Dialect/TritonGPU/Transforms/`, `include/triton/Dialect/TritonGPU/Transforms/` |
| LLVM lowering | lib/include | `lib/Conversion/TritonGPUToLLVM/`, `include/triton/Conversion/TritonGPUToLLVM/` |
| Sanitizer / instrumentation / race detection | Python + IR + lib | `python/triton/experimental/gsan/`, `python/triton/tools/gsan.py`, `include/triton/Dialect/TritonInstrument/`, `lib/Dialect/TritonInstrument/` |

## Common Coding Patterns

### Basic Triton Pattern

参考 `python/tutorials/01-vector-add.py`：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
```

### Triton Autotune Pattern

参考 `python/tutorials/03-matrix-multiplication.py`：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
                      num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                      num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    ...
```

### Gluon Basic Pattern

参考 `python/tutorials/gluon/01-intro.py`：

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    value = gl.load(in_ptr)
    gl.store(out_ptr, value)

def copy_scalar(input, output):
    copy_scalar_kernel[(1,)](input, output, num_warps=1)
```

### Gluon Multi-CTA Launch Pattern

参考 `python/tutorials/gluon/14-multicta.py`：

```python
multicta_softmax_kernel[(M,)](
    x,
    out,
    x.stride(0),
    out.stride(0),
    BLOCK_N=N,
    num_warps=cfg["num_warps"],
    num_ctas=cfg["num_ctas"],
)
```

如果用户问的是 **CGA / cluster / multi-CTA**，优先看 `14-multicta.py` 和 `examples/gluon/03-matmul-multicta.py`，不要只停留在单 CTA 教程。

## Troubleshooting

| 问题 | 可能原因 | 优先参考 |
|------|---------|---------|
| `tl.dot` / `tl.dot_scaled` 结果不对 | 输入 dtype、scale 布局或 accumulator dtype 不对 | `python/tutorials/03-matrix-multiplication.py`, `10-block-scaled-matmul.py`, `python/examples/gluon/04-2cta-block-scale-matmul.py` |
| autotune 没生效 | `key` 没覆盖真实变化维度，或 launch meta 没同步 | `python/tutorials/03-matrix-multiplication.py`, `python/tutorials/gluon/01-intro.py` |
| Tensor descriptor / TMA 出错 | block shape、连续性、host/device descriptor 或 layout 不匹配 | `python/tutorials/08-grouped-gemm.py`, `python/tutorials/09-persistent-matmul.py`, `python/tutorials/gluon/04-tma.py`, `python/triton/tools/tensor_descriptor.py` |
| im2col convolution 边界错误 | `TensorDescriptorIm2Col` 参数、padding、offset 或 block shape 不对 | `python/tutorials/gluon/13-conv-im2col.py`, `python/examples/gluon/02-convolution.py` |
| multi-CTA / CLC 行为异常 | `num_ctas`、`cga_layout`、cross-CTA sync、distributed shared memory 用错 | `python/tutorials/gluon/12-cluster-launch-control.py`, `14-multicta.py`, `python/examples/gluon/03-matmul-multicta.py` |
| Gluon layout inference / encoding 推断失败 | auto encoding / infer coalesced encoding / inline 相关 pass 问题 | `include/triton/Dialect/Gluon/`, `lib/Dialect/Gluon/`, `test/Gluon/` |
| 性能低 | 没用 persistent kernel、warp specialization、TMA、block-scaled path 或 multi-CTA | `python/tutorials/09-persistent-matmul.py`, `python/tutorials/gluon/07-persistence.py`, `python/tutorials/gluon/08-warp-specialization.py`, `python/tutorials/gluon/14-multicta.py`, `python/triton_kernels/triton_kernels/matmul_details/` |
| 数据竞争 / 可见性问题 | async copy、mbarrier、跨 CTA 同步或 global memory ordering 出错 | `docs/programming-guide/chapter-3/debugging.rst`, `python/triton/experimental/gsan/`, `include/triton/Dialect/TritonInstrument/`, `lib/Dialect/TritonInstrument/` |

## Debugging / Profiling Checklist

先看：

```
TRITON_REPO/docs/programming-guide/chapter-3/debugging.rst
TRITON_REPO/README.md
TRITON_REPO/python/triton/knobs.py
```

常用开关：

```bash
# Triton interpreter
TRITON_INTERPRET=1 python your_script.py

# device_assert 需要 TRITON_DEBUG=1
TRITON_DEBUG=1 python your_script.py

# MLIR / LLVM dump
MLIR_ENABLE_DUMP=1 python your_script.py
MLIR_DUMP_PATH=/tmp/triton-mlir python your_script.py
LLVM_IR_ENABLE_DUMP=1 python your_script.py

# 编译 reproducer
TRITON_REPRODUCER_PATH=/tmp/triton-reproducer.mlir python your_script.py

# autotune / kernel dump / timing
TRITON_PRINT_AUTOTUNING=1 python your_script.py
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/triton-kernel-dump python your_script.py
TRITON_ALWAYS_COMPILE=1 python your_script.py
MLIR_ENABLE_TIMING=1 LLVM_ENABLE_TIMING=1 python your_script.py
```

常用辅助：

```bash
# NVIDIA 内存访问 / race / 越界排查
compute-sanitizer python your_script.py

# 官方 testing helper 里也有 compute-sanitizer/roofline 入口
TRITON_REPO/python/triton_kernels/triton_kernels/testing.py
TRITON_REPO/python/triton_kernels/triton_kernels/roofline.py
```

如果问题看起来是 compiler / lowering bug，而不是 kernel 算法问题：

1. 先打开 `MLIR_ENABLE_DUMP=1` 和 `TRITON_REPRODUCER_PATH=...`
2. 再去 `include/triton/Dialect/*`、`lib/Dialect/TritonGPU/Transforms/` 和 `lib/Conversion/TritonGPUToLLVM/`
3. 同时对照 `test/Conversion/*.mlir`、`test/Gluon/*.mlir`

## Repo Update / Sync

当前工作区里 Triton 是 **git submodule**。  
同步代码时，优先用 submodule 工作流，不要再依赖旧的 `update-repos.sh triton` 说明。

```bash
# 初始化 / 同步到 superproject 记录的 revision
git submodule update --init --recursive third_party/triton

# 查看当前 submodule revision
git -C third_party/triton rev-parse HEAD
```

如果要研究某个问题，**始终以当前 submodule checkout 的实际文件为准**，不要假设教程数量、examples 目录或 pass 名称与旧版本一致。

## Additional References

- Triton 官方文档: https://triton-lang.org
- Triton Python API: `TRITON_REPO/docs/python-api/triton.rst`
- Triton Language API: `TRITON_REPO/docs/python-api/triton.language.rst`
- Triton semantics: `TRITON_REPO/docs/python-api/triton-semantics.rst`
- Triton debugging guide: `TRITON_REPO/docs/programming-guide/chapter-3/debugging.rst`
- Triton language source: `TRITON_REPO/python/triton/language/`
- Gluon experimental API: `TRITON_REPO/python/triton/experimental/gluon/`
