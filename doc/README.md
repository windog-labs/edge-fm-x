# EdgeFM 文档索引

## 主文档

- `doc/design.md` - 代码结构、运行时分层、配置与调度
- `doc/edge_fm_benchmark_tables.md` - Benchmark 数据表
- `doc/orin_r36.4.3_qwen_benchmark_guide.md` - Jetson Orin benchmark 指南

## 性能优化

### 当前状态 (2026-04-21)

**平台**: RTX 3060, Qwen2.5-1.5B, prefill=1024

| 指标 | TRT-Edge-LLM | Edge-FM | 差距 |
|------|--------------|---------|------|
| Prefill | ~70 ms | ~124 ms | **1.77x 慢** |
| Decode | ~10.1 ms | ~10.0 ms | 1.01x 快 |

### 瓶颈分析

**Prefill 时间分布**:
- GEMM operations: 110 ms (88.7%) ⚠️ **主要瓶颈**
- FlashInfer Attention: 5.9 ms (4.8%) ✅
- Activation: 4.7 ms (3.8%) ✅
- RMSNorm: 2.1 ms (1.7%) ✅

**根本原因**: 
- GEMM kernel 数量过多 (28 layers × 4-6 GEMM/layer)
- 单个 GEMM 性能优秀 (0.3 ms, 20 TFLOPS)
- 但累积起来占用 110 ms

### 优化策略

**Phase 1: Kernel Fusion** (最高优先级)
- 目标: 减少 GEMM kernel 数量
- 方法: GEMM + Bias + Activation fusion, Layer fusion
- 预期: 110 ms → 80 ms (-27%)

**Phase 2: Tile 优化**
- 目标: 优化单个 GEMM
- 方法: CUTLASS 3.x, Warp specialization
- 预期: 额外 10-15%

**Phase 3: Memory 优化**
- 目标: 减少 memory traffic
- 方法: Layout 优化, In-place operations
- 预期: 额外 5-10%

### 详细分析

- `doc/tmp/final_performance_analysis.md` - 完整性能分析报告
- `doc/tmp/optimization_direction_reevaluation.md` - 优化方向调整
- `deliverables/kernel_opt/` - Kernel 优化工作目录

## 快速开始

### Benchmark
```bash
# Prefill performance
python scripts/profile/profile_edgefm_generate_case.py \
  --model-path examples/qwen2.5-1.5b-instruct/qwen2.5-1.5b-instruct \
  --prefill-len 1024 --decode-len 32 --use-cuda-graph
```

### Profiling
```bash
# Nsys profiling
nsys profile -o output.nsys-rep \
  --trace=cuda,nvtx --capture-range=cudaProfilerApi \
  python scripts/profile/profile_edgefm_generate_case.py ...
```

---

**下一步**: 实施 Kernel Fusion 优化
