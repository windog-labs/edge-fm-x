---
name: ncu-cuda-profiling
description: NCU (Nsight Compute) CUDA profiling workflow for edge-fm. Collects full metrics, persists reports to ncu_reports/, and auto-diagnoses bottlenecks (DRAM/L1/compute/occupancy). Use when profiling CUDA kernels, optimizing performance, analyzing ncu-rep files, or when the user mentions NCU, kernel performance, or GPU profiling.
---

# NCU CUDA 性能分析

edge-fm 的 NCU 自动化性能分析流程，支持全量指标采集、持久化存储和瓶颈诊断。

## 快速开始

### 采集（edge-fm 通过 Python 调用 CUDA）

```bash
# 对 pytest 用例采集（典型用法）
ncu --set full -o ncu_reports/attn_profile --target-processes all \
    python -m pytest tests/layers/test_attn.py -v -k "test_xxx"

# 或对任意 Python 脚本
ncu --set full -o ncu_reports/linear_profile --target-processes all \
    python tests/model_qwen2_one_layer.py
```

### 从已有报告提取

```bash
ncu --import ncu_reports/attn_profile.ncu-rep --print-summary per-kernel
ncu --import ncu_reports/attn_profile.ncu-rep --page raw --csv > ncu_reports/attn_profile.csv
```

### 自动化分析

```bash
python .cursor/skills/ncu-cuda-profiling/scripts/ncu_analyzer.py \
    --import ncu_reports/attn_profile.ncu-rep -o ncu_reports/attn_profile_analysis.md
```

## 报告存储

```
project_root/
├── ncu_reports/
│   ├── attn_profile.ncu-rep
│   ├── attn_profile.csv
│   └── attn_profile_analysis.md
└── ...
```

## 诊断规则

| 条件 | 瓶颈类型 | 优化方向 |
|------|----------|----------|
| dram_throughput > 70%, roofline < 30% | DRAM_MEMORY_BOUND | Block Tiling, float4 加载, Prefetching |
| l1tex_throughput > 80%, dram < 30% | L1_PRESSURE_BOUND | Shared Memory Padding (+1), Data Transpose |
| sm_busy < 50%, occupancy > 60% | LATENCY_BOUND | Double Buffering, ILP, Loop Unroll |
| roofline > 60%, sm_busy > 80% | COMPUTE_BOUND | FMA, FP16/TF32, Tensor Core |
| occupancy < 30%, sm_busy > 70% | OCCUPANCY_BOUND | 减少寄存器, 调整 block size, __launch_bounds__ |

## 输出模板

分析报告应包含：

- **报告信息**：Kernel 名、采集时间、报告路径
- **执行摘要**：主要瓶颈、置信度、关键指标
- **关键指标表**：Roofline、SM Busy、Occupancy、DRAM/L1/L2 Throughput
- **诊断详情**：瓶颈类型、判断依据
- **优化建议**：按优先级列出
- **下一步**：建议的 NCU 重采命令、验证清单

## 常见误区

- 高 Throughput 不一定高效：Compute + Memory 都高但 Roofline 低 = GPU 在“忙等”
- DRAM Throughput 低可能是好事：优化后降低说明缓存复用
- Occupancy 不是越高越好：目标是足够隐藏延迟即可

## 脚本

- **ncu_analyzer.py**：从 ncu-rep 提取指标、自动诊断、生成 Markdown 报告
