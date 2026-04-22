# Edge-FM 性能优化总结报告

**日期**: 2026-04-22  
**平台**: RTX 3060 (sm_86)  
**模型**: Qwen2.5-1.5B  
**工作负载**: prefill=1024, decode=32

---

## 📊 性能现状

### Baseline 性能

| 指标 | TRT-Edge-LLM | Edge-FM | 对比 |
|------|--------------|---------|------|
| **Prefill** | ~70 ms | **124 ms** | 1.77x 慢 |
| **Decode** | ~10.1 ms | **10.0 ms** | 1.01x 快 ✅ |
| **端到端** | 438 ms | 482 ms | 1.10x 慢 |

**vs Transformers**: Edge-FM 比 PyTorch Transformers 快 **31.6%** ✅

---

## 🔍 深度性能分析

### Prefill Kernel 分布 (nsys profiling)

| 类别 | 时间 (ms) | 占比 | Calls | 说明 |
|------|-----------|------|-------|------|
| **GEMM** | **110.0** | **88.6%** | 112 | 主要瓶颈 |
| Attention | 8.1 | 6.5% | 85 | FlashInfer 已优化 |
| Activation | 4.7 | 3.8% | 28 | SiLU (可融合) |
| Other | 1.4 | 1.1% | 3 | - |
| **总计** | **124.2** | **100%** | **228** | - |

### GEMM 详细分析

**Top GEMM Kernels**:
1. CUTLASS 128x256x32: 62.3 ms (28 calls @ 2.2 ms)
2. CUTLASS 256x128x32: 34.9 ms (28 calls @ 1.2 ms)
3. Ampere 128x128: 7.4 ms (28 calls @ 0.26 ms)
4. Ampere 128x64: 5.4 ms (28 calls @ 0.19 ms)

**关键发现**:
- 单个 GEMM 性能优秀 (20+ TFLOPS)
- 问题是 **GEMM 数量过多** (112 calls)
- 每层 ~4 个 GEMM (QKV, O, Gate+Up, Down)

---

## 🎯 优化尝试总结

### 尝试 1: SiLU Fusion
**目标**: 融合 GEMM + SiLU activation (节省 4.7 ms)

**方法**: 
- 添加 `try_forward_prefill_swiglu_fused()` 方法
- 尝试使用 TensorRT-LLM fused kernel

**结果**: ❌ **未生效**
- Fused path 返回 false，fallback 到分离路径
- 可能原因: operator table 配置或 state 初始化问题

**收益**: 0 ms

---

### 尝试 2: cuBLASLt Algo 调优
**目标**: 优化 GEMM algo 选择

**方法**: 测试不同 algo_index (0-6)

**结果**: ❌ **收益微小**
- algo_index 4 → 0: 仅 0.12 ms 改善
- cuBLASLt 已经选择了较优 algo

**收益**: 0.12 ms

---

### 尝试 3: Fused MOE Tile 调优
**目标**: 优化 decode 阶段的 Fused MOE

**方法**: 
- 测试 Tile16x128x64Stage2 → Stage3
- 测试 Tile32x128x64Stage2

**结果**: ❌ **无改善**
- Stage3 vs Stage2: 无差异
- 说明不是 memory/pipeline bound

**收益**: 0 ms

---

### 尝试 4: CUTLASS GEMM + SiLU
**目标**: 使用 CUTLASS 实现自定义 fused kernel

**方法**: 
- 创建 CUTLASS GEMM with SiLU epilogue
- 自定义 epilogue functor

**结果**: ❌ **实现有问题**
- Kernel 编译成功但正确性测试失败
- 需要更深入理解 CUTLASS epilogue 机制

**收益**: 0 ms

---

## 💡 核心发现

### 1. 单个 Kernel 性能优秀
- GEMM: 20+ TFLOPS (vs PyTorch 18.5 TFLOPS)
- Attention: FlashInfer 高度优化
- 单个 kernel 已接近硬件极限

### 2. 真正的瓶颈：Kernel 数量
- **112 个 GEMM calls** 占用 110 ms
- 平均每个 GEMM: 0.98 ms
- Launch overhead + 累积效应

### 3. 架构层面的差异
**Edge-FM**:
- 每层 4-6 个独立 GEMM
- 分离的 activation kernels
- 较少的 fusion

**TRT-Edge-LLM**:
- 更激进的 kernel fusion
- 减少 kernel 数量
- Graph-level 优化

---

## 📈 性能对比

### Edge-FM 的优势
1. ✅ **比 Transformers 快 31.6%**
2. ✅ **Decode 性能与 TRT 持平**
3. ✅ **单个 kernel 性能优秀**
4. ✅ **代码清晰，易于维护**

### 与 TRT 的差距
1. ⚠️ **Prefill 慢 1.77x (54 ms)**
2. ⚠️ **GEMM 数量多 (112 vs ~60)**
3. ⚠️ **Fusion 程度低**

---

## 🚀 未来优化方向

### 短期 (1-2 周)
1. **修正 SiLU Fusion**
   - 调试 fused path 失败原因
   - 预期: -5 ms

2. **CUTLASS Epilogue 优化**
   - 修正 SiLU epilogue 实现
   - 预期: -5 ms

### 中期 (1-2 月)
1. **Layer Fusion**
   - QKV projection fusion
   - Linear + RMSNorm fusion
   - 预期: -10-15 ms

2. **Graph-level 优化**
   - 减少 kernel launch 次数
   - 优化 memory layout
   - 预期: -10-20 ms

### 长期 (3-6 月)
1. **架构重构**
   - 学习 TRT 的 fusion 策略
   - 实施更激进的 fusion
   - 预期: 接近或超越 TRT

2. **CUTLASS 3.x**
   - Warp specialization
   - Persistent kernels
   - 预期: 额外 10-15%

---

## 📊 预期优化路径

| 阶段 | 优化 | Prefill (ms) | vs Baseline | vs TRT |
|------|------|--------------|-------------|--------|
| **当前** | - | **124** | - | 1.77x 慢 |
| 短期 | SiLU + Epilogue | 114 | -8% | 1.63x 慢 |
| 中期 | + Layer Fusion | 90 | -27% | 1.29x 慢 |
| 长期 | + 架构重构 | 65 | -48% | **0.93x (快 7%)** |

---

## 🎓 经验教训

### 1. 性能分析要全面
- ❌ 只看 profiling 数据不够
- ✅ 需要 isolated benchmark 验证
- ✅ 区分 prefill vs decode

### 2. 优化要找对方向
- ❌ 优化已经很快的 kernel 收益有限
- ✅ 减少 kernel 数量更重要
- ✅ 架构层面的优化收益更大

### 3. 单点优化 vs 系统优化
- ❌ 单个 kernel 优化 10% → 总体 1%
- ✅ 减少 50% kernel 数量 → 总体 20-30%
- ✅ 系统性优化更有效

### 4. 工程实践
- ✅ 保持代码清晰可维护
- ✅ 及时回退无效尝试
- ✅ 文档化分析过程

---

## 📁 相关文件

### 分析工具
- `deliverables/kernel_opt/prefill_gemm_128x256x32/` - GEMM 对比分析
- `scripts/profile/profile_edgefm_generate_case.py` - 性能测试脚本

### 文档
- `doc/README.md` - 主文档索引
- `doc/orin_r36.4.3_qwen_benchmark_guide.md` - Orin 平台指南

### Profiling 数据
- `.tmp_codex/nsys/edgefm_prefill_only.nsys-rep` - Prefill-only profile
- `.tmp_codex/nsys/edgefm_prefill_1024_no_graph.nsys-rep` - 完整 profile

---

## 🎯 结论

**Edge-FM 当前性能**:
- ✅ 已经比 PyTorch Transformers 快 31.6%
- ✅ Decode 性能与 TRT 持平
- ⚠️ Prefill 比 TRT 慢 1.77x

**优化潜力**:
- 通过 kernel fusion 可以缩小与 TRT 的差距
- 长期有潜力超越 TRT
- 需要更深入的架构优化

**建议**:
- 短期: 接受当前性能，专注其他功能
- 中长期: 系统性实施 fusion 优化
- 学习 TRT 的优化策略

---

**报告完成日期**: 2026-04-22  
**分析工具**: nsys, CUTLASS, cuBLASLt  
**测试平台**: RTX 3060 12GB (sm_86)
