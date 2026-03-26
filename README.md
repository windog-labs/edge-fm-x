# EdgeFM

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CMake](https://img.shields.io/badge/CMake-3.15+-green.svg)](https://cmake.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/)
[![CUDA](https://img.shields.io/badge/CUDA-Required-orange.svg)](https://developer.nvidia.com/cuda-toolkit)

EdgeFM（Edge Foundation Model）是一个专为边缘端场景优化的通用大模型推理引擎。EdgeFM 针对边缘端推理的独特需求，提供高效的多模态理解、语言生成和决策推理能力，广泛应用于自动驾驶、具身智能、机器人控制等边缘端智能系统，助力边缘端大模型应用的快速部署。

## 特性

- 🎯 **极简设计**：针对边缘端大模型推理的独特需求，大幅简化推理框架设计。相比云端复杂的 continuous-batching 和动态前缀匹配机制，EdgeFM 采用固定前缀缓存和单请求处理模式，显著降低系统复杂度，提升可维护性
- ⚡ **极致性能**：深度集成 FlashInfer 等高性能算子库，并支持 SageAttention、MLA（Multi-head Latent Attention）等前沿高效 LLM 算子。针对边缘端大模型的特殊尺寸（如多模态 token 序列长度）进行专门的算子优化，充分发挥硬件算力
- 🛠️ **简单易用**：通过配置文件统一管理推理参数（如采样策略、KV cache 配置等），简化 `generate` 接口调用。同时提供基于 pybind11 的 Python 绑定，支持快速验证和便捷集成
- 🔌 **良好扩展性**：采用模块化架构设计，支持跨平台部署。目前已支持 NVIDIA RTX 5090 平台，并计划扩展支持 NVIDIA Thor 边缘计算平台以及地平线（Horizon）等主流边缘端芯片平台

## 硬件支持

| 硬件平台 | 状态 | 说明 |
|---------|------|------|
| x86 (NVIDIA RTX 5090) | ✅ 已支持 | 支持基于 x86 的 CPU 和 NVIDIA RTX 5090 GPU 平台<br>推荐配置：NVIDIA RTX 5090（CUDA 11.0+）<br>测试平台：x86_64 Linux |
| NVIDIA Thor | 🔄 计划支持 | NVIDIA Thor 边缘计算平台 |
| 地平线 J6M | 🔄 计划支持 | 地平线征程 J6M 边缘端芯片平台 |

## 系统要求

- **CMake**: 3.15 或更高版本
- **C++ 编译器**: 支持 C++17 标准（GCC 7+, Clang 5+, MSVC 2017+）
- **Python**: 3.10+（用于 Python 绑定和测试）

### 平台特定要求

- **NVIDIA 平台（5090、thor）**：
  - **CUDA**: 需要 CUDA 工具包（推荐 12.8+）
  - **cuDNN**: 需要 cuDNN 库
- **地平线 J6M 平台**：
  - 平台特定依赖（待补充）

## 安装

### 前置依赖

根据目标平台安装相应依赖：

#### NVIDIA 平台（5090、thor）

1. **CUDA 工具包**
   ```bash
   # 检查 CUDA 是否安装
   nvcc --version
   ```

#### 地平线 J6M 平台

平台特定依赖（待补充）

### 构建步骤

1. **克隆仓库并初始化子模块**
   ```bash
   git clone git@github.com:MenglingD/edge-fm.git
   cd edge-fm
   git submodule update --init --recursive
   ```

2. **配置和构建**

   使用默认平台（5090）：
   ```bash
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   make install
   ```

   指定目标平台：
   ```bash
   mkdir build && cd build
   cmake .. -DPLATFORM=5090    # NVIDIA RTX 5090
   # 或
   cmake .. -DPLATFORM=thor    # NVIDIA Thor
   # 或
   cmake .. -DPLATFORM=j6m     # 地平线 J6M
   make -j$(nproc)
   make install
   ```

   支持的平台选项：
   - `5090`: NVIDIA RTX 5090（x86_64）
   - `thor`: NVIDIA Thor 边缘计算平台
   - `j6m`: 地平线征程 J6M 边缘端芯片平台
### Python 绑定

构建完成后，Python 模块将生成在 `build/python/` 目录中。

将 Python 模块路径添加到 `PYTHONPATH`：
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/edge-fm/build/python
```

## 使用样例

### C++ 接口

```cpp
#include <edge-fm/edge-fm.h>
#include <vector>

using namespace edge_fm;

// 初始化推理引擎
EdgeFM engine("examples/qwen2.5-vl/config.json");

// 创建请求（仅文本）
std::vector<int32_t> token_ids = {151643, 151644, 198, 2610, 525, 198};
Request request(0, token_ids);

// 生成响应
Response response = engine.generate(request);

// 获取生成的 token IDs
const auto& generated_tokens = response.token_ids();
```

### Python 接口

```python
import edge_fm

# 初始化推理引擎
engine = edge_fm.EdgeFM("examples/qwen2.5-vl/config.json")

# 创建请求（仅文本）
token_ids = [151643, 151644, 198, 2610, 525, 198]
request = edge_fm.Request(request_id=0, token_ids=token_ids)

# 生成响应
response = engine.generate(request)

# 获取生成的 token IDs
generated_tokens = response.token_ids()
```

### Qwen2.5-VL 使用示例

仓库提供了完整的 Qwen2.5-VL 使用示例，位于 `examples/qwen2.5-vl/` 目录：

1. **下载模型**（如需要）：
   ```bash
   cd examples/qwen2.5-vl
   ./download.sh
   ```

2. **运行推理**：
   ```bash
   # Python 示例
   python3 generate.py
   ```

3. **配置文件**：`examples/qwen2.5-vl/config.json` 包含了完整的配置示例，包括：
   - 两阶段模型路径配置（prefill/decode）
   - 投机采样配置（EAGLE3）
   - KV cache 配置（包含 prefix token ids）
   - 采样参数配置

### 推理配置文件（JSON）

配置文件采用 JSON 格式，核心字段说明：

- **`prefill_model_path` / `decode_model_path`**：两阶段模型路径配置
- **`speculative`**：投机采样（Speculative Sampling）配置
- **`runtime`**：引擎运行时/执行策略配置
- **`kvcache`**：KV cache 管理策略，包括压缩配置和请求槽位配置
- **`sampling`**：采样参数配置（temperature、top_k、top_p、max_new_tokens）

更详细的配置说明请参考 `examples/qwen2.5-vl/config.json` 和 `examples/config/engine_default.json`。

## 支持模型列表

| 模型系列 | 状态 | 说明 |
|---------|------|------|
| Qwen2.5 | ✅ 已支持 | 通义千问2.5系列模型<br>支持模型文件格式转换（参考 `scripts/convert_qwen3.py`） |
| 更多模型 | 🔄 计划支持 | 更多模型支持正在开发中... |

## 性能测试

### 推理性能

以下性能数据基于 EdgeFM 在不同硬件平台上的测试结果：

| 模型 | 硬件平台 | 量化精度 | 序列长度 | 推理速度 (tokens/s) | 首包延迟 (ms) | 备注 |
|------|---------|---------|---------|-------------------|--------------|------|
| Qwen2.5-7B | RTX 5090 | FP16 | 2048 | - | - | 测试中 |
| Qwen2.5-14B | RTX 5090 | FP16 | 2048 | - | - | 测试中 |

> **说明**：
> - 推理速度：decode 阶段的平均生成速度
> - 首包延迟：从输入到第一个 token 输出的时间
> - 测试环境：单请求、无批处理模式
> - 更多性能数据持续更新中...

### 运行性能测试

你可以使用项目提供的性能测试工具进行基准测试：

```bash
# Python 性能测试
cd tests/benchmark
python test_attn.py --model <model_path> --config <config_path>
```

## 项目结构

```
edge-fm/
├── cmake/                  # CMake 模块和工具
├── include/                # 公共头文件
│   └── edge-fm/
│       ├── core.h        # 核心类型定义
│       └── edge-fm.h    # 主接口
├── src/                  # 源代码
│   ├── engine/           # 推理引擎
│   │   ├── speculative/  # 投机采样引擎
│   │   └── ...
│   ├── layers/           # 神经网络层
│   ├── models/           # 模型实现
│   │   └── qwen2_5/      # Qwen2.5-VL 模型
│   ├── python/           # Python 绑定
│   ├── utils/            # 工具函数
│   └── edge-fm.cpp      # 主实现
├── examples/             # 使用示例
│   ├── config/           # 配置文件示例
│   └── qwen2.5-vl/       # Qwen2.5-VL 示例
├── tests/                # 测试文件
│   ├── benchmark/        # 性能测试
│   └── models/           # 模型测试
├── scripts/              # 工具脚本
├── third_party/          # 第三方依赖（Git 子模块）
```

## 性能优化

EdgeFM 针对边缘端大模型推理场景，从多个维度进行深度优化，实现极致性能：

### 高效算子实现

- **高性能算子库集成**：深度集成 FlashInfer 等业界领先的高性能算子库，提供优化的注意力机制和矩阵运算
- **前沿算子支持**：支持 SageAttention、MLA（Multi-head Latent Attention）等前沿高效 LLM 算子，充分发挥硬件算力
- **多模态优化**：针对边缘端大模型的特殊尺寸（如多模态 token 序列长度）进行专门的算子优化，支持视觉、语言、动作等多种模态

### 简化逻辑设计

- **单请求处理模式**：针对边缘端单用户场景，摒弃复杂的 continuous-batching 和动态调度机制，大幅简化系统复杂度
- **固定前缀缓存**：采用固定 prefix KV cache 机制，预缓存常见请求前缀，避免重复计算，显著提升推理效率
- **轻量级架构**：去除不必要的批处理和并行调度逻辑，专注于单请求低延迟推理

### 极致性能优化

- **两阶段量化策略**：支持为 prefill 和 decode 阶段配置不同的量化模型，针对各阶段的计算特点选择最优量化精度，平衡首包延迟与续写吞吐
- **任务特定优化**：针对自动驾驶、具身智能等特定应用场景，通过词表裁剪、模型压缩等技术，减少模型参数量和计算量，提升推理速度
- **高效投机采样**：集成高效的投机采样模型（如 EAGLE3），通过草稿模型快速生成候选 token 序列，显著提升生成吞吐
- **KV 压缩算法**：支持 FlashMLA 等前沿 KV cache 压缩算法，在保证推理质量的前提下大幅降低内存占用，提升系统资源利用率

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过 Issue 联系我们。