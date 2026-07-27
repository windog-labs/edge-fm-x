# VLAForge

[English](README.en.md) | 中文

VLAForge 是一个面向机器人与自动驾驶 VLA 部署的有状态调用编译器。它把
Adapter 声明的模型调用编译为经过 schema 校验、无需 Python 的 C++ Session，
由底软准备输入并主动调用 `Session::Run()`。

## 设计边界

VLAForge 只负责模型部署语义：

- 带可选 `InputRevision` 的 Tensor/Scalar 输入；
- authoritative persistent state 与可失效 derived cache；
- TensorRegion、结构化分支和有界循环；
- exact reuse contract；
- 原子提交的多 named outputs；
- 稳定 C ABI、模型强类型 C++ wrapper 与外部 Region plugin ABI。

它不采集或同步传感器，不维护周期、deadline 或帧率，不接入
ROS/Cyber，不发布底盘动作，也不实现车辆安全层。

## 仓库结构

```text
vlaforge/
  python/vlaforge/   Semantic IR、编译器、Plan、codegen、deployment
  runtime/           无 Python C/C++ runtime
  backends/          AOTInductor/TorchScript Region backend
  include/           稳定 C/C++ ABI
  examples/          确定性模型范式 fixture
  tests/             Python、C++、CUDA 与真实模型门禁
  tools/             capture、artifact、benchmark 与论文审计工具
doc/
  model_cards/       模型适配卡
  reports/           可复现实验与 claim-evidence
```

旧 EdgeFM engine、LLM/VLM operator、Qwen 示例、TensorRT/CUTLASS 子模块及
自定义 CUDA kernel 不属于本分支，也不进入 VLAForge 构建图。

## Python 开发

```bash
python -m pip install -e 'vlaforge[test]'
python -m pytest -q vlaforge/tests
```

## C/C++ runtime

CPU runtime 与 ABI 测试：

```bash
cmake --preset host-cpu
cmake --build --preset host-cpu --parallel
ctest --preset host-cpu
```

CUDA AOTI backend 使用当前 PyTorch 的 CMake package：

```bash
export VLAFORGE_TORCH_CMAKE_PREFIX_PATH="$(
  python -c 'import torch; print(torch.utils.cmake_prefix_path)'
)"
cmake --preset host-cuda
cmake --build --preset host-cuda --parallel
ctest --preset host-cuda
```

## 当前证据边界

真实 SmolVLA、DiffusionDrive、OpenVLA-7B 和 MindDrive 0.5B 均已完成
Host-CUDA generated no-Python C++ Session 路径。OpenVLA 使用稳定 raw
wrapper/cubin provider 做 invocation-resident weight paging；该 L4 是
correctness/deployment evidence，不是 latency benchmark。正式统计、消融、
Model Cards、复现清单和 release gate 位于
[doc/reports](doc/reports/README.md)。

当前性能结论只覆盖 RTX 3060 (`sm_86`) / CUDA 12.8。Orin、跨 GPU 和第二台
机器复现是可选增强，不影响当前 Host-CUDA 论文工程完成条件。

核心文档：

- [Invocation IR v0.2](doc/vlaforge_invocation_ir_v0_2.md)
- [开发计划](doc/vlaforge_development_plan.md)
- [论文设计](doc/vlaforge_paper_design.md)
- [模型适配矩阵](doc/vlaforge_model_adaptation_matrix.md)
- [模型适配卡](doc/model_cards/README.md)
