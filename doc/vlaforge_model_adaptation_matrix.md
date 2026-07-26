# VLAForge 模型适配与证据矩阵

本文是 Invocation IR v0.2 的模型覆盖实施清单。它不把论文结构映射、deterministic
fixture、真实模型 capture 和完整 C++ 部署混为一谈。

## 1. 证据等级

| 等级 | 含义 | 必需证据 |
|---|---|---|
| L0 | source/paper contract mapping | upstream revision、入口、I/O、状态与控制流审计 |
| L1 | deterministic executable fixture | 可执行 Adapter、Semantic IR/Plan 等价测试 |
| L2 | real frontend capture/eager parity | 真实源码与 checkpoint、eager/capture 数值对齐 |
| L3 | real compiled artifact parity | backend artifact 与 eager/capture 对齐 |
| L4 | generated no-Python C++ Session parity | 独立 C++ build/run、named outputs 与 state trace 对齐 |

任何 L1 fixture 都不得标成真实模型支持。模型卡片必须逐项列出当前等级和缺失证据。

## 2. 范式覆盖

### 2.1 机器人/具身

| 范式 | 对象 | 重点 IR 能力 | 目标 |
|---|---|---|---|
| RT-1-like | source-faithful fixture | history tensor+mask、离散 token、detokenize | L1 |
| RT-2-like | contract mapping | VLM autoregressive action token | L0/L1 |
| OpenVLA | 真实开源对象 | prefill exact cache、bounded decode、无 queue | real L3；L4 可选 |
| ACT-like | ChunkedAction template | queue/cursor 是 Adapter state，不是 core | L1 |
| Octo/Diffusion Policy-like | Octo 优先 | optional modality、bounded denoise、action chunk | L2–L4 |
| π0/SmolVLA-like | SmolVLA 真实优先 | prefix、flow loop、continuous chunk、cache | L4 |
| GR00T N1-like | 官方开源对象 | 多相机/embodiment schema、DiT、组合 artifact | L2–L4 |

### 2.2 自动驾驶

| 范式 | 对象 | 重点 IR 能力 | 目标 |
|---|---|---|---|
| StatelessTrajectory | DrivingTrajectory fixture | multi-camera、ego history、route、valid-count | L1 |
| AutoregressiveTrajectory | DrivingAR/AutoVLA | fast-slow branch、bounded token decode | L1，真实 L2+ |
| DiffusionPlanner | DrivingDiffusion/DiffusionDrive | 2-step denoise、K candidates+score | fixture-L4 + real L2/L3/L4 |
| HybridVLMPlanner | ReCogDrive-like | VLM + diffusion 跨 artifact 组合 | L1/L2+ |
| MultiTaskDriving | UniDriveVLA/OpenDriveVLA-like | 2D/3D token、多专家、多 named outputs | L1/L2+ |
| ExternalFeature Hybrid | DriveVLM-Dual-like fixture | C++ BEV/agent/map Region plugin | L1/L4 |

## 3. Model Adaptation Card

每个目录 `doc/model_cards/<model>.md` 使用同一模板：

```text
upstream repository/revision/checkpoint hash/license
source entry points
input/output schema and bounded profiles
authoritative persistent state
derived cache and loop-carried SSA
TensorRegion and artifact partition
loop/branch/cache semantics
Adapter LOC and shared template reuse
new core op count
unsupported items
L0/L1/L2/L3/L4 evidence paths
memory/performance results
```

通用性验收目标：

- 每个新增模型的 core op 增量为 0；
- 新代码优先只包含 Adapter、Region、Validator 与 Artifact binding；
- 统计 Adapter LOC 和模板复用率；
- 新特性先尝试封装进 TensorRegion 或已有结构化 `if/for`；
- extension op 必须带新 schema version、type verifier、reference semantics、
  Plan lowering、runtime/codegen 和 serialization tests。

## 4. 当前实现状态

| 对象 | 当前证据 | v0.2 fixture | generated C++ | 真实 checkpoint 缺口 |
|---|---|---:|---:|---|
| RT-1 | L0 + L1 | 是 | 否 | L2–L4 |
| ACT | L0 + L1 | 是 | 否 | L2–L4 |
| Octo | L0 + L1 | 是 | 否 | JAX capture 与 L2–L4 |
| OpenVLA | L0 + L1 + real L2 + real L3 + fixture-L4 | 是 | fixture | real generated Session L4 |
| π0 | L0 + L1 | 是 | 否 | real capture/artifact |
| SmolVLA | L0 + L1 + real L2 + real L3 + real L4 | 是 | real checkpoint | complete on Host CUDA |
| GR00T N1.7 | L0 + L1 | 是 | 否 | real capture/artifact |
| DiffusionDrive | L0 + L1 + real L2 + real L3 + real L4 | 是 | real checkpoint | complete on Host CUDA |
| AutoVLA | L0 + L1 + real L2 partition + L3-candidate | 是 | 否 | 完整 camera/VLM/decode 与通过门槛的 real L3/L4 |
| ReCogDrive | L0 + structural L1 | hybrid | hybrid fixture | real hybrid artifacts |
| UniDriveVLA | L0 + structural L1 | multitask structure | 否 | license/checkpoint/L2–L4 |
| OpenDriveVLA | L0 + structural L1 | multitask structure | 否 | gated checkpoint/L2–L4 |

当前 11 类 executable/structural fixtures 均由 v0.2 通用 op 表达，新增 core
opcode 数为 0。完整 pinned revision 和 unsupported items 见
[model_cards/README.md](./model_cards/README.md)。

## 5. 分阶段交付

1. 稳定 generic InputPort、bounded profile、transactional output group 和 plugin ABI；
2. 完成四类 driving fixture 与 OpenVLA/SmolVLA/π0 fixture 迁移；
3. 完成通用 C ABI、typed wrapper 和 clean C++ parity；
4. SmolVLA、DiffusionDrive real L4 与 OpenVLA 分 Region real L3 已完成；
5. Host-CUDA 五 workload 统计矩阵、正式消融、10k Run 长稳和 profile 已完成；
6. 冻结 core 后完成 Octo、GR00T N1.7、AutoVLA held-out L0/L1 审计，
   三者 core-op 增量均为 0；
7. AutoVLA 发布 checkpoint 的真实 decoder 分区已达到 L2；conservative
   AOTI 结果因中间 Region NRMSE 超过预声明门槛，严格保留为 L3-candidate。

当前 RTX 3060/CUDA 12.8 投稿范围已经完成。OpenVLA real L4、完整端到端
AutoVLA、Octo/GR00T 的更多真实 checkpoint 层级、跨 GPU、第二机复现和
Orin 只属于可选增强，不是当前 Host-CUDA completion gate。
