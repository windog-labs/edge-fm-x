# VLAForge 论文设计：Stateful Invocation Whole-Program Compilation

> 文档状态：当前权威论文方案
> 更新时间：2026-07-25
> 推荐方向：MLSys / 编译器与机器学习系统
> 推荐题目：**VLAForge: Compiling Stateful VLA Invocations to Verified C++ Edge Sessions**

## 1. 一句话主线

VLAForge 将一次被底软调用的、带跨调用模型状态的 VLA 程序表示成一个小型
Invocation IR，显式区分输入身份、权威状态、可重算缓存和事务输出；编译器据此
证明跨 Run 复用与内存优化合法，并生成无 Python 的 C++ Session。

它不是传感器同步器、控制周期调度器或动作发布框架。

## 2. 研究问题

今天的 Tensor compiler 能高效编译 `forward()` 内部的计算图，但 VLA 部署
常包含 Tensor graph 之外、又直接影响正确性的程序语义：

- 输入是否与上次 Run 是同一份逻辑数据；
- 哪些值必须跨 Run 保留，哪些只是可失效缓存；
- autoregressive、diffusion、flow 的有界循环如何携带状态；
- 多个 Region 与多个 named outputs 如何原子提交；
- backend failure 或 validation failure 后 state/output 是否一致；
- 如何让底软用稳定 C/C++ ABI 接入，而不嵌入 Python。

论文问题是：

> 能否用一个 VLA 专用、足够小的 whole-program IR，自动生成忠实且可优化的
> C++ deployment session，同时让新机器人和驾驶 VLA 主要只增加 Adapter、
> Region 与 artifact，而不修改 core runtime？

## 3. 为什么 Tensor Graph 不够

`torch.export` 捕获的是一次 Python callable 的 Tensor computation；
AOTInductor 可将 exported program 编译成非 Python artifact；ExecuTorch
提供通用 edge AOT/runtime 和 backend delegation。它们是 VLAForge 的
TensorRegion backend，而不是要替代的对象。

缺少的不是另一个 ATen IR，而是 Region 之间的 VLA invocation contract：

```text
external typed inputs + revisions
  -> read latest authoritative state
  -> invoke bounded Region program
  -> validate named output group
  -> atomically commit state and outputs
```

把这些逻辑继续手写在 Python/C++ glue 中，会产生三类难以由普通 Tensor graph
发现的错误：

1. 复用遗漏输入 revision 或 state version，得到 stale prefix/condition；
2. queue、cursor、hidden 等权威状态在失败路径被提前修改或静默丢失；
3. 多输出或 state/output 分别可见，底软读取到不一致的结果。

## 4. 系统边界

### 4.1 输入

底软负责传感器采集、同步、历史窗口与 ragged packing，再 push：

- static Tensor；
- Scalar/POD；
- optional fixed-shape input/default；
- `InputStamp.revision`；
- bounded profile 的 tensor + valid_count/mask。

timestamp 仅是 freshness metadata。VLAForge 不等待、对齐、丢帧或维持帧率。

### 4.2 输出

Session 返回 committed named output group。它可以是 action chunk，也可以是
trajectory、K candidates + scores、prediction、map、detection 或 VQA token。
底软决定执行前缀、选择轨迹、发布控制和安全仲裁。

### 4.3 扩展

protobuf/ROS/Cyber/任意 host object 不进入 IR。外部 NV12/点云/BEV/CAN
preprocessing 用静态 Tensor/Scalar `RegionExecutable` ABI 接入。

## 5. Invocation IR

### 5.1 六类核心语义

1. Stamped Input；
2. Versioned Authoritative State；
3. Pure TensorRegion Invoke；
4. structured `if` 与 bounded `for`；
5. exact Cache/Reuse Contract；
6. Transactional Named Output Group。

operation 集合仅为：

```text
input.read, txn.begin, state.read_latest, snapshot.value,
invoke, if, for, yield, state.stage_write, validate,
output.create, output.group, txn.commit/abort, return
```

没有 clock、tick、deadline、async scheduler 或 publish。

### 5.2 状态与缓存

论文最重要的语义区分不是“有状态/无状态”，而是：

| 类别 | 错误丢弃的后果 | 例子 |
|---|---|---|
| Authoritative state | 改变后续模型语义 | queue/cursor、hidden、RNG |
| Derived cache | 只损失性能，可重算 | VLM prefix、condition、DiT feature |

state 使用 immutable snapshot/version；只有成功 commit 才由 StateStore 分配
新 version。exact cache key 为：

$$
K_f = (\text{model}, \text{artifact}, \text{region}, \text{episode},
\text{InputRevision}^*, \text{StateVersion}^*)
$$

缺少 revision 时默认每次变化。近似 reuse 必须走显式 guard，不得冒充 exact。

### 5.3 事务输出

一次事务把 staged state 和一个 validated output group 原子提交。validation、
Region 或 backend 失败时：

- state version 不增加；
- staged state 丢弃；
- 上一 committed output 不被覆盖；
- Run 返回错误。

这比硬编码 `action.publish` 更通用，也覆盖自动驾驶多输出。

## 6. 编译器贡献

### Contribution 1：VLA-specific Invocation IR

一个小型、可验证、可序列化的 IR，首次在 VLA 部署编译边界中联合显式表达：

- 静态 typed I/O 与 exact input identity；
- authoritative state 与 derived cache 的语义分离；
- bounded generation loop；
- generic transactional output group。

措辞应限定为“在所比较的 VLA deployment compiler/runtime 中”，不能声称
首次发明 state、transaction、cache、loop 或 C++ runtime。

### Contribution 2：Revision/version-guided legality

编译器从 input revision 与 state snapshot version 推导：

- exact cross-Run memoization；
- condition/prefix loop-invariant hoisting；
- loop-carried decode/denoise state；
- persistent/cache/per-Run memory placement；
- static arena reuse；
- failure-safe state physicalization。

需要至少三个真实收益或必要性 case，不能只有 verifier。

### Contribution 3：Verified C++ session generation

从同一 IR/Plan 生成：

- generic C ABI；
- model-specific typed C++ wrapper；
- schema digest 检查；
- StateStore/Transaction/OutputStore；
- Region plugin/artifact bindings；
- no-Python clean executable。

用 normalized trace 对齐 eager/fixture、Semantic IR、Plan 和 generated C++ 的
输入 revision、cache、Region、state version、transaction 与 output group。

## 7. 创新性判断

### 7.1 当前创新点是否足够

只做“一个 VLA C++ runtime”不够，因为已有工作已经覆盖通用 VLA inference
runtime、edge execution、cache、pipeline 和异步控制。只做“persistent state
IR”也不够，因为通用 stateful compilation 和近期 agent-driven persistent-state
IR 已经存在。

当前组合在以下条件同时满足时有论文潜力：

1. IR 保持小而 VLA-specific，且机器人与驾驶模型新增 core op 数接近 0；
2. state/cache/output 的区分能阻止真实错误，不是纯软件工程抽象；
3. legality analysis 自动产生可测性能收益；
4. 至少一个真实 checkpoint 达到 generated C++ L4；
5. held-out model 证明适配主要发生在 Adapter/Region；
6. trace/failure injection 证明语义忠实。

目前仓库已经有完整 v0.2 架构、fixture 与 C++ substrate，但真实 checkpoint 的
v0.2 L2–L4 和性能数据尚未补齐。因此：

> 设计创新点已经形成，工程 substrate 足够支持投稿实验；但在真实模型
> L4、优化收益和硬件数据完成前，论文贡献尚未闭环。

### 7.2 最强差异点

推荐强调：

- **semantic identity, not time scheduling**：revision/version 决定 exactness；
- **authoritative state vs recomputable cache**：不同 failure/lifetime 合同；
- **state/output atomicity for generated model sessions**；
- **generic multi-output coverage across manipulation and driving**；
- **schema-safe customer C++ integration**；
- **adaptability evidence**：冻结 core 后模型适配成本可量化。

不建议强调：

- multi-clock、deadline、freshness scheduler；
- 首个 VLA C++ runtime；
- 首次 action chunk/cache/transaction；
- fixture 数量本身；
- 仅靠 Agent 自动写部署代码。

## 8. 相关工作与定位

### 8.1 通用 PyTorch 部署

- [torch.export](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export.html)
  捕获一次 callable 的 Tensor graph；
- [AOTInductor](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_aot_inductor.html)
  生成非 Python artifact；
- [ExecuTorch](https://docs.pytorch.org/executorch/stable/getting-started-architecture)
  提供 edge AOT、portable runtime 与 backend delegation。

VLAForge 位于其上方，表达跨 Region invocation/state/output contract，并可以
把这些系统作为 TensorRegion backend。

### 8.2 VLA runtime 与推理优化

- [vla.cpp](https://arxiv.org/abs/2606.08094) 统一多种 VLA 架构的 portable
  C++ inference runtime，并覆盖 flow/diffusion 与 prefix cache；
- [Embodied.cpp](https://arxiv.org/abs/2607.02501) 强调异构机器人上的
  portable embodied runtime、输入/头部插件和 multi-rate execution；
- [ActionFlow system](https://arxiv.org/abs/2512.20276) 通过跨请求 pipeline
  与 KV buffer 提升 OpenVLA edge throughput；
- [Reflex](https://arxiv.org/abs/2607.14695) 研究 flow VLA 的 streaming
  inference 与 cache correctness；
- [VLASH](https://arxiv.org/abs/2512.01031) 和
  [RTC](https://arxiv.org/abs/2506.07339) 研究异步 action execution 和
  chunk continuity。

VLAForge 不与这些工作争夺 robot scheduler/async control claim；它研究的是
被动 invocation 的可编译 state/cache/output 语义和 generated C++ contract。

### 8.3 Stateful/agent-driven systems

- [FlashRT](https://arxiv.org/abs/2607.18171) 使用 agent、persistent-state IR
  与 measurement-gated transformation 生成多 GPU multimodal deployment；
- [Execution-State Capsules](https://arxiv.org/abs/2606.20537) 研究 device
  execution state 的 snapshot/restore/fork/rollback。

因此 VLAForge 不把“persistent-state IR”或“snapshot/rollback”单独作为
新颖性。差异应落在 VLA-specific input identity、authoritative/cache
classification、transactional named outputs、legality 和 edge C++ ABI。

### 8.4 模型范式

实验矩阵以真实 upstream source 为准：

- robot：RT-1、ACT、OpenVLA、Octo、π0/SmolVLA、GR00T；
- driving：DiffusionDrive、AutoVLA、ReCogDrive、UniDriveVLA、
  OpenDriveVLA。

具体 pinned revision、license、checkpoint 与 unsupported items 见
[model_cards/README.md](./model_cards/README.md)。fixture 与真实模型必须分栏。

## 9. 实验设计

### 9.1 研究问题

- RQ1 Expressiveness：一个固定 core 是否覆盖不同 VLA 范式？
- RQ2 Correctness：Semantic/Plan/C++ 是否保持 output/state trace？
- RQ3 Optimization：revision/version-guided transforms 收益多大？
- RQ4 Integration：新模型和客户 C++ 输入需要多少 Adapter/core 修改？
- RQ5 Deployment：无 Python Session 在 Host CUDA/Orin 的 latency、memory、
  energy 和稳定性如何？

### 9.2 证据等级

| 等级 | 证据 |
|---|---|
| L0 | pinned source/paper contract |
| L1 | deterministic executable fixture |
| L2 | real checkpoint eager/frontend capture parity |
| L3 | real compiled artifact parity |
| L4 | real checkpoint generated no-Python C++ parity |

`fixture-L4` 只能证明编译链能力，不能证明真实 checkpoint 支持。

### 9.3 模型集合

最小投稿集合建议：

1. SmolVLA：flow + continuous action chunk + Adapter queue；
2. OpenVLA：autoregressive action token，无 persistent queue；
3. DiffusionDrive：多相机/ego、two-step diffusion、K candidates + score；
4. Octo 或 GR00T：optional modality/多 embodiment/DiT；
5. AutoVLA 或 ReCogDrive：真实驾驶 AR 或 VLM+diffusion hybrid；
6. 一个 held-out model：冻结 core 后测适配成本。

至少 SmolVLA/OpenVLA/DiffusionDrive 中的代表对象需要真实 L4；如果受
checkpoint/backend 限制，必须缩小论文 claim，不得用 fixture 替代。

### 9.4 Baselines

- eager PyTorch；
- TensorRegion-only export/AOT backend + 手写 host glue；
- VLAForge conservative profile；
- VLAForge verified profile；
- 对适用模型比较现有官方/runtime 实现。

`TensorRegion-only + handwritten glue` 是最关键 baseline，用来回答为什么需要
Invocation IR，而不只是更好的 Tensor kernels。

### 9.5 指标

Correctness：

- output max/mean error、token/action/trajectory equality；
- state version 与 normalized trace equality；
- cache hit/miss；
- failure injection 后 state/output unchanged；
- 10k+ Run deterministic soak。

Performance：

- end-to-end Run p50/p95/p99；
- Region latency、cache/LICM saved work；
- peak/per-class memory；
- startup/bundle size；
- Host CUDA 与 Orin power/energy；
- conservative vs verified transformation 消融。

Generality：

- Adapter LOC；
- shared template reuse rate；
- new core op count；
- required custom Region/backend LOC；
- unsupported upstream features。

### 9.6 必要性消融

至少构造并在真实路径复现：

1. revision 被忽略导致 stale condition cache；
2. authoritative queue 被当 cache 丢弃；
3. validation failure 后 state version 错误增长；
4. trajectory 与 candidate score 非原子可见；
5. schema 变更后 input ID 静默错绑；
6. denoise loop 中错误 hoist timestep-dependent value。

对每个 case 比较：

- handwritten/Tensor-only 无保护版本；
- verifier 拒绝或 conservative 执行；
- VLAForge verified transform；
- 数值、trace、性能结果。

## 10. 论文结构

1. Introduction：VLA deployment 的“Tensor graph 外正确性”；
2. Motivation：robot/driving 三个错误案例；
3. Design：输入身份、状态/缓存、事务输出、系统边界；
4. IR semantics and verifier；
5. Compiler analyses and memory/codegen；
6. C/C++ integration and Region ABI；
7. Evaluation：correctness、optimization、generality、deployment；
8. Related work；
9. Limitations；
10. Conclusion。

## 11. 图表清单

- Figure 1：host → Invocation IR → Plan → C++ Session 边界；
- Figure 2：四类 memory 与 transaction visibility；
- Figure 3：robot AR、flow/chunk、driving diffusion 三种 lowering；
- Table 1：与 Tensor compiler、VLA runtime、async control、state systems 的定位；
- Table 2：L0–L4 model coverage；
- Table 3：Adapter LOC/core-op increment；
- Table 4：Semantic/Plan/C++ correctness；
- Table 5：latency/memory/energy；
- Figure 4：exact cache 与 LICM 消融；
- Figure 5：failure injection。

## 12. 当前完成度

已完成：

- Invocation IR v0.2、parser/serializer/verifier；
- Reference Interpreter 与 Scheduled Plan；
- exact revision cache、bounded loop 与四类 memory plan；
- versioned state、transactional named outputs；
- generic C ABI、typed wrapper、schema digest；
- external Tensor/Scalar Region ABI；
- verified artifact schema v3、Compile Bundle v4，以及真实 CUDA AOTI
  package 到 generated no-Python C++ Session 的 production-path audit；
- robot/driving deterministic fixtures；
- generated no-Python C++ fixture parity；
- OpenVLA-7B 与 SmolVLA 真实 checkpoint eager/IR L2；
- SmolVLA 真实 prefix/solver-step/trim 的 `sm_86` AOTInductor L3 数值
  parity；BF16 artifact 不声称 bit-exact；
- SmolVLA 真实八 Region verified bundle 与 generated no-Python C++ L4：
  direct artifact/C++ action chunk bit-exact，并覆盖 revision cache、CUDA
  authoritative state、事务失败回滚、reset 和 typed/generic ABI；
- pinned upstream source audit 与 Model Adaptation Cards。

尚未完成、且决定投稿强度：

- DiffusionDrive/OpenVLA 的真实 L3/L4；
- real-model optimization speedup 与 memory 消融；
- frozen-core held-out model 数据；
- Host CUDA 长稳与 profile；
- Orin 真机 latency/power/closed-loop。

小型 CUDA AOTI audit 只证明 production artifact substrate 已经真实执行，
不能计入模型覆盖表中的 real-model L3/L4。SmolVLA 的 L3 是独立的固定
checkpoint、真实前端 capture、三 Region artifact 和逐步数值报告；同目录
的 L4 报告再将这些 artifacts 连接到 generated no-Python C++ Session。
证据见 `doc/reports/vlaforge_real_v03/`。

## 13. 投稿 go/no-go

满足以下条件再写成完整系统论文：

- ≥3 个不同范式真实模型达到 L3；
- ≥1 个 robot VLA 和 ≥1 个 driving planner 达到真实 L4；
- ≥3 个 legality-guided optimization/error-prevention case；
- held-out model 新增 core op 为 0；
- no-Python C++、failure injection、trace parity 均通过；
- 至少 Host CUDA 有完整性能数据；若写 Orin claim，则必须有真机证据。

若只完成 fixture-L4 和 source audit，应定位为 design/prototype，不应声称完整
real-model deployment compiler。
