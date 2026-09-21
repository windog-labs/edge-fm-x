# ICRA 2027 EdgeFM: 当前设计与新论文的差距评估

日期: 2026-09-06
评估代码: `codex/vlaforge-paper-artifact` @ `592fc320d57398571dcb4ab686d5fb55fe9bac86`
论文: [ICRA_2027_EdgeFM.pdf](/home/zhangzimo/Repos/private/edge-fm-x/doc/ICRA_2027_EdgeFM.pdf)
范围: 以当前本地代码为准的设计审计、已有实验记录审计和小范围 CPU 回归，不是新增边缘设备性能实验。本轮不修改运行时代码，不把历史报告等同于本轮复现。

## 1. 结论

**方向兼容，底座可复用，但当前分支还不满足新论文的完整描述。不能只改名字、README 和表格，就把现有 VLAForge 视作论文中的 EdgeFM。**

当前版本已经实现了有价值的部署底座: 有状态调用 IR、结构化循环、显式 KV/噪声输入、静态内存规划、exact cache、事务式输出、独立 C++ Session 和可替换的 Region 后端。这些应保留。

新论文把重心移到四项尚未闭合的能力: **整段去噪回放、异构边缘后端、按步骤低精度部署、Agent 算子调优与复用**。这是中到大规模的增量工程，主要集中在执行后端、模型 Adapter、数值工具链和实验工具，不需要推翻现有 IR/Session。

此外，论文的若干强表述本身需要收敛，尤其是“激进量化但严格无损”“单流即可保证 deadline”“任意多模态前缀可离线缓存”。这些不能靠照着文字改代码来解决。

## 2. 论文主张与代码对应

| 论文主张 | 当前证据 | 判断与需要的工作 |
|---|---|---|
| 面向 VLA、batch=1、无 Python 的确定性调用 | [README](/home/zhangzimo/Repos/private/edge-fm-x/README.md:5)、[编译入口](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/compiler.py:296) | 基本符合部署定位。无动态请求调度不等于消除了操作系统、GPU、热管理等全部抖动。 |
| 编码一次、动作头执行 N 次 | [SmolVLA PrefixModule](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/adapters/smolvla_frontend.py:78)、[SolverModule](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/adapters/smolvla_frontend.py:189) | 已有真实 SmolVLA 分阶段实现，可以复用。其他模型仍需分别确认其 conditioning 形式，并非都输出 KV。 |
| 整段 N 步只做一次图回放 | [Region run/synchronize](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/codegen/cpp.py:1617)、[host for lowering](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/codegen/cpp.py:1672) | 当前仍为 C++ 循环和逐 Region 同步，没有当前分支整段 CUDA Graph replay 实现证据。有界循环不等于设备图回放。 |
| 固定单流执行 | [TensorRT backend](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/backends/tensorrt_region_executable.cpp:235)、[Region Interface](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/include/vlaforge/runtime/region_executable.h:119) | TensorRT Region 各自持有 stream，公共创建参数没有 Session 共享 stream/execution-context 契约。逻辑串行成立，物理共享单流尚未成立。 |
| 所有 activation/KV/workspace 由统一 arena 管理，运行期零分配 | [AOTI workspace](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/backends/aoti_region_executable.cpp:378)、[AOTI Run](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/backends/aoti_region_executable.cpp:393) | 部分符合。AOTI 明确自行管理 workspace，每次 Run 建立 vector/Tensor 包装并复制输出。当前 arena 不能证明整个部署栈运行期零堆分配。 |
| 静态指令 KV 离线计算并跨相机帧复用 | [SmolVLA prefix](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/adapters/smolvla_frontend.py:98)、[cache 依赖测试](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/tests/models/test_smolvla_artifact.py:26) | 现有 exact cache 依赖图像、state、tokens、mask，不是独立的离线文本 KV。必须证明被缓存量不依赖新图像/state/位置才可跨帧复用。 |
| 摄像头到 action buffer 的零拷贝流水线 | [TensorView](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/include/vlaforge/runtime/region_executable.h:55)、[README 范围](/home/zhangzimo/Repos/private/edge-fm-x/README.md:19) | 当前是借用 tensor 存储的模型 Interface，没有完整相机缓冲导入、设备一致性、执行缓冲交接证据。应在外部 I/O Adapter 实现，不把传感器/控制调度塞进 core。 |
| encoder 高精度、action head 激进低精度、逐步校准 | 当前 frontend/compiler/后端构建链未找到 step-indexed calibration/precision-plan 实现 | 不是已有功能改名，需要校准数据、量化策略、后端 lowering 和数值验收闭环。 |
| Orin 与 Horizon BPU 同一部署框架 | [后端构建项](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/CMakeLists.txt:16)、[Orin 报告](/home/zhangzimo/Repos/private/edge-fm-x/doc/reports/vlaforge_orin_validation.md:3) | 当前有 AOTI/TorchScript/TensorRT，无当前分支 BPU provider。Orin 记录是模拟 ARM64 编译及 CPU 执行，不是实机 VLA 性能验证。 |
| Agent 搜索 kernel，验证后沉淀 skill library | 当前分支未包含旧 EdgeFM 自定义算子树；[README 明确说明](/home/zhangzimo/Repos/private/edge-fm-x/README.md:38) | 需接回独立离线工具链。旧分支有资产，不应判断为全部从零开发。 |
| 生成式 VLM 是去掉 action head 后的单次前向 | 当前分支并没有通用 Qwen3.5 部署路径 | 论文表述不完整: 生成式 VLM 仍有 text autoregressive decode、KV 或 recurrent state，不能只测 encoder。 |

### 可以回收的旧分支资产

只读检查了本地 `coda_kernel` @ `18f5c2f1aff53f35fe43a30c27ee07e19fb2ead9`。该分支包含:

- `.codex/skills/edge-fm-cuda-kernel-optimizer/scripts/orchestrate.py`
- `benchmark.py`、`preflight.py`、`profile_ncu.py`、`run_iteration.py`、`state.py`、`summarize.py`
- `.codex/skills/edge-fm-benchmark-report/scripts/report_qwen_3way_cuda_graph_vs_trt.py`
- 旧 EdgeFM 算子与模型实现。

这些是 **其他分支的可复用工程资产**，不是当前 VLAForge 的已集成能力，也不是新论文实验已完成的证据。推荐挑选必要模块接入 Region provider 和离线工具，不整体恢复旧 engine，不把两套 runtime 叠加在一起。

## 3. 模型覆盖与指标缺口

下表的“已有”主要依据仓库中的报告，并非本轮重新执行真实 checkpoint。

| 目标模型 | 当前分支真实覆盖 | 新主表仍缺什么 |
|---|---|---|
| SmolVLA | Host-CUDA real L2/L3/L4，已有正式统计矩阵 | Orin 实机、BPU 编译运行、真实观测轨迹、完整 action chunk 数值核验、新 runtime/量化/Agent 消融 |
| pi0 | [仅 L1 fixture，明确不声称预训练权重支持](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/adapters/pi0/pi0.py:1) | 真实 checkpoint、source parity、capture、artifact、C++、两类边缘设备 |
| pi0.5 | 未发现独立真实部署链 | 同上；不能用 pi0 fixture 代替。需要确认版本、state/token 表达和对应配置 |
| RDT-1B | 未发现真实 Adapter/部署报告 | 文本/图像编码器、conditioning、diffusion scheduler、完整动作头和两类边缘设备 |
| CogACT | 未发现真实 Adapter/部署报告 | VLM、cognition feature、DiT、CFG/DDIM、反归一化、完整动作 chunk 和两类边缘设备 |
| Qwen3.5 0.8B / 2B | 当前分支没有对应完整部署报告 | vision/prefill/decode、混合注意力状态、真实输入、TTFT/生成吞吐和目标设备结果 |

其他已有模型不能自动抵扣这五个 VLA: OpenVLA 的 weight-paged L4 被明确标为 correctness audit，不是性能基准；DiffusionDrive/MindDrive 可保留为补充泛化证据。依据 [模型矩阵](/home/zhangzimo/Repos/private/edge-fm-x/doc/vlaforge_model_adaptation_matrix.md:78)。

**现有可引用数字，但只能按原硬件和测量范围引用:** [RTX 3060 Host-CUDA 矩阵](/home/zhangzimo/Repos/private/edge-fm-x/doc/reports/vlaforge_cuda_matrix_v01/README.md) 中，SmolVLA baseline 的 eager/direct/generated 均值为 113.198/45.466/45.752 ms；DiffusionDrive 为 19.451/16.663/16.688 ms。这些不是 H100、H20、Orin 或 BPU 成绩，也不是本文新增低比特/Agent 方法的成绩。

当前 SmolVLA 的 Session 是单动作输出并自带动作队列。**旧正式基准已经排除了队列快路径**: [benchmark 定义](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/tools/benchmark_cuda_paper_matrix.py:1)，[full 模式每轮 ResetEpisode](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/tools/benchmark_generated_l4.py:412)。因此不是已有主表误算队列，而是新主表建议增加明确的 `generate_action_chunk` 入口，直接测 fresh chunk，避免通过 episode reset 强制刷新，并对齐完整 chunk 输出而非只检查首个动作。

### 模型规格先锁定

- CogACT 的“3B”应核对具体 checkpoint。官方论文按约 7B 规模与 OpenVLA 比较；官方仓库区分 Small/Base/Large 的 DiT 变体。不能把 action head 的大小、主干大小和整套模型混为一谈。[CogACT 论文](https://arxiv.org/abs/2411.19650)、[官方实现](https://github.com/microsoft/CogACT)
- RDT 的官方部署还涉及 T5-XXL 和 SigLIP。“RDT-1B”的名字不能直接当成完整端到端 pipeline 的驻留参数量。固定任务文本可以另报预计算模式，但完整输入模式不能悄悄省略文本编码器。[RDT 官方实现](https://github.com/thu-ml/RoboticsDiffusionTransformer)
- Qwen3.5 0.8B/2B 使用包含 Gated DeltaNet 的混合结构，不能当成旧 Qwen 的换权重实验。需要相应 recurrent/conv state 与常规 attention KV 管理。[0.8B 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[2B 模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)
- SmolVLA、pi0、pi0.5 的最终参数量同样以锁定 checkpoint 实际统计为准。主表同时记录总参数、在线活跃参数、驻留权重字节数，避免名义参数量误导内存可部署性。

## 4. 去噪叙述: 需要修改的核心问题

### 4.1 统一表达可以保留，但不能把不同生成机制写成同一个速度更新

PDF 第 3 页式 (2) 使用 `A_next = A + v(A, KV)`，缺少显式 timestep 和积分步长，并把 diffusion/flow/AR token 都归入该式。

建议先使用通用“有界迭代程序”:

$$
C_t = f_{enc}(O_t), \qquad (X^{n+1}, Z^{n+1}) = F_{\theta,n}(X^n, Z^n; C_t, \tau_n, \xi_n).
$$

其中 `O_t` 应包含模型实际需要的图像、指令、proprio/state 和掩码；`C_t` 是 conditioning tensors，不强制只叫 KV；`Z` 可表示 scheduler 历史或 decode state；确定性 solver 不需要随机量 `xi_n`。

对 Euler flow 再特化为:

$$
X^{n+1} = X^n + \Delta\tau_n v_\theta(X^n, C_t, \tau_n).
$$

当前 SmolVLA 真实实现已经显式传 timestep 并做 `sample - velocity / num_steps`，这一点比论文简式更准确。官方 OpenPI 也在循环外先填 prefix KV，再以 `dt=-1/N` 积分。故“编码从 N 次变一次”不能直接当作相对官方实现的新收益；新增价值应落在显式分区、跨后端实现、回放和内存管理。[SmolVLA 代码](/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/python/vlaforge/adapters/smolvla_frontend.py:209)、[OpenPI 源码](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0.py)

RDT/CogACT 要保留各自 scheduler/CFG 语义。AR token 模型应另写 token/state 更新，不能把 token 数和 diffusion step 数都解释为相同的速度场积分。N 是锁定部署 profile 的配置，不宜统称为“模型架构天然固定”；动态停止须定义静态上界与 mask，或明确不进入该回放 profile。

### 4.2 激进低比特与“严格无损”存在直接冲突

PDF 第 4 页提出 aggressive low precision 和逐步骤量化；第 6 页又说没有 lossy quantization，并用 MSE/余弦相似度推导 task behavior mathematically preserved。

建议拆成两个验收通道:

| 通道 | 可以声称 | 需要的验收 |
|---|---|---|
| 同精度结构/执行优化 | 在声明容差内与参考实现数值一致 | 相同权重、输入、noise、N、scheduler、CFG；离散输出逐项一致；连续输出的 MSE/max-abs/相对误差/余弦 |
| 低比特部署 | 在指定数据和误差预算下保持高 action fidelity | 逐层/逐步校准、held-out 数据、逐步误差及最终 chunk 误差；若声称任务性能不降，还需任务评测 |

余弦相似度不是无损证明: `cos(a, 2a)=1`，动作幅值却翻倍。MSE 很小也不能自动保证 gripper 阈值、闭环轨迹或成功率不变。若暂不做仿真/真机任务成功率，论文可以聚焦部署数值保真度，删除“任务行为数学上完全保持/zero accuracy degradation”的强结论。

### 4.3 逐步校准必须有可执行定义

式 (5) 的“每步一个 scale”还不够定义可复现实验。至少记录:

- checkpoint、输入规范化、N、时间网格、scheduler/solver、CFG、校准集/验证集划分。
- 每层/算子、每步或 step-group 的 activation dtype、位宽、scale、zero-point、粒度及 clipping 规则；权重量化单独定义。
- FP reference 轨迹上同输入的逐步算子误差，以及量化模型 free-running 后累积的整段轨迹误差。
- 全局尺度、分层但跨步共享尺度、逐步/分组尺度的同位宽对照；报告饱和率、误差、时延、内存、artifact 大小。
- BPU 静态量化图是否允许复用权重和不同 step scales，要用实际 OE 编译验证。若生成 N 份图，不能忽略权重/编译产物膨胀。

推荐把 `PrecisionPlan` 放在离线部署工具中，生成 backend artifact 和 manifest，不先向通用 IR 添加大量模型专属量化 opcode。

### 4.4 其他需要限制适用条件的表述

- 多项式近似 GELU 与 erf 形式不是天然严格相同；若参考实现使用近似版本，换成 erf 是数值实现变更，需要验证，不能写成无条件数学等价。
- 将 attention 的负无穷替换为有限值，必须验证掩码语义、全 masked 行和量化动态范围；不能只看“编译通过”。
- 固定文本 embedding 可缓存，不代表混合了当前图像的文本 KV 可缓存。缓存有效性必须与 attention mask、位置和全部依赖一致。
- `N*C_act/(C_enc+N*C_act)` 只说明在该模型成本假设下随 N 增长，不证明实际部署中 action head 一定占主导。需要逐模型实测 encoder/head/copy/launch 时间。
- 高精度 encoder 只运行一次，也可能占主要时延或大量驻留权重，不能直接写“影响可忽略”。
- 单次图 replay 减少的是重复的 host dispatch 开销，不把所有 GPU kernel 工作合并成一次常数成本。头部内部 kernel 数、依赖和设备执行成本仍存在。
- 将“on-chip memory”与 SoC 共享 DRAM、GPU device memory 区分。阶段拆分不自动减少常驻权重；若需要卸载/换入，代价必须计入 E2E，并会限制全图回放。
- 同步单流、无 Python 和 arena 有利于稳定时延，但不足以保证硬实时 deadline。应表述为在指定硬件/负载/功耗下实测的尾时延与 deadline miss rate。
- 论文一处声称固定静态计划，另一处说 dynamically orchestrated。建议明确为“离线选择后端与分区，初始化绑定固定执行计划，运行期按数据依赖执行”，避免暗示每周期重新调度。
- Agent 搜索式 (6) 只列了 memory 约束，但搜索空间还包括 precision。需要显式加入数值误差/输出契约约束，否则最快的候选可能不满足式 (3) 的 C3。

## 5. 推荐的代码改动范围

| 模块 | 保留/新增设计 | 相对工作量 |
|---|---|---|
| Invocation IR / Plan / Session | 保留输入版本、state/cache 分类、有界控制流、事务和输出契约 | 小，优先不增加 core opcode |
| 模型 Adapter | 统一 fresh action-chunk 工作流；补真实 pi0/pi0.5/RDT/CogACT；分别保留 scheduler 和预处理 | 中到大，模型数量驱动 |
| Execution backend | Session 共享 execution context；可捕获静态 action program；warmup 后固定地址；设备端 timestep/noise 输入；有声明的 fallback | 大，是当前执行层最关键改造 |
| Region Interface | 版本化扩展 capability/context/workspace 外部绑定能力，旧 provider 继续走现有路径 | 中，跨后端回归风险较高 |
| 数值工具链 | PrecisionPlan、step calibration、误差预算、量化 artifact/provenance | 中到大；如果保留论文量化主张则必需 |
| Horizon provider | OE/HBM 装载、输入布局/对齐、外部内存生命周期、cache flush/invalidate、任务完成和错误传播 | 大，需确切 BPU/SDK 验证 |
| Agent tooling / skill registry | 回收旧离线调优资产，连接真实 op specification、correctness gate、target profiler、产物注册 | 中到大；CUDA kernel 不能原样用于 BPU |
| Benchmark/report | 复用现有 raw samples、进程级统计、校验；补设备、E2E、CDF、动作误差和新消融 | 中 |

建议的对外调用仍保持小 Interface: **输入观测和显式噪声，返回已完成并验证的动作块**。Context 编码和 action 迭代是内部可替换模块。两个逻辑阶段可以对应多个物理 artifacts，不必强行限定为两个文件。

不能直接删掉所有 `synchronize`: 当前错误传播、输入借用期限和事务提交依赖完成语义。异步化后必须先完成必要的设备依赖和数值验证，再发布新输出；失败时保留上一 committed output，释放/复用缓冲前确保任务结束。

全局“无分配”难以马上覆盖所有 AOTI 第三方行为。可先对选定 capture-capable 后端建立可测的 steady-state 契约，其他 provider 明确不具备该 capability，不把 fallback 性能混入图回放结果。

BPU 的离线优化应使用 OE 允许的图分解、融合、layout、量化与编译配置，以及实际受支持的扩展机制；不能以 CUDA thread-block 搜索的成功声称 BPU 同样完成 Agent 自动 kernel 合成。

## 6. 必须补的实验

### P0: 主表与平台落地

目标是 **5 个 VLA x 2 类边缘平台** 的明确覆盖矩阵。每格状态用 passed / unsupported / OOM / pending，并附原因；只跑 BPU 子图而其余在 CPU 上运行时，必须标为异构 E2E，报告 CPU/BPU 时间与占比，不写成全 BPU。

主表至少包含:

`checkpoint | hardware/SDK | precision(enc/head) | N | chunk H | K_exec | E2E mean/p50/p95/p99/max ms | fresh chunks/s | peak resident memory | CPU fallback | fidelity gate`

统一边界为观测进入推理接口到完整动作块可供调用方消费，显式列出预处理、数据搬运、encoder、action loop、反归一化/后处理。纯预处理后 tensor-to-tensor 可以另报，但不能与含完整输入处理的基线混比。摄像头采集等待不必加入模型 E2E，除非另做 sensor-to-action 实验。

固定 batch=1、模型原始相机数/分辨率、token 长度、N、CFG、noise、输出 H 和动作维度；同一模型的基线设置完全一致。不同模型不要求人为改成同一个 N/H，但要展示设置。

吞吐使用实际 fresh chunks/s。输出队列消费 Hz、一次生成 H 个 action 的摊销速率和新观测重规划 Hz 分开，不能互相替代。RDT 等固定任务文本缓存模式与完整文本输入模式也分行报告。

### P0: 连续时延 CDF 与长稳

- 建议预热后至少 1,000 次连续 fresh-chunk 推理，保存逐次原始时延；至少 5 个独立进程给出进程级统计区间。
- 输出 CDF、尾部 CCDF、p50/p95/p99/max/std，以及给定 `K_exec * delta_t` 的 miss rate。不同 N/H/功耗模式分开。
- 覆盖连续变化的真实观测；same-revision cache-hit soak 另列，不能替代该实验。
- Orin 记录实际功耗模式、频率、温度、内存和 throttling，并做持续热稳态测试。不能把刚开机短跑称为稳定实时。
- 若 1,000 次无超时，仍不能证明任意小 epsilon 或最坏时延硬保证。在独立同分布近似下，零失败的 95% 上界约为 3/n；相关连续样本需要更谨慎解释。

### P0: 动作保真度

同 checkpoint、预处理、反归一化、scheduler、N、CFG 和显式保存的初始 noise；只设同 seed 不一定能跨后端得到相同噪声。

输出全 chunk 的 cosine、MSE/RMSE、max-abs、分动作维度误差、分位数和最坏样本。原始归一化空间与物理单位都检查；gripper/离散 token 单独核对，近零向量余弦需有定义。使用与 calibration 不重叠的真实观测/任务；现有人工 linspace fixture 不能充当泛化保真度主证据。

### P0: Agent 单算子消融

从五类 VLA 的实际 trace 抽取 shape、dtype、layout 和调用次数，覆盖:

`Attention Core | GEMM | LayerNorm/RMSNorm | Embedding | RoPE | VLA-specific operators`

VLA-specific 可包括 timestep embedding、AdaLN/调制、action projection、cross-attention、solver update；以真实模型热点为准。

同平台比较现有默认实现、Agent 优化后实现，另列 native/vendor 强基线。保存 correctness、warmup、计时边界、launch 配置和 raw samples。只对“默认 unfused baseline”更快不足以证明超越强 vendor baseline。还需把选中 kernel 放回完整模型测一次 E2E，避免单算子收益被 layout/copy/调度成本抵消。

### P0: 新叙事直接要求的补充消融

仅补主表与算子表还不够支持方法章节，应增加:

1. **阶段时间分解:** encoder、N 次 head、copy/launch/sync；验证不同模型上的去噪占比。
2. **执行层消融:** 同一 artifact 的现有逐 Region 执行、共享流、整段 replay；arena 和零拷贝收益按能力单独开关。保持数值和模型设置不变。
3. **量化消融:** 若保留量化主张，对比同精度基线、encoder 高精度/head 低精度，以及全局/分层/逐步尺度；固定 N，报告累积误差和内存。
4. **Agent 与库复用:** 固定候选/时间预算，比较无调优、无历史 skill 的调优、带 skill 的调优；报告耗时、尝试数、有效候选数、失败数、人工参与和 held-out 新模型的复用率。

“weeks to minutes”需要实际 onboarding 流程和计时定义支撑，不能由 kernel speedup 推导。没有可比的人工基线时，改报可复现的绝对时间及复用节省，而非虚设人力对照。

### P1: Qwen3.5 0.8B / 2B

作为 VLM 泛化补充，测真实 image+text 输入的 TTFT、固定生成长度的 tokens/s、总时延、内存、greedy 输出/数值保真度。必须包含 vision 和 decode，不只跑 text-only 或 prefill。

Qwen3.5 混合状态适配可能是独立的中等规模任务，不建议挤占第一个 VLA 在 Orin/BPU 上闭合的优先级。旧 VLM 数据可按真实版本保留，不能把旧 Qwen 结果重标为 3.5。

### 基线要按当前可用能力选择

TensorRT-Edge-LLM 与 TensorRT-LLM 不是同一仓库。当前官方 Edge-LLM 文档已有 action 模型和 Qwen3.5 0.8B/2B 条目，不能沿用“vendor 只支持纯文本”的笼统叙述。具体五个 VLA 是否被直接支持，要逐 checkpoint 核验，不支持就明确 N/A；可另用标准 TensorRT 构建可比较基线，但应正确标名。[官方 Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM)、[支持模型列表](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)

Orin 精度和 SDK 不能照搬 H100/H20。选择板端实际 SDK 对应的 baseline release，核验支持矩阵；不能把 Hopper 低精度性能当成 Orin 的可实现能力。[官方平台矩阵](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html)

## 7. 论文现有数字与文稿修订清单

PDF 第 5-6 页的方法对象、实验对象和表格尚未统一:

- 新列表写 SmolVLA/RDT/pi0/CogACT，用户又明确包含 pi0.5；主表应统一成五个指定对象。
- Table I 实际为 Horizon 上 Octo/OpenVLA，而正文“Table 1”描述 Orin/OpenVLA；后文又指向“Table 2”。
- “Fig. 1 是时延 CDF”与当前 Agent 流程图不对应；仍有 Fig. ?? / Section ??。
- VLM 正文指向 Table 3，但当前 PDF 没有对应完整表格；Qwen2.5-VL 的“2B”也需核对是否实际上为 Qwen3-VL-2B 或自定义版本。
- 当前本地 Markdown/CSV/JSON 检索未找到能够对应 PDF 中 Orin 126.4 ms、BPU 38.4/148.2 ms、1.49x、25.6 tokens/s、cosine > 0.9999 的完整 provenance 链。应暂标“待原始日志核验”，不能据此认定已经复现，也不能直接断言这些旧数字无效。
- “first documented Horizon VLA”还需要单独的先前工作检索与范围限定；本次架构审计不证明优先权。可以先写可验证的部署事实。

旧结果若复用，应绑定旧 source SHA、hardware、checkpoint、precision、workload 和 raw logs。旧 kernel 的原平台数据可作为历史对照；新 runtime 的 E2E/CDF/量化结论必须重新实测。

## 8. H100/H20 与周一 Orin 的执行安排

这是后续建议，没有把同步/运行写成已经完成。

| 资源 | 建议用途 | 不能替代什么 |
|---|---|---|
| zzm-h100-x8 | 真权重 eager reference、捕获、CUDA artifact parity、校准数据/逐步误差、模型适配 | Orin/BPU 延迟、功耗和尾时延结论 |
| zzm-h20-x8-1 | 第二种 CUDA 设备的功能复验、批量准备、兼容的 OE 离线编译环境 | BPU 实机执行、Orin 专用 kernel 调优 |
| 周一 Orin 台架 | 先闭合 SmolVLA，再扩到 pi0/pi0.5/RDT/CogACT；正式板端 profiling 和基准 | 仅交叉编译不算该阶段通过 |
| 待确认 BPU 台架 | 先用最小真实模型验证 HBM/内存/同步/数值，再扩覆盖 | 只有 HBM 文件或子图 checker 不能算完整 VLA E2E |

代码来源按用户指定固定为本地 workspace。后续 rsync 到:

- H20: `zzm-h20-x8-1:/xs-train-nas/zzm/repos/<isolated-run-directory>/`
- H100: `zzm-h100-x8:/mnt/data/LingXiTeam/workspace/zzm/repos/<isolated-run-directory>/`

同步前预览差异，隔离源码/环境/assets/results，记录本地 commit、dirty patch 和 source file hashes；不覆盖远端已有工作，不默认用 `--delete`。远端仅执行，本地产生代码修订。CUDA/TRT artifacts 按 GPU/SDK 分开构建，不把 H100/H20 的二进制直接搬到 Orin。

推荐执行顺序:

1. 锁定论文数值口径、五个 checkpoint、真实数据集和 BPU SKU/OE 版本。
2. 用 SmolVLA 完成 fresh chunk + 完整数值输出 + 统一测量入口，在现有 CUDA 上形成可复现参考。
3. 完成 CUDA 执行层共享 context/replay，并在 Orin 做同精度闭合；保持现有串行路径作为 correctness 对照。
4. 优先验证 BPU provider 和第一个真实模型，尽早发现 unsupported operator、内存和精度风险；不要等五个 CUDA 模型全部完成才开始。
5. 扩真实模型覆盖，接入 Agent 调优；如果论文保留量化贡献，补 step calibration 和误差消融。
6. 收集两类边缘平台主表/CDF/算子表，再补 Qwen3.5 泛化和统一论文表述。

周一的合理首个交付是“一个真实 VLA 在 Orin 上完整可测”，不是在未锁定配置和后端前承诺五模型双平台全部完成。

## 9. 本轮验证记录

结构探索使用现有 CodeGraph 索引，并对关键源码、历史分支资产和论文正文逐项读取。未运行 `codegraph init`，未切换当前分支，未覆盖用户论文。

执行:

```bash
/home/zhangzimo/.venvs/vlaforge-validation-py313-20260906/bin/python -m pytest -q \
  vlaforge/tests/models/test_smolvla_artifact.py \
  vlaforge/tests/models/test_model_fixtures.py \
  vlaforge/tests/unit/test_generated_l4_benchmark.py \
  vlaforge/tests/unit/test_cuda_paper_matrix.py \
  vlaforge/tests/plan/test_memory.py
```

结果: **59 passed, 1 warning in 0.98s**。警告为当前 CPU 测试环境未安装 NumPy；上述测试通过。这不是完整测试套件，也不执行真实 checkpoint、CUDA/HBM 或 Orin/BPU。

本轮新增文件只有本评估文档；用户已有 PDF 保持不动。此前 AutoVLA 补全阶段的下载/安装已暂停，没有把其未完成状态计入本次完成情况。

## 10. 通用 Python 接口补充实现

本节记录用户确认后追加的代码修改，时间为 2026-09-06。第 1-9 节保留修改前的评估快照；本节不能替代其中尚未完成的平台、真实模型和论文实验要求。论文 PDF 未修改。

本轮沿用 Python -> Invocation IR -> Plan -> C++ Session，不新增模型专用 opcode，也不按模型名选择框架实现。

| 层次 | 本轮实现 | 模型 Adapter 的责任 |
|---|---|---|
| Python 接口 | `InvocationBuilder` 自动生成 SSA、输入/状态读取、校验、事务与命名输出 | 声明类型、纯计算 Region 和状态语义 |
| Prefill + 多次生成 | 通用 `call` / `iterate`，异构多值 carry、嵌套循环、同步更新 | 提供 prefill / step 计算和固定步数 profile |
| 内存生命周期 | Plan 规划 carry scratch，固定循环外上下文和循环状态的存活区间 | 不手工分配/复用 C++ buffer |
| 上下文状态 | 推导传递输入/状态依赖；区分显式版本、自动版本和默认值；复用既有状态提交/回滚/重置 | 声明哪些值跨 Run 持久化，初始化状态；需要跨 Run 复用时提供正确输入版本 |
| 模型接入 | 连续生成/自回归两类示例；现有 pi0 L1 fixture 迁移到公共接口并保留原输出 | 替换纯计算实现，不修改公共调度代码 |

接口、可运行示例及限制见 [Generic Python Invocation Interface](../vlaforge/spec/python-invocation.md)。该接口是受限的 Python 符号编排，不是任意 Python 自动翻译；TensorRegion 捕获/编译继续使用已有 `torch.export` 和后端 artifact 链。

验证命令在 `vlaforge/` 下运行：

```bash
/home/zhangzimo/.venvs/vlaforge-validation-py313-20260906/bin/python -m pytest -q -ra \
  --junitxml=build-generic-frontend/python-tests.xml
cmake -S . -B build-generic-frontend -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-generic-frontend --parallel 4
ctest --test-dir build-generic-frontend --output-on-failure
```

结果：**323 passed, 11 skipped**；C/C++ runtime **8/8 passed**。新增 22 项 frontend 测试和 1 项 pi0 迁移回归，覆盖 IR/Plan 一致性、序列化、异构/嵌套循环、别名交换、上下文生命周期、三类输入身份、状态回滚/重置、`torch.export` 捕获以及实际编译执行的无 Python CPU Session。C++ fixture 使用测试内的确定性算子，不把它计作真实模型 artifact 或 GPU 数值验证。

11 个跳过项是显式开启的 CUDA AOTI / 真实 checkpoint 测试。此次未同步远端、未运行 H100/H20/Orin/BPU，也未生成论文性能表。整段 CUDA Graph、多设备统一内存、AOTI 内部 workspace/权重接管、BPU provider、逐去噪步量化与真实模型基准仍属于后续工作。
