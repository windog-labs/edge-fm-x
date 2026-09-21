# Goal Prompt: EdgeFM 新论文适配与 VLA 部署验证

本文件正文可作为长期 Goal 的执行 prompt。保存本文件不等于已经启动 Goal。

## 目标与范围

以 `/home/zhangzimo/Repos/private/edge-fm-x/doc/ICRA_2027_EdgeFM.pdf` 为新版论文依据，持续推进当前本地分支的代码设计、真实模型适配、部署验证与实验工具，使其符合面向 VLA 部署的新叙事，并生成可复现、可追溯的论文证据。

当前 Goal 优先完成本机、H100、H20 上能够实现和验证的工作。需要 Orin、地平线 BPU 或其他尚不可用硬件的实机任务可以延期，但不能因此暂停通用接口、CUDA 实现、模型捕获、数值验证、校准工具和实验入口的开发。不要停在评估、计划、接口空壳或仅 fixture 通过；每轮持续推进实现、运行和证据归档。

分别跟踪“当前可用硬件阶段完成”和“整篇论文最终验收完成”。前者不代表后者，H100/H20 结果不能填入 Orin/BPU 主表。

## 输入与执行环境

- 本地代码唯一来源：`/home/zhangzimo/Repos/private/edge-fm-x`。以启动和恢复时的本地 HEAD、未提交改动和最新论文为准，保留用户改动。
- 初始设计评估：`/home/zhangzimo/Repos/private/edge-fm-x/doc/ICRA_2027_EdgeFM_design_gap_assessment_20260906.md`。第 1-9 节为修改前快照，第 10 节为第一批通用接口实现；必须结合当前代码复核。
- 已补接口说明：`/home/zhangzimo/Repos/private/edge-fm-x/vlaforge/spec/python-invocation.md`。已记录的 CPU 基线是 323 项通过、11 项跳过，C/C++ runtime 8 项通过；这些不是本 Goal 新运行的结果，也不是 GPU/真实模型部署证据。
- H20：`zzm-h20-x8-1`，源码同步到 `/xs-train-nas/zzm/repos/` 下独立运行目录。
- 另可使用 `zzm-h20-x8-2`，与 H20-1 共用上述 NAS 挂载；源码可共用，构建、缓存和结果按主机/任务隔离，避免并发写冲突。
- H100：`zzm-h100-x8`，源码同步到 `/mnt/data/LingXiTeam/workspace/zzm/repos/` 下独立运行目录。
- 另可使用 `zzm-h100-x4`（2026-09-08 用户追加授权），同步根目录相同；按主机/任务隔离运行目录，不假定两台 H100 的文件系统互不共享。
- 先检查 SSH、GPU 占用、显存、磁盘、驱动和 SDK。不得终止或挤占其他人的任务，不假定任一机器一直空闲；有可用资源就继续其他独立任务。
- 用 rsync 预览后同步，不默认使用 `--delete`，不覆盖远端已有工作。远端主要负责执行，代码修订回到本地源；源码、环境、权重和结果分目录管理。
- 隔离依赖环境；优先复用已授权的本地/远端权重，必要时从官方公开来源获取。记录许可证和来源，不绕过 gated 权重授权，不自动购买资源或修改凭据。
- 目标 capability、编译器和 artifact 兼容性实测确认；按目标构建，不假定 H100/H20/Orin 的二进制可以互换。

## 一、通用架构与论文方法适配（P0）

1. 保留 Python -> Invocation IR -> Plan -> C++ Session 主链。纯 TensorRegion 通过已有捕获/后端 artifact 链接入；不要恢复第二套旧 runtime，也不要把模型专用逻辑塞入公共 IR、内存管理或调度器。
2. 对外提供小而完整的通用 Interface。模型 Adapter 只声明纯计算、类型/形状、状态语义和必要的 profile；模型作者不需要自己写 C++ 内存分配、缓存失效、循环调度或事务管理。扩展核心能力时至少用两种不同生成范式验证复用。
3. 通用内存设计覆盖 activation、派生 context、循环 carry、持久状态和 backend workspace 的声明、生命周期、复用与内存统计。区分权重、Plan 可控内存和第三方后端内部内存；没有覆盖实测前，不宣称整个栈零分配、全零拷贝或所有权重由 arena 接管。
4. 通用执行采用 context/prefill + 有界迭代生成。condition 可以是多个 tensor，不强制都是 KV；支持异构多值 carry。逻辑两阶段可以对应多个物理 artifact，不为匹配论文图示强制生成两个文件。
5. 通用状态管理覆盖 snapshot、版本、reset、derived cache、loop-local state、提交和回滚；正确区分显式输入版本、自动版本和可选默认值。缓存只依赖真实传递依赖，图像/state/mask/位置变化时正确失效，不能把混合视觉信息的前缀当成静态文本缓存。
6. 实现并验证 Session 共享 execution context/stream、稳定地址、warmup、可捕获的完整 N 步动作程序及 replay。只生成 C++ for、只捕获单个 Region 或每步重新捕获，不算整段 replay 完成。
7. 后端 capability、workspace/stream 绑定和 fallback 使用通用、可版本化接口。异步执行必须同时保证依赖、错误传播、借用输入期限和输出提交完成语义，不能仅删除 synchronize 来制造速度提升。
8. 对选定支持的 CUDA 路径实测运行期分配、内存高水位、长稳行为、跨 Session 隔离、失败/重试/reset。普通执行和 replay 都保留数值对照，fallback 单独标记，不混入 replay 成绩。
9. 对照论文 III-A 至 III-D 建立“主张 -> 模块 -> 测试 -> 实验 -> 当前限制”映射；输出文稿修改建议，不覆盖用户 PDF，不为匹配文稿伪造功能或结果。

## 二、真实 VLA 适配（P0）

目标模型必须逐项跟踪：SmolVLA、RDT-1B、pi0、pi0.5、CogACT。模型名称中的规模及用户给出的名义参数量只是待核对信息；锁定官方 checkpoint/revision、完整 pipeline 参数量、在线活跃参数和实际驻留权重，不混淆 action head 与全模型。

每个模型推进官方参考/eager -> TensorRegion 捕获与对齐 -> 真实编译 artifact -> 生成无 Python C++ Session -> 完整 action chunk 数值核验 -> 性能实验。每层分别记录结果和失败原因；pi0 fixture 不能替代真实 pi0，更不能替代 pi0.5。

保持官方预处理、相机数/分辨率、指令/state 表达、mask、归一化/反归一化、scheduler、CFG、N、chunk 长度和动作维度。不得遗漏 RDT 等模型实际需要的编码器或 CogACT 完整主干。固定任务文本预计算允许作为显式独立模式，不能暗中从端到端基线删去。

先用已有真实覆盖最成熟的 SmolVLA 闭合公共链，再扩展其他四个模型；不得把五套独立部署脚本当作通用框架完成。模型专属差异放在 Adapter/后端扩展中；多个模型共享的实现放入公共模块。

## 三、去噪与低精度叙述（P0）

统一抽象是 conditioning + 有界迭代状态转移，不是把 diffusion、flow matching 和 AR 都写成同一 Euler 速度更新。显式保留 timestep、积分步长、scheduler 历史、CFG、噪声和 carry；AR 保留 token/KV/recurrent state 语义。N 是锁定的部署 profile，动态停止须有明确表示或明确不支持。

验证论文所说的 encoder/head 成本占比，不把官方实现本来已有的 prefix-once 当成新增优化收益。分别测编码、动作迭代、搬运、launch/sync 和后处理。

论文 III-B 的逐步低精度是独立待验证贡献，不能因为没有 BPU 就延期全部数值工具链。实现通用 PrecisionPlan/校准记录、每步或 step-group 激活统计、backend lowering 和误差预算；校准集与 held-out 验证集分离。

在可用 CUDA 上完成可支持的真实低精度 artifact 实验，并区分 fake-quant 数值模拟与真实低精度 kernel。对比同精度基线、encoder/head 非对称精度、同位宽全局/分层/逐步尺度；记录逐步误差、free-running 累积误差、饱和率、最终动作误差、时延、内存和 artifact 大小。某硬件或后端不支持时记录明确证据，继续可支持路径。

同精度执行优化和低比特部署分开验收。不得同时无条件声称“激进量化”和“严格无损”；对 GELU、mask、RoPE 等改写逐项验证，不能仅以编译成功证明等价。无法支撑的论文表述列入修订清单，不能通过事后放宽容差掩盖失败。

## 四、VLA 主表、CDF 与动作保真（P0）

最终主表目标是 5 个指定 VLA x Orin / 地平线 BPU 两类平台。当前先在 H100/H20 完成同一测量入口、协议和 CUDA 实测表，分别标明硬件；为板端保留独立单元格和一键执行入口，禁止代填数据。

固定 batch=1，逐模型声明输入 profile、精度、N、CFG、输出 horizon 和实际执行动作数。测量边界为观测进入约定推理接口到完整 fresh action chunk 可由调用者消费，明确预处理、搬运和后处理是否包含；设备时间与同步完成的 host wall time 分开报告。

主表包含 checkpoint、硬件/SDK、精度、N、chunk profile、E2E mean/p50/p95/p99/max、fresh chunks/s、峰值驻留内存、fallback、数值验收和证据链接。fresh chunks/s、队列取动作 Hz、摊销 actions/s 和重规划 Hz 不得混用。张量到张量、完整输入、静态文本缓存模式分行报告。

对比原生官方 PyTorch 与本平台真实可用的强 vendor baseline，锁定版本和等价工作负载；核验 TensorRT、TensorRT-LLM 与 TensorRT-Edge-LLM 的实际支持范围并正确标名。不支持填 N/A 和原因，不虚构 vendor baseline。

连续时延实验使用变化的真实观测和连续状态，默认预热后至少 1,000 次 fresh-chunk 推理、至少 5 个独立进程重复；确有成本限制时显式记录缩减配置和未达到正式统计门槛。保存逐次 raw samples，生成 CDF、尾部曲线、p50/p95/p99/max/std；存在声明的控制周期时报告 miss rate。cache-hit soak、队列快路径、每次 reset 的短测不能冒充连续推理。

动作核验使用同 checkpoint、完整预处理/后处理、N、scheduler、CFG，以及保存下来的同一份实际 noise tensor，而不只设置相同 seed。必须输出完整 chunk 的 cosine，同时保留 MSE/RMSE、max-abs、分维度与最坏样本；处理近零向量，核对 gripper/离散 token，在归一化空间和物理单位分别检查。

容差、数据集划分和计时协议在正式运行前固定。用户当前需要部署数值保真证据，不把机器人任务成功率或闭环机器人评测新增为当前硬性要求；没有这些证据时也不能声称任务行为数学上严格无损。

## 五、Agent 调优与算子消融（P0）

从真实 VLA trace 提取 Attention Core、GEMM、LayerNorm/RMSNorm、Embedding、RoPE 和 VLA 专用热点的 shape、dtype、layout、调用次数。专用热点按实际模型选择，如 timestep embedding、AdaLN、action projection、cross-attention、solver update；无该算子的模型标 N/A，不造负载。

同一平台、相同数值契约比较默认实现、Agent 调优后实现和可用的强 native/vendor baseline。保存 correctness gate、预热、计时边界、原始时延、launch 配置、编译结果和 profiler 证据。

建立离线候选生成 -> 编译 -> 数值检查 -> 微基准 -> 选择 -> skill 注册流程，优先按当前 Interface 回收旧分支可用资产，不整体恢复旧 engine。skill 按算子签名、适用范围和目标能力管理，不能只硬编码某模型或某张 GPU。

预先固定并记录搜索/时间预算、候选数、失败数、优化耗时、人工参与和复用命中；比较无调优、无历史 skill 与复用 skill 的路径。必须把选中算子装回至少代表性的真实完整模型重测 E2E，不能把孤立算子加速直接当成模型加速。

执行层另外保留同 artifact 的普通执行、共享流、整段 replay 以及可独立开关的内存优化消融。CUDA 调优结果不能当作 BPU 算子优化结果。

## 六、VLM 泛化补充（P1）

在不挤占核心 VLA 闭合的前提下，补 Qwen3.5 0.8B 和 2B 的真实 image+text 部署。锁定实际版本并按模型结构处理 attention KV、recurrent/conv state，复用公共接口，不为 Qwen 新建独立 runtime。

测完整 vision + prefill + text decode，报告 TTFT、固定生成长度的 tokens/s、总 E2E 时延、内存以及 greedy 输出/数值对齐。不能只测文本、encoder 或 prefill，也不能把旧 Qwen/VLM 数字重标成 Qwen3.5。P1 不得无限扩张为无关模型项目；无法完成时明确其可选优先级和具体原因。

## 七、延期硬件任务

延期的是必须依赖实机的验收，而不是所有涉及该平台的代码工作。现在准备通用 provider 契约、平台 manifest、输入/noise/reference 包、构建/运行/收集脚本、结果 schema 和板端检查清单；已有可用 SDK 时可做真实离线编译，但明确它不代表实机运行。

Orin/BPU 的最终 5 模型矩阵、板端 E2E/CDF/保真、同平台算子消融、功耗/温度/频率/热稳态数据保留独立延期任务。确认具体 BPU SKU、OE/SDK、可用内存、编译能力和设备访问后再执行；不能拿 mock provider、CPU 模拟、交叉编译成功、HBM 文件或子图 checker 代替完整板端 E2E。

异构 CPU/BPU 执行必须明确 fallback 和各阶段时间，不写成全 BPU。每个延期项记录依赖、设备/SDK、待运行命令和验收条件；硬件到位后可以直接续跑。

## 八、追踪、证据与完成标准

首次执行创建或更新 `/home/zhangzimo/Repos/private/edge-fm-x/doc/edgefm_vla_goal_status.md`，复用现有日志/报告工具；不要为跟踪而另造复杂管理系统。

使用稳定任务编号：G0 基线/论文映射，G1 通用运行时/内存/状态，G2 五个真实 VLA，G3 CUDA 主表/CDF/保真，G4 Agent/算子/执行消融，G5 逐步低精度，G6 Qwen 补充，D1 Orin 验收，D2 BPU 验收。

每项记录状态、最近已验证证据、剩余工作、阻塞原因、下一条命令。模型 x 平台 x 阶段分别记录 passed/running/failed/unsupported/OOM/pending；硬件延期使用 deferred-hardware。记录失败不等于把未完成的必需项判为通过。

结果至少绑定 source SHA、dirty patch/source hashes、checkpoint revision/hash、输入/noise hash、依赖锁定、硬件/驱动/SDK、实际 GPU/精度、完整命令、raw samples、日志和报告。长任务报告 PID、日志及退出状态，支持根据产物验证后的 resume，不能只凭 manifest 存在跳过失败阶段。

区分 historical-recorded、fixture、real eager/capture、real artifact、real no-Python C++ 与正式 benchmark 证据。历史数据只在来源完整且口径一致时复用，并保留原模型、硬件和运行时名称。开发态证据与冻结源码的正式论文结果分开。

表格和图从原始数据生成，保存 CSV/JSON、统计配置和绘图脚本；人工撰写的 Markdown 数字不能成为唯一证据。阶段进展简短报告“完成了什么、证据在哪里、还差什么、下一步做什么”，不能仅重复计划或把启动写成完成。

当前阶段的完成条件：G0-G5 在本地/可用 CUDA 范围内的必需实现与验证闭合，五个真实 VLA 各有可追溯部署/数值结果，CUDA 基准、CDF、算子与方法消融已生成，公共回归通过，G6 已完成或明确说明其可选任务处置，D1/D2 的待运行包和验收清单齐备。必需项若有外部授权/资源/依赖阻塞，保留为未完成并报告，不以“已列 TODO”代替完成。

全论文的最终完成条件：在当前阶段之上完成 D1/D2 实机验收，五个 VLA 的平台矩阵逐格有真实结果或可审查的不支持/OOM 结论；对于未支持单元格如实收敛论文覆盖主张，不能称五模型双平台全部成功。论文所有数字、方法表述和强结论均与可复现证据一致。只有当前阶段完成时，明确表述为“现有硬件阶段完成，边缘实机验收待补”。

持续自主推进可执行工作，适当并行独立子任务；只有缺失关键授权/资源、存在不可安全决定的接口契约或需要用户调整目标时才询问。不得擅自删减用户指定模型、降低数值门槛或把失败结果修饰为成功。
