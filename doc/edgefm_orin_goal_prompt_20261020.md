# EdgeFM Orin 补充实验 Goal Prompt

> 进度入口：[网页版任务看板](http://127.0.0.1:4317)及[Markdown 镜像](edgefm_orin_task_board.md)。
> 二者共用 `doc/edgefm_orin_task_board_state.json`，有进展更新此 JSON；本地服务同步生成 Markdown，网页自动刷新。
> 未运行服务时执行 `node tools/orin-task-board/sync.mjs`，不要把生成的 Markdown 作为另一份状态源。
> 用户于 2026-10-21 要求：
> 本任务每次有阶段进展、状态变化、新验证结果或失败/阻塞，都更新同一看板，
> 包括核验时间、当前位置、任务状态、下一步和证据。该同步要求在 Goal 恢复后继续生效。
> 下方旧规划仍是历史快照，不能据此恢复已取消的 Orin PyTorch 路线。

> Superseded execution route: the user changed the active Goal to TensorRT-only
> on 2026-10-20. Orin must not build/install/depend on Torch/torchvision or run
> PyTorch model baselines. Export/reference preparation stays on existing host
> environments. Compare same-target TRT conventional execution against
> EdgeFM + TRT optimized execution. The text below is the original planning
> snapshot, not authorization to resume its PyTorch preparation steps. Current
> execution and preserved stop evidence are tracked in
> [Orin Goal Status](edgefm_orin_goal_status.md).

> 最新验收澄清（用户于 2026-10-22 明确确认）：同一 Orin、同精度 / 输入 /
> noise 的常规 TRT 与优化路径，以完整标准化 action 的 MSE < 1e-5、
> cosine > 0.9999 为主门槛；native 误差及 H20 跨设备差异单列，不删除或改阈值。
> 历史 all-output failed 收据保持原样，以新收据绑定此决定。Runtime 对照固定
> engine 字节；图改写消融显式记录 engine 差异，其他计算与测量边界保持一致。
> 这只允许 Norm 通过主门槛后继续完整 E2E，不放行标准化 MSE 已失败的 GEMM。

> 最新排程调整（用户于 2026-10-22 要求尽量并行）：主机侧可独立并行推进
> pi0/pi0.5 与 CogACT 的真实资产定位、神经图导出和通用适配准备；不再等待
> 前一模型的 Orin 正式采样结束才做主机准备。Orin 正式采样仍独占 GPU，
> 目标构建/完整链验收保持串行；主机 GPU 导出先检查资源并协调。所有正式
> 样本数、数值门槛和目标计算边界不变，不新增其他平台性能实验。

生成日期：2026-10-20。下方代码块是可完整复制到 Goal 的执行指令。
本文件依据作者最新要求、当前论文 PDF、已有正式报告及本轮 Orin 只读检查编写。
本轮没有启动模型实验或安装软件，没有修改论文和既有测量结果。

## 当前起点

| 项目 | 2026-10-20 核验结果 |
|---|---|
| 代码 | 工作区 `/home/zhangzimo/.codex/worktrees/f801/edge-fm-x`，HEAD `aa5558fd33628a0e46e65706025ced3a25eb59c0`，detached HEAD；检查时工作区干净 |
| 论文 | `doc/ICRA_2027_EdgeFM.pdf`，SHA256 `92a3af4618a08c7ea66e62b928c3bb32d326e0445a491eee079ecccf70adf997` |
| 台架 | `ssh orin` 可连接，hostname `ugv`，aarch64，Jetson AGX Orin Developer Kit |
| 内存 | 系统报告总内存约 62,800 MiB，可用约 29,894 MiB；共享内存机器，不能把总容量当作当前可分配量 |
| 软件 | Ubuntu 20.04.6，L4T 35.4.1，CUDA Toolkit 11.4，`libnvinfer8 8.5.2-1+cuda11.4` |
| Python | 默认 Python 3.8.10；该解释器未发现 torch、transformers 或 TensorRT Python binding；尚未穷举其他环境 |
| 存储 | 根分区可用约 7 GiB；NVMe 挂载 `/data/datalog`，可用约 263 GiB；落盘前确认目录写权限 |
| 工具 | `/usr/local/cuda/bin/nvcc`、`/usr/src/tensorrt/bin/trtexec`、`tegrastats` 和 `jetson_clocks` 存在；默认 PATH 未覆盖全部工具 |
| 功耗 | `nvpmodel -q` 显示 MAXN，但部分 EMC 查询权限不足；不能据此称为已验证的 60W 配置 |
| 既有 Orin 资产 | 历史报告是 r36.4/arm64 仿真与 driverless 编译、fixture 正确性；现有交付脚本面向 TensorRT 10。与本台架的 TensorRT 8.5 兼容性尚未验证 |

这些是一次环境快照。后续开始实验时重新确认版本、资源、权限和实际进程。
TensorRT-Edge-LLM 的适用平台与版本须查阅并锁定其
[官方支持矩阵](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/support-matrix.md)
和[安装说明](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/docs/source/user_guide/getting_started/installation.md)，
不能根据普通 TensorRT 已安装推断 Edge-LLM 可以运行。

## 论文主张对应的缺口

| 原文位置 | Orin 需要提供的证据或修订 |
|---|---|
| 作者新要求、IV-B | SmolVLA、RDT、pi0、pi0.5、CogACT 五个独立模型的完整动作块 E2E 主表；作者新名单优先于草稿的四模型措辞 |
| 摘要、IV-B/IV-D | 官方 PyTorch 与实际可用 vendor/runtime 的同工作负载对比；重新检验 1.49x/1.32x 等加速数字 |
| III-A、IV-B | 连续 fresh-chunk CDF、尾延迟、std、deadline miss；runtime 分阶段、共享流和完整迭代 replay 消融 |
| III-A/III-C、IV-A | 真实驻留/峰值内存、KV/workspace、稳态分配、实际功耗模式、功耗/温度/频率与热稳态 |
| III-B、IV-E | 同精度完整动作核验；若保留非对称量化/逐步尺度主张，还需独立低精度证据 |
| III-D、IV-E/Table II | Orin 上真实模型签名的算子前后对比、Agent 成本与复用证据；至少一个代表模型的集成 E2E 复测 |
| 作者新要求、IV-D | Qwen3.5-0.8B/2B 图文输入的 TTFT、固定长度完整生成、tokens/s、内存及输出核验；旧 Qwen2.5-VL 数字不可改名复用 |
| III-A 的 zero-copy、全文强保证 | 只有真实实现和测量才支持相应主张；否则列出准确的可用表述，不增加物理机器人成功率或摄像头接入任务 |

旧 Octo/OpenVLA、Qwen2.5-VL、固定倍率和时延数字是文稿待核对内容，不作为
本轮必须复现的目标值。H20 与 BPU 证据继续复用，并保留各自原有范围：其中
J6M SmolVLA 正式交付为阶段级，不能自动改标为完整五模型 BPU E2E。

## 可复制的 Goal Prompt

```text
Goal：在 ssh orin 可访问的真实 NVIDIA Jetson AGX Orin 台架上，补齐 EdgeFM
论文的 Orin 部署实验证据，交付可复现的 VLA 主表、连续推理 CDF、完整动作
核验、同平台算子与 runtime 消融，以及优先级较低的 Qwen3.5 VLM 补充结果。

工作区：/home/zhangzimo/.codex/worktrees/f801/edge-fm-x
论文：doc/ICRA_2027_EdgeFM.pdf
任务设计与初始环境：doc/edgefm_orin_goal_prompt_20261020.md
目标机器：ssh orin；以实查 SKU、L4T、CUDA、TensorRT、内存和权限为准。
本轮只新增 Orin 证据，复用已完成 H20/BPU 的代码、输入和审计结果。
不启动其他平台的测评，不把其他平台的时延写入 Orin 表格。

一、开始前读取并冻结事实
1. 读取论文 III-A～III-D、IV-A～IV-E，建立逐条 claim -> 实验 -> 验收 ->
   证据路径清单；作者最新的五模型名单与 Qwen3.5 要求优先于草稿旧名单。
2. 读取 doc/reports/README.md、edgefm_h20_j6m_experiment_summary_20260914.md、
   h20_vla_experiment_delivery_20260910.md、h20_paper_scope_review_20260910.md、
   qwen35_native_formal_20260910.md、smolvla_j6m_performance_20260914.md，及
   doc/specs/board_target_handoff_v1.md、doc/reports/vlaforge_orin_validation.md。
3. 检查当前 HEAD、工作区和既有任务；复用终态验证通过的资产，保留原始报告
   与冻结源，不重复已完成的 H20/J6M 采样，不重新开展已完成的整理工作。
4. 本文件的环境快照只是起点。重新记录 board identity、OS/L4T、driver、SM、
   CUDA/cuDNN/TensorRT、Python/依赖、存储、可用内存、进程、设备访问和温度。

二、D1-00：Orin 环境与通用后端兼容性，P0
1. 当前台架是 L4T 35.4.1/CUDA 11.4/TensorRT 8.5.2，默认 Python 3.8；
   仓库 Python 要求 >=3.10，历史 Orin 脚本面向 r36.4/TensorRT 10。先检查
   已有虚拟环境/离线 SDK，选择实际兼容的 arm64 构建和运行路径。
2. 优先独立目录和可回退环境，在确认有写权限与空间的 NVMe 上存放大文件。
   不把几十 GB 模型装进仅约 7 GiB 空间的根分区，不擅自重刷系统、替换
   板载 CUDA/TensorRT 或终止其他服务。确需系统升级时给出实证依赖和方案。
3. 同时评估官方参考路径与 EdgeFM native Session 路径，补齐通用后端适配。
   先完成小型真实 CUDA/backend smoke、dtype/shape/I-O、同步与 ELF/DSO 检查。
   smoke、仿真通过和交叉编译成功均单独记载，不算模型或论文性能结果。
4. H20 的 SM90 二进制、x86 动态库和不同 TensorRT 版本 engine 不直接复用。
   权重/已验证处理后输入可复用；engine/artifact 必须绑定实际 Orin target。
5. Jetson 无 nvidia-smi/NVML 时使用 CUDA API、sysfs、tegrastats 和可获得的
   设备进程证据；无法证明资源可独占时等待，不用猜测的空闲状态开始正式采样。

三、D1-01：锁定模型和输入合同，P0
1. 必须逐一评估 SmolVLA、RDT-1B、pi0、pi0.5、CogACT，不自行缩减模型名单。
   顺序优先 SmolVLA，再 RDT、pi0、pi0.5，最后容量风险较高的 CogACT。
2. 锁定 checkpoint/revision/hash、实际参数量、processor/统计、图像与文本
   profile、action horizon/维度、N、scheduler、CFG、dtype 和每阶段放置。
   RDT 使用真实原调度器；pi0 与 pi0.5 分开测；不统一改成 Euler 更新。
3. CogACT 复用配置的完整 checkpoint 约 7.63B，并带公开依赖限定；不能写成
   已测名义 3B 或已验证原 gated Meta 配置。若换配置，作为新的明确对照项。
4. 复用已审计的图像/指令/state/noise/reference 包；完整记录采样覆盖与 episode
   数量。同 seed 不替代同一份实际 noise；不能将循环使用的样本当独立轨迹。

四、D1-02：完整 VLA E2E 主表，P0
1. 每模型至少包括官方 PyTorch 可用参考与 EdgeFM optimized 两行；另评估同
   平台可用的 TensorRT/Edge-LLM/vendor 路径。准确记录版本与支持范围，
   普通 TensorRT、TensorRT-LLM、TensorRT-Edge-LLM 不能互换名称。
2. 优先同精度 FP16 部署；保留上游必要的高精度累加/后处理并如实记录。
   不以全 FP32 回退完成主表，不默认启动大规模量化校准或反复精度搜索。
3. batch=1、单请求；从约定的 CPU 内存观测到完整 CPU action chunk 可消费，
   包含在线文本/视觉编码、所有动作迭代及必要预处理、搬运、后处理与同步。
   把文件解码、初始化、日志/质量比较和机器人通信明确列在边界之外。
4. complete original-input、host-model-tensor、resident-device 边界分别记录。
   主表报告实际完整 E2E；不能只测 action head、首步、prefix 或队列取动作，
   不能用各阶段均值之和或 SSH/文件链墙钟替代常驻完整 E2E。
5. 字段至少含 model/profile、backend、功耗模式、精度、N、chunk shape、
   mean/p50/p95/p99/max/std、fresh chunks/s、初始化耗时、峰值内存、
   fallback、完整输出状态与证据链接。actions/s、servo Hz 与 chunks/s 分开。

五、D1-03：正式连续采样与 CDF，P0
1. 正式运行前冻结协议：每模型/配置 5 个独立进程，每进程 128 warmup +
   1024 measured，顺序运行；每行保留 5120 个 measured raw latency samples。
   在 pilot 先估计耗时/内存；不能静默减样本、减少进程或只完成一个轻量模型。
2. 使用变化的真实观测与该 profile 的正常连续状态；每个 fresh chunk 都真正
   执行完整模型。只在新 episode/明确状态变化时 reset；保留输入修订与 cache
   命中/失效证据，固定输入 soak 和 action-queue 快路径不能替代这张 CDF。
3. 使用实际 GPU completion；host wall、CUDA device time、拷贝和同步按真实
   可测边界分列，不可将未暴露项写 0。CUDA event 不单独代表 host E2E。
4. 保存逐进程与 pooled 的 CDF/尾部曲线、raw CSV/JSON、PNG/PDF 和脚本。
   不剔除慢样本；外部资源争用使整进程失效，保留原始证据后补跑完整进程。
5. 报告 1000+ 连续周期上的经验抖动，不承诺 std<1.2ms、零抖动或硬实时。
   有明确 K_exec、动作周期和 deadline 时报告 miss rate；没有则标 N/A。

六、D1-04：完整动作输出核验，P0
1. 先在同一 Orin 比较同 checkpoint/输入/noise/数值合同的官方与 optimized
   完整 chunk，再将 H20 冻结输出作为补充参考，区分设备差异和执行路径差异。
2. 对全部实际输出保存原始数据、dtype、shape、hash；报告完整动作 cosine、
   MSE/RMSE、max/mean-abs、最坏样本和逐维误差，检查 NaN/Inf、gripper、
   离散索引/token。近零向量不给误导性的 cosine；归一化和 native 单位分列。
3. 正式运行前固定容差和参考集；按原文额外核对 MSE<1e-5、cosine>0.9999，
   不通过就保留失败，不事后放宽。byte exact 与 within-tolerance 分开报告；
   高 cosine 本身不证明量化无损、模型行为数学等价或机器人任务成功率。
4. 热点/单步一致性不替代完整 free-running 动作核验，warmup 输出也保留校验。

七、D1-05：Orin 同平台算子/Agent 消融，P0
1. 从真实模型 trace 提取 Attention、GEMM/Linear、LayerNorm、RMSNorm、
   Embedding、RoPE 及实际 VLA 专用热点的 shape/dtype/layout/调用次数。
   有效模型没有某类算子时标 N/A 并给定位依据；不制造形状凑表。
2. 同一 Orin、输入和数值合同比较官方/默认实现、Agent 候选与可用 vendor
   实现；记录至少 5 个独立进程对、预热、批内重复、同步边界和统计口径。
   小算子可 graph-batch 测量，但明确 per-process median/mean，不能冒充 E2E CDF。
3. 在调优前记录候选数量/时间预算、Agent/模型/规则版本、编译参数、失败数、
   搜索耗时、人工参与与 skill 命中。保留变慢或零收益结果，不挑最好一次。
4. 最优候选经完整 tensor 验证后，至少集成进一个代表性真实 VLA，并复测
   同边界 E2E 和动作输出。隔离编译器 recipe 选择、Agent 搜索和 runtime
   replay 收益；没有自主 kernel 合成或跨模型复用实证时不得按此命名。

八、D1-06：Runtime、内存与确定性消融，P0
1. 对可执行代表模型测同 artifact 的普通执行、共享流/batch-only、完整 N 步
   replay，以及可独立开关的 stage/context 复用、静态内存优化。
   验证实际 capture/replay/ordinary/fallback 计数，不以源码有 for 循环证明 replay。
2. 分解一次编码、全部动作迭代、搬运、launch/sync 和后处理；官方本来已有
   prefix-once 时不虚构重复编码基线或把它归为新收益。
3. 记录加载/首轮/稳态峰值、weights、KV/conv/recurrent、workspace、host RSS、
   CUDA allocated/reserved 和可观察的运行期 allocation/retry/OOM；解释不同
   计数包含范围。Jetson CPU/GPU 共享 DRAM，不能重复相加作为总物理占用。
4. 做同进程重复调用、多次 create/destroy、reset/失败重试、多 Session 隔离
   和内存高水位检查。先用短稳态 probe，再完成至少 30 分钟的代表模型热稳态
   连续测试，复用正式采样区间时必须实际满足时长与输入语义。

九、D1-07：功耗模式与热稳态，P0
1. 核对本 SKU 真正支持的 30W/60W/MAXN 配置及实测频率/功耗上限。先锁定一个
   主模式完成五模型主表；再在硬件支持的另一模式至少复测 SmolVLA 和一个
   已通过的大模型。原文若保留所有模型双模式结论，需要补齐对应完整矩阵。
2. 保存 nvpmodel、jetson_clocks、风扇、CPU/GPU/EMC、温度、tegrastats、
   throttling、进程占用与内存的时间序列；记录环境、预热和采样间隔。
   模式切换仅在台架可独占时执行并记录恢复；MAXN 不能直接改称 60W。
3. 有可用传感器时报告整板平均/峰值功率、测量边界和 energy/fresh chunk；
   整板功率不能标作纯 GPU 功耗。缺失传感器/权限给出 N/A 与具体证据。

十、D1-08：Qwen3.5-0.8B/2B，P1
1. 核心 VLA 路径闭合后，分别评估两个规模真实 image+text 的官方与 native
   路径；复用 H20 的参考包，但为 Orin 重新构建 artifact 并验证全部输出。
2. 覆盖 vision + prefill + text decode，报告 TTFT、固定 16-token 完整生成
   延迟/CDF、total tokens/s、decode tokens/s 与峰值内存；first-token 和
   fixed-length profile 分开。N 个输出需 N-1 次 decode，decode 分母要一致。
3. 管理完整 KV、conv/recurrent state、RoPE、有效长度和 reset；核验完整
   greedy tokens 及 logits/数值误差，验证状态隔离、容量与 shape 拒绝。
4. 当前 adapter 的固定 grid/RoPE、无 padding 等限制必须保留；匹配 profile
   或明确适配后再运行，不能直接宣称任意图像/多轮/流式支持。
5. 不直接拿预处理边界不同的官方/native 表计算加速比；必要时补同边界表。
   此项为可选补充，遇到真实平台限制独立报告，不无限挤占核心 VLA 任务。

十一、D1-09：低精度与论文强主张核对
1. 默认先交付同精度执行优化；只有论文保留 III-B 非对称低精度/逐 step scale
   贡献时，安排独立且有预算的代表性实验：同精度、全局尺度、encoder/head
   非对称、逐 step/group 尺度，使用分离的 calibration/held-out 数据。
2. 区分真实低精度 kernel 和 fake-quant；报告逐步误差、饱和、最终 free-running
   动作误差、时延和内存。不支持的精度给出证据，不以反复量化搜索追求无损。
3. 原稿中的 1.49x、126.4ms、std<1.2ms、1.3~1.9x 算子收益、weeks-to-minutes、
   零动态分配、zero-copy 和 strict equivalence 逐项关联实测或改写建议。
   没有硬件计数/真实相机路径证据时，不把 DMA/统一内存称为摄像头零拷贝。
4. 保留原稿 Octo/OpenVLA/Qwen2.5-VL 数字的来源待核对状态；本轮不自动
   新增这些旧模型实验，更不将其数字改标为作者新名单。
5. 输出 claim-gap 表与建议文本，不擅自重写 PDF；editable manuscript 集成
   单独说明。物理机器人闭环成功率不作为本轮新增硬性实验。

十二、通用设计、执行与资源约束
1. 继续使用 Python Adapter -> Invocation IR -> Plan -> native C++ Session，
   复用已有 benchmark、fidelity、CDF、board handoff 与 audit 工具。
   通用编排不能按某个模型名称硬编码，模型专属语义在 Adapter/声明中表达。
2. 实现导出/build/pilot/formal/collect/audit 的可恢复流程；在正式长跑前
   冻结 source、数值与输入协议。单卡顺序采样；CPU 准备可并行但避免干扰计时。
3. 每次恢复检查真实 PID/start time/argv、日志增长和当前状态；等待超时
   不等于失败，不因丢失监控会话就重复启动模型。
4. 不抢占外部任务。资源争用时只停止本任务自己的进程，保留无效整组证据，
   等空闲后新建输出目录并复用已审计的完整进程。
5. 保留工作区既有修改、正式输入输出、失败记录与 source 快照；不得清理
   原始实验数据、降低质量阈值、裁剪输出维度或用 mock provider 填表。

十三、跟踪与交付
1. 使用稳定编号 D1-00～D1-09，在 doc/edgefm_orin_goal_status.md 逐项记录
   priority、status、执行范围、最新证据、阻塞、下一条可执行命令。
   statuses：pending/running/pilot/passed/failed/compile-failed/unsupported/
   OOM/blocked/deferred；性能、输出质量与编译通过分别判定。
2. 每个 model × backend × power/profile 单元格保留真实状态。必须有实机
   完整执行、完整输出、正式采样与独立审计才标正式 passed。
   OOM/unsupported 必须有合理部署配置下可复现的原因，不等同于未尝试。
3. 本地按 artifacts/orin/<run_id>/ 归档 protocol/source/environment/inputs/
   build/pilot/formal/telemetry/audit/figures；板端使用经验证可写的 NVMe 目录，
   保留两端路径、传输 hash、真实 PID、完整命令、退出码、target 和 runtime DSO。
4. 生成 doc/reports/edgefm_orin_experiment_delivery.md、VLA 主表 CSV/JSON、
   所有 CDF PNG/PDF、完整输出质量表、算子与 runtime 消融表、功耗/内存/热稳态
   表、Qwen 补充表或受限说明、逐项 claim-gap 表，以及可复跑的命令与证据索引。
5. 公开报告中的数字必须由原始样本生成，目视检查最终图。代码改动执行适用
   CPU/C++/目标后端回归，未运行的 GPU/SDK 测试写明原因。

十四、完成标准
1. 五模型 Orin 正式 E2E、CDF、完整动作核验及必需算子/runtime/内存与功耗
   证据闭合，所有主表单元格可追溯；Qwen 可选项有完成结果或具体受限处置。
2. 必需项有无法解决的依赖、模型身份问题、失败或缺失证据时，继续保留为
   未完成；不能仅提交一份 TODO/unsupported 表就宣称五模型部署完成。
   若必须收窄论文支持范围，应明确提出，由用户决定，不自行减少目标。
3. 所有收益与限制对应真实配置和测量边界，原文缺证据的强结论有明确处理；
   H20/BPU 的历史范围不被此 Orin 任务自动提升。
4. 只在必要实验、审计和交付全部达到本标准时标记 Goal complete。
```
