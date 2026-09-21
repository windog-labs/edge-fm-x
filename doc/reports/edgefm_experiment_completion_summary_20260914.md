# EdgeFM 补充实验完成情况与可交付指标

整理日期：2026-09-14。本文为现有证据的状态快照，不是新一轮实验执行计划。

工作区：`/home/zhangzimo/.codex/worktrees/f801/edge-fm-x`。

依据：作者提出的三类补充实验、现有 H20 正式报告、J6M 原始归档，以及 2026-09-14 14:44 CST 更新的 J6M 状态文档。本次仅读取本地材料并新增本汇总，不连接远端、不启动实验、不改动既有结果。下文“进行中/排队”来自主线程已记录的状态，不代表本次重新核实了远端实时进程。

## 1. 结论与范围

**作者最初要求的整套实验尚未全部完成；H20 上限定配置的结果已经可以交付，Orin/BPU 五模型端到端主表仍缺。**

- H20：五个 VLA 的原始输入工作流对比、连续时延 CDF、完整输出一致性、六类算子微基准，以及 Qwen3.5 0.8B/2B 固定配置原生部署，均已有正式结果。
- J6M：仅 SmolVLA 有模型阶段结果。单 Solver Step 的正式 CDF 已完成；Prefix V2/V5 同条件短测已完成，新的阶段正式采样尚未验收完成。
- Orin、j6m-1 暂缓。RDT、pi0、pi0.5、CogACT 和 Qwen 的 J6M 实验暂缓；这不影响它们已有的 H20 结果。
- 当前 J6M 工作按用户收窄后的要求，以已有 SmolVLA HBM 的耗时为主，不新增 FP32 部署实验，不开展大规模量化搜索，不为追逐精度阈值反复编译。
- 完整链动作核验的历史失败必须保留，但不阻止报告真实阶段耗时；阶段性能完成、输出核验完成和整篇论文实验完成是三个不同的验收口径。

### 1.1 对照作者要求

| 编号 | 作者要求 | 完成情况 | 可以交付 | 仍缺或需限定的内容 |
|---|---|---|---|---|
| A1 | 五个 VLA 在 Orin/BPU 的 E2E 吞吐、时延主表 | 未完成 | H20 五模型工作流对比；J6M SmolVLA 阶段测时 | Orin 全部；BPU 五模型完整 E2E。H20 不可替代板端数据 |
| A2 | 连续推理时延抖动 CDF | 部分完成 | H20 五模型；J6M 单 Step 正式 CDF | J6M Prefix 新批次验收、完整链 CDF；Orin CDF |
| A3 | 动作余弦相似度、加速前后输出一致性 | 部分完成 | H20 完整输出逐字节一致；J6M 单 Step 单样本误差 | J6M 完整链尚无严格一致性通过证据；Orin 缺失 |
| B1 | Agent 调优前后单算子消融 | H20 限定范围完成 | 六类真实模型签名的固定编译候选对比 | 不是自主发明内核或整模型收益归因；J6M 历史探针仅探索性证据 |
| C1 | Qwen3.5 0.8B/2B 吞吐、时延和原生适配 | H20 固定配置完成 | 官方图文输入基线、原生首 token/16 token、CDF、输出及生命周期验证 | 原生计时不含图像预处理；任意输入/生成长度与板端部署未覆盖 |

### 1.2 模型与平台状态

| 模型 | H20 | J6M/BPU | Orin |
|---|---|---|---|
| SmolVLA | 原始输入工作流正式对比完成 | Step 完成；Prefix 短测完成，正式批次未闭合；非完整 E2E | 暂缓，未有可交付结果 |
| RDT-1B | 原始输入工作流正式对比完成 | 暂缓，未执行该模型板端实验 | 暂缓，未有可交付结果 |
| pi0 | 原始输入工作流正式对比完成 | 暂缓，未执行该模型板端实验 | 暂缓，未有可交付结果 |
| pi0.5 | 原始输入工作流正式对比完成 | 暂缓，未执行该模型板端实验 | 暂缓，未有可交付结果 |
| CogACT | 公共依赖配置正式对比完成 | 暂缓，未执行该模型板端实验 | 暂缓，未有可交付结果 |
| Qwen3.5 0.8B | 官方基线与固定配置原生部署完成 | 暂缓，未执行该模型板端实验 | 暂缓，未有可交付结果 |
| Qwen3.5 2B | 官方基线与固定配置原生部署完成 | 暂缓，未执行该模型板端实验 | 暂缓，未有可交付结果 |

## 2. H20 五个 VLA 的可交付结果

### 2.1 原始输入工作流时延与吞吐

正式配置为 `official-pytorch` 和 `session-required`。每个模型每个配置有 5 个独立进程，每进程 128 次预热、1024 次正式调用；合计 50 个进程、51,200 次正式调用。

计时从 CPU 内存中的已解码观测开始，到完整 CPU 动作输出结束，包括预处理、H2D、实际推理、后处理和 D2H。不含文件/视频解码、模型初始化、验证/日志 I/O、机器人通信。Session 周围仍有 Python 主机处理，不应标为完整无 Python 的传感器到执行器系统。

| 模型 | 官方平均 ms | Session 平均 ms | 官方 P99 ms | Session P99 ms | 官方 chunks/s | Session chunks/s | 平均加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SmolVLA | 206.161639 | 50.726747 | 220.579781 | 53.682128 | 4.850563 | 19.713466 | 4.064x |
| RDT-1B | 239.685959 | 221.804152 | 301.828993 | 230.311815 | 4.172126 | 4.508482 | 1.081x |
| pi0 | 221.305113 | 108.152217 | 350.640813 | 113.976963 | 4.518648 | 9.246227 | 2.046x |
| pi0.5 | 256.840307 | 116.853524 | 274.030360 | 122.893219 | 3.893470 | 8.557722 | 2.198x |
| CogACT，公共依赖配置 | 120.286270 | 79.971651 | 127.398996 | 82.199409 | 8.313501 | 12.504431 | 1.504x |

吞吐为新生成的动作块数，不是机器人伺服频率。RDT 使用原始 5 步调度器，其余配置使用 10 步。重复调用遍历有限的观测/噪声集，不能将调用次数写成独立轨迹数量。加速比为官方平均时延除以 Session 平均时延。

### 2.2 完整输出与 CDF

| 模型 | 核验完整张量数，含预热及两种配置 | 浮点输出最小余弦相似度 | 最大绝对误差 | MSE | 逐字节一致 |
|---|---:|---:|---:|---:|---|
| SmolVLA | 23,040 | 0.9999999999999998 | 0 | 0 | 是 |
| RDT-1B | 23,040 | 0.9999999999999998 | 0 | 0 | 是 |
| pi0 | 23,040 | 0.9999999999999998 | 0 | 0 | 是 |
| pi0.5 | 23,040 | 0.9999999999999998 | 0 | 0 | 是 |
| CogACT | 57,600 | 0.9999999999999998 | 0 | 0 | 是 |
| 合计 | 149,760 | 0.9999999999999998 | 0 | 0 | 是 |

比较对象为各自冻结参考的完整输出，包含归档中的动作、后处理输出和 CogACT 整数 RNG 回执等，不全是独立动作样本。整数回执只做精确比较，不计算虚假的余弦。F64/BF16 输出保留真实 dtype。结论限于所测输入和数值配置，不证明机器人成功率、物理标定或任意低精度量化无损。

每模型和合并 CDF、原始采样 CSV 均已完成；预热不进入时延统计，实测离群值保留。

证据：[H20 交付报告](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/h20_vla_experiment_delivery_20260910.md)、[原始结果 CSV](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-five-summary-001/host-pipeline-table.csv)、[完整输出 CSV](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-five-summary-001/complete-output-quality.csv)、[CDF PNG](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-five-summary-001/host-pipeline-cdf.png)、[CDF PDF](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-five-summary-001/host-pipeline-cdf.pdf)。

### 2.3 CogACT 的模型身份与独立原生实验

实测 checkpoint 为 **7,630,224,071 参数，约 7.63B**，使用公共依赖配置。不能直接标成作者列出的“CogACT 3B”；与原始受限 Meta 依赖配置的等价性尚未验证。上面的原始输入实验使用归档外部 RNG tape，另有下列原生 C++ RNG 的独立实验。

| 配置 | 平均 ms | P50 ms | P95 ms | P99 ms | 调用/s |
|---|---:|---:|---:|---:|---:|
| 官方 PyTorch | 129.258 | 124.006 | 177.383 | 232.080 | 7.736 |
| 自主 RNG 的原生 Session | 88.751 | 89.199 | 90.683 | 95.811 | 11.268 |

同边界平均加速比 **1.456x**，95% bootstrap 区间 `[1.450, 1.463]`。每配置 5120 次正式调用，完整输出及 RNG 与同一参考逐字节一致。官方第 4 个 worker 的长尾保留，配对 GPU 平均时延比的中位数为 1.393x。

该实验包含 RNG、模型/原始 VLM 与 DDIM 采样、CUDA 完成等待，不包含静态输入 H2D、绑定、预处理和输出转换。不能与 2.1 的工作流表拼接计算。SmolVLA 的 SO100 统计、RDT 的相机配置、pi0/pi0.5 单 ALOHA episode 和 CogACT 单 Fractal episode 等输入限定，详见交付报告。

证据：[CogACT 同边界报告](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/cogact_timing_boundary_formal_20260910.md)。另外已有 [CUDA 常驻表](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/h20_vla_formal_table_20260909.md) 和 [主机模型张量边界表](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/h20_host_tensor_formal_table_20260909.md)，各 15 行、76,800 次正式调用；它们是不同计时边界的补充证据，不能改名为原始输入 E2E。

## 3. H20 算子微基准

每类 5 对独立进程，使用同一归档真实模型工作负载签名。单位为微秒；数值是 5 个进程各自 graph-batch 设备计时中位数的均值，不是逐调用 CDF。

| 类别 | ATen 基线 us | 候选 us | 时延降低 | 候选方案 |
|---|---:|---:|---:|---|
| Attention | 103.301055 | 103.313151 | -0.0117% | Inductor ATen |
| GEMM / Linear | 11.540214 | 12.870522 | -11.5276% | Inductor autotune |
| LayerNorm | 13.816088 | 13.716502 | 0.7208% | Inductor ATen preserving |
| Embedding | 2.142171 | 1.865898 | 12.8969% | Inductor ATen |
| RoPE frequencies | 21.332679 | 8.486607 | 60.2178% | Inductor ATen |
| VLA solver/update | 4.289051 | 2.203624 | 48.6221% | Inductor ATen |

完整目标/参考和候选输出均逐字节一致。平台为 H20-2 GPU6，Torch 2.10.0+cu128。

本表支持“Agent 辅助固定编译方案选择”的限定消融，不证明自主内核发明、完整技能复用循环或候选已集成到模型。Attention 基本无变化，Linear 变慢；负结果必须保留。LayerNorm 覆盖归一化类别，但没有独立 H20 RMSNorm 正式结果；solver 行仅是三算子更新片段。

证据：[算子报告](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/h20_operator_table_20260909.md)、[算子 CSV](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/operator-six-summary-002/operator-table.csv)、[对比图](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/operator-six-summary-002/operator-comparison.png)。

## 4. Qwen3.5 的 H20 结果

### 4.1 固定配置原生部署

327 输入 token，固定图像/张量形状；首 token 与完整 16 token 是分开的正式实验。每组 5 进程，每进程 128 次预热和 1024 次正式调用。

| 模型 | 输出配置 | 平均 ms | P50 ms | P95 ms | P99 ms | 请求/s | 完整输出 token/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.8B | 首 token | 96.905373 | 99.480645 | 104.489179 | 114.008338 | 10.319345 | 不适用 |
| 0.8B | 16 token | 369.289954 | 322.653369 | 573.796106 | 625.620183 | 2.707899 | 43.326388 |
| 2B | 首 token | 90.657933 | 85.696679 | 104.990503 | 137.604724 | 11.030474 | 不适用 |
| 2B | 16 token | 370.497338 | 333.449038 | 583.324144 | 629.293585 | 2.699075 | 43.185196 |

计时包含 H2D、输入绑定、Session run、完成等待和完整 typed D2H；不含图像预处理、初始化、日志/验证 I/O 和 detokenization。完整输出吞吐为 `16 / 完整调用平均秒数`，不是纯 decode 吞吐。首 token/完整生成使用独立测量和不同 GPU，不通过两均值之差构造正式 streaming 逐 token 结果。

| 验证项 | 已完成证据 |
|---|---|
| 正式采样 | 20 个独立进程，20,480 次正式调用，4 组独立审计及 CDF |
| 完整输出 | 46,080 个张量、5,721,488,640 个值，含预热；token 和 BF16 logits 对完整 direct/eager 参考逐字节一致 |
| 原生运行 | 生成的 C++ Session；报告记录 worker 无 libpython/libtorch_python 映射 |
| 契约拒绝 | 每 bundle 7 类非法形状拒绝；accepted=false 拒绝；失败后已提交输出哈希保持不变 |
| 生命周期 | 4 个 bundle 的重复调用、双 Session 交错、reset、销毁重建；80 项独立断言复核 |

这是固定配置完成，不是任意 prompt、图像、生成长度或板端通用部署完成。四组实验在同一主机的不同 GPU 上运行并共享主机资源，长尾不是独占设备峰值极限。Qwen 是 token/logit 输出，不应套用 VLA 动作余弦指标。

证据：[原生正式报告](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/qwen35_native_formal_20260910.md)、[证据索引](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-evidence-index-002.json)、[首 token CDF](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-first/latency-cdf.png)、[16 token CDF](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-full/latency-cdf.png)。

### 4.2 官方图文输入基线

真实 640x480 图像和文本，327 输入 token 中含 300 图像 token，生成 16 token。每模型 5120 次正式调用；5760 个完整 token 输出（含预热）与同后端参考一致。

| 模型 | TTFT 平均 ms | TTFT P99 ms | 完整 16 token 平均 ms | 完整 P99 ms | 纯 decode token/s |
|---|---:|---:|---:|---:|---:|
| 0.8B | 152.937227 | 164.270234 | 558.793883 | 595.405434 | 36.958862 |
| 2B | 156.301746 | 171.681141 | 545.668130 | 592.475612 | 38.524127 |

这里从主机内存中的编码图像字节开始，包括图像预处理、H2D、vision、prefill 和生成，不含文件读取、初始化及 detokenization。decode 吞吐按剩余 15 token 总数除以首 token 后的时间总和计算，和 4.1 的 `16 / full latency` 不同。

**不得用 4.2 与 4.1 直接计算官方到原生的加速比。** 原生适配已证实，但若论文需要“原始图文输入同边界原生加速”结论，还缺包含相同预处理边界的原生对照。当前固定配置验收不要求额外扩展这一实验。

证据：[官方自然图像基线报告](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/qwen35_natural_profile_20260909.md)。

## 5. J6M SmolVLA 的完成项与限制

板卡为 j6m-2，归档序列号 `0e190a0d3392371c`；已测环境为 UCP 3.14.7、HBRT 4.9.7、BPU library 2.2.6、HBM compiler 4.5.5。本节不是全模型 E2E 结果。

### 5.1 已有 HBM 的同条件短测

每行 5 帧 `hrt_model_exec perf`，单线程、默认核心选择；Prefix V2/V5 使用相同的 7 个真实捕获输入文件，顺序执行。

| 阶段 | 平均 ms | perf 报告调用/s | BPU ms | 外部 CPU ms | 最小 HBM 内存计划 MiB |
|---|---:|---:|---:|---:|---:|
| Prefix V2 | 1406.642 | 0.711 | 186.942 | 1216.335 | 560.451 |
| Prefix V5 | 555.395 | 1.800 | 165.712 | 388.700 | 543.660 |
| Solver Step-060，新短测 | 91.050 | 10.945 steps/s | 31.659 | 57.498 | 119.401 |

Prefix V5/V2 平均加速 **2.533x**。V2 的 63 个外部 CPU `InplaceScatterND` 共 812.164 ms，V5 为 0 个。对比还包含 V2/V5 的其他编译差异，不是单独移除 ScatterND 的严格因果消融。

计时边界为 `hbDNNInferV2` 前到 `hbUCPWaitTaskDone` 后，包含任务构建、提交和等待，不含输入文件读取、输出 dump、模型加载。含 CPU fallback，不能写成纯 BPU 时延；内部复制/同步没有独立分项，不能假定为零。内存列是编译器静态计划，不是 RSS 或完整设备实占内存。

### 5.2 已完成的 Step 正式 CDF 与局部核验

| 项目 | 已完成结果 | 限定 |
|---|---|---|
| Step 正式 CDF | 5 进程，每进程 128 预热 + 1024 正式，共 5120 样本 | 2026-09-11 既有正式批次；一个 solver step，不是完整 10 步模型 |
| 平均 / P50 / P95 / P99 / 最大 | 89.738050 / 89.645 / 91.068 / 95.593 / 96.937 ms | 原始离群值保留，不与 5 帧 perf 均值混合 |
| Step 动作单样本核验 | cosine 0.9999994983908165；max-abs 0.0036740303；MSE 8.9804977301e-7 | 通过该阶段局部阈值，但非逐字节相同、非完整链验收 |
| Prefix V5 单样本检查 | 33 输出均有限，mask 精确；最低 cache cosine 0.9940398655，最大 cache 误差 1.402682066 | cache 不是最终动作，不证明量化无损 |
| Prefix V5 有效 run-1 | 1024 正式样本；平均 553.938838 ms，P99 556.520 ms | 仅 1 个进程，不能标五进程正式完成 |

Step-060 使用 FP16 主体与 INT16 Conv/MatMul，仍有既有 FP32 算子或 CPU 回退，不能标为全图 FP16/INT16 或完全无 FP32。当前约束是不新增全 FP32 部署路线，不抹去既有混合精度事实。

证据：[J6M 当前测时报告](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/smolvla_j6m_performance_20260914.md)、[Step-060 报告](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/reports/smolvla_j6m_stage060_20260911.md)、[Step 正式统计 JSON](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/j6m-feasibility-20260911/smolvla-stage-060-horizon-nash-m-fp16-linear-int16/board-evidence/latency-summary/summary.json)、[V5 单样本检查](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/j6m-performance-20260914/prefix-v5-output-check.json)。

### 5.3 进行中、历史失败与探索性证据

- 原 `formal` 批次因外部 BPU 用户出现而停止，JSON 为 `failed`，错误为 `BPU idle check failed: ratio='23'`。V5 run-1 有效；run-2 整轮保留但排除，依据是外部资源占用，而非挑选时延值。
- 14:44 CST 状态报告及干扰审计记录 `formal-r2` 已恢复，复用有效 run-1，替换受干扰 run-2；Finish 的新队列等待它成功结束。本汇总读取时没有本地完整 r2 终态验收结果，因此仍记为“报告记录进行中，尚未完成”，不能把原批次失败写成恢复批次也失败。
- 历史 `prefix -> 10 step -> finish` 文件编排 pilot 已执行：最终动作 cosine 0.9999031720、max-abs 0.0508375019、mean-abs 0.0160986385，未通过当时的 `max_abs <= 0.05`、`mean_abs <= 0.01` 严格门槛。148.238 s 是含 SSH、文件传输、加载等开销的命令墙钟时间，不能当作模型 E2E 时延。
- J6M 历史 INT8 算子探针与 FP32 基线已归档，但采用合成固定形状；Attention 为 200/200 帧，其他所复用类别为 100/200 帧，GEMM/LayerNorm/Embedding 是组合探针。它们不是当前 FP16/INT16 模型配方的正式逐算子五进程消融，不用于宣称完整动作质量保持，也不要求重新跑 FP32。
- Finish 现有 HBM 是单个 Slice，F32 输入/输出仅做裁剪，不含浮点算术；未来耗时应标“输出处理”，不能标作 FP32 模型部署。

证据：[J6M 状态记录](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/doc/edgefm_j6m_goal_status.md)、[原正式批次 JSON](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/j6m-performance-20260914/formal/campaign.json)、[干扰排除及恢复审计](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/j6m-performance-20260914/interference-audit.json)。历史 full-chain 和算子探针的指标及来源见 Step-060 报告。

## 6. 未完成项清单与验收条件

此表区分当前 SmolVLA 阶段交付与作者原始论文缺口，不授权新增执行或扩展模型范围。

| ID | 未完成项 | 当前状态 | 纳入当前阶段交付？ | 完成所需证据 |
|---|---|---|---|---|
| J1 | Prefix V5/V2 新批次正式时延、CDF | 状态记录为恢复采样中；V5 仅有一个已确认有效进程 | 是 | 每阶段 5 个独立进程，各 128 预热 + 1024 正式；无资源干扰、原始日志、统计/CDF及审计齐全 |
| J2 | Step 新批次重复测量 | 状态记录为同批次后续任务 | 是，按已记录批次；既有 Step 正式结果不因此失效 | 新批次完整采样及审计，与旧正式数据分开保存 |
| J3 | Finish 输出裁剪测时 | 状态记录为排队 | 是 | 独立阶段 smoke/正式测时、真实边界和输出处理标签 |
| J4 | 本轮阶段性能交付包 | 短测已具备，正式证据未闭合 | 是 | 阶段正式表/CDF、协议、输入与 HBM 哈希、内存/环境记录、负结果和证据索引 |
| P1 | SmolVLA J6M 完整模型 E2E 吞吐/时延/CDF | 文件 pilot 不能替代；常驻 runner 尚未实现 | 否，后续论文扩展 | 模型常驻、真实初始化及完整链执行、清楚的 E2E 边界、正式连续采样；不可仅累加阶段均值 |
| P2 | SmolVLA J6M 完整动作无损或严格一致性证明 | 历史 full-chain 严格阈值失败；V5 只有局部单样本检查 | 否，不以反复量化达标作为本轮任务 | 对完整最终动作的明确样本集及误差证据；没通过就保留限制，不因 cosine 高而宣称无损 |
| P3 | RDT、pi0、pi0.5、CogACT 的 J6M 主表 | 用户明确暂缓，未执行模型板端实验 | 否 | 将来重新授权后，先支持/容量评估再决定能否正式部署；不得预承诺 |
| P4 | Orin 五个 VLA 的主表、CDF、动作核验 | 暂缓 | 否 | 可用硬件/runtime 后的真实 Orin 正式实验 |
| P5 | 板端正式单算子 Agent 消融 | 仅 J6M 历史探索性探针 | 否 | 若保留板端优化叙事，需同平台同工作负载、明确候选/精度及一致协议；旧探针不足以替代 |
| P6 | CogACT 与作者列出模型身份对齐 | 已测约 7.63B 公共依赖配置 | 文稿必须处理，不强制重跑 | 按真实 checkpoint/依赖命名并征求作者确认；若坚持其他配置，现有结果不能替代 |
| P7 | Qwen 原始图文输入同边界原生加速对照 | 原生与官方已有表的预处理边界不同 | 否，当前固定配置已完成 | 仅在需要该加速结论时补齐同边界测量；不能直接用现有两表求比 |
| P8 | 结果写入论文与叙事对齐 | 已有报告/图表，论文未因本汇总而修改 | 否 | 可编辑论文源及作者确认；区分 H20/板端、stage/E2E、pilot/正式及模型身份 |

Qwen 任意长度、任意图像与流式生成属于当前固定配置之外的泛化范围，不应自动视为本轮未完成任务。J6M 多样本量化搜索、额外重编译或其他模型测试也不在当前执行范围内。

## 7. 交付时的统一口径

| 可以写的结论 | 不应写的结论 |
|---|---|
| H20 五个限定 VLA 配置的完整主机工作流、CDF 和完整输出核验已完成 | 五个 VLA 在 Orin/BPU 的 E2E 全部完成 |
| H20 固定工作负载下，SmolVLA 平均工作流加速 4.064x；各模型增益见表 | 将 H20 增益套用到 J6M/Orin，或宣称任意输入都同等加速 |
| J6M Prefix V5 在同条件 5 帧短测中比 V2 快 2.533x | 完整 SmolVLA E2E 快 2.533x，或已经有正式五进程 Prefix CDF |
| H20 已测输出与冻结参考逐字节一致；J6M 局部误差如实列出 | J6M 量化已无损、机器人实际成功率不变、全模型完全无 FP32 |
| Qwen 两种规模的固定配置原生 C++ Session 已验证 | Qwen 任意输入/流式/板端部署已验证，或两张不同边界表证明原生加速 |
| Agent 辅助选择固定编译方案有收益也有负结果 | 所有算子都加速、Agent 自主发明所有内核、微基准收益已解释整模型收益 |

本文是一次状态汇总。后续只应在主线程生成新的完整结果、终态报告及审计后更新对应状态；不根据运行 PID、排队状态、编译成功或一次高 cosine 提前标记实验完成。
