# EdgeFM 三平台最终实验数据汇总

更新日期：2026-10-26，Asia/Shanghai。

存储说明：用户随后授权退役本机实验大目录，本文部分原机来源链接不再可访问；数值与历史审计结论未改。论文修改使用已独立核验的[自包含交接包](../../paper_handoff/edgefm_three_platform_20261026/README.md)，无需已删除的路径。范围及复现限制见[清理记录](local_bulk_cleanup_20261026.md)。

本文是论文写作用的数据入口，统一汇总已经审计的 **H20、J6M 和 Orin** 结果；不启动新推理、不改原始统计、不重写历史失败，也不修改论文 PDF。数字按原始 CSV/JSON 四舍五入，完整精度保留在各节来源中。

**当前可支持的论文范围：H20 和 Orin 的五模型完整 VLA 部署，H20 的 Qwen3.5-0.8B/2B、Orin 的 Qwen3.5-0.8B 补充，以及 J6M 的 SmolVLA 阶段级部署。J6M 五模型完整 E2E 主表仍然缺失，不能用阶段耗时求和补齐。** Orin 已选实验范围完成，不等于论文所有更强主张都已验证。

## 1. 覆盖与统一口径

### 1.1 覆盖矩阵

| 模型 / 实验 | H20 | J6M | Orin |
|---|---|---|---|
| SmolVLA | 完整 VLA，正式时延/CDF/动作核验 | Prefix V2/V5、单步 Solver、Finish 正式阶段时延/CDF；完整链仅有质量未通过的文件编排 pilot | 完整 VLA，常规/共享流/完整循环 replay 正式对比 |
| RDT-1B | 完整 VLA，正式时延/CDF/动作核验 | deferred，未运行 | 完整 VLA，常规/共享流/完整循环 replay 正式对比 |
| pi0 | 完整 VLA，正式时延/CDF/动作核验 | deferred，未运行 | 完整 VLA，仅 Arena 布局对比，无 replay |
| pi0.5 | 完整 VLA，正式时延/CDF/动作核验 | deferred，未运行 | 完整 VLA，仅 Arena 布局对比，无 replay |
| CogACT | 完整 VLA，实际公开依赖配置约 7.63B | deferred，未运行 | 完整 VLA，同样保留实际约 7.63B 身份，仅 Arena，无 replay |
| Qwen3.5-0.8B | 原生 C++ Session 首 token / 16-token；另有官方图文基线 | deferred，未运行 | 完整 native TRT vision/prefill/15decode，正式 TTFT/16-token/CDF/质量/资源 |
| Qwen3.5-2B | 已有正式原生部署及官方图文基线 | deferred，未运行 | 用户取消，不计为测试、通过或不支持 |
| 算子消融 | 六类真实模型签名的固定 compiler recipe 对比 | HBM 版本/放置变化及历史合成 FP32/INT8 支持性探针 | 六类真实签名微基准；完整模型回集成含负收益与未过质量门槛的候选 |

模型名只标识实测 checkpoint/profile；不将 CogACT 的实测约 7.63B 写成名义 3B。SmolVLA 保留恢复的 SO100 统计配置；RDT、OpenPI 和 CogACT 的观测、动作维度、求解步数、精度与依赖限制均随原始报告保留。

### 1.2 计时与统计边界

| 数据组 | 计时范围 | 不应混淆的事项 |
|---|---|---|
| H20 VLA 原始输入 | 已解码 CPU 观测/状态/文本/实际 noise -> 预处理、H2D、完整模型、后处理、完整 CPU 动作 | 外围有 Python；不是完全无 Python 的传感器到执行器链路 |
| H20 resident VLA | 常驻模型输入 -> Session 计算、输出处理与同步完成 | 不含原始输入处理和 H2D/D2H；不是上一行的 E2E |
| H20 host-model-tensor / native Qwen | 已准备 CPU 模型 tensor -> H2D、binding、Session、完整 typed D2H | 不含在线图像/文本预处理；Qwen 首 token 与 full16 是不同进程组 |
| J6M HBM stage | `hbDNNInferV2` 前 -> `hbUCPWaitTaskDone` 后 | 含任务构造/提交/等待；不含文件加载、输出 dump、模型加载和 SSH；是 stage calls/s |
| Orin VLA | 已解码 CPU 观测 -> native 在线准备、搬运、完整视觉/语言/全部动作步、后处理、完整 CPU action chunk | 原生 TRT/C++/CUDA，不运行 Torch 模型；同平台常规 TRT 对优化路径 |
| Orin Qwen | raw CPU RGB+instruction -> native 在线分词/patchify、H2D、17 次神经+16 次控制调用、CPU token/最终 logits | fresh vision/prefill；固定 16 新 tokens；TTFT 和 full16 来自同一次请求，不是两批独立样本 |

主要完整模型与最终 J6M 阶段结果均为 batch=1，每配置 **5 个独立进程 × (128 warmup + 1024 measured)**，每配置 5120 条测量。warmup 不计入时延，但质量记录保留。没有按延迟删除慢样本；资源冲突/中断的排除过程按原始审计记录保留。

VLA 吞吐为 `1000 / mean_ms` 个 **fresh action chunks/s**，不是动作 horizon 乘出来的控制 Hz。Qwen full16 吞吐为 `16 * 1000 / mean_full16_ms`。没有定义控制 deadline 时，不填 deadline-miss 概率。分位数保留各 campaign 的原方法，例如 H20 native Qwen 是 nearest-rank，Orin Qwen 是 `(n-1)*q` 线性插值，不重新统一后覆盖原值。

## 2. VLA 性能主表

### 2.1 H20：完整原始输入工作流

单位：ms；最后一列为 chunks/s。每行 n=5120，10 配置共 50 进程、51200 条 measured；包含 warmup 的完整输出核验共 149760 个 tensor。

| 模型 | 配置 | Mean | P50 | P95 | P99 | Max | Std | Chunks/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SmolVLA | official-pytorch | 206.161639 | 210.240524 | 216.365152 | 220.579781 | 234.840338 | 9.333303 | 4.850563 |
| SmolVLA | session-required | 50.726747 | 51.901111 | 52.617350 | 53.682128 | 69.352764 | 1.808891 | 19.713466 |
| RDT-1B | official-pytorch | 239.685959 | 236.215243 | 247.706501 | 301.828993 | 373.057891 | 11.895341 | 4.172126 |
| RDT-1B | session-required | 221.804152 | 221.369958 | 224.783646 | 230.311815 | 268.850709 | 3.366552 | 4.508482 |
| pi0 | official-pytorch | 221.305113 | 221.656810 | 246.965037 | 350.640813 | 372.507159 | 24.208986 | 4.518648 |
| pi0 | session-required | 108.152217 | 108.430270 | 109.614178 | 113.976963 | 156.662682 | 1.901058 | 9.246227 |
| pi0.5 | official-pytorch | 256.840307 | 262.418709 | 268.599969 | 274.030360 | 294.215221 | 10.815259 | 3.893470 |
| pi0.5 | session-required | 116.853524 | 116.317470 | 118.850791 | 122.893219 | 170.638020 | 2.031607 | 8.557722 |
| CogACT (public dependencies) | official-pytorch | 120.286270 | 120.234825 | 123.989071 | 127.398996 | 136.697471 | 2.700216 | 8.313501 |
| CogACT (public dependencies) | session-required | 79.971651 | 79.558869 | 81.669945 | 82.199409 | 90.627153 | 0.982177 | 12.504431 |

RDT 保留原 5 步求解器，其他 VLA profile 为 10 步；不能通过减少求解次数解释加速。H20 该表的 baseline 是 official PyTorch，不是 TensorRT baseline。CogACT 该组使用归档外部 RNG tape，不与后文自主 C++ RNG 组混合。

来源：[H20 交付](h20_vla_experiment_delivery_20260910.md)、[原始输入主表 CSV](../../artifacts/recovery-audit-20260909/raw-five-summary-001/host-pipeline-table.csv)、[质量 CSV](../../artifacts/recovery-audit-20260909/raw-five-summary-001/complete-output-quality.csv)、[审计汇总](../../artifacts/recovery-audit-20260909/raw-five-summary-001/report.json)。

### 2.2 Orin：完整 native TRT 工作流

单位：ms、chunks/s；每行 n=5120。12 配置、60 进程、69120 raw / 61440 measured。设备为实际 Jetson AGX Orin，TensorRT 8.5.2.2、CUDA 11.4、L4T 35.4.1，既有 MAXN，**不写成 60W**。

| 模型 | 配置 | Mean | P50 | P95 | P99 | Max | Std | Chunks/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SmolVLA | Conventional TRT | 134.409630 | 134.349024 | 136.952802 | 138.496310 | 142.241472 | 1.509459 | 7.439943 |
| SmolVLA | Shared stream | 133.067807 | 133.085408 | 135.528658 | 136.765336 | 138.972224 | 1.492815 | 7.514966 |
| SmolVLA | Full-loop replay | 113.139929 | 113.600224 | 115.987411 | 117.187907 | 121.009632 | 2.034434 | 8.838613 |
| RDT-1B | Conventional TRT | 3065.625218 | 3063.296512 | 3102.937584 | 3109.544230 | 3121.115744 | 23.509168 | 0.326198 |
| RDT-1B | Shared stream | 3064.553677 | 3071.761888 | 3106.344786 | 3111.908306 | 3122.125760 | 29.516018 | 0.326312 |
| RDT-1B | Full-loop replay | 2975.587343 | 2964.070992 | 3006.751920 | 3012.872988 | 3029.676640 | 21.612032 | 0.336068 |
| pi0 | Ordinary repeated-call TRT | 1129.869939 | 1127.602320 | 1163.024701 | 1170.859402 | 1177.885024 | 16.205243 | 0.885058 |
| pi0 | Verified arena; no replay | 1125.379221 | 1127.359920 | 1146.530686 | 1150.569614 | 1157.528480 | 13.313744 | 0.888589 |
| pi0.5 | Ordinary repeated-call TRT | 1258.792712 | 1255.848832 | 1275.217485 | 1279.311340 | 1290.113440 | 9.709894 | 0.794412 |
| pi0.5 | Verified arena; no replay | 1256.955605 | 1260.030848 | 1269.932627 | 1273.155440 | 1280.483680 | 9.474035 | 0.795573 |
| CogACT (actual ~7.63B public-dependency configuration) | Ordinary repeated-call TRT | 694.393388 | 694.044368 | 699.434048 | 704.337246 | 717.167104 | 2.973361 | 1.440106 |
| CogACT (actual ~7.63B public-dependency configuration) | Verified arena; no replay | 693.790506 | 693.427904 | 698.610686 | 704.232248 | 718.729504 | 2.970001 | 1.441357 |

同一 Orin 内的均值比分别约为 SmolVLA 1.187995x、RDT 1.030259x、pi0 1.003990x、pi0.5 1.001462x、CogACT 1.000869x。后三者只是 Arena 布局变化，差异很小且有负收益轮次，不能写成已实现 replay 或显著加速。CogACT 优化路径 Max 更高，SmolVLA replay Std 高于常规路径。

来源：[最终已选范围验收](orin_selected_goal_acceptance_20261026.md)、[主表 CSV](../../artifacts/orin/20261024-07/tables-001/main-table.csv)、[质量 CSV](../../artifacts/orin/20261024-07/tables-001/quality-table.csv)、[五模型输入/证据清单](../../artifacts/orin/20261024-07/input-manifest-001.json)。H20 与 Orin 的 baseline、精度和 runtime 不同，**不计算跨表硬件加速比**。

### 2.3 J6M：SmolVLA HBM 阶段表，不是模型 E2E

实测板卡 `j6m-2`，序列号 `0e190a0d3392371c`；UCP 3.14.7、HBRT 4.9.7、BPU library 2.2.6。每行 n=5120，单位为 ms、stage calls/s；Step 的调用是一轮 solver update，不是完整 action chunk。

| Stage | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Stage calls/s | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Prefix V2 | 1402.998 | 1402.430 | 1408.200 | 1412.990 | 1431.410 | 2.797 | 0.712759 | 35.8125 |
| Prefix V5 | 553.930 | 553.805 | 555.397 | 557.017 | 570.170 | 0.908 | 1.805283 | 33.7500 |
| Solver Step-060 | 89.921 | 89.834 | 91.249 | 95.632 | 96.615 | 1.164 | 11.120922 | 34.0000 |
| Finish crop | 0.261185 | 0.254 | 0.304 | 0.373 | 1.154 | 0.034997 | 3828.710091 | 27.0625 |

Prefix V2/V5 正式均值比为 2.532809x。选定模型阶段采用 FP16 body / INT16 Conv、MatMul 等既有混合精度，并非全模型 FP32。Finish 仅是 `[1,50,32] -> [1,50,6]` 的 Slice 裁剪：F32 I/O 不等于 FP32 神经网络部署。

该表优先使用 2026-09-14 最终 stage campaign；2026-09-11 Step 的 `89.738050 ms` CDF 与 `90.088 ms` 短测仍是历史独立批次，不替换本表的 `89.921 ms`。中断/争用排除和复用的完整、已核验进程在来源中保留，没有拼接半个进程的样本。

来源：[J6M 最终阶段报告](smolvla_j6m_performance_20260914.md)、[阶段 CSV](../../artifacts/j6m-performance-20260914/final-stage-summary/stage-performance.csv)、[Finish CSV](../../artifacts/j6m-performance-20260914/final-finish-summary/stage-performance.csv)、[最终证据索引](../../artifacts/j6m-performance-20260914/evidence-index-final.md)。

## 3. Qwen3.5 VLM 补充实验

### 3.1 H20 native C++ Session

每行 5 进程、n=5120；输入为已准备 `input_ids [1,327]`、`attention_mask [1,327]`、`pixel_values [1200,1536]` 和显式状态。计时含 H2D/binding/完整 typed D2H，不含图文在线预处理。首 token 与 full16 是独立固定 profile，不能相减冒充逐 token 实测。

| 模型 | Profile | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Full16 tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | First token | 96.905373 | 99.480645 | 104.489179 | 114.008338 | 167.637292 | 8.145816 | - |
| Qwen3.5-0.8B | Fixed16 | 369.289954 | 322.653369 | 573.796106 | 625.620183 | 1097.094486 | 85.094859 | 43.326388 |
| Qwen3.5-2B | First token | 90.657933 | 85.696679 | 104.990503 | 137.604724 | 169.666646 | 11.079401 | - |
| Qwen3.5-2B | Fixed16 | 370.497338 | 333.449038 | 583.324144 | 629.293585 | 1005.713665 | 77.170312 | 43.185196 |

四组共 20 进程、20480 measured、46080 个完整输出 tensor；完整 token/BF16 logits 对 direct/eager reference byte-exact。各组使用不同 H20 GPU UUID，并行组共享主机 CPU/存储，保留全部长尾；0.8B/2B 小差异不构成模型大小性能排名。

来源：[H20 native Qwen 正式报告](qwen35_native_formal_20260910.md)、[四组审计和 CDF 根目录](../../artifacts/recovery-audit-20260909/qwen-native-formal-20260910/)、[生命周期审计](../../artifacts/recovery-audit-20260909/qwen-native-lifecycle-probe-20260910-001/lifecycle-audit.json)。

### 3.2 H20 官方图文基线，单列参考

计时从主机内存中的编码图像字节开始，包含图像解码/预处理和 H2D，到 CPU tokens；不含文件 I/O、初始化、detokenize。固定一幅 640×480 图像和 prompt，每模型 n=5120；不是 16 图像正式泛化集。

| 模型 | TTFT mean / P99 ms | Full16 mean / P99 ms | 后15 tokens/s | Torch peak allocated bytes |
|---|---|---|---:|---:|
| Qwen3.5-0.8B | 152.937227 / 164.270234 | 558.793883 / 595.405434 | 36.958862 | 1805916160 |
| Qwen3.5-2B | 156.301746 / 171.681141 | 545.668130 / 592.475612 | 38.524127 | 4532012032 |

“后15 tokens/s”来自同请求 `full16 - TTFT` 区间，**不是**上一表的 E2E tokens/s。此官方图文边界与 native host-model-tensor 边界不同，不能直接以两表均值相除宣称部署加速比。来源：[官方图文正式报告](qwen35_natural_profile_20260909.md)。

### 3.3 Orin Qwen3.5-0.8B

同一 Orin、同 3 个神经 engine + 2 个无参数控制 engine、同精度，普通布局 `off` 对 `verified Arena`，**无 whole-loop replay**。16 个变化的真实 RGB 观测循环，每次都重新在线分词和执行 vision/prefill。每配置 5 进程、n=5120，共 11520 raw / 10240 measured。

| 配置 | 指标 | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | Std ms | Full16 tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| off | TTFT | 334.328554 | 333.731760 | 339.407525 | 343.079333 | 357.044384 | 2.586327 | - |
| verified | TTFT | 333.349658 | 332.869280 | 338.417126 | 341.265658 | 357.665152 | 2.655338 | - |
| off | Full16 | 1593.061317 | 1592.458800 | 1600.865789 | 1605.455708 | 1618.577408 | 4.441399 | 10.043556 |
| verified | Full16 | 1586.510069 | 1586.291328 | 1593.629232 | 1598.201941 | 1628.726880 | 4.247181 | 10.085029 |

这四行是 **同一万余条请求的两个相关计时指标**，不是 20480 条独立测量。Full16 均值降低约 0.4112%，不宣称显著加速；两项 Max 均变差，TTFT Std 也更高。Full16 包含全部 16 tokens 和最终 logits 的 D2H，TTFT 为 CPU-ready 首 token；初始化、文件解码、reset bookkeeping 和日志在计时外。

来源：[最终 Qwen 测量](../../artifacts/orin/20261025-21/final-export-006-retry-001/measurements.md)、[performance summary](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/summary.json)、[完整原始 CSV](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/raw.csv)、[31 项发布后验收](../../artifacts/orin/20261025-21/selected-goal-review-002.json)。2B 在 Orin 用户取消；在 H20 已测，不全局删除该已有结果。

## 4. 输出质量核验

### 4.1 正式通过结果

Orin 主门槛按用户确认口径为 **同一 Orin、相同精度/输入/noise 的常规 TRT 对优化路径，标准化 action MSE < 1e-5、cosine > 0.9999**。native action 量纲误差和跨 H20 差异单列；整数 RNG/count/token 用精确比较，不人为赋予浮点 cosine。

| 平台 / 对象 | 实际核验覆盖 | 结果 | 可支持结论 |
|---|---|---|---|
| H20 原始输入五 VLA | 149760 个完整输出 tensor，含 warmup；保留 BF16/F64 及 CogACT 整数辅助端口 | byte-exact；浮点 MSE/max-abs=0，最小报告 cosine 0.9999999999999998 | 记录输入、固定部署 profile 相对于所绑定 reference 的一致性 |
| Orin SmolVLA | 34560 个完整标准化/native tensor | MSE=0、cosine=1、byte-exact | 同 Orin 优化路径一致，不是官方源 BF16 等价 |
| Orin RDT-1B | 34560 个完整 typed tensor | MSE=0、cosine=1、byte-exact | 同上 |
| Orin pi0 | 23040 个完整标准化/native tensor | MSE=0、cosine=1、byte-exact | 同上 |
| Orin pi0.5 | 23040 个完整标准化/native tensor | MSE=0、cosine=1、byte-exact | 同上 |
| Orin CogACT | 57600 个完整 typed tensor，其中 23040 个 exact-only 辅助输出 | 浮点 MSE=0、cosine=1；RNG/count exact | 同上，且保留约 7.63B 公开依赖配置身份 |
| H20 native Qwen 四 profile | 46080 个完整 token/logit tensor；另有 80 项生命周期断言 | 完整 tokens/BF16 logits byte-exact；reset、双 Session interleave、拒绝不改 committed outputs、重建通过 | 固定 native profile 和所绑定参考的一致性，不是任意流式/文本生成质量 |
| Orin Qwen0.8B | 所有 184320 个 token 值；320 组完整最终 logits/IDs 见证，覆盖每进程首/末16观测周期，其中160组成对比较 | 全 finite、byte-exact，MSE=0、cosine=1；reset 和独立进程重复通过 | 完整 tokens 每请求已核验；不是每请求/每中间步 logits 都持久化；并发 Qwen Session 未测 |

来源：[H20 complete-output CSV](../../artifacts/recovery-audit-20260909/raw-five-summary-001/complete-output-quality.csv)、[Orin 质量 CSV](../../artifacts/orin/20261024-07/tables-001/quality-table.csv)、[Orin Qwen 完整数据审计](../../artifacts/orin/20261025-21/final-export-006-retry-001/data-audit.json)。既有 Orin 完整字节审计是质量依据；本次汇总不声称重新读取每一个历史动作 tensor。

### 4.2 J6M 有限检查与未过项

| 对象 | 指标 | 分类 |
|---|---|---|
| Prefix V5，1 个历史校准样本 | 33 输出 finite，mask exact；最低 cache cosine=0.9940398655074391，最大 cache abs error=1.4026820659637451 | 有限 cache 检查；不是完整 action、held-out 或无损量化通过 |
| Step-060，1 次 solver update | action cosine=0.9999994983908165；MSE=8.980497730075133e-7；max-abs=0.003674030303955078；离散 step index exact | 局部门槛通过，但非 bit-exact、非最终完整动作 |
| Finish crop，1 个 smoke 输入 | `[1,50,6]` 的 300 元素有限，精确等于 pinned ONNX Slice，max/mean-abs=0 | 只证明裁剪正确，不证明上游 action 质量 |
| full-chain-v1 文件链 pilot | cosine=0.999903171999195；MSE=0.0003810012909923388；max-abs=0.050837501883506775；mean-abs=0.01609863852150738 | **未过**原 `max-abs<=0.05、mean-abs<=0.01` 门槛；高 cosine 不覆盖失败 |
| Stage-061 扩大量化候选 | cosine=0.3523425661、MSE=7.7069833839；无 HBM 编译结果 | 质量失败，不纳入部署结果 |

`full-chain-v1` 的 148.238 s 是包含 SSH/文件传输/加载/dump 的命令墙钟，不是 E2E；V5 的阶段性能通过也不追溯证明该旧文件链质量通过。本轮不追加量化搜索、校准 sweep 或全模型 FP32 部署。

来源：[最终 Prefix 检查](../../artifacts/j6m-performance-20260914/prefix-v5-output-check.json)、[Step 检查](../../artifacts/j6m-performance-20260914/step-output-check.json)、[Finish 检查](../../artifacts/j6m-performance-20260914/finish-smoke-output-check.json)、[历史完整链及量化负结果](smolvla_j6m_stage060_20260911.md)。

### 4.3 跨平台与官方源差异

| 比较 | 已有反例 / 限制 | 写作要求 |
|---|---|---|
| Orin Qwen0.8B 对 H20/source | 只有 7/16 组完整 tokens 相同，9/16 不同；最终 logits 最坏 MSE=8.65417583235，最低 cosine=0.697306153915 | 补充误差必须保留，不能写“与官方源完全等价” |
| Orin pi0 / pi0.5 对 H20 | 原 normalized gate 均为 0/16 通过；pi0 聚合 normalized cosine约0.6685354882、MSE约0.06826529088 | 不把同 Orin Arena exactness 等同于跨精度/跨平台源等价 |
| Orin RDT 对源/CPU/H20 | 历史 vision/text 和最终 action 非零误差保留；完整动作 H20 最坏 native MSE约0.0444664986 | 不删除失败，不作为新门槛替代已经冻结的同 Orin比较 |
| 机器人任务质量 | 有限观测、固定 profile/noise；没有实际成功率、闭环安全或 physical action calibration 评测 | 使用“记录工作负载的一致性”，不用“保证机器人任务无损” |

来源：[Orin claim-gap](orin_claim_gap_20261021.md)、[pi0 H20 补充审计](../../artifacts/orin/20261023-25/audit-001.json)、[Qwen 最终数据审计](../../artifacts/orin/20261025-21/final-export-006-retry-001/data-audit.json)。

## 5. CDF、抖动与原始样本入口

以下图均引用已有审计的原始曲线，不把不同平台、模型、阶段或计时边界的样本合并成一条 CDF。每条正式配置曲线通常 n=5120；同请求 TTFT/full16 是两条相关分布。保留长尾、独立进程差异和旧失败，不能截尾后重新宣布达标。

| 平台 / 内容 | PNG | PDF | 原始数据 / 审计入口 |
|---|---|---|---|
| H20 五 VLA 原始输入，独立系列 | [PNG](../../artifacts/recovery-audit-20260909/raw-five-summary-001/host-pipeline-cdf.png) | [PDF](../../artifacts/recovery-audit-20260909/raw-five-summary-001/host-pipeline-cdf.pdf) | [raw-host-formal](../../artifacts/recovery-audit-20260909/raw-host-formal/)、[report](../../artifacts/recovery-audit-20260909/raw-five-summary-001/report.json) |
| H20 native Qwen 首 token | [PNG](../../artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-first/latency-cdf.png) | [PDF](../../artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-first/latency-cdf.pdf) | [四 profile 原始 CDF / report](../../artifacts/recovery-audit-20260909/qwen-native-formal-20260910/) |
| H20 native Qwen full16 | [PNG](../../artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-full/latency-cdf.png) | [PDF](../../artifacts/recovery-audit-20260909/qwen-native-formal-20260910/figures-full/latency-cdf.pdf) | 同上，first/full 分开 |
| Orin SmolVLA | [PNG](../../artifacts/orin/20261021-00/smolvla-raw-formal-001/audit-output/smolvla-e2e-cdf.png) | [PDF](../../artifacts/orin/20261021-00/smolvla-raw-formal-001/audit-output/smolvla-e2e-cdf.pdf) | [模型/CSV/图哈希清单](../../artifacts/orin/20261024-07/input-manifest-001.json) |
| Orin RDT | [PNG](../../artifacts/orin/20261022-01/formal-cdf-001/e2e-cdf.png) | [PDF](../../artifacts/orin/20261022-01/formal-cdf-001/e2e-cdf.pdf) | 同上；同目录保留 process-level 曲线 |
| Orin pi0 | [PNG](../../artifacts/orin/20261023-42/publication-curves-001/cdf-pooled.png) | [PDF](../../artifacts/orin/20261023-42/publication-curves-001/cdf-pooled.pdf) | 同上；同目录保留 process-level 曲线 |
| Orin pi0.5 | [PNG](../../artifacts/orin/20261023-58/final-001/publication-curves-001/cdf-pooled.png) | [PDF](../../artifacts/orin/20261023-58/final-001/publication-curves-001/cdf-pooled.pdf) | 同上；同目录保留 process-level 曲线 |
| Orin CogACT | [PNG](../../artifacts/orin/20261024-05/publication-curves-001/cdf-pooled.png) | [PDF](../../artifacts/orin/20261024-05/publication-curves-001/cdf-pooled.pdf) | 同上；同目录保留 process-level 曲线 |
| Orin Qwen0.8B TTFT | [PNG](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/ttft_cdf/latency-cdf.png) | [PDF](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/ttft_cdf/latency-cdf.pdf) | [全部 raw / 四份 exact-rank CDF / stats](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/) |
| Orin Qwen0.8B full16 | [PNG](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/full16_cdf/latency-cdf.png) | [PDF](../../artifacts/orin/20261025-21/final-export-006-retry-001/performance/full16_cdf/latency-cdf.pdf) | 同上；[实际视觉核验](../../artifacts/orin/20261025-21/cdf-visual-review-006.json) |
| J6M Prefix V2/V5 | [PNG](../../artifacts/j6m-performance-20260914/final-prefix-v2-v5-cdf/latency-cdf.png) | [PDF](../../artifacts/j6m-performance-20260914/final-prefix-v2-v5-cdf/latency-cdf.pdf) | [stage summary](../../artifacts/j6m-performance-20260914/final-stage-summary/report.json) |
| J6M Step-060 | [PNG](../../artifacts/j6m-performance-20260914/final-step-060-cdf/latency-cdf.png) | [PDF](../../artifacts/j6m-performance-20260914/final-step-060-cdf/latency-cdf.pdf) | 同上 |
| J6M Finish crop | [PNG](../../artifacts/j6m-performance-20260914/final-finish-crop-cdf/latency-cdf.png) | [PDF](../../artifacts/j6m-performance-20260914/final-finish-crop-cdf/latency-cdf.pdf) | [Finish summary](../../artifacts/j6m-performance-20260914/final-finish-summary/report.json) |

**抖动结论：**Orin 五个 VLA 优化配置的 pooled Std 分别为 2.034434、21.612032、13.313744、9.474035、2.970001 ms，均未达到草稿的 `<1.2 ms`。单独 35 分钟 SmolVLA 单 worker 的 1.060070 ms 不替代正式 pooled 结果。J6M Step Std=1.164 ms 是单步阶段，不是完整模型达标。H20 native Qwen 的 full16 最大值超过 1 s，长尾不得隐藏。

代表曲线如下，其他配置和 PDF 见上表。

![H20 五模型原始输入 CDF](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/recovery-audit-20260909/raw-five-summary-001/host-pipeline-cdf.png)

![Orin SmolVLA 完整 E2E CDF](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261021-00/smolvla-raw-formal-001/audit-output/smolvla-e2e-cdf.png)

![J6M Prefix V2 V5 阶段 CDF](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/j6m-performance-20260914/final-prefix-v2-v5-cdf/latency-cdf.png)

![Orin Qwen0.8B full16 CDF](/home/zhangzimo/.codex/worktrees/f801/edge-fm-x/artifacts/orin/20261025-21/final-export-006-retry-001/performance/full16_cdf/latency-cdf.png)

## 6. 算子、Runtime 与回集成消融

### 6.1 H20：固定 compiler recipe，真实模型签名

单位为 us；每项是 **5 个进程各自 graph-batch device timing 中位数的均值**，不是逐调用 E2E 均值/CDF。候选完整输出均 byte-exact，未因此自动选入完整模型。

| Family | ATen us | Candidate us | 时延降低 % | Candidate |
|---|---:|---:|---:|---|
| Attention | 103.301055 | 103.313151 | -0.0117 | Inductor ATen |
| GEMM / Linear | 11.540214 | 12.870522 | -11.5276 | Inductor autotune |
| LayerNorm | 13.816088 | 13.716502 | 0.7208 | Inductor ATen preserving |
| Embedding | 2.142171 | 1.865898 | 12.8969 | Inductor ATen |
| RoPE frequencies | 21.332679 | 8.486607 | 60.2178 | Inductor ATen |
| VLA solver/update | 4.289051 | 2.203624 | 48.6221 | Inductor ATen |

Attention 基本不变，Linear 变慢；Norm 一行是 LayerNorm，不宣称已独立覆盖 RMSNorm。VLA solver 是实际三算子更新片段，不是完整 action model。该表支持固定配方选择的微基准，不支持“自主发明全部 kernel”或“微基准比率就是模型加速比”。

来源：[六算子报告](h20_operator_table_20260909.md)、[CSV](../../artifacts/recovery-audit-20260909/operator-six-summary-002/operator-table.csv)、[来源及配方哈希](../../artifacts/recovery-audit-20260909/operator-six-summary-002/report.json)。

### 6.2 Orin：真实签名、分开报告数值合同

单位为 us，比率为 before/after。resident 不含逐请求搬运；IO 列也不是完整主机预处理/E2E。

| Family | Resident before | Resident after | 比率 | IO before | IO after | 比率 |
|---|---:|---:|---:|---:|---:|---:|
| LayerNorm | 186.408479 | 43.062342 | 4.328805 | 489.500161 | 297.729034 | 1.644113 |
| GEMM | 38.142144 | 26.338613 | 1.448146 | 59.466626 | 58.993376 | 1.008022 |
| Attention Core | 2746.298173 | 1883.267162 | 1.458263 | 3852.482448 | 3107.671844 | 1.239668 |
| VLA state projection | 29.811383 | 30.402696 | 0.980551 | 39.361035 | 40.169228 | 0.979880 |
| Language Embedding | 16.460231 | 16.269644 | 1.011714 | 28.954250 | 28.883931 | 1.002435 |
| RoPE/cache | 41.206087 | 30.265394 | 1.361492 | 58.652144 | 51.242712 | 1.144595 |

必须随表保留的合同：

- Norm 为原 FP32 island，GEMM 为已观测 FP16 数据/权重，两者容差通过但非 byte-exact；Attention 输出 byte-exact，已观测 FP16 QK/PV 与 FP32 Softmax。
- State 只能称 **configured-policy 对比**：baseline Myelin 内部不透明，未建立相同 effective precision，且候选更慢。
- Embedding/RoPE 是新建 standalone FLOAT-only SDK numerical policy，完整输出 byte-exact；不是全模型 FP32 回退，也没有证明隐藏存储/指令精度。Embedding 两边界各有 3/5 负收益对，RoPE resident Max 更高。
- 原四类用 vendor event / 1000 ms warmup；新两类用原生 CUDA-event bracket / 128 warmup+1024 measured，不混为一个测量实现。Attention 接受重试保留每进程1040条，总20800条，不截断为1024；此前20479条的不完整批次仍拒绝。
- 12 个已接受构建控制器耗时合计 116.830536192 s 是对应构建成本，不是全部历史搜索/Agent reasoning 时间；未知成本不填 0。

来源：[Orin 六类消融](orin_operator_ablation_results_20261022.md)、[完整合同和构建成本](../../artifacts/orin/20261024-07/operator-costs-001/summary.json)、[Embedding/RoPE 新配方及 CDF](orin_embedding_rope_float_islands_20261024.md)、[两类原始统计](../../artifacts/orin/20261024-06/publication-001/operator-statistics.csv)。该报告末尾旧“4/5、Qwen未测”是历史状态，完成范围以本汇总和最终验收为准。

### 6.3 Orin 完整 VLA 回集成，不能只报正向微基准

| 候选 | 完整模型质量 | 完整模型结果 | 论文结论 |
|---|---|---|---|
| Norm | 标准化 action 最坏 MSE=5.30081810e-6、最低 cosine=0.999998499934，通过主门槛；native 最坏 MSE=0.0121000582 单列 | 10240 measured / 23040完整 tensor；E2E mean 125.900306 -> 133.994095 ms，所有5对均更慢，慢约6.4287% | 微基准4.329x不转化为整模型收益，保留负结果 |
| GEMM | normalized MSE=1.324344964e-5；对 unchanged-source rebuild 仍为1.375922937e-5，均失败 | 未准入正式完整模型 E2E | 不用高 cosine 覆盖 MSE 失败，不计集成加速 |
| Attention | 34组完整 normalized/native输出对；normalized最坏MSE=1.2796944331e-5；对 rebuild=1.2985515613e-5，失败 | 完整链执行过，但未启动正式 raw E2E 采样 | 不能把微基准1.458x写成模型加速 |
| State / Embedding / RoPE | State 负收益/精度可见性有限；其余是独立数值合同微基准 | 未补充合格的完整 VLA 集成收益 | 不乘以 replay 比率、不外推为所有模型生效 |

来源：[Norm 正式 E2E](orin_operator_layernorm_raw_e2e_20261022.md)、[回集成验收口径](orin_integration_requirement_reconciliation_20261024.md)、[GEMM 控制实验](orin_operator_gemm_integration_20261022.md)、[Attention 质量失败](orin_operator_attention_integration_20261024.md)。

### 6.4 J6M：部署放置收益与历史支持性探针分开

| 比较 | Before | After | 实际支持的结论 |
|---|---:|---:|---|
| Prefix V2 -> V5 正式均值 | 1402.998 ms | 553.930 ms | 2.532809x，两个完整 HBM 版本比较；不是单算子因果隔离 |
| Prefix 短测外部 CPU Scatter | 63个，约812.164 ms | 0个 | 实板确认预期 placement 变化；不能据此声称最终 action 质量无损 |
| Step-057 -> Step-060 五帧短测 | 8893.975 ms | 90.088 ms | 约98.73x，Conv/MatMul 的 CPU -> BPU 放置/精度配方变化；非同精度 kernel 消融、非正式 E2E |

以下是**已经存在的历史合成固定 shape FP32 -> INT8 支持性探针**，不是本轮新跑、不是部署 FP32 候选、不是同精度 Agent 调优，也不是 held-out/模型质量证据。实际 Step-060 是 FP16+INT16，不是此 INT8 recipe。

| 历史探针 | FP32 ms | INT8 ms | 原因与边界 |
|---|---:|---:|---|
| Attention | 29.123835 | 0.257960 | CPU fallback节点5 -> 0，包含精度与placement变化 |
| GEMM + LayerNorm + Embedding | 7.384600 | 0.314685 | 复合探针，不拆成三个独立实测结论；fallback4 -> 1 |
| RMSNorm | 0.874710 | 0.432790 | 合成固定shape，不替代真实模型回集成 |
| RoPE | 0.243380 | 0.267290 | INT8更慢，负结果保留 |
| Conv + vision | 133.215942 | 0.412560 | fallback5 -> 0，不是同精度单kernel收益 |
| Flow-matching step | 0.700370 | 0.276120 | 更新片段，不是完整VLA |

来源：[历史探针复核清单](../../artifacts/j6m-performance-20260914/reused-operator-audit.json)、[Stage-060报告](smolvla_j6m_stage060_20260911.md)、[最终Prefix短测/正式口径](smolvla_j6m_performance_20260914.md)。探针帧数为已记录的100/200帧，不套用五进程正式模型协议。

### 6.5 H20 native Session 边界补充，不与2.1混合

单位 ms。每个数据组独立为15配置/75进程/76800 measured；off/shared/replay 是同编译 Session 内的三种 runtime policy，不是官方 PyTorch baseline。

| 模型 | Resident off / shared / replay mean | Resident replay P99 | Host-model-tensor off / shared / replay mean | Host-model-tensor replay P99 |
|---|---|---:|---|---:|
| SmolVLA | 117.632 / 118.380 / 47.113 | 49.798 | 121.939924 / 120.539683 / 47.810259 | 48.941977 |
| RDT-1B | 176.711 / 172.734 / 164.398 | 168.415 | 177.103208 / 172.187518 / 164.879751 | 166.872083 |
| pi0 | 139.629 / 139.362 / 84.651 | 86.143 | 143.922713 / 139.762972 / 86.381092 | 90.053202 |
| pi0.5 | 180.263 / 169.153 / 95.344 | 113.096 | 169.710285 / 177.257704 / 94.886679 | 97.238034 |
| CogACT | 84.870 / 86.191 / 76.231 | 78.861 | 86.785191 / 85.320263 / 76.242278 | 77.524266 |

这两组不是同一次请求截取出来的分段，不用组间差值估算搬运成本。H20 pi0.5 resident replay 的 sampled device memory 从7397增至8185 MiB，不能宣称 replay 一定省内存。

另有 CogACT 自主 C++ RNG 的同边界正式对照：official mean/P99=129.258/232.080 ms，autonomous native=88.751/95.811 ms；均值比1.456x，保留官方重尾。此边界含 RNG/model/sampler/CUDA completion，排除静态输入 H2D、binding、预处理和输出转换，不能替换2.1的79.971651 ms一行。

来源：[resident表](h20_vla_formal_table_20260909.md)、[resident CSV](../../artifacts/recovery-audit-20260909/h20-five-summary/cuda-session-table.csv)、[host-model-tensor表](h20_host_tensor_formal_table_20260909.md)、[host CSV](../../artifacts/recovery-audit-20260909/host-five-summary/cuda-session-table.csv)、[CogACT同边界正式报告](cogact_timing_boundary_formal_20260910.md)。

## 7. 资源、功耗与运行稳定性

| 平台 / 实验 | 已有数据 | 不能据此推断 |
|---|---|---|
| Orin 五VLA | 60正式进程、78667条telemetry；最大 sampled process VmHWM：SmolVLA3114.136719、RDT6811.164062、pi04161.277344、pi0.53228.792969、CogACT4352.203125 MiB | VmHWM含初始化/warmup，不是模型私有GPU峰值；CPU/GPU共享RAM不重复相加 |
| Orin Qwen0.8B | off/verified sampled VmHWM=6729.117188/6706.246094 MiB；4段RSS观察3598/5083/7216/1734行，覆盖10个native出生 | 前3段保留cgroup metadata错误；有效段之间缺口574.391277/317.417639/163.190024s，未插值；不是连续1Hz或匹配的请求峰值，也不证实省内存 |
| Orin热稳态/生命周期 | 单worker35.016分钟；2022.466141s实测模型工作；12对Session生命周期；覆盖的CUDART设备申请/释放217/217，1024稳态ModelRun未观察到覆盖API分配事件 | 不等于全部CPU/private SDK/driver无分配、全局zero-malloc或普遍无泄漏；CUPTI权限失败仍保留 |
| Orin匹配功耗窗口 | 常规/共享流/replay分别38.438594/38.552769/42.267198 W；2423 raw /1854 eligible功率样本 | 是客户端观测输入rail窗口，不是独立模型能量/请求；replay功耗更高，MAXN不是60W |
| J6M | Prefix V5静态HBM plan约543.660 MiB；正式进程Peak RSS33.75 MiB；其他阶段见2.3。温度/频率before/after快照及部分1Hz观察保留 | 静态plan、RSS和driver-managed device allocation不是同一个量；快照不证明全程恒频/无争用 |
| H20 Qwen native | 0.8B full16 peak sampled device约2393 MiB；2B约4995 MiB | 不等于官方Torch allocated计数，不能与Orin进程VmHWM直接作节省百分比 |

来源：[Orin核心资源/功耗口径](orin_core_acceptance_20261024.md)、[Orin60进程资源表](../../artifacts/orin/20261024-07/resources-001/per-process-resources.csv)、[Qwen资源审计](../../artifacts/orin/20261025-21/final-export-006-retry-001/resources/summary.json)、[J6M阶段报告](smolvla_j6m_performance_20260914.md)。Orin最终独立核验已记录原Supervisor恢复、125资产/6映射SDK文件与11个模型/控制器出生终态；恢复进程不等于应用功能健康验收。

## 8. 未测项与结论边界

| 项目 | 当前分类 | 论文中如何处理 |
|---|---|---|
| J6M完整SmolVLA resident Prefix -> 10 Step -> Finish | **未测完整resident E2E；旧文件链质量失败** | 主表写“未取得合格E2E结果”，阶段结果单独表；不能阶段均值求和或填墙钟 |
| J6M上RDT/pi0/pi0.5/CogACT/Qwen | deferred，未运行 | 不写成0、不写成已证明不支持或OOM；不得借用H20/Orin值 |
| j6m-1、J6P实机VLM | 本次无相应正式结果；J6P已有部署设计不等于实测 | 不把设计文档或J6M结果视为J6P验证 |
| Orin Qwen2B | 用户取消 | 保留H20已有结果；Orin单元格标“cancelled”，不是unsupported |
| Orin第二功耗模式 | 权限/切换路径限制已评估；无第二模式性能实测 | 可说明限制，不补写30W/60W主表，MAXN不改名 |
| 跨设备官方源质量完全等价 | 存在明确非零误差/失败 | 主门槛和补充误差分表，不删除失败、不放宽阈值 |
| 原始输入范围 | VLA有限观测循环；H20 Qwen固定图像/prompt；Orin Qwen16帧单episode循环 | 不写成1000条独立轨迹、大规模held-out泛化或真实机器人成功率 |
| 同精度独立RMSNorm全面覆盖 | H20/Orin Norm主消融为LayerNorm；J6M RMSNorm仅历史不同精度合成探针 | 不能合并包装成全部真实模型RMSNorm优化已验收 |
| 自主Agent kernel发明/全局搜索收益 | 固定配方、特定签名与实际构建成本，有负结果；全局推理/历史搜索成本未知 | 使用“Agent辅助的固定compiler recipe/图改写评估”，不宣称普遍自主创新 |
| zero-copy / 全局零分配 / 精确私有GPU峰值 | 未由本组数据证明 | 明确H2D/D2H、CPU文本分配、SDK/driver可见性和共享RAM边界 |
| TensorRT-Edge-LLM / 草稿1.49x、3.8x通用结论 | 当前Orin baseline是同engine常规TRT，无该vendor baseline | 使用真实对照名称和本表数值，不套用草稿倍数 |
| deadline/物理闭环连续性/机器人质量 | 未提供K_exec、控制间隔、队列消费预算和机器人任务评测 | 不从吞吐或高cosine直接推断硬实时、安全或闭环成功 |

模型/engine等大复现产物已按用户许可进行过清理，尤其Orin完成的五VLA；保留测量/质量审计/曲线/来源记录并不代表所有历史engine仍可直接执行。后续重跑需要重新准备已清理payload。本次只核对汇总引用和保存证据，不恢复大模型或触发新实验。

## 9. 写论文时的建议组织

1. **VLA主表分平台/比较对象：**H20 official PyTorch vs Session；Orin同精度常规TRT vs已声明优化模式；J6M仅阶段表或完整E2E空缺标注。不要合成统一“EdgeFM跨硬件加速倍数”。
2. **CDF单独交代边界：**每模型/配置保留原始5120点及完整tail；给出pooled Std和最大时延，不用单worker soak掩盖pooled目标未达。
3. **质量表分三层：**同平台优化一致性、跨H20/官方源补充误差、J6M单样本阶段近似/失败；整数输出精确核验和浮点误差分别写。
4. **消融同时报micro和integration：**保留H20 Linear负收益、Orin State/Embedding有限收益、Norm完整E2E变慢和GEMM/Attention质量失败；J6M不同精度历史探针放附录而非同精度Agent消融主表。
5. **Qwen作VLM适配补充：**H20两尺寸与Orin0.8B展示完整视觉/语言/状态链可运行，明确固定profile和计时边界；Orin2B取消不影响保留H20已测2B。

## 10. 证据与复查入口

| 数据包 | 文档 | 机器可读入口 |
|---|---|---|
| H20五VLA原始输入 | [正式交付](h20_vla_experiment_delivery_20260910.md) | [raw-five report](../../artifacts/recovery-audit-20260909/raw-five-summary-001/report.json) |
| H20六算子 | [正式算子表](h20_operator_table_20260909.md) | [operator report](../../artifacts/recovery-audit-20260909/operator-six-summary-002/report.json) |
| H20 Qwen native | [完整部署/质量/生命周期](qwen35_native_formal_20260910.md) | [四profile数据](../../artifacts/recovery-audit-20260909/qwen-native-formal-20260910/) |
| J6M SmolVLA | [最终阶段性能](smolvla_j6m_performance_20260914.md) | [final-stage audit](../../artifacts/j6m-performance-20260914/final-stage-summary/report.json)、[Finish audit](../../artifacts/j6m-performance-20260914/final-finish-summary/report.json) |
| Orin五VLA与公共消融 | [交付索引](edgefm_orin_experiment_delivery.md)、[核心验收](orin_core_acceptance_20261024.md) | [五模型summary](../../artifacts/orin/20261024-07/tables-001/summary.json) |
| Orin Qwen0.8B及最终范围 | [最终31项验收报告](orin_selected_goal_acceptance_20261026.md) | [完整export](../../artifacts/orin/20261025-21/final-export-006-retry-001/summary.json)、[发布后31项review](../../artifacts/orin/20261025-21/selected-goal-review-002.json) |
| 原始论文强主张边界 | [H20审查](h20_paper_scope_review_20260910.md)、[Orin claim-gap](orin_claim_gap_20261021.md) | 原PDF未修改，SHA256 `92a3af4618a08c7ea66e62b928c3bb32d326e0445a491eee079ecccf70adf997` |

本汇总不重新认证已清理模型的二进制可运行性。数值、主表行、CDF文件与源记录的本机核对可运行下列命令；只读取已保存的数据，不连接板卡、不运行模型、不控制服务：

```bash
cd /home/zhangzimo/.codex/worktrees/f801/edge-fm-x
python3 -B artifacts/paper-summary-20261026/verify_summary.py --output /tmp/edgefm-three-platform-summary-NEW.json
```

该复查是汇总一致性检查，不替代既有完整动作字节审计或新的硬件实验。对应首次复查结果会保存在 [汇总验证记录](../../artifacts/paper-summary-20261026/verification-001.json)。
