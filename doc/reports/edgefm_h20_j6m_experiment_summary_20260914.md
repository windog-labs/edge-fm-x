# EdgeFM H20 and J6M Experiment Summary

Updated: 2026-09-14

本文汇总当前已经完成并有原始证据支持的 H20 与 J6M 实验。H20 结果来自
`zzm-h20-x8-1/2` 的 GPU 部署；J6M 结果来自 `j6m-2` 的 Horizon BPU
部署。两类结果的计时边界不同，不能直接把数值混合成跨硬件加速比。

## Coverage

| Platform | Completed models / stages | Main evidence | Current boundary |
|---|---|---|---|
| H20 | SmolVLA, RDT-1B, pi0, pi0.5, CogACT；Qwen3.5-0.8B/2B | [H20 delivery](h20_vla_experiment_delivery_20260910.md) | Host and resident CUDA profiles; fixed workloads |
| J6M | SmolVLA Prefix V2/V5, Solver Step-060, Finish crop | [J6M delivery](smolvla_j6m_performance_20260914.md) | Stage-level HBM timing on `j6m-2` |
| Orin | None | Deferred | No hardware result in this collection |

H20 的五模型 VLA 结果包含完整动作输出核验；J6M 本轮聚焦已有低精度 HBM
的阶段耗时、抖动、内存和有限输出检查。J6M 没有执行 RDT、pi0、pi0.5、
CogACT 或 Qwen，Orin 也没有执行。

## H20 VLA Original-Input Pipeline

每个 worker 使用 128 次 warmup 和 1024 次测量，计时从解码后的观测进入
CPU RAM 开始，到完整 CPU 动作输出结束，包含预处理、H2D、推理、后处理
和 D2H。吞吐是新计算 action chunks/s，不是机器人伺服频率。

| Model | Official PyTorch mean ms | Official p99 ms | Official chunks/s | Native Session mean ms | Native p99 ms | Native chunks/s |
|---|---:|---:|---:|---:|---:|---:|
| SmolVLA | 206.162 | 220.580 | 4.851 | 50.727 | 53.682 | 19.713 |
| RDT-1B | 239.686 | 301.829 | 4.172 | 221.804 | 230.312 | 4.508 |
| pi0 | 221.305 | 350.641 | 4.519 | 108.152 | 113.977 | 9.246 |
| pi0.5 | 256.840 | 274.030 | 3.893 | 116.854 | 122.893 | 8.558 |
| CogACT* | 120.286 | 127.399 | 8.314 | 79.972 | 82.199 | 12.504 |

`*` CogACT 使用实际公开依赖配置；模型身份和参数规模限制见原始报告，
不要将其简单等同为名义 3B 模型。五个模型的完整输出均通过记录 workload
上的 byte-exact 检查，最小报告 cosine 为 `0.9999999999999998`；这属于
固定输入和固定 profile 的一致性结果，不是物理机器人成功率。

原始输入表、CDF 图、完整输出质量和审计证据：

- [H20 original-input table](h20_vla_experiment_delivery_20260910.md)
- [H20 formal table](h20_vla_formal_table_20260909.md)
- [H20 host-model-tensor table](h20_host_tensor_formal_table_20260909.md)

## H20 Resident Session Comparison

该表使用同一编译 C++ Session 的 resident model-tensor 边界，计时包含模型
执行、输出处理和同步 Session 完成，不包含输入预处理、H2D/D2H、初始化和
机器人传输。因此它不能和上一节的 sensor-to-action 原始输入表合并。

| Model | Off mean ms | Shared-stream mean ms | Replay mean ms | Replay p99 ms | Replay chunks/s |
|---|---:|---:|---:|---:|---:|
| SmolVLA | 117.632 | 118.380 | 47.113 | 49.798 | 21.226 |
| RDT-1B | 176.711 | 172.734 | 164.398 | 168.415 | 6.083 |
| pi0 | 139.629 | 139.362 | 84.651 | 86.143 | 11.813 |
| pi0.5 | 180.263 | 169.153 | 95.344 | 113.096 | 10.488 |
| CogACT | 84.870 | 86.191 | 76.231 | 78.861 | 13.118 |

五个模型均为五进程、每进程 128 warmup + 1024 measured。该表的输出与对应
官方或同 artifact reference 在记录 workload 上 byte exact。

## H20 CogACT Boundary-Aligned Comparison

CogACT 另有一组边界对齐的官方实现与 autonomous native Session 对比。两边
都包含随机数准备、模型或原始 VLM/DDIM sampler 和 CUDA completion，排除静态
输入 H2D、binding、预处理和输出转换。

| Configuration | Mean ms | P50 ms | P95 ms | P99 ms | Calls/s |
|---|---:|---:|---:|---:|---:|
| Official PyTorch | 129.258 | 124.006 | 177.383 | 232.080 | 7.736 |
| Autonomous native Session | 88.751 | 89.199 | 90.683 | 95.811 | 11.268 |

全样本均值比为 `1.456x`，native 平均时延降低 31.34%。官方 worker 的重尾
样本保留在统计中，没有删除异常值。该结果与上一节的 CogACT 原始输入结果
使用不同计时边界，不能互换。

## H20 Operator Ablation

下表来自五进程算子微基准的历史正式结果，单位为微秒。候选实现是在归档的
真实模型 shape 上选择的固定 compiler recipe；它们不是已经集成进所有完整
模型的通用 kernel。

| Family | Baseline us | Candidate us | Latency reduction |
|---|---:|---:|---:|
| Attention | 103.301 | 103.313 | -0.012% |
| GEMM / Linear | 11.540 | 12.871 | -11.528% |
| LayerNorm | 13.816 | 13.717 | 0.721% |
| Embedding | 2.142 | 1.866 | 12.897% |
| RoPE | 21.333 | 8.487 | 60.218% |
| VLA solver/update | 4.289 | 2.204 | 48.622% |

Attention 基本不变，Linear 变慢；LayerNorm 这一行覆盖 normalization family，
不额外宣称独立 RMSNorm 或 autonomous kernel invention 已完成。详见
[H20 operator table](h20_operator_table_20260909.md)。

## H20 Qwen3.5 Native Deployment

Qwen 的结果是 H20 固定 profile 的原生 C++ Session 部署。输入包含
`input_ids [1,327]`、`attention_mask [1,327]`、`pixel_values [1200,1536]`
和显式 `accepted` 状态；输出包含 token 与 BF16 logits。

| Model | Profile | Mean ms | P50 ms | P95 ms | P99 ms | Calls/s | 16-token tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | First token | 96.905 | 99.481 | 104.489 | 114.008 | 10.319 | - |
| Qwen3.5-2B | First token | 90.658 | 85.697 | 104.991 | 137.605 | 11.030 | - |
| Qwen3.5-0.8B | Fixed 16 tokens | 369.290 | 322.653 | 573.796 | 625.620 | 2.708 | 43.326 |
| Qwen3.5-2B | Fixed 16 tokens | 370.497 | 333.449 | 583.324 | 629.294 | 2.699 | 43.185 |

四个 native bundle 均通过完整 token/logit byte-exact 检查和状态生命周期审计。
这是固定图像、prompt 和生成长度 profile，不代表任意流式输入或每 token 的
独立实时延迟。

详见 [Qwen native report](qwen35_native_formal_20260910.md) 和
[Qwen multimodal report](qwen35_natural_profile_20260909.md)。

## J6M SmolVLA Existing HBM

J6M 结果来自 `j6m-2`，序列号 `0e190a0d3392371c`，运行时为 UCP 3.14.7、
HBRT 4.9.7、BPU library 2.2.6。每阶段使用五个独立顺序进程，每个进程
128 warmup + 1024 measured，共 5120 个有效样本。计时是 `hrt_model_exec`
的单次模型调用边界，包含任务构造、提交和完成等待；输入文件加载、输出
处理、SSH 和文件编排不在时延内。

| Stage | HBM role | Mean ms | P50 ms | P95 ms | P99 ms | Max ms | Stage calls/s | Peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Prefix V2 | original low-precision prefix | 1402.998 | 1402.430 | 1408.200 | 1412.990 | 1431.410 | 0.712759 | 35.8125 |
| Prefix V5 | lowered prefix | 553.930 | 553.805 | 555.397 | 557.017 | 570.170 | 1.805283 | 33.7500 |
| Solver Step-060 | iterative solver step | 89.921 | 89.834 | 91.249 | 95.632 | 96.615 | 11.120922 | 34.0000 |
| Finish crop | output Slice only | 0.261 | 0.254 | 0.304 | 0.373 | 1.154 | 3828.710 | 27.0625 |

Prefix V5 相对 V2 的正式均值比为 `2.532809x`。短测 profiler 显示 V2 有
63 个外部 CPU `InplaceScatterND` 节点、合计约 812.164 ms，而 V5 为 0 个；
这是两个完整 HBM 版本的比较，不是单算子因果隔离实验。

Finish 是 `[1,50,32] -> [1,50,6]` 的 Slice 输出裁剪，虽然 I/O 表示为 F32，
其源图没有浮点运算，不应被写成 FP32 神经网络部署结果。

## J6M Output and Memory Boundaries

- Prefix V5 一次有限核验：33 个输出有限，mask exact，最小 cache cosine
  `0.9940398655`，最大 cache absolute error `1.402682066`。
- Step-060 一次 solver update 核验：cosine `0.9999994984`，最大绝对误差
  `0.0036740303`，离散 step index exact。
- Finish 一次 smoke 输出核验：300 个元素有限，输出精确匹配 pinned ONNX
  Slice，最大和平均绝对误差均为 0。

这些是固定 sample 的有限一致性检查，不是无损量化、held-out accuracy、完整
动作质量或机器人任务成功率证明。J6M 本轮没有增加 FP32 部署、量化搜索或
校准 sweep。

HBM 静态 memory plan 与 Linux 进程 RSS 分开报告；RSS 不包含所有 driver-managed
device allocation。温度和频率是每个进程的 before/after snapshot，不是连续遥测。

## CDF and Evidence

J6M CDF 图和原始 CSV：

- [Prefix V2/V5 CDF](../../artifacts/j6m-performance-20260914/final-prefix-v2-v5-cdf/latency-cdf.png)
- [Step-060 CDF](../../artifacts/j6m-performance-20260914/final-step-060-cdf/latency-cdf.png)
- [Finish crop CDF](../../artifacts/j6m-performance-20260914/final-finish-crop-cdf/latency-cdf.png)
- [Final evidence index](../../artifacts/j6m-performance-20260914/evidence-index-final.md)

H20 证据索引和原始结果见：

- [H20 experiment delivery](h20_vla_experiment_delivery_20260910.md)
- [H20 operator table](h20_operator_table_20260909.md)
- [Qwen native formal report](qwen35_native_formal_20260910.md)

## Remaining Scope

| Item | Status |
|---|---|
| H20 five-model VLA tables, CDF and output checks | Complete for recorded profiles |
| H20 Qwen3.5 0.8B/2B native deployment | Complete for fixed H20 profiles |
| H20 operator ablation | Complete as archived microbenchmarks |
| J6M SmolVLA stage performance | Complete on `j6m-2` |
| J6M resident full-chain Prefix -> 10 Step -> Finish timing | Subsequent extension |
| Orin VLA main-table experiments | Not performed; hardware/result unavailable |
| J6M RDT, pi0, pi0.5, CogACT, Qwen | Deferred in the SmolVLA-only round |

本文件只汇总已有证据，不把 H20 结果替代 Orin 或 J6M，也不把 J6M 阶段耗时
相加后冒充端到端 E2E 时延。
