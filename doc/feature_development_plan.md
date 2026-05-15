# EdgeFM 开发计划整理

本文用于重新整理 EdgeFM 当前一轮开发计划。目标不是把所有技术细节一次性铺开，而是先把背景、边界、feature 总览、owner 拆分和详细任务组织清楚，便于 3 个人并行推进。

## 0. Owner A 当前状态校正（2026-05-14）

下面是基于当前代码实现的 Owner A 状态校正，优先级高于本文后续历史排期里的旧描述：

- Prefix KV 已完成，不应再列为未完成项。当前实现是连续 per-request/per-layer KV slot：`KVManager` 解析 `prefix_token_ids` 并偏移 write pointer，`Scheduler` 校验 request prefix，`StandardEngine::warmup()` 预填 prefix KV，真实 prefill 跳过 prefix 只写 suffix。
- 已完成的 Owner A 闭环包括：`sampling.max_new_tokens`、device-side token finalize/stop 语义、`last_generate_metrics()`、compact vocab runtime remap/restore 基础能力、非 identity compact vocab 测试、greedy sampler 直接 argmax 优化、decode breakdown profiler、默认关闭的 `lm_head_top1` 实验路径、DeepGEMM probe-only 决策门。
- 未完成或仍处于 probe/deferred 的 Owner A 项：完整 benchmark matrix 报告、DeepGEMM 默认关闭候选接入前置条件、FP8/W8A8 artifact/scale contract、连续 INT8 KV engine side、compact vocab 多模型/大规模验收。
- `lm_head_top1` 当前已有默认关闭实现：`runtime.lm_head_top1.enabled=true` 且 greedy decode 时可绕过 full logits + sampler；除非后续证明 CUDA graph 目标 slice 端到端提升 >= 1% 且 token alignment 全通过，否则继续保持 full logits 默认路径。

## 1. 背景 && 现状

### 1.1 项目当前结构和定位

当前代码库已经有比较清晰的 CUDA runtime 分层：

- `src/engine/`: request 生命周期、scheduler、KV cache、CUDA graph、generate loop。
- `src/models/`: 模型 runtime，目前 Qwen2.5 LLM/VLM 主要走 `src/models/qwen2_5/`。
- `src/layers/`: layer contract、shape/dtype/device 校验、weight 绑定、operator query 构造。
- `src/operators/`: operator registry、operator implementation、`operator_impl_table` 路由和底层 CUDA kernel。
- `src/backends/`: Horizon whole-graph artifact 和 runtime backend。

整体项目的设计理念不是做一个过度通用的大框架，而是围绕现有 runtime 做小步、可验证、容易回退的增量开发。

### 1.2 当前能力现状

| 能力 | 当前状态 | 主要缺口 |
| --- | --- | --- |
| 单请求性能优化 | 已有标准 generate 路径、decode CUDA graph、device-side finalize/stop 语义、greedy direct argmax 和 decode breakdown metrics | 完整 benchmark matrix 报告、`m=1` decode 专项优化仍未进入默认路径 |
| W4A16 量化 | 已有 INT4 groupwise kernel 和 `LinearLayer::forward_int4_groupwise()` 测试入口 | 不是正式 `linear` operator impl，不能通过 `operator_impl_table` 路由，端到端流程不完整 |
| W8A8 量化 | `DType::Int8` 基础类型存在 | 没有 activation quant、scale contract、decode-first int8 linear 正式实现 |
| 词表裁剪 | 已有 compact vocab artifact contract、runtime input/output remap、response restore、非 identity 测试、TRT-Edge-LLM 风格 `vocab_map` packaging/validator 工具、0.5B 真实 checkpoint packaging smoke | 仍缺多模型/大规模验收 |
| 投机采样 | `src/engine/experimental/speculative/EagleEngine` 只有原型头文件，facade 拒绝启用 | 没有 draft model runtime、verify loop、accept/reject/commit、临时 KV workspace |
| KV cache 压缩 | `KVManager` 当前是连续 per-request/per-layer KV buffer | 没有压缩格式、scale metadata、dequant-on-load attention、压缩态连续 buffer 接入 |
| Prefix KV | 已完成连续 KV slot 下的 prefix warmup/reuse | 没有 paged attention、语义化 prefix cache lookup 或 INT8 KV scale contract |

### 1.3 当前关键代码事实

- `src/engine/engine_factory.cpp` 当前在 `speculative.enabled=true` 时直接抛错。
- `src/engine/experimental/speculative/eagle_engine.cpp` 当前为空文件。
- `src/engine/tasks/token_generation/kv_manager.cpp` 当前按 request/layer 分配连续 KV buffer，没有 paged attention 语义。
- `src/layers/linear.cu` 已支持 `.qweight + .scaling_factors` 识别 INT4 groupwise 权重。
- `src/operators/linear_impl.cu` 目前正式注册的 linear impl 主要是 `cublasLt`、`cutlass` 等。
- `src/layers/sampler.cu` 当前实际主路径主要消费 `temperature` 和 `seed`。

### 1.4 必须遵守的框架设计和开发原则

下面这些原则需要写死在本轮开发里，后续 feature 实现都不能绕开：

1. 简单优先。优先做贴着现有结构的小步增量，不为了“看起来更通用”提前引入新的框架层。
2. 不引入 paged attention。KV cache 继续保持连续 per-request/per-layer buffer，不引入非连续索引表、block table 或额外地址翻译层。
3. 不做新的通用 IR 或 decode framework。CUDA path 继续保持当前直接 runtime 结构，speculative 第一版也只是在现有 `StandardEngine::generate()` 附近加最小分支。
4. 共享边界保持稳定。`engine -> model -> layer -> operator` 这条责任链不打乱，layer 继续负责 contract 和 operator query，operator 继续通过 registry 和 `operator_impl_table` 选择。
5. 量化只加载已经量化好的模型。不开发离线量化转换工具，不把复杂格式转换塞进热路径。
6. 默认行为不能回退。所有新能力都必须显式开关；关闭后，原有 FP16/BF16 generate 行为和 correctness 不应变化。
7. benchmark 先行。大一点的设计改动必须有统一 benchmark 口径，否则后续收益判断会失真。
8. B/C 尽量独立开发。Owner B 和 Owner C 的任务默认都应限制在局部目录和局部接口内，不主动发起共享框架重构。
9. 共享接口改动必须先和 Owner A 对齐。尤其是 `StandardEngine::generate()`、`Scheduler::create_context()`、`Context`、`Response`、`KVManager`、`operator_impl_table` schema、`LinearShapeSignature` 这类公共边界。
10. 第一版先做 lossless 或近似 lossless 能力，再做会改变输出分布的 approximate 优化。

### 1.5 当前总体判断

- Owner A 对项目整体最熟，应该承担 runtime owner 和接口 owner 的角色。
- Owner B 和 Owner C 相对不那么熟悉，因此拆分任务时应优先让他们在局部完成闭环，而不是去动大量已有共享接口。
- 第一轮最值得优先闭环的，是单请求标准路径优化、W4A16 已量化模型端到端、speculative greedy correctness。
- W8A8、KV cache 压缩、DeepGEMM 实际接入都应该建立在前面三条线已经有 baseline 和接口边界的前提上。

## 2. 开发需求 && Feature 项目 && 整体开发计划 Overview

### 2.1 本轮开发需求

本轮明确要覆盖的开发需求包括：

1. 量化算子的实现，优先支持 `w4a16`，再推进 `w8a8`。
2. 词表裁剪。
3. 投机采样。
4. KV cache 压缩算法。
5. 单请求场景的性能优化和设计更新。
6. 在 Owner A 侧加入 DeepGEMM 候选路径评估。

### 2.2 Feature 项目总览

| Feature 项目 | 主要 Owner | 第一版范围 | 备注 |
| --- | --- | --- | --- |
| 单请求性能优化和运行时设计更新 | Owner A | benchmark、metrics、`max_new_tokens`、device-side stop flag、token finalize、greedy fast path、`m=1` decode 优化 | 第一优先级 |
| 词表裁剪 | Owner A | compact vocab artifact contract、runtime input/output remap、special token 保留校验 | 归 Owner A，不再单独漂浮 |
| 投机采样 | Owner B | greedy only、same tokenizer/same vocab、draft-only、target verify、accept/reject/commit | 第一版只做最小 speculative greedy |
| W4A16 / W8A8 量化算子 | Owner C | W4A16 operator 化和端到端；W8A8 decode-first | 只加载已量化模型 |
| KV cache 压缩 | Owner A + Owner C | 连续 INT8 KV first，A 负责 engine side，C 负责 attention/kernel side | 第二轮偏后推进 |
| DeepGEMM 候选路径 | Owner A 主导，Owner C 配合 | 先 benchmark 和路由设计，再决定是否轻量接入 | 不作为默认路径 |

### 2.3 整体开发计划 Overview

整体上建议按 4 个阶段推进：

| 阶段 | 核心目标 | 主要产出 |
| --- | --- | --- |
| Phase 0 | 统一边界和 benchmark 口径 | baseline benchmark、metrics、共享接口边界、各 feature 第一版范围确认 |
| Phase 1 | 各 owner 先做各自最小闭环 | A 跑通标准路径 MVP；B 跑通 draft-only；C 跑通 W4A16 operator MVP |
| Phase 2 | 接到真实 generate 路径 | A 完成 greedy 热路径和词表裁剪设计；B 完成 verify/accept；C 完成 W4A16 端到端并开始 W8A8 |
| Phase 3 | 第二功能和稳定化 | A 收口 DeepGEMM/`lm_head_top1` 决策并验证已完成 Prefix KV；B 做 metrics/fallback；C 完成 W8A8，A+C 视情况推进连续 INT8 KV |

### 2.4 第一轮的主成果和取舍

第一轮建议优先确保以下三项闭环：

1. 单请求标准路径优化，且默认 FP16/BF16 correctness 不回退。
2. W4A16 已量化模型端到端可运行，并能稳定 benchmark。
3. speculative greedy correctness 闭环，且失败可退回标准 decode。

如果第一轮资源紧张，下面这些项目可以保留到第二轮或作为 stretch goal：

- W8A8 全量完善。
- 连续 INT8 KV cache 真正落地。
- DeepGEMM 真正接入默认 runtime。
- 更激进的 prefix reuse 和近似压缩。

## 3. 三个 Owner 的职责、大体工作安排和协作方式

### 3.1 拆分原则

本轮拆分不是平均切 feature，而是按“谁最适合动共享边界”来拆：

- Owner A 是 runtime owner，也是共享接口 owner。
- Owner B 和 Owner C 的任务要尽量独立，默认不改大量已有接口。
- 如果 B/C 发现接口不够用，先提出最小需求，再和 Owner A 一起收敛接口改动，不建议各自直接改大框架。

### 3.2 三个 Owner 的职责总表

| Owner | 负责范围 | 默认写入范围 | 不应该主动扩大的范围 |
| --- | --- | --- | --- |
| Owner A | 单请求性能、词表裁剪、运行时设计、共享接口边界、DeepGEMM 评估、KV compression engine side | `src/engine/`、部分 `src/models/`、benchmark/metrics/config、compact vocab runtime | 不写量化 kernel，不写 speculative 算法主体 |
| Owner B | speculative greedy 最小闭环 | `src/engine/experimental/speculative/`，以及 `StandardEngine::generate()` 附近的最小 glue code | 不重构 generate framework，不改 KV 连续设计，不改大范围 engine API |
| Owner C | W4A16/W8A8 量化算子、已量化模型加载兼容、相关 correctness/benchmark | `src/layers/linear*`、`src/operators/*linear*`、quant loading/tests | 不改 scheduler/generate loop，不开发离线量化转换工具 |

### 3.3 共享接口协作规则

下面这些接口或 schema，如果要改，必须先和 Owner A 对齐：

- `StandardEngine::generate()` 的主循环语义。
- `Scheduler::create_context()` 和 request/context 生命周期。
- `Context` 内 tensor naming、tensor slot、生命周期约束。
- `Response` 写入语义和 token 对外语义。
- `KVManager` 的 layout、capacity、连续写指针规则。
- `operator_impl_table` schema。
- `LinearShapeSignature` schema。
- `sampling.max_new_tokens`、`speculative.*`、`kvcache.*` 这类共享 config 语义。

对 B/C 的具体要求：

- Owner B 如果需要改 token finalize、response 写入、target main KV 提交流程，先和 Owner A 对齐。
- Owner C 如果需要改 shared quant contract、shape signature、operator route schema，先和 Owner A 对齐。
- 如果只是局部实现细节，例如 speculative helper 内部逻辑、W4A16 kernel route、局部测试补齐，则应尽量在各自 owner 范围内独立完成。

### 3.4 大体工作安排

建议按 6-8 周做第一轮排期：

| 周期 | Owner A | Owner B | Owner C |
| --- | --- | --- | --- |
| Week 1 | baseline benchmark、metrics、`max_new_tokens` 语义、compact vocab contract、DeepGEMM 评估范围 | speculative config 和状态流转设计 | W4A16 contract、已量化权重命名、INT4 reference |
| Week 2 | stop check 降同步、token finalize 方案 | draft model load、draft KV context | `w4a16_groupwise` impl skeleton |
| Week 3 | 标准路径 MVP、compact vocab remap POC | draft-only K token 生成 | W4A16 correctness |
| Week 4 | greedy 热路径 benchmark、`lm_head_top1`/DeepGEMM 决策、compact vocab runtime 设计收敛 | target verify temporary KV workspace | W4A16 loading compat 和 operator table |
| Week 5 | 单请求标准路径性能收敛 | accept/reject/commit，exact match | W4A16 Qwen2.5 end-to-end |
| Week 6 | prefix reuse 或 compact vocab 第一版落地 | metrics/fallback | W8A8 decode prototype |
| Week 7-8 | 视结果推进 DeepGEMM、连续 INT8 KV engine side；Prefix KV 做验证和文档收口 | CUDA graph verify 评估 | W8A8 correctness/report 或 INT8 KV attention kernel POC |

## 4. 详细任务拆解

### 4.1 Owner A: 单请求性能、词表裁剪和共享接口

#### 4.1.1 角色定位

Owner A 是本轮的 runtime owner 和接口 owner，负责：

- 标准单请求路径性能优化。
- 词表裁剪的整体落地。
- 和其他两条线有关的共享接口边界。
- DeepGEMM 候选路径的评估和是否接入的决策。
- 如果进入 KV cache 压缩阶段，负责 engine/KVManager 一侧的连续 buffer 接入。

#### 4.1.2 核心任务

1. 基线和 benchmark

- 固定 benchmark matrix：模型、prompt len、decode len、CUDA graph on/off、greedy/sampling、FP16/BF16/W4A16/W8A8。
- 统一输出 `prefill_ms`、`decode_ms`、`decode_step_avg_ms`、`tokens/s`。
- 增加关键 NVTX/CUDA event 边界，至少覆盖 lm_head、sampler、stop check、decode graph replay。

2. generate 语义清理

- 明确 `kvcache.requests[].max_tokens` 仍表示 KV 容量。
- 引入或正式启用 `sampling.max_new_tokens` 表示单次生成上限。
- 实际生成上限按 `min(max_new_tokens, slot.max_tokens - prompt_len + 1)` 控制。

3. 单请求 decode 热路径优化

- 把 stop check 从每 token host copy + stream sync 改成 device-side stop flag。
- token finalize kernel 同时写 decode token、response token 和 stop flag。
- greedy 路径下评估轻量 argmax path。
- `lm_head_top1` fast path 已有默认关闭实验入口；是否默认化只看 benchmark 收益和 token alignment。
- 推动 `m=1` decode GEMV 专项优化，但这里的量化 kernel 由 Owner C 配合。

4. 词表裁剪

- 定义 compact vocab artifact contract：
  - `original_vocab_size`
  - `compact_vocab_size`
  - `old_to_new`
  - `new_to_old`
  - TRT-Edge-LLM 风格 `vocab_map.safetensors`，其中 `vocab_map == new_to_old`
  - `special_token_ids`
  - pruned embedding / `lm_head`
  - 更新后的 `config.json`
- 完成 runtime input remap 和 output remap。
- 确保对外 `Response.token_ids()` 仍返回原 tokenizer id。
- 被裁 token 默认报错，不做静默任意映射。
- 第一轮重点是 runtime 和 contract 落地，不要求先做很复杂的工具化。

5. DeepGEMM 候选路径

- 只做 benchmark、shape 筛选、路由策略和默认开关策略。
- 优先评估 prefill bucket、FP8/W8A8 相关 dense linear。
- 第一版不把 DeepGEMM 作为 `m=1` decode 主方案。
- unsupported shape/hardware/build artifact 必须直接 fallback。

6. Prefix KV 和 KV compression engine side

- Prefix KV 已完成连续 KV layout 下的配置式 prefix warmup/reuse：按 `request_id` slot 和 `prefix_token_ids` 精确匹配，不做 paged attention 或非连续 layout。
- Prefix KV 不是语义化/近似 prefix cache，也不负责 INT8 KV scale buffer。
- 如果未来推进连续 INT8 KV，Owner A 负责把连续 int8 K/V buffer 和 scale buffer 接入 `prepare_kvcache_tensors()` 一侧。

#### 4.1.3 需要和其他 Owner 配合的点

- 给 Owner B 提供稳定的 token finalize、response 写入和 generated token 推进边界。
- 如果 `lm_head_top1`、`m=1` GEMV、DeepGEMM 最终涉及 operator 入口，和 Owner C 一起确定最小接口。
- 如果要改 `Response`、`Context`、`KVManager`、`operator_impl_table` schema，Owner A 应该主导改动。

#### 4.1.4 分阶段计划

| 阶段 | 目标 | 交付 |
| --- | --- | --- |
| Phase 0 | 边界确认 | benchmark matrix、metrics 方案、`max_new_tokens` 语义、compact vocab contract、DeepGEMM 范围 |
| Phase 1 | 标准路径 MVP | stop flag、token finalize、标准路径 baseline、compact vocab remap POC |
| Phase 2 | 接入真实路径 | greedy 热路径优化、compact vocab runtime 设计收敛、B 需要的共享接口边界 |
| Phase 3 | 第二功能 | `lm_head_top1`/DeepGEMM 决策、Prefix KV 验证收口、compact vocab 第一版落地、连续 INT8 KV engine side 配合 |

#### 4.1.5 验收标准

- 标准 greedy 输出不变。
- decode step latency 不回退，且关键热点有可解释的 benchmark。
- compact vocab 不改变外部 token id 语义。
- DeepGEMM 如接入，必须可独立开关并自动 fallback。

### 4.2 Owner B: 投机采样

#### 4.2.1 角色定位

Owner B 负责 speculative greedy 的最小闭环。这里的重点是 correctness 和独立性，不是做一个通用 decode 框架。

#### 4.2.2 核心任务

1. 配置和入口

- 保留 `speculative.enabled`。
- 第一版只支持 `algorithm="greedy"`。
- 增加 `num_draft_tokens`。
- `enabled=false` 时标准路径完全不受影响。

2. Draft model 接入

- draft 和 target 必须同 tokenizer。
- draft 和 target 必须同 vocab size。
- draft 独立 `Model`、独立连续 `KVManager`、独立 scheduler/context。

3. Draft-only MVP

- 从 target 首 token 开始，draft 能连续生成 K 个 token。
- draft KV 全部写入自己的连续 KV buffer。
- draft-only 失败时不影响标准路径。

4. Target verify 和 accept/reject/commit

- target verify 使用临时连续 K/V workspace。
- 不直接污染主 target KV。
- greedy 下逐位比较 draft token 和 target token。
- accepted token 写入 response，并把对应 K/V 提交到主 KV 当前写指针。
- rejected 后丢弃临时 workspace，并写入 target token。

5. 第二阶段能力

- acceptance rate、accepted/rejected tokens、draft latency、target verify latency。
- low acceptance fallback。
- 如需要，再评估固定 `num_draft_tokens` 的 CUDA graph verify。

#### 4.2.3 对 Owner B 的约束

- 不重构成通用 decode framework。
- 不支持跨 tokenizer draft model。
- 不改变主 KV 连续 buffer 设计。
- 默认写入范围尽量限制在 `src/engine/experimental/speculative/` 和 `StandardEngine::generate()` 附近的最小 glue code。

#### 4.2.4 必须先和 Owner A 对齐的改动

- token finalize / response 写入语义。
- generated token 计数和 stop check 交互。
- target main KV 提交流程。
- `Context`、`Response`、`KVManager`、`Scheduler` 的公共接口。

Owner B 更适合提出最小接口诉求，由 Owner A 带着一起把共享边界改稳，而不是自己大范围重构 engine。

#### 4.2.5 分阶段计划

| 阶段 | 目标 | 交付 |
| --- | --- | --- |
| Phase 0 | 范围确认 | speculative config、状态流转说明、draft-only 最小测试样例 |
| Phase 1 | 独立 MVP | draft model load、draft KV context、draft-only K token |
| Phase 2 | 真正闭环 | target verify、accept/reject/commit、exact match correctness |
| Phase 3 | 稳定化 | metrics、fallback、CUDA graph verify 评估 |

#### 4.2.6 验收标准

- `speculative.enabled=true` 不再被 facade 直接拒绝。
- greedy speculative 输出和 standard greedy 完全一致。
- acceptance/fallback 指标可见。
- draft/target KV 不互相污染。

### 4.3 Owner C: 量化算子

#### 4.3.1 角色定位

Owner C 负责量化 operator、已量化 checkpoint 加载兼容，以及后续 W8A8 和可选的 INT8 KV attention kernel。Owner C 不负责离线量化转换工具，也不负责 generate loop 主流程。

#### 4.3.2 核心任务

1. W4A16 operator 化

- 新增 `w4a16_groupwise` linear impl。
- 注册到 `LinearOpRegistry`。
- 通过 `operator_impl_table` 路由。
- decode `m=1` 走 GEMV，prefill `m>1` 走 GEMM。

2. 已量化模型加载兼容

- 支持 `<prefix>.qweight`。
- 支持 `<prefix>.scales` 和 `<prefix>.scaling_factors`。
- 可选支持 `<prefix>.qzeros`、`<prefix>.zeros`。
- 读取必要 metadata，如 `bits`、`group_size`、`quant_method`。
- metadata 缺失时按 shape/dtype 推断，推断失败时报错。

3. correctness 和 benchmark

- 增加 Python dequant reference。
- 覆盖 `m=1`、`m=2/4`、`m=64/128/512`。
- 覆盖 `fused_qkv`、`attention_output`、`fused_gate_up`、`mlp_down`，可选 `lm_head`。
- 输出 layer benchmark、误差报告、token alignment 结果。

4. W8A8 decode-first

- 假设 checkpoint 已存在 int8 weight 和 scale metadata。
- runtime 做 activation quant。
- 第一版只支持 `m=1` decode linear。
- 输出回 FP16/BF16 或 FP32。
- `lm_head` 单独评估，不默认打开。

5. 第二阶段协作项

- 为 Owner A 的 `m=1` decode GEMV、`lm_head_top1` 提供量化路径或 operator 入口。
- 如 Owner A 决定接入 DeepGEMM，提供最小 `linear` operator binding 或 build/link 支持。
- 如果进入 KV cache 压缩阶段，提供 INT8 KV attention dequant-on-load kernel。

#### 4.3.3 对 Owner C 的约束

- 不开发离线量化转换工具。
- 不负责 speculative decode loop。
- 不主动改 scheduler、generate loop、KVManager 主结构。
- 默认写入范围尽量限制在 `src/layers/linear*`、`src/operators/*linear*`、模型加载和 tests。

#### 4.3.4 必须先和 Owner A 对齐的改动

- `LinearShapeSignature` schema 变更。
- `operator_impl_table` schema 变更。
- 量化 contract 如果会影响共享 config 或 engine 路由逻辑。
- 任何需要修改 `Context`、`Response`、`KVManager` 的需求。

Owner C 更适合把工作收敛在 layer/operator/model-loading 范围内，把 engine 侧共享接口留给 Owner A 主导。

#### 4.3.5 分阶段计划

| 阶段 | 目标 | 交付 |
| --- | --- | --- |
| Phase 0 | contract 和测试准备 | W4A16 weight contract、已量化命名梳理、INT4 reference test |
| Phase 1 | W4A16 MVP | `w4a16_groupwise` impl、operator table route、loading compat、layer correctness |
| Phase 2 | W4A16 端到端 + W8A8 起步 | Qwen2.5 greedy alignment、latency report、W8A8 decode prototype |
| Phase 3 | 第二功能 | W8A8 correctness/report、DeepGEMM binding 配合、INT8 KV attention kernel 视情况推进 |

#### 4.3.6 验收标准

- W4A16 能通过 `operator_impl_table` 命中并端到端跑通。
- unsupported quant format 报错清晰。
- W8A8 decode-first 至少有 layer-level correctness 和 error report。
- 关闭量化路径后，原有 FP16/BF16 行为不回退。

### 4.4 共享接口修改与并行开发注意事项

这部分是整个三人并行开发时最容易出问题的地方，需要单独提醒：

1. 共享接口修改默认由 Owner A 主导。

- Owner B 和 Owner C 可以提出需求，但不要各自直接改大范围 engine/shared schema。
- 如果确实需要改，先收敛成最小接口改动，再安排 owner 配合落地。

2. PR 边界尽量按 owner 收敛。

- Owner B 的 PR 尽量只改 speculative 和最小 glue code。
- Owner C 的 PR 尽量只改 layer/operator/quant loading/test。
- Owner A 的 PR 负责 shared runtime、benchmark、词表裁剪、公共语义和配置。

3. 不建议并行启动的组合。

- speculative 和 KV cache 压缩不要同时改主 KV 提交流程。
- W8A8 和 `lm_head_top1` 不要在同一个 PR 中改同一段 lm_head 路径。
- DeepGEMM 和 W8A8 不要在同一个 PR 中同时改 `linear` quant dtype、operator route 和 benchmark 口径。
- 词表裁剪和 speculative 不要同时改 response token id 语义。

4. 第一版不要做的事情。

- 不引入 paged attention。
- 不引入新的通用 decode 框架。
- 不引入新的通用 backend framework 只为了 DeepGEMM。
- 不在量化路径里开发离线量化转换工具。
- 不做会改变输出分布的近似 KV 压缩作为第一版主线。
