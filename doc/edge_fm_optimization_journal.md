# EdgeFM 优化日志

最近更新：2026-04-07

这份文档是 EdgeFM 性能优化工作的长期事实来源。
每次开始新一轮优化前，先读这份文档，避免重复走已经验证过且没有收益的分支。

每次有意义的实验结束后，必须补充以下信息：

- 实验目标
- 严格 A/B 条件
- 代码改动
- 实测结果
- 保留还是回退

## 1. 核心目标

- 主目标：持续优化 `EdgeFM(cuda-graph)`，直到打平并尽量超越 `TRT-Edge-LLM`
- 主 benchmark 模型：`Qwen2.5-1.5B BF16, batch=1`
- 主 benchmark 矩阵：
  - `prefill=512, decode=32`
  - `prefill=512, decode=64`
  - `prefill=1024, decode=32`
  - `prefill=1024, decode=64`
  - `prefill=2048, decode=32`
  - `prefill=2048, decode=64`
- 主比较口径：
  - 只重点比较 `EdgeFM(cuda-graph)` 和 `TRT-Edge-LLM`
  - `Transformers` 只作为慢基线保留
  - 不再花时间分析 `EdgeFM(no-graph)`，除非是在排查 graph 正确性

## 2. 可用工具与资源

默认优先使用项目内 skill，不要凭印象拍脑袋试。

- `cuda-skill`
  - 用于 CUDA Runtime / Driver / PTX / best practices 查询
  - 用于 `nsys` / `ncu` / compute-sanitizer 工作流
  - 用于 CUDA Graph、kernel、内存搬运、runtime 设计分析
- `ncu-cuda-profiling`
  - 用于 NCU 采集和指标解释
  - 当前容器里 `ncu` 不可用，但换机器后可继续使用
- `edge-fm-benchmark-report`
  - 用于标准三方 benchmark：`Transformers` vs `EdgeFM(cuda-graph)` vs `TRT-Edge-LLM`
- `edge-fm-add-operator`
  - 新增 `impl_id`、更新 operator registry、更新 `operator_impl_table.json` 时使用
- `cutlass-skill`
  - 生产级 CUDA/CUTLASS kernel 的首选参考
- `triton-skill`
  - 用于快速原型验证和 shape-specific 实验
- `cutile-python-skill`
  - 用于快速 cuTile 原型和 autotune 验证

## 3. 不可违反的原则

- 正确性优先
  - 任何优化都必须同时保证 LLM 和 VLM 正确性
  - 算子级别正确不代表可接受，最终必须过端到端 gate
- 保持代码干净
  - 没有收益的方向必须及时回退
  - 不要把 dead code、临时 `impl_id`、一次性 debug hook 留在 `src/` 里
- 不破坏既有 engine 架构约束
  - 优化必须在现有设计下进行，除非有足够证据证明架构必须改
- 用数据下结论
  - 理论直觉重要，但最终结论必须来自公平 A/B 和 profiling
- 原型和生产实现要区分
  - Triton / cuTile 只负责证明“有没有上限空间”
  - 只有当原型证明收益明确后，才进入 CUTLASS / CUDA 生产实现

## 4. Engine 架构约束

这些约束在当前代码里是真实存在的，优化时不能破坏。

- request 和 slot 的匹配关系是固定的
  - `Scheduler::create_context()` 会按 `request_id` 找到匹配 slot
- slot 的 prefix 匹配是固定的
  - 如果 slot 带 `prefix_token_ids`，请求 token 必须严格匹配该 prefix
- slot 的资源边界是固定的
  - 每个 slot 都有固定 `prefix_size` 和 `max_tokens`
  - 请求必须适配 slot，不能反向篡改 slot 语义
- `Context` 是叠加在 slot 上的每请求运行时状态
  - `Context` 维护 `generated_tokens`、`decode_cache_kv_len`、响应 token 指针、tensor 视图和模型相关状态
  - `Context` 不能侵入或重写 slot 定义本身
- decode 的动态状态优先收敛到固定 device-side buffer
  - 当前已经有稳定的 decode 设备端状态：
    - `TOKEN_IDS`
    - `D_KV_LEN`
    - 可选的 `POSITION_IDS`
  - 这条方向和 CUDA Graph replay 是兼容的，应继续保持

相关代码：

- [scheduler.cpp](/xs-train-nas/zzm/repos/edge-fm-x/src/engine/scheduler.cpp)
- [scheduler.h](/xs-train-nas/zzm/repos/edge-fm-x/src/engine/scheduler.h)
- [stardard_engine.cpp](/xs-train-nas/zzm/repos/edge-fm-x/src/engine/stardard_engine.cpp)

## 5. 实验规则

- 先写假设，再开始改代码
  - 例如：“`2048/64` 的 prefill gap 主要来自 GEMM 选择，而不是 D2D copy”
- A/B 必须公平
  - 同模型
  - 同 token
  - 同 `prefill_len`
  - 同 `decode_len`
  - 同 stop-token 行为
  - 同 dtype
  - 同 device
- 对比对象要对
  - 比 fusion，就要用同一类 kernel 家族下的 fused vs unfused 做比较
  - 不能因为一个很慢的 Triton 原型表现差，就直接得出“生产 fusion 没价值”
- 算子结论和端到端结论要分开
  - microbench 负责决定“值不值得集成”
  - end-to-end benchmark 负责决定“集成后是否真的重要”
- 重写前先 profiling
  - 当前容器内统一用 `nsys`
  - `ncu` 不可用，不作为 blocker
- GPU benchmark / profiling 不要并行跑
  - 之前已经验证过，并行跑容易出无效数据
- 没收益的分支尽快回退
  - 不要让临时代码在树里长期存活

## 6. 当前环境事实

- 目标平台：`A800-SXM4-80GB / sm80 / device=1`
- 当前构建类型：`Release`
  - 已从 `build/CMakeCache.txt` 验证
  - 因此当前性能差距不是 `Debug vs Release` 导致的
- `nsys`：当前容器可用
  - 有时输出 `.qdstrm`
  - 需要用 `QdstrmImporter` 转成 `.nsys-rep` 再做 `nsys stats`
- `ncu`：当前容器不可用
  - 不阻塞当前优化
- Python / pytest 运行时需要：
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`

相关产物：

- benchmark helper：
  - [.codex benchmark script](/xs-train-nas/zzm/repos/edge-fm-x/.codex/skills/edge-fm-benchmark-report/scripts/report_qwen_3way_cuda_graph_vs_trt.py)
- 最近 benchmark 快照：
  - `/tmp/edgefm_bench_512_64_after_fused512.json`
  - `/tmp/edgefm_bench_2048_64_latest.json`
  - `/tmp/edgefm_bench_fresh_512_1024_2048_x64.json`
- 最近 `nsys` 产物：
  - `/tmp/edgefm_profile_2048_64.nsys-rep`
  - `/tmp/edgefm_profile_2048_64.sqlite`
  - `/tmp/edgefm_profile_2048_64_stats_cuda_gpu_kern_sum.csv`
  - `/tmp/edgefm_profile_2048_64_stats_cuda_api_sum.csv`
  - `/tmp/edgefm_profile_current_2048_64.nsys-rep`
  - `/tmp/edgefm_profile_current_2048_64.sqlite`
  - `/tmp/edgefm_profile_current_2048_64_stats_cuda_gpu_kern_sum.csv`
  - `/tmp/edgefm_profile_current_2048_64_stats_cuda_api_sum.csv`

## 7. 当前可信 benchmark 基线

### 7.1 恢复 tuned attention 后的可信矩阵

这是恢复 decode tuned attention 路径后的可信六组数据，用于指导后续优化优先级。

| Case | EdgeFM total | TRT total | Gap | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill=512, decode=32` | `115.259 ms` | `112.003 ms` | `+3.165 ms` / `+2.83%` | `12.169 ms` | `12.282 ms` | `-0.113 ms` | `102.883 ms` | `99.606 ms` | `+3.277 ms` |
| `prefill=1024, decode=32` | `122.980 ms` | `119.182 ms` | `+3.614 ms` / `+3.04%` | `18.733 ms` | `15.699 ms` | `+3.034 ms` | `103.934 ms` | `103.353 ms` | `+0.581 ms` |
| `prefill=2048, decode=32` | `140.943 ms` | `141.125 ms` | `-0.339 ms` / `-0.24%` | `33.629 ms` | `29.457 ms` | `+4.172 ms` | `106.974 ms` | `111.485 ms` | `-4.511 ms` |
| `prefill=512, decode=64` | `221.818 ms` | `213.766 ms` | `+7.932 ms` / `+3.71%` | `12.233 ms` | `11.209 ms` | `+1.024 ms` | `209.333 ms` | `202.425 ms` | `+6.908 ms` |
| `prefill=1024, decode=64` | `230.557 ms` | `226.185 ms` | `+4.232 ms` / `+1.87%` | `18.712 ms` | `15.823 ms` | `+2.889 ms` | `211.541 ms` | `210.199 ms` | `+1.343 ms` |
| `prefill=2048, decode=64` | `263.140 ms` | `264.530 ms` | `-1.525 ms` / `-0.58%` | `37.932 ms` | `37.737 ms` | `+0.195 ms` | `224.899 ms` | `226.619 ms` | `-1.720 ms` |

### 7.2 当前最可信重跑快照

后续又对两组代表性 case 做了重跑确认，结果表明此前出现的“全局大回归”并不稳定复现。

| Case | EdgeFM total | TRT total | Gap | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefill=512, decode=64` | `221.646 ms` | `212.752 ms` | `+8.894 ms` / `+4.18%` | `12.168 ms` | `10.893 ms` | `+1.275 ms` | `209.208 ms` | `201.749 ms` | `+7.459 ms` |
| `prefill=2048, decode=64` | `251.009 ms` | `256.591 ms` | `-5.582 ms` / `-2.18%` | `33.544 ms` | `29.472 ms` | `+4.072 ms` | `217.123 ms` | `226.946 ms` | `-9.823 ms` |

### 7.3 基线解释

- 长 context case 已经基本打平甚至领先 TRT
- 真正剩下的主要问题是短 context decode，尤其是 `512/64`
- 当前没有证据表明存在“整条 runtime 路径已经坏掉”的大回归
- 剩余空间仍然存在，但已经不是那种显而易见的大 easy win

## 8. 已保留的有效优化

- decode attention 的 tuned 路径已经恢复并保留
  - `impl_id`: `flashinfer_attention_decode_sm80_tuned`
  - 代码位置：
    - [attention_op.cu](/xs-train-nas/zzm/repos/edge-fm-x/src/operators/attention_op.cu)
    - [operator_impl_table.json](/xs-train-nas/zzm/repos/edge-fm-x/examples/config/operator_impl_table.json)
- decode linear 的 tuned `cublasLt` 记录已经保留：
  - `fused_qkv`
  - `attention_output`
  - `mlp_down`
  - `fused_gate_up`
  - `lm_head`
- prefill `fused_qkv` 的 tuned `cublasLt` 记录已经保留：
  - `m=512 -> algo_index=0`
  - `m=1024 -> algo_index=4`
  - `m=2048 -> algo_index=3`
- 已确认的 `fused_qkv` 算子级收益：
  - `m=512`: `0.044880 ms -> 0.038720 ms`
  - `m=1024`: `0.062272 ms -> 0.059216 ms`
  - `m=2048`: `0.114112 ms -> 0.084736 ms`
- decode runtime state 已经完成收敛：
  - 稳定 device-side `TOKEN_IDS`
  - 稳定 device-side `D_KV_LEN`
  - 可选稳定 device-side `POSITION_IDS`

## 9. 已回退或已明确否决的方向

- 不再重新开启“新的 speculative decode attention 分支”
  - 原因：
    - 当前 tuned FlashInfer decode path 就是现阶段的生产路径
    - 之前移除它会导致真实端到端回归
    - 当前剩余 gap 更像短 context decode 的固定成本，而不是 attention 大幅落后
  - 只有当新的 `nsys --cuda-graph-trace=node` 证明 attention 重新成为最大热点时，才允许重新打开 attention 主线
- 已回退：prefill `mlp_down m=2048 -> algo_index=0`
  - 原因：operator test 明确更慢
- 已明确否决“activation-only retune 是主要方向”
  - 原因：真实组合路径收益几乎为零
- 已明确否决“naive CUDA gate_up + act 融合可以直接进生产”
  - 原因：数值正确，但比现有生产路径慢约 `8.31x`
- 已明确否决“一个慢的 Triton 原型就能证明 fusion 没价值”
  - 原因：Triton 只能证明原型本身不适合集成，不能否定生产级 fusion 的理论和实测 headroom

## 10. 当前未决问题与风险

- `mlp_down prefill m=1024 -> algo_index=2` 还需要更严格验证
  - 之前严格 BF16 对比出现过轻微 drift
  - 在没有补齐 torch reference + end-to-end correctness 前，不要把它视为最终稳定收益
- benchmark 输出文件可能在 JSON 前面带日志前缀
  - 解析时必须抓取末尾 JSON payload
- `/tmp` 磁盘空间偏紧
  - 做 `nsys` 时要控制输出规模，并及时清理老产物

## 11. 当前下一步优先级

1. 保持当前 tuned decode attention 路径，不再反复重开 attention 分支
2. 用 `nsys --cuda-graph-trace=node` 对 `prefill=512, decode=64` 做更精确的剩余热点归因
3. 在数据支持下继续推进以下分支：
   - 生产级 `fused_gate_up + silu_and_mul` fusion
   - decode `m=1` linear fixed-cost 优化
   - residual / norm / runtime fixed-cost 优化
4. 只有当新的 profiling 再次证明 prefill attention 或其他路径重新上升为主热点时，才切换主线

当前默认优先级：

1. 保住已恢复的 decode tuned attention 路径
2. 对短 context gap 做 node-level profiling
3. 先做短 context decode 固定成本优化
4. 不再机械性地回到 attention 大改

## 12. 当前正确性 gate

每次进入新一轮正式优化前，至少保证这些 gate 是通过的：

- `tests/operators/test_attention_decode.py`
- `tests/engine/test_qwen2_generate.py -k test_generate_token_alignment_cuda_graph`
- `tests/engine/test_qwen2_generate.py -k test_generate_vl_token_alignment_cuda_graph`

最近一次重跑结果：

- 命令：
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 pytest -q tests/engine/test_qwen2_generate.py -k 'test_generate_token_alignment_cuda_graph or test_generate_vl_token_alignment_cuda_graph'`
- 结果：
  - `2 passed, 10 deselected`

## 13. 实验记录

### 2026-04-06：恢复 decode tuned attention 路径

- 现象：
  - 某一轮工作树里出现了明显 decode 回归
- 根因：
  - 工作树里把 decode 专用 tuned attention 路径删掉了
  - `operator_impl_table.json` 中对应的 decode `impl_id` 路由也丢了
- 处理：
  - 恢复 `flashinfer_attention_decode_sm80_tuned`
  - 重新编译 `Release`
  - 重跑 operator correctness / perf、LLM correctness、VLM correctness、graph benchmark
- 恢复后的算子级证据：
  - decode attention median latency：
    - `kv=512`: `0.024032 ms`
    - `kv=1024`: `0.024736 ms`
    - `kv=2048`: `0.028768 ms`
  - 恢复前 `kv=2048` 的当前构建参考值约为：`0.0682 ms`
- 结论：
  - 这个 tuned attention 路径必须保留
  - 除非将来有严格 A/B 证明它长期回归，否则不要再次移除

### 2026-04-06：Triton 验证 `gate_up + silu_and_mul` 的 fusion headroom

- 假设：
  - 短 context decode 剩余 gap 里，`fused_gate_up + silu_and_mul` 可能还有真实 fusion 空间
- A/B 条件：
  - 目标 shape：Qwen2.5-1.5B decode `m=1`, `hidden=1536`, `intermediate=8960`, `bf16`
  - 使用真实 layer-0 权重
  - Triton unfused 路径：
    - 一个自定义 gate/up decode GEMV-style kernel 输出 packed `[2, 8960]`
    - 一个 Triton `silu_mul` kernel 消费 packed 输出
  - Triton fused 路径：
    - 一个 decode GEMV-style kernel，内部直接完成 `silu(gate) * up`
  - 生产参考路径：
    - EdgeFM `FusedGateUpLinearLayer` + `ActivationLayer`
- 原型脚本：
  - [bench_triton_gate_up_decode.py](/xs-train-nas/zzm/repos/edge-fm-x/scripts/bench_triton_gate_up_decode.py)
- 实测结果：
  - 最好 Triton fused：`0.144208 ms`
  - 最好成对 Triton unfused：`0.149024 ms`
  - 同家族 fusion 收益：约 `0.004816 ms` / call，约 `+3.23%`
  - 另一组接近结果：
    - `0.156768 ms -> 0.151536 ms`
  - 当前 EdgeFM 生产路径：
    - `fused_gate_up + silu_and_mul` 组合：`0.054848 ms`
    - `fused_gate_up` 单独：`0.047520 ms`
    - `silu_and_mul` 单独：`0.014592 ms`
    - 同路径下的理论上限：
      - `0.054848 - 0.047520 = 0.007328 ms` / call
- 结论：
  - 这条线不是零价值，存在真实 fusion headroom
  - 但 Triton 原型本身仍比当前生产路径慢约 `2.6x`
  - 因此：
    - 不集成 Triton runtime 路径
    - 如果继续做，必须走生产级 CUDA / CUTLASS 路线
  - 按 `28 * 64 = 1792` 次调用估算：
    - 现实可争取收益约 `~9 ms`
    - 理论上限约 `~13 ms`

### 2026-04-06：activation-only tuned kernel 验证

- 假设：
  - 当前 decode `silu_and_mul` 可能因为 `batch=1` 只起一个 CTA 而留下明显性能
- A/B 条件：
  - 在 `src/operators/activation_op.cu` 中加 decode-only `cuda_silu_and_mul_decode_sm80_tuned`
  - 只对 `qwen2_5 + sm80 + stage=decode + batch=1 + hidden=8960 + bf16` 生效
  - 使用独立子进程 benchmark driver 做基线对比
- 脚本：
  - [bench_activation_ab.py](/xs-train-nas/zzm/repos/edge-fm-x/scripts/bench_activation_ab.py)
- 实测结果：
  - baseline activation：`0.014192 ms`
  - tuned activation：`0.013920 ms`
  - activation-only 收益：`0.000272 ms`
  - baseline 组合路径：`0.052880 ms`
  - tuned 组合路径：`0.052864 ms`
  - 组合收益仅：`0.000016 ms`
- 结论：
  - activation-only 不是当前 decode 的主要瓶颈
  - 这条生产分支已经被证伪
- 处理：
  - 回退 runtime 改动
  - 保留脚本和记录，避免后续重复试同一路径

### 2026-04-06：benchmark 再确认

- 目标：
  - 确认此前看到的 `+30%` 级别大回归是否真实存在
- 条件：
  - 走和 `tests/engine/test_qwen2_generate.py` 一致的 graph-only benchmark 口径
  - 同 token、同 `ignore_stop_tokens=True`
  - 比较 `EdgeFM(cuda-graph)` 与 `TRT-Edge-LLM`
  - 代表 case：
    - `prefill=512, decode=64`
    - `prefill=2048, decode=64`
- 实测结果：
  - `512/64`
    - EdgeFM total：`221.646 ms`
    - TRT total：`212.752 ms`
    - gap：`+8.894 ms` / `+4.18%`
    - EdgeFM prefill/decode：`12.168 / 209.208 ms`
    - TRT prefill/decode：`10.893 / 201.749 ms`
  - `2048/64`
    - EdgeFM total：`251.009 ms`
    - TRT total：`256.591 ms`
    - gap：`-5.582 ms` / `-2.18%`
    - EdgeFM prefill/decode：`33.544 / 217.123 ms`
    - TRT prefill/decode：`29.472 / 226.946 ms`
- 结论：
  - 之前看到的大回归并不稳定复现
  - 当前代码仍然处在“长 context 已打平甚至领先，短 context decode 还有小 gap”的状态

### 2026-04-06：naive CUDA 版 `gate_up + silu_and_mul` 融合尝试

- 假设：
  - 既然 Triton 证明有 fusion headroom，生产级 CUDA 直接融合也许能吃到收益
- A/B 条件：
  - 在 `FusedGateUpLinearLayer` 上加 decode-only CUDA fused path
  - 对比对象为当前生产路径：
    - `FusedGateUpLinearLayer::forward_fp16_bf16(..., Decode)`
    - 加上 `ActivationLayer::forward_silu_and_mul(..., Decode)`
  - 测试 shape：
    - `m=1`, `hidden=1536`, `intermediate=8960`, `bf16`
- 实测结果：
  - 数值：对齐 baseline
  - 性能：
    - baseline 两段式：`0.053472 ms`
    - naive fused CUDA kernel：`0.444336 ms`
    - 约慢 `8.31x`
- 结论：
  - 这个 naive CUDA 融合实现完全不具备生产价值
  - 如果后续重开这条线，必须直接走更强实现：
    - CUTLASS / CUDA tensor-core 级实现
    - 或者先由 profiling 证明还有更急迫的短 context 热点
- 处理：
  - 已完整回退该 runtime 分支
  - 不保留任何临时 hook 或绑定

### 2026-04-07：当前状态收敛

- 目标：
  - 把“当前还剩什么 gap、应该继续做什么、不该再重复什么”落成中文结论
- 额外核验：
  - 已确认 `flashinfer_attention_decode_sm80_tuned` 仍在 active runtime 和 operator table 中
  - 已重跑端到端 LLM/VLM CUDA Graph correctness gate
- 当前结论：
  - 当前没有证据表明存在大的全局回归
  - 长 context 已经达到与 TRT 持平甚至略优
  - 主要剩余问题是短 context decode，特别是 `512/64`
  - 还有优化空间，但已经没有非常明显的大 easy win
- 当前最值得做的三条线：
  - 生产级 `fused_gate_up + silu_and_mul` fusion
  - decode `m=1` linear fixed-cost 优化
  - `512/64` 的 node-level `nsys` 精确归因
- 当前明确不该重复的方向：
  - 不要在没有新证据前再次重开 attention 大改
  - 不要再次尝试 activation-only retune

### 2026-04-07：恢复 decode graph steady-state 的 host fast path

- 假设：
  - 当前 `512/64` 比之前可信基线更差，不是 decode kernel 本身突然退化，而是 decode graph steady-state 仍在重复执行 host 侧 `prepare_decode_tensors()/prepare_kvcache_tensors()`，把短 context case 的固定成本重新拉高了
- A/B 条件：
  - 同一份当前源码重建后做公平对比
  - 同模型、同 token、同 `ignore_stop_tokens=True`
  - 同 benchmark 入口：`tests/engine/test_qwen2_generate.py` 内 helper
  - 重点 case 先看 `prefill=512, decode=64`
  - 然后补跑完整 6 组 `EdgeFM(cuda-graph) vs TRT-Edge-LLM`
- 代码改动：
  - 在 `Model` 上增加 `has_static_decode_runtime_tensors()` 能力位
  - `Qwen2.5` 显式声明 decode graph steady-state 只依赖稳定设备端 buffer：
    - `TOKEN_IDS`
    - `D_KV_LEN`
    - 可选 `POSITION_IDS`
  - `StandardEngine` 在 decode graph 已捕获后，跳过重复的 decode tensor/kvcache host-side 准备
  - `sync_decode_graph()` 改为直接从 `Context` 当前 K/V 写指针计算 graph 动态节点的下一目的地址，而不是依赖每步重建 tensor map
  - 保留 capture 阶段的临时 K/V redirect 逻辑，不改变 graph capture 语义
- 额外核验：
  - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 EDGE_FM_DEVICE_ID=1 pytest -q tests/engine/test_qwen2_generate.py -k 'test_generate_token_alignment_cuda_graph or test_generate_vl_token_alignment_cuda_graph'`
  - 结果：`2 passed, 10 deselected`
- 实测结果：
  - 单 case `512/64`：
    - EdgeFM total：`222.005 ms`
    - TRT total：`213.273 ms`
    - gap：`+8.732 ms` / `+4.09%`
    - EdgeFM prefill/decode：`13.077 / 208.567 ms`
    - TRT prefill/decode：`11.048 / 202.073 ms`
  - 这组结果已经回到此前可信 `512/64` 水平（之前可信快照约 `221.6 ~ 221.8 ms`）
  - 完整六组当前源码矩阵：
    - `512/32`：EdgeFM `115.540 ms`，TRT `113.706 ms`，gap `+1.61%`
    - `512/64`：EdgeFM `222.005 ms`，TRT `213.273 ms`，gap `+4.09%`
    - `1024/32`：EdgeFM `123.162 ms`，TRT `121.053 ms`，gap `+1.74%`
    - `1024/64`：EdgeFM `230.585 ms`，TRT `226.686 ms`，gap `+1.72%`
    - `2048/32`：EdgeFM `140.866 ms`，TRT `141.290 ms`，gap `-0.30%`
    - `2048/64`：EdgeFM `251.391 ms`，TRT `274.494 ms`，gap `-8.42%`
  - 说明：
    - `2048/64` 本轮 TRT timed runs 有明显 outlier（`313.568 ms`），因此长 context 结论应优先结合 stage time 与 median 看，不要只看单次均值
- 结论：
  - 这个 host fast path 是真实有效优化，不应再被当成“没有收益的噪声改动”回退
  - 它已经把当前源码重新拉回到此前可信指标附近，尤其是最关键的 `512/64`
  - 目前剩余 gap 重新收敛为“短 context decode 仍略落后 TRT”，而不是“当前 runtime 存在新的大回归”
- 处理：
  - 保留该 runtime 改动
  - 后续继续围绕短 context decode 固定成本推进，但不要再把重复 decode prepare 路径当成已无关因素
