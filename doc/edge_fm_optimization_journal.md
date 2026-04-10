# EdgeFM 优化 Journal

最近更新：2026-04-10

这份文档是当前有效的优化事实来源。
它不再保留完整历史流水账；历史报告、原始 benchmark、profiling 产物统一留在 `doc/benchmark_reports/` 和仓库内 `.tmp_codex/`。
后续优化、benchmark、profiling、提交、飞书同步，都以这份文档为准。

## 1. 当前目标

- 主目标：
  - 持续优化 `EdgeFM(cuda-graph)`，优先推进 VLM 主线，直到主 VLM benchmark 上与 `TRT-Edge-LLM` 打平并尽量超越
- 主 VLM benchmark：
  - `Qwen2.5-VL-3B-Instruct`
  - `Qwen2.5-VL-7B-Instruct`
  - 公平口径：`prepared multimodal / 不计 ViT`
- LLM 哨兵模型：
  - `Qwen2.5-0.5B-Instruct`
  - `Qwen2.5-1.5B-Instruct`
  - `Qwen2.5-3B-Instruct`
- 主 benchmark 矩阵：
  - `prefill=512, decode=32`
  - `prefill=512, decode=64`
  - `prefill=1024, decode=32`
  - `prefill=1024, decode=64`
  - `prefill=2048, decode=32`
  - `prefill=2048, decode=64`
- 当前阶段的明确重点：
  - LLM：完整 3 模型矩阵上已整体领先 `TRT-Edge-LLM`；当前主 blocker 收缩到 `3B 512/*` short-context decode，其次是 `1.5B 512/*` 的轻微 decode residual；`0.5B` 以回归保护为主
  - VLM：优先把 `VL-3B / VL-7B` 拉回并压到 `<= 5%` gap 区间

## 2. 必须遵守的工作规则

- 正确性优先：
  - 任何保留的优化都必须先过 correctness gate，再谈 benchmark
- Benchmark / profiling 默认使用 `device=0`：
  - 除非 `device=0` 被占用且文档中明确记录原因，否则不要把主结论建立在 `device=1`
  - 测试过程中必须持续监控 GPU，确认没有其他 compute 进程插入
- 不要并行跑 GPU benchmark / profiling：
  - 并行结果无效，必须避免
- 构建统一使用：
  - `CUDA 12.6`
  - `scripts/build_cuda_fast.sh`
- 当前 Python 环境统一使用：
  - `/xs-train-nas/zzm/conda/e2e_zk`
- 临时文件必须落在仓库内：
  - 默认使用 `.tmp_codex/`
  - 不要再把大体积临时文件写到 `/tmp`
- 阶段性进展必须外发：
  - 用 `cc-connect` 同步飞书
- 阶段性稳定结果必须提交：
  - commit
  - push 到远端 `dml-dev`
- 优化方向必须优先使用成熟库：
  - `FlashInfer`
  - `cuBLASLt`
  - `CUTLASS`
  - `TRT-LLM / TRT-Edge-LLM` 风格实现
- 如果要优化 attention：
  - 优先参考并扩展 `FlashInfer` 的实现
  - 不要自己手写裸 CUDA attention kernel 作为主线方案
- 没收益的方向要快速回退：
  - 不保留 dead code
  - 不保留临时 `impl_id`
  - 不保留一次性 debug 分支
- 定时清理过期信息：
  - 及时删除已被新 benchmark / profiling / correctness 结论覆盖或证伪的旧信息
  - 保证这份 journal 持续反映当前有效事实，保持信息有效性和新鲜度

## 3. 当前环境与默认入口

- 平台：
  - `NVIDIA A800-SXM4-80GB / sm80`
- 默认 device：
  - `device=0`
- CUDA：
  - `/usr/local/cuda-12.6`
- Conda：
  - `/xs-train-nas/zzm/conda/e2e_zk`
- 默认构建：
  - `Release`
  - 如果要做 TRT 3-way benchmark，需带上 `BUILD_TRT_EDGELLM_PYBIND=ON`
- 默认构建脚本：
  - `scripts/build_cuda_fast.sh`
- 默认 repo-local temp root：
  - `.tmp_codex/`
- 当前 Feishu 通路：
  - `cc-connect` 已接通，可正常发送阶段性进展

推荐环境变量：

```bash
export TMPDIR=/xs-train-nas/zzm/repos/edge-fm-x/.tmp_codex/tmp
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/xs-train-nas/zzm/conda/e2e_zk/bin:$PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONPATH=/xs-train-nas/zzm/repos/edge-fm-x/build/python:/xs-train-nas/zzm/repos/edge-fm-x/build/install/python${PYTHONPATH:+:$PYTHONPATH}
```

## 4. 当前 tuning 表与分流规则

- LLM 主线 tuning 表：
  - `examples/config/operator_impl_table_llm.json`
- VLM 主线 tuning 表：
  - `examples/config/operator_impl_table_vlm.json`
- 共享表：
  - `examples/config/operator_impl_table.json`
  - 仅保留兼容用途，不再作为主线结果落点

当前 benchmark / tuning 脚本的默认分流规则已经确认正确：

- `Qwen2.5-*` 会自动命中：
  - `examples/config/operator_impl_table_llm.json`
- `Qwen2.5-VL-*` 会自动命中：
  - `examples/config/operator_impl_table_vlm.json`

另外需要明确：

- `0.5B / 1.5B / 3B` 虽然共用同一个 operator `model_name=qwen2_5`
- 但当前算子匹配是按：
  - `op_kind`
  - `layer_role`
  - `stage`
  - `shape_sig`
  - `hw_profile`
  - `model_name`
  综合打分
- 因此不同模型尺寸的 tuned 记录只要 `shape_sig` 不同，就不会互相冲突
- 同一个 `shape_sig` 下，prefill / decode 记录也不会互相覆盖：
  - `stage` 本身就是匹配维度之一
  - 因而 decode 调优不会直接把 prefill 调优表“冲掉”

这条判断已经通过当前表内容和 `scripts/operator_table_utils.py` 的分流逻辑复核过。

## 5. 当前可信 benchmark 基线

这里只保留仍值得引用的最新结果。
更老的阶段性数字不再作为当前决策依据。

### 5.1 当前 LLM 最新完整 device0 基线

来源：

- `.tmp_codex/bench/qwen_llm_3model_fullsuite_20260410_post3bfinal.json`
- `.tmp_codex/validation/llm_3model_alignment_20260410.json`

使用说明：

- 这里所有 gap 定义统一为：
  - `(EdgeFM - TRT) / TRT`
  - 负值表示 `EdgeFM` 比 `TRT-Edge-LLM` 更快
- `2026-04-10` 之后的 tuning / microbench 结论只引用 `b93c548` 之后的数据：
  - `b93c548`
  - `Fix Qwen tuning tensor bridges`
- 这个提交修复了 tuning 脚本里基于 DLPack 的输出观测问题
- 因此更早的局部 tuning 数字不再作为当前主决策依据

当前最值得引用的完整 LLM 结果：

| Model | Source scope | Avg total gap vs TRT | Avg prefill gap | Avg decode gap |
| --- | --- | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct` | `2026-04-10` 三模型完整 6-shape 套件复测 | `-3.45%` | `-27.33%` | `-0.85%` |
| `Qwen2.5-1.5B-Instruct` | `2026-04-10` 三模型完整 6-shape 套件复测 | `-2.87%` | `-12.52%` | `-1.84%` |
| `Qwen2.5-3B-Instruct` | `2026-04-10` 三模型完整 6-shape 套件复测 | `-0.74%` | `-3.67%` | `-0.71%` |

如果只看同一份 `2026-04-10 post3bfinal` 三模型完整套件，那么：

| Scope | Avg total gap vs TRT | Avg prefill gap | Avg decode gap |
| --- | ---: | ---: | ---: |
| `LLM 3-model fullsuite` | `-2.35%` | `-14.51%` | `-1.13%` |

当前准确判断：

- 三个 LLM 模型在完整矩阵均值上都已经不落后于 `TRT-Edge-LLM`
- LLM 当前不是“普遍 prefill 还落后”的状态
- 当前更准确的描述是：
  - `0.5B / 1.5B` 已有较稳定整体领先
  - `3B` 也已经在完整矩阵均值上转成整体领先
  - 但 `3B 512/*` short-context decode 仍有小尾巴
- 后续 LLM 只需要：
  - 守住已经拿到的整体领先
  - 把 `3B 512/*` 的短 context decode residual 视为回归保护项，而不是再开大主线

### 5.2 当前 LLM 仍需盯紧的 targeted case

来源：

- `.tmp_codex/bench/qwen_llm_3model_fullsuite_20260410_post3bfinal.json`
- `.tmp_codex/bench/qwen3b_shortctx_after_runtime_revert_20260410.json`

当前保留为可信事实的 LLM targeted 结论：

- `0.5B`
  - 更晚的完整 6-shape 套件已经更新到：
    - avg total gap `-3.45%`
    - avg prefill gap `-27.33%`
    - avg decode gap `-0.85%`
  - 它已经不是当前主 blocker
  - 后续只需要把这组收益守住，避免 decode 定向调优回退
- `1.5B`
  - 最新完整套件里的 short-context 观测：
    - `512/32`: total `-5.35%`, prefill `-46.68%`, decode `+1.41%`
    - `512/64`: total `+0.29%`, prefill `-22.24%`, decode `+1.53%`
  - 结论：
    - short-context total 已基本到 parity 区间
    - 剩余差距主要是轻微 decode residual
- `3B`
  - `2026-04-10` 已落地：
    - 去掉 `decode fused_gate_up` 的显式 `algo_index=2`，回退到 heuristic baseline
  - 最新完整套件和回归文件共同确认：
    - `512/32`: total `+1.99%`, prefill `-7.65%`, decode `+3.00%`
    - `512/64`: total `+1.94%`, prefill `-11.70%`, decode `+2.67%`
    - `512/32` 当前 `decode_step_avg_ms`：
      - EdgeFM `5.536778`
      - TRT `5.375735`
    - `512/64` 当前 `decode_step_avg_ms`：
      - EdgeFM `5.539040`
      - TRT `5.395036`
  - 结论：
    - `3B` 完整矩阵均值已整体领先 TRT
    - 当前残余差距主要集中在 `512/*` short-context decode
    - `decode fused_gate_up` 这条 tail 已经收掉，但剩余小尾巴还在

另外，这一轮最后补做的两个 decode 候选也已经确认不落地：

- `device-side sampled token append` 的 runtime 路线：
  - 对 `1.5B` 有局部收益
  - 但会让 `3B 512/*` decode 明显回退
  - 因而不保留
- `3B decode lm_head algo_3`：
  - 单算子 microbench 更快
  - 但端到端 `3B 512/*` 的 EdgeFM 自身 stage / decode 没有形成稳定收益
  - 因而也不保留

### 5.3 当前可引用的 VLM 模型级结果

来源：

- `doc/benchmark_reports/qwen_vlm_suite_20260409_device0_postretune.md`

当前保留为可信事实的 VLM 结论：

- `Qwen2.5-VL-3B-Instruct`
  - full avg total gap：`+8.07%`
  - short avg total gap（`prefill<=1024`）：`+10.86%`
  - full avg prefill gap：`+23.72%`
  - full avg decode gap：`+6.16%`
- `Qwen2.5-VL-7B-Instruct`
  - full avg total gap：`+6.87%`
  - short avg total gap（`prefill<=1024`）：`+8.39%`
  - full avg prefill gap：`+16.08%`
  - full avg decode gap：`+5.92%`

当前对 VLM 的准确判断：

- 还没有达到 `<= 5%` 的目标线
- 短 context 仍然明显偏慢
- 与 LLM 不同，VLM 当前是 prefill 和 decode 都还需要继续压

## 6. 计时口径与解释规则

这部分是为了避免再误解 benchmark 数字。

当前 `EdgeFM` 的 `prefill_ms` 不是“纯 kernel 时间”，而是整个 prefill phase 的阶段时间。

具体来说：

- `prefill_ms` 的计时开始于：
  - `prepare_tensors(ModelStage::Prefill)` 之前
- `prefill_ms` 的计时结束于：
  - prefill 首 token sampling 之后

这意味着 `prefill_ms` 会同时包含：

- GPU kernel 时间
- on-stream memcpy
- prefill prepare 阶段引入的 host / runtime 边界成本

另外一个已经实测确认的重要事实：

- CUDA event 计时会把两次 `cudaEventRecord` 之间的 host 空转也算进去
- 已通过一次 `time.sleep(0.2)` 的最小实验复核：
  - 观测值约 `200.38 ms`

因此后续分析时必须区分：

- 纯 GPU kernel 瓶颈
- prefill phase 总体时间

不能再把 `prefill_ms` 直接当作“纯 device compute 时间”来解释。

## 7. 已经验证的有效结论

### 7.1 prefill replay-state 是当前第一条被验证有效的高 ROI runtime 主线

`2026-04-10` 在 `device=0` 上完成了以下变更：

- `StandardEngine` 为 plain Qwen LLM prefill 增加 replay-state
- graph replay 命中时，不再无条件重新走完整 `prepare_tensors(ModelStage::Prefill)`
- 仅刷新：
  - `token_ids` H2D
  - `response_tokens` / `sampler_token_out` bookkeeping
- 同时校验：
  - `(request_id, seq_len)` graph key
  - 当前 `kv_read_ptrs / kv_write_ptrs` 是否与 capture 时一致

这条路径已通过：

- correctness：
  - `tests/operators/test_attention_prefill.py -k correctness`
  - `tests/engine/test_qwen2_generate.py -k 'test_generate_token_alignment or test_generate_token_alignment_cuda_graph'`
- build：
  - `.tmp_codex/logs/build_prefill_replay_20260410.log`
- benchmark：
  - `.tmp_codex/bench/qwen_llm_1p5b_3b_shortctx_prefill_replay_20260410_device0.raw.json`

当前保留的准确结论：

- 这不是噪声，而是稳定收益
- `1.5B / 3B` 的 short-context prefill 都稳定下降约 `6.7 ~ 7.1 ms`
- `1.5B`
  - prefill gap 已从 `+3.0 ~ +3.7 ms / +3.2 ms` 一档，转成 `-7.0 ~ -1.7 / -5.5 / -3.5 ms`
  - stage total 在 4 个 short-context case 上都已经 `<= TRT`
- `3B`
  - prefill gap 已从 `+5.3 ~ +6.0 ms` 压到 `-3.6 ~ -0.6 ms`
  - `1024/32` 与 `1024/64` 的 stage total 已反超 TRT
  - 剩余 stage total 差距主要集中在 `512/*` 的 decode

因此当前准确判断是：

- 上一轮 profiling 的主结论成立：
  - 当前大头不是 attention tile，也不是单条 cuBLASLt algo pin
  - 而是 prefill runtime prepare / graph coverage
- plain LLM 的 runtime 静态化已经成为当前确认有效的主线方向
- 后续若继续做 runtime 优化，应优先沿这条线扩展，而不是退回到零散算子 pin

### 7.2 `1.5B / 3B` short-context gap 不是“再补几条 512 线性 pin”就能解决

以下几类试验已经做过，且没有形成足够收益：

- `1.5B fused_qkv prefill m=512`
  - baseline heuristic 已经等价最优
- `1.5B mlp_down prefill m=512`
  - 只有噪声级收益
- `3B attention_output prefill m=512`
  - baseline heuristic 已经等价最优

结论：

- 当前 `1.5B / 3B` short-context prefill gap
- 不能再简单归因于“缺少几个 512 shape 的 `algo_index` pin”

### 7.3 FlashInfer prefill attention tile 不是当前 LLM prefill gap 的主线突破口

`2026-04-10` 在 `device=0` 上又补做了两轮专门 A/B：

- `1.5B / 3B` 统一强制 `prefill_cta_tile_q=128`
  - 产物：
    - `.tmp_codex/bench/qwen_llm_1p5b_3b_shortctx_attn128_20260410_device0.raw.json`
    - `.tmp_codex/monitor/gpu0_llm_1p5b_3b_shortctx_attn128_20260410.log`
- `3B` 分档：
  - `qo_len<=512 -> 128`
  - `qo_len>512 -> 64`
  - 产物：
    - `.tmp_codex/bench/qwen_llm_3b_shortctx_attn_split_20260410_device0.raw.json`
    - `.tmp_codex/monitor/gpu0_llm_3b_shortctx_attn_split_20260410.log`

当前保留的准确结论：

- `1.5B`
  - 强制 `128` 会明显拖坏 prefill：
    - `512/32` prefill gap 从 `+24.31%` 恶化到 `+68.60%`
    - `1024/32` prefill gap 从 `-8.92%` 恶化到 `+29.45%`
  - 因此：
    - 当前 `1.5B` 不能把 attention prefill tile 从 `64` 直接改成 `128`
- `3B`
  - `512` short-context 上，`128` 或分档有时能带来一点 prefill 改善
  - 但从 `EdgeFM` 自身 stage 均值看，真实 `prefill_ms` 变化大多仍接近噪声级：
    - `512/32`: baseline `22.978 ms`，分档 `22.997 ms`
    - `1024/32`: baseline `35.658 ms`，分档 `35.388 ms`
    - `1024/64`: baseline `35.779 ms`，分档 `38.336 ms`
  - 也就是说：
    - `3B` 的 case-level gap 摆动里仍混有 TRT phase 波动
    - 这条线没有形成稳定、可保留的大收益

因此当前准确判断是：

- `1.5B / 3B` 的剩余 prefill gap
- 不能主要归因于 FlashInfer prefill attention `cta_tile_q` 选错
- 这条线可以保留为次级调参项
- 但不应再作为当前主线继续消耗时间

### 7.4 prefill GPU 时间的主成分是 BF16 GEMM + FlashInfer prefill attention + activation

`nsys` prefill-only 证据：

- `1.5B`
  - `.tmp_codex/nsys/edgefm_1p5b_prefillonly_nograph_20260410_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- `3B`
  - `.tmp_codex/nsys/edgefm_3b_prefillonly_nograph_20260410_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`

当前可保留的解释：

- 主耗时是：
  - BF16 GEMM
  - FlashInfer prefill attention
  - activation
- `cudaMemcpy2DAsync` 主要对应每层 K/V 从 fused QKV 输出拆到 `k_write / v_write`
- 这部分存在，但不是当前 20% 到 30% prefill gap 的主要来源

### 7.5 失败分支已经回退，不再重走

以下方向已经确认不应作为当前主线：

- prefill `fused gate_up + SwiGLU`
  - 当前平台上明显更慢
  - 还伴随数值漂移
  - 已回退
  - 相关稳定提交：
    - `8aa445b`
    - `Rollback failed prefill swiglu fused path`

### 7.6 Qwen host-path 微优化没有带来可保留的 prefill 收益

`2026-04-10` 在 `device=0` 上做过一轮 Qwen runtime host-path 微优化试验：

- 改动点：
  - 缓存 decoder layer 指针
  - 减少 per-layer string 拼接 / map lookup
  - 去掉 decode 热路径里的重复 `dynamic_cast`
  - hoist `q/k/v` 相关常量和部分 tensor 查找
- correctness：
  - `tests/operators/test_fused_gate_up_activation.py`
  - `tests/engine/test_qwen2_generate.py -k test_generate_token_alignment`
  - 均通过
- benchmark 产物：
  - baseline：
    - `.tmp_codex/bench/qwen_llm_1p5b_3b_shortctx_3way_20260410_device0.raw.json`
  - host-path 试验：
    - `.tmp_codex/bench/qwen_llm_1p5b_3b_shortctx_3way_hostpathopt_20260410_device0.raw.json`
  - GPU 监控：
    - `.tmp_codex/monitor/gpu0_llm_hostpath_opt_20260410_monitor.log`

当前保留的结论：

- 这组改动没有带来稳定、可保留的 `EdgeFM` prefill 收益
- 从 `EdgeFM` 自身的 stage 均值看：
  - prefill 平均仅变化约 `+0.044 ms`
  - 不足以解释 `1.5B / 3B` short-context prefill gap
- 本轮观测到的 gap 大幅摆动，主要来自 TRT phase time 的复测波动：
  - `1.5B 1024/32` 中 TRT prefill 明显变快
  - `3B 512/*` 中 TRT prefill 明显变慢
  - EdgeFM 自身 prefill 基本未动
- 因此：
  - 这不是当前高 ROI 的主线
  - 已回退，不再作为主线继续投入

### 7.7 `3B decode fused_gate_up` 显式 pin 已确认应回退到 heuristic

`2026-04-10` 在 `device=0` 上补做了 `3B decode fused_gate_up` 的单算子 retune：

- microbench：
  - `.tmp_codex/bench/qwen3b_decode_fused_gate_up_cublaslt_20260410T1535Z.json`
- short-context A/B：
  - candidate：
    - `.tmp_codex/bench/qwen3b_shortctx_fused_gate_up_candidate_20260410_device0.json`
  - 回归：
    - `.tmp_codex/bench/qwen_llm_1p5b_3b_shortctx_after_3b_gateup_unpin_20260410_device0.json`
- correctness：
  - `.tmp_codex/logs/qwen3b_decode_alignment_after_fused_gate_up_unpin_20260410.log`

当前保留的结论：

- `3B decode fused_gate_up` 当前主表里的显式 `algo_index=2` 不是最优
- 单算子上：
  - heuristic baseline `0.064768 ms`
  - 显式 `algo_2` `0.065824 ms`
  - heuristic 更快约 `1.63%`
- 端到端上，`3B 512/*` 的 EdgeFM 自身 decode 也有同步下降：
  - `512/32`：`decode_step_avg_ms 5.550 -> 5.516`
  - `512/64`：`decode_step_avg_ms 5.548 -> 5.515`
- correctness 已过：
  - `3B` CUDA graph token alignment `20/20`，`mismatch_count=0`

因此这条 decode 表项已经作为稳定优化保留：

- 从 `examples/config/operator_impl_table_llm.json` 移除：
  - `3B decode fused_gate_up`
  - `shape_sig=m=1|input=2|weight=2|output=2|in_features=2048|out_features=22016`
  - `algo_index=2`

### 7.8 `device-side sampled token append` runtime 路线确认不落地

`2026-04-10` 对 decode response append 做过一版 device-side runtime 实验：

- 产物：
  - `.tmp_codex/bench/llm_1p5b_3b_shortctx_current_20260410_runtime_experiment.json`

当前保留的结论：

- 这条路对 `1.5B` 的 short-context total 有正向波动
- 但会让 `3B 512/*` 的 decode / total 明显回退
- 由于它不能同时守住 `1.5B` 和 `3B`，因此不保留
- 这条 runtime 路线已回退到基线实现

### 7.9 `3B decode lm_head algo_3` microbench 更快，但端到端不落地

`2026-04-10` 在最后一轮 `3B` 收尾时，重新扫了一遍 decode linear：

- microbench：
  - `.tmp_codex/bench/qwen3b_decode_lm_head_retune_current_20260410.json`
- end-to-end A/B：
  - `.tmp_codex/bench/qwen3b_shortctx_lmhead_algo3_candidate_20260410.json`

当前保留的结论：

- `3B decode lm_head` 单算子上：
  - `algo_3` 比当前主表 `algo_0` 更快
- 但端到端上：
  - `3B 512/32` 的 EdgeFM 自身 `decode / total_stage` 没有改善
  - `3B 512/64` 的 EdgeFM 自身 `decode / total_stage` 也基本持平
- 因而：
  - `algo_3` 不落地
  - 当前主表继续保持 `algo_0`

## 8. 当前不应再继续重复尝试的方向

除非出现新的 profiling 证据，否则不要优先回到这些分支：

- 继续盲目穷举 `1.5B / 3B` 的少量 `512` 线性 shape pin
- 继续盲目 sweep `1.5B` prefill attention tile
- 继续把 `3B` prefill attention short/long tile split 当作主线
- 重新尝试已经失败的 prefill fused `gate_up + SwiGLU`
- 自己手写 attention CUDA kernel 作为主线优化手段
- 继续在 Qwen host-path 上做 layer pointer / map lookup / `dynamic_cast` 级别微优化

## 9. 当前建议的下一步

### 9.1 LLM

优先级最高：

1. 当前 LLM 主线已经完成整体领先收尾：
   - 后续以回归保护为主
   - 不再继续投入新的 runtime 主线
2. 如果后续还看 LLM，只盯：
   - `3B 512/*` 的 decode 小尾巴
   - 所有 LLM 模型的完整矩阵回归保护
3. LLM 当前不应再把“普遍 prefill gap”当作主问题：
   - prefill 优势已经拿到
   - 现在没有证据支持继续扩大 runtime 主线
4. 任何新的 decode 候选都必须同时过两层 gate：
   - 先过 targeted short-context A/B
   - 再过完整 6-shape 矩阵，避免把 `0.5B` 或 `1.5B` 已拿到的收益拉回去
5. 继续保证 `0.5B / 1.5B / 3B` correctness 和阶段指标不回退
6. `3B` 当前已经落地的 decode 调优不要回退：
   - `decode fused_gate_up` 保持 heuristic baseline
   - 已证伪的 `runtime append` 和 `lm_head algo_3` 不要重新加回去

### 9.2 VLM

在 LLM prefill 归因继续推进的同时，VLM 主线不能中断：

1. 坚持使用独立的 `operator_impl_table_vlm.json`
2. benchmark 口径继续固定为：
   - prepared multimodal
   - 不计 ViT
   - `device=0`
3. 目标保持不变：
   - `VL-3B / VL-7B` total gap 压到 `<= 5%`
4. 优先关注：
   - short-context prefill
   - short-context decode

## 10. 需要持续更新的最小信息集

每次有意义的实验结束后，只需要补这 5 项：

- 实验目标
- A/B 条件
- 代码改动
- 实测结果
- 保留 / 回退结论

如果结果稳定且准备保留，还必须同步两件事：

- 用 `cc-connect` 发飞书
- commit 并 push 到 `dml-dev`

这份文档后续只维护“当前仍然有效的结论”。
详细历史过程统一去看：

- `doc/benchmark_reports/`
- `.tmp_codex/`
