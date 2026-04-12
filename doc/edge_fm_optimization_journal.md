# EdgeFM 优化 Journal

最近更新：2026-04-12

这份文档只保留当前有效事实，不再保留长流水账。
历史 benchmark、profiling、临时实验产物统一留在 `doc/benchmark_reports/` 和仓库内 `.tmp_codex/`。
后续优化、benchmark、profiling、提交、飞书同步，都以这份文档为准。

## 1. 当前目标

- 主目标：
  - 持续优化 `EdgeFM(cuda-graph)`，优先推进 VLM 主线，直到主 VLM benchmark 上与 `TRT-Edge-LLM` 打平并尽量超越
- 主 VLM benchmark：
  - `Qwen2.5-VL-0.5B`
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
- 当前重点：
  - LLM：完整 3 模型矩阵均值已整体领先 `TRT-Edge-LLM`；后续以回归保护为主
  - VLM：当前主工作重心已经收缩到 decode residual，尤其是 `VL-3B / VL-7B`

## 2. 必须遵守的工作规则

- 正确性优先：
  - 任何保留的优化都必须先过 correctness gate，再谈 benchmark
- Benchmark / profiling 默认使用 `device=0`
- 不要并行跑 GPU benchmark / profiling
- 构建统一使用：
  - `CUDA 12.6`
  - `scripts/build_cuda_fast.sh`
- Python 环境统一使用：
  - `/xs-train-nas/zzm/conda/e2e_zk`
- 临时文件统一落在仓库内：
  - 默认使用 `.tmp_codex/`
- 阶段性稳定结果需要：
  - 同步飞书
  - commit
  - push 到远端 `dml-dev`
- 优化方向优先使用成熟实现：
  - `FlashInfer`
  - `cuBLASLt`
  - `CUTLASS`
  - `TRT-LLM / TRT-Edge-LLM`
- 能直接复用或对齐 `TensorRT-Edge-LLM` 的算子、插件、kernel 形态、shape 策略时，优先复用或对齐；只有在不适配时才自己补实现
- 如果要优化 attention：
  - 优先参考并扩展 `FlashInfer`
  - 不要自己手写裸 CUDA attention kernel 作为主线方案
- VLM runtime 改动必须先做 LLM 风险评估：
  - VLM runtime 与 LLM 共享路径时，默认视为高风险改动
  - 任何保留的共享 runtime 改动，都必须先验证不会让 LLM 哨兵模型回退
  - 最低回归检查集至少包含 `Qwen2.5-1.5B-Instruct` 和 `Qwen2.5-3B-Instruct` 的 short-context case
- 没收益的方向要快速回退：
  - 不保留 dead code
  - 不保留临时 `impl_id`
  - 不保留一次性 debug 分支
- 定时清理过期信息：
  - 及时删除已被新 benchmark、profiling、correctness 结论覆盖或证伪的旧信息
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
- prepared-case profiling 约束：
  - `build/python` 必须排在 `build/install/python` 前面，避免误加载旧模块

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

当前分流规则已确认正确：

- `Qwen2.5-*` 默认命中 `examples/config/operator_impl_table_llm.json`
- `Qwen2.5-VL-*` 默认命中 `examples/config/operator_impl_table_vlm.json`

当前算子匹配按以下维度综合打分：

- `op_kind`
- `layer_role`
- `stage`
- `shape_sig`
- `hw_profile`
- `model_name`

因此：

- 不同模型尺寸只要 `shape_sig` 不同，就不会互相冲突
- 同一个 `shape_sig` 下，prefill / decode 记录也不会互相覆盖
- decode 调优不会直接把 prefill 调优表冲掉

## 5. 当前可信 benchmark 基线

### 5.1 LLM 最新完整 retained 基线

来源：

- `.tmp_codex/bench/qwen_llm_3model_fullsuite_20260410_post3bfinal.json`
- `.tmp_codex/validation/llm_3model_alignment_20260410.json`

这里所有 gap 定义统一为：

- `(EdgeFM - TRT) / TRT`
- 负值表示 `EdgeFM` 比 `TRT-Edge-LLM` 更快

当前最值得引用的完整 LLM 结果：

| Model | Avg total gap vs TRT | Avg prefill gap | Avg decode gap |
| --- | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct` | `-3.45%` | `-27.33%` | `-0.85%` |
| `Qwen2.5-1.5B-Instruct` | `-2.87%` | `-12.52%` | `-1.84%` |
| `Qwen2.5-3B-Instruct` | `-0.74%` | `-3.67%` | `-0.71%` |

如果只看三模型完整套件均值：

| Scope | Avg total gap vs TRT | Avg prefill gap | Avg decode gap |
| --- | ---: | ---: | ---: |
| `LLM 3-model fullsuite` | `-2.35%` | `-14.51%` | `-1.13%` |

当前准确判断：

- 三个 LLM 模型在完整矩阵均值上都已经不落后于 `TRT-Edge-LLM`
- LLM 当前不是“普遍 prefill 还落后”的状态
- 当前真正需要盯的是：
  - `3B 512/*` short-context decode residual
  - `1.5B 512/*` 的轻微 decode residual
  - `0.5B` 只做回归保护

### 5.2 LLM 当前回归哨兵

来源：

- `.tmp_codex/bench/qwen1p5b_51232_sentinel_after_vlm_prefill_replay_20260412.json`
- `.tmp_codex/bench/qwen3b_51232_sentinel_after_vlm_prefill_replay_20260412.json`
- `.tmp_codex/bench/qwen_llm_3model_fullsuite_20260410_post3bfinal.json`
- `.tmp_codex/bench/llm_1p5b_3b_51232_after_decode_swiglu_launchcompact_20260412.json`

当前最新哨兵结果：

- `Qwen2.5-1.5B-Instruct 512/32`
  - 当前单次复测：
    - `prefill_ms = 8.6238`
    - `decode_ms = 97.5996`
    - `total_stage_ms = 106.2234`
- 历史 retained 完整套件均值：
    - `prefill_ms = 8.6073`
    - `decode_ms = 100.1371`
    - `total_stage_ms = 108.7811`
- `Qwen2.5-3B-Instruct 512/32`
  - 当前单次复测：
    - `prefill_ms = 16.0588`
    - `decode_ms = 164.2139`
    - `total_stage_ms = 180.2726`
    - `decode_step_avg_ms = 5.2972`
  - 历史 retained 完整套件均值：
    - `prefill_ms = 16.0571`
    - `decode_ms = 171.6401`
    - `total_stage_ms = 187.6973`
    - `decode_step_avg_ms = 5.5368`

当前准确判断：

- 当前没有看到任何 “LLM 被拉下来” 的信号
- 这次共享 decode 路径改动在 `1.5B / 3B 512/32` 哨兵上反而带来了额外收益：
  - `1.5B`：
    - `decode_ms` 相比历史 retained 均值再降 `~2.54 ms`
    - `total_stage_ms` 再降 `~2.56 ms`
  - `3B`：
    - `decode_ms` 相比历史 retained 均值再降 `~7.43 ms`
    - `total_stage_ms` 再降 `~7.42 ms`
- 但完整 3 模型 x 6 shape 全套 benchmark 还没有基于这次最新 VLM replay-state 改动重跑
- 因此当前最准确的表述是：
  - `没有回退，而且 short-context 哨兵更快了`
  - `但完整套件仍待最终回归确认`

### 5.3 VLM 当前可引用结果

来源：

- `.tmp_codex/bench/vlm7b_51232_prepared_prefill_replay_vlm_inputs_20260412.json`
- `.tmp_codex/bench/vlm7b_51232_decode_swiglu_launchcompact_20260412.json`
- `.tmp_codex/bench/vlm3b_shortctx_runtime_candidate_20260411.json`
- `.tmp_codex/bench/vlm3b_51232_decode_swiglu_launchcompact_20260412.json`
- `scripts/profile_vlm_prepared_case.py`
- `.tmp_codex/nsys/vlm7b_51232_edgefm_prepared_20260412_decodefused.nsys-rep`

当前最重要的 key case：

- `VL-7B 512/32 prepared`
  - 当前最新保留结果：
    - `prefill 30.54 ms`
    - `decode 300.67 ms`
    - `total 331.21 ms`
  - TRT 当前参考：
    - `prefill 30.21 ms`
    - `decode 301.58 ms`
    - `total 331.79 ms`
  - 当前 gap：
    - `prefill +1.08%`
    - `decode -0.30%`
    - `total -0.17%`
  - 相比上一版保留结果：
    - `prefill 30.45 -> 30.54 ms`
    - `decode 320.33 -> 300.67 ms`
    - `total 350.78 -> 331.21 ms`

- `VL-3B 512/32 prepared`
  - 当前最新复测结果：
    - `prefill 16.11 ms`
    - `decode 172.68 ms`
    - `total 188.78 ms`
  - 当前参考 TRT：
    - `prefill 17.76 ms`
    - `decode 164.63 ms`
    - `total 182.39 ms`
  - 当前 gap：
    - `prefill -9.27%`
    - `decode +4.88%`
    - `total +3.51%`
  - 相比上一版 retained 结果：
    - `prefill 23.52 -> 16.11 ms`
    - `decode 181.18 -> 172.68 ms`
    - `total 204.70 -> 188.78 ms`

- `VL-0.5B 512/32 prepared`
  - 当前不能再引用旧数字
  - 使用当前默认 candy prepared case 时：
    - base prepared input length = `612`
    - image token span reaches position `598`
    - 因此 `prefill_len=512` 不能安全截断
  - 在 `0.5B` 的 `512/32 prepared` 口径修复前，不再保留任何过期的 `0.5B 512/32` benchmark 结论

当前准确判断：

- `VL-7B 512/32 prepared` 这个最关键的 VLM case 已经基本打平并略微超过当前 TRT 参考
- 当前主 residual 已经不再是 `VL-7B 512/32 prepared`
- `VL-3B 512/32 prepared` 也已经显著收敛，但 decode 仍有 `~4.9%` residual
- `VL-0.5B` 当前的首要问题不是 kernel residual，而是 `512/32 prepared` case 本身与默认 prompt/image 组合不兼容
- 最新代码上的 VLM fullsuite 还没有重跑完，所以这里不再保留旧的模型均值表，避免把过期数字当成当前结论
- `Qwen2.5-VL-0.5B / 3B / 7B` 的最新模型级均值 gap，需要等下一轮 fullsuite 后再回填

## 6. 计时口径与解释规则

当前 `EdgeFM` 的 `prefill_ms` 不是“纯 kernel 时间”，而是整个 prefill phase 的阶段时间。

它会同时包含：

- GPU kernel 时间
- on-stream memcpy
- prefill prepare 阶段引入的 host / runtime 边界成本

已经实测确认：

- CUDA event 会把两次 `cudaEventRecord` 之间的 host 空转也算进去

因此后续分析时必须区分：

- 纯 GPU kernel 瓶颈
- prefill phase 总体时间

不能再把 `prefill_ms` 直接当作“纯 device compute 时间”来解释。

## 7. 当前已验证有效的结论

### 7.1 plain LLM prefill replay-state 是已验证有效的主线

`2026-04-10` 已验证：

- `StandardEngine` 为 plain Qwen LLM prefill 增加 replay-state
- graph replay 命中时，不再重走完整 `prepare_tensors(ModelStage::Prefill)`
- `1.5B / 3B` short-context prefill 都稳定下降约 `6.7 ~ 7.1 ms`

这条结论仍然有效。

### 7.2 prepared VLM prefill replay-state 现在也已经验证有效

`2026-04-12` 当前最新保留变更：

- `prepared VLM request` 也纳入 prefill replay-state
- 仅在以下条件同时满足时命中 replay：
  - `embedding / position_ids` 已经在目标 GPU 上
  - replay 时输入指针与 size 保持稳定
  - `kv_read_ptrs / kv_write_ptrs` 保持稳定

这条路径已通过：

- `test_generate_token_alignment`
- `test_generate_vl_token_alignment`
- `test_generate_vl_token_alignment_cuda_graph`

当前保留收益：

- `VL-7B 512/32 prepared`
  - `prefill_ms: 37.50 -> 30.58`
  - `total_stage_ms: 361.44 -> 354.41`

因此当前准确判断是：

- VLM prepared prefill 之前的大头确实主要来自 runtime / replay coverage
- 不是 prefill kernel 本体缺一个大优化

### 7.3 `request-owned device token_ids` 不是有效方向

这条候选：

- correctness 通过
- 但没有形成可保留的端到端收益

因此已经退出主线，不应继续投入。

### 7.4 decode-only `M-RoPE + KV write` 融合现在也已经验证有效

`2026-04-12` 当前最新保留变更：

- decode `seq_len == 1` 且启用 `mrope` 时：
  - 直接走 fused `M-RoPE + KV write` 路径
  - 避免旧路径里的分离式 `V copy -> apply_mrope(Q/K) -> K copy`

这条路径已通过：

- `test_generate_token_alignment`
- `test_generate_vl_token_alignment`
- `test_generate_vl_token_alignment_cuda_graph`

当前保留收益：

- `VL-7B 512/32 prepared`
  - `decode_ms: 323.83 -> 320.82`
  - `total_stage_ms: 354.41 -> 351.23`

因此当前准确判断是：

- 这次 decode 融合是有效的，应保留
- 但它只收掉了约 `3.18 ms` 总时间，说明当前 decode residual 仍然不只是一条 `mrope` 路径的问题

### 7.5 TRT-Edge-LLM 的 M-RoPE 形态已经确认，但 node-level 对照还需要补齐

本地 TensorRT-Edge-LLM 源码已确认存在专门的 M-RoPE 路径：

- `initializeMRopeCosSin`
- `applyRopeWriteKV`

它的实现形态是：

- 先初始化 `MRope cos/sin cache`
- 再把 `rope + KV write` 融到同一条路径里

当前准确判断：

- 我们已经开始按 TRT 形态收敛 decode 路径
- 但最新版本的 node-level `nsys` 对照还在补采
- 在新的 kernel 对照完成前，不再保留旧的 `mrope 占比 1.6%` 这类已经被新实现覆盖的数字

### 7.6 当前 VLM residual 已经从 `VL-7B` 主 case 收缩到更小范围

现有保留 benchmark 与最新 replay/decode 融合实测共同说明：

- `prefill` residual：
  - 大头之前主要来自 runtime / replay coverage
  - 这一点已经被 `VL-7B 512/32 prepared` 的 `prefill 37.50 -> 30.58 -> 30.41 ms` 直接证明
- `decode` residual：
  - `VL-7B 512/32 prepared` 的关键 gap 已被新的 decode fused-gate-up launch 路径基本收平
  - 当前真正剩余的 compute residual 主要集中在：
    - `VL-3B` decode path
    - 后续 fullsuite 中还未重测的其他 shape
  - 下一轮 `nsys` 的重点不再是 `VL-7B 512/32`，而是新的 `VL-3B` decode residual
- `attention / rope-write / runtime launch` 仍需继续按 node-level 对照判断，但优先级已经低于 `VL-3B` 新 residual 的再定位

### 7.7 `VL-7B` decode attention 参数有一条已验证保留的小收益

`2026-04-12` 当前最新保留变更：

- `examples/config/operator_impl_table_vlm.json`
  - `num_qo_heads=28|num_kv_heads=4|head_dim=128`
  - `flashinfer_attention_decode_sm80_tuned`
  - 从
    - `short_seq_bdz=3`
    - `no_split_kv_threshold=192`
    - `chunk_candidates=[64,128,256,512]`
  - 调整到
    - `short_seq_bdz=4`
    - `no_split_kv_threshold=256`
    - `min_chunk_size=128`
    - `chunk_alignment=128`
    - `chunk_candidates=[128,256,512,1024]`

已验证：

- decode attention microbench（`kv_len=512/544`）有稳定收益
- `VL-7B 512/32 prepared` 同进程 A/B：
  - `avg_ms: 351.52 -> 351.04`
  - `total_stage_ms: 351.10 -> 350.78`
  - `decode_ms: 320.63 -> 320.33`
- correctness 通过：
  - `test_generate_vl_token_alignment`
  - `test_generate_vl_token_alignment_cuda_graph`

当前准确判断：

- 这是一条小收益、低风险、应保留的 VLM-only 调优
- 但它只能继续收掉 `~0.3 ms` 量级，当前主 residual 依然不在 attention 调参本身

### 7.8 decode fused-gate-up launch compaction 现在已经验证有效

`2026-04-12` 当前最新保留变更：

- `src/operators/fused_gate_up_activation_op.*`
  - decode fused SwiGLU 不再在热路径里每步重复查询 occupancy
  - `prepare` 阶段缓存选中 kernel 的 occupancy 与 CTA 数
  - 运行阶段按真实 tile 数压缩 threadblock count，避免 `batch_rows=1` decode 小问题上明显过发空 CTA

这条路径已通过：

- `test_generate_token_alignment`
- `test_generate_vl_token_alignment`
- `test_generate_vl_token_alignment_cuda_graph`

当前保留收益：

- decode fused SwiGLU layer microbench（`VL-7B`, `m=1`, `in=3584`, `out=18944`）：
  - `auto = 0.1662 ms`
  - `default = 0.1947 ms`
  - 当前 `auto` 相比 `default` 快 `~14.6%`
- `VL-7B 512/32 prepared`：
  - `decode_ms: 320.33 -> 300.67`
  - `total_stage_ms: 350.78 -> 331.21`
- `VL-3B 512/32 prepared`：
  - `decode_ms: 181.18 -> 172.68`
  - `total_stage_ms: 204.70 -> 188.78`
- LLM 哨兵：
  - `1.5B 512/32`：
    - `decode_ms: 100.14 -> 97.60`
    - `total_stage_ms: 108.78 -> 106.22`
  - `3B 512/32`：
    - `decode_ms: 171.64 -> 164.21`
    - `total_stage_ms: 187.70 -> 180.27`

因此当前准确判断是：

- 这次 decode fused-gate-up launch 收敛是当前最关键、最有效的一次保留优化
- 它既提升了 VLM，也没有拉坏 LLM 哨兵，反而改善了共享 decode 路径
- 后续 VLM decode profiling 应基于这版结果继续往下做，而不是回到修改前的 residual 判断

## 8. 当前不应继续重复尝试的方向

除非出现新的 profiling 证据，否则不要优先回到这些方向：

- `request-owned device token_ids`
- 共享 `act_and_mul` block-size 微调
- 盲目穷举少量 `512` shape 的线性 pin
- host-path map lookup / `dynamic_cast` 一类微优化
- 自己手写裸 attention CUDA kernel 作为主线
- 未经过 LLM 哨兵验证的共享 runtime 改动

## 9. 当前建议的下一步

### 9.1 LLM

- 当前 LLM 主线已经完成整体领先收尾
- 后续以回归保护为主
- 任何新的共享 runtime 或 VLM 相关改动，都先过：
  - `1.5B 512/32` 哨兵
  - `3B 512/32` 哨兵
- 只有在候选确定保留时，才重跑完整 3 模型 x 6 shape 套件

### 9.2 VLM

后续 VLM 主线优先级：

1. 先基于这版新结果重做 `VL-3B` decode residual 的 `nsys` 定位
2. `VL-7B` 主 key case 已基本收平，后续主要做回归保护与 fullsuite 扩展验证
3. `VL-0.5B` 先修 benchmark contract：
   - 要么给 `512/32 prepared` 换一个合法的短 prompt/image case
   - 要么明确把 `1024/32` 作为首个有效 prepared 对照点
4. 在新的 residual 上继续优先看：
   - `decode qkv / attention_output / mlp_down / lm_head`
   - TRT multimodal 路径里已经存在的 plugin / kernel / shape 策略
5. 如果候选只在 microbench 更快、但端到端没有稳定收益：
   - 不保留

## 10. 需要持续更新的最小信息集

每轮迭代后，至少更新这几项：

- 最新 retained benchmark 数字
- correctness gate 是否通过
- 当前主 blocker 是什么
- 哪些方向已被证伪，不应再重复尝试
- 是否需要清理过期信息，保证文档持续新鲜
