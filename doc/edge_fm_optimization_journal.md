# EdgeFM 优化 Journal

最近更新：2026-04-13

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

来源：

- `.tmp_codex/bench/qwen_6model_fullsuite_20260413_runtime_final.json`
- `.tmp_codex/validation/qwen_correctness_suite_20260413_runtime_final.json`
- `doc/edge_fm_benchmark_tables.md`

这里所有 gap 定义统一为：

- `(EdgeFM - TRT) / TRT`
- 负值表示 `EdgeFM` 比 `TRT-Edge-LLM` 更快

当前 retained 口径：

- 覆盖 `6` 个模型：
  - LLM：`0.5B / 1.5B / 3B`
  - VLM：`0.5B / 3B / 7B`
- fullsuite 共 `34` 个 3-way case：
  - `18` 个 LLM case
  - `16` 个 VLM case
- 当前 final JSON 中，少数 TRT case 出现单次 prefill outlier
  - 例如 `Qwen2.5-0.5B-Instruct 512/32` 的 TRT `prefill_ms` 五次分别为：
    - `187.64 / 6.97 / 7.14 / 115.52 / 7.03`
- 因此当前 retained benchmark 与 `doc/edge_fm_benchmark_tables.md` 统一采用：
  - 当前 fullsuite JSON 内 `5` 次 timed run 的逐阶段 `median`
- 完整逐 shape 数据统一看：
  - `doc/edge_fm_benchmark_tables.md`

### 5.1 correctness gate

- correctness fullsuite 已通过：
  - `.tmp_codex/validation/qwen_correctness_suite_20260413_runtime_final.json`
  - `all_passed = true`
- 覆盖范围：
  - LLM `0.5B / 1.5B / 3B` eager + cuda-graph
  - VLM `0.5B / 3B` `prefill_len=2048` eager + cuda-graph
  - VLM `7B` `prefill_len=2048` 与 `prefill_len=15580` eager + cuda-graph

当前准确判断：

- 当前 `6` 个模型都已经 correctness 对齐
- `VLM 7B` 不再是 correctness blocker
- 后续如果继续推进，主线应回到性能优化与回归保护

### 5.2 LLM 最新完整 retained 基线

| Model | Avg total gap vs TRT | Avg prefill gap | Avg decode gap | Avg decode-step gap |
| --- | ---: | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct` | `-8.42%` | `-26.23%` | `-6.42%` | `-6.42%` |
| `Qwen2.5-1.5B-Instruct` | `-5.40%` | `-7.52%` | `-5.61%` | `-5.61%` |
| `Qwen2.5-3B-Instruct` | `-4.26%` | `-2.85%` | `-4.77%` | `-4.77%` |
| `LLM 3-model fullsuite` | `-6.03%` | `-12.20%` | `-5.60%` | `-5.60%` |

当前准确判断：

- 三个 LLM 模型在 fullsuite retained median 口径下都已经领先 `TRT-Edge-LLM`
- LLM 当前不是 correctness 风险源，也不是主要性能瓶颈源
- 后续对 LLM 只需要做回归保护，不需要再专门开一条大优化支线

### 5.3 VLM 最新完整 retained 基线

| Model | Avg total gap vs TRT | Avg prefill gap | Avg decode gap | Avg decode-step gap |
| --- | ---: | ---: | ---: | ---: |
| `Qwen2.5-VL-0.5B` | `-12.82%` | `-10.73%` | `-13.03%` | `-13.03%` |
| `Qwen2.5-VL-3B-Instruct` | `-0.00%` | `-3.64%` | `+0.21%` | `+0.21%` |
| `Qwen2.5-VL-7B-Instruct` | `-1.15%` | `+2.01%` | `-1.69%` | `-1.69%` |
| `VLM 3-model fullsuite` | `-3.63%` | `-3.29%` | `-3.81%` | `-3.81%` |

当前准确判断：

- `VL-0.5B` 当前全部合法 shape 都已经整体领先 `TRT-Edge-LLM`
- `VL-3B` 在 fullsuite retained median 口径下总体已经打平，剩余 residual 主要集中在 `512/*` 与 `1024/*` short-context decode
- `VL-7B` 当前只是在 prefill 上略慢，但从 `1024/*` 开始 decode 已经稳定领先 TRT
- 当前 VLM 主线已经从“先修 correctness 和 benchmark contract”切换成“只做有把握的 decode residual 优化”

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

### 7.2.1 `BF16 linear -> CUBLAS_COMPUTE_32F_FAST_16BF` 不是有效保留方向

`2026-04-13` 已验证：

- 在 `LinearLayer` 的 BF16 路径上，把 `cublasLtMatmulDescCreate` 的 compute type 从
  - `CUBLAS_COMPUTE_32F`
  - 切到 `CUBLAS_COMPUTE_32F_FAST_16BF`
- 并补齐 descriptor cache 的 dtype 重建条件

`VL-3B 512/32 prepared` 复测：

- baseline retained：
  - `prefill_ms ~= 16.07`
  - `decode_ms ~= 172.79`
  - `total_stage_ms ~= 188.86`
- candidate：
  - `prefill_ms ~= 16.06`
  - `decode_ms ~= 172.81`
  - `total_stage_ms ~= 188.87`

当前准确判断：

- 这条方向没有形成端到端收益
- 已经回退，不保留到主线
- 后续不再继续把时间投入到单纯更换 BF16 compute type 这一条线上

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

### 7.9 `2026-04-13` 两条新候选都已证伪并回退

- `cuBLASLt row-major descriptor` 候选：
  - 目标：
    - 通过改写 Lt descriptor 表达，避免 decode 小 shape 线性层落到 GEMV-like tactic
  - 结果：
    - `fused_qkv (2048 -> 2560)` 在候选路径下直接触发 `cuBLASLt status 15`
    - `attention_output / mlp_down / lm_head` 三个 decode shape 虽然能跑，但 `heuristic_candidate_count = 0`
    - `attention_output` layer microbench 也没有形成收益：
      - baseline：`~0.01763 ms`
      - candidate：`~0.01792 ms`
  - 结论：
    - 这条路径不能稳定运行
    - 而且即便能跑的 shape 也没有形成更好的 tactic 选择
    - 已回退，不进入主线

- `decode mrope vec8` 候选：
  - 目标：
    - 参考 `TRT-Edge-LLM applyRopeWriteKV` 的向量化形态，缩小 `decode_mrope_apply_q_write_kv` 与 TRT 的差距
  - correctness：
    - `test_generate_vl_token_alignment`
    - `test_generate_vl_token_alignment_cuda_graph`
    - 均通过
  - 端到端结果：
    - `VL-3B 512/32 prepared`
    - `decode_ms: 172.63 -> 172.60 ms`
    - `total_stage_ms: 188.72 -> 188.71 ms`
  - 结论：
    - 收益远低于噪声带
    - 不保留，已回退

- `explicit cublas` decode linear 候选：
  - 目标：
    - 对 `VL-3B` 的 `m=1` decode 线性层直接 A/B `cublasLt` 与 `cublas`
    - 判断是否能绕开当前 `gemvx / gemv2T` 风格 residual
  - 微基准结果：
    - `attention_output`：`0.01850 -> 0.01888 ms`，`-2.1%`
    - `mlp_down`：`0.05645 -> 0.06102 ms`，`-8.1%`
    - `fused_gate_up`：`0.06581 -> 0.06669 ms`，`-1.3%`
    - `lm_head`：`0.36744 -> 0.36882 ms`，`-0.37%`
    - `fused_qkv`：由于带 `q/k/v bias`，当前 no-bias `cublas` 实现根本不支持
  - 结论：
    - 这条路径不能覆盖 `fused_qkv` 主路径
    - 其余关键 decode 线性层也都不如现有 `cublasLt`
    - 不保留，已回退

### 7.10 `2026-04-13` VLM operator_impl_table 命中 bug 已修复并验证

- 根因：
  - `src/operators/operator_impl_table.cpp` 与 `src/engine/engine.cpp` 的 identifier 归一化规则不一致
  - 对 `qwen2_5_vl` / `cuda_sm80` 这类带下划线的 VLM 标识，query 侧和 JSON table 侧会被归一化成不同字符串
  - 结果是：
    - VLM JSON tuning records 可能直接 miss
    - 退回到 builtin/default `cublasLt` fallback
- 代码修复：
  - 统一保留下划线
  - 同时兼容 `qwen25vl` 等无下划线别名
- 命中验证：
  - 新增测试：
    - `tests/operators/test_decode_linear.py::test_vlm_decode_linear_uses_shape_tuned_record`
  - 当前已通过，且 `selected_impl_params.algo_index` 能正确读到 VLM tuned record
- 当前量化结论：
  - `VL-3B decode linear` layer microbench，相比 stripped fallback：
    - `fused_qkv`：`+26.1%`
    - `attention_output`：`+5.2%`
    - `mlp_down`：`+14.6%`
    - `lm_head`：`+0.38%`
  - `VL-3B 512/32 prepared` 的 `EdgeFM-only` A/B：
    - 保留 tuned decode linear records：
      - `prefill 16.08 ms`
      - `decode 172.77 ms`
      - `total 188.85 ms`
    - 删除这些 tuned records：
      - `prefill 16.07 ms`
      - `decode 184.47 ms`
      - `total 200.54 ms`
    - 当前记录相对 stripped fallback：
      - `decode -11.70 ms`
      - `total -11.69 ms`
      - `decode_step_avg -6.34%`
- 当前准确判断：
  - 不需要再怀疑 `operator_impl_table_vlm.json` 的 decode linear tuned records 是否命中
  - 它们现在已经命中，而且是当前 `VL-3B` 基线不可缺少的一部分
  - 在此基础上，`VL-3B` 相比 TRT 仍有 `~7.8 ms decode residual`，说明剩余问题已经不是“表没命中”这一层

### 7.10.1 `2026-04-13` 当前 VLM-3B decode linear algo table 基本已经收敛

在 `operator_impl_table_vlm.json` 已正确命中的前提下，重新对 `VL-3B decode` 热点 shape 做了 `cublasLt algo_index` 复扫：

- `fused_qkv (2048 -> 2560)`：
  - 当前表项 `algo_index=3`
  - 仍然是最优，`~0.01757 ms`
- `mlp_down (11008 -> 2048)`：
  - 当前表项 `algo_index=1`
  - 仍然是最优，`~0.04704 ms`
- `attention_output (2048 -> 2048)`：
  - 当前表项 `algo_index=1`
  - baseline auto `~0.01734 ms`
  - `algo_index=1` `~0.01744 ms`
  - 差异只有 `~0.00010 ms`，端到端意义很弱
- `lm_head (2048 -> 151936)`：
  - 当前表项 `algo_index=3`
  - 最优候选是 `algo_index=2`，`~0.36346 ms`
  - 当前 `algo_index=3` `~0.36480 ms`
  - 差异只有 `~0.00134 ms`

当前准确判断：

- `fused_qkv` 和 `mlp_down` 当前 decode table 已经处在正确最优点
- `attention_output` / `lm_head` 还存在极小的微基准差异，但量级只在 `0.0001 ~ 0.0013 ms`
- 就 `VL-3B 512/32 prepared` 这个残余规模看，这两项即便调整，也不足以解释或消除剩余 decode gap

### 7.10.2 `2026-04-13` decode fused SwiGLU occupancy cap 不是有效新方向

为了确认当前 `decode fused-gate-up launch compaction` 是否仍然过保守，额外验证了 `decode fused SwiGLU` 的 occupancy cap：

- `cap=2`：`~0.05942 ms`
- `cap=3`：`~0.06037 ms`
- `cap=4`：`~0.05946 ms`

当前准确判断：

- 当前默认 `cap=2` 仍然最好
- 把 occupancy cap 再抬高没有形成额外收益
- 这条探针已经回退，不保留到主线

### 7.10.3 `2026-04-13` `fp16 kvcache` 不是一个可直接通过 benchmark 配置切换的公平候选

为了验证 `VL-3B` residual 是否有一部分来自 `bf16 KV cache / attention dtype` 形态，额外做了一个最小化实验：

- 保持 `Qwen2.5-VL-3B` 其余路径不变
- 只把 engine config 里的 `kvcache.dtype` 从 `bf16` 改成 `fp16`
- 继续跑 `VL-3B 512/32 prepared`

结果：

- 当前 `bf16 kvcache` 复测：
  - `prefill_ms ~= 16.10`
  - `decode_ms ~= 172.82`
  - `total_stage_ms ~= 188.93`
- `fp16 kvcache` 直接失败：
  - `attention` 入口报错
  - `k tensor dtype mismatch. Expected 2, got 1`

当前准确判断：

- 这说明 `fp16 kvcache` 不是当前 benchmark 脚本口径不公平导致的“假问题”
- 也不是一个可以靠改配置就立即收收益的 quick win
- 如果后续要继续推进这条线，必须补齐真正的 mixed-dtype 支持：
  - prefill / decode 的 KV write cast
  - attention 层的 `q_dtype / kv_dtype / o_dtype` 分离
  - 对应实现是否继续走 `FlashInfer`，还是直接对齐 `TRT-Edge-LLM` decode attention 形态
- 在这些支持补齐前，不应再重复做“只改 `kvcache.dtype=fp16`”的 benchmark

### 7.10.4 `2026-04-13` decode fused `M-RoPE + KV write` 的 `inv_freq` 预计算没有形成端到端收益

为了确认当前 fused decode `M-RoPE + KV write` 内核里每线程的 `powf` 是否还是明显浪费，额外验证了一个候选：

- 在 `Qwen2_5` 初始化阶段预计算 `inv_freq`
- decode fused kernel 改为直接读取预计算表，不再逐线程计算 `powf`

`VL-3B 512/32 prepared` 结果：

- baseline retained：
  - `prefill_ms ~= 16.10`
  - `decode_ms ~= 172.82`
  - `total_stage_ms ~= 188.93`
- candidate：
  - `prefill_ms ~= 16.06 ~ 16.19`
  - `decode_ms ~= 172.88 ~ 172.92`
  - `total_stage_ms ~= 188.91 ~ 189.06`

当前准确判断：

- 这条候选没有形成稳定正收益
- 从端到端口径看，效果落在噪声带内，甚至略差
- 相关性能改动已经回退，不保留到主线
- 但顺手保留了一个清理项：
  - `Qwen2_5` 现在会正确释放 `mrope_section_cumsum_gpu_`，避免该模型路径的显存泄漏

### 7.10.5 `2026-04-13` mixed-dtype `bf16 Q / fp16 KV / bf16 O` 是一条真正的结构性方向，但这轮没有保留

这轮最后一次 serious pass 里，确实开始沿着 `q_dtype / kv_dtype / o_dtype` 分离去做一版 mixed-dtype 候选，目标是：

- 保留 `VL-3B / VL-7B` 当前 tuned decode attention
- 同时尝试把 KV cache 压到 `fp16`
- 看能否进一步缩小 short-context decode residual

但在真正拿到端到端 benchmark 之前，这条候选已经暴露出明显的工程复杂度信号：

- `attention_op.cu` 的模板实例数量显著膨胀
- 单次 `scripts/build_cuda_fast.sh` 在 `attention_op.cu` 上：
  - `cicc` 编译阶段持续跑了数分钟
  - `ptxas` 也持续跑了数分钟
  - 整个 attention 相关编译尾巴已经明显超出当前可接受范围
- 在还没有证明端到端稳定收益之前，这样的 shared runtime / attention 复杂度不值得直接保留

当前准确判断：

- mixed-dtype KV cache 仍然是一个可能继续追的结构性方向
- 但它不再属于“快速补一个 runtime/path 小修就能收掉尾巴”的范畴
- 如果后续继续做，必须满足两个前提：
  - 只保留最小必要的 dtype 组合，避免模板组合继续爆炸
  - 在首个可编译版本上立刻证明 `VL-3B 512/*` short-context decode 有明确端到端收益
- 在这两个前提没满足前，这条线不进入 retained baseline
- 因此本轮最终对外口径仍以 `.tmp_codex/bench/vlm_suite_20260413_all_shapes.json` 里的 fullsuite 基线为准

### 7.10.6 `2026-04-13` `VL-0.5B` decode table retune 是有效保留方向

为了把 `Qwen2.5-VL-0.5B` 从“prefill 基本打平、decode 还落后”推进到真正反超 TRT，这轮直接针对 `0.5B` 的 decode 路径补了一次单独 retune。

先确认的结构性问题：

- `examples/config/operator_impl_table_vlm.json` 之前没有 `0.5B` 对应的 `hidden_size=896` decode 线性记录
- 也没有 `num_qo_heads=14|num_kv_heads=2|head_dim=64` 的 decode attention tuned record
- 因此 `0.5B` decode 之前大概率一直在吃 generic fallback / 默认 heuristic

为此保留了两类修复：

- `scripts/tune_qwen_cublaslt.py`
  - 补齐 `llava/text_config` 兼容，允许 `0.5B` 正常读到 `hidden_size / intermediate_size / vocab_size`
- `operator_impl_table_vlm.json`
  - 新增 `0.5B` decode attention tuned record：
    - `num_qo_heads=14|num_kv_heads=2|head_dim=64`
  - 新增 `0.5B` decode linear tuned records：
    - `fused_qkv 896 -> 1152 : algo_index=5`
    - `attention_output 896 -> 896 : algo_index=5`
    - `lm_head 896 -> 151936 : algo_index=3`

这轮明确不保留的 `0.5B` decode linear 记录：

- `mlp_down 4864 -> 896`
  - baseline 最优
- `fused_gate_up 896 -> 9728`
  - baseline 最优

微基准结果：

- decode attention
  - baseline avg median `~0.02770 ms`
  - best tuned avg median `~0.02645 ms`
- `fused_qkv`
  - baseline `~0.01831 ms`
  - best `algo_5` `~0.01702 ms`
- `attention_output`
  - baseline `~0.01709 ms`
  - best `algo_5` `~0.01654 ms`
- `lm_head`
  - baseline `~0.17510 ms`
  - best `algo_3` `~0.16746 ms`

端到端结果：

- `1024/32`
  - 旧 retained：
    - `prefill_ms ~= 6.28`
    - `decode_ms ~= 66.67`
    - `total_stage_ms ~= 72.95`
  - 新 retained：
    - `prefill_ms ~= 6.59`
    - `decode_ms ~= 52.15`
    - `total_stage_ms ~= 58.75`
  - 对 TRT：
    - `decode gap ~= -3.40 ms`
    - `total gap ~= -6.66 ms`
- `2048/32`
  - 旧 retained：
    - `prefill_ms ~= 12.32`
    - `decode_ms ~= 63.47`
    - `total_stage_ms ~= 75.79`
  - 新 retained：
    - `prefill_ms ~= 12.32`
    - `decode_ms ~= 50.86`
    - `total_stage_ms ~= 63.18`
  - 对 TRT：
    - `decode gap ~= -10.09 ms`
    - `total gap ~= -11.42 ms`

当前准确判断：

- 这轮 `0.5B` decode table retune 是明确有效并且应该保留的方向
- 收益主头来自：
  - `lm_head`
  - 小 hidden decode linear tactic 选择
  - `0.5B` 专属 decode attention 参数
- 这轮变更只落在：
  - `VLM operator table`
  - `0.5B` 专属 tuning script 兼容
- 不涉及共享 runtime 行为，因此不构成新的 LLM 回退风险

### 7.10.7 `2026-04-13` VLM 一致性根因已修复，6 模型 correctness fullsuite 已全绿

这轮最终保留的 correctness 修复有四类：

- `src/models/model.*` + `src/models/qwen2_5/*` + `src/engine/stardard_engine.cpp`
  - 将 `mrope_last_pos` 的 fallback 下沉到 `Model::derive_mrope_last_pos(...)`
  - 默认行为仍是 3D `position_ids` 的全局 max 复制到三个维度
  - `Qwen2_5` 仅对真正 `Qwen2.5-VL` 且 `hidden_size >= 3584` 的 `7B` 口径做 `+1` 修正
  - 这样模型相关行为保留在 model 侧，而不是继续堆到通用 runtime 路径里

- `src/layers/attention.*` + `src/utils/device/decode_runtime_kernels.cu` + `src/models/qwen2_5/qwen2_5.*`
  - 明确 `M-RoPE` 的 section 语义是对单个 half-dim 分段，不是 `section * 2`
  - `section_hi = section_lo`
  - prefill / decode 两条路径统一一致

- `src/edge-fm.cpp` + `src/python/pybind_debug.cpp`
  - VLM text-tower 权重过滤补上顶层 `lm_head.*`
  - 避免某些 checkpoint 静默回落到 `embed_tokens.weight` 并产生错误 logits

- `src/operators/linear_impl.cu`
  - 保留 cuBLASLt bias epilogue 外挂 bias 的 workaround
  - 但严格收窄到 `qwen2_5_vl + fused_qkv`
  - 避免把 `LLM 0.5B` 与 `VLM 0.5B` 一起拉坏

同时已经清理掉本轮不应保留的内容：

- `tests/engine/test_qwen2_generate.py` 不再携带临时 `mrope_last_pos` workaround
- `.tmp_codex/` 下的验证 / 定位脚本只保留为实验产物，不进入提交

验证结果：

- correctness：
  - `.tmp_codex/validation/qwen_correctness_suite_20260413_runtime_final.json`
  - `all_passed = true`
- benchmark：
  - `.tmp_codex/bench/qwen_6model_fullsuite_20260413_runtime_final.json`
  - `18` 个 LLM 3-way case
  - `16` 个 VLM 3-way case

当前准确结论：

- 一致性问题已经从“`VLM 7B` 不对齐”收敛并修复到“`6` 个模型全部对齐”
- 后续如果继续推进，只需要做性能优化和回归保护，不再需要围绕 correctness 根因做大范围试探

## 8. 当前不应继续重复尝试的方向

除非出现新的 profiling 证据，否则不要优先回到这些方向：

- `request-owned device token_ids`
- 共享 `act_and_mul` block-size 微调
- 盲目穷举少量 `512` shape 的线性 pin
- host-path map lookup / `dynamic_cast` 一类微优化
- 自己手写裸 attention CUDA kernel 作为主线
- 未经过 LLM 哨兵验证的共享 runtime 改动
- `cuBLASLt row-major descriptor` decode 线性候选
- no-bias `explicit cublas` decode linear 候选
- 只改善单个小 kernel、但端到端增益落在噪声带内的 `mrope` 向量化尝试
- 只改 `kvcache.dtype=fp16`、但没有补齐 mixed-dtype attention / KV write 支持的伪候选

## 9. 当前建议的下一步

### 9.1 共享 gate

- 任何新的共享 runtime / kernel / operator 改动，都先过：
  - `.tmp_codex/validation/qwen_correctness_suite_20260413_runtime_final.json` 同口径的 correctness fullsuite
- 只有 correctness 通过且端到端收益稳定的候选，才进入 retained baseline
- 如果原始 timed run 出现明显单次 outlier：
  - retained benchmark 继续使用当前 fullsuite JSON 的逐阶段 `median`

### 9.2 LLM

- 当前 LLM 主线已经完成整体领先收尾
- 后续以回归保护为主
- 共享改动优先盯：
  - `1.5B 512/32`
  - `3B 512/32`
  - `0.5B 2048/64`
- 如果没有新的回退信号，不再单独开 LLM 优化支线

### 9.3 VLM

1. 如果继续优化，优先盯 `VL-3B` 的 `512/*` 与 `1024/*` short-context decode residual。
2. `VL-7B` 当前 correctness 已完成收口，主要做回归保护；性能上只剩 prefill 小幅慢、short decode 近乎持平。
3. `VL-0.5B` 当前合法 shape 已整体领先 TRT，不再需要先修 benchmark contract 或 TRT 适配。
4. 不再优先回到已证伪方向：
   - `row-major Lt`
   - `explicit cublas`
   - 单独 `mrope` micro-opt
   - 只改 `fp16 kvcache`

## 10. 需要持续更新的最小信息集

每轮迭代后，至少更新这几项：

- 最新 retained benchmark 数字
- correctness gate 是否通过
- 当前主 blocker 是什么
- 哪些方向已被证伪，不应再重复尝试
- 是否需要清理过期信息，保证文档持续新鲜
