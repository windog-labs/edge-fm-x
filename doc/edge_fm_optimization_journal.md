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

### 5.1 LLM 最新完整 retained 基线

来源：

- `.tmp_codex/bench/qwen_llm_3model_fullsuite_20260410_post3bfinal.json`
- `.tmp_codex/validation/llm_3model_alignment_20260410.json`

这里所有 gap 定义统一为：

- `(EdgeFM - TRT) / TRT`
- 负值表示 `EdgeFM` 比 `TRT-Edge-LLM` 更快

#### 5.1.1 LLM 模型级均值

| Model | Avg total gap vs TRT | Avg prefill gap | Avg decode gap |
| --- | ---: | ---: | ---: |
| `Qwen2.5-0.5B-Instruct` | `-3.45%` | `-27.33%` | `-0.85%` |
| `Qwen2.5-1.5B-Instruct` | `-2.87%` | `-12.52%` | `-1.84%` |
| `Qwen2.5-3B-Instruct` | `-0.74%` | `-3.67%` | `-0.71%` |

如果只看三模型完整套件均值：

| Scope | Avg total gap vs TRT | Avg prefill gap | Avg decode gap |
| --- | ---: | ---: | ---: |
| `LLM 3-model fullsuite` | `-2.35%` | `-14.51%` | `-1.13%` |

#### 5.1.2 `Qwen2.5-0.5B-Instruct` retained all-shape

| Shape | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap | Decode-step gap | EdgeFM total | TRT total | Total gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512/32` | `3.671 ms` | `8.061 ms` | `-4.390 ms` | `56.547 ms` | `51.452 ms` | `+5.095 ms` | `+0.164 ms` | `60.217 ms` | `59.513 ms` | `+0.705 ms` |
| `512/64` | `3.509 ms` | `7.038 ms` | `-3.529 ms` | `111.479 ms` | `104.882 ms` | `+6.597 ms` | `+0.105 ms` | `114.988 ms` | `111.920 ms` | `+3.068 ms` |
| `1024/32` | `6.374 ms` | `8.501 ms` | `-2.127 ms` | `54.924 ms` | `54.720 ms` | `+0.204 ms` | `+0.007 ms` | `61.298 ms` | `63.221 ms` | `-1.923 ms` |
| `1024/64` | `6.088 ms` | `9.281 ms` | `-3.193 ms` | `109.560 ms` | `111.206 ms` | `-1.645 ms` | `-0.026 ms` | `115.648 ms` | `120.487 ms` | `-4.839 ms` |
| `2048/32` | `11.886 ms` | `11.820 ms` | `+0.066 ms` | `54.134 ms` | `60.262 ms` | `-6.128 ms` | `-0.198 ms` | `66.020 ms` | `72.082 ms` | `-6.062 ms` |
| `2048/64` | `11.866 ms` | `11.930 ms` | `-0.064 ms` | `110.004 ms` | `122.215 ms` | `-12.211 ms` | `-0.194 ms` | `121.870 ms` | `134.144 ms` | `-12.274 ms` |

#### 5.1.3 `Qwen2.5-1.5B-Instruct` retained all-shape

| Shape | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap | Decode-step gap | EdgeFM total | TRT total | Total gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512/32` | `8.607 ms` | `16.142 ms` | `-7.534 ms` | `100.137 ms` | `98.747 ms` | `+1.390 ms` | `+0.045 ms` | `108.744 ms` | `114.888 ms` | `-6.144 ms` |
| `512/64` | `8.613 ms` | `11.076 ms` | `-2.463 ms` | `203.570 ms` | `200.499 ms` | `+3.070 ms` | `+0.049 ms` | `212.182 ms` | `211.575 ms` | `+0.607 ms` |
| `1024/32` | `15.294 ms` | `15.710 ms` | `-0.417 ms` | `101.320 ms` | `102.646 ms` | `-1.326 ms` | `-0.043 ms` | `116.613 ms` | `118.356 ms` | `-1.743 ms` |
| `1024/64` | `15.303 ms` | `16.252 ms` | `-0.948 ms` | `205.955 ms` | `208.512 ms` | `-2.557 ms` | `-0.041 ms` | `221.259 ms` | `224.764 ms` | `-3.505 ms` |
| `2048/32` | `30.073 ms` | `29.405 ms` | `+0.668 ms` | `104.723 ms` | `110.739 ms` | `-6.016 ms` | `-0.194 ms` | `134.796 ms` | `140.144 ms` | `-5.348 ms` |
| `2048/64` | `29.993 ms` | `29.990 ms` | `+0.003 ms` | `211.888 ms` | `225.484 ms` | `-13.596 ms` | `-0.216 ms` | `241.881 ms` | `255.474 ms` | `-13.593 ms` |

#### 5.1.4 `Qwen2.5-3B-Instruct` retained all-shape

| Shape | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap | Decode-step gap | EdgeFM total | TRT total | Total gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512/32` | `16.057 ms` | `17.386 ms` | `-1.329 ms` | `171.640 ms` | `166.648 ms` | `+4.992 ms` | `+0.161 ms` | `187.697 ms` | `184.034 ms` | `+3.663 ms` |
| `512/64` | `16.087 ms` | `18.218 ms` | `-2.131 ms` | `348.960 ms` | `339.887 ms` | `+9.072 ms` | `+0.144 ms` | `365.047 ms` | `358.106 ms` | `+6.941 ms` |
| `1024/32` | `28.767 ms` | `29.943 ms` | `-1.176 ms` | `171.036 ms` | `172.533 ms` | `-1.497 ms` | `-0.048 ms` | `199.803 ms` | `202.475 ms` | `-2.673 ms` |
| `1024/64` | `28.760 ms` | `29.280 ms` | `-0.520 ms` | `346.214 ms` | `350.497 ms` | `-4.283 ms` | `-0.068 ms` | `374.974 ms` | `379.777 ms` | `-4.803 ms` |
| `2048/32` | `58.110 ms` | `57.025 ms` | `+1.084 ms` | `177.585 ms` | `184.774 ms` | `-7.189 ms` | `-0.232 ms` | `235.695 ms` | `241.800 ms` | `-6.105 ms` |
| `2048/64` | `57.872 ms` | `57.220 ms` | `+0.652 ms` | `359.416 ms` | `374.117 ms` | `-14.701 ms` | `-0.233 ms` | `417.289 ms` | `431.338 ms` | `-14.049 ms` |

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
    - `prefill_ms = 8.6060`
    - `decode_ms = 97.8189`
    - `total_stage_ms = 106.4249`
- 历史 retained 完整套件均值：
    - `prefill_ms = 8.6073`
    - `decode_ms = 100.1371`
    - `total_stage_ms = 108.7811`
- `Qwen2.5-3B-Instruct 512/32`
  - 当前单次复测：
    - `prefill_ms = 16.0420`
    - `decode_ms = 164.5125`
    - `total_stage_ms = 180.5545`
    - `decode_step_avg_ms = 5.3069`
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

### 5.3 VLM 当前最新 retained fullsuite

来源：

- `.tmp_codex/bench/vlm_suite_20260413_all_shapes.json`
- `tests/engine/test_qwen2_generate.py`
- `scripts/profile_vlm_prepared_case.py`
- `.tmp_codex/nsys/vlm3b_51232_edgefm_nodes_20260413.nsys-rep`
- `.tmp_codex/nsys/vlm3b_51232_trt_nodes_20260413.nsys-rep`

这轮已经基于当前保留代码，重新跑完主 VLM prepared benchmark 矩阵：

- 模型：
  - `Qwen2.5-VL-3B-Instruct`
  - `Qwen2.5-VL-7B-Instruct`
- shape：
  - `512/32`
  - `512/64`
  - `1024/32`
  - `1024/64`
  - `2048/32`
  - `2048/64`
- 公平口径：
  - `prepared multimodal / 不计 ViT`
  - `EdgeFM(cuda-graph) vs TRT-Edge-LLM`

#### 5.3.1 模型级均值

这里 gap 统一定义为：

- `(EdgeFM - TRT) / TRT`
- 负值表示 `EdgeFM` 更快

| Model | Avg total gap vs TRT | Avg prefill gap | Avg decode gap | Avg decode-step gap |
| --- | ---: | ---: | ---: | ---: |
| `Qwen2.5-VL-3B-Instruct` | `-0.95%` | `-4.77%` | `-0.73%` | `-0.73%` |
| `Qwen2.5-VL-7B-Instruct` | `-1.39%` | `+1.15%` | `-1.88%` | `-1.88%` |

当前准确判断：

- `VL-3B / VL-7B` 在完整 6-shape 均值上都已经不落后于 `TRT-Edge-LLM`
- `VL-3B` 当前 residual 已经不再是 fullsuite 平均意义上的大 gap，而是集中在 short-context decode
- `VL-7B` 当前已经从“主 key case 落后”转成：
  - `prefill` 略慢一点
  - `decode` 在中长 context 上明显更快

#### 5.3.2 `VL-3B` 最新 all-shape 对 TRT gap

| Shape | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap | Decode-step gap | EdgeFM total | TRT total | Total gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512/32` | `16.055 ms` | `17.798 ms` | `-1.743 ms` | `169.883 ms` | `164.621 ms` | `+5.262 ms` | `+0.170 ms` | `185.938 ms` | `182.419 ms` | `+3.519 ms` |
| `512/64` | `16.050 ms` | `17.739 ms` | `-1.690 ms` | `345.357 ms` | `334.481 ms` | `+10.876 ms` | `+0.173 ms` | `361.406 ms` | `352.220 ms` | `+9.187 ms` |
| `1024/32` | `28.661 ms` | `29.281 ms` | `-0.620 ms` | `170.918 ms` | `170.133 ms` | `+0.785 ms` | `+0.025 ms` | `199.579 ms` | `199.414 ms` | `+0.165 ms` |
| `1024/64` | `28.659 ms` | `29.191 ms` | `-0.531 ms` | `346.527 ms` | `345.751 ms` | `+0.776 ms` | `+0.012 ms` | `375.186 ms` | `374.942 ms` | `+0.244 ms` |
| `2048/32` | `56.011 ms` | `57.682 ms` | `-1.671 ms` | `172.185 ms` | `182.960 ms` | `-10.776 ms` | `-0.348 ms` | `228.196 ms` | `240.642 ms` | `-12.446 ms` |
| `2048/64` | `56.222 ms` | `57.645 ms` | `-1.423 ms` | `349.029 ms` | `369.865 ms` | `-20.837 ms` | `-0.331 ms` | `405.250 ms` | `427.510 ms` | `-22.260 ms` |

当前最重要结论：

- `VL-3B` 剩余问题已经高度收缩到 `512/*` short-context decode
- `1024/*` 基本已经打平
- `2048/*` 已经整体反超 TRT

#### 5.3.3 `VL-7B` 最新 all-shape 对 TRT gap

| Shape | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap | Decode-step gap | EdgeFM total | TRT total | Total gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512/32` | `30.525 ms` | `29.883 ms` | `+0.642 ms` | `300.683 ms` | `300.502 ms` | `+0.181 ms` | `+0.006 ms` | `331.207 ms` | `330.384 ms` | `+0.823 ms` |
| `512/64` | `30.572 ms` | `29.798 ms` | `+0.773 ms` | `610.981 ms` | `610.172 ms` | `+0.809 ms` | `+0.013 ms` | `641.553 ms` | `639.970 ms` | `+1.583 ms` |
| `1024/32` | `58.343 ms` | `57.854 ms` | `+0.490 ms` | `300.932 ms` | `305.951 ms` | `-5.019 ms` | `-0.162 ms` | `359.276 ms` | `363.805 ms` | `-4.529 ms` |
| `1024/64` | `58.648 ms` | `58.522 ms` | `+0.126 ms` | `611.449 ms` | `619.784 ms` | `-8.335 ms` | `-0.132 ms` | `670.097 ms` | `678.307 ms` | `-8.209 ms` |
| `2048/32` | `120.855 ms` | `120.110 ms` | `+0.745 ms` | `301.360 ms` | `315.002 ms` | `-13.642 ms` | `-0.440 ms` | `422.215 ms` | `435.112 ms` | `-12.897 ms` |
| `2048/64` | `120.841 ms` | `120.290 ms` | `+0.550 ms` | `612.216 ms` | `638.760 ms` | `-26.544 ms` | `-0.421 ms` | `733.056 ms` | `759.050 ms` | `-25.994 ms` |

当前最重要结论：

- `VL-7B` 现在只在 `512/*` short-context 上还略慢于 TRT
- 从 `1024/*` 开始已经稳定反超
- `2048/*` 的 decode 优势已经很明确

#### 5.3.4 `VL-0.5B` 当前有效 3-way key cases

`2026-04-13` 已经补齐 `TRT-Edge-LLM` 的 `llava prepared-only` 路径，因此：

- `Qwen2.5-VL-0.5B` 现在可以在本地走公平 3-way prepared benchmark
- 当前合法 shape 仍然只看：
  - `1024/*`
  - `2048/*`
- `512/*` 仍然不是合法 prepared case：
  - base multimodal prompt 长度已经超过 `512`
  - 不再引用任何旧的 `0.5B 512/*` 结果

当前最新保留 valid-shape 结果：

| Shape | EdgeFM prefill | TRT prefill | Prefill gap | EdgeFM decode | TRT decode | Decode gap | Decode-step gap | EdgeFM total | TRT total | Total gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024/32` | `6.595 ms` | `9.851 ms` | `-3.257 ms` | `52.155 ms` | `55.555 ms` | `-3.400 ms` | `-0.110 ms` | `58.749 ms` | `65.406 ms` | `-6.657 ms` |
| `1024/64` | `6.282 ms` | `8.870 ms` | `-2.588 ms` | `107.977 ms` | `112.473 ms` | `-4.496 ms` | `-0.071 ms` | `114.259 ms` | `121.343 ms` | `-7.084 ms` |
| `2048/32` | `12.317 ms` | `13.645 ms` | `-1.328 ms` | `50.861 ms` | `60.949 ms` | `-10.088 ms` | `-0.325 ms` | `63.177 ms` | `74.594 ms` | `-11.416 ms` |
| `2048/64` | `12.315 ms` | `13.206 ms` | `-0.890 ms` | `103.029 ms` | `123.903 ms` | `-20.875 ms` | `-0.331 ms` | `115.344 ms` | `137.109 ms` | `-21.765 ms` |

当前准确判断：

- `VL-0.5B` 已经不再是“只能看 2-way”或“没有 TRT 公平口径”的状态
- 最新这轮保留优化已经把 `0.5B` 的主 residual 从“decode 明显落后 TRT”改成：
  - 当前全部合法 shape 都已经整体反超 TRT
- 当前 `0.5B` 最大的已知收益来自 decode，不是 prefill
- `1024/64` 虽然还有一次 timed run outlier，但 median / trimmed mean 口径依然领先 TRT

#### 5.3.5 当前结构性判断

- `VL-3B` 最新 node-level `nsys` 结论仍然成立：
  - 当前 residual 主要还是 decode 小 shape 线性层实现形态差异
  - attention 次之
  - `mrope` 仍有差距，但不是当前最大头
- 最新全矩阵结果进一步证明：
  - 不是“VLM 全面落后 TRT”
  - 而是：
    - `3B/7B` 在 short-context decode 还有尾巴
    - 中长 context 已经接近打平甚至反超

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
   - 这一步已经完成
   - 当前重点不再是“哪里慢”，而是只针对下面三块做有把握的候选：
     - decode 小 shape linear 实现形态
     - attention decode 实现形态
     - TRT 已有可直接复用的 multimodal / decode plugin 与 kernel 形态
   - 如果继续沿 `fp16 kvcache / fp16 decode attention` 方向推进：
     - 这已经不是 benchmark 脚本修修补补的问题
     - 必须作为一条明确的 mixed-dtype runtime/kernel 改造来做
2. `VL-7B` 主 key case 已基本收平，后续主要做回归保护与 fullsuite 扩展验证
3. `VL-0.5B` 先修 benchmark contract：
   - `llava` prepared 输入链路已经打通
   - 下一步要么给 `512/32 prepared` 换一个合法的短 prompt/image case
   - 要么继续把 `1024/32` 作为首个有效 prepared 对照点扩展出更完整矩阵
   - 同时评估是否需要为 `llava` 补 TRT 可比口径，而不是继续沿用 `image_grid_thw` 假设
4. 在新的 residual 上继续优先看：
   - `decode qkv / attention_output / mlp_down / lm_head` 的 kernel-level residual
   - TRT multimodal 路径里已经存在的 plugin / kernel / shape 策略
   - 继续围绕 `attention / rope-write / fused decode path` 做 node-level 对照
   - 不再优先投入：
     - `row-major Lt`
     - `explicit cublas`
     - 只改 `mrope` 一条小 kernel 的局部尝试
5. 如果候选只在 microbench 更快、但端到端没有稳定收益：
   - 不保留

## 10. 需要持续更新的最小信息集

每轮迭代后，至少更新这几项：

- 最新 retained benchmark 数字
- correctness gate 是否通过
- 当前主 blocker 是什么
- 哪些方向已被证伪，不应再重复尝试
- 是否需要清理过期信息，保证文档持续新鲜
