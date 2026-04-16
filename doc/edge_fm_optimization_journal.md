# EdgeFM 算子调优 Journal

最近更新：2026-04-15

这份文档是当前 `RTX 3060 12GB / sm86` 环境下的主线工作文档，只服务于“算子调优”。
`A800 / sm80` 的历史 benchmark、profiling、runtime 结论已经归档到 `doc/a800_edge_fm_optimization_journal.md`。
除非重新在当前 `3060 / sm86` 环境上复测，否则不要把 A800 的 retained benchmark 和优化结论直接当成当前事实。

## 1. 当前范围

- 本轮只做：
  - 算子实现与算子参数调优
  - 算子表记录更新
  - 算子级 microbench
  - 算子调优带来的端到端 benchmark 验证
- 本轮不做：
  - runtime 路径改动
  - request prepare / prepared multimodal 流程改动
  - cuda-graph capture / replay / replay-state 改动
  - scheduler / sampling / KV cache 管理逻辑改动
  - benchmark contract 改写
  - TensorRT-Edge-LLM runtime 集成逻辑改动

## 2. 当前平台与映射关系

- 当前机器：
  - `NVIDIA GeForce RTX 3060 12GB / sm86`
- 当前仓库没有单独的桌面 `sm86` 多平台 preset；统一走 `3060` 平台入口
- 因此这轮 `3060 / sm86` 调优统一映射到：
  - `PLATFORM=3060`
  - `CMAKE_CUDA_ARCHITECTURES=86`
  - `hw_profile=cuda_sm86`
  - `build-3060/`
- 当前机器实际就是 `3060 / sm86`，这里直接使用仓库内现有的桌面 `sm86` 平台入口

## 3. 默认构建与运行入口

- Docker 构建入口：
  - `scripts/docker/build_cuda.sh`
- 默认平台环境变量：
  - `EDGE_FM_PLATFORM=3060`
- 默认 device：
  - `EDGE_FM_DEVICE_ID=0`
- 默认构建目录：
  - `build-3060/`
- 默认平台配置目录：
  - `examples/config/platform/3060/`

推荐环境变量：

```bash
export EDGE_FM_PLATFORM=3060
export EDGE_FM_DEVICE_ID=0
export EDGE_FM_BUILD_DIR=$PWD/build-3060
export EDGE_FM_HOST_TRT_DIR=/usr/local/TensorRT-10.15.1.29
```

推荐入口：

```bash
# 构建 CUDA tools image
EDGE_FM_PLATFORM=3060 \
EDGE_FM_HOST_TRT_DIR=/usr/local/TensorRT-10.15.1.29 \
bash scripts/docker/build_cuda.sh image

# 配置 / 编译 / 安装
EDGE_FM_PLATFORM=3060 \
EDGE_FM_HOST_TRT_DIR=/usr/local/TensorRT-10.15.1.29 \
bash scripts/docker/build_cuda.sh configure

EDGE_FM_PLATFORM=3060 \
EDGE_FM_HOST_TRT_DIR=/usr/local/TensorRT-10.15.1.29 \
bash scripts/docker/build_cuda.sh build

EDGE_FM_PLATFORM=3060 \
EDGE_FM_HOST_TRT_DIR=/usr/local/TensorRT-10.15.1.29 \
bash scripts/docker/build_cuda.sh install

# 需要 correctness + smoke benchmark 时再显式执行
EDGE_FM_PLATFORM=3060 \
EDGE_FM_HOST_TRT_DIR=/usr/local/TensorRT-10.15.1.29 \
bash scripts/docker/build_cuda.sh verify
```

## 4. 必须遵守的工作规则

- 正确性优先：
  - 任何保留的算子优化都必须先过 correctness gate，再谈 benchmark
- Benchmark / profiling 默认使用 `device=0`
- 不要并行跑 GPU benchmark / profiling
- 只保留 `3060 / sm86` 上重新验证过的结论
- 任何保留的算子优化至少要满足：
  - 算子级 microbench 有稳定收益，或至少不退
  - 目标端到端 case 有稳定收益
  - 不拉坏哨兵 correctness
- 优先复用成熟实现与成熟策略：
  - `FlashInfer`
  - `cuBLASLt`
  - `CUTLASS`
  - `TRT-LLM / TRT-Edge-LLM` 的算子形态与 shape 策略
- 不保留 dead code、一次性 `impl_id`、临时 debug 分支
- 若某条算子候选在 `3060` 上没有形成收益，应直接回退，不继续用 A800 结果硬保留

## 5. 当前 tuning 表与落点

- LLM 主线 tuning 表：
  - `examples/config/base/operator_impl_table_llm.json`
- VLM 主线 tuning 表：
  - `examples/config/base/operator_impl_table_vlm.json`
- 共享表：
  - `examples/config/base/operator_impl_table.json`
  - 仅保留兼容用途，不作为当前主线结果落点
- `3060 / sm86` 这轮物化目标：
  - `examples/config/platform/3060/`

当前分流规则：

- `Qwen2.5-*` 默认命中 `operator_impl_table_llm.json`
- `Qwen2.5-VL-*` 默认命中 `operator_impl_table_vlm.json`
- 最终通过平台物化生成 `examples/config/platform/3060/*.json`

每次改完 base 表后，都要同步物化 `3060`：

```bash
python3 scripts/operator_table/materialize_platform_configs.py --platform 3060
python3 scripts/operator_table/validate_operator_tables.py --platform 3060
```

## 6. 当前 3060 retained 事实

- 当前已经有一批 `3060 / sm86` 可保留的 LLM decode-oriented 调优事实：
  - tuning 工具链已经支持显式 `hw_profile=cuda_sm86`
  - `examples/config/base/operator_impl_table_llm.json` 和 `examples/config/platform/3060/operator_impl_table_llm.json` 已经写入 `sm86` 的 decode tuned records
  - `python3 scripts/operator_table/validate_operator_tables.py --platform 3060` 已通过
- 当前 `3060` 上能长期保留的端到端 spot benchmark 只有：
  - `Qwen2.5-0.5B-Instruct`
    - baseline total median: `152.13 ms`
    - tuned total median: `134.81 ms`
    - baseline decode avg: `130.29 ms`
    - tuned decode avg: `112.75 ms`
    - decode TPS: `210.54 -> 237.99`
  - `Qwen2.5-1.5B-Instruct`
    - baseline total median: `405.23 ms`
    - tuned total median: `369.26 ms`
    - baseline decode avg: `343.20 ms`
    - tuned decode avg: `307.23 ms`
    - decode TPS: `78.97 -> 86.48`
- `Qwen2.5-3B-Instruct` 当前在本机 `prefill=512, decode=32` 的 spot case 上会触发 OOM：
  - 可以保留 `sm86` microbench 调优记录
  - 但当前不能宣称稳定的端到端 retained 收益
- 当前还没有可长期保留的 `3060` profiling 结论
- 当前已经有一条可保留的 `3060` VLM decode-oriented 结果：
  - `Qwen2.5-VL-0.5B`
    - baseline total median: `165.42 ms`
    - tuned total median: `151.69 ms`
    - baseline decode avg: `128.32 ms`
    - tuned decode avg: `114.51 ms`
    - decode TPS: `193.17 -> 210.73`
    - prefill avg 基本不变：`37.23 -> 37.25 ms`
  - 当前保留的 `sm86` decode record 共 `4` 条：
    - attention decode `1` 条
    - linear decode `3` 条：`fused_qkv / attention_output / mlp_down`
- `Qwen2.5-VL-0.5B` 当前的 correctness gate 说明：
  - baseline `sm86` 表与 tuned `sm86` 表在同一 `prefill=1024` prepared 请求上的生成 token 完全一致
  - 直接复用当前 Transformers dump/alignment 脚本对 `0.5B` 做 gold reference 仍有 checkpoint / processor 兼容性问题
  - 因此这次 VLM 0.5B 的 correctness 证据是“tuned 不改变生成 token”，而不是“已经完成稳定的 Transformers token alignment”
- `tests/engine/test_qwen2_generate.py` 的 benchmark helper 仍然把 runtime `hw_profile` 写死成 `cuda_sm80`
  - 因此 `3060 / sm86` 的 retained benchmark 目前使用自定义 engine config / Python snippet 验证
  - 这不影响当前“只做算子调优、不改 runtime/test 逻辑”的工作边界
- `A800` 文档里的以下信息都不再视为当前事实：
  - retained fullsuite gap
  - 哪些 VLM residual 已经收平
  - 哪些 runtime 结论已经验证有效
  - 哪些 decode / prefill 优化应默认保留

当前准确判断：

- 第一条真正有效的 `3060 / sm86` 调优方向，是让 operator tuning 和算子表真正命中 `cuda_sm86`
- 当前 `LLM 0.5B / 1.5B` 的收益主要来自 decode path，而不是 prefill path
- 当前 `VLM 0.5B` 的收益也主要来自 decode path，而不是 prefill path
- 下一轮 VLM 调优应优先扩大到更重的 shape，而不是继续在 `0.5B` 上反复细抠

## 7. 当前 benchmark 口径

- 如果需要算 gap，统一定义为：
  - `(EdgeFM - TRT) / TRT`
  - 负值表示 `EdgeFM` 更快
- `prefill_ms` 仍然按 prefill phase 时间解释，不直接当作纯 kernel 时间
- 因此每次算子调优都要同时看两层：
  - 算子 microbench
  - 端到端 benchmark

如果出现以下情况，不能直接宣布优化有效：

- 算子 microbench 变快，但端到端无收益
- 端到端变快，但 correctness 不稳定
- 单个 run 好看，但多次重复不稳定

## 8. 本轮 3060 的起始调优范围

优先级从高到低：

- VLM decode attention
- VLM decode linear / `cuBLASLt` tactic
- VLM decode fused gate_up + SwiGLU
- LLM decode attention / linear 的回归保护
- prefill 相关算子只在 decode 方向没有收益时再展开

允许做的实现侧动作：

- 调整现有算子的 tile / split / launch / vectorization / tactic 选择
- 增删 `operator_impl_table_*` 中的 tuned record
- 在算子内部补充低风险、局部性的 shape-specialized fast path
- 对齐 `TRT-Edge-LLM` 或 `FlashInfer` 的算子参数和 kernel 形态

不允许做的实现侧动作：

- 改 `StandardEngine` 或其他 runtime 主流程
- 改 prepared request contract
- 改 graph capture/replay 逻辑
- 改 KV cache 生命周期与接口
- 改 benchmark 脚本的公平口径来“制造收益”

## 9. 3060 首轮 baseline 建议

在显存允许范围内，建议先建立以下 baseline：

- LLM 哨兵：
  - `Qwen2.5-0.5B-Instruct`
  - `Qwen2.5-1.5B-Instruct`
  - `Qwen2.5-3B-Instruct`
- VLM 哨兵：
  - `Qwen2.5-VL-0.5B`
  - `Qwen2.5-VL-3B-Instruct`
  - `Qwen2.5-VL-7B-Instruct` 仅在本机可稳定装载时纳入
- 默认端到端 case：
  - `prefill=512, decode=32`
  - `prefill=1024, decode=32`

说明：

- `3060 12GB` 的显存约束与 `A800` 不同，不继承 `A800` 的 fullsuite 覆盖承诺
- 若 `VLM-7B` 在当前机器上不稳定或 OOM，应先收敛到 `0.5B / 3B`

## 10. 通过标准

一条算子优化要进入保留状态，至少需要：

- 对应模型或热点 shape 的 correctness 通过
- 算子级 microbench 有稳定收益或至少不退
- 至少一个目标端到端 case 有稳定收益
- 不拉坏以下回归集：
  - `Qwen2.5-0.5B-Instruct`
  - `Qwen2.5-1.5B-Instruct`
  - 至少一个当前正在优化的 VLM 哨兵

## 11. 文档维护规则

- 当前文档只记录 `3060 / sm86` 上重新验证过的有效事实
- A800 的历史记录只追加到 `doc/a800_edge_fm_optimization_journal.md`
- 一旦有新的 `3060` retained benchmark 或 profiling 结论，应直接更新这份文档
- 一旦旧结论被新测量覆盖，应及时删除，不保留“也许还成立”的旧判断
