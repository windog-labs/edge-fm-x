# EdgeFM CUDA Kernel Optimizer 使用指南

本文说明如何使用仓库内的
`.codex/skills/edge-fm-cuda-kernel-optimizer` 为新的 NVIDIA 硬件平台做
EdgeFM 模型性能调优。目标不是盲目写 kernel，而是建立一套可复现流程：
先让 EdgeFM 与 `TRT-Edge-LLM` 在同一模型、同一 shape、同一运行参数下
对齐，再用 NSYS/NCU 定位 gap，最后通过 source-op、plugin-op 或
Humanize/KernelPilot 长循环逐步追平甚至超过 reference。

## 1. 安装和启用 skill

在 `edge-fm-x` 仓库内，skill 已随仓库提供：

```bash
.codex/skills/edge-fm-cuda-kernel-optimizer/SKILL.md
```

通常不需要额外安装。新开 Codex 会话后，在调优请求中显式引用
`$edge-fm-cuda-kernel-optimizer`，Codex 会读取该 skill 并按它的流程执行。

如果要在另一个仓库复用该能力，复制整个目录到目标仓库：

```bash
mkdir -p .codex/skills
cp -a /path/to/edge-fm-x/.codex/skills/edge-fm-cuda-kernel-optimizer \
  .codex/skills/
```

长时间 autonomous 优化需要 Humanize hooks。只安装 hooks，不会自动启动优化：

```bash
bash .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/install_humanize_hooks.sh
```

常用入口：

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/check_env.py \
  --out .tmp_codex/env/gpu_env.json

python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/analyze_edgefm_nsys_profile.py \
  --mapping-input .tmp_codex/nsys/edgefm_graph_off.nsys-rep \
  --formal-input .tmp_codex/nsys/edgefm_graph_on.nsys-rep
```

## 2. 配置 GPU profiling 环境

新平台调优前先确认这些工具可用：

```bash
nvidia-smi
nvcc --version
nsys --version
ncu --version
python3 --version
cmake --version
```

EdgeFM CUDA 平台常用环境变量：

```bash
export EDGE_FM_BUILD_DIR=/path/to/edge-fm-x/build-<platform>
export EDGE_FM_PLATFORM=<platform>
export EDGE_FM_DEVICE_ID=0
export EDGE_FM_TEST_DEVICE_ID=0
export LD_LIBRARY_PATH=$EDGE_FM_BUILD_DIR/lib:$EDGE_FM_BUILD_DIR/install/lib:${LD_LIBRARY_PATH:-}
```

NCU 可执行不等于 counter 权限可用。先做最小 smoke：

```bash
ncu --set basic --target-processes all --version
```

如果 profiling 时报 `ERR_NVGPUCTRPERM`，不要把缺失 counter 当成真实性能结论。
应先配置 GPU counter 权限，或临时退回 NSYS attribution 与 operator microbench。
需要 sudo 时只使用窄权限方式，不要把 sudo 密码写入命令、脚本或文档。

如果要和 `TRT-Edge-LLM` 对比，需提前准备：

- EdgeFM 模型 artifact 和 config。
- TRT-Edge-LLM engine workspace。
- 对应 plugin library，例如 `libNvInfer_edgellm_plugin.so`。
- 同一组 model size、prefill length、decode length、warmup、runs、CUDA graph 设置。

## 3. 新硬件平台调优流程

推荐按下面顺序推进，每一步都要留下 artifact 路径和接受/拒绝原因。

### 3.1 建立 paired benchmark baseline

先跑 EdgeFM 与 TRT-Edge-LLM 的 paired matrix。至少覆盖目标模型尺寸和常用
prefill/decode shape：

```bash
python3 scripts/profile/profile_edgefm_generate_case.py \
  --model-path /path/to/model \
  --prefill-len 2048 \
  --decode-len 64 \
  --use-cuda-graph \
  --runs 3 \
  --json

python3 scripts/profile/profile_trt_edgellm_generate_case.py \
  --model-path /path/to/model \
  --engine-dir /path/to/trt_workspace \
  --plugin-path /path/to/libNvInfer_edgellm_plugin.so \
  --prefill-len 2048 \
  --decode-len 64 \
  --runs 3 \
  --json
```

输出表至少包含：

- EdgeFM total / prefill / decode
- TRT total / prefill / decode
- total gap
- prefill gap
- decode gap
- tokens/s 或 decode step avg

### 3.2 先做 NSYS attribution，再决定 NCU 目标

CUDA graph 会隐藏 kernel attribution。先采 graph-off mapping trace，再采
graph-on formal trace：

```bash
nsys profile -o .tmp_codex/nsys/edgefm_graph_off \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model \
    --prefill-len 2048 \
    --decode-len 64 \
    --profile-range

nsys profile -o .tmp_codex/nsys/edgefm_graph_on \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 scripts/profile/profile_edgefm_generate_case.py \
    --model-path /path/to/model \
    --prefill-len 2048 \
    --decode-len 64 \
    --use-cuda-graph \
    --profile-range
```

然后用 skill 脚本生成 kernel table、known-path table 和 action table：

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/analyze_edgefm_nsys_profile.py \
  --mapping-input .tmp_codex/nsys/edgefm_graph_off.nsys-rep \
  --formal-input .tmp_codex/nsys/edgefm_graph_on.nsys-rep
```

只把满足这些条件的 kernel 放进 NCU 或 Humanize 队列：

- 目标 slice 有明确端到端 gap。
- kernel/operator 占总 gap 比例足够大。
- 有 correctness reference 或 operator/layer 级测试。
- 现有 operator table、cuBLASLt、FlashInfer、CUTLASS 小参数 sweep 已经到平台期。

### 3.3 按 operator gap 排队

常见优先级：

1. Prefill attention / FMHA / KV write / RoPE。
2. QKV、OProj、MLP GateUp/Down 等 dense linear。
3. Decode attention 和 decode linear。
4. Norm、sampler、finalize、response copy。
5. Runtime 层面的 launch、CUDA graph、stream overlap、host sync。

先尝试低风险路线：

- 更新 platform operator table。
- 调整已有 source-op / FlashInfer / cuBLASLt / CUTLASS 参数。
- 增加 model/shape/stage 级选择记录。

当短程调优没有稳定收益，再进入 Humanize + KernelPilot。

### 3.4 Humanize + KernelPilot 长循环

进入条件：

- 有明确热点、shape、baseline、reference 和验证入口。
- 预计需要多轮候选、NCU 证据、source provenance 和 rejected ledger。
- TRT 或其他 reference 明显更快，但不能直接依赖 serialized TensorRT engine。

建议目录：

```bash
deliverables/kernel_opt/<platform>_<operator>_<date>/
```

每个长循环至少保留：

- baseline / reference / dims
- NCU baseline digest
- attempt ledger
- optimization ledger
- source idea ledger
- accepted / rejected summary

回迁规则：

1. standalone/repro 先证明 correctness 和 latency。
2. 回迁到 `src/operators`、`src/layers` 或 operator table。
3. rebuild 并安装 Python binding。
4. 跑 operator/layer/engine 回归。
5. 跑 paired benchmark，确认收益没有在真实 generate 链路消失。

## 4. 调优提示词模板

### 4.1 生成新平台调优方案

```text
请使用 $edge-fm-cuda-kernel-optimizer 为 <platform> 制定 EdgeFM LLM
性能调优方案。目标是追平并尽量超过 TRT-Edge-LLM。

约束：
1. 不做大架构改造。
2. 先建立 EdgeFM vs TRT-Edge-LLM paired benchmark matrix。
3. 用 graph-off NSYS 做 kernel attribution，用 graph-on trace 确认端到端行为。
4. 输出 operator gap 表，至少包括 prefill attention、QKV/OProj、MLP、
   decode attention、norm/sampler/finalize。
5. 优先尝试 operator table、source-op、FlashInfer、cuBLASLt、CUTLASS 小范围调优。
6. 短程平台期后再启动 Humanize + KernelPilot 长循环。
7. 每个 accepted/rejected 节点都更新文档并记录 artifact。
```

### 4.2 持续优化某个 operator

```text
请使用 $edge-fm-cuda-kernel-optimizer 持续优化 <operator>。

目标 slice:
- model: <model>
- prefill_len: <N>
- decode_len: <M>
- platform: <platform>

要求：
1. 先确认当前 EdgeFM 与 TRT-Edge-LLM 的 prefill/decode/total gap。
2. 找到该 operator 在 NSYS/NCU 中的真实 kernel 名和耗时。
3. 优先建立 operator/layer 级 benchmark 或 standalone repro。
4. 每轮只吸收 correctness 通过且端到端不回退的改动。
5. 如果两轮短程优化进入平台期，升级到 Humanize + KernelPilot。
```

### 4.3 判断能否移除 TRT bridge

```text
请评估当前 <platform>/<model> 是否可以移除 TRT bridge。

请区分三种模式：
- native/source-op: EdgeFM 自有 CUDA/CUTLASS/FlashInfer 路径。
- plugin-op: 可复用 source-visible TRT-Edge-LLM plugin/kernel，但不加载 serialized engine。
- trt-reference: 只作为 benchmark reference 或 fallback。

验收标准：
1. source-op 或 plugin-op 在目标 matrix 上达到 practical parity。
2. 正确性回归通过。
3. 相邻 shape 没有明显退化。
4. README/doc 中明确默认路径是否还依赖 TensorRT engine/context。
```

### 4.4 启动 Humanize + KernelPilot

```text
请使用 $edge-fm-cuda-kernel-optimizer 和 $humanize-kernel-agent-loop
为 <kernel/operator> 启动长循环优化。

前置条件：
- 已有 hotspot 证据。
- 已有 correctness reference。
- 已有 standalone 或 operator benchmark。
- 已有 NCU baseline digest。

输出要求：
- deliverables/kernel_opt/<name>/ standalone repo。
- refined plan。
- source/lineage ledger。
- attempt/optimization ledger。
- 每轮 profile evidence digest。
- 回迁条件和最终 accept/reject 结论。
```

### 4.5 阶段性汇报

```text
请汇总当前调优阶段结果：
1. 当前 best matrix 与 TRT-Edge-LLM 的 gap。
2. accepted 优化项和收益。
3. rejected 路线和原因。
4. 剩余最大 operator gap。
5. 下一轮计划。
6. artifact 路径。
```

## 5. 验收标准

每个新平台调优阶段至少满足：

- correctness 优先于 latency。
- paired benchmark 同 shape、同 runs、同 CUDA graph 设置。
- accepted 改动有 artifact 和回归结果。
- rejected 路线写清楚原因，避免重复烧时间。
- 默认路径不依赖未说明的 TensorRT engine/context。
- README 或 doc 中保留当前有效性能矩阵，临时 tuning log 不长期堆在 `doc/`。
