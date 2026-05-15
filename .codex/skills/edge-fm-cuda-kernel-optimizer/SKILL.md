---
name: edge-fm-cuda-kernel-optimizer
description: NCU-guided CUDA/CUTLASS kernel optimization workflow for edge-fm-x, including short standalone/repo-in-place tuning and long Humanize + KernelPilot loops. Use when optimizing kernels under src/, tuning standalone CUDA/CUTLASS reproductions, or when the user asks to speed up an Edge-FM CUDA kernel and mentions Nsight Compute, Nsight Systems, NCU, benchmark plateaus, Humanize, KernelPilot, long-running kernel search, source provenance, ledgers, CUTLASS, latency regressions, or iterative optimization.
---

# Edge-FM CUDA Kernel Optimizer

这个 skill 把 NSYS 热点预诊断、短程 NCU 优化和长程 Humanize + KernelPilot 优化都收口到 `edge-fm-x` 的一个入口。重点不是直接盲改生产 kernel，而是先定位热点、建立可复现 baseline，再做 profile -> 选方法 -> 改代码 -> 校验 -> benchmark 的闭环；当任务需要多小时、多版本、多来源证据时，再升级到带外部审查和 ledgers 的长循环。

## 什么时候用

- 用户明确要优化 `src/` 下的 CUDA / CUTLASS kernel
- 已经知道热点 kernel，或至少已经缩小到某个 operator / layer
- 用户要分析 Edge-FM `.nsys-rep` / `.sqlite`，或提到 `nsys` / `Nsight Systems` / CUDA graph profile
- 用户提到 `ncu` / `Nsight Compute` / kernel tuning / CUTLASS / “make this kernel faster”
- 需要做多轮迭代，而不是一次性盲猜优化
- 用户提到 Humanize / KernelPilot / 长期 autonomous search / 版本 lineage / profile evidence / source provenance

## 什么时候先不要用

- 只知道“整模型慢”，但没有 trace、profile 入口或可复现实验
- 当前先是功能错误、编译失败、接口不匹配，不是性能问题
- 没有 reference 或最小可验证输入

如果热点还不清楚，先读 `references/edge_fm_workflow.md`，用仓库现有测试和 profile 脚本把问题收敛出来，再进入本 skill。

## 先选工作模式

### 模式 0: NSYS 热点预诊断

适用于已有 Edge-FM `.nsys-rep` / `.sqlite`，或需要先判断整条 generate/VLM profile 里哪个 stage、layer、kernel 最值得继续优化。

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/analyze_edgefm_nsys_profile.py \
  --input /path/to/profile.nsys-rep
```

CUDA graph 场景优先用 graph-off mapping trace 补 attribution，再用 graph-on formal trace 看最终行为：

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/analyze_edgefm_nsys_profile.py \
  --mapping-input /path/to/graph_off.nsys-rep \
  --formal-input /path/to/graph_on.nsys-rep
```

这个脚本固定输出 kernel table、known-path table、action table。先按 action table 检查既有 Edge-FM operator/tuning 路线；不要在这些路径排除前把热点当成全新 kernel 机会。

### 模式 A: 短程独立基准闭环

适用于你已经有一个独立 CUDA/CUTLASS kernel 文件，能直接接入本 skill 自带的 `benchmark.py`：

- CUDA / CUTLASS: `extern "C" void solve(...)`
- reference: `reference(**kwargs)` in `ref.py`

这种模式直接用自带脚本跑完整优化循环。`benchmark.py` 仍保留 Triton 兼容入口，只用于已有 baseline 或用户明确指定的外部对照；默认不要把 Triton 作为 edge-fm 的候选实现路线。

### 模式 B: 短程 edge-fm 仓库内 kernel

适用于目标在 `src/layers/*.cu`、`src/operators/*.cu`、`src/utils/device/*.cu` 等正式实现中。

做法：

1. 先按 `references/edge_fm_workflow.md` 找到对应测试和 profile 入口。
2. 如果当前 kernel 不能直接接入 `benchmark.py`，优先做一个最小复现。
3. 推荐把复现文件放在 `deliverables/kernel_opt/<kernel_name>/`，避免一开始就在生产代码上做多轮试错。
4. 只有当复现版本已经证明正确且更快，再把有效改动回迁到 `src/`。
5. 回迁后必须 rebuild 并运行目标 pytest / benchmark 做回归。

### 模式 C: Humanize + KernelPilot 长循环

适用于 QKV/OProj、W4A16/W8A8 linear、prefill attention/KV、fused gate-up、`lm_head_top1`、DeepGEMM/TensorRT bridge 这类需要长期探索的优化任务。

进入条件：

- 已经有明确热点、shape、baseline、验证入口和收益口径。
- 预计需要多轮候选、NCU 证据、失败记录、来源记录和外部 review。
- 用户明确要求 Humanize / KernelPilot，或连续短程优化已经 plateau。

先读 `references/humanize_kernelpilot_long_loop.md`。不要在未知热点、没有 correctness reference、或仓库不干净时启动 RLCR；先建立 standalone repo 和计划。

## 必读文件

- `references/edge_fm_workflow.md`
  目标在当前仓库内部，需要知道路径、构建、测试映射、热点定位方式时读取。
- `references/optimization_catalog.md`
  选优化方法时必须先读。每个 axis 必须按优先级严格扫描。
- `references/ncu_metrics_guide.md`
  把 `ncu_top.json` 的指标映射到 compute / memory / latency 三个轴。
- `references/profile_known_paths.md`
  分析 Edge-FM NSYS trace 后，判断 hotspot 是否已经对应现有 operator/tuning 路线时读取。
- `references/humanize_kernelpilot_long_loop.md`
  当任务需要 Humanize + KernelPilot 长循环、source ledgers、profile evidence digest 或 RLCR review gate 时读取。

## 自带资源

- `scripts/benchmark.py`
  独立 benchmark + correctness harness
- `scripts/check_env.py`
  采集 GPU / nvcc / ncu / CUTLASS / torch 环境，并记录可选 Triton 环境用于对照
- `scripts/preflight.py`
  校验 baseline / reference / dims 契约
- `scripts/orchestrate.py`
  串起 setup / close-iter / finalize
- `scripts/profile_ncu.py`
  采集 `.ncu-rep` 并生成 `ncu_top.json`
- `scripts/analyze_edgefm_nsys_profile.py`
  分析 Edge-FM `.nsys-rep` / `.sqlite`，输出 kernel、known-path、action 三张表
- `scripts/state.py`
  维护 run 状态和方法有效性
- `scripts/install_humanize_hooks.sh`
  安装/更新 Codex Stop hook，使其指向本 skill 内 vendored Humanize runtime；只做 hook 配置，不启动 RLCR。
- `templates/iteration_report.md`
  每轮分析模板
- `vendor/humanize/`
  Humanize runtime：scripts、hooks、prompt-template、templates、config、agents，以及内部 `humanize-kernel-agent-loop` 参考。
- `vendor/kernel-pilot/`
  KernelPilot 知识库、source catalog、`kernel-knowledge` 和 `profile-evidence` 内部参考。

## 需要的输入

做模式 A 时，开始前至少确认：

1. `baseline`
   例如 `./gemm.cu`
2. `ref`
   例如 `./ref.py`
3. `dims`
   例如 `{"M":4096,"N":4096,"K":4096}`
4. `iterations`
   默认 3
5. `ncu_num`
   每个 axis 提取多少个 top 指标，默认 5

做模式 B 时，还要确认：

1. 目标生产文件路径
2. 对应验证入口
3. 当前 build 目录或平台配置
4. 是否已有最小复现；没有就先建立

## 推荐流程

### 1. 建立最小可验证入口

- 对仓库内 kernel，不要一开始就在整条 generate 链路里迭代
- 优先缩到 operator / layer 级测试；必要时再抽成独立 repro
- reference 和 correctness 不稳定时，不要进入迭代优化

### 2. 先做环境和契约检查

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/check_env.py --out ./env.json

python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/preflight.py \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}'
```

如果目标是仓库内 kernel，但还没有独立 baseline / ref 契约，先回到 `references/edge_fm_workflow.md` 建立复现。

### 3. 初始化 run 并播种 baseline

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --iterations 3 \
  --ncu-num 5 \
  --dims '{"M":4096,"N":4096,"K":4096}'
```

这一步会：

- 记录环境
- 校验 baseline / ref / dims
- 在 baseline 同目录生成 `run_YYYYMMDD_HHMMSS/`
- 跑 baseline correctness + benchmark
- 采第一轮 `best_input.ncu-rep` 和 `ncu_top.json`

### 4. 每轮都按固定闭环执行

对第 `i` 轮，顺序固定：

1. 读 `iterv{i}/ncu_top.json`
2. 读 `state.json`
3. 读当前 `best_file`
4. 读 `references/optimization_catalog.md`
5. 读 `references/ncu_metrics_guide.md`
6. 每个 axis 只选一个方法，且严格按 catalog 优先级扫描
7. 写出：
   - `iterv{i}/kernel.<ext>`
   - `iterv{i}/methods.json`
   - `iterv{i}/analysis.md`
8. 执行：

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/orchestrate.py close-iter \
  --run-dir <run_dir> \
  --iter <i>
```

如果 correctness 失败，先修复本轮 kernel，再重新执行 `close-iter`。单轮最多重试 3 次。

### 5. 结束后生成总结

```bash
python .codex/skills/edge-fm-cuda-kernel-optimizer/scripts/orchestrate.py finalize \
  --run-dir <run_dir>
```

输出 `summary.md`，包含环境快照、每轮方法与耗时、最终 best kernel、有效 / 无效方法清单和后续建议。

## 选方法时的硬约束

- 必须一轮三选：`compute` / `memory` / `latency` 各一个
- 必须优先级扫描，不能跳过更高优先级的可用方法
- 已在 `selected_methods` 里的方法不能重复选
- `ineffective_methods` 中的方法默认禁止复用，除非 bottleneck 已明显变化
- `memory.multi_stage_pipeline` 和 `latency.async_pipeline` 不能同时选
- 选中的方法必须和当前 `sm_arch` 兼容
- `analysis.md` 必须写清楚为什么不是更高优先级的方法

## edge-fm 回迁规则

当独立 repro 上的优化已经被证明正确且更快，才回迁到正式代码：

1. 修改 `src/` 对应实现
2. rebuild 当前 build 目录
3. 安装 Python 绑定
4. 运行目标 pytest
5. 必要时再跑一遍 NCU 或 benchmark，确认收益没有在正式链路里消失

具体命令和测试映射看 `references/edge_fm_workflow.md`。

## 常见失败模式

- `ncu` 可执行但读不到 perf counter
  这时 `ncu_top.json` 可能退化，不能把 0 指标当成真实瓶颈
- baseline 自身 correctness 不通过
  先修 baseline / ref 契约，不要继续迭代
- 整模型 benchmark 更快，但单 kernel 指标没动
  说明收益可能来自 launch 次数、图捕获或调度，不一定是目标 kernel 本身
- 只在 standalone repro 上有效，回到 edge-fm 后无收益
  优先检查真实输入 shape、调度路径、launch config、编译选项是否一致
- 最终 SASS 基本不变
  说明“意图优化”没有真正落到代码生成上，需回看 ncu 对比和编译产物

## 输出要求

完成一次优化任务后，尽量留下这些工件：

- 可复现的 baseline / ref / dims
- `run_*/state.json`
- 每轮 `analysis.md`
- 每轮 `methods.json`
- `summary.md`
- 如果改回 `src/`，还要给出对应测试和 benchmark 结果

这个 skill 的重点不是只给一个“看起来更快”的 patch，而是留下一个还能继续复验和迭代的性能工作流。
