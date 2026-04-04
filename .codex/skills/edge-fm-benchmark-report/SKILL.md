---
name: edge-fm-benchmark-report
description: Generate detailed benchmark reports inside this edge-fm repo, with the default fair comparison set to Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM on Qwen2.5 CUDA workloads. Use when the user asks for benchmark指标、性能对比、详细报告、cuda-graph vs TRT-Edge-LLM analysis, or wants the existing tests/engine/test_qwen2_generate.py benchmark flow wrapped into a repeatable local workflow.
---

# edge-fm Benchmark Report

优先复用仓库现有 benchmark helper，而不是手写另一套测量逻辑。当前默认入口是 `tests/engine/test_qwen2_generate.py`，其中已经固定了 warmup、timed runs、`ignore_stop_tokens=True` 和 Qwen2.5 的 dump 数据。

## 什么时候用

- 用户要 `Transformers`、`EdgeFM`、`TRT-Edge-LLM` 的详细性能报告。
- 用户明确要求看 `EdgeFM(cuda-graph)` 和 `TRT-Edge-LLM` 的公平对比。
- 用户要判断差距来自 `cuda-graph`、kernel 实现，还是 tail latency。
- 用户希望把 benchmark 结果整理成稳定口径，而不是只看单次均值。

## 默认口径

- 默认主报告必须是三框架：`Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM`
- `EdgeFM` 默认使用 `cuda-graph`，不能默认退回 no-graph
- 如果需要 no-graph，只作为附加项，不作为主对比口径

## 工作流

1. 先确认环境变量：
   - `LD_LIBRARY_PATH=/xs-train-nas/zzm/packages/TensorRT-10.16.0.72/lib:${LD_LIBRARY_PATH:-}`
   - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
   - `EDGE_FM_DEVICE_ID=1`（或用户指定 GPU）
2. 优先运行本 skill 自带脚本：
   - `python3 .codex/skills/edge-fm-benchmark-report/scripts/report_qwen_3way_cuda_graph_vs_trt.py`
3. 如果用户要和现有 tests 对齐，再补跑：
   - `pytest -s tests/engine/test_qwen2_generate.py -q -k test_benchmark_llm`
   - `pytest -s tests/engine/test_qwen2_generate.py -q -k test_benchmark_trt_edgellm`
4. 输出报告时，默认同时给三类口径：
   - 原始均值
   - 中位数
   - 去掉最大值后的 trimmed mean
5. 如果 `EdgeFM(cuda-graph)` 出现首个 timed run 异常偏大，不要只报均值；必须说明 steady-state 口径。

## 报告要求

- 主表必须包含：`Transformers`、`EdgeFM(cuda-graph)`、`TRT-Edge-LLM`
- 重点分析 `TRT-Edge-LLM` 相对 `EdgeFM(cuda-graph)` 的差距
- 明确 benchmark 配置：模型、batch、prefill、decode、warmup、runs、device
- 列出每次 timed run
- 给出 latency / total throughput / decode throughput
- 明确写出：
  - `TRT vs EdgeFM(cuda-graph)` 的 mean speedup
  - `TRT vs EdgeFM(cuda-graph)` 的稳态 speedup（median 或 trimmed mean）
  - `EdgeFM(cuda-graph)` 是否存在 outlier / tail latency
- 如果引用 `tests/engine/test_qwen2_generate.py` 的实现细节，标明对应文件路径

## 直接脚本

- 主脚本：`scripts/report_qwen_3way_cuda_graph_vs_trt.py`
- 默认行为：直接输出 `Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM`
- 可选：加 `--also-edgefm-no-graph` 把 `EdgeFM(no-graph)` 也一起带上
- 可选：加 `--json-only` 输出结构化结果

## 注意事项

- `TRT-Edge-LLM` 的 engine 和 runtime 必须和当前 TensorRT 版本匹配
- 如果用户只看“最终 steady-state 性能”，优先强调 median / trimmed mean，不要只给 mean
- 如果用户问“为什么差这么大”，先区分是 graph capture、初始化抖动，还是 steady-state kernel 性能差
