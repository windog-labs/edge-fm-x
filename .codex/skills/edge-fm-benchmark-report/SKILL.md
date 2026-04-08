---
name: edge-fm-benchmark-report
description: Generate detailed benchmark reports inside this edge-fm repo, covering configurable Qwen2.5 LLM/VLM model matrices on CUDA. The default fair comparison remains Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM for LLM, while VLM currently uses Transformers vs EdgeFM(cuda-graph). Use when the user asks for benchmark指标、性能对比、详细报告、cuda-graph vs TRT-Edge-LLM analysis, VLM benchmark reports, or wants the existing tests/engine/test_qwen2_generate.py benchmark flow wrapped into a repeatable local workflow.
---

# edge-fm Benchmark Report

优先复用仓库现有 benchmark helper，而不是手写另一套测量逻辑。当前默认入口是 `tests/engine/test_qwen2_generate.py`，其中已经固定了 warmup、timed runs、`ignore_stop_tokens=True`，并支持按模型矩阵运行。

## 什么时候用

- 用户要 `Transformers`、`EdgeFM`、`TRT-Edge-LLM` 的详细性能报告。
- 用户明确要求看 `EdgeFM(cuda-graph)` 和 `TRT-Edge-LLM` 的公平对比。
- 用户要 `VLM` 的 `Transformers vs EdgeFM(cuda-graph)` 性能报告。
- 用户要一口气看多种模型 size，而不是单个 1.5B case。
- 用户要判断差距来自 `cuda-graph`、kernel 实现，还是 tail latency。
- 用户希望把 benchmark 结果整理成稳定口径，而不是只看单次均值。

## 默认口径

- LLM 默认主报告优先是三框架：`Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM`
- VLM 默认主报告会优先尝试三框架：`Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM`
- 如果本地缺少 VLM 的 TRT LLM / multimodal engine，VLM 自动退回两框架：`Transformers vs EdgeFM(cuda-graph)`
- `EdgeFM` 固定使用 `cuda-graph` 作为唯一主对比口径，不再分析 no-graph 中间态
- 默认模型矩阵：
  - LLM: `0.5B, 1.5B, 3B`
  - VLM: `3B, 7B`
- 默认输入矩阵：
  - prefill: `512, 1024, 2048`
  - decode: `32, 64`
- 若某个模型权重缺失，或 LLM 缺少 TRT engine，该模型应单独标记为 skipped，而不是让整组 benchmark 失败

## 工作流

1. 先确认环境变量：
   - `LD_LIBRARY_PATH=/xs-train-nas/zzm/packages/TensorRT-10.16.0.72/lib:${LD_LIBRARY_PATH:-}`
   - `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
   - `EDGE_FM_DEVICE_ID=1`（或用户指定 GPU）
2. 优先运行本 skill 自带 suite 脚本：
   - `python3 .codex/skills/edge-fm-benchmark-report/scripts/report_qwen_benchmark_suite.py`
3. 如果用户只看 legacy 单模型 LLM 3-way，对齐旧报告时再运行：
   - `python3 .codex/skills/edge-fm-benchmark-report/scripts/report_qwen_3way_cuda_graph_vs_trt.py`
4. 如果用户要和现有 tests 对齐，再补跑：
   - `pytest -s tests/engine/test_qwen2_generate.py -q -k test_benchmark_llm`
   - `pytest -s tests/engine/test_qwen2_generate.py -q -k test_benchmark_trt_edgellm`
   - `pytest -s tests/engine/test_qwen2_generate.py -q -k test_benchmark_vlm`
5. 如需限制模型矩阵，可用：
   - `EDGE_FM_BENCH_LLM_MODELS=0.5b,1.5b,3b`
   - `EDGE_FM_BENCH_VLM_MODELS=3b,7b`
   - `EDGE_FM_BENCH_PREFILL_LIST=512,1024,2048`
   - `EDGE_FM_BENCH_DECODE_LIST=32,64`
6. 输出报告时，默认同时给三类口径：
   - 原始均值
   - 中位数
   - 去掉最大值后的 trimmed mean
7. 如果 `EdgeFM(cuda-graph)` 出现首个 timed run 异常偏大，不要只报均值；必须说明 steady-state 口径。

## 报告要求

- LLM 主表必须包含：`Transformers`、`EdgeFM(cuda-graph)`、`TRT-Edge-LLM`
- VLM 主表至少必须包含：`Transformers`、`EdgeFM(cuda-graph)`
- 若本地 VLM TRT engine 可用，则 VLM 主表也应包含：`TRT-Edge-LLM`
- 重点分析 `TRT-Edge-LLM` 相对 `EdgeFM(cuda-graph)` 的差距；对 VLM，则重点看 `EdgeFM(cuda-graph)` 相对 `Transformers` 的收益和稳定性
- 明确 benchmark 配置：模型、模型类型（LLM/VLM）、batch、prefill、decode、warmup、runs、device
- 列出每次 timed run
- 给出 latency / total throughput / decode throughput
- 明确写出：
  - LLM: `TRT vs EdgeFM(cuda-graph)` 的 mean speedup
  - LLM: `TRT vs EdgeFM(cuda-graph)` 的稳态 speedup（median 或 trimmed mean）
  - VLM: `EdgeFM(cuda-graph) vs Transformers` 的 mean speedup
  - VLM: `EdgeFM(cuda-graph) vs Transformers` 的稳态 speedup（median 或 trimmed mean）
  - 若 VLM TRT 可用：`TRT-Edge-LLM vs EdgeFM(cuda-graph)` 的 mean / 稳态 speedup
  - `EdgeFM(cuda-graph)` 是否存在 outlier / tail latency
- 如果某些模型被 skipped，必须明确标出来是因为缺权重还是缺 TRT engine
- 如果引用 `tests/engine/test_qwen2_generate.py` 的实现细节，标明对应文件路径

## 直接脚本

- 主 suite 脚本：`scripts/report_qwen_benchmark_suite.py`
- legacy 单模型 LLM 3-way 脚本：`scripts/report_qwen_3way_cuda_graph_vs_trt.py`
- 默认行为：
  - suite 脚本：输出可用模型的多模型 benchmark 报告，LLM 优先 3-way；VLM 若 TRT assets 可用则 3-way，否则 2-way
  - legacy 脚本：直接输出单模型 `Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM`
- 可选：加 `--json-only` 输出结构化结果

## 注意事项

- `TRT-Edge-LLM` 的 engine 和 runtime 必须和当前 TensorRT 版本匹配
- 当前仓库的 VLM benchmark 可以在本地 TRT VLM assets 可用时走 3-way；若缺 assets，应明确退回 2-way，而不是伪造 3-way
- 多模型 benchmark 时，优先按“模型缺失单独 skip”处理，不要因为 3B/7B 尚未下载就阻塞 0.5B/1.5B/3B(VLM) 已有结果
- 如果用户只看“最终 steady-state 性能”，优先强调 median / trimmed mean，不要只给 mean
- 如果用户问“为什么差这么大”，先区分是 graph capture、初始化抖动，还是 steady-state kernel 性能差
