# Qwen2.5 LLM Prefill Attention Retune (2026-04-08 Device1)

## Config

- Device: `cuda:1`
- GPU: `NVIDIA A800-SXM4-80GB`
- Scope: `Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`
- Prefill list: `512,1024`
- Decode list: `32,64`
- Warmup runs: `3`
- Timed runs: `5`
- Comparison: `Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM`
- Build:
  - `CUDA_HOME=/usr/local/cuda-12.6 PYTHON_EXECUTABLE=/xs-train-nas/zzm/conda/e2e_zk/bin/python bash scripts/build_cuda_fast.sh`
- Benchmark entry:
  - `python .codex/skills/edge-fm-benchmark-report/scripts/report_qwen_benchmark_suite.py --kind llm --llm-models 0.5b,1.5b --prefill-list 512,1024 --decode-list 32,64 --json-only`

## Artifacts

- Raw stdout with TRT logs mixed in:
  - `.tmp_codex/benchmarks/qwen_0p5b_1p5b_prefill_retune_20260408T143327Z.json`
- Clean JSON payload:
  - `.tmp_codex/benchmarks/qwen_0p5b_1p5b_prefill_retune_20260408T143327Z.clean.json`
- GPU monitor:
  - `.tmp_codex/benchmarks/gpu1_monitor_qwen0p5b_1p5b_prefill_retune_20260408T143327Z.log`

## GPU Monitor Check

- `device:1` 在整轮期间只观察到 benchmark 自身 PID：
  - `1919868`
- 没有外部 compute PID 插入 `device:1`
- 本轮结果有效

## Tuning Applied

- `Qwen2.5-0.5B-Instruct`
  - `attention prefill shape_sig=num_qo_heads=14|num_kv_heads=2|head_dim=64`
  - `prefill_cta_tile_q=128`
- `Qwen2.5-1.5B-Instruct`
  - `attention prefill shape_sig=num_qo_heads=12|num_kv_heads=2|head_dim=128`
  - `prefill_cta_tile_q=128`
- `Qwen2.5-3B-Instruct`
  - 保持 `prefill_cta_tile_q=64`

## Summary

- `0.5B`：
  - 四点平均 total gap：`+1.47%`
  - 四点平均 prefill gap：`-1.35%`
  - 四点平均 decode gap：`+2.00%`
  - 相比此前相同四点子集：
    - total gap：`+3.06% -> +1.47%`
    - prefill gap：`+38.95% -> -1.35%`
- `1.5B`：
  - 四点平均 total gap：`+0.06%`
  - 四点平均 prefill gap：`+6.43%`
  - 四点平均 decode gap：`-0.62%`
  - 相比此前相同四点子集：
    - total gap：`+3.45% -> +0.06%`
    - prefill gap：`+34.31% -> +6.43%`

## Qwen2.5-0.5B-Instruct

| Prefill | Decode | Edge vs TRT gap | Edge prefill gap | Edge decode gap |
| --- | ---: | ---: | ---: | ---: |
| 512 | 32 | `+2.00%` | `-21.13%` | `+5.75%` |
| 512 | 64 | `+3.74%` | `+2.58%` | `+3.82%` |
| 1024 | 32 | `+1.07%` | `+9.90%` | `-0.32%` |
| 1024 | 64 | `-0.92%` | `+3.23%` | `-1.25%` |

## Qwen2.5-1.5B-Instruct

| Prefill | Decode | Edge vs TRT gap | Edge prefill gap | Edge decode gap |
| --- | ---: | ---: | ---: | ---: |
| 512 | 32 | `+1.69%` | `+8.89%` | `+0.93%` |
| 512 | 64 | `+1.07%` | `+4.10%` | `+0.90%` |
| 1024 | 32 | `-0.75%` | `+6.25%` | `-1.91%` |
| 1024 | 64 | `-1.76%` | `+6.48%` | `-2.41%` |

## Interpretation

- 这轮最明确的收益来自 `prefill attention`，不是 decode。
- `0.5B` 的 prefill 已经从此前明显落后 TRT，压到四点均值略快于 TRT；总差距也收敛到 `+1.47%`。
- `1.5B` 的 prefill gap 也被明显压缩，四点均值 total 基本打平 TRT。
- `0.5B` 当前剩余问题不再主要是 prefill，而是 short-context decode / fixed-cost。
- 这进一步支持当前主线判断：
  - `flashinfer prefill` 需要按模型尺寸单独调参
  - `stage + shape_sig + model-size` 的 tuning 表策略是正确方向
