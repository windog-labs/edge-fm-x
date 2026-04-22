#!/usr/bin/env python3
"""
Edge-FM(cuda graph) vs TRT-Edge-LLM 算子耗时对比

使用 nsys 采集 GPU kernel 耗时，对比两个框架的算子分布，找出 Edge-FM 中较慢的算子。
默认 workload 对齐当前主 benchmark 口径：
  - Qwen2.5-1.5B BF16
  - batch=1
  - prefill=2048
  - decode=64
  - Edge-FM 仅分析 cuda-graph 路径

用法（在项目根目录）:
  # 1. 采集
  bash scripts/profile/profile_operator_comparison.sh

  # 2. 或手动采集
  nsys profile -o ncu_reports/edgefm_profile --stats=true \
      --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      python scripts/profile/profile_operator_comparison.py edgefm
  nsys profile -o ncu_reports/trt_profile --stats=true \
      --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
      python scripts/profile/profile_operator_comparison.py trt

  # 3. 提取 kernel 汇总
  nsys stats --report cuda_gpu_kern_sum --format csv --output ncu_reports/edgefm_kernels ncu_reports/edgefm_profile.nsys-rep
  nsys stats --report cuda_gpu_kern_sum --format csv --output ncu_reports/trt_kernels ncu_reports/trt_profile.nsys-rep
"""

import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
project_root = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
for build_python in [
    project_root / "build" / "python",
    project_root / "build" / "install" / "python",
]:
    build_python_str = str(build_python)
    if build_python.is_dir() and build_python_str not in sys.path:
        sys.path.insert(0, build_python_str)

from operator_table.utils import resolve_target_hw_profile

# 与当前主 benchmark 默认口径保持一致
DEVICE_ID = int(os.environ.get("EDGE_FM_DEVICE_ID", "1"))
PROFILE_WARMUP = 2
PROFILE_RUNS = 3  # 少量 run 便于 nsys 采集
PROFILE_PREFILL_LEN = int(os.environ.get("EDGE_FM_PROFILE_PREFILL_LEN", "2048"))
PROFILE_DECODE_LEN = int(os.environ.get("EDGE_FM_PROFILE_DECODE_LEN", "64"))
CUDA_HW_PROFILE = resolve_target_hw_profile()


def _default_trt_build_dir() -> Path:
    edge_fm_build_dir = os.environ.get("EDGE_FM_BUILD_DIR")
    if edge_fm_build_dir:
        return Path(edge_fm_build_dir) / "trt-edgellm"
    return project_root / "build-trt-edgellm"


def _load_dump():
    dump_dir = project_root / "tests" / "data" / "decode_dump"
    manifest_path = dump_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dump not found at {dump_dir}. Run: pytest -s tests/engine/test_qwen2_generate.py -k benchmark_trt_edgellm (once) to generate."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    model_path = manifest["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path in manifest not found: {model_path}")
    token_ids = __import__("numpy").load(dump_dir / "token_ids.npy")
    return {
        "manifest": manifest,
        "model_path": model_path,
        "token_ids": token_ids.flatten().tolist(),
        "prompt": manifest.get("prompt", "Hello, how are you today?"),
    }


def _build_prefill_token_ids(base_token_ids: list[int], prefill_len: int) -> list[int]:
    if prefill_len <= 0:
        raise ValueError(f"prefill_len must be > 0, got {prefill_len}")
    if not base_token_ids:
        raise ValueError("token_ids is empty")
    if len(base_token_ids) >= prefill_len:
        return base_token_ids[:prefill_len]
    mul = (prefill_len + len(base_token_ids) - 1) // len(base_token_ids)
    return (base_token_ids * mul)[:prefill_len]


def _create_engine_config(model_path: str, prefill_len: int, num_steps: int) -> str:
    import json
    config_path = Path(model_path) / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    num_heads = text_config.get("num_attention_heads", 8)
    num_kv_heads = text_config.get("num_key_value_heads", num_heads)
    attention_type = "gqa" if num_kv_heads < num_heads else "mha"
    torch_dtype = str(text_config.get("torch_dtype", "float16")).lower()
    kvcache_dtype = "bf16" if ("bfloat" in torch_dtype or "bf16" in torch_dtype) else "fp16"
    max_tokens = prefill_len + num_steps - 1
    engine_config_dir = tempfile.mkdtemp()
    engine_config_path = Path(engine_config_dir) / "engine_config.json"
    with open(engine_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": "Qwen2.5",
            "runtime": {
                "device": "cuda",
                "device_id": DEVICE_ID,
                "hw_profile": CUDA_HW_PROFILE,
                "use_cuda_graph": True,
            },
            "prefill_model_path": str(Path(model_path).resolve()),
            "kvcache": {
                "dtype": kvcache_dtype,
                "attention_type": attention_type,
                "requests": [{"request_id": 0, "prefix_token_ids": [], "max_tokens": max_tokens}],
            },
            "sampling": {"temperature": 0.0, "seed": 42},
        }, f, indent=2)
    return str(engine_config_path)


def run_edgefm():
    """运行 Edge-FM(cuda graph) 推理（供 nsys 采集）。"""
    import time
    import torch
    import edge_fm

    data = _load_dump()
    token_ids_list = _build_prefill_token_ids(data["token_ids"], PROFILE_PREFILL_LEN)
    num_steps = PROFILE_DECODE_LEN
    prefill_len = len(token_ids_list)
    engine_config_path = _create_engine_config(data["model_path"], prefill_len, num_steps)
    engine = edge_fm.EdgeFM(engine_config_path)

    def make_request():
        req = edge_fm.Request(0, token_ids_list)
        req.set_ignore_stop_tokens(True)
        return req

    print(
        f"[profile] Edge-FM cuda-graph config: device=cuda:{DEVICE_ID}, "
        f"prefill={prefill_len}, decode={num_steps}"
    )
    for _ in range(PROFILE_WARMUP):
        engine.generate(make_request())
    torch.cuda.synchronize()

    times = []
    for _ in range(PROFILE_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.generate(make_request())
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(
        f"[profile] Edge-FM(cuda graph) inference completed. "
        f"Timed {PROFILE_RUNS} runs: avg={avg_ms:.1f}ms, times={[f'{t:.0f}' for t in times]}ms"
    )


def run_trt():
    """运行 TRT-Edge-LLM 推理（供 nsys 采集）。"""
    try:
        import edge_fm_trt
    except ImportError:
        print("ERROR: edge_fm_trt not found. Build with BUILD_TRT_EDGELLM_PYBIND=ON")
        sys.exit(1)

    data = _load_dump()
    engine_dir = os.environ.get(
        "TRT_EDGELLM_ENGINE_DIR",
        str(project_root / "tests" / "data" / "trt_edgellm_workspace" / "qwen2.5-1.5b" / "engines"),
    )
    if not Path(engine_dir).exists() or not (Path(engine_dir) / "llm.engine").exists():
        raise FileNotFoundError(
            f"TRT engine not found at {engine_dir}. Run: bash tests/scripts/setup_trt_edgellm_benchmark.sh"
        )

    plugin_path = Path(os.environ.get("EDGE_FM_TRT_BUILD_DIR", str(_default_trt_build_dir()))) / "libNvInfer_edgellm_plugin.so"
    if plugin_path.exists():
        os.environ["EDGELLM_PLUGIN_PATH"] = str(plugin_path)

    runtime = edge_fm_trt.TrtEdgeLlmRuntime(engine_dir, "", DEVICE_ID)
    token_ids_list = _build_prefill_token_ids(data["token_ids"], PROFILE_PREFILL_LEN)
    num_steps = PROFILE_DECODE_LEN

    import time
    print(
        f"[profile] TRT-Edge-LLM config: device=cuda:{DEVICE_ID}, "
        f"prefill={len(token_ids_list)}, decode={num_steps}"
    )
    for _ in range(PROFILE_WARMUP):
        runtime.generate_from_token_ids(
            token_ids_list, num_steps, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=True
        )

    times = []
    for _ in range(PROFILE_RUNS):
        t0 = time.perf_counter()
        runtime.generate_from_token_ids(
            token_ids_list, num_steps, temperature=0.0, top_p=1.0, top_k=1, ignore_stop_tokens=True
        )
        times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(
        f"[profile] TRT-Edge-LLM inference completed. "
        f"Timed {PROFILE_RUNS} runs: avg={avg_ms:.1f}ms, times={[f'{t:.0f}' for t in times]}ms"
    )


def _normalize_kernel_name(name: str) -> str:
    """将 kernel 名简化为可对比的算子类别。"""
    n = name.strip()
    # TensorRT 自定义算子 (__myl_*)
    if n.startswith("__myl_"):
        return "TensorRT/Plugin"
    # GEMM / Linear (CUTLASS, cuBLAS GEMM)
    if "gemm" in n.lower() or "cutlass" in n.lower() or "volta" in n.lower():
        return "GEMM/Linear"
    # cuBLAS GEMV (小 batch decode 时大量使用)
    if "gemv" in n.lower() or "cublas" in n.lower() or "gemvx" in n.lower():
        return "cuBLAS/GEMV"
    # Attention
    if "flash" in n.lower() or "attention" in n.lower() or "fused_attn" in n.lower() or "fmha" in n.lower() or "kernel_mha" in n.lower():
        return "Attention"
    if "softmax" in n.lower() and "attn" not in n.lower():
        return "Softmax"
    # Embedding
    if "embed" in n.lower() or "gather" in n.lower():
        return "Embedding/Gather"
    # LayerNorm / RMSNorm
    if "norm" in n.lower() or "layernorm" in n.lower() or "rmsnorm" in n.lower():
        return "Norm"
    # Activation
    if "silu" in n.lower() or "swish" in n.lower() or "gelu" in n.lower() or "relu" in n.lower():
        return "Activation"
    # Elementwise
    if "elementwise" in n.lower() or "element_wise" in n.lower():
        return "Elementwise"
    # TensorRT 内置 kernel (trt_ampere_*, sm80_xmma_*)
    if "trt_" in n.lower() or "sm80_xmma" in n.lower() or "sm50_xmma" in n.lower():
        return "TensorRT/GEMM"
    # 保留部分原名便于识别
    for prefix in ["void ", "cuda::", "__global__"]:
        if prefix in n:
            n = n.split(prefix)[-1]
    if "(" in n:
        n = n.split("(")[0]
    return n.strip() or name[:60]


def analyze():
    """解析 nsys gpukernsum 输出，生成 Edge-FM vs TRT 对比报告。"""
    report_dir = project_root / "ncu_reports"
    def find_kernel_csv(prefix: str) -> Path | None:
        candidates = [
            report_dir / f"{prefix}_cuda_gpu_kern_sum.csv",
            report_dir / f"{prefix}_gpukernsum.csv",
            report_dir / f"{prefix}.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    efm_csv = find_kernel_csv("edgefm_kernels")
    trt_csv = find_kernel_csv("trt_kernels")

    if efm_csv is None or trt_csv is None:
        print("Run profile first: bash scripts/profile/profile_operator_comparison.sh")
        return

    import csv
    def load_kernels(path: Path) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        return rows

    efm_rows = load_kernels(efm_csv)
    trt_rows = load_kernels(trt_csv)

    # 解析列名（nsys 可能用 "Total Time(ns)" 等）
    def parse_time(val):
        if val is None or val == "":
            return 0.0
        s = str(val).replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return 0.0

    def get_total_time(rows):
        total = 0.0
        for r in rows:
            for k, v in r.items():
                if "total" in k.lower() and "time" in k.lower() and "ns" in k.lower():
                    total += parse_time(v)
                    break
        return total

    def get_name_col(rows):
        for r in rows:
            for k in r:
                if "name" in k.lower():
                    return k
        return "Name"

    efm_name_col = get_name_col(efm_rows)
    trt_name_col = get_name_col(trt_rows)
    efm_total_ns = get_total_time(efm_rows)
    trt_total_ns = get_total_time(trt_rows)

    # 按算子类别聚合
    def get_row_time(r):
        for k, v in r.items():
            if "total" in k.lower() and "time" in k.lower() and "ns" in k.lower():
                return parse_time(v)
        return 0.0

    def aggregate(rows, name_col):
        agg = {}
        for r in rows:
            name = r.get(name_col, r.get("Name", ""))
            cat = _normalize_kernel_name(name)
            t = get_row_time(r)
            agg[cat] = agg.get(cat, 0) + t
        return agg

    efm_agg = aggregate(efm_rows, efm_name_col)
    trt_agg = aggregate(trt_rows, trt_name_col)

    # 输出报告
    out = report_dir / "operator_comparison.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Edge-FM vs TRT-Edge-LLM 算子耗时对比\n\n")
        f.write("## ⚠️ 重要说明\n\n")
        f.write(
            "**nsys 对 CUDA Graph replay 的 kernel 统计可能不完整，"
            "因此图模式下的 raw kernel sum 更适合看热点排序，不适合直接当端到端时延。**\n\n"
        )
        f.write(
            f"本次 profile 默认 workload: prefill={PROFILE_PREFILL_LEN}, decode={PROFILE_DECODE_LEN}, "
            f"device=cuda:{DEVICE_ID}。\n\n"
        )
        efm_ms = efm_total_ns / 1e6
        trt_ms = trt_total_ns / 1e6
        f.write("## 总 GPU Kernel 时间（raw nsys，TRT 可能低估）\n\n")
        f.write(f"| 框架 | 总时间 (ms) | 说明 |\n")
        f.write(f"|------|-------------|------|\n")
        f.write(f"| Edge-FM(cuda graph) | {efm_ms:.2f} | raw nsys kernel sum |\n")
        f.write(f"| TRT-Edge-LLM | {trt_ms:.2f} | CUDA Graph 导致可能漏记 |\n")
        ratio = (efm_ms / trt_ms) if trt_ms > 0 else float('inf')
        f.write(f"| raw 比值 | {ratio:.1f}x | 仅供热点排序参考，不等同于端到端时延 |\n\n")

        f.write("## 按算子类别耗时分布\n\n")
        f.write("（TRT 列因 CUDA Graph 可能不完整，**Edge-FM 占比**更有参考价值）\n\n")
        all_cats = sorted(set(efm_agg.keys()) | set(trt_agg.keys()))
        f.write("| 算子类别 | Edge-FM (ms) | Edge-FM % | TRT (ms) |\n")
        f.write("|----------|--------------|-----------|----------|\n")
        for cat in all_cats:
            e = efm_agg.get(cat, 0) / 1e6
            t = trt_agg.get(cat, 0) / 1e6
            pct = 100 * e / efm_ms if efm_ms > 0 else 0
            f.write(f"| {cat} | {e:.2f} | {pct:.1f}% | {t:.2f} |\n")

        f.write("\n## Edge-FM Top 10 最耗时 Kernel\n\n")
        sorted_efm = sorted(efm_rows, key=get_row_time, reverse=True)
        f.write("| Kernel | Total Time (ms) | % |\n")
        f.write("|--------|----------------|---|\n")
        for r in sorted_efm[:10]:
            t = get_row_time(r)
            pct = 100 * t / efm_total_ns if efm_total_ns > 0 else 0
            nm = r.get(efm_name_col, r.get("Name", ""))[:70]
            f.write(f"| {nm} | {t/1e6:.2f} | {pct:.1f}% |\n")

        f.write("\n## TRT-Edge-LLM Top 10 最耗时 Kernel\n\n")
        sorted_trt = sorted(trt_rows, key=get_row_time, reverse=True)
        f.write("| Kernel | Total Time (ms) | % |\n")
        f.write("|--------|----------------|---|\n")
        for r in sorted_trt[:10]:
            t = get_row_time(r)
            pct = 100 * t / trt_total_ns if trt_total_ns > 0 else 0
            nm = r.get(trt_name_col, r.get("Name", ""))[:70]
            f.write(f"| {nm} | {t/1e6:.2f} | {pct:.1f}% |\n")

        f.write("\n## 结论与优化建议（基于 Edge-FM 算子占比）\n\n")
        f.write("### Edge-FM 主要耗时算子（按占比）\n\n")
        efm_sorted = sorted(efm_agg.items(), key=lambda x: -x[1])
        for cat, t_ns in efm_sorted[:5]:
            pct = 100 * t_ns / efm_total_ns if efm_total_ns > 0 else 0
            f.write(f"- **{cat}**: {pct:.1f}%\n")
        f.write("\n### 优化方向\n\n")
        f.write("1. **cuBLAS/GEMV** 占比最高：batch=1 decode 时大量使用 GEMV。建议尝试融合 decode GEMM 或 CUTLASS batch=1 优化。\n\n")
        f.write("2. **GEMM/Linear**：CUTLASS 小 tile 较多。可评估统一为大 tile 的收益。\n\n")
        f.write("3. **Attention**：FlashInfer decode。可对比 fmha 等实现。\n\n")
        f.write("### 下一步\n\n")
        f.write(
            "- `ncu --set full -o ncu_reports/edgefm_gemv --kernel-name gemv2T "
            "python scripts/profile/profile_operator_comparison.py edgefm` 深入分析 GEMV\n"
        )

    print(f"Report written to {out}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile/profile_operator_comparison.py <edgefm|trt|analyze>")
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "edgefm":
        run_edgefm()
    elif mode == "trt":
        run_trt()
    elif mode == "analyze":
        analyze()
    else:
        print("Usage: python scripts/profile/profile_operator_comparison.py <edgefm|trt|analyze>")
        sys.exit(1)


if __name__ == "__main__":
    main()
