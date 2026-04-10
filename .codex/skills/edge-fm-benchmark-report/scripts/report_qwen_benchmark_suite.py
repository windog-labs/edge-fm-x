#!/usr/bin/env python3
import argparse
import contextlib
import importlib.util
import io
import json
import os
import statistics as stats
import sys
from pathlib import Path

DEFAULT_TRT_PACKAGE = "/xs-train-nas/zzm/packages/TensorRT-10.16.0.72"


def load_test_module(repo_root: Path, device_id: int):
    os.environ["EDGE_FM_DEVICE_ID"] = str(device_id)
    for python_dir in [
        repo_root / "build" / "python",
        repo_root / "build" / "install" / "python",
    ]:
        if python_dir.exists():
            sys.path.insert(0, str(python_dir))
    module_path = repo_root / "tests" / "engine" / "test_qwen2_generate.py"
    spec = importlib.util.spec_from_file_location("edgefm_bench_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def summary(result: dict) -> dict:
    existing = result.get("latency_summary")
    if existing:
        return existing
    xs = list(result["times_ms"])
    trimmed = sorted(xs)[:-1] if len(xs) > 1 else xs
    stdev = stats.stdev(xs) if len(xs) > 1 else 0.0
    mean = stats.mean(xs)
    return {
        "mean_ms": mean,
        "median_ms": stats.median(xs),
        "trimmed_mean_drop_max_ms": stats.mean(trimmed),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "stdev_ms": stdev,
        "cv_pct": (stdev / mean * 100.0) if mean else 0.0,
    }


def compare(baseline_result, target_result):
    baseline_mean = baseline_result["avg_ms"]
    target_mean = target_result["avg_ms"]
    baseline_sum = summary(baseline_result)
    target_sum = summary(target_result)
    return {
        "mean_latency_speedup_target_vs_baseline": baseline_mean / target_mean,
        "mean_latency_reduction_pct_target_vs_baseline": (baseline_mean - target_mean) / baseline_mean * 100.0,
        "mean_total_throughput_gain_pct_target_vs_baseline": (target_result["tokens_per_sec"] - baseline_result["tokens_per_sec"]) / baseline_result["tokens_per_sec"] * 100.0,
        "mean_decode_throughput_gain_pct_target_vs_baseline": (target_result["decode_tokens_per_sec"] - baseline_result["decode_tokens_per_sec"]) / baseline_result["decode_tokens_per_sec"] * 100.0,
        "median_latency_speedup_target_vs_baseline": baseline_sum["median_ms"] / target_sum["median_ms"],
        "trimmed_latency_speedup_target_vs_baseline": baseline_sum["trimmed_mean_drop_max_ms"] / target_sum["trimmed_mean_drop_max_ms"],
        "baseline_latency_summary": baseline_sum,
        "target_latency_summary": target_sum,
    }


def compare_stage(edge_result, trt_result):
    edge_stage = edge_result.get("stage_avg_ms", {})
    trt_stage = trt_result.get("stage_avg_ms", {})
    if not edge_stage or not trt_stage:
        return {}

    result = {}
    for key in ["prefill_ms", "decode_ms", "total_stage_ms", "decode_step_avg_ms"]:
        edge_val = float(edge_stage.get(key, 0.0))
        trt_val = float(trt_stage.get(key, 0.0))
        result[key] = {
            "edgefm_cuda_graph_ms": edge_val,
            "trt_edgellm_ms": trt_val,
            "gap_ms": edge_val - trt_val,
            "gap_pct_vs_trt": ((edge_val - trt_val) / trt_val * 100.0) if trt_val > 0 else 0.0,
        }
    return result


def build_case_report(raw_case: dict) -> dict:
    report = {
        "config": raw_case["config"],
        "transformers": raw_case["transformers"],
        "edgefm_cuda_graph": raw_case["edgefm_cuda_graph"],
        "edgefm_cuda_graph_vs_transformers": compare(raw_case["transformers"], raw_case["edgefm_cuda_graph"]),
    }
    trt_result = raw_case.get("trt_edgellm")
    if trt_result is not None:
        report["trt_edgellm"] = trt_result
        report["trt_vs_transformers"] = compare(raw_case["transformers"], trt_result)
        report["trt_vs_edgefm_cuda_graph"] = compare(raw_case["edgefm_cuda_graph"], trt_result)
        report["trt_vs_edgefm_cuda_graph_stage"] = compare_stage(raw_case["edgefm_cuda_graph"], trt_result)
    return report


def print_case_report(report: dict):
    cfg = report["config"]
    tf = report["transformers"]
    edge = report["edgefm_cuda_graph"]
    title = (
        f"{cfg['model_label']} [{cfg['kind']}] prefill={cfg['prefill_tokens']} decode={cfg['decode_tokens']}"
    )
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"  Transformers:       mean={tf['avg_ms']:.3f} ms  tok/s={tf['tokens_per_sec']:.3f}  decode_tok/s={tf['decode_tokens_per_sec']:.3f}")
    print(f"  EdgeFM(cuda-graph): mean={edge['avg_ms']:.3f} ms  tok/s={edge['tokens_per_sec']:.3f}  decode_tok/s={edge['decode_tokens_per_sec']:.3f}")
    print(f"  Transformers runs:  {tf['times_ms']}")
    print(f"  EdgeFM runs:        {edge['times_ms']}")
    print(f"  EdgeFM vs TF:       mean={report['edgefm_cuda_graph_vs_transformers']['mean_latency_speedup_target_vs_baseline']:.3f}x  "
          f"median={report['edgefm_cuda_graph_vs_transformers']['median_latency_speedup_target_vs_baseline']:.3f}x  "
          f"trimmed={report['edgefm_cuda_graph_vs_transformers']['trimmed_latency_speedup_target_vs_baseline']:.3f}x")
    if "trt_edgellm" in report:
        trt = report["trt_edgellm"]
        cmp = report["trt_vs_edgefm_cuda_graph"]
        print(f"  TRT-Edge-LLM:       mean={trt['avg_ms']:.3f} ms  tok/s={trt['tokens_per_sec']:.3f}  decode_tok/s={trt['decode_tokens_per_sec']:.3f}")
        print(f"  TRT runs:           {trt['times_ms']}")
        print(f"  TRT vs EdgeFM:      mean={cmp['mean_latency_speedup_target_vs_baseline']:.3f}x  "
              f"median={cmp['median_latency_speedup_target_vs_baseline']:.3f}x  "
              f"trimmed={cmp['trimmed_latency_speedup_target_vs_baseline']:.3f}x")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run multi-model Qwen benchmark suite for LLM/VLM.")
    parser.add_argument("--repo-root", default="/xs-train-nas/zzm/repos/edge-fm-x")
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("EDGE_FM_DEVICE_ID", "0")))
    parser.add_argument("--kind", choices=["llm", "vlm", "all"], default="all")
    parser.add_argument("--llm-models", default="")
    parser.add_argument("--vlm-models", default="")
    parser.add_argument("--prefill-list", default="")
    parser.add_argument("--decode-list", default="")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    os.environ.setdefault("TRT_PACKAGE_DIR", DEFAULT_TRT_PACKAGE)
    if args.llm_models:
        os.environ["EDGE_FM_BENCH_LLM_MODELS"] = args.llm_models
    if args.vlm_models:
        os.environ["EDGE_FM_BENCH_VLM_MODELS"] = args.vlm_models
    if args.prefill_list:
        os.environ["EDGE_FM_BENCH_PREFILL_LIST"] = args.prefill_list
    if args.decode_list:
        os.environ["EDGE_FM_BENCH_DECODE_LIST"] = args.decode_list

    t = load_test_module(repo_root, args.device_id)

    def quiet_context():
        return contextlib.redirect_stdout(io.StringIO()) if args.json_only else contextlib.nullcontext()

    payload = {
        "device_id": args.device_id,
        "llm_3way_cases": [],
        "llm_2way_cases": [],
        "vlm_3way_cases": [],
        "vlm_2way_cases": [],
        "missing_models": {},
    }

    if args.kind in ("llm", "all"):
        llm_models, llm_missing = t._resolve_bench_model_specs("llm")
        payload["missing_models"]["llm"] = llm_missing
        for model_spec in llm_models:
            with quiet_context():
                threeway_cases = t._benchmark_llm_model(model_spec, include_trt=True)
            if threeway_cases:
                payload["llm_3way_cases"].extend(build_case_report(case) for case in threeway_cases)
            else:
                with quiet_context():
                    twoway_cases = t._benchmark_llm_model(model_spec, include_trt=False)
                payload["llm_2way_cases"].extend(build_case_report(case) for case in twoway_cases)

    if args.kind in ("vlm", "all"):
        vlm_models, vlm_missing = t._resolve_bench_model_specs("vlm")
        payload["missing_models"]["vlm"] = vlm_missing
        for model_spec in vlm_models:
            with quiet_context():
                raw_cases = t._benchmark_vlm_model(model_spec, include_trt=True)
            if raw_cases and any(case.get("trt_edgellm") is not None for case in raw_cases):
                payload["vlm_3way_cases"].extend(build_case_report(case) for case in raw_cases)
            else:
                payload["vlm_2way_cases"].extend(build_case_report(case) for case in raw_cases)

    if args.json_only:
        print(json.dumps(payload, indent=2))
        return

    if payload["missing_models"].get("llm"):
        print("[missing] llm models:")
        for item in payload["missing_models"]["llm"]:
            print(f"  - {item['label']} ({item['model_size']})")
        print()
    if payload["missing_models"].get("vlm"):
        print("[missing] vlm models:")
        for item in payload["missing_models"]["vlm"]:
            print(f"  - {item['label']} ({item['model_size']})")
        print()

    for section in ["llm_3way_cases", "llm_2way_cases", "vlm_3way_cases", "vlm_2way_cases"]:
        for case in payload[section]:
            print_case_report(case)


if __name__ == "__main__":
    main()
