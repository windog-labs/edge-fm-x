#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import statistics as stats
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_TRT_PACKAGE = "/xs-train-nas/zzm/packages/TensorRT-10.16.0.72"


def load_test_module(repo_root: Path, device_id: int):
    os.environ["EDGE_FM_DEVICE_ID"] = str(device_id)
    sys.path.insert(0, str(repo_root / "build" / "install" / "python"))
    module_path = repo_root / "tests" / "engine" / "test_qwen2_generate.py"
    spec = importlib.util.spec_from_file_location("edgefm_bench_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_dump_data(t):
    manifest_path = t.DUMP_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    model_path = manifest["model_path"]
    if not Path(model_path).exists():
        fallback = t._find_qwen_model_path()
        if fallback is None:
            raise RuntimeError(f"Model path in dump manifest not found: {model_path}")
        model_path = fallback
        manifest["model_path"] = model_path
    token_ids = np.load(t.DUMP_DIR / "token_ids.npy")
    return manifest, model_path, token_ids.flatten().tolist()


def summary(times_ms):
    xs = list(times_ms)
    trimmed = sorted(xs)[:-1] if len(xs) > 1 else xs
    stdev = stats.stdev(xs) if len(xs) > 1 else 0.0
    return {
        "mean_ms": stats.mean(xs),
        "median_ms": stats.median(xs),
        "trimmed_mean_drop_max_ms": stats.mean(trimmed),
        "min_ms": min(xs),
        "max_ms": max(xs),
        "stdev_ms": stdev,
        "cv_pct": (stdev / stats.mean(xs) * 100.0) if stats.mean(xs) else 0.0,
    }


def compare(baseline_result, target_result):
    baseline_mean = baseline_result["avg_ms"]
    target_mean = target_result["avg_ms"]
    baseline_sum = summary(baseline_result["times_ms"])
    target_sum = summary(target_result["times_ms"])
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


def run_transformers(t, model_path, token_ids_list, num_steps):
    result = t._bench_transformers_llm(
        model_path,
        token_ids_list,
        num_steps,
        warmup=t.BENCH_WARMUP_RUNS,
        runs=t.BENCH_TIMED_RUNS,
    )
    torch.cuda.empty_cache()
    return result


def run_edgefm(t, edge_fm_mod, model_path, token_ids_list, prefill_len, num_steps, use_cuda_graph):
    engine_config_path = t._create_engine_config(
        model_path,
        prefill_len,
        num_steps,
        use_cuda_graph=use_cuda_graph,
        generated_tokens_total=num_steps,
    )
    engine = edge_fm_mod.EdgeFM(engine_config_path)

    def make_request():
        req = edge_fm_mod.Request(0, token_ids_list)
        req.set_ignore_stop_tokens(True)
        return req

    result = t._bench_edgefm(
        engine,
        make_request,
        num_steps,
        prefill_len,
        warmup=t.BENCH_WARMUP_RUNS,
        runs=t.BENCH_TIMED_RUNS,
    )
    del engine
    torch.cuda.empty_cache()
    return result


def run_trt(t, token_ids_list, prefill_len, num_steps, prompt, engine_dir, inference_bin):
    result = t._bench_trt_edgellm(
        engine_dir,
        inference_bin,
        token_ids_list,
        num_steps,
        prefill_len,
        warmup=t.BENCH_WARMUP_RUNS,
        runs=t.BENCH_TIMED_RUNS,
        ignore_stop_tokens=True,
        prompt=prompt,
    )
    torch.cuda.empty_cache()
    return result


def build_report(config, tf_result, edge_cg_result, trt_result, edge_ng_result=None):
    report = {
        "config": config,
        "transformers": tf_result,
        "edgefm_cuda_graph": edge_cg_result,
        "trt_edgellm": trt_result,
        "edgefm_cuda_graph_vs_transformers": compare(tf_result, edge_cg_result),
        "trt_vs_transformers": compare(tf_result, trt_result),
        "trt_vs_edgefm_cuda_graph": compare(edge_cg_result, trt_result),
    }
    if edge_ng_result is not None:
        report["edgefm_no_graph"] = edge_ng_result
        report["edgefm_no_graph_vs_transformers"] = compare(tf_result, edge_ng_result)
        report["trt_vs_edgefm_no_graph"] = compare(edge_ng_result, trt_result)
    return report


def print_text_report(report):
    cfg = report["config"]
    tf = report["transformers"]
    edge = report["edgefm_cuda_graph"]
    trt = report["trt_edgellm"]
    trt_vs_edge = report["trt_vs_edgefm_cuda_graph"]

    print("=== Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM ===")
    print(json.dumps(cfg, indent=2))
    print()
    print("Mean latency (ms):")
    print("  Transformers:        %.3f" % tf["avg_ms"])
    print("  EdgeFM(cuda-graph):  %.3f" % edge["avg_ms"])
    print("  TRT-Edge-LLM:        %.3f" % trt["avg_ms"])
    print()
    print("Mean throughput (tok/s):")
    print("  Transformers:        %.3f" % tf["tokens_per_sec"])
    print("  EdgeFM(cuda-graph):  %.3f" % edge["tokens_per_sec"])
    print("  TRT-Edge-LLM:        %.3f" % trt["tokens_per_sec"])
    print()
    print("Mean decode throughput (tok/s):")
    print("  Transformers:        %.3f" % tf["decode_tokens_per_sec"])
    print("  EdgeFM(cuda-graph):  %.3f" % edge["decode_tokens_per_sec"])
    print("  TRT-Edge-LLM:        %.3f" % trt["decode_tokens_per_sec"])
    print()
    print("TRT vs EdgeFM(cuda-graph):")
    print("  Mean latency speedup:     %.3fx" % trt_vs_edge["mean_latency_speedup_target_vs_baseline"])
    print("  Mean latency reduction:   %.2f%%" % trt_vs_edge["mean_latency_reduction_pct_target_vs_baseline"])
    print("  Mean throughput gain:     %.2f%%" % trt_vs_edge["mean_total_throughput_gain_pct_target_vs_baseline"])
    print("  Mean decode gain:         %.2f%%" % trt_vs_edge["mean_decode_throughput_gain_pct_target_vs_baseline"])
    print("  Median latency speedup:   %.3fx" % trt_vs_edge["median_latency_speedup_target_vs_baseline"])
    print("  Trimmed latency speedup:  %.3fx" % trt_vs_edge["trimmed_latency_speedup_target_vs_baseline"])
    print()
    print("Timed runs (ms):")
    print("  Transformers:        %s" % tf["times_ms"])
    print("  EdgeFM(cuda-graph):  %s" % edge["times_ms"])
    print("  TRT-Edge-LLM:        %s" % trt["times_ms"])
    print()
    print("EdgeFM(cuda-graph) latency summary:")
    print(json.dumps(trt_vs_edge["baseline_latency_summary"], indent=2))
    print("TRT-Edge-LLM latency summary:")
    print(json.dumps(trt_vs_edge["target_latency_summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformers vs EdgeFM(cuda-graph) vs TRT-Edge-LLM using test_qwen2_generate helpers.")
    parser.add_argument("--repo-root", default="/xs-train-nas/zzm/repos/edge-fm-x")
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("EDGE_FM_DEVICE_ID", "1")))
    parser.add_argument("--also-edgefm-no-graph", action="store_true")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    os.environ.setdefault("TRT_PACKAGE_DIR", DEFAULT_TRT_PACKAGE)

    t = load_test_module(repo_root, args.device_id)
    import edge_fm  # noqa: F401
    edge_fm_mod = sys.modules["edge_fm"]

    manifest, model_path, base_token_ids = load_dump_data(t)
    prefill_len = len(base_token_ids)
    num_steps = t.BENCH_NUM_STEPS
    prompt = manifest.get("prompt", t.DEFAULT_PROMPT)
    token_ids_list = t._build_prefill_token_ids(base_token_ids, prefill_len)
    engine_dir = (t.project_root / "tests" / "data" / "trt_edgellm_workspace" / "qwen2.5-1.5b" / "engines").resolve()
    inference_bin = (t.project_root / "third_party" / "TensorRT-Edge-LLM" / "build" / "examples" / "llm" / "llm_inference").resolve()

    tf_result = run_transformers(t, model_path, token_ids_list, num_steps)
    trt_result = run_trt(t, token_ids_list, prefill_len, num_steps, prompt, engine_dir, inference_bin)
    edge_cg_result = run_edgefm(t, edge_fm_mod, model_path, token_ids_list, prefill_len, num_steps, use_cuda_graph=True)
    edge_ng_result = None
    if args.also_edgefm_no_graph:
        edge_ng_result = run_edgefm(t, edge_fm_mod, model_path, token_ids_list, prefill_len, num_steps, use_cuda_graph=False)

    report = build_report(
        {
            "model_path": str(model_path),
            "device_id": args.device_id,
            "prefill_tokens": prefill_len,
            "decode_tokens": num_steps,
            "warmup_runs": t.BENCH_WARMUP_RUNS,
            "timed_runs": t.BENCH_TIMED_RUNS,
            "engine_dir": str(engine_dir),
            "inference_bin": str(inference_bin),
        },
        tf_result,
        edge_cg_result,
        trt_result,
        edge_ng_result=edge_ng_result,
    )

    if args.json_only:
        print(json.dumps(report, indent=2))
        return

    print_text_report(report)


if __name__ == "__main__":
    main()
