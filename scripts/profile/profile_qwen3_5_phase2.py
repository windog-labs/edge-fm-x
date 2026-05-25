#!/usr/bin/env python3
"""Run the Qwen3.5 phase-2 paired benchmark and logging workflow.

The runner intentionally orchestrates existing profile entrypoints instead of
embedding model-specific benchmark logic here. It keeps CURRENT.md and
state.json as the fresh source of truth for the CUDA graph / TRT comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "deliverables" / "kernel_opt"
MODEL_CASES = {
    "0p8b": {
        "display": "0.8B",
        "path": PROJECT_ROOT / "examples" / "qwen3.5-0.8b" / "qwen3.5-0.8b",
        "trt_engine_env": "EDGE_FM_QWEN3_5_0P8B_TRT_ENGINE_DIR",
        "trt_plugin_env": "EDGE_FM_QWEN3_5_0P8B_TRT_PLUGIN_PATH",
    },
    "2b": {
        "display": "2B",
        "path": PROJECT_ROOT / "examples" / "qwen3.5-2b" / "qwen3.5-2b",
        "trt_engine_env": "EDGE_FM_QWEN3_5_2B_TRT_ENGINE_DIR",
        "trt_plugin_env": "EDGE_FM_QWEN3_5_2B_TRT_PLUGIN_PATH",
    },
}
ACCEPTANCE_RATIO = 1.01
STRETCH_SPEEDUP = 1.05


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 phase-2 benchmark runner")
    parser.add_argument("--models", default="0p8b,2b", help="Comma separated model cases: 0p8b,2b")
    parser.add_argument("--prefill-lens", default="128,512,1024")
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("EDGE_FM_DEVICE_ID", "0")))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--edgefm-build-dir", default=os.environ.get("EDGE_FM_BUILD_DIR", "build"))
    parser.add_argument("--notify", action="store_true", help="Send progress notifications through codex-notify-wechat")
    parser.add_argument("--skip-trt", action="store_true", help="Only run EdgeFM graph-off/graph-on baselines")
    parser.add_argument("--lm-head-top1", action="store_true", help="Also pass --lm-head-top1 to EdgeFM profiles")
    return parser.parse_args()


def notify(enabled: bool, message: str) -> None:
    if not enabled:
        return
    notifier = Path.home() / ".local" / "bin" / "codex-notify-wechat"
    if notifier.exists():
        subprocess.run([str(notifier), f"【Codex/edge-fm-x】{message}"], check=False)


def ensure_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_PARENT / f"qwen3_5_phase2_{stamp}"
    (output_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    return output_dir


def run_profile(cmd: list[str], log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    return {"returncode": result.returncode, "log": str(log_path)}


def load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def edgefm_cmd(
    *,
    model_path: Path,
    prefill_len: int,
    decode_len: int,
    warmup: int,
    runs: int,
    device_id: int,
    output_json: Path,
    use_cuda_graph: bool,
    lm_head_top1: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/profile/profile_edgefm_generate_case.py",
        "--model-path",
        str(model_path),
        "--model-name",
        "Qwen3.5",
        "--prefill-len",
        str(prefill_len),
        "--decode-len",
        str(decode_len),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--device-id",
        str(device_id),
        "--output-json",
        str(output_json),
        "--json",
    ]
    if use_cuda_graph:
        cmd.append("--use-cuda-graph")
    if lm_head_top1:
        cmd.append("--lm-head-top1")
    return cmd


def trt_cmd(
    *,
    model_path: Path,
    engine_dir: Path,
    plugin_path: Path,
    prefill_len: int,
    decode_len: int,
    warmup: int,
    runs: int,
    device_id: int,
    output_json: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/profile/profile_trt_edgellm_generate_case.py",
        "--model-path",
        str(model_path),
        "--engine-dir",
        str(engine_dir),
        "--plugin-path",
        str(plugin_path),
        "--prefill-len",
        str(prefill_len),
        "--decode-len",
        str(decode_len),
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--device-id",
        str(device_id),
        "--output-json",
        str(output_json),
        "--json",
    ]


def summarize_case(edge_graph: dict[str, Any] | None, trt: dict[str, Any] | None) -> dict[str, Any]:
    if not edge_graph:
        return {"status": "missing_edgefm_graph_on"}
    edge_ms = float(edge_graph.get("avg_ms", 0.0))
    if not trt:
        return {"status": "missing_trt_baseline", "edgefm_graph_on_ms": edge_ms}
    trt_ms = float(trt.get("avg_ms", 0.0))
    if edge_ms <= 0.0 or trt_ms <= 0.0:
        return {"status": "invalid_latency", "edgefm_ms": edge_ms, "trt_ms": trt_ms}
    ratio = edge_ms / trt_ms
    return {
        "status": "pass" if ratio <= ACCEPTANCE_RATIO else "gap",
        "edgefm_graph_on_ms": edge_ms,
        "trt_ms": trt_ms,
        "gap_ratio": ratio,
        "accepted": ratio <= ACCEPTANCE_RATIO,
        "stretch": trt_ms / edge_ms >= STRETCH_SPEEDUP,
    }


def write_current(output_dir: Path, state: dict[str, Any]) -> None:
    lines = [
        "# Qwen3.5 Phase 2 Current State",
        "",
        f"- Updated: `{state['updated_at']}`",
        f"- Workload: prefill `{state['prefill_lens']}`, decode `{state['decode_len']}`",
        f"- Acceptance: EdgeFM graph-on avg_ms <= TRT avg_ms * `{ACCEPTANCE_RATIO}`",
        "",
        "## Results",
        "",
        "| Model | Prefill | EdgeFM graph-on ms | TRT ms | Gap ratio | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in state["matrix"]:
        summary = row["summary"]
        lines.append(
            "| {model} | {prefill} | {edge:.3f} | {trt:.3f} | {ratio:.4f} | {status} |".format(
                model=row["model"],
                prefill=row["prefill_len"],
                edge=float(summary.get("edgefm_graph_on_ms", 0.0)),
                trt=float(summary.get("trt_ms", 0.0)),
                ratio=float(summary.get("gap_ratio", 0.0)),
                status=summary.get("status", "unknown"),
            )
        )
    if state["blockers"]:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in state["blockers"])
    lines.extend(["", "## Freshness Policy", "", "- This file and `state.json` are the current source of truth."])
    (output_dir / "CURRENT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_ledger(output_dir: Path, state: dict[str, Any]) -> None:
    path = output_dir / "iteration-ledger.md"
    if not path.exists():
        path.write_text("# Qwen3.5 Phase 2 Iteration Ledger\n\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"## Baseline {state['updated_at']}\n\n")
        handle.write("- Change: establish EdgeFM graph-off/graph-on and optional TRT paired baseline.\n")
        handle.write(f"- Correctness: expected to be validated separately by `tests/engine/test_qwen3_5_generate.py`.\n")
        handle.write(f"- Accepted cases: {state['accepted_cases']}/{len(state['matrix'])}.\n")
        if state["blockers"]:
            handle.write("- Blockers: " + "; ".join(state["blockers"]) + "\n")
        handle.write("\n")


def main() -> int:
    args = parse_args()
    output_dir = ensure_output_dir(args)
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    prefill_lens = parse_csv_ints(args.prefill_lens)
    env = os.environ.copy()
    env["EDGE_FM_BUILD_DIR"] = args.edgefm_build_dir

    notify(args.notify, f"阶段2: 开始 Qwen3.5 paired benchmark; 输出 {output_dir}")
    matrix: list[dict[str, Any]] = []
    blockers: list[str] = []

    for model_key in selected_models:
        if model_key not in MODEL_CASES:
            blockers.append(f"unknown model case: {model_key}")
            continue
        case = MODEL_CASES[model_key]
        model_path = Path(case["path"])
        if not (model_path / "config.json").exists():
            blockers.append(f"missing Qwen3.5 {case['display']} model path: {model_path}")
            continue

        trt_engine = os.environ.get(case["trt_engine_env"], "")
        trt_plugin = os.environ.get(case["trt_plugin_env"], "")
        trt_available = bool(trt_engine and trt_plugin and Path(trt_engine).exists() and Path(trt_plugin).exists())
        if not trt_available and not args.skip_trt:
            blockers.append(
                f"missing TRT baseline for {case['display']}: set {case['trt_engine_env']} and {case['trt_plugin_env']}"
            )

        for prefill_len in prefill_lens:
            prefix = f"{model_key}_prefill{prefill_len}_decode{args.decode_len}"
            off_json = output_dir / "baseline" / f"edgefm_graph_off_{prefix}.json"
            on_json = output_dir / "baseline" / f"edgefm_graph_on_{prefix}.json"
            trt_json = output_dir / "baseline" / f"trt_{prefix}.json"

            off_run = run_profile(
                edgefm_cmd(
                    model_path=model_path,
                    prefill_len=prefill_len,
                    decode_len=args.decode_len,
                    warmup=args.warmup,
                    runs=args.runs,
                    device_id=args.device_id,
                    output_json=off_json,
                    use_cuda_graph=False,
                    lm_head_top1=args.lm_head_top1,
                ),
                output_dir / "logs" / f"edgefm_graph_off_{prefix}.log",
                env,
            )
            on_run = run_profile(
                edgefm_cmd(
                    model_path=model_path,
                    prefill_len=prefill_len,
                    decode_len=args.decode_len,
                    warmup=args.warmup,
                    runs=args.runs,
                    device_id=args.device_id,
                    output_json=on_json,
                    use_cuda_graph=True,
                    lm_head_top1=args.lm_head_top1,
                ),
                output_dir / "logs" / f"edgefm_graph_on_{prefix}.log",
                env,
            )

            trt_run: dict[str, Any] | None = None
            if trt_available and not args.skip_trt:
                trt_run = run_profile(
                    trt_cmd(
                        model_path=model_path,
                        engine_dir=Path(trt_engine),
                        plugin_path=Path(trt_plugin),
                        prefill_len=prefill_len,
                        decode_len=args.decode_len,
                        warmup=args.warmup,
                        runs=args.runs,
                        device_id=args.device_id,
                        output_json=trt_json,
                    ),
                    output_dir / "logs" / f"trt_{prefix}.log",
                    env,
                )

            edge_graph = load_json_if_present(on_json)
            trt_result = load_json_if_present(trt_json)
            matrix.append(
                {
                    "model": case["display"],
                    "model_key": model_key,
                    "prefill_len": prefill_len,
                    "decode_len": args.decode_len,
                    "edgefm_graph_off": {"run": off_run, "json": str(off_json), "metrics": load_json_if_present(off_json)},
                    "edgefm_graph_on": {"run": on_run, "json": str(on_json), "metrics": edge_graph},
                    "trt": {"run": trt_run, "json": str(trt_json), "metrics": trt_result},
                    "summary": summarize_case(edge_graph, trt_result),
                }
            )
            notify(args.notify, f"阶段2: {case['display']} prefill={prefill_len} decode={args.decode_len} 完成")

    accepted_cases = sum(1 for row in matrix if row["summary"].get("accepted"))
    state = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "prefill_lens": prefill_lens,
        "decode_len": args.decode_len,
        "runs": args.runs,
        "warmup": args.warmup,
        "acceptance_ratio": ACCEPTANCE_RATIO,
        "stretch_speedup": STRETCH_SPEEDUP,
        "accepted_cases": accepted_cases,
        "blockers": blockers,
        "matrix": matrix,
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    write_current(output_dir, state)
    append_ledger(output_dir, state)
    notify(args.notify, f"阶段2: baseline 完成 {accepted_cases}/{len(matrix)} case 达标; 查看 {output_dir / 'CURRENT.md'}")
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
