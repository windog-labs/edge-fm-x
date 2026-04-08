#!/usr/bin/env python3
"""
Benchmark baseline vs tuned decode activation in isolated subprocesses.

This is intentionally subprocess-based to avoid:
- conda libstdc++ preload issues
- WeightLoader / static buffer cross-talk between baseline and tuned runs
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "examples" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct"
OP_TABLE_PATH = REPO_ROOT / "examples" / "config" / "operator_impl_table.json"
SHAPE_SIG = "batch=1|hidden=8960|dtype=2"
LD_PRELOAD_LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"


def write_operator_table(records: list[dict]) -> Path:
    td = Path(tempfile.mkdtemp(prefix="efm_act_table_"))
    path = td / "operator_impl_table.json"
    path.write_text(json.dumps({"schema": "edgefm_operator_impl_table_v1", "records": records}))
    return path


def make_engine_config(table_path: Path) -> Path:
    td = Path(tempfile.mkdtemp(prefix="efm_act_cfg_"))
    path = td / "engine_config.json"
    path.write_text(json.dumps({
        "model_name": "Qwen2.5",
        "runtime": {"device": "cuda", "device_id": 1, "hw_profile": "cuda_sm80"},
        "prefill_model_path": str(MODEL_PATH),
        "operator_impl_table_path": str(table_path),
    }))
    return path


def run_single(table_path: Path) -> dict[str, float]:
    config_path = make_engine_config(table_path)
    code = f"""
import json
import statistics
import sys
from pathlib import Path

root = Path({str(REPO_ROOT)!r})
sys.path.insert(0, str(root / 'build' / 'install' / 'python'))
import edge_fm
import torch

torch.cuda.set_device(1)
loader = edge_fm.WeightLoader.instance()
loader.clear_stage(edge_fm.ModelStage.Prefill)
loader.clear_stage(edge_fm.ModelStage.Decode)

linear = edge_fm.FusedGateUpLinearLayer('model.layers.0.mlp', {str(config_path)!r}, 1536, 8960, 8960)
activation = edge_fm.ActivationLayer({str(config_path)!r})

x = torch.randn((1, 1536), device='cuda:1', dtype=torch.bfloat16)
packed = torch.empty((1, 17920), device='cuda:1', dtype=torch.bfloat16)
out = torch.empty((1, 8960), device='cuda:1', dtype=torch.bfloat16)

x_efm = edge_fm.Tensor.from_dlpack(x.contiguous().__dlpack__())
packed_efm = edge_fm.Tensor.from_dlpack(packed.contiguous().__dlpack__())
out_efm = edge_fm.Tensor.from_dlpack(out.contiguous().__dlpack__())

def bench(fn, warmup=40, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end))
    return statistics.median(vals)

def run_linear():
    linear.forward_fp16_bf16(x_efm, packed_efm, 0, 'Decode')

def run_act():
    activation.forward_silu_and_mul(packed_efm, out_efm, 0, 'Decode')

def run_both():
    run_linear()
    run_act()

run_both()
torch.cuda.synchronize()
print(json.dumps({{'act_ms': bench(run_act), 'both_ms': bench(run_both)}}))
"""
    env = os.environ.copy()
    env["LD_PRELOAD"] = LD_PRELOAD_LIBSTDCPP
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    records = json.loads(OP_TABLE_PATH.read_text())["records"]
    baseline_records = [
        r for r in records
        if not (
            r.get("model_name") == "qwen2_5"
            and r.get("hw_profile") == "cuda_sm80"
            and r.get("op_kind") == "activation"
            and r.get("layer_role") == "mlp_activation"
            and r.get("op_name") == "silu_and_mul"
            and r.get("stage") == "decode"
            and r.get("shape_sig") == SHAPE_SIG
        )
    ]
    baseline_table = write_operator_table(baseline_records)
    baseline = run_single(baseline_table)
    tuned = run_single(OP_TABLE_PATH)
    print(json.dumps({
        "baseline": baseline,
        "tuned": tuned,
        "act_gain_ms": baseline["act_ms"] - tuned["act_ms"],
        "both_gain_ms": baseline["both_ms"] - tuned["both_ms"],
    }, indent=2))


if __name__ == "__main__":
    main()
