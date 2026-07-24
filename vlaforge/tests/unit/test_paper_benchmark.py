from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "benchmark_paper_artifact.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_paper_artifact",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_summary_is_deterministic_and_retains_samples() -> None:
    benchmark = _module()
    values = [float(index) for index in range(30)]
    first = benchmark._summarize(values, 200, seed=7)
    second = benchmark._summarize(values, 200, seed=7)
    assert first == second
    assert first["samples"] == values
    assert first["p50"]["estimate"] == 14.5
    assert first["p95"]["ci95"][0] <= first["p95"]["estimate"]
    assert first["p95"]["ci95"][1] >= first["p95"]["estimate"]


def test_declared_backend_tensor_bytes_are_separate_and_recursive() -> None:
    benchmark = _module()
    specification = {
        "inputs": [
            {"shape": [1, 3, 4, 4], "dtype": "f32"},
            {"shape": [1, 8], "dtype": "i64"},
        ],
        "outputs": ({"shape": [2, 5], "dtype": "bf16"},),
    }
    assert benchmark._declared_tensor_bytes(specification) == (
        1 * 3 * 4 * 4 * 4 + 1 * 8 * 8 + 2 * 5 * 2
    )


def test_reports_disclose_reused_measurements(tmp_path: Path) -> None:
    benchmark = _module()
    cell = {
        "model": "openvla",
        "workload": "repeat",
        "mode": "off",
        "measurement_reused_from": "nominal/off",
        "post_warm_samples": 30,
        "latency_us": {
            metric: {"estimate": 1.0, "ci95": [0.9, 1.1]}
            for metric in ("p50", "p95", "p99")
        },
        "cache_hits": 0,
        "cache_misses": 30,
        "memory": {
            "process_rss_peak_bytes": 1024,
            "process_vram_peak_bytes": 0,
            "compiler_arena_bytes": 64,
            "backend_declared_tensor_bytes": 512,
        },
        "exact_vs_off": True,
    }
    csv_path = tmp_path / "report.csv"
    benchmark._write_csv(csv_path, [cell])
    row = next(csv.DictReader(io.StringIO(csv_path.read_text())))
    assert row["measurement_reused_from"] == "nominal/off"

    markdown = benchmark._markdown(
        {
            "revision": "deadbeef",
            "gate_passed": True,
            "evidence_exact": True,
            "measurements": [cell],
        }
    )
    assert "| openvla | repeat | off | reused from nominal/off |" in markdown
