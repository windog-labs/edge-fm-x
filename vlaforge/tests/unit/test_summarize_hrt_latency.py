"""CPU-only tests for HRT infer latency aggregation."""

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "tools/summarize_hrt_latency.py"
_SPEC = importlib.util.spec_from_file_location("summarize_hrt_latency", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_parse_latency_ns_preserves_decimal_milliseconds(tmp_path):
    path = tmp_path / "latency.log"
    path.write_text(
        "[INFO] noise\n"
        "Infer time: 1.25 ms\n"
        "Infer time: 2 ms\n"
    )
    assert tool.parse_latency_ns(path) == [1_250_000, 2_000_000]


def test_parse_latency_ns_rejects_log_without_samples(tmp_path):
    path = tmp_path / "latency.log"
    path.write_text("No measurements\n")
    with pytest.raises(ValueError, match="no Infer time"):
        tool.parse_latency_ns(path)


def test_summary_uses_nearest_rank_and_population_std():
    summary = tool.summarize([100, 200, 300, 400])
    assert summary == {
        "samples": 4,
        "mean_ns": 250.0,
        "min_ns": 100,
        "p50_ns": 200,
        "p95_ns": 400,
        "p99_ns": 400,
        "max_ns": 400,
        "std_ns": pytest.approx(111.80339887498948),
    }
