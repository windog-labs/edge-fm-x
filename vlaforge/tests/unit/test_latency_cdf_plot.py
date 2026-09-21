"""CPU-only validation for exact CDF-figure inputs."""

import importlib.util
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "tools/plot_latency_cdf.py"
_SPEC = importlib.util.spec_from_file_location("latency_cdf_plot", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_load_cdf_keeps_duplicate_samples_and_exact_ranks(tmp_path):
    path = tmp_path / "cdf.csv"
    path.write_text(
        "latency_ns,cdf\n"
        "100,0.25\n"
        "100,0.5\n"
        "200,0.75\n"
        "300,1.0\n"
    )
    latencies, cdf = tool.load_cdf(path)
    assert latencies == [100, 100, 200, 300]
    assert cdf == [0.25, 0.5, 0.75, 1.0]


def test_load_cdf_rejects_removed_or_reordered_samples(tmp_path):
    path = tmp_path / "cdf.csv"
    path.write_text("latency_ns,cdf\n300,1.0\n100,0.5\n")
    with pytest.raises(ValueError, match="sorted"):
        tool.load_cdf(path)
