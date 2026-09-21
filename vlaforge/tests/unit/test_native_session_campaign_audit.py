"""CPU-only checks for native campaign audit statistics."""

import importlib.util
from pathlib import Path


_PATH = Path(__file__).resolve().parents[2] / "tools/audit_native_session_campaign.py"
_SPEC = importlib.util.spec_from_file_location("native_campaign_audit", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_summary_uses_nearest_rank_and_population_std():
    summary = tool.summarize([10, 20, 30, 40])
    assert summary["count"] == 4
    assert summary["mean_ns"] == 25
    assert summary["p50_ns"] == 20
    assert summary["p90_ns"] == 40
    assert summary["p99_ns"] == 40
    assert summary["std_ns"] == 11.180339887498949


def test_summary_does_not_remove_duplicate_outliers():
    summary = tool.summarize([1, 1, 100])
    assert summary["count"] == 3
    assert summary["max_ns"] == 100
