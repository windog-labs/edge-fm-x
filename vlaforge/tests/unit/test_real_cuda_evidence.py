from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "aggregate_real_cuda_evidence.py"
    )
    specification = importlib.util.spec_from_file_location(
        "aggregate_real_cuda_evidence",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_speedup_is_baseline_over_optimized() -> None:
    evidence = _module()

    assert evidence._speedup(20.0, 5.0) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="positive"):
        evidence._speedup(0.0, 5.0)
