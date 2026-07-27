from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_real_autovla_artifacts.py"
    )
    specification = importlib.util.spec_from_file_location(
        "audit_real_autovla_artifacts",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_artifact_metrics_distinguish_exact_tokens_and_float_error() -> None:
    audit = _module()
    tokens = torch.tensor([151665, 151666], dtype=torch.int64)
    assert audit._metrics(tokens, tokens.clone())["exact"] is True
    changed = audit._metrics(tokens, tokens + 1)
    assert changed["exact"] is False
    assert changed["maximum_absolute_error"] == 1.0

    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    actual = torch.tensor([1.0, 2.25], dtype=torch.float32)
    metrics = audit._metrics(expected, actual)
    assert metrics["exact"] is False
    assert metrics["maximum_absolute_error"] == 0.25
    assert metrics["normalized_root_mean_square_error"] > 0.0


def test_target_chain_accepts_destination_native_artifacts() -> None:
    audit = _module()
    audit._validate_target_chain(
        runtime_target="sm_80",
        expected_target="sm_80",
        compile_target="sm_80",
    )
    audit._validate_target_chain(
        runtime_target="sm_90",
        expected_target=None,
        compile_target="sm_90",
    )


def test_target_chain_rejects_cross_gpu_artifact_reuse() -> None:
    audit = _module()
    with pytest.raises(RuntimeError, match="must be rebuilt"):
        audit._validate_target_chain(
            runtime_target="sm_90",
            expected_target="sm_90",
            compile_target="sm_86",
        )
    with pytest.raises(RuntimeError, match="legacy"):
        audit._validate_target_chain(
            runtime_target="sm_80",
            expected_target=None,
            compile_target=None,
        )
