from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_cuda_aoti_region.py"
    )
    specification = importlib.util.spec_from_file_location(
        "audit_cuda_aoti_region",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("capability", "expected_target", "expected_arch"),
    [
        ((8, 0), "sm_80", "8.0"),
        ((8, 6), "sm_86", "8.6"),
        ((9, 0), "sm_90", "9.0"),
    ],
)
def test_runtime_target_follows_destination_capability(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    expected_target: str,
    expected_arch: str,
) -> None:
    audit = _module()
    monkeypatch.setattr(
        audit.torch.cuda,
        "get_device_capability",
        lambda _device: capability,
    )
    assert audit._runtime_target("cuda") == (
        expected_target,
        expected_arch,
    )


def test_runtime_target_preserves_cpu_identity() -> None:
    audit = _module()
    assert audit._runtime_target("cpu") == ("cpu", None)
