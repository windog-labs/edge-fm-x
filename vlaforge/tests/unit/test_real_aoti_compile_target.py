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
        / "compile_real_aoti_exports.py"
    )
    specification = importlib.util.spec_from_file_location(
        "compile_real_aoti_exports",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_native_cuda_arch_guard_accepts_only_current_target() -> None:
    compiler = _module()
    compiler._validate_native_cuda_arch("8.0", major=8, minor=0)
    compiler._validate_native_cuda_arch("9.0+PTX", major=9, minor=0)
    compiler._validate_native_cuda_arch(None, major=8, minor=6)
    with pytest.raises(RuntimeError, match="destination-native"):
        compiler._validate_native_cuda_arch("8.6", major=9, minor=0)
    with pytest.raises(RuntimeError, match="destination-native"):
        compiler._validate_native_cuda_arch(
            "8.0;9.0",
            major=9,
            minor=0,
        )
