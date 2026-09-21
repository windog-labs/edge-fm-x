"""CPU fault injection against the actual context implementation, not CUDA evidence."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def context_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    fixture = ROOT / "tests/fixtures/execution_context_pool"
    binary = tmp_path_factory.mktemp("context-pool") / "probe"
    subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
        "-DVLAFORGE_ENABLE_CUDA_ARENA=1", "-I", str(fixture),
        "-I", str(ROOT / "include"), str(fixture / "probe.cpp"),
        str(ROOT / "runtime/execution_context.cpp"),
        str(ROOT / "runtime/region_executable.c"), "-o", str(binary),
    ], check=True, capture_output=True, text=True)
    return binary


@pytest.mark.parametrize("mode", [
    "reuse", "live-isolation", "device-isolation", "poison", "drain-failure",
    "capture-active", "device-failure", "creation-failure", "concurrent-leases", "late-global-release",
    "sticky-drain-failure", "sticky-device-failure", "invalid-input-does-not-poison",
])
def test_actual_context_stream_leases_with_injected_cuda(context_probe, mode):
    result = subprocess.run([str(context_probe), mode], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
