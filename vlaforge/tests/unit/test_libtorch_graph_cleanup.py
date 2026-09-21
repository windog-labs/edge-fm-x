"""Checked native cleanup control flow with injected API results, not GPU evidence."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/graph_cleanup"


def compile_probe(output, *, minor=10):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    return subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
        f"-DTORCH_VERSION_MINOR={minor}", "-I", str(FIXTURE),
        "-I", str(ROOT / "backends"), str(FIXTURE / "probe.cpp"),
        "-o", str(output),
    ], capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def checked_probe(tmp_path_factory):
    binary = tmp_path_factory.mktemp("checked-graph") / "probe"
    result = compile_probe(binary)
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("stage", range(6))
def test_checked_cleanup_preserves_remaining_owners_on_failure(checked_probe, stage):
    result = subprocess.run([str(checked_probe), str(stage)], capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("minor", [9, 11])
def test_private_sdk_layout_rejects_unaudited_versions(tmp_path, minor):
    result = compile_probe(tmp_path / "probe", minor=minor)
    assert result.returncode != 0
    assert "requires the audited LibTorch 2.10" in result.stderr


def test_cmake_rejects_scoped_policy_without_cuda_provider(tmp_path):
    result = subprocess.run([
        "cmake", "-S", str(ROOT), "-B", str(tmp_path / "build"),
        "-DVLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=ON", "-DBUILD_TESTING=OFF",
    ], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "requires the LibTorch CUDA graph backend" in " ".join(result.stderr.split())


@pytest.mark.parametrize("version", ["2.9.1", "2.11.0"])
def test_cmake_rejects_unaudited_torch_before_any_cuda_target(tmp_path, version):
    package = tmp_path / "fake-torch"
    package.mkdir()
    (package / "TorchConfig.cmake").write_text(f'set(Torch_FOUND TRUE)\nset(Torch_VERSION "{version}")\n')
    result = subprocess.run([
        "cmake", "-S", str(ROOT), "-B", str(tmp_path / "build"),
        "-DVLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=ON", "-DBUILD_TESTING=OFF",
        "-DVLAFORGE_BUILD_TORCHSCRIPT_BACKEND=ON", "-DVLAFORGE_TORCHSCRIPT_ENABLE_CUDA=ON",
        f"-DTorch_DIR={package}",
    ], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "requires audited LibTorch 2.10" in " ".join(result.stderr.split())
