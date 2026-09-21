"""Optional real SDK/GPU proof of the public graph provider's scoped policy."""

import json
import os
import struct
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("VLAFORGE_RUN_CUDA_TORCHSCRIPT") != "1",
                    reason="opt-in actual LibTorch scoped graph cleanup")
def test_actual_scoped_graph_cleanup_and_quarantine(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    root = Path(__file__).resolve().parents[2]
    major, minor = torch.cuda.get_device_capability(0)
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}"}
    (tmp_path / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(scoped_graph_regression LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(VLAFORGE_BUILD_TORCHSCRIPT_BACKEND ON CACHE BOOL "" FORCE)
set(VLAFORGE_TORCHSCRIPT_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM ON CACHE BOOL "" FORCE)
add_subdirectory("{root}" runtime)
add_executable(probe "{root}/tests/cpp/libtorch_graph_cleanup_cuda_smoke.cpp")
target_link_libraries(probe PRIVATE vlaforge_libtorch_graph_backend torch_cuda c10_cuda)
add_library(destroy_failure SHARED "{root}/tests/fixtures/graph_cleanup/destroy_failure.cpp")
target_link_libraries(destroy_failure PRIVATE CUDA::cudart dl)
''')
    build = tmp_path / "build"
    for index, command in enumerate([
        ["cmake", "-S", str(tmp_path), "-B", str(build),
         f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--parallel", "2"],
    ]):
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        (tmp_path / f"build-{index}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
    binary = build / "probe"
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    (tmp_path / "runner.ldd.txt").write_text(linked)
    assert all(word not in linked for word in ("libpython", "libtorch_python", "not found"))
    for mode in ("success", "exec", "graph", "poison"):
        folder = tmp_path / mode
        folder.mkdir()
        result = subprocess.run([str(binary), str(folder), mode],
                                env={**env, "PYTHONHOME": "/no/python", "PYTHONPATH": "/no/python",
                                     "LD_PRELOAD": str(build / "libdestroy_failure.so")},
                                capture_output=True, text=True, timeout=60, check=False)
        (folder / "stdout.jsonl").write_text(result.stdout)
        (folder / "stderr.log").write_text(result.stderr)
        assert result.returncode == (0 if mode == "success" else 78), result.stdout + result.stderr
        records = [json.loads(line) for line in result.stdout.splitlines()]
        if mode == "success":
            assert len(records) == 5
            assert all(row["private_pool_count"] == 0 and row["device_frees_this_destroy"] > 0 for row in records)
        else:
            assert records == [{"quarantine_retained": True, "failed_destroy_device_frees": 0,
                                "new_pool_isolated": True, "published_outputs_exact": True}]
            assert "[VLAFORGE-GRAPH-QUARANTINE]" in result.stderr
            if mode != "poison":
                assert "[GRAPH-FAILURE] injected" in result.stderr
        for path in folder.glob("*.f32"):
            scale = (5 if path.stem == "replacement" else 3 if path.stem == "survivor"
                     else 7 if "partial" in path.stem else int(path.stem[-1]) + 2)
            assert path.read_bytes() == struct.pack("<4096f", *(element * scale for element in range(4096)))
