import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("VLAFORGE_RUN_CUDA_TORCHSCRIPT") != "1",
                    reason="opt-in actual CUDA stream lease allocator regression")
def test_actual_cuda_stream_leases_bound_workspace_growth(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    root = Path(__file__).resolve().parents[2]
    major, minor = torch.cuda.get_device_capability(0)
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}"}
    (tmp_path / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.18)
project(context_pool_cuda LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
add_executable(probe "{root}/tests/cpp/execution_context_pool_cuda_smoke.cpp"
  "{root}/runtime/execution_context.cpp" "{root}/runtime/region_executable.c")
target_include_directories(probe PRIVATE "{root}/include")
target_compile_definitions(probe PRIVATE VLAFORGE_ENABLE_CUDA_ARENA=1)
target_link_libraries(probe PRIVATE ${{TORCH_LIBRARIES}} torch_cuda c10_cuda CUDA::cudart)
''')
    build = tmp_path / "build"
    commands = [
        ["cmake", "-S", str(tmp_path), "-B", str(build),
         f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--parallel", "2"],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, env=env, check=False, capture_output=True, text=True)
        (tmp_path / f"build-{index}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
    binary = build / "probe"
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    (tmp_path / "runner.ldd.txt").write_text(linked)
    assert "libpython" not in linked and "libtorch_python" not in linked and "not found" not in linked
    result = subprocess.run([str(binary)], env={**env, "PYTHONPATH": "/no/python", "PYTHONHOME": "/no/python"},
                            check=False, capture_output=True, text=True, timeout=60)
    (tmp_path / "runner.stdout.jsonl").write_text(result.stdout)
    (tmp_path / "runner.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(result.stdout.splitlines()) == 10
