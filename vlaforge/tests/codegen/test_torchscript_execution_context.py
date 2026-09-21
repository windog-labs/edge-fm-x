import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("VLAFORGE_RUN_CUDA_TORCHSCRIPT") != "1",
                    reason="opt-in native TorchScript shared-stream/replay CUDA audit")
def test_native_torchscript_replays_and_quarantines_invalid_capture(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    root = Path(__file__).resolve().parents[2]
    major, minor = torch.cuda.get_device_capability(0)
    env = {**os.environ, "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}"}
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(root), "-B", str(build),
                    "-DVLAFORGE_BUILD_TORCHSCRIPT_BACKEND=ON", "-DVLAFORGE_TORCHSCRIPT_ENABLE_CUDA=ON",
                    "-DBUILD_TESTING=ON", f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}"],
                   env=env, check=True, capture_output=True, text=True)
    subprocess.run(["cmake", "--build", str(build), "-j", "2", "--target",
                    "vlaforge_torchscript_bounded_replay_smoke"],
                   env=env, check=True, capture_output=True, text=True)
    binary = build / "tests/cpp/vlaforge_torchscript_bounded_replay_smoke"
    linked = subprocess.check_output(["ldd", str(binary)], text=True)
    assert "libpython" not in linked and "not found" not in linked
    (tmp_path / "runner.ldd.txt").write_text(linked)
    for mode, expected in (("success", 0), ("fatal", 78)):
        result = subprocess.run([str(binary), str(tmp_path / f"{mode}.pt"), f"sm_{major}{minor}", mode],
                                env={**env, "PYTHONHOME": "/no/python", "PYTHONPATH": "/no/python"},
                                text=True, capture_output=True, check=False)
        (tmp_path / f"{mode}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == expected, result.stdout + result.stderr
        assert ("full eager bytes exact" if mode == "success" else "fatal_poisoned=1 ordinary_count=0") in result.stdout
