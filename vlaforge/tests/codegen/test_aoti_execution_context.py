from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.cuda_aoti
@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_CUDA_AOTI") != "1",
    reason="set VLAFORGE_RUN_CUDA_AOTI=1 for the shared-context CUDA audit",
)
@pytest.mark.parametrize(
    ("bounded_replay", "artifact_mode"),
    [(False, "package"), (True, "package"), (True, "raw")],
)
def test_aoti_shared_execution_context(
    tmp_path: Path, bounded_replay: bool, artifact_mode: str
) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    class AuditRegion(torch.nn.Module):
        def forward(self, values, gain):
            return (torch.sin(values) + values.square()) * gain

    inputs = (
        torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4) / 8,
        torch.tensor(0.75, device="cuda"),
    )
    exported = torch.export.export(AuditRegion().eval(), inputs, strict=True)
    if artifact_mode == "raw":
        artifact = Path(
            torch._inductor.aot_compile(
                exported.module(),
                inputs,
                options={
                    "aot_inductor.output_path": str(tmp_path / "audit.so"),
                    "aot_inductor.package": False,
                },
            )
        )
        assert artifact.suffix == ".so" and artifact.is_file()
    else:
        artifact = tmp_path / "audit.pt2"
        torch._inductor.aoti_compile_and_package(exported, package_path=str(artifact))
    major, minor = torch.cuda.get_device_capability(0)
    environment = {**os.environ, "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}"}
    source_root = Path(__file__).resolve().parents[2]
    build = tmp_path / "build"
    runner_name = (
        "vlaforge_aoti_bounded_replay_smoke"
        if bounded_replay
        else "vlaforge_aoti_execution_context_smoke"
    )
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(build),
            "-DVLAFORGE_BUILD_AOTI_BACKEND=ON",
            "-DBUILD_TESTING=ON",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "-j",
            "2",
            "--target",
            runner_name,
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    runner = build / "tests" / "cpp" / runner_name
    dependencies = subprocess.run(
        ["ldd", str(runner)], check=True, capture_output=True, text=True
    )
    assert "libpython" not in dependencies.stdout.lower()
    result = subprocess.run(
        [str(runner), str(artifact)]
        + (
            [f"sm_{major}{minor}", "success"]
            if bounded_replay
            else ["cuda", f"sm_{major}{minor}"]
        ),
        env={**environment, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"},
        check=True,
        capture_output=True,
        text=True,
    )
    if bounded_replay:
        assert "BOUNDED_REPLAY real AOTI CUDA two-region N4 audit passed" in result.stdout
        fatal = subprocess.run(
            [str(runner), str(artifact), f"sm_{major}{minor}", "fatal"],
            env={**environment, "PYTHONHOME": "/nonexistent", "PYTHONPATH": "/nonexistent"},
            check=False,
            capture_output=True,
            text=True,
        )
        assert fatal.returncode == 78, fatal.stdout + fatal.stderr
        assert "fatal_poisoned=1 ordinary_count=0" in fatal.stdout
    else:
        assert "EXECUTION_CONTEXT passed device=cuda regions=2 runs=3" in result.stdout
