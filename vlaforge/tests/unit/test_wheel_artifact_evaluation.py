from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType


def _module(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "tools" / name
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_wheel_runtime_inventory_and_directory_size(tmp_path: Path) -> None:
    evaluation = _module("evaluate_wheel_artifact.py")
    wheel = tmp_path / "vlaforge.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "vlaforge-0.2.data/data/share/vlaforge/CMakeLists.txt",
            "cmake",
        )
        archive.writestr("vlaforge/__init__.py", "")
        archive.writestr(
            "vlaforge-0.2.data/data/share/vlaforge/runtime/session.cpp",
            "runtime",
        )
    assert evaluation._wheel_runtime_entries(wheel) == [
        "vlaforge-0.2.data/data/share/vlaforge/CMakeLists.txt",
        "vlaforge-0.2.data/data/share/vlaforge/runtime/session.cpp",
    ]
    assert evaluation._directory_size(tmp_path) >= wheel.stat().st_size


def test_built_wheel_contains_private_aoti_backend_header(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(project),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(tmp_path.glob("vlaforge-*.whl"))
    entries = _module("evaluate_wheel_artifact.py")._wheel_runtime_entries(
        wheel
    )
    assert any(
        entry.endswith("/share/vlaforge/backends/aoti_callable.h")
        for entry in entries
    )


def test_artifact_evaluation_markdown_has_claim_boundaries() -> None:
    evaluation = _module("evaluate_wheel_artifact.py")
    report = {
        "status": "passed",
        "repository": {"evaluated_revision": "deadbeef"},
        "wheel": {
            "sha256": "1" * 64,
            "runtime_source_entries": 24,
        },
        "artifact_evaluation": {
            "target": "sm_86",
            "session_resident_bundle": {
                "bundle_digest": "2" * 64,
                "io_schema_digest": "3" * 64,
            },
        },
    }
    markdown = evaluation.render_markdown(report)
    assert "installed wheel" in markdown
    assert "`sm_86`" in markdown
    assert "not real-model" in markdown
    assert "Generated runner links libpython | no" in markdown


def test_cuda_aoti_audit_exercises_schema_and_abi_rejection() -> None:
    audit = _module("audit_cuda_aoti_region.py")
    runner = audit._generated_runner_source()
    assert 'mode == "schema-mismatch"' in runner
    assert 'mode == "abi-mismatch"' in runner
    assert "vlaforge_session_api_validate" in runner
    assert "invalid_api.abi_version += 1u" in runner
