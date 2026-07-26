#!/usr/bin/env python3
"""Run the minimal Host-CUDA artifact evaluation from an installed wheel.

The evaluator deliberately separates the package under test from its source
checkout:

1. build a wheel from the selected revision;
2. install it into a fresh venv;
3. switch to a non-Git working directory with no source ``PYTHONPATH``;
4. compile a small real ``sm_86`` CUDA AOTI artifact;
5. build and verify a Compile Bundle using runtime sources from the wheel;
6. execute the generated C++ Session with an invalid Python environment.

The synthetic tensor program is an artifact-evaluation smoke test, not
real-model evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any


_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
_SCHEMA = "vlaforge.wheel_artifact_evaluation/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(
    command: list[str],
    *,
    cwd: Path,
    log: Path,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - started
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(cwd),
                "elapsed_seconds": elapsed,
                "returncode": completed.returncode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n\n[stdout]\n"
        + completed.stdout
        + "\n[stderr]\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    text = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return f"unavailable (exit {completed.returncode}): {text}"
    return text


def _environment() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total,"
            "power.limit",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "host": platform.platform(),
        "machine": platform.machine(),
        "python_builder": platform.python_version(),
        "gpu": gpu,
        "nvcc": _version(["nvcc", "--version"]).splitlines()[-1],
        "cmake": _version(["cmake", "--version"]).splitlines()[0],
        "compiler": _version(["c++", "--version"]).splitlines()[0],
        "nsys": _version(["nsys", "--version"]).splitlines()[0],
        "ncu": _version(["ncu", "--version"]).splitlines()[-1],
    }


def _wheel_runtime_entries(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        return sorted(
            name
            for name in archive.namelist()
            if "/share/vlaforge/" in name
        )


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def evaluate(
    work_dir: Path,
    *,
    baseline_revision: str,
) -> dict[str, Any]:
    if work_dir.exists() and any(work_dir.iterdir()):
        raise ValueError(f"work directory must be absent or empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    logs = work_dir / "logs"
    wheels = work_dir / "wheels"
    venv = work_dir / "venv"
    non_git_cwd = work_dir / "non-git-cwd"
    audit_root = work_dir / "artifact-audit"
    wheels.mkdir()
    non_git_cwd.mkdir()

    current_revision = _git(["rev-parse", "HEAD"])
    tracked_status = _git(
        ["status", "--short", "--untracked-files=no"]
    )
    if tracked_status:
        raise RuntimeError(
            "wheel artifact evaluation requires a clean tracked worktree"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline_revision, "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("baseline revision is not an ancestor of HEAD")

    commands: list[dict[str, Any]] = []

    def run_step(
        name: str,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        log = logs / f"{name}.log"
        result = _run(
            command,
            cwd=cwd,
            log=log,
            environment=environment,
        )
        commands.append(
            {
                "name": name,
                "command": command,
                "cwd": str(cwd),
                "log": str(log),
            }
        )
        return result

    run_step(
        "build-wheel",
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheels),
            str(_SOURCE_ROOT),
        ],
        cwd=_REPOSITORY_ROOT,
    )
    built_wheels = tuple(wheels.glob("*.whl"))
    if len(built_wheels) != 1:
        raise RuntimeError("expected exactly one built VLAForge wheel")
    wheel = built_wheels[0]

    run_step(
        "create-venv",
        [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
        cwd=non_git_cwd,
    )
    venv_python = venv / "bin" / "python"
    run_step(
        "install-wheel",
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheel),
        ],
        cwd=non_git_cwd,
    )

    clean_environment = dict(os.environ)
    clean_environment.pop("PYTHONPATH", None)
    clean_environment.update(
        {
            "VLAFORGE_AUDIT_INSTALLED_PACKAGE": "1",
            "VLAFORGE_SOURCE_REVISION": current_revision,
            "VLAFORGE_SOURCE_DIRTY": "0",
        }
    )
    package_probe = run_step(
        "probe-installed-package",
        [
            str(venv_python),
            "-c",
            (
                "import json,sys,vlaforge;"
                "print(json.dumps({'prefix':sys.prefix,"
                "'package':vlaforge.__file__}))"
            ),
        ],
        cwd=non_git_cwd,
        environment=clean_environment,
    )
    installed = json.loads(package_probe.stdout)
    package_path = Path(installed["package"]).resolve()
    if venv.resolve() not in package_path.parents:
        raise RuntimeError(
            f"VLAForge was not imported from the clean venv: {package_path}"
        )
    runtime_root = venv / "share" / "vlaforge"
    for required in (
        "CMakeLists.txt",
        "runtime/state_store.cpp",
        "backends/aoti_region_executable.cpp",
        "include/vlaforge/runtime/session.h",
    ):
        if not (runtime_root / required).is_file():
            raise FileNotFoundError(runtime_root / required)

    git_probe = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=non_git_cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if git_probe.returncode == 0:
        raise RuntimeError("artifact evaluation cwd unexpectedly belongs to Git")

    audit_report_path = work_dir / "installed-wheel-aoti-audit.json"
    audit_tool = _SOURCE_ROOT / "tools" / "audit_cuda_aoti_region.py"
    run_step(
        "installed-wheel-sm86-audit",
        [
            str(venv_python),
            str(audit_tool),
            "--work-dir",
            str(audit_root),
            "--report",
            str(audit_report_path),
            "--runtime-root",
            str(runtime_root),
            "--source-revision",
            current_revision,
        ],
        cwd=non_git_cwd,
        environment=clean_environment,
    )
    audit = json.loads(audit_report_path.read_text(encoding="utf-8"))
    generated = audit["generated_session"]
    paged = audit["invocation_resident_generated_session"]
    required_negative_cases = {
        "abi-mismatch",
        "corrupt-artifact",
        "missing-artifact",
        "schema-mismatch",
        "wrong-device",
        "wrong-dtype",
        "wrong-layout",
        "wrong-shape",
    }
    if (
        audit.get("status") != "passed"
        or audit.get("target") != "sm_86"
        or not audit.get("installed_package_mode")
        or Path(audit["package_import"]).resolve() != package_path
        or Path(audit["runtime_root"]).resolve() != runtime_root.resolve()
        or audit.get("python_linked")
        or generated.get("artifact_target") != "sm_86"
        or generated.get("source_revision") != current_revision
        or not generated.get("schema_validated")
        or not generated.get("abi_validated")
        or set(generated.get("negative_cases", {}))
        != required_negative_cases
        or set(paged.get("negative_cases", {}))
        != required_negative_cases
    ):
        raise RuntimeError("installed-wheel CUDA artifact audit is incomplete")

    runtime_entries = _wheel_runtime_entries(wheel)
    return {
        "schema": _SCHEMA,
        "status": "passed",
        "passed": True,
        "classification": (
            "synthetic Host-CUDA artifact-evaluation smoke; not real-model "
            "or Orin evidence"
        ),
        "repository": {
            "baseline_revision": baseline_revision,
            "evaluated_revision": current_revision,
            "source_dirty": False,
        },
        "environment": _environment(),
        "isolation": {
            "non_git_cwd": True,
            "source_pythonpath_removed": True,
            "venv_system_site_packages": True,
            "package_import": str(package_path),
            "runtime_root": str(runtime_root.resolve()),
            "installed_from_wheel_only": True,
        },
        "wheel": {
            "filename": wheel.name,
            "sha256": _sha256(wheel),
            "size_bytes": wheel.stat().st_size,
            "runtime_source_entries": len(runtime_entries),
        },
        "artifact_evaluation": {
            "target": audit["target"],
            "torch": audit["torch_version"],
            "cuda": audit["cuda_version"],
            "artifact": audit["package"],
            "compile_seconds": audit["compile_seconds"],
            "direct_cpp": {
                "maximum_absolute_error": audit["max_abs_error"],
                "invalid_python_environment": (
                    audit["invalid_python_environment_run"]
                ),
                "links_libpython": audit["python_linked"],
                "negative_cases": audit["backend_negative_cases"],
            },
            "session_resident_bundle": generated,
            "invocation_resident_bundle": paged,
        },
        "commands": commands,
        "audit_tool": {
            "path": str(audit_tool),
            "sha256": _sha256(audit_tool),
        },
        "work_dir": {
            "path": str(work_dir),
            "size_bytes": _directory_size(work_dir),
            "large_files_committed": False,
        },
        "claim_boundary": {
            "host_cuda": True,
            "orin": False,
            "real_model": False,
            "model_kernel_optimization": False,
            "source_checkout_used_for_package_import": False,
        },
        "reproduction": {
            "command_template": [
                sys.executable,
                str(Path(__file__).resolve()),
                "--work-dir",
                "<empty-work-dir>",
                "--baseline-revision",
                baseline_revision,
                "--report",
                "<report.json>",
            ]
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    wheel = report["wheel"]
    artifact = report["artifact_evaluation"]
    bundle = artifact["session_resident_bundle"]
    return "\n".join(
        (
            "# VLAForge installed-wheel artifact evaluation",
            "",
            f"Status: **{report['status']}**.",
            "",
            "| Gate | Result |",
            "|---|---|",
            f"| Evaluated revision | `{report['repository']['evaluated_revision']}` |",
            f"| Wheel SHA256 | `{wheel['sha256']}` |",
            f"| Runtime source entries | {wheel['runtime_source_entries']} |",
            "| Working directory | non-Git |",
            "| Package import | installed wheel |",
            f"| CUDA target | `{artifact['target']}` |",
            f"| Bundle digest | `{bundle['bundle_digest']}` |",
            f"| I/O schema digest | `{bundle['io_schema_digest']}` |",
            "| C ABI validation | passed |",
            "| Schema mismatch rejection | passed |",
            "| Artifact hash rejection | passed |",
            "| Target mismatch rejection | passed |",
            "| Invalid-Python generated C++ run | passed |",
            "| Generated runner links libpython | no |",
            "",
            "This is a synthetic Host-CUDA artifact-evaluation smoke test. "
            "It is not real-model, Orin, latency, power, or kernel-optimization "
            "evidence.",
            "",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-revision",
        default="f0fc1be",
        help="Required ancestor representing the frozen starting baseline.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)

    report = evaluate(
        args.work_dir.resolve(),
        baseline_revision=args.baseline_revision,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(report["artifact_evaluation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
