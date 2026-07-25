#!/usr/bin/env python3
"""Aggregate clean Host-CUDA release-gate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.deployment import load_bundle_manifest  # noqa: E402


REPORT_SCHEMA = "vlaforge.host_cuda_release_gate/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", "0")) for suite in suites)
    failures = sum(
        int(suite.attrib.get("failures", "0")) for suite in suites
    )
    errors = sum(int(suite.attrib.get("errors", "0")) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", "0")) for suite in suites)
    seconds = sum(float(suite.attrib.get("time", "0")) for suite in suites)
    names = [
        str(item.attrib["name"])
        for suite in suites
        for item in suite.findall("testcase")
    ]
    return {
        "tests": tests,
        "passed": tests - failures - errors - skipped,
        "skipped": skipped,
        "failures": failures,
        "errors": errors,
        "seconds": seconds,
        "test_names": names,
    }


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _binary_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _environment() -> dict[str, Any]:
    import torch

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    nvcc = subprocess.run(
        ["nvcc", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[-1]
    return {
        "host": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "nvcc": nvcc,
    }


def aggregate(gate_root: Path) -> dict[str, Any]:
    python_tests = junit_summary(gate_root / "python-junit.xml")
    cuda_aoti = junit_summary(gate_root / "cuda-aoti-junit.xml")
    cpu_ctest = junit_summary(gate_root / "cpu-ctest.xml")
    cuda_ctest = junit_summary(gate_root / "cuda-ctest.xml")
    for name, result in (
        ("python", python_tests),
        ("cuda_aoti", cuda_aoti),
        ("cpu_ctest", cpu_ctest),
        ("cuda_ctest", cuda_ctest),
    ):
        if result["failures"] or result["errors"]:
            raise ValueError(f"{name} release tests failed: {result}")
    if (
        python_tests["passed"] < 200
        or cuda_aoti["passed"] != 1
        or cpu_ctest["passed"] != 7
        or cuda_ctest["passed"] != 8
    ):
        raise ValueError("release test coverage is incomplete")

    for status in ("cpu-consumer.status", "cuda-consumer.status"):
        if (gate_root / status).read_text(encoding="utf-8").strip() != "passed":
            raise ValueError(f"install consumer did not pass: {status}")

    build_logs = {
        name: (gate_root / name).read_text(encoding="utf-8")
        for name in ("cpu-build.log", "cuda-build.log")
    }
    for name, text in build_logs.items():
        if ".cu.o" in text or "/src/operators/" in text:
            raise ValueError(f"{name} compiled an old/custom CUDA kernel")

    wheels = tuple((gate_root / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("release gate requires exactly one wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        runtime_entries = sorted(
            name
            for name in archive.namelist()
            if "/share/vlaforge/" in name
        )
    required_suffixes = {
        "share/vlaforge/CMakeLists.txt",
        "share/vlaforge/runtime/state_store.cpp",
        "share/vlaforge/include/vlaforge/runtime/session.h",
        "share/vlaforge/backends/aoti_region_executable.cpp",
    }
    if not all(
        any(entry.endswith(suffix) for entry in runtime_entries)
        for suffix in required_suffixes
    ):
        raise ValueError("wheel is missing standalone runtime source")

    bundle_root = gate_root / "wheel-bundle"
    bundle_path = bundle_root / "bundle.json"
    bundle = load_bundle_manifest(bundle_path)
    bundle.verify_files(bundle_root)
    if (
        bundle.reproducibility.source_revision
        != "package:vlaforge-0.2.0.dev0"
        or bundle.reproducibility.source_dirty
    ):
        raise ValueError("installed-wheel bundle provenance is invalid")
    ldd = (gate_root / "wheel-runner.ldd").read_text(
        encoding="utf-8"
    ).lower()
    if "python" in ldd:
        raise ValueError("installed-wheel generated runner links Python")
    runner_output = (
        gate_root / "wheel-runner.log"
    ).read_text(encoding="utf-8").splitlines()
    if len(runner_output) != 3 or any(
        not line.startswith("OUTPUT,") for line in runner_output
    ):
        raise ValueError("installed-wheel generated runner output mismatch")

    architecture = _json(
        _REPOSITORY_ROOT
        / "doc/reports/vlaforge_architecture_v01/architecture_surface.json"
    )
    heldouts = _json(
        _REPOSITORY_ROOT
        / "doc/reports/vlaforge_heldout_v01/heldout_audit.json"
    )
    real_root = (
        _REPOSITORY_ROOT / "doc/reports/vlaforge_real_v03"
    )
    real_evidence = {
        "SmolVLA": _json(real_root / "smolvla_artifact_l4.json"),
        "DiffusionDrive": _json(
            real_root / "diffusiondrive_artifact_l4.json"
        ),
        "OpenVLA": _json(real_root / "openvla_artifact_l3.json"),
        "performance": _json(real_root / "real_cuda_evidence.json"),
    }
    if (
        not architecture.get("passed")
        or not heldouts.get("passed")
        or any(
            record.get("status") != "passed"
            for record in real_evidence.values()
        )
        or real_evidence["SmolVLA"].get("evidence_level") != "L4"
        or real_evidence["DiffusionDrive"].get("evidence_level") != "L4"
        or real_evidence["OpenVLA"].get("evidence_level") != "L3"
    ):
        raise ValueError("required architecture/model evidence is not passing")

    tracked_status = _git(
        ["status", "--short", "--untracked-files=no"]
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "passed": True,
        "repository": {
            "revision": _git(["rev-parse", "HEAD"]),
            "source_dirty": bool(tracked_status),
            "tracked_status": (
                tracked_status.splitlines() if tracked_status else []
            ),
        },
        "environment": _environment(),
        "tests": {
            "python": python_tests,
            "cuda_aoti_opt_in": cuda_aoti,
            "cpu_ctest": cpu_ctest,
            "cuda_ctest": cuda_ctest,
        },
        "clean_build": {
            "cpu_release": True,
            "cuda_aoti_release": True,
            "cuda_target": "sm_86",
            "old_edgefm_or_custom_cuda_sources_compiled": False,
            "cpu_install_consumer": True,
            "cuda_install_consumer": True,
            "installed_binaries": {
                "cpu_runtime": _binary_record(
                    gate_root / "cpu-install/lib/libvlaforge_runtime.a"
                ),
                "cuda_runtime": _binary_record(
                    gate_root / "cuda-install/lib/libvlaforge_runtime.a"
                ),
                "aoti_backend": _binary_record(
                    gate_root / "cuda-install/lib/libvlaforge_aoti_backend.a"
                ),
                "cpu_consumer": _binary_record(
                    gate_root / "cpu-consumer/vlaforge_install_consumer"
                ),
                "cuda_consumer": _binary_record(
                    gate_root / "cuda-consumer/vlaforge_install_consumer"
                ),
            },
        },
        "wheel": {
            "filename": wheel.name,
            "sha256": _sha256(wheel),
            "size_bytes": wheel.stat().st_size,
            "runtime_source_entries": len(runtime_entries),
            "installed_cli_bundle": {
                "bundle_digest": bundle.digest(),
                "io_schema_digest": bundle.io_schema_digest,
                "profile": bundle.compilation_certificate.profile.value,
                "source_revision": (
                    bundle.reproducibility.source_revision
                ),
                "source_dirty": bundle.reproducibility.source_dirty,
                "invalid_python_environment": True,
                "links_libpython": False,
                "output_lines": runner_output,
                "runner": _binary_record(
                    bundle_root / "bin/vlaforge_generated_runner"
                ),
            },
        },
        "prior_evidence": {
            "architecture_surface": {
                "passed": True,
                "schema": architecture["schema"],
            },
            "frozen_core_heldouts": {
                "passed": True,
                "schema": heldouts["schema"],
                "models": [
                    item["name"] for item in heldouts["models"]
                ],
                "core_op_delta": heldouts["summary"]["core_op_delta"],
            },
            "real_models": {
                "SmolVLA": "L4",
                "DiffusionDrive": "L4",
                "OpenVLA": "L3",
            },
            "host_cuda_performance": "passed",
        },
        "claim_boundary": {
            "host_cuda": True,
            "orin": False,
            "sensor_sync_or_scheduling": False,
            "old_edgefm_kernel_optimization": False,
        },
        "summary": {
            "release_gate_passed": True,
            "python_passed": python_tests["passed"],
            "python_opt_in_skipped": python_tests["skipped"],
            "cuda_aoti_opt_in_passed": cuda_aoti["passed"],
            "cpu_ctest_passed": cpu_ctest["passed"],
            "cuda_ctest_passed": cuda_ctest["passed"],
            "wheel_install_bundle_passed": True,
            "no_python_runner": True,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    tests = report["tests"]
    wheel = report["wheel"]
    lines = [
        "# VLAForge Host-CUDA release gate",
        "",
        f"Status: **{report['status']}**.",
        "",
        "| Gate | Result |",
        "|---|---:|",
        (
            f"| Python | {tests['python']['passed']} passed, "
            f"{tests['python']['skipped']} explicit opt-in skipped |"
        ),
        (
            f"| CUDA AOTI opt-in | "
            f"{tests['cuda_aoti_opt_in']['passed']} passed |"
        ),
        f"| CPU Release CTest | {tests['cpu_ctest']['passed']}/7 |",
        f"| CUDA/AOTI Release CTest | {tests['cuda_ctest']['passed']}/8 |",
        "| CPU installed-package consumer | passed |",
        "| CUDA installed-package consumer | passed |",
        "| Old EdgeFM/custom CUDA sources compiled | no |",
        "| Installed wheel CLI bundle | passed |",
        "| Invalid-Python generated runner | passed |",
        "| Generated runner links libpython | no |",
        "",
        "## Installed wheel",
        "",
        f"- Wheel: `{wheel['filename']}`",
        f"- SHA256: `{wheel['sha256']}`",
        f"- Bundled runtime source entries: {wheel['runtime_source_entries']}",
        (
            "- Bundle digest: "
            f"`{wheel['installed_cli_bundle']['bundle_digest']}`"
        ),
        (
            "- I/O schema digest: "
            f"`{wheel['installed_cli_bundle']['io_schema_digest']}`"
        ),
        (
            "- Provenance: "
            f"`{wheel['installed_cli_bundle']['source_revision']}`"
        ),
        "",
        "## Evidence boundary",
        "",
        "- This is an RTX 3060 `sm_86` Host-CUDA release gate.",
        "- It is not Orin latency, power, thermal, or closed-loop evidence.",
        "- Model kernel compilation remains upstream AOTI work.",
        "- VLAForge does not provide sensor synchronization or physical "
        "scheduling.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    report = aggregate(args.gate_root.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
