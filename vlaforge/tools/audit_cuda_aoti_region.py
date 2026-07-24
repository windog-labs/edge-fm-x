#!/usr/bin/env python3
"""Compile and audit one real CUDA AOTInductor RegionExecutable artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch


class AuditTensorRegion(torch.nn.Module):
    """Small real tensor program used to exercise the production AOTI ABI."""

    def forward(
        self, values: torch.Tensor, gain: torch.Tensor
    ) -> torch.Tensor:
        return (torch.sin(values) + values.square()) * gain


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        **kwargs,
    )


def _parse_output(text: str) -> list[float]:
    line = next(
        (item for item in text.splitlines() if item.startswith("OUTPUT,")),
        None,
    )
    if line is None:
        raise RuntimeError(f"C++ runner did not emit OUTPUT: {text}")
    return [float(item) for item in line.split(",")[1:]]


def _audit(root: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    source_root = Path(__file__).resolve().parents[1]
    package_path = root / "audit_tensor_region.pt2"
    build_dir = root / "build"

    model = AuditTensorRegion().eval().cuda()
    values = (
        torch.arange(16, dtype=torch.float32, device="cuda")
        .reshape(4, 4)
        .div(8.0)
    )
    gain = torch.tensor(0.75, dtype=torch.float32, device="cuda")
    expected = model(values, gain).detach().cpu().flatten().tolist()

    compile_start = time.perf_counter()
    exported = torch.export.export(model, (values, gain), strict=True)
    actual_package = Path(
        torch._inductor.aoti_compile_and_package(
            exported, package_path=str(package_path)
        )
    )
    compile_seconds = time.perf_counter() - compile_start
    if actual_package != package_path or not package_path.is_file():
        raise RuntimeError(
            f"AOTI package path mismatch: {actual_package} != {package_path}"
        )

    configure = _run(
        [
            "cmake",
            "-S",
            str(source_root),
            "-B",
            str(build_dir),
            "-DVLAFORGE_BUILD_AOTI_BACKEND=ON",
            "-DBUILD_TESTING=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
        ]
    )
    build = _run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "vlaforge_aoti_region_smoke",
            "--parallel",
            "4",
        ]
    )
    runner = build_dir / "tests" / "cpp" / "vlaforge_aoti_region_smoke"
    clean_environment = dict(os.environ)
    clean_environment.update(
        {
            "PYTHONHOME": "/definitely/not/a/python/home",
            "PYTHONPATH": "/definitely/not/a/python/path",
        }
    )
    run_start = time.perf_counter()
    completed = _run(
        [str(runner), str(package_path)], env=clean_environment
    )
    run_seconds = time.perf_counter() - run_start
    actual = _parse_output(completed.stdout)
    if len(actual) != len(expected):
        raise RuntimeError(
            f"output length mismatch: {len(actual)} != {len(expected)}"
        )
    max_abs_error = max(
        (abs(left - right) for left, right in zip(actual, expected, strict=True)),
        default=0.0,
    )
    if max_abs_error > 1e-6:
        raise RuntimeError(f"AOTI numeric mismatch: {max_abs_error}")

    linked = _run(["ldd", str(runner)]).stdout
    linked_libraries = [
        line.strip().split()[0].lower()
        for line in linked.splitlines()
        if line.strip()
    ]
    if any(name.startswith("libpython") for name in linked_libraries):
        raise RuntimeError("C++ AOTI runner unexpectedly links Python")

    return {
        "schema": "vlaforge.cuda_aoti_audit/1",
        "status": "passed",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "package": {
            "path": str(package_path),
            "sha256": _sha256(package_path),
            "size_bytes": package_path.stat().st_size,
        },
        "compile_seconds": compile_seconds,
        "run_seconds": run_seconds,
        "max_abs_error": max_abs_error,
        "output_elements": len(actual),
        "cpp_runner": str(runner),
        "python_linked": False,
        "invalid_python_environment_run": True,
        "configure_tail": configure.stdout.splitlines()[-5:],
        "build_tail": build.stdout.splitlines()[-5:],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        help="Persistent audit directory; a temporary directory is used by default.",
    )
    parser.add_argument("--report", help="Optional JSON report path.")
    args = parser.parse_args(argv)

    context = (
        nullcontext(Path(args.work_dir).resolve())
        if args.work_dir
        else tempfile.TemporaryDirectory(prefix="vlaforge-aoti-audit-")
    )
    with context as selected:
        root = (
            selected
            if isinstance(selected, Path)
            else Path(selected).resolve()
        )
        root.mkdir(parents=True, exist_ok=True)
        report = _audit(root)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
