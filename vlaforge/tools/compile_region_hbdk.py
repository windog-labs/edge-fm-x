#!/usr/bin/env python3
"""Compile a fixed ONNX region to a Horizon HBM with reproducible evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--march", default="nash-m")
    parser.add_argument("--opt", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    onnx_path = args.onnx.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.suffix != ".hbm":
        raise ValueError("Horizon compiler output must end with .hbm")
    if output.exists():
        raise ValueError(f"compiler output already exists: {output}")
    if report_path.exists():
        raise ValueError(f"compiler report already exists: {report_path}")
    if args.opt < 0 or args.jobs < 1:
        raise ValueError("optimization level must be non-negative and jobs positive")

    import onnx
    from hbdk4.compiler import apis
    from hbdk4.compiler import onnx as hbdk_onnx

    advice_dir = report_path.with_name(report_path.stem + "-advice")
    advice_path = advice_dir / f"{output.stem}_advice.json"
    started_ns = time.time_ns()
    report: dict[str, Any] = {
        "schema": "vlaforge.horizon_onnx_compile/1",
        "status": "started",
        "march": args.march,
        "source": str(onnx_path),
        "source_sha256": digest(onnx_path),
        "output": str(output),
        "report": str(report_path),
        "advice_json": str(advice_path),
        "opt": args.opt,
        "jobs": args.jobs,
        "versions": {
            "hbdk4_compiler": package_version("hbdk4-compiler"),
            "hbdk4_march": package_version("hbdk4-march"),
            "onnx": package_version("onnx"),
        },
        "started_ns": started_ns,
    }
    write_json(report_path, report)
    try:
        model = onnx.load(str(onnx_path), load_external_data=True)
        onnx.checker.check_model(model)
        if not model.graph.input:
            raise ValueError(
                "HBDK4 does not compile constant-only graphs without a runtime "
                "input; materialize the constant through the stage contract"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        advice_dir.mkdir(parents=True, exist_ok=True)
        module = hbdk_onnx.export(model, name=output.stem)
        converted = apis.convert(
            module,
            march=args.march,
            advice=True,
            advice_path=str(advice_dir) + "/",
        )
        apis.compile(
            converted,
            path=str(output),
            march=args.march,
            opt=args.opt,
            jobs=args.jobs,
            progress_bar=False,
        )
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("HBDK4 did not produce an HBM")
        report.update(
            {
                "status": "passed",
                "hbm": str(output),
                "hbm_sha256": digest(output),
                "hbm_size": output.stat().st_size,
                "advice_size": (
                    advice_path.stat().st_size if advice_path.exists() else 0
                ),
                "advice_files": (
                    sorted(path.name for path in advice_dir.glob("*"))
                    if advice_dir.exists()
                    else []
                ),
                "elapsed_ns": time.time_ns() - started_ns,
            }
        )
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_ns": time.time_ns() - started_ns,
            }
        )
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
