#!/usr/bin/env python3
"""Extract named ONNX outputs and their transitive graph dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-name", action="append", required=True)
    args = parser.parse_args()

    import onnx

    input_path = args.input.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists():
        raise ValueError(f"extracted output already exists: {output}")
    if report_path.exists():
        raise ValueError(f"extraction report already exists: {report_path}")

    source = onnx.load(str(input_path), load_external_data=True)
    output_names = set(args.output_name)
    declared_outputs = {value.name for value in source.graph.output}
    missing = output_names - declared_outputs
    if missing:
        raise ValueError(f"requested outputs are not declared: {sorted(missing)}")
    input_names = [value.name for value in source.graph.input]
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(
        str(input_path),
        str(output),
        input_names,
        list(args.output_name),
    )
    model = onnx.load(str(output), load_external_data=True)
    onnx.checker.check_model(model)
    report: dict[str, Any] = {
        "schema": "vlaforge.onnx_output_extraction/1",
        "status": "passed",
        "source": str(input_path),
        "source_sha256": digest(input_path),
        "output": str(output),
        "output_sha256": digest(output),
        "requested_outputs": list(args.output_name),
        "graph_inputs": [value.name for value in model.graph.input],
        "graph_outputs": [value.name for value in model.graph.output],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
