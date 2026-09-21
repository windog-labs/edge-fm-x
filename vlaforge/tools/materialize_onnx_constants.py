#!/usr/bin/env python3
"""Materialize ONNX Constant nodes as initializers without changing semantics."""

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


def tensor_from_constant(node: Any) -> Any:
    import onnx

    if len(node.output) != 1:
        raise ValueError(f"Constant node must have one output: {node.name}")
    values = [
        attribute.t
        for attribute in node.attribute
        if attribute.name == "value"
    ]
    if len(values) != 1:
        raise ValueError(
            f"Constant node must use exactly one value attribute: {node.name}"
        )
    value = onnx.TensorProto()
    value.CopyFrom(values[0])
    value.name = node.output[0]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    import onnx

    input_path = args.input.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists():
        raise ValueError(f"materialized output already exists: {output}")
    if report_path.exists():
        raise ValueError(f"materialization report already exists: {report_path}")

    model = onnx.load(str(input_path), load_external_data=True)
    initializer_names = {value.name for value in model.graph.initializer}
    retained = []
    materialized = []
    for node in model.graph.node:
        if node.op_type != "Constant":
            retained.append(node)
            continue
        value = tensor_from_constant(node)
        if value.name in initializer_names:
            raise ValueError(f"duplicate initializer name: {value.name}")
        initializer_names.add(value.name)
        materialized.append(value)
    del model.graph.node[:]
    model.graph.node.extend(retained)
    model.graph.initializer.extend(materialized)
    materialized_names = {value.name for value in materialized}
    retained_info = [
        value
        for value in model.graph.value_info
        if value.name not in materialized_names
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    report = {
        "schema": "vlaforge.onnx_constant_materialization/1",
        "status": "passed",
        "input": str(input_path),
        "input_sha256": digest(input_path),
        "output": str(output),
        "output_sha256": digest(output),
        "constant_nodes": len(materialized),
        "remaining_nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
