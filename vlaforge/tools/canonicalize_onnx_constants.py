#!/usr/bin/env python3
"""Represent flat ONNX numeric Constant literals using tensor-valued attributes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def canonicalize(model):
    import onnx

    if model.functions or model.training_info or any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attr in node.attribute):
        raise ValueError("flat inference graph required")
    result = copy.deepcopy(model)
    converted = []
    kinds = {
        "value_int": (onnx.AttributeProto.INT, onnx.TensorProto.INT64, "i", False),
        "value_ints": (onnx.AttributeProto.INTS, onnx.TensorProto.INT64, "ints", True),
        "value_float": (onnx.AttributeProto.FLOAT, onnx.TensorProto.FLOAT, "f", False),
        "value_floats": (onnx.AttributeProto.FLOATS, onnx.TensorProto.FLOAT, "floats", True),
    }
    for node in result.graph.node:
        if node.op_type != "Constant" or node.domain not in ("", "ai.onnx"):
            continue
        if len(node.attribute) != 1 or node.input or len(node.output) != 1:
            raise ValueError("one literal attribute and one result required for Constant")
        attr = node.attribute[0]
        if attr.name not in kinds:
            continue
        attr_type, dtype, field, vector = kinds[attr.name]
        if attr.type != attr_type or attr.ref_attr_name:
            raise ValueError("literal attribute type mismatch")
        values = list(getattr(attr, field)) if vector else [getattr(attr, field)]
        if dtype == onnx.TensorProto.FLOAT and not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite float literal payload requires a separate bit contract")
        old_sha = hashlib.sha256(node.SerializeToString()).hexdigest()
        tensor = onnx.helper.make_tensor("", dtype, [len(values)] if vector else [], values)
        node.ClearField("attribute")
        node.attribute.append(onnx.helper.make_attribute("value", tensor))
        converted.append({"node": node.name, "output": node.output[0],
                          "source_node_sha256": old_sha, "source_attribute": attr.name,
                          "dtype": onnx.TensorProto.DataType.Name(dtype),
                          "shape": list(tensor.dims)})
    return result, {"schema": "vlaforge.onnx_literal_canonicalization/1",
                    "status": "canonicalized-not-target-validated", "converted": converted,
                    "source_nodes": len(model.graph.node), "output_nodes": len(result.graph.node),
                    "changes_numeric_dtype": False, "changes_public_io": False,
                    "changes_initializer_payloads": False, "target_execution_verified": False}


def main():
    import onnx
    from export_torchscript_region_onnx import digest, publish_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.absolute()
    report_path = output.with_suffix(".report.json")
    if source.parent != output.parent.resolve() or os.path.lexists(output) or os.path.lexists(report_path):
        raise ValueError("fresh output/report beside immutable source weights required")
    pin = digest(source)
    model = onnx.load(source, load_external_data=False)
    onnx.checker.check_model(str(source))
    result, report = canonicalize(model)
    with tempfile.NamedTemporaryFile(prefix=".literal-", suffix=".onnx", dir=source.parent) as temporary:
        staged = Path(temporary.name)
        # Serialize metadata only; never ask ONNX to rewrite external weight payloads.
        staged.write_bytes(result.SerializeToString())
        onnx.checker.check_model(str(staged))
        if digest(source) != pin:
            raise ValueError("immutable source changed during canonicalization")
        report.update(source=str(source), source_sha256=pin, output=str(output),
                      output_sha256=digest(staged), generator_sha256=digest(Path(__file__)))
        publish_files(staged, output, [])
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "converted"}, indent=2))


if __name__ == "__main__":
    main()
