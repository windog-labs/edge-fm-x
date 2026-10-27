#!/usr/bin/env python3
"""Collapse proven leading batch axes around Trilu for rank-three importers."""
import argparse
import copy
import json
import math
from pathlib import Path
import tempfile


def lower(model):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    if model.functions or model.training_info or any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attr in node.attribute):
        raise ValueError("flat inference graph required")
    result = copy.deepcopy(model)
    values = {value.name: value for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]}
    occupied = {value.name for value in [*model.graph.input, *model.graph.output, *model.graph.value_info, *model.graph.initializer]}
    occupied.update(name for node in model.graph.node for name in [node.name, *node.input, *node.output])
    report = {"schema": "vlaforge.onnx_trilu_batch_axes/1", "rewritten": [], "skipped": [],
              "changes_public_io": False, "changes_data_dtype": False, "numerical_values_recomputed": False}
    nodes = []
    def fresh(name):
        while name in occupied:
            name += "_"
        occupied.add(name)
        return name
    for index, node in enumerate(result.graph.node):
        if node.domain not in ("", "ai.onnx") or node.op_type != "Trilu":
            nodes.append(node)
            continue
        port = values.get(node.input[0])
        shape = [dim.dim_value if dim.HasField("dim_value") else None
                 for dim in port.type.tensor_type.shape.dim] if port is not None else None
        if shape is not None and 2 <= len(shape) <= 3:
            nodes.append(node)
            continue
        if (shape is None or not 4 <= len(shape) <= 8
                or any(dim is None or dim <= 0 for dim in shape[-2:])
                or any(dim is not None and not 0 < dim <= 2147483647 for dim in shape)
                or (all(dim is not None for dim in shape[:-2]) and math.prod(shape[:-2]) > 2147483647)
                or not 1 <= len(node.input) <= 2 or len(node.output) != 1):
            report["skipped"].append({"node": node.name, "shape": shape, "reason": "rank4..8 and static positive matrix dimensions required"})
            nodes.append(node)
            continue
        dynamic_batch = any(dim is None for dim in shape[:-2])
        flat_shape = [-1 if dynamic_batch else math.prod(shape[:-2]), *shape[-2:]]
        label = node.name or "trilu_" + str(index)
        flat, triangular = fresh(label + "__flat"), fresh(label + "__triangular")
        shape_in, shape_out = fresh(label + "__flat_shape"), fresh(label + "__original_shape")
        result.graph.initializer.append(numpy_helper.from_array(np.asarray(flat_shape, dtype=np.int64), shape_in))
        before, after = node.input[0], node.output[0]
        if dynamic_batch:
            nodes.append(helper.make_node("Shape", [before], [shape_out], name=fresh(label + "__runtime_shape")))
        else:
            result.graph.initializer.append(numpy_helper.from_array(np.asarray(shape, dtype=np.int64), shape_out))
        nodes.append(helper.make_node("Reshape", [before, shape_in], [flat], name=fresh(label + "__flatten_batch")))
        node.input[0], node.output[0] = flat, triangular
        nodes.append(node)
        nodes.append(helper.make_node("Reshape", [triangular, shape_out], [after], name=fresh(label + "__restore_batch")))
        report["rewritten"].append({"node": node.name, "source_shape": shape, "flattened_shape": flat_shape,
                                    "batch_inferred_from_runtime_volume": dynamic_batch,
                                    "diagonal_input_unchanged": True, "attributes_unchanged": True})
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    return result, report


def main():
    import onnx
    from export_torchscript_region_onnx import digest, external_files, publish_files
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(strict=True), args.output.absolute()
    report_path = output.with_suffix(".report.json")
    if source.parent != output.parent.resolve() or output.exists() or report_path.exists():
        raise ValueError("fresh sibling output required; external weights stay immutable")
    model = onnx.load_model(str(source), load_external_data=False)
    sidecars = external_files(model, source.parent)
    source_pin = digest(source)
    weights = [{"location": str(path), "bytes": (source.parent / path).stat().st_size,
                "sha256": digest(source.parent / path)} for path in sidecars]
    result, report = lower(model)
    with tempfile.NamedTemporaryFile(prefix=".trilu-", suffix=".onnx", dir=source.parent) as temporary:
        staged = Path(temporary.name)
        staged.write_bytes(result.SerializeToString())
        onnx.checker.check_model(str(staged))
        assert digest(source) == source_pin
        assert all(digest(source.parent / row["location"]) == row["sha256"] for row in weights)
        report.update(source_sha256=source_pin, output_sha256=digest(staged), external_data=weights,
                      generator_sha256=digest(Path(__file__)), target_execution_verified=False)
        publish_files(staged, output, [])
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"rewritten": len(report["rewritten"]), "skipped": len(report["skipped"])}))


if __name__ == "__main__":
    main()
