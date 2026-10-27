#!/usr/bin/env python3
"""Prepare explicit child-graph input contracts; never implement implicit casts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile


def prepare(model, *, drop_unused=False, externalize_casts=(), narrow_gather_indices=()):
    import onnx

    if (model.functions or model.training_info or any(
            attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for attr in node.attribute)):
        raise ValueError("flat child graph without implicit captures required")
    if (len(set(externalize_casts)) != len(externalize_casts)
            or len(set(narrow_gather_indices)) != len(narrow_gather_indices)):
        raise ValueError("input requests must be unique")
    result = copy.deepcopy(model)
    inputs = list(result.graph.input)
    names = [value.name for value in inputs]
    if len(set(names)) != len(names):
        raise ValueError("unique source inputs required")
    initializers = {value.name for value in result.graph.initializer}
    outputs = {value.name for value in result.graph.output}
    info = {value.name: value for value in [*result.graph.input, *result.graph.output, *result.graph.value_info]}
    nodes = list(result.graph.node)
    if len({name for node in nodes for name in node.output}) != sum(len(node.output) for node in nodes):
        raise ValueError("source nodes require unique SSA results")
    casts, narrowed, removed = [], [], set()
    routes = [{"source": "input", "index": index, "name": value.name} for index, value in enumerate(inputs)]
    for selected in externalize_casts:
        matches = [node for node in nodes if selected in node.output]
        if len(matches) != 1:
            raise ValueError("one explicit Cast producer required")
        node = matches[0]
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        if (node.op_type != "Cast" or node.domain not in ("", "ai.onnx")
                or len(node.input) != 1 or len(node.output) != 1 or set(attrs) != {"to"}
                or node.input[0] not in names or node.input[0] in initializers):
            raise ValueError("external Cast must consume one non-overridable graph input")
        index = names.index(node.input[0])
        source = inputs[index]
        if selected not in info:
            raise ValueError("Cast output requires an explicit tensor contract")
        target = info[selected]
        src, dst = source.type.tensor_type, target.type.tensor_type
        if ((src.elem_type, dst.elem_type) not in {
                (onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT),
                (onnx.TensorProto.INT64, onnx.TensorProto.INT32)}
                or attrs["to"] != dst.elem_type or not src.HasField("shape")
                or not dst.HasField("shape") or src.shape != dst.shape):
            raise ValueError("only declared same-shape F64/F32 or I64/I32 Cast boundaries are supported")
        if source.name in outputs or any(
                source.name in other.input for other in nodes if other.SerializeToString() != node.SerializeToString()):
            raise ValueError("source input has another observer; cannot remove it")
        entry = {"source_input": source.name, "source_input_index": index,
                 "target_input": selected, "source_dtype": onnx.TensorProto.DataType.Name(src.elem_type),
                 "target_dtype": onnx.TensorProto.DataType.Name(dst.elem_type),
                 "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in src.shape.dim],
                 "source_cast_sha256": hashlib.sha256(node.SerializeToString()).hexdigest(),
                 "checked_runtime_cast_required": True, "hoisting_proved": False}
        casts.append(entry)
        routes[index] = {"source": "external-cast", "index": len(casts) - 1, "name": selected}
        inputs[index] = copy.deepcopy(target)
        removed.add(source.name)
        names[index] = selected
        nodes = [other for other in nodes if other.SerializeToString() != node.SerializeToString()]
    for name in narrow_gather_indices:
        if name not in names or name in initializers:
            raise ValueError("non-overridable index input required")
        index = names.index(name)
        value = inputs[index]
        if value.type.WhichOneof("value") != "tensor_type" or value.type.tensor_type.elem_type != onnx.TensorProto.INT64:
            raise ValueError("original index input must be INT64")
        consumers = [node for node in nodes if name in node.input]
        if not consumers or name in outputs or any(
                node.domain not in ("", "ai.onnx") or node.op_type != "Gather" or len(node.input) != 2
                or [position for position, operand in enumerate(node.input) if operand == name] != [1]
                for node in consumers):
            raise ValueError("narrowed input must be used only as standard Gather indices")
        value.type.tensor_type.elem_type = onnx.TensorProto.INT32
        narrowed.append({"name": name, "source_input_index": index,
                         "source_dtype": "INT64", "target_dtype": "INT32",
                         "range": [-2147483648, 2147483647], "checked_runtime_cast_required": True,
                         "consumer_nodes": [node.name for node in consumers]})
    live = outputs | {operand for node in nodes for operand in node.input if operand}
    keep = [index for index, value in enumerate(inputs) if not drop_unused or value.name in live]
    dropped = [{"name": value.name, "source_input_index": index}
               for index, value in enumerate(inputs) if index not in keep]
    removed.update(value["name"] for value in dropped)
    del result.graph.input[:]
    result.graph.input.extend(inputs[index] for index in keep)
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    kept_names = {value.name for value in result.graph.input}
    retained_info = [value for value in result.graph.value_info if value.name not in removed | kept_names]
    del result.graph.value_info[:]
    result.graph.value_info.extend(retained_info)
    if [value.SerializeToString() for value in result.graph.output] != [value.SerializeToString() for value in model.graph.output]:
        raise ValueError("output contract changed")
    if [value.SerializeToString() for value in result.graph.initializer] != [value.SerializeToString() for value in model.graph.initializer]:
        raise ValueError("source weights changed")
    return result, {"schema": "vlaforge.explicit_onnx_input_preparation/1",
                    "status": "input-contract-prepared-not-runtime-validated", "external_casts": casts,
                    "narrowed_gather_indices": narrowed, "dropped_unused_inputs": dropped,
                    "input_routes": [routes[index] for index in keep],
                    "input_names": [value.name for value in result.graph.input],
                    "source_nodes": len(model.graph.node), "prepared_nodes": len(nodes),
                    "all_other_source_nodes_unchanged": True, "all_initializer_metadata_unchanged": True,
                    "output_contract_unchanged": True, "runtime_boundaries_implemented": False,
                    "target_execution_verified": False, "source_numerical_parity_verified": False}


def main():
    import onnx
    from export_torchscript_region_onnx import digest, external_files, publish_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-unused", action="store_true")
    parser.add_argument("--externalize-cast", action="append", default=[])
    parser.add_argument("--narrow-gather-index", action="append", default=[])
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    output = args.output.absolute()
    report_path = output.with_suffix(".report.json")
    if source.parent != output.parent.resolve() or os.path.lexists(output) or os.path.lexists(report_path):
        raise ValueError("choose fresh graph/report names beside the immutable source weights")
    pin = digest(source)
    model = onnx.load(source, load_external_data=False)
    if digest(source) != pin:
        raise ValueError("source graph changed while loading")
    onnx.checker.check_model(str(source))
    sidecars = external_files(model, source.parent)
    weights = [{"location": str(path), "sha256": digest(source.parent / path),
                "bytes": (source.parent / path).stat().st_size} for path in sidecars]
    result, report = prepare(model, drop_unused=args.drop_unused, externalize_casts=args.externalize_cast,
                             narrow_gather_indices=args.narrow_gather_index)
    with tempfile.NamedTemporaryFile(prefix=".trt-inputs-", suffix=".onnx", dir=source.parent) as temporary:
        staged = Path(temporary.name)
        onnx.save_model(result, staged)
        onnx.checker.check_model(str(staged))
        if digest(source) != pin or any(digest(source.parent / row["location"]) != row["sha256"] for row in weights):
            raise ValueError("immutable graph/weights changed during preparation")
        report.update(source=str(source), source_sha256=pin, output=str(output),
                      output_sha256=digest(staged), shared_immutable_external_data=weights,
                      generator_sha256=digest(Path(__file__)))
        publish_files(staged, output, [])
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
