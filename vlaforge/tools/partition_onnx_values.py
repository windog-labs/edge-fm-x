#!/usr/bin/env python3
"""Partition a functional ONNX graph at a dependency-closed producer module."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


def _has_external_data(model):
    import onnx

    tensors = list(model.graph.initializer)
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attribute.tensors)
    return any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in tensors)


def _check_model(model, external_base_dir):
    import onnx

    if not _has_external_data(model):
        onnx.checker.check_model(model)
        return
    if external_base_dir is None:
        raise ValueError("external ONNX weights require their explicit base directory")
    from export_torchscript_region_onnx import external_files

    root = Path(external_base_dir).resolve(strict=True)
    external_files(model, root)
    # Check metadata beside its original weights without creating hard links;
    # modern ONNX checkers intentionally reject multiply-linked weight files.
    with tempfile.NamedTemporaryFile(prefix=".partition-check-", suffix=".onnx", dir=root) as staged:
        Path(staged.name).write_bytes(model.SerializeToString())
        onnx.checker.check_model(staged.name)


def partition(model, values, *, external_base_dir=None, trusted_plugins=(), hoist_constants=False):
    import onnx
    from onnx import helper
    from onnx.utils import Extractor

    if model.functions or model.training_info or any(
        (node.domain not in ("", "ai.onnx") and not trusted_plugins) or node.op_type in {
            "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Bernoulli", "Multinomial", "Dropout"}
        or any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attr in node.attribute)
        for node in model.graph.node
    ):
        raise ValueError("producer partition requires functional flat standard ONNX")
    prepared = copy.deepcopy(model)
    hoists = []
    if hoist_constants:
        occupied = {v.name for v in [*prepared.graph.input, *prepared.graph.initializer]}
        retained = []
        for node in prepared.graph.node:
            if node.domain in ("", "ai.onnx") and node.op_type == "Constant":
                if (node.input or len(node.output) != 1 or len(node.attribute) != 1
                        or node.attribute[0].name != "value" or node.attribute[0].type != onnx.AttributeProto.TENSOR
                        or not node.output[0] or node.output[0] in occupied):
                    raise ValueError("literal hoisting requires canonical unique tensor Constants")
                tensor = copy.deepcopy(node.attribute[0].t)
                tensor.name = node.output[0]
                prepared.graph.initializer.append(tensor)
                occupied.add(tensor.name)
                hoists.append({"node": node.name, "output": tensor.name,
                               "source_node_sha256": hashlib.sha256(node.SerializeToString()).hexdigest(),
                               "dtype": tensor.data_type, "shape": list(tensor.dims)})
            else:
                retained.append(node)
        prepared.graph.ClearField("node")
        prepared.graph.node.extend(retained)
    shape_proof = None
    if trusted_plugins:
        spec = importlib.util.spec_from_file_location("_vf_reviewed_plugin_shapes", Path(__file__).with_name("onnx_trt_plugin_shapes.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inferred, shape_proof = module.infer_reviewed_shapes(prepared, trusted_plugins)
    else:
        inferred = onnx.shape_inference.infer_shapes(prepared, strict_mode=True)
    info = {item.name: item for item in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    inputs = [item.name for item in inferred.graph.input]
    outputs = [item.name for item in inferred.graph.output]
    constants = {item.name for item in inferred.graph.initializer}
    producers = {value: index for index, node in enumerate(inferred.graph.node) for value in node.output}
    selected = tuple(values)
    if not selected or len(set(selected)) != len(selected) or any(value not in producers for value in selected):
        raise ValueError("select unique produced tensor values")
    head_nodes = set()

    pending = list(selected)
    while pending:
        value = pending.pop()
        if value in inputs or value in constants:
            continue
        if value not in producers:
            raise ValueError("producer dependency cannot be resolved")
        index = producers[value]
        if index in head_nodes:
            continue
        head_nodes.add(index)
        pending.extend(name for name in inferred.graph.node[index].input if name)
    if len(head_nodes) == len(inferred.graph.node):
        raise ValueError("producer partition would absorb the complete model")
    consumed = set(outputs)
    for index, node in enumerate(inferred.graph.node):
        if index not in head_nodes:
            consumed.update(name for name in node.input if name)
    frontier = [name for index, node in enumerate(inferred.graph.node) if index in head_nodes
                for name in node.output if name in consumed]
    if not set(selected) <= set(frontier):
        raise ValueError("selected values must be consumed outside the producer module")
    for name in frontier:
        if (name not in info or info[name].type.WhichOneof("value") != "tensor_type"
                or not info[name].type.tensor_type.HasField("shape") or not info[name].type.tensor_type.elem_type
                or any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in info[name].type.tensor_type.shape.dim)):
            raise ValueError("producer interface requires static positive tensor dimensions")
    extractor = Extractor(inferred)
    first = extractor.extract_model(inputs, frontier)
    second = extractor.extract_model(inputs + frontier, outputs)
    source_initializers = {tensor.name: tensor for tensor in inferred.graph.initializer}
    retained_output_initializers = {}
    for label, part in (("producer", first), ("consumer", second)):
        defined = {value.name for value in part.graph.input} | {tensor.name for tensor in part.graph.initializer}
        defined.update(value for node in part.graph.node for value in node.output)
        retained = []
        # Extractor does not retain a direct initializer output with no node consumer.
        for value in part.graph.output:
            if value.name not in defined and value.name in source_initializers:
                part.graph.initializer.append(copy.deepcopy(source_initializers[value.name]))
                defined.add(value.name)
                retained.append(value.name)
        retained_output_initializers[label] = retained
    original_first = {tuple(node.output) for node in first.graph.node}
    if any(tuple(node.output) in original_first for node in second.graph.node):
        raise ValueError("producer partition duplicated a computation")
    routes, aliases = [], {}
    occupied = set(info) | constants | set(producers)
    for index, value in enumerate(second.graph.input):
        name = value.name
        route = {"source": "input", "index": inputs.index(name)} if name in inputs else {
            "source": "first", "index": frontier.index(name)}
        if name in outputs:
            alias = "vf_producer_passthrough_" + str(index)
            while alias in occupied:
                alias += "_"
            occupied.add(alias)
            aliases[name] = alias
            value.name = alias
            for node in second.graph.node:
                for position, operand in enumerate(node.input):
                    if operand == name:
                        node.input[position] = alias
            second.graph.node.append(helper.make_node("Identity", [alias], [name], name=alias + "_identity"))
        routes.append({"name": value.name, **route})
    first.graph.name += "__producer"
    second.graph.name += "__consumer"
    _check_model(first, external_base_dir)
    _check_model(second, external_base_dir)
    return first, second, {"schema": "vlaforge.onnx_producer_partition/1", "status": "partitioned-not-runtime-validated",
        "selected_values": list(selected), "source_inputs": inputs, "source_outputs": outputs,
        "first_inputs": inputs, "first_outputs": frontier, "second_inputs": routes,
        "second_outputs": outputs, "passthrough_aliases": aliases,
        "source_nodes": len(model.graph.node), "first_nodes": len(first.graph.node), "second_nodes": len(second.graph.node),
        "first_operator_counts": {op: sum(node.op_type == op for node in first.graph.node) for op in sorted({node.op_type for node in first.graph.node})},
        "computation_duplication": False, "formula_rewritten": False, "compiled_artifact_verified": False,
        "constant_hoists": hoists, "shape_proof": shape_proof,
        "retained_output_initializers": retained_output_initializers}


def main():
    import onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value-name", action="append")
    parser.add_argument("--operator", action="append")
    parser.add_argument("--trusted-plugin", action="append", default=[])
    parser.add_argument("--hoist-constants", action="store_true")
    args = parser.parse_args()
    model = onnx.load(args.input, load_external_data=False)
    selected = list(args.value_name or ())
    for node in model.graph.node:
        if node.op_type in (args.operator or ()):
            selected.extend(node.output)
    first, second, report = partition(model, selected, external_base_dir=args.input.resolve().parent,
                                     trusted_plugins=args.trusted_plugin, hoist_constants=args.hoist_constants)
    args.output.mkdir(parents=True, exist_ok=False)
    from export_torchscript_region_onnx import digest, external_files

    sidecars = sorted(set(external_files(first, args.input.resolve().parent))
                      | set(external_files(second, args.input.resolve().parent)))
    for relative in sidecars:
        if relative in (Path("producer.onnx"), Path("consumer.onnx"), Path("partition.json")):
            raise ValueError("external weight path aliases a partition output")
        destination = args.output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.input.resolve().parent / relative, destination)
    onnx.save(first, args.output / "producer.onnx")
    onnx.save(second, args.output / "consumer.onnx")
    onnx.checker.check_model(str(args.output / "producer.onnx"))
    onnx.checker.check_model(str(args.output / "consumer.onnx"))
    report.update(source=str(args.input.resolve()), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files={name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in ("producer.onnx", "consumer.onnx")},
                  external_data=[{"location": str(path), "bytes": (args.output / path).stat().st_size,
                                  "sha256": digest(args.output / path)} for path in sidecars])
    (args.output / "partition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
