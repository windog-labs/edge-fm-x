#!/usr/bin/env python3
"""Extract an explicit FLOAT arithmetic tail without rewriting its formula."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def partition(model, output_names=None):
    import onnx
    from onnx import helper
    from onnx.utils import Extractor

    if model.functions or model.training_info or any(
        node.domain not in ("", "ai.onnx") or node.op_type in {
            "RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Bernoulli", "Multinomial", "Dropout"}
        or any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attr in node.attribute)
        for node in model.graph.node
    ):
        raise ValueError("partition requires a reviewed functional flat standard ONNX graph")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    info = {value.name: value for value in [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    constants = {value.name for value in inferred.graph.initializer}
    inputs = [value.name for value in inferred.graph.input]
    outputs = [value.name for value in inferred.graph.output]
    producers = {name: index for index, node in enumerate(inferred.graph.node) for name in node.output}
    selected = list(output_names) if output_names is not None else [
        name for name in outputs if info[name].type.tensor_type.elem_type == onnx.TensorProto.FLOAT]
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(outputs):
        raise ValueError("select unique declared FLOAT graph outputs")
    arithmetic = {"Add", "Sub", "Mul", "Div"}
    tail_nodes, raw_inputs, frontier = set(), set(), set()

    def float_type(name):
        return name in info and info[name].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    def collect(name):
        if name in constants:
            return
        if name in inputs:
            raw_inputs.add(name)
            return
        if name not in producers:
            raise ValueError("tail has an unresolved dependency: " + name)
        index = producers[name]
        node = inferred.graph.node[index]
        if node.op_type in arithmetic and all(float_type(value) or value in constants for value in node.input) and all(float_type(value) for value in node.output):
            if index not in tail_nodes:
                tail_nodes.add(index)
                for value in node.input:
                    collect(value)
        else:
            frontier.add(name)

    for name in selected:
        if not float_type(name):
            raise ValueError("selected tail output is not FLOAT")
        collect(name)
    if not tail_nodes or not frontier or not any(float_type(name) for name in raw_inputs):
        raise ValueError("tail requires original FLOAT input and a produced neural frontier")
    if any(name in frontier for name in selected):
        raise ValueError("selected output is not an extractable FLOAT arithmetic tail")
    ordered_frontier = [name for node in inferred.graph.node for name in node.output if name in frontier]
    passthrough = [name for name in outputs if name not in selected]
    core_outputs = list(dict.fromkeys(ordered_frontier + [name for name in passthrough if name not in inputs]))
    tail_inputs = [name for name in inputs if name in raw_inputs or name in passthrough]
    tail_inputs += [name for name in core_outputs if name not in tail_inputs]
    for name in core_outputs + tail_inputs + outputs:
        if name not in info or any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in info[name].type.tensor_type.shape.dim):
            raise ValueError("partition interface requires known positive static tensor dimensions")
    extractor = Extractor(inferred)
    core = extractor.extract_model(inputs, core_outputs)
    tail = extractor.extract_model(tail_inputs, outputs)
    core_names = {tuple(node.output) for node in core.graph.node}
    if any(tuple(inferred.graph.node[index].output) in core_names for index in tail_nodes):
        raise ValueError("partition would duplicate selected arithmetic in both graphs")
    expected_tail = [copy.deepcopy(inferred.graph.node[index]) for index in sorted(tail_nodes)]
    if [node.SerializeToString() for node in tail.graph.node] != [node.SerializeToString() for node in expected_tail]:
        raise ValueError("extracted tail changed the selected original nodes")
    routes = []
    occupied = set(info) | constants | set(producers)
    for index, value in enumerate(tail.graph.input):
        old = value.name
        route = {"source": "input", "index": inputs.index(old)} if old in inputs else {
            "source": "first", "index": core_outputs.index(old)}
        if old in outputs:
            candidate = "vf_fp32_tail_passthrough_" + str(index)
            while candidate in occupied:
                candidate += "_"
            occupied.add(candidate)
            value.name = candidate
            for node in tail.graph.node:
                for position, name in enumerate(node.input):
                    if name == old:
                        node.input[position] = candidate
            tail.graph.node.append(helper.make_node("Identity", [candidate], [old], name=candidate + "_identity"))
        routes.append({"name": value.name, **route})
    core.graph.name += "__numeric_core"
    tail.graph.name += "__fp32_tail"
    onnx.checker.check_model(core)
    onnx.checker.check_model(tail)
    return core, tail, {"schema": "vlaforge.onnx_fp32_tail_partition/1", "status": "partitioned-not-runtime-validated",
        "source_inputs": inputs, "source_outputs": outputs, "selected_outputs": selected,
        "core_inputs": [value.name for value in core.graph.input], "core_outputs": core_outputs,
        "tail_inputs": routes, "tail_outputs": [value.name for value in tail.graph.output],
        "original_tail_nodes": [inferred.graph.node[index].name for index in sorted(tail_nodes)],
        "source_nodes": len(model.graph.node), "core_nodes": len(core.graph.node), "tail_nodes": len(tail.graph.node),
        "formula_rewritten": False, "source_tail_node_bytes_exact": True,
        "arithmetic_node_duplication": False, "compiled_artifact_verified": False}


def main():
    import onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-name", action="append")
    args = parser.parse_args()
    core, tail, report = partition(onnx.load(args.input), args.output_name)
    args.output.mkdir(parents=True, exist_ok=False)
    onnx.save(core, args.output / "core.onnx")
    onnx.save(tail, args.output / "tail.onnx")
    report.update(source=str(args.input.resolve()), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files={name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in ("core.onnx", "tail.onnx")})
    (args.output / "partition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
