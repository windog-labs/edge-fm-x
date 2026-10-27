#!/usr/bin/env python3
"""Lower FLOAT IsNaN/IsInf to ordered comparisons, preserving non-finite classes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import os


def lower(model, *, operators=("IsNaN", "IsInf")):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    selected = set(operators)
    if selected - {"IsNaN", "IsInf"}:
        raise ValueError("only explicit IsNaN/IsInf predicate rewrites are supported")
    if model.functions or any(any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                                  for attribute in node.attribute) for node in model.graph.node):
        raise ValueError("FLOAT predicate lowering requires a flat graph without local functions")
    result = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True)
    kinds = {value.name: value.type.tensor_type.elem_type for value in
             [*result.graph.input, *result.graph.output, *result.graph.value_info]}
    kinds.update({value.name: value.data_type for value in result.graph.initializer})
    producers = {name: node for node in result.graph.node for name in node.output}
    occupied = set(kinds) | {value for node in result.graph.node for value in [node.name, *node.input, *node.output]}

    def unique(name):
        while name in occupied:
            name += "_"
        occupied.add(name)
        return name

    report = {"schema": "vlaforge.onnx_float_predicates/1", "status": "lowered-not-target-validated",
              "source_equivalence_verified": False, "full_model_verified": False, "rewritten": [],
              "selected_operators": sorted(selected),
              "classification_widening_casts": [],
              "required_precision": "FLOAT inputs and full-range bounds must reach comparisons unchanged; independently validate exact target flags",
              "reduced_precision_caution": "Layer FP32 constraints alone may not preserve full-range constants; isolate when target conformance requires it"}
    constants = {}

    def constant(label, numeric):
        if label not in constants:
            name = unique("vf_predicate_" + label)
            constants[label] = name
            result.graph.initializer.append(numpy_helper.from_array(np.asarray(numeric, np.float32), name))
        return constants[label]

    operations = []
    bypassed = set()
    for index, node in enumerate(result.graph.node):
        if node.domain not in ("", "ai.onnx") or node.op_type not in selected:
            operations.append(node)
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            raise ValueError("non-finite predicate requires an explicit FLOAT input")
        attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        prefix = unique("vf_predicate_" + str(index))
        x, output = node.input[0], node.output[0]
        if kinds.get(x) == onnx.TensorProto.DOUBLE:
            producer = producers.get(x)
            attrs = ({attribute.name: helper.get_attribute_value(attribute) for attribute in producer.attribute}
                     if producer is not None else {})
            if (producer is None or producer.domain not in ("", "ai.onnx") or producer.op_type != "Cast"
                    or len(producer.input) != 1 or attrs != {"to": onnx.TensorProto.DOUBLE}
                    or kinds.get(producer.input[0]) != onnx.TensorProto.FLOAT):
                raise ValueError("non-finite predicate requires an explicit FLOAT input or a proven exact widening cast")
            report["classification_widening_casts"].append({"node": producer.name, "from": "FLOAT", "to": "DOUBLE",
                "classification_only": True, "value": x})
            bypassed.add(x)
            x = producer.input[0]
        if kinds.get(x) != onnx.TensorProto.FLOAT:
            raise ValueError("non-finite predicate requires an explicit FLOAT input")

        def emit(kind, inputs, destination=None):
            target = destination if destination is not None else unique(prefix + "_" + kind.lower())
            operations.append(helper.make_node(kind, inputs, [target], name=unique(prefix + "_" + kind.lower() + "_node")))
            return target

        if node.op_type == "IsNaN":
            if attributes:
                raise ValueError("unexpected IsNaN attributes")
            zero = constant("zero", 0)
            smaller = emit("Less", [x, zero])
            equal = emit("Equal", [x, zero])
            greater = emit("Greater", [x, zero])
            ordered = emit("Or", [emit("Or", [smaller, equal]), greater])
            emit("Not", [ordered], output)
        else:
            if (set(attributes) - {"detect_positive", "detect_negative"}
                    or any(value not in (0, 1) for value in attributes.values())):
                raise ValueError("unreviewed IsInf detection flags")
            positive, negative = attributes.get("detect_positive", 1), attributes.get("detect_negative", 1)
            if positive and negative:
                emit("Greater", [emit("Abs", [x]), constant("maximum", np.finfo(np.float32).max)], output)
            elif positive:
                emit("Greater", [x, constant("maximum", np.finfo(np.float32).max)], output)
            elif negative:
                emit("Less", [x, constant("minimum", -np.finfo(np.float32).max)], output)
            else:
                emit("Less", [emit("Abs", [x]), constant("zero", 0)], output)
        report["rewritten"].append({"operator": node.op_type, "source_node": node.name, "output": output})
    used = {value.name for value in result.graph.output} | {name for node in operations for name in node.input}
    removed = {name for name in bypassed if name not in used}
    del result.graph.node[:]
    result.graph.node.extend(node for node in operations if not (set(node.output) & removed))
    information = [value for value in result.graph.value_info if value.name not in removed]
    del result.graph.value_info[:]
    result.graph.value_info.extend(information)
    report["dead_widening_cast_outputs_removed"] = sorted(removed)
    return result, report


def main():
    import onnx
    from export_torchscript_region_onnx import digest, external_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--operator", action="append", choices=("IsNaN", "IsInf"),
                        help="Rewrite only the operators known unsupported by the selected target")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("predicate lowering never overwrites evidence")
    model = onnx.load(str(args.input), load_external_data=False)
    sidecars = external_files(model, args.input.resolve().parent)
    if sidecars and args.input.resolve().parent != args.output.resolve().parent:
        raise ValueError("external-weight predicate rewrites require a sibling output graph")
    result, report = lower(model, operators=args.operator if args.operator is not None else ("IsNaN", "IsInf"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".float-predicates-", suffix=".onnx", dir=args.output.parent) as staged:
        onnx.save_model(result, staged.name)
        onnx.checker.check_model(staged.name)
        os.link(staged.name, args.output)
    try:
        report.update(source_sha256=digest(args.input), output=str(args.output.resolve()),
            output_sha256=digest(args.output), generator_sha256=digest(Path(__file__)),
            external_data=[{"location": str(path), "size": (args.output.parent / path).stat().st_size,
                            "sha256": digest(args.output.parent / path)} for path in sidecars])
        with args.report.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    except Exception:
        args.output.unlink()
        raise
    print(json.dumps({"rewritten": len(report["rewritten"]), "output_sha256": report["output_sha256"]}))


if __name__ == "__main__":
    main()
