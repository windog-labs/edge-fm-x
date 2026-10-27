#!/usr/bin/env python3
"""Represent BF16 values as FLOAT with explicit BF16 rounding boundaries.

This does not certify a compute precision or numerical equivalence. Dense
precision, finite/range checks and complete source-output validation are
separate requirements. The portable reference uses BF16 Cast, while deployment
uses the optional TensorRT FLOAT-carrier rounding plugin.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

from export_torchscript_region_onnx import digest, external_files, publish_files


def reference_from_carrier(model):
    """Expand reviewed precision nodes; retain existing weight locations."""
    import onnx
    from onnx import helper

    result = copy.deepcopy(model)
    if result.functions or any(any(attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
        for attribute in node.attribute) for node in result.graph.node):
        raise ValueError("carrier reference requires a flat graph without local functions")
    occupied = {name for node in result.graph.node for name in [node.name, *node.input, *node.output]}
    operations = []

    def fresh(name):
        while name in occupied:
            name += "_"
        occupied.add(name)
        return name

    def cast(value, dtype, label):
        output = fresh(label)
        operations.append(helper.make_node("Cast", [value], [output], name=fresh(label + "_cast"), to=dtype))
        return output

    for node in result.graph.node:
        if node.domain in ("", "ai.onnx"):
            operations.append(node)
            continue
        attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        arity = {"VLAForgeBFloat16Round": 1, "VLAForgeBFloat16MatMul": 2}.get(node.op_type)
        if (node.domain != "vlaforge.trt" or arity is None
                or len(node.input) != arity or len(node.output) != 1
                or attributes != {"plugin_version": b"1", "plugin_namespace": b""}):
            raise ValueError("carrier reference encountered an unreviewed plugin or policy")
        raw = node.input[0]
        if arity == 2:
            operands = []
            for index, value in enumerate(node.input):
                label = node.output[0] + "__vf_portable_operand_" + str(index)
                bf16 = cast(value, onnx.TensorProto.BFLOAT16, label)
                operands.append(cast(bf16, onnx.TensorProto.FLOAT, label + "_float"))
            raw = fresh(node.output[0] + "__vf_portable_accumulation")
            operations.append(helper.make_node("MatMul", operands, [raw], name=fresh(raw + "_matmul")))
        middle = cast(raw, onnx.TensorProto.BFLOAT16, node.output[0] + "__vf_portable_bf16")
        operations.append(helper.make_node("Cast", [middle], list(node.output),
                          name=fresh(middle + "_up"), to=onnx.TensorProto.FLOAT))
    del result.graph.node[:]
    result.graph.node.extend(operations)
    imports = [entry for entry in result.opset_import if entry.domain != "vlaforge.trt"]
    del result.opset_import[:]
    result.opset_import.extend(imports)
    return result


def lower(model, *, plugin=True, external_base_dir=None, native_bf16_matmul=False):
    import numpy as np
    import onnx
    from onnx import external_data_helper, helper, numpy_helper

    if native_bf16_matmul and not plugin:
        raise ValueError("native BF16 MatMul requires deployment plugins; use its portable shadow for reference")
    if (model.functions or model.training_info or model.graph.sparse_initializer
            or any(node.domain not in ("", "ai.onnx")
                   or node.op_type in {"RandomNormal", "RandomNormalLike", "RandomUniform",
                                       "RandomUniformLike", "Bernoulli", "Multinomial", "Dropout",
                                       "QuantizeLinear", "DequantizeLinear"}
                   or any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                          for attr in node.attribute) for node in model.graph.node)):
        raise ValueError("BF16 carrier lowering requires a flat functional standard ONNX graph")
    result = onnx.shape_inference.infer_shapes(copy.deepcopy(model), strict_mode=True, data_prop=native_bf16_matmul)
    all_values = [*result.graph.input, *result.graph.output, *result.graph.value_info]
    values = {value.name: value for value in all_values}
    types = {name: value.type.tensor_type.elem_type for name, value in values.items()}
    types.update({tensor.name: tensor.data_type for tensor in result.graph.initializer})
    if any(name and not types.get(name) for node in result.graph.node for name in node.output):
        raise ValueError("every source tensor output must have an inferred element type")
    report = {"schema": "vlaforge.onnx_bfloat16_carrier/1", "status": "lowered-not-runtime-validated",
              "carrier": "FLOAT", "rounding": "BF16 round-to-nearest-even at arithmetic boundaries",
              "implementation": "VLAForgeBFloat16Round/1" if plugin else "standard BF16 Cast reference",
              "compute_precision_certified": False, "source_equivalence_verified": False,
              "full_model_verified": False, "integer_dtypes_changed": False,
              "promoted_tensors": 0, "rounding_boundaries": [], "input_rounding": [],
              "native_bf16_matmul": [], "native_bf16_matmul_skipped": [],
              "static_shape_data_propagation": native_bf16_matmul}
    native_nodes = set()
    if native_bf16_matmul:
        shapes = {name: tuple(d.dim_value if d.HasField("dim_value") else None
                             for d in value.type.tensor_type.shape.dim) for name, value in values.items()}
        shapes.update({tensor.name: tuple(tensor.dims) for tensor in result.graph.initializer})
        for index, node in enumerate(result.graph.node):
            if node.op_type != "MatMul":
                continue
            if len(node.input) != 2 or len(node.output) != 1 or any(
                    types.get(value) != onnx.TensorProto.BFLOAT16 for value in [*node.input, *node.output]):
                report["native_bf16_matmul_skipped"].append({"node": node.name, "reason": "not an all-BF16 source MatMul"})
                continue
            inputs = [shapes.get(value, ()) for value in node.input]
            if any(not 2 <= len(shape) <= 8 or any(d is None or not 0 < d <= 2147483647 for d in shape) for shape in inputs):
                raise ValueError("native BF16 MatMul requires static positive rank 2..8 inputs: " + node.name)
            rank = max(map(len, inputs))
            left, right = ((1,) * (rank - len(shape)) + shape for shape in inputs)
            if left[-1] != right[-2] or any(a != b and a != 1 and b != 1 for a, b in zip(left[:-2], right[:-2], strict=True)):
                raise ValueError("native BF16 MatMul has incompatible reduction/batch dimensions: " + node.name)
            native_nodes.add(index)
            report["native_bf16_matmul"].append({"node": node.name, "input_shapes": inputs,
                "implementation": "VLAForgeBFloat16MatMul/1", "compute": "BF16 operands, FLOAT accumulation, BF16 RNE result in FLOAT storage",
                "external_result_rounding_retained": True})

    def promote(tensor):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            if external_base_dir is None:
                raise ValueError("supply the external tensor base directory before BF16 carrier lowering")
            external_data_helper.load_external_data_for_tensor(tensor, str(external_base_dir))
            tensor.data_location = onnx.TensorProto.DEFAULT
            del tensor.external_data[:]
        if tensor.data_type != onnx.TensorProto.BFLOAT16:
            return
        data = numpy_helper.to_array(tensor).astype(np.float32)
        tensor.CopyFrom(numpy_helper.from_array(data, tensor.name))
        report["promoted_tensors"] += 1

    if external_base_dir is not None:
        external_files(result, Path(external_base_dir))
    for tensor in result.graph.initializer:
        promote(tensor)
    for value in all_values:
        if value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    occupied = set(types) | {node.name for node in result.graph.node}

    def unique(proposed):
        while proposed in occupied:
            proposed += "_"
        occupied.add(proposed)
        return proposed

    # These operations only select/copy already-rounded values or exact constants.
    bit_preserving = {"Identity", "Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze",
                      "Expand", "Concat", "Split", "Gather", "GatherElements", "GatherND",
                      "Where", "Tile", "Flatten", "Constant", "ConstantOfShape"}
    operations = []

    def append_rounding(raw, output, name):
        if plugin:
            operations.append(helper.make_node("VLAForgeBFloat16Round", [raw], [output],
                name=name, domain="vlaforge.trt", plugin_version="1", plugin_namespace=""))
        else:
            rounded = unique(output + "__vf_bf16_bits")
            operations.extend([
                helper.make_node("Cast", [raw], [rounded], name=name + "_down", to=onnx.TensorProto.BFLOAT16),
                helper.make_node("Cast", [rounded], [output], name=name + "_up", to=onnx.TensorProto.FLOAT)])

    aliases = {}
    for value in result.graph.input:
        if types[value.name] != onnx.TensorProto.BFLOAT16:
            continue
        if any(output.name == value.name for output in result.graph.output):
            raise ValueError("BF16 source input/output aliases require an explicit port adapter")
        canonical = unique(value.name + "__vf_bf16_input")
        aliases[value.name] = canonical
        info = copy.deepcopy(value)
        info.name = canonical
        result.graph.value_info.append(info)
        types[canonical] = onnx.TensorProto.BFLOAT16
        name = unique("vf_bf16_input_round_" + str(len(aliases) - 1))
        append_rounding(value.name, canonical, name)
        report["input_rounding"].append({"input": value.name, "canonical_value": canonical,
                                         "rounding_node": name})
    for node_index, node in enumerate(result.graph.node):
        source_operator = node.op_type
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
        if (any(types.get(name) == onnx.TensorProto.DOUBLE for name in node.input)
                and any(types.get(name) == onnx.TensorProto.BFLOAT16 for name in node.output)):
            raise ValueError("DOUBLE-to-BF16 requires a separately proven direct rounding implementation")
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                promote(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    promote(tensor)
            elif node.op_type == "Cast" and attr.name == "to" and attr.i == onnx.TensorProto.BFLOAT16:
                attr.i = onnx.TensorProto.FLOAT
            elif attr.name == "dtype" and attr.type == onnx.AttributeProto.INT and attr.i == onnx.TensorProto.BFLOAT16:
                raise ValueError("unhandled explicit BF16 dtype attribute: " + node.op_type)
        if node_index in native_nodes:
            node.op_type, node.domain = "VLAForgeBFloat16MatMul", "vlaforge.trt"
            del node.attribute[:]
            node.attribute.extend([helper.make_attribute("plugin_version", "1"), helper.make_attribute("plugin_namespace", "")])
        operations.append(node)
        if node.op_type in bit_preserving or (
                node.op_type == "Cast" and types.get(node.input[0]) == onnx.TensorProto.BFLOAT16):
            continue
        for index, output in enumerate(node.output):
            if types.get(output) != onnx.TensorProto.BFLOAT16:
                continue
            raw = unique(output + "__vf_before_bf16_round")
            node.output[index] = raw
            info = copy.deepcopy(values[output])
            info.name = raw
            result.graph.value_info.append(info)
            name = unique("vf_bf16_round_" + str(len(report["rounding_boundaries"])))
            append_rounding(raw, output, name)
            report["rounding_boundaries"].append({"producer": node.name, "operator": source_operator,
                                                   "value": output, "rounding_node": name})
    del result.graph.node[:]
    result.graph.node.extend(operations)
    if plugin and (report["rounding_boundaries"] or report["input_rounding"]):
        result.opset_import.append(helper.make_opsetid("vlaforge.trt", 1))
    report["source_nodes"], report["lowered_nodes"] = len(model.graph.node), len(operations)
    return result, report


def main():
    import onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true", help="Emit portable BF16 Cast reference, not TRT plugins")
    parser.add_argument("--native-bf16-matmul", action="store_true",
                        help="Use the optional native BF16 provider only for proven static BF16 MatMul nodes")
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = output.with_suffix(".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError("output or report already exists")
    metadata = onnx.load(str(args.input), load_external_data=False)
    source_sidecars = external_files(metadata, args.input.resolve().parent)
    source_external_data = [{"location": str(path), "size": (args.input.parent / path).stat().st_size,
                             "sha256": digest(args.input.parent / path)} for path in source_sidecars]
    # Shape inference runs before materializing weights, so its protobuf remains small.
    result, report = lower(metadata, plugin=not args.reference, external_base_dir=args.input.resolve().parent,
                           native_bf16_matmul=args.native_bf16_matmul)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bf16-carrier-", dir=output.parent) as directory:
        staged = Path(directory) / output.name
        onnx.save_model(result, str(staged), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=output.name + ".data", size_threshold=1024,
                        convert_attribute=True)
        metadata = onnx.load(str(staged), load_external_data=False)
        sidecars = external_files(metadata, staged.parent)
        onnx.checker.check_model(str(staged))
        report.update(source=str(args.input.resolve()), source_sha256=digest(args.input),
                      source_external_data=source_external_data,
                      generator_sha256=digest(Path(__file__)), output=str(output),
                      output_sha256=digest(staged),
                      external_data=[{"location": str(path), "size": (staged.parent / path).stat().st_size,
                                      "sha256": digest(staged.parent / path)} for path in sidecars])
        published = publish_files(staged, output, sidecars)
        try:
            with report_path.open("x") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        except Exception:
            for path in reversed(published):
                path.unlink()
            raise
    print(json.dumps({"output": str(output), "rounding_boundaries": len(report["rounding_boundaries"]),
                      "compute_precision_certified": False, "source_equivalence_verified": False}))


if __name__ == "__main__":
    main()
