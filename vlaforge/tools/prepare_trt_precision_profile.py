#!/usr/bin/env python3
"""Declare FP16 dense compute and FP32 arithmetic for typed ONNX graphs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def build_profile(model, *, dense_output_type="fp32", fp32_dense_nodes=()):
    import onnx

    if dense_output_type not in ("fp32", "fp16"):
        raise ValueError("dense output type must be fp32 or fp16")
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    types = {value.name: value.type.tensor_type.elem_type for value in
             [*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info]}
    types.update({value.name: value.data_type for value in inferred.graph.initializer})
    names = [node.name for node in model.graph.node if node.name]
    if len(set(names)) != len(names):
        raise ValueError("TensorRT precision selection requires unique node names")
    dense = {"Conv", "MatMul", "Gemm"}
    exceptions = set(fp32_dense_nodes)
    declared_dense = {node.name for node in model.graph.node if not node.domain and node.op_type in dense}
    if exceptions - declared_dense:
        raise ValueError("FP32 dense exceptions must name exact existing dense nodes")
    arithmetic = {"Add", "Sub", "Mul", "Div", "Pow", "ReduceMean", "ReduceMin", "ReduceMax",
                  "ReduceSum", "Sqrt", "Sin", "Cos", "Tanh", "Softmax", "Sigmoid", "Reciprocal", "Abs",
                  "LayerNormalization", "Where", "Neg", "Clip", "Exp", "Softplus"}
    comparisons = {"Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"}
    precisions, outputs, skipped = {}, {}, []
    counts = Counter()
    for node in model.graph.node:
        normalization_plugin = node.domain == "trt.plugins" and node.op_type == "LayerNorm"
        if normalization_plugin:
            attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
            if attributes.get("plugin_version") != b"1" or attributes.get("plugin_namespace") != b"":
                raise ValueError("unreviewed LayerNorm plugin precision contract")
        if not normalization_plugin and (node.domain or node.op_type not in dense | arithmetic | comparisons):
            continue
        kinds = [types.get(value) for value in node.output]
        if (not node.name or any(character in node.name for character in ",:*?\n\r")
                or any(kind is None for kind in kinds)):
            raise ValueError("precision selection requires exact names and known output types")
        comparison = node.op_type in comparisons and any(types.get(value) == onnx.TensorProto.FLOAT for value in node.input)
        if kinds != [onnx.TensorProto.FLOAT] and not comparison:
            skipped.append({"name": node.name, "op": node.op_type, "output_types": kinds})
            continue
        precision = "fp16" if node.op_type in dense and node.name not in exceptions else "fp32"
        precisions[node.name] = precision
        if kinds == [onnx.TensorProto.FLOAT]:
            outputs[node.name] = dense_output_type if node.op_type in dense and node.name not in exceptions else "fp32"
        counts[precision + "_" + node.op_type] += 1
    return {"schema": "edgefm.trt_precision_profile/1", "policy": (
            "fp16_dense_fp32_arithmetic_declared_outputs/1" if dense_output_type == "fp32"
            else "fp16_dense_fp16_outputs_fp32_arithmetic/1"),
        "dense_output_type": dense_output_type,
        "fp32_dense_exceptions": sorted(exceptions),
        "dense_result_rounding": ("FLOAT output requested" if dense_output_type == "fp32"
                                  else "additional FP16 result rounding before FLOAT consumers; must validate complete outputs"),
        "scope": "selected FLOAT-producing operators only; no integer cast, quantization search or full-FP32 request",
        "layer_precisions": precisions, "layer_output_types": outputs,
        "skipped_nonfloating_or_non_float32": skipped, "counts": dict(counts),
        "correctness_verified": False,
        "verification_required": "builder must obey constraints; inspect engine placement and gate complete outputs"}


def main():
    import onnx

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense-output-type", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--fp32-dense-node", action="append", default=[],
                        help="Exact dense node with an independently observed precision/range requirement")
    args = parser.parse_args()
    model = onnx.load(args.input, load_external_data=False)
    record = build_profile(model, dense_output_type=args.dense_output_type, fp32_dense_nodes=args.fp32_dense_node)
    record.update(source=str(args.input.resolve()), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"profile": str(args.output), "counts": record["counts"],
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
