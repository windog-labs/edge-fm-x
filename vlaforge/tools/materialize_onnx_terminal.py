#!/usr/bin/env python3
"""Emit model-neutral FLOAT slice/affine stages with complete finite guards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def materialize(spec: dict):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    shape = tuple(spec["shape"])
    if not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("terminal stages require a positive static FLOAT shape")
    nodes, initializers = [], []

    def constant(name, value):
        initializers.append(numpy_helper.from_array(np.asarray(value), name=name))
        return name

    def tensor(name, dtype, dimensions):
        return helper.make_tensor_value_info(name, dtype, dimensions)

    def finite(value, name):
        maximum = constant(name + "_max", np.asarray(np.finfo(np.float32).max, np.float32))
        nodes.extend([
            helper.make_node("Abs", [value], [name + "_abs"], name=name + "_abs"),
            helper.make_node("LessOrEqual", [name + "_abs", maximum], [name + "_finite"], name=name + "_compare"),
            helper.make_node("Cast", [name + "_finite"], [name + "_integer"], name=name + "_cast", to=onnx.TensorProto.INT32),
            helper.make_node("ReduceMin", [name + "_integer"], [name + "_all"], name=name + "_all", keepdims=0),
            helper.make_node("Cast", [name + "_all"], [name + "_bool"], name=name + "_bool", to=onnx.TensorProto.BOOL),
            helper.make_node("Reshape", [name + "_bool", one_shape], [name], name=name + "_reshape"),
        ])
        return name

    one_shape = constant("one_shape", np.asarray([1], np.int64))
    inputs = [tensor("input_0", onnx.TensorProto.FLOAT, shape)]
    kind = spec["kind"]
    if kind == "slice-finite":
        width = spec["width"]
        if type(width) is not int or not 0 < width <= shape[-1]:
            raise ValueError("slice width is outside the last-axis profile")
        starts = constant("starts", np.asarray([0], np.int64))
        ends = constant("ends", np.asarray([width], np.int64))
        axes = constant("axes", np.asarray([len(shape) - 1], np.int64))
        steps = constant("steps", np.asarray([1], np.int64))
        nodes.append(helper.make_node("Slice", ["input_0", starts, ends, axes, steps], ["output_0"], name="trim"))
        accepted = finite("input_0", "complete_input_finite")
        result_shape = (*shape[:-1], width)
    elif kind == "affine-finite":
        scale, bias = np.asarray(spec["scale"], np.float32), np.asarray(spec["bias"], np.float32)
        if scale.shape != (shape[-1],) or bias.shape != scale.shape or not np.isfinite(scale).all() or not np.isfinite(bias).all():
            raise ValueError("affine scale/bias must be finite last-axis FLOAT vectors")
        inputs.append(tensor("input_1", onnx.TensorProto.BOOL, [1]))
        constant("scale", scale)
        constant("bias", bias)
        nodes.extend([
            helper.make_node("Mul", ["input_0", "scale"], ["scaled"], name="affine_scale"),
            helper.make_node("Add", ["scaled", "bias"], ["output_0"], name="affine_bias"),
        ])
        source_ok = finite("input_0", "complete_input_finite")
        result_ok = finite("output_0", "complete_output_finite")
        nodes.extend([
            helper.make_node("And", [source_ok, result_ok], ["both_finite"], name="both_finite"),
            helper.make_node("And", ["both_finite", "input_1"], ["incoming_and_finite"], name="incoming_and_finite"),
        ])
        accepted, result_shape = "incoming_and_finite", shape
    else:
        raise ValueError("unsupported terminal stage kind")
    nodes.append(helper.make_node("Identity", [accepted], ["output_1"], name="accepted"))
    model = helper.make_model(helper.make_graph(nodes, kind, inputs, [
        tensor("output_0", onnx.TensorProto.FLOAT, result_shape),
        tensor("output_1", onnx.TensorProto.BOOL, [1]),
    ], initializers), opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import onnx

    model = materialize(json.loads(args.spec.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, args.output)
    print(json.dumps({"status": "materialized-not-target-validated", "output": str(args.output),
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                      "precision_requirement": "FLOAT arithmetic/comparisons must not narrow the finite bound"}))


if __name__ == "__main__":
    main()
