#!/usr/bin/env python3
"""Materialize bounded integer-only ONNX control dataflow as literal Constants."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

MAX_ELEMENTS = 4096
MAX_RANK = 64
MAX_DEPTH = 64
CONTROL_INPUTS = {"Slice": (1, 2, 3, 4), "Reshape": (1,), "Expand": (1,),
                  "Unsqueeze": (1,), "Squeeze": (1,)}
INTEGER_OPS = {"Constant", "Identity", "Cast", "Reshape", "Concat", "Unsqueeze", "Squeeze"}


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "_vf_control_tensor_reader", Path(__file__).with_name("lower_onnx_scalar_sincos_f64.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bounded_shape(shape):
    if (len(shape) > MAX_RANK or any(d < 0 or d > MAX_ELEMENTS for d in shape)
            or math.prod(max(1, d) for d in shape) > MAX_ELEMENTS):
        raise ValueError("integer tensor shape exceeds evaluation budget")


class _Evaluator:
    def __init__(self, model, base_dir):
        self.inputs = {v.name for v in model.graph.input}
        self.initializers = {v.name: v for v in model.graph.initializer}
        if len(self.initializers) != len(model.graph.initializer):
            raise ValueError("duplicate initializers")
        self.producers = {}
        for node in model.graph.node:
            for name in node.output:
                if not name:
                    continue
                if name in self.producers or name in self.inputs or name in self.initializers:
                    raise ValueError("unique SSA producer names required")
                self.producers[name] = node
        versions = [v.version for v in model.opset_import if v.domain in ("", "ai.onnx")]
        if not versions or len(set(versions)) != 1 or not 13 <= versions[0] <= 18:
            raise ValueError("explicit standard opset 13..18 required")
        self.version = versions[0]
        self.base_dir = base_dir
        self.cache = {}
        self.active = set()
        self.reader = _helpers()._bounded_tensor

    def tensor(self, tensor):
        import onnx

        if tensor.data_type not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            raise ValueError("only INT32/INT64 literal tensors may be evaluated")
        _bounded_shape(tuple(tensor.dims))
        return self.reader(tensor, self.base_dir)

    def evaluate(self, name, depth=0):
        import numpy as np
        import onnx
        from onnx import helper
        from onnx.reference import ReferenceEvaluator

        if name in self.inputs:
            raise ValueError("runtime or overridable input: " + name)
        if name in self.cache:
            return self.cache[name]
        if depth >= MAX_DEPTH or name in self.active:
            raise ValueError("cyclic or too-deep integer dataflow")
        self.active.add(name)
        try:
            if name in self.initializers:
                result = self.tensor(self.initializers[name])
            else:
                node = self.producers.get(name)
                if (node is None or node.domain not in ("", "ai.onnx")
                        or node.op_type not in INTEGER_OPS or len(node.output) != 1):
                    raise ValueError("unsupported integer control producer: " + name)
                if len({a.name for a in node.attribute}) != len(node.attribute) or any(a.ref_attr_name for a in node.attribute):
                    raise ValueError("ambiguous or linked attributes")
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                op = node.op_type
                if op == "Constant":
                    if node.input or len(node.attribute) != 1:
                        raise ValueError("one integer literal required")
                    attr = node.attribute[0]
                    if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                        result = self.tensor(attr.t)
                    elif attr.name == "value_int" and attr.type == onnx.AttributeProto.INT:
                        result = np.asarray(attr.i, dtype=np.int64)
                    elif attr.name == "value_ints" and attr.type == onnx.AttributeProto.INTS and len(attr.ints) <= MAX_ELEMENTS:
                        result = np.asarray(attr.ints, dtype=np.int64)
                    else:
                        raise ValueError("bounded integer literal required")
                else:
                    values = [self.evaluate(v, depth + 1) for v in node.input if v]
                    if len(values) != len(node.input):
                        raise ValueError("missing integer dataflow input")
                    if op == "Identity":
                        if len(values) != 1 or attrs:
                            raise ValueError("invalid Identity control")
                    elif op == "Cast":
                        if len(values) != 1 or set(attrs) != {"to"} or attrs["to"] not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
                            raise ValueError("only Cast to INT32/INT64 may be evaluated")
                        if attrs["to"] == onnx.TensorProto.INT32 and values[0].size and (
                                values[0].min() < -(2**31) or values[0].max() > 2**31 - 1):
                            raise ValueError("out-of-range INT32 cast is left unresolved")
                    elif op == "Reshape":
                        if len(values) != 2 or set(attrs) - {"allowzero"} or attrs.get("allowzero", 0) not in (0, 1):
                            raise ValueError("invalid Reshape control")
                        shape = values[1]
                        if shape.dtype != np.int64 or shape.ndim != 1 or shape.size > MAX_RANK:
                            raise ValueError("bounded INT64 Reshape shape vector required")
                        dimensions = [int(d) for d in shape]
                        if dimensions.count(-1) > 1 or any(d < -1 or d > MAX_ELEMENTS for d in dimensions):
                            raise ValueError("Reshape dimensions exceed evaluation budget")
                        if not attrs.get("allowzero", 0):
                            for axis, d in enumerate(dimensions):
                                if d == 0:
                                    if axis >= values[0].ndim:
                                        raise ValueError("Reshape zero-copy axis outside input rank")
                                    dimensions[axis] = values[0].shape[axis]
                        known = math.prod(d for d in dimensions if d != -1)
                        if -1 in dimensions:
                            if known == 0 or values[0].size % known:
                                raise ValueError("ambiguous or incompatible inferred Reshape dimension")
                            dimensions[dimensions.index(-1)] = values[0].size // known
                        _bounded_shape(tuple(dimensions))
                        if math.prod(dimensions) != values[0].size:
                            raise ValueError("Reshape changes element count")
                    elif op == "Concat":
                        if not values or set(attrs) != {"axis"} or sum(v.size for v in values) > MAX_ELEMENTS:
                            raise ValueError("Concat exceeds evaluation budget")
                        rank = values[0].ndim
                        axis = int(attrs["axis"])
                        if not rank or not -rank <= axis < rank or any(v.ndim != rank or v.dtype != values[0].dtype for v in values):
                            raise ValueError("invalid integer Concat ranks/types/axis")
                        axis %= rank
                        dimensions = list(values[0].shape)
                        dimensions[axis] = sum(v.shape[axis] for v in values)
                        _bounded_shape(tuple(dimensions))
                    elif op in ("Unsqueeze", "Squeeze"):
                        if attrs or len(values) not in ((2,) if op == "Unsqueeze" else (1, 2)):
                            raise ValueError("explicit opset-13 integer axes required")
                        if len(values) == 2 and (values[1].dtype != np.int64 or values[1].ndim != 1):
                            raise ValueError("INT64 axes vector required")
                        if op == "Unsqueeze" and values[0].ndim + values[1].size > MAX_RANK:
                            raise ValueError("Unsqueeze rank exceeds evaluation budget")
                    evaluation_node = copy.deepcopy(node)
                    evaluation_node.domain = ""
                    result = ReferenceEvaluator(evaluation_node, opsets={"": self.version}).run(
                        None, dict(zip(node.input, values)))[0]
            if result.dtype not in (np.dtype("int32"), np.dtype("int64")) or result.size > MAX_ELEMENTS:
                raise ValueError("integer result exceeds evaluation budget")
            _bounded_shape(result.shape)
            self.cache[name] = result
            return result
        finally:
            self.active.remove(name)


def fold(model, *, external_base_dir=None):
    import onnx
    from onnx import helper, numpy_helper

    if model.functions or model.training_info or any(
            a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for node in model.graph.node for a in node.attribute):
        raise ValueError("flat inference graph required")
    evaluator = _Evaluator(model, external_base_dir)
    report = {"schema": "vlaforge.onnx_static_controls/1", "status": "folded-not-target-validated",
              "max_elements": MAX_ELEMENTS, "rewritten": [], "resolved": [], "unresolved": [],
              "changes_public_io": False, "changes_initializer_payloads": False,
              "target_execution_verified": False, "full_model_fidelity_verified": False}
    replacements = {}
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type not in CONTROL_INPUTS:
            continue
        for slot in CONTROL_INPUTS[node.op_type]:
            if slot >= len(node.input) or not node.input[slot]:
                if node.op_type == "Slice" and slot == 3:
                    report["unresolved"].append({"consumer": node.name, "slot": slot, "value": "", "reason": "Slice axes absent; no axes invented"})
                continue
            name = node.input[slot]
            item = {"consumer": node.name, "slot": slot, "value": name}
            try:
                value = evaluator.evaluate(name)
                if value.ndim != 1:
                    raise ValueError("control must be an integer vector")
                producer = evaluator.producers.get(name)
                report["resolved"].append({**item, "dtype": str(value.dtype), "shape": list(value.shape)})
                if producer is None or producer.op_type == "Constant":
                    continue
                replacements[name] = value
            except (ValueError, TypeError, IndexError, RuntimeError, OSError, OverflowError) as error:
                report["unresolved"].append({**item, "reason": str(error)})
    result = copy.deepcopy(model)
    for node in result.graph.node:
        if len(node.output) != 1 or node.output[0] not in replacements:
            continue
        value = replacements[node.output[0]]
        before = hashlib.sha256(node.SerializeToString()).hexdigest()
        node.op_type = "Constant"
        node.ClearField("input")
        node.ClearField("attribute")
        node.attribute.append(helper.make_attribute("value", numpy_helper.from_array(value)))
        report["rewritten"].append({"node": node.name, "value": node.output[0], "source_node_sha256": before,
                                    "dtype": str(value.dtype), "shape": list(value.shape),
                                    "payload_sha256": hashlib.sha256(value.tobytes()).hexdigest()})
    return result, report


def main():
    import onnx
    from export_torchscript_region_onnx import digest, publish_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(strict=True), args.output.absolute()
    report_path = output.with_suffix(".report.json")
    if source.parent != output.parent.resolve() or os.path.lexists(output) or os.path.lexists(report_path):
        raise ValueError("fresh output/report beside immutable source weights required")
    pin = digest(source)
    model = onnx.load(source, load_external_data=False)
    onnx.checker.check_model(str(source))
    result, report = fold(model, external_base_dir=source.parent)
    with tempfile.NamedTemporaryFile(prefix=".controls-", suffix=".onnx", dir=source.parent) as temporary:
        staged = Path(temporary.name)
        staged.write_bytes(result.SerializeToString())
        onnx.checker.check_model(str(staged))
        if digest(source) != pin:
            raise ValueError("immutable source changed during folding")
        report.update(source=str(source), source_sha256=pin, output=str(output),
                      output_sha256=digest(staged), generator_sha256=digest(Path(__file__)))
        publish_files(staged, output, [])
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"rewritten": len(report["rewritten"]), "unresolved": len(report["unresolved"]), "output": str(output)}))


if __name__ == "__main__":
    main()
