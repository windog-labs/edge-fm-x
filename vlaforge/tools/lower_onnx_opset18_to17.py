#!/usr/bin/env python3
"""Conservatively lower flat ONNX opset-18 graphs for older importers.

No tensor precision is changed. Only statically known reduction axes and
operator forms with unchanged opset-17 semantics are accepted. Numerical
validation of the complete model remains a separate gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def lower(source: Path, output: Path) -> dict:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_hash = digest(source)
    model = onnx.load(source, load_external_data=False)
    if digest(source) != source_hash:
        raise ValueError("source graph changed while loading")
    default = [opset for opset in model.opset_import if not opset.domain]
    if len(default) != 1 or default[0].version != 18:
        raise ValueError("requires an explicit default opset 18")
    if model.functions or any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                              for node in model.graph.node for a in node.attribute):
        raise ValueError("nested graphs/functions need independent schema conversion")
    external = {}
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                path = (source.parent / entry.value).resolve(strict=True)
                if not path.is_relative_to(source.parent):
                    raise ValueError("external weights escape source directory")
                if entry.value not in external:
                    external[entry.value] = digest(path)
    if external and output.parent != source.parent:
        raise ValueError("retain external weights in the source directory")

    graph_inputs = {value.name for value in model.graph.input}
    constants = {tensor.name: numpy_helper.to_array(copy.deepcopy(tensor), base_dir=str(source.parent))
                 for tensor in model.graph.initializer
                 if tensor.name not in graph_inputs and tensor.data_type == onnx.TensorProto.INT64}
    for node in model.graph.node:
        if not node.domain and node.op_type == "Constant" and len(node.output) == 1:
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if "value" in attrs and attrs["value"].data_type == onnx.TensorProto.INT64:
                constants[node.output[0]] = numpy_helper.to_array(
                    copy.deepcopy(attrs["value"]), base_dir=str(source.parent))
            elif "value_ints" in attrs:
                constants[node.output[0]] = np.asarray(attrs["value_ints"], dtype=np.int64)
    # Exporters may wrap literal axes in shape-only dataflow. Never derive an
    # axes constant from a runtime input, including overridable initializers.
    from onnx.reference import ReferenceEvaluator

    axes_dataflow_resolved = 0
    for node in model.graph.node:
        if (node.domain or node.op_type not in {"Identity", "Reshape", "Concat", "Unsqueeze", "Squeeze"}
                or len(node.output) != 1 or not node.input
                or any(name and name not in constants for name in node.input)):
            continue
        arrays = [constants[name] for name in node.input if name]
        if any(array.dtype != np.int64 or array.size > 4096 for array in arrays):
            continue
        if node.op_type == "Concat" and sum(array.size for array in arrays) > 4096:
            continue
        if node.op_type == "Reshape" and (len(arrays) != 2 or arrays[1].ndim != 1 or arrays[1].size > 8):
            continue
        value = ReferenceEvaluator(node, opsets={"": 18}).run(
            None, {name: constants[name] for name in node.input if name})[0]
        if value.dtype == np.int64 and value.size <= 4096:
            constants[node.output[0]] = value
            axes_dataflow_resolved += 1

    reductions = {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1",
                  "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceSumSquare"}
    counts = Counter()
    for node in model.graph.node:
        if node.domain:
            continue
        if onnx.defs.get_schema(node.op_type, 18).since_version <= 17:
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type in reductions:
            axes = None
            if len(node.input) > 1 and node.input[1]:
                axes = constants.get(node.input[1])
                if axes is None or axes.ndim != 1 or axes.dtype != np.int64:
                    raise ValueError(f"{node.name}: reduction axes must be constant int64 vector")
                axes = [int(value) for value in axes]
                if len(set(axes)) != len(axes):
                    raise ValueError(f"{node.name}: duplicate reduction axes")
            if not axes and attrs.pop("noop_with_empty_axes", 0):
                node.op_type = "Identity"
                del node.input[1:]
                del node.attribute[:]
                counts["empty_axes_identity"] += 1
            else:
                attrs.pop("noop_with_empty_axes", None)
                if axes:
                    attrs["axes"] = axes
                del node.input[1:]
                del node.attribute[:]
                node.attribute.extend(helper.make_attribute(name, value) for name, value in attrs.items())
                counts["constant_reduction_axes"] += 1
        elif node.op_type == "Resize":
            if attrs.get("antialias", 0) != 0 or "axes" in attrs or attrs.get(
                "keep_aspect_ratio_policy", b"stretch"
            ) != b"stretch":
                raise ValueError(f"{node.name}: Resize has non-default opset-18 semantics")
            keep = [a for a in node.attribute if a.name not in ("antialias", "keep_aspect_ratio_policy")]
            del node.attribute[:]
            node.attribute.extend(keep)
            counts["unchanged_resize"] += 1
        elif node.op_type == "Pad":
            if (len(node.input) > 3 and any(node.input[3:])) or attrs.get("mode", b"constant") not in (
                b"constant", b"edge", b"reflect"
            ):
                raise ValueError(f"{node.name}: Pad axes/mode require explicit lowering")
            del node.input[3:]
            counts["unchanged_pad"] += 1
        elif node.op_type == "Split":
            if "num_outputs" in attrs or len(node.input) != 2 or not node.input[1]:
                raise ValueError(f"{node.name}: Split requires explicit split sizes")
            counts["explicit_split_sizes"] += 1
        elif node.op_type in ("ScatterND", "ScatterElements"):
            if attrs.get("reduction", b"none") not in (b"none", b"add", b"mul"):
                raise ValueError(f"{node.name}: {node.op_type} reduction needs opset 18")
            counts["unchanged_scatter_nd" if node.op_type == "ScatterND"
                   else "unchanged_scatter_elements"] += 1
        else:
            raise ValueError(f"{node.name}: unsupported opset-18 change for {node.op_type}")
    default[0].version = 17
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".opset17-", suffix=".onnx", dir=output.parent)
    os.close(descriptor)
    staged = Path(temporary)
    try:
        onnx.save(model, staged)
        onnx.checker.check_model(str(staged))
        for name, expected in external.items():
            if digest(source.parent / name) != expected:
                raise ValueError("conversion changed external weights")
        if digest(source) != source_hash:
            raise ValueError("source graph changed during conversion")
        os.link(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    return {"schema": "vlaforge.onnx_opset18_to17/1", "status": "converted-checker-passed",
            "source": str(source), "source_sha256": source_hash,
            "output": str(output), "output_sha256": digest(output),
            "external_weights_sha256": external, "rewrites": dict(counts),
            "constant_integer_dataflow_resolved": axes_dataflow_resolved,
            "precision_changed": False, "requires_numerical_verification": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = lower(args.input, args.output)
    with args.report.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
