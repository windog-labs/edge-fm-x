#!/usr/bin/env python3
"""Fold bounded integer/boolean constants without evaluating floating dataflow."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import tempfile

INTEGER_OPS = {
    "Identity", "Cast", "Gather", "GatherElements", "Concat", "Unsqueeze", "Squeeze",
    "Reshape", "Transpose", "Slice", "Expand", "Range", "Add", "Sub", "Mul", "Div",
    "Mod", "Min", "Max", "Equal", "Less", "Greater", "LessOrEqual", "GreaterOrEqual",
    "And", "Or", "Xor", "Not", "Where", "ReduceSum", "ReduceProd", "ConstantOfShape",
    "Neg", "Abs", "Pad", "Tile", "Shape", "Size", "Trilu", "ScatterElements",
}
MAX_ELEMENTS = 1 << 20


def _constant_scatter_is_deterministic(node, max_elements):
    import numpy as np
    import onnx_graphsurgeon as gs

    if (len(node.inputs) != 3 or len(node.outputs) != 1
            or any(not isinstance(value, gs.Constant) for value in node.inputs)
            or set(node.attrs) - {"axis", "reduction"}
            or node.attrs.get("reduction", "none") not in ("none", b"none")):
        return False
    data, indices, updates = [value.values for value in node.inputs]
    axis = node.attrs.get("axis", 0)
    if (not isinstance(axis, (int, np.integer)) or not data.ndim
            or not -data.ndim <= axis < data.ndim
            or indices.dtype not in (np.dtype("int32"), np.dtype("int64"))
            or indices.ndim != data.ndim or updates.shape != indices.shape
            or any(value.size > max_elements for value in (data, indices, updates))):
        return False
    axis %= data.ndim
    if (data.shape[axis] == 0
            or any(size > data.shape[index] for index, size in enumerate(indices.shape) if index != axis)
            or np.any(indices < -data.shape[axis]) or np.any(indices >= data.shape[axis])):
        return False
    # Repeated destinations have undefined ordering for reduction=none.
    normalized = np.where(indices < 0, indices + data.shape[axis], indices)
    return not bool(np.any(np.diff(np.sort(normalized, axis=axis), axis=axis) == 0))


def _shape(tensor):
    if not tensor.HasField("shape") or any(not d.HasField("dim_value") for d in tensor.shape.dim):
        return None
    return tuple(d.dim_value for d in tensor.shape.dim)


def _literal(tensor):
    value = copy.deepcopy(tensor)
    value.ClearField("name")
    return value.SerializeToString()


def _node_key(node):
    attributes = []
    for attr in sorted(node.attribute, key=lambda a: a.name):
        value = copy.deepcopy(attr)
        if value.HasField("t"):
            value.t.ClearField("name")
        for tensor in value.tensors:
            tensor.ClearField("name")
        attributes.append(value.SerializeToString())
    return (node.domain, node.op_type, node.name, tuple(node.input), tuple(node.output),
            tuple(attributes))


def _floating_literals(model):
    import onnx

    integer = {onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.BOOL}
    result = {t.name: _literal(t) for t in model.graph.initializer if t.data_type not in integer}
    for node in model.graph.node:
        if node.domain in ("", "ai.onnx") and node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type not in integer:
                    result[node.output[0]] = _literal(attr.t)
    return result


def fold(model, *, trusted_plugins=(), max_passes=4, max_elements=MAX_ELEMENTS, external_base_dir=None):
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs
    from onnx import helper, numpy_helper
    from onnx_trt_plugin_shapes import infer_reviewed_shapes

    if not 1 <= max_passes <= 8 or not 1 <= max_elements <= MAX_ELEMENTS:
        raise ValueError("bounded folding budget required")
    if model.functions or model.training_info or model.graph.sparse_initializer or any(
            n.op_type in {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
                          "Bernoulli", "Multinomial", "Dropout"} or any(
                a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for a in n.attribute)
            for n in model.graph.node):
        raise ValueError("flat, dense inference graph required")
    integer = {onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.BOOL}
    dtypes = {np.dtype("int32"), np.dtype("int64"), np.dtype("bool")}
    original_literals = _floating_literals(model)
    original_nodes = {tuple(n.output): _node_key(n) for n in model.graph.node if n.op_type != "Constant"}
    original_protos = {tuple(n.output): n for n in model.graph.node if n.op_type != "Constant"}
    result = copy.deepcopy(model)
    from lower_onnx_scalar_sincos_f64 import _bounded_tensor

    external_controls = 0
    tensors = [*result.graph.initializer, *(a.t for n in result.graph.node for a in n.attribute
                                         if a.type == onnx.AttributeProto.TENSOR)]
    for tensor in tensors:
        if tensor.data_location == onnx.TensorProto.EXTERNAL and tensor.data_type in integer:
            value = _bounded_tensor(tensor, external_base_dir)
            tensor.CopyFrom(numpy_helper.from_array(value, tensor.name))
            external_controls += 1
    passes = []
    for _ in range(max_passes):
        inferred, proof = infer_reviewed_shapes(result, trusted_plugins)
        info = {v.name: v.type.tensor_type for v in
                [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]}
        static_shapes = []
        for node in inferred.graph.node:
            if node.domain not in ("", "ai.onnx") or node.op_type not in ("Shape", "Size"):
                continue
            if len(node.input) != 1 or len(node.output) != 1 or node.input[0] not in info:
                continue
            dimensions = _shape(info[node.input[0]])
            if dimensions is None or len(dimensions) > 64 or any(d < 0 for d in dimensions):
                continue
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            if len(attrs) != len(node.attribute) or any(a.ref_attr_name for a in node.attribute):
                continue
            if node.op_type == "Shape" and not set(attrs) - {"start", "end"}:
                value = np.asarray(dimensions[slice(attrs.get("start"), attrs.get("end"))], dtype=np.int64)
            elif node.op_type == "Size" and not attrs and math.prod(dimensions) < 2**63:
                value = np.asarray(math.prod(dimensions), dtype=np.int64)
            else:
                continue
            static_shapes.append(node.output[0])
            node.CopyFrom(helper.make_node("Constant", [], list(node.output), name=node.name,
                                           value=numpy_helper.from_array(value)))
        # Prevent constant evaluation from importing overridable initializer values.
        overridable = {v.name for v in inferred.graph.input} & {t.name for t in inferred.graph.initializer}
        if overridable:
            raise ValueError("overridable initializer inputs are not foldable")
        graph = gs.import_onnx(inferred)

        def exclude(node):
            if node.domain not in (None, "", "ai.onnx") or node.op not in INTEGER_OPS:
                return True
            tensors = [t for t in [*node.inputs, *node.outputs] if not t.is_empty()]
            try:
                unsupported_dtype = any(t.dtype is None or np.dtype(t.dtype) not in dtypes for t in tensors)
            except (TypeError, ValueError):
                # Unsupported external float literals can retain an ONNX dtype enum.
                return True
            if unsupported_dtype:
                return True
            if node.op == "ScatterElements" and not _constant_scatter_is_deterministic(node, max_elements):
                return True
            for tensor in node.outputs:
                if tensor.shape is None or any(not isinstance(d, (int, np.integer)) or d < 0 for d in tensor.shape):
                    return True
                if math.prod(tensor.shape) > max_elements:
                    return True
            # Integer narrowing and bool casts need value-range proofs, not just dtype inference.
            if node.op == "Cast" and not (node.inputs[0].dtype == node.outputs[0].dtype or (
                    np.dtype(node.inputs[0].dtype) == np.dtype("int32") and
                    np.dtype(node.outputs[0].dtype) == np.dtype("int64"))):
                return True
            return False

        before = len(graph.nodes)
        graph.toposort().fold_constants(fold_shapes=False, recurse_subgraphs=False,
                                       recurse_functions=False, flatten_subgraphs=False,
                                       error_ok=False, size_threshold=max_elements * 8,
                                       should_exclude_node=exclude)
        graph.cleanup(remove_unused_graph_inputs=False).toposort()
        result = gs.export_onnx(graph)
        result.ir_version = model.ir_version
        result.graph.ClearField("input")
        result.graph.input.extend(model.graph.input)
        result.graph.ClearField("output")
        result.graph.output.extend(model.graph.output)
        for node in result.graph.node:
            if node.op_type != "Constant" and tuple(node.output) in original_nodes:
                if _node_key(node) != original_nodes[tuple(node.output)]:
                    raise ValueError("retained computation changed rather than constant-folded: " + node.name)
                node.CopyFrom(original_protos[tuple(node.output)])
        if any(original_literals.get(name) != value for name, value in _floating_literals(result).items()):
            raise ValueError("floating literal payload or descriptor changed")
        passes.append({"source_nodes": before, "remaining_nodes": len(result.graph.node),
                       "materialized_static_shapes": len(static_shapes), "shape_proof": proof})
        if not static_shapes and len(result.graph.node) == before:
            break
    return result, {"schema": "vlaforge.onnx_integer_shape_folding/1", "passes": passes,
                    "source_nodes": len(model.graph.node), "remaining_nodes": len(result.graph.node),
                    "remaining_ops": dict(Counter(n.op_type for n in result.graph.node)),
                    "max_elements": max_elements, "materialized_external_integer_controls": external_controls,
                    "public_io_unchanged": True,
                    "retained_computations_unchanged": True, "floating_literals_unchanged": True,
                    "floating_arithmetic_evaluated": False, "target_inference_executed": False}


def main():
    import onnx
    from export_torchscript_region_onnx import digest, external_files, publish_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-plugin", action="append", default=[])
    parser.add_argument("--max-passes", type=int, default=4)
    args = parser.parse_args()
    source, output = args.input.resolve(strict=True), args.output.absolute()
    report_path = output.with_suffix(".report.json")
    if source.parent != output.parent.resolve() or output.exists() or report_path.exists():
        raise ValueError("fresh output/report beside immutable external weights required")
    model = onnx.load_model(source, load_external_data=False)
    source_hash = digest(source)
    weights = [{"location": str(p), "sha256": digest(source.parent / p),
                "bytes": (source.parent / p).stat().st_size} for p in external_files(model, source.parent)]
    result, report = fold(model, trusted_plugins=args.trusted_plugin, max_passes=args.max_passes,
                          external_base_dir=source.parent)
    with tempfile.NamedTemporaryFile(prefix=".integer-shapes-", suffix=".onnx", dir=source.parent) as temporary:
        staged = Path(temporary.name)
        staged.write_bytes(result.SerializeToString())
        onnx.checker.check_model(str(staged))
        if digest(source) != source_hash or any(digest(source.parent / w["location"]) != w["sha256"] for w in weights):
            raise ValueError("immutable source or weights changed during folding")
        report.update(source_sha256=source_hash, output_sha256=digest(staged), external_data=weights,
                      generator_sha256=digest(Path(__file__)), onnx_checker="passed")
        publish_files(staged, output, [])
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"source_nodes": report["source_nodes"], "remaining_nodes": report["remaining_nodes"],
                      "passes": len(report["passes"])}))


if __name__ == "__main__":
    main()
