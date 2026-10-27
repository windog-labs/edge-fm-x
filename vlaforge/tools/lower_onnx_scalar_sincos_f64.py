#!/usr/bin/env python3
"""Lower one proven scalar DOUBLE sin/cos block to a FLOAT-interface TRT plugin.

Frequency computation is evaluated only from bounded constant dataflow. This
prepares a plugin graph, not a numerical or target-support certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile

MAX_FREQUENCIES = 16384
MAX_CONSTANT_ELEMENTS = 4 * MAX_FREQUENCIES
PLUGIN = "VLAForgeScalarSinCosF64"
DOMAIN = "vlaforge.trt"
CONSTANT_OPS = {"Constant", "Identity", "Cast", "Add", "Sub", "Mul", "Div",
                "Pow", "Reciprocal", "Neg", "Less", "Equal", "Where", "Range",
                "Unsqueeze", "Squeeze", "Concat", "Expand"}


def _index(model):
    producers = {}
    for node in model.graph.node:
        for name in node.output:
            if not name or name in producers:
                raise ValueError("unique nonempty SSA outputs required")
            producers[name] = node
    return producers


def _attributes(node):
    from onnx import helper

    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def _type_shape(value):
    tensor = value.type.tensor_type
    if (value.type.WhichOneof("value") != "tensor_type" or not tensor.HasField("shape")
            or any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in tensor.shape.dim)):
        raise ValueError("static positive tensor dimensions required")
    return tensor.elem_type, tuple(dim.dim_value for dim in tensor.shape.dim)


def _bounded_tensor(tensor, base_dir):
    import onnx

    if (any(dim < 0 for dim in tensor.dims)
            or math.prod(tensor.dims) > MAX_CONSTANT_ELEMENTS
            or tensor.data_type not in {onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE,
                                        onnx.TensorProto.INT32, onnx.TensorProto.INT64,
                                        onnx.TensorProto.BOOL}):
        raise ValueError("constant tensor exceeds the bounded numeric domain")
    snapshot = copy.deepcopy(tensor)
    if tensor.data_location == onnx.TensorProto.EXTERNAL:
        if base_dir is None:
            raise ValueError("external frequency data requires an explicit base directory")
        spec = importlib.util.spec_from_file_location(
            "_vf_scalar_sincos_external", Path(__file__).with_name("export_torchscript_region_onnx.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.external_files(tensor, Path(base_dir))
        entries = {entry.key: entry.value for entry in tensor.external_data}
        if len(entries) != len(tensor.external_data):
            raise ValueError("duplicate frequency external-data keys")
        for key in ("offset", "length"):
            if key in entries and (not entries[key].isascii() or not entries[key].isdigit()):
                raise ValueError("invalid frequency external-data range")
        widths = {onnx.TensorProto.FLOAT: 4, onnx.TensorProto.DOUBLE: 8,
                  onnx.TensorProto.INT32: 4, onnx.TensorProto.INT64: 8, onnx.TensorProto.BOOL: 1}
        count = math.prod(tensor.dims) * widths[tensor.data_type]
        path = Path(base_dir) / entries["location"]
        offset = int(entries.get("offset", "0"))
        length = int(entries.get("length", str(path.stat().st_size - offset)))
        if offset < 0 or length != count or offset + count > path.stat().st_size:
            raise ValueError("external frequency tensor has an invalid byte count")
        with path.open("rb") as stream:
            stream.seek(offset)
            raw = stream.read(count)
        if len(raw) != count:
            raise ValueError("truncated frequency tensor")
        snapshot.data_location = onnx.TensorProto.DEFAULT
        del snapshot.external_data[:]
        snapshot.raw_data = raw
    return onnx.numpy_helper.to_array(snapshot)


def _preflight(node, values):
    import numpy as np

    op, attrs = node.op_type, _attributes(node)
    if op == "Range":
        if len(values) != 3 or any(value.size != 1 for value in values):
            raise ValueError("bounded scalar Range inputs required")
        start, end, step = (value.item() for value in values)
        if not all(math.isfinite(value) for value in (start, end, step)) or step == 0:
            raise ValueError("invalid constant Range")
        count = max(0, math.ceil((end - start) / step))
        if count > MAX_CONSTANT_ELEMENTS - 2:
            raise ValueError("constant Range exceeds evaluation budget")
    elif op == "Expand":
        if len(values) != 2 or values[1].dtype.kind not in "iu" or values[1].ndim != 1:
            raise ValueError("explicit bounded Expand shape required")
        shape = tuple(int(dim) for dim in values[1])
        if len(shape) > 8 or any(dim <= 0 for dim in shape) or math.prod(shape) > MAX_CONSTANT_ELEMENTS:
            raise ValueError("constant Expand exceeds evaluation budget")
        if math.prod(np.broadcast_shapes(values[0].shape, shape)) > MAX_CONSTANT_ELEMENTS:
            raise ValueError("constant broadcast exceeds evaluation budget")
    elif op == "Concat":
        if sum(value.size for value in values) > MAX_CONSTANT_ELEMENTS:
            raise ValueError("constant Concat exceeds evaluation budget")
    elif op in {"Add", "Sub", "Mul", "Div", "Pow", "Less", "Equal", "Where"}:
        if math.prod(np.broadcast_shapes(*(value.shape for value in values))) > MAX_CONSTANT_ELEMENTS:
            raise ValueError("constant broadcast exceeds evaluation budget")


def constant_value(model, name, *, external_base_dir=None):
    """Return a small literal-derived value and its exact producer closure."""
    import numpy as np
    import onnx
    from onnx.reference import ReferenceEvaluator

    producers = _index(model)
    inputs = {value.name for value in model.graph.input}
    initializers = {value.name: value for value in model.graph.initializer}
    opsets = {item.domain: item.version for item in model.opset_import}
    values, active, nodes, tensors = {}, set(), set(), set()

    def evaluate(value):
        if value in inputs:
            raise ValueError("frequency depends on runtime input: " + value)
        if value in values:
            return values[value]
        if value in active:
            raise ValueError("cyclic constant dataflow")
        active.add(value)
        if value in initializers:
            result = _bounded_tensor(initializers[value], external_base_dir)
            tensors.add(value)
        else:
            node = producers.get(value)
            if (node is None or node.domain not in ("", "ai.onnx")
                    or node.op_type not in CONSTANT_OPS or len(node.output) != 1
                    or any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
                           for attr in node.attribute)):
                raise ValueError("unreviewed constant producer: " + value)
            arrays = [evaluate(operand) for operand in node.input if operand]
            literal_tensor = None
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.TENSOR:
                        literal_tensor = _bounded_tensor(attr.t, external_base_dir)
                    elif attr.type in (onnx.AttributeProto.INTS, onnx.AttributeProto.FLOATS):
                        if len(attr.ints if attr.type == onnx.AttributeProto.INTS else attr.floats) > MAX_CONSTANT_ELEMENTS:
                            raise ValueError("constant attribute exceeds evaluation budget")
            _preflight(node, arrays)
            result = literal_tensor if literal_tensor is not None else ReferenceEvaluator(node, opsets=opsets).run(
                None, {operand: array for operand, array in zip(
                    (operand for operand in node.input if operand), arrays, strict=True)})[0]
            nodes.add(value)
        result = np.asarray(result)
        if (result.size > MAX_CONSTANT_ELEMENTS or result.dtype.kind not in "fbiu"
                or not np.isfinite(result).all()):
            raise ValueError("invalid bounded constant result")
        values[value] = result
        active.remove(value)
        return result

    return evaluate(name), nodes, tensors


def lower(model, output_value, *, external_base_dir=None):
    import numpy as np
    import onnx
    from onnx import helper

    if model.functions or model.training_info or any(
            any(attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for attr in node.attribute)
            for node in model.graph.node):
        raise ValueError("flat functional graph required")
    source = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=False)
    producers = _index(source)
    info = {value.name: value for value in [*source.graph.input, *source.graph.output, *source.graph.value_info]}
    for initializer in source.graph.initializer:
        info.setdefault(initializer.name, helper.make_tensor_value_info(
            initializer.name, initializer.data_type, list(initializer.dims)))

    def node(value, op):
        result = producers.get(value)
        if (result is None or result.domain not in ("", "ai.onnx") or result.op_type != op
                or len(result.output) != 1):
            raise ValueError("expected " + op + " producer for " + value)
        return result

    final = node(output_value, "Cast")
    if len(final.input) != 1 or _attributes(final) != {"to": onnx.TensorProto.FLOAT}:
        raise ValueError("explicit terminal DOUBLE-to-FLOAT Cast required")
    concat = node(final.input[0], "Concat")
    if len(concat.input) != 2 or _attributes(concat) != {"axis": 1}:
        raise ValueError("sin-then-cos last-axis Concat required")
    sine, cosine = node(concat.input[0], "Sin"), node(concat.input[1], "Cos")
    if len(sine.input) != 1 or list(sine.input) != list(cosine.input) or _attributes(sine) or _attributes(cosine):
        raise ValueError("Sin and Cos must share the same unmodified phase")
    multiply = node(sine.input[0], "Mul")
    if len(multiply.input) != 2 or _attributes(multiply):
        raise ValueError("binary DOUBLE phase multiplication required")
    matches = []
    for frequency, time in (tuple(multiply.input), tuple(reversed(multiply.input))):
        try:
            array, cone, initializers = constant_value(source, frequency, external_base_dir=external_base_dir)
        except ValueError:
            continue
        cast = producers.get(time)
        if (cast is None or cast.op_type != "Cast" or cast.domain not in ("", "ai.onnx")
                or len(cast.input) != 1 or _attributes(cast) != {"to": onnx.TensorProto.DOUBLE}):
            continue
        if cast.input[0] not in info or _type_shape(info[cast.input[0]])[0] != onnx.TensorProto.FLOAT:
            continue
        time_shape = _type_shape(info[cast.input[0]])[1]
        if math.prod(time_shape) != 1 or array.dtype != np.dtype("float64"):
            continue
        if array.ndim not in (1, 2) or (array.ndim == 2 and array.shape[0] != 1):
            continue
        if not 1 <= array.size <= MAX_FREQUENCIES:
            continue
        matches.append((array, cone, initializers, cast))
    if len(matches) != 1:
        raise ValueError("one bounded constant frequency vector and FLOAT scalar time required")
    array, cone, initializers, cast = matches[0]
    n = array.size
    if _type_shape(info[output_value]) != (onnx.TensorProto.FLOAT, (1, 2 * n)):
        raise ValueError("source FLOAT output must have shape [1,2*N]")
    for value in [*multiply.input, *multiply.output, *sine.output, *cosine.output, *concat.output]:
        if value not in info or _type_shape(info[value])[0] != onnx.TensorProto.DOUBLE:
            raise ValueError("all phase/sin/cos computation must remain DOUBLE")
    group = {value for item in [cast, multiply, sine, cosine, concat, final] for value in item.output}
    exposed = {value.name for value in source.graph.output} | {
        operand for item in source.graph.node if not any(value in group for value in item.output)
        for operand in item.input}
    if (group - {output_value}) & exposed:
        raise ValueError("matched intermediate has an external observer")
    frequencies = np.ascontiguousarray(array.reshape(-1), dtype="<f8")
    words = list(struct.unpack("<" + "i" * (2 * n), frequencies.tobytes()))
    if struct.pack("<" + "i" * len(words), *words) != frequencies.tobytes():
        raise ValueError("frequency bit encoding changed bytes")
    result = copy.deepcopy(source)
    occupied = {item.name for item in result.graph.node}
    name = "vf_scalar_sincos_f64"
    while name in occupied:
        name += "_"
    plugin = helper.make_node(PLUGIN, [cast.input[0]], [output_value], name=name, domain=DOMAIN,
                              frequency_words=words, plugin_version="1", plugin_namespace="")
    rebuilt = []
    for item in result.graph.node:
        if output_value in item.output:
            rebuilt.append(plugin)
        elif not any(value in group for value in item.output):
            rebuilt.append(item)
    uses = Counter(operand for item in rebuilt for operand in item.input if operand)
    uses.update(value.name for value in result.graph.output)
    removed_constants = set()
    for item in reversed(rebuilt):
        if set(item.output) <= cone and all(uses[value] == 0 for value in item.output):
            removed_constants.update(item.output)
            uses.subtract(operand for operand in item.input if operand)
    del result.graph.node[:]
    result.graph.node.extend(item for item in rebuilt if not set(item.output) <= removed_constants)
    removed_initializers = initializers - {name for name in initializers if uses[name] > 0}
    retained_initializers = [item for item in result.graph.initializer if item.name not in removed_initializers]
    del result.graph.initializer[:]
    result.graph.initializer.extend(retained_initializers)
    removed_values = (group - {output_value}) | removed_constants | removed_initializers
    retained_info = [item for item in result.graph.value_info if item.name not in removed_values]
    del result.graph.value_info[:]
    result.graph.value_info.extend(retained_info)
    imports = {item.domain: item.version for item in result.opset_import}
    if DOMAIN in imports and imports[DOMAIN] != 1:
        raise ValueError("conflicting plugin domain version")
    if DOMAIN not in imports:
        result.opset_import.append(helper.make_opsetid(DOMAIN, 1))
    report = {"schema": "vlaforge.scalar_sincos_f64_lowering/1", "status": "prepared-not-target-validated",
              "plugin": PLUGIN, "version": "1", "input": cast.input[0], "output": output_value,
              "frequency_count": n, "frequency_sha256": hashlib.sha256(frequencies.tobytes()).hexdigest(),
              "frequency_words": words, "source_nodes": len(source.graph.node),
              "output_nodes": len(result.graph.node), "removed_operator_outputs": sorted(group - {output_value}),
              "removed_constant_outputs": sorted(removed_constants),
              "removed_constant_initializers": sorted(removed_initializers),
              "constant_dataflow_only": True, "f64_arithmetic_demoted": False,
              "source_numerical_parity_verified": False, "target_execution_verified": False}
    return result, report


def portable_shadow(input_value, output_value, words):
    """Standard ONNX oracle for the plugin's declared FLOAT/DOUBLE contract."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if not words or len(words) % 2 or len(words) > 2 * MAX_FREQUENCIES:
        raise ValueError("paired bounded frequency words required")
    frequencies = np.frombuffer(struct.pack("<" + "i" * len(words), *words), dtype="<f8")
    if not np.isfinite(frequencies).all():
        raise ValueError("finite frequency values required")
    occupied = {input_value.name, output_value.name}

    def fresh(label):
        while label in occupied:
            label += "_"
        occupied.add(label)
        return label

    time, frequency, phase, sine, cosine, joined = (fresh(label) for label in (
        "time_f64", "frequencies", "phase", "sine", "cosine", "joined"))
    nodes = [helper.make_node("Cast", [input_value.name], [time], to=onnx.TensorProto.DOUBLE),
             helper.make_node("Mul", [time, frequency], [phase]),
             helper.make_node("Sin", [phase], [sine]),
             helper.make_node("Cos", [phase], [cosine]),
             helper.make_node("Concat", [sine, cosine], [joined], axis=1),
             helper.make_node("Cast", [joined], [output_value.name], to=onnx.TensorProto.FLOAT)]
    return helper.make_model(helper.make_graph(nodes, "scalar_sincos_f64_shadow", [input_value], [output_value],
        [numpy_helper.from_array(frequencies.reshape(1, -1), frequency)]),
        opset_imports=[helper.make_opsetid("", 18)], ir_version=8)


def main():
    import onnx
    from export_torchscript_region_onnx import digest, external_files, publish_files
    from lower_onnx_split_sequences import copy_external, publication_paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-value", required=True)
    args = parser.parse_args()
    source_path = args.input.resolve(strict=True)
    source_hash = digest(source_path)
    source = onnx.load(source_path, load_external_data=False)
    if digest(source_path) != source_hash:
        raise ValueError("source graph changed while loading")
    onnx.checker.check_model(str(source_path))
    sidecars = external_files(source, source_path.parent)
    result, report = lower(source, args.output_value, external_base_dir=source_path.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "model.onnx"
    publication_paths(output, sidecars)
    with tempfile.TemporaryDirectory(prefix=".sincos-stage-", dir=args.output) as temporary:
        staged = Path(temporary) / output.name
        copies = [copy_external(source_path.parent / name, staged.parent / name, name) for name in sidecars]
        # Frequency specialization must agree with the actual published snapshot.
        _, staged_report = lower(source, args.output_value, external_base_dir=staged.parent)
        if staged_report["frequency_words"] != report["frequency_words"]:
            raise ValueError("frequency data changed between evaluation and staging")
        onnx.save_model(result, staged)
        onnx.checker.check_model(str(staged))
        if digest(source_path) != source_hash:
            raise ValueError("source graph changed during preparation")
        publish_files(staged, output, sidecars)
    info = {value.name: value for value in [*result.graph.input, *result.graph.output, *result.graph.value_info]}
    shadow = portable_shadow(info[report["input"]], info[report["output"]], report["frequency_words"])
    onnx.save_model(shadow, args.output / "portable-shadow.onnx")
    onnx.checker.check_model(str(args.output / "portable-shadow.onnx"))
    report.update(source=str(source_path), source_sha256=source_hash,
                  generator_sha256=digest(Path(__file__)), output_sha256=digest(output),
                  external_data=copies, shadow_sha256=digest(args.output / "portable-shadow.onnx"))
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "frequency_words"}, indent=2))


if __name__ == "__main__":
    main()
