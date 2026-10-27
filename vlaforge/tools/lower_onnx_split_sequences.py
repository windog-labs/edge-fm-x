#!/usr/bin/env python3
"""Lower Horizon-unsupported ONNX forms without changing graph semantics.

Some exporters represent ``torch.split(...)`` as ``SplitToSequence`` followed
by constant ``SequenceAt`` nodes. Horizon's ONNX frontend supports ``Slice``
but not the sequence pair. It also does not support ``CastLike``. This pass
performs equivalent static-shape and static-dtype lowering while preserving
all unrelated graph nodes and data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def scalar_constant(node_value: Any) -> int:
    if node_value.size != 1:
        raise ValueError(f"expected scalar constant, got shape {node_value.shape}")
    return int(node_value.reshape(-1)[0])


def tensor_element_type(value: Any) -> int | None:
    value_type = value.type
    if value_type.WhichOneof("value") != "tensor_type":
        return None
    element_type = value_type.tensor_type.elem_type
    return int(element_type) if element_type else None


def tensor_shape(value: Any) -> tuple[int, ...] | None:
    value_type = value.type
    if (value_type.WhichOneof("value") != "tensor_type"
            or not value_type.tensor_type.HasField("shape")):
        return None
    dimensions = []
    for dimension in value_type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            return None
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def tensor_messages(message):
    """Include sparse tensors, nested graph attributes, and local functions."""
    import onnx

    if isinstance(message, onnx.TensorProto):
        yield message
        return
    for field, value in message.ListFields():
        if field.message_type is not None:
            for child in value if field.is_repeated else (value,):
                yield from tensor_messages(child)


def external_helpers():
    # Also work when this script is loaded directly with importlib by callers.
    spec = importlib.util.spec_from_file_location(
        "_vf_sequence_export_helpers",
        Path(__file__).with_name("export_torchscript_region_onnx.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.external_files, module.publish_files


def external_span(tensor, root: Path) -> tuple[Path, int, int]:
    entries = {entry.key: entry.value for entry in tensor.external_data}
    if len(entries) != len(tensor.external_data):
        raise ValueError(f"{tensor.name}: duplicate external data keys")
    for key in ("offset", "length"):
        if key in entries and (not entries[key].isascii() or not entries[key].isdigit()):
            raise ValueError(f"{tensor.name}: invalid external {key}")
    source = root / entries["location"]
    offset = int(entries.get("offset", "0"))
    size = source.stat().st_size
    length = int(entries["length"]) if "length" in entries else size - offset
    if offset > size or length < 0 or length > size - offset:
        raise ValueError(f"{tensor.name}: external byte range exceeds file size")
    if any(tensor.HasField(key) for key in ("raw_data", "segment")) or any(
        getattr(tensor, key) for key in (
            "float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data"
        )
    ):
        raise ValueError(f"{tensor.name}: external tensor also has inline data or segment")
    return source, offset, length


def publication_paths(output: Path, sidecars: list[Path]) -> None:
    paths = [*sidecars, Path(output.name)]
    if len(set(paths)) != len(paths) or any(
        left in right.parents for left in paths for right in paths if left != right
    ):
        raise ValueError("external tensor location aliases the ONNX graph or another file")
    for relative in paths:
        destination = output.parent / relative
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        for parent in relative.parents:
            candidate = output.parent / parent
            if parent != Path(".") and candidate.is_symlink():
                raise ValueError("external tensor destination must not use symlinks")


def copy_external(source: Path, destination: Path, location: Path) -> dict[str, Any]:
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(8 << 20), b""):
            checksum.update(block)
            writer.write(block)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
    ):
        raise ValueError(f"external source changed while copying: {source}")
    copied_hash = digest(destination)
    if copied_hash != checksum.hexdigest():
        raise ValueError(f"external copy checksum mismatch: {source}")
    return {"location": str(location), "size_bytes": before.st_size,
            "source_sha256": checksum.hexdigest(), "output_sha256": copied_hash}


def lower(
    input_path: Path,
    output_path: Path,
    *,
    demote_f64_island: bool = False,
    promote_bfloat16_to_f32: bool = False,
    lower_shape_nodes: bool = True,
    demote_spurious_complex_casts: bool = False,
    normalize_cumsum_bool: bool = False,
) -> dict[str, Any]:
    import onnx
    import numpy as np
    from onnx import helper, numpy_helper

    input_path = input_path.resolve(strict=True)
    output_path = output_path.absolute()
    if os.path.lexists(output_path):
        raise FileExistsError(output_path)
    source_hash = digest(input_path)
    model = onnx.load(str(input_path), load_external_data=False)
    if digest(input_path) != source_hash:
        raise ValueError("source ONNX graph changed while loading")
    external_files, publish_files = external_helpers()
    sidecars = external_files(model, input_path.parent)
    publication_paths(output_path, sidecars)
    external_tensors = []
    for tensor in tensor_messages(model):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            external_span(tensor, input_path.parent)
            external_tensors.append(tensor.SerializeToString())
        elif tensor.external_data:
            raise ValueError(f"{tensor.name}: external metadata without EXTERNAL data location")
    if sidecars and any((demote_f64_island, promote_bfloat16_to_f32,
                         demote_spurious_complex_casts, normalize_cumsum_bool)):
        raise ValueError("dtype-changing flags are unsupported with external tensors; "
                         "use an explicit precision conversion before structural lowering")

    # Index tensor metadata; only SplitToSequence/SequenceAt scalar controls
    # are decoded. In particular, do not convert neural weights to NumPy.
    constants = {value.name: value for value in model.graph.initializer}
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if attribute.name == "value" and attribute.type == onnx.AttributeProto.TENSOR:
                constants[node.output[0]] = attribute.t
            elif attribute.name == "value_int":
                constants[node.output[0]] = helper.make_tensor(
                    node.output[0], onnx.TensorProto.INT64, [], [attribute.i]
                )
    controls: dict[str, int] = {}
    external_controls: dict[str, tuple[Path, int, bytes]] = {}
    external_control_bytes = 0
    graph_input_names = {value.name for value in model.graph.input}
    for node in model.graph.node:
        if (node.domain not in ("", "ai.onnx")
                or node.op_type not in ("SplitToSequence", "SequenceAt") or len(node.input) < 2):
            continue
        name = node.input[1]
        if name in graph_input_names:
            raise ValueError(f"{name}: sequence control is a runtime graph input, not a constant")
        if name in controls or name not in constants:
            continue
        control = constants[name]
        if control.dims or control.data_type not in (onnx.TensorProto.INT32, onnx.TensorProto.INT64):
            raise ValueError(f"{name}: sequence control must be a scalar integer tensor")
        size = 8 if control.data_type == onnx.TensorProto.INT64 else 4
        if control.data_location == onnx.TensorProto.EXTERNAL:
            source, offset, length = external_span(control, input_path.parent)
            if length != size:
                raise ValueError(f"{name}: external scalar byte length must be {size}")
            with source.open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(size)
            if len(raw) != size:
                raise ValueError(f"{name}: truncated external scalar")
            controls[name] = int.from_bytes(raw, byteorder="little", signed=True)
            external_controls[name] = (source.relative_to(input_path.parent), offset, raw)
            external_control_bytes += size
        else:
            if control.HasField("raw_data") and len(control.raw_data) != size:
                raise ValueError(f"{name}: invalid inline scalar byte length")
            if len(control.int32_data) + len(control.int64_data) > 1:
                raise ValueError(f"{name}: invalid scalar payload")
            controls[name] = scalar_constant(numpy_helper.to_array(control))

    def infer_metadata(current):
        # Shape inference may need the split size. Inline only the already
        # resolved scalar controls in a metadata copy, then restore descriptors.
        probe = onnx.ModelProto()
        probe.CopyFrom(current)
        for initializer in probe.graph.initializer:
            if initializer.name in controls:
                initializer.CopyFrom(helper.make_tensor(
                    initializer.name, initializer.data_type, [], [controls[initializer.name]]
                ))
        for node in probe.graph.node:
            if node.op_type == "Constant" and len(node.output) == 1 and node.output[0] in controls:
                for attribute in node.attribute:
                    if attribute.name == "value":
                        attribute.t.CopyFrom(helper.make_tensor(
                            attribute.t.name, attribute.t.data_type, [], [controls[node.output[0]]]
                        ))
        inferred = onnx.shape_inference.infer_shapes(probe, strict_mode=False, data_prop=False)
        for initializer in inferred.graph.initializer:
            if initializer.name in controls:
                initializer.CopyFrom(constants[initializer.name])
        for node in inferred.graph.node:
            if node.op_type == "Constant" and len(node.output) == 1 and node.output[0] in controls:
                for attribute in node.attribute:
                    if attribute.name == "value":
                        attribute.t.CopyFrom(constants[node.output[0]])
        return inferred

    source_ports = ([value.SerializeToString() for value in model.graph.input],
                    [value.SerializeToString() for value in model.graph.output])
    model = infer_metadata(model)

    occupied = {value.name for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    occupied.update(value.name for value in model.graph.initializer)
    for node in model.graph.node:
        occupied.update((*node.input, *node.output, node.name))

    def unique(name):
        actual = name
        while actual in occupied:
            actual += "_"
        occupied.add(actual)
        return actual

    element_types: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    for value in (
        *model.graph.input,
        *model.graph.output,
        *model.graph.value_info,
    ):
        element_type = tensor_element_type(value)
        if element_type is not None:
            element_types[value.name] = element_type
        shape = tensor_shape(value)
        if shape is not None:
            shapes[value.name] = shape
    for initializer in model.graph.initializer:
        if initializer.data_type:
            element_types[initializer.name] = int(initializer.data_type)
        shapes[initializer.name] = tuple(int(value) for value in initializer.dims)

    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                element_types.setdefault(node.output[0], int(attribute.t.data_type))
                shapes.setdefault(node.output[0], tuple(attribute.t.dims))

    bfloat16_values = {
        name
        for name, element_type in element_types.items()
        if element_type == onnx.TensorProto.BFLOAT16
    }
    bfloat16_promoted_nodes = 0
    if promote_bfloat16_to_f32 and bfloat16_values:
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        ):
            if tensor_element_type(value) == onnx.TensorProto.BFLOAT16:
                value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        for index, initializer in enumerate(model.graph.initializer):
            if initializer.data_type != onnx.TensorProto.BFLOAT16:
                continue
            value = numpy_helper.to_array(initializer).astype(np.float32)
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(value, name=initializer.name)
            )
        for node in model.graph.node:
            if not any(
                name in bfloat16_values
                for name in (*node.input, *node.output)
            ):
                continue
            bfloat16_promoted_nodes += 1
            for attribute in node.attribute:
                if (
                    attribute.name == "value"
                    and attribute.t.data_type == onnx.TensorProto.BFLOAT16
                ):
                    value = numpy_helper.to_array(
                        attribute.t
                    ).astype(np.float32)
                    attribute.t.CopyFrom(
                        numpy_helper.from_array(value, name=attribute.t.name)
                    )
                if (
                    node.op_type == "Cast"
                    and attribute.name == "to"
                    and int(attribute.i) == onnx.TensorProto.BFLOAT16
                ):
                    attribute.i = onnx.TensorProto.FLOAT
        for name in bfloat16_values:
            element_types[name] = onnx.TensorProto.FLOAT

    f64_values = {
        name
        for name, element_type in element_types.items()
        if element_type == onnx.TensorProto.DOUBLE
    }
    f64_demoted_nodes = 0
    if demote_f64_island and f64_values:
        graph_io = {
            value.name for value in (*model.graph.input, *model.graph.output)
        }
        exposed = sorted(f64_values & graph_io)
        if exposed:
            raise ValueError(
                "F64 graph inputs or outputs require an explicit external "
                f"boundary: {exposed}"
            )
        for node in model.graph.node:
            if not any(
                name in f64_values for name in (*node.input, *node.output)
            ):
                continue
            f64_demoted_nodes += 1
            for attribute in node.attribute:
                if (
                    attribute.name == "value"
                    and attribute.t.data_type == onnx.TensorProto.DOUBLE
                ):
                    value = numpy_helper.to_array(attribute.t).astype(
                        np.float32
                    )
                    attribute.t.CopyFrom(
                        numpy_helper.from_array(value, name=attribute.t.name)
                    )
                if (
                    node.op_type == "Cast"
                    and attribute.name == "to"
                    and int(attribute.i) == onnx.TensorProto.DOUBLE
                ):
                    attribute.i = onnx.TensorProto.FLOAT
        for index, initializer in enumerate(model.graph.initializer):
            if initializer.data_type != onnx.TensorProto.DOUBLE:
                continue
            value = numpy_helper.to_array(initializer).astype(np.float32)
            model.graph.initializer[index].CopyFrom(
                numpy_helper.from_array(value, name=initializer.name)
            )
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        ):
            if tensor_element_type(value) == onnx.TensorProto.DOUBLE:
                value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        for name in f64_values:
            element_types[name] = onnx.TensorProto.FLOAT

    complex_cast_demoted = 0
    if demote_spurious_complex_casts:
        graph_output_names = {value.name for value in model.graph.output}
        for node in model.graph.node:
            if node.op_type != "Cast" or len(node.input) != 1:
                continue
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            if attributes.get("to") not in (
                onnx.TensorProto.COMPLEX64,
                onnx.TensorProto.COMPLEX128,
            ):
                continue
            output_names = set(node.output)
            if output_names & graph_output_names:
                raise ValueError(
                    f"{node.name}: complex Cast is exposed as a graph output"
                )
            source_type = element_types.get(node.input[0])
            if source_type in (
                onnx.TensorProto.COMPLEX64,
                onnx.TensorProto.COMPLEX128,
            ):
                raise ValueError(
                    f"{node.name}: complex-to-complex Cast is not a "
                    "spurious exporter artifact"
                )
            for attribute in node.attribute:
                if attribute.name == "to":
                    attribute.i = onnx.TensorProto.FLOAT
                    break
            for name in output_names:
                element_types[name] = onnx.TensorProto.FLOAT
            complex_cast_demoted += 1
        retained = []
        for value in model.graph.value_info:
            element_type = tensor_element_type(value)
            if element_type in (
                onnx.TensorProto.COMPLEX64,
                onnx.TensorProto.COMPLEX128,
            ):
                continue
            retained.append(value)
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained)

    cumsum_bool_normalized = 0
    normalized_cumsum_outputs: set[str] = set()
    if normalize_cumsum_bool:
        rebuilt = []
        for node in model.graph.node:
            if (
                node.op_type != "CumSum"
                or element_types.get(node.input[0]) != onnx.TensorProto.BOOL
            ):
                rebuilt.append(node)
                continue
            cast_output = unique(f"{node.name}_input_int64")
            rebuilt.append(
                helper.make_node(
                    "Cast",
                    [node.input[0]],
                    [cast_output],
                    name=unique(f"{node.name}_bool_to_int64"),
                    to=onnx.TensorProto.INT64,
                )
            )
            node.input[0] = cast_output
            rebuilt.append(node)
            normalized_cumsum_outputs.update(node.output)
            for name in node.output:
                element_types[name] = onnx.TensorProto.INT64
            cumsum_bool_normalized += 1
        del model.graph.node[:]
        model.graph.node.extend(rebuilt)
        if normalized_cumsum_outputs:
            retained = [
                value
                for value in model.graph.value_info
                if value.name not in normalized_cumsum_outputs
            ]
            del model.graph.value_info[:]
            model.graph.value_info.extend(retained)

    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    counter = 0

    def constant_node(name: str, value) -> Any:
        nonlocal counter
        counter += 1
        actual = unique(f"{name}_{counter}")
        return helper.make_node(
            "Constant",
            [],
            [actual],
            name=unique(f"{actual}_node"),
            value=numpy_helper.from_array(value, name=f"{actual}_value"),
        )

    replacements: dict[str, Any] = {}
    split_nodes = set()
    lowered = 0
    slices = 0
    for split in model.graph.node:
        if split.domain not in ("", "ai.onnx") or split.op_type != "SplitToSequence":
            continue
        uses = consumers.get(split.output[0], [])
        if (not uses or any(node.domain not in ("", "ai.onnx") or node.op_type != "SequenceAt"
                            or node.input[0] != split.output[0] for node in uses)):
            raise ValueError(
                f"{split.name}: only constant SequenceAt consumers are supported"
            )
        if split.output[0] in {value.name for value in model.graph.output}:
            raise ValueError(f"{split.name}: sequence is exposed as a graph output")

        def nested_uses(message):
            if isinstance(message, onnx.NodeProto) and split.output[0] in message.input:
                return True
            for field, value in message.ListFields():
                if field.message_type is not None:
                    if any(nested_uses(child) for child in (value if field.is_repeated else (value,))):
                        return True
            return False

        if any(nested_uses(attribute) for node in model.graph.node for attribute in node.attribute):
            raise ValueError(f"{split.name}: nested sequence captures are unsupported")
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in split.attribute
        }
        axis = int(attributes.get("axis", 0))
        keepdims = int(attributes.get("keepdims", 1))
        if keepdims != 1:
            raise ValueError(f"{split.name}: keepdims={keepdims} is unsupported")
        if len(split.input) < 2 or split.input[1] not in controls:
            raise ValueError(f"{split.name}: split size is not a constant")
        split_size = controls[split.input[1]]
        if split_size <= 0:
            raise ValueError(f"{split.name}: split size must be positive")

        for sequence_at in uses:
            if len(sequence_at.input) != 2 or sequence_at.input[1] not in controls:
                raise ValueError(
                    f"{sequence_at.name}: sequence index is not a constant"
                )
            index = controls[sequence_at.input[1]]
            if index < 0:
                raise ValueError(
                    f"{sequence_at.name}: negative sequence index unsupported"
                )
            start = index * split_size
            end = start + split_size
            shape = shapes.get(split.input[0])
            if shape is None or not -len(shape) <= axis < len(shape):
                raise ValueError(f"{split.name}: split axis extent must be statically known")
            if start >= shape[axis]:
                raise ValueError(f"{sequence_at.name}: sequence index is out of range")
            starts = constant_node(
                f"vlaforge_slice_starts_{lowered}_{index}",
                np.asarray([start], dtype=np.int64),
            )
            ends = constant_node(
                f"vlaforge_slice_ends_{lowered}_{index}",
                np.asarray([end], dtype=np.int64),
            )
            axes = constant_node(
                f"vlaforge_slice_axes_{lowered}_{index}",
                np.asarray([axis], dtype=np.int64),
            )
            steps = constant_node(
                f"vlaforge_slice_steps_{lowered}_{index}",
                np.asarray([1], dtype=np.int64),
            )
            replacements[sequence_at.output[0]] = [
                starts,
                ends,
                axes,
                steps,
                helper.make_node(
                    "Slice",
                    [
                        split.input[0],
                        starts.output[0],
                        ends.output[0],
                        axes.output[0],
                        steps.output[0],
                    ],
                    list(sequence_at.output),
                    name=unique(f"vlaforge_lower_{sequence_at.name}"),
                ),
            ]
            slices += 1
        split_nodes.add(split.output[0])
        lowered += 1

    old_values = split_nodes

    rebuilt = []
    for node in model.graph.node:
        if any(name in split_nodes for name in node.output):
            continue
        replacement = replacements.get(node.output[0]) if node.output else None
        if replacement is not None:
            rebuilt.extend(replacement)
            continue
        rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)

    lowered_cast_like = 0
    cast_like_to_cast = 0
    cast_like_to_identity = 0
    lowered_shape = 0
    normalized_scatter_nd_reduction = 0
    normalized_cast = 0
    rebuilt = []
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            rebuilt.append(node)
            continue
        if node.op_type == "ScatterND":
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            if attributes.get("reduction") == b"none":
                reduction_index = next(
                    index
                    for index, attribute in enumerate(node.attribute)
                    if attribute.name == "reduction"
                )
                del node.attribute[reduction_index]
                normalized_scatter_nd_reduction += 1
            rebuilt.append(node)
            continue
        if node.op_type == "Shape":
            if not lower_shape_nodes:
                rebuilt.append(node)
                continue
            if len(node.input) != 1 or len(node.output) != 1:
                raise ValueError(
                    f"{node.name}: Shape requires one input and one output"
                )
            shape = shapes.get(node.input[0])
            if shape is None:
                raise ValueError(
                    f"{node.name}: Shape input shape is not statically known"
                )
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            start = int(attributes.get("start", 0))
            end = attributes.get("end")
            end = int(end) if end is not None else None
            value = np.asarray(shape[start:end], dtype=np.int64)
            rebuilt.append(
                helper.make_node(
                    "Constant",
                    [],
                    list(node.output),
                    name=unique(f"vlaforge_lower_{node.name}"),
                    value=numpy_helper.from_array(
                        value, name=f"vlaforge_shape_{lowered_shape}"
                    ),
                )
            )
            element_types[node.output[0]] = onnx.TensorProto.INT64
            shapes[node.output[0]] = tuple(value.shape)
            lowered_shape += 1
            continue
        if node.op_type == "Cast":
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            target_type = attributes.get("to")
            source_type = element_types.get(node.input[0])
            if target_type is not None and source_type == int(target_type):
                rebuilt.append(
                    helper.make_node(
                        "Identity",
                        [node.input[0]],
                        list(node.output),
                        name=unique(f"vlaforge_lower_{node.name}"),
                    )
                )
                normalized_cast += 1
            else:
                rebuilt.append(node)
            continue
        if node.op_type != "CastLike":
            rebuilt.append(node)
            continue
        if len(node.input) != 2 or len(node.output) != 1:
            raise ValueError(
                f"{node.name}: CastLike requires two inputs and one output"
            )
        target_type = element_types.get(node.input[1])
        if target_type is None:
            target_type = element_types.get(node.output[0])
        if target_type is None:
            raise ValueError(
                f"{node.name}: CastLike target dtype is not statically known"
            )
        source_type = element_types.get(node.input[0])
        if source_type == target_type:
            replacement = helper.make_node(
                "Identity",
                [node.input[0]],
                list(node.output),
                name=unique(f"vlaforge_lower_{node.name}"),
            )
            cast_like_to_identity += 1
        else:
            replacement = helper.make_node(
                "Cast",
                [node.input[0]],
                list(node.output),
                name=unique(f"vlaforge_lower_{node.name}"),
                to=target_type,
            )
            cast_like_to_cast += 1
        element_types[node.output[0]] = target_type
        rebuilt.append(replacement)
        lowered_cast_like += 1
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)

    remaining_f64 = sum(
        1
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
        if tensor_element_type(value) == onnx.TensorProto.DOUBLE
    )
    remaining_bfloat16 = sum(
        1
        for value in (
            *model.graph.input,
            *model.graph.output,
            *model.graph.value_info,
        )
        if tensor_element_type(value) == onnx.TensorProto.BFLOAT16
    )

    model = infer_metadata(model)

    reserved_names = {
        value.name for value in model.graph.input
    } | {
        value.name for value in model.graph.output
    } | {
        initializer.name for initializer in model.graph.initializer
    }
    retained_value_info = []
    seen_value_info = set()
    removed_value_info = 0
    for value in model.graph.value_info:
        if (
            value.name in old_values
            or value.name in reserved_names
            or value.name in seen_value_info
        ):
            removed_value_info += 1
            continue
        retained_value_info.append(value)
        seen_value_info.add(value.name)
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    # TensorProto stores contiguous bytes already. A NumPy contiguity sweep
    # would load every external weight and can exceed protobuf's 2 GiB limit.
    if not any((demote_f64_island, promote_bfloat16_to_f32,
                demote_spurious_complex_casts, normalize_cumsum_bool)):
        for values, original in zip((model.graph.input, model.graph.output), source_ports):
            del values[:]
            values.extend(onnx.ValueInfoProto.FromString(value) for value in original)
    if external_tensors != [tensor.SerializeToString() for tensor in tensor_messages(model)
                            if tensor.data_location == onnx.TensorProto.EXTERNAL]:
        raise ValueError("structural lowering changed an external tensor descriptor")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_data = []
    with tempfile.TemporaryDirectory(prefix=".sequence-lowering-", dir=output_path.parent) as directory:
        staged = Path(directory) / output_path.name
        for location in sidecars:
            external_data.append(copy_external(input_path.parent / location, staged.parent / location, location))
        # Bind specialization to the actual staged weight snapshot, even if
        # a control changed before the copy's own consistency window began.
        for name, (location, offset, expected) in external_controls.items():
            with (staged.parent / location).open("rb") as stream:
                stream.seek(offset)
                if stream.read(len(expected)) != expected:
                    raise ValueError(f"{name}: external scalar control changed between resolution and staging")
        onnx.save_model(model, str(staged))
        onnx.checker.check_model(str(staged))
        if digest(input_path) != source_hash:
            raise ValueError("source ONNX graph changed during lowering")
        publication_paths(output_path, sidecars)
        publish_files(staged, output_path, sidecars)
    return {
        "schema": "vlaforge.onnx_split_sequence_lowering/1",
        "status": "passed",
        "source": str(input_path.resolve()),
        "source_sha256": source_hash,
        "output": str(output_path.resolve()),
        "output_sha256": digest(output_path),
        "lowered_split_to_sequence": lowered,
        "lowered_slices": slices,
        "lowered_cast_like": lowered_cast_like,
        "cast_like_to_cast": cast_like_to_cast,
        "cast_like_to_identity": cast_like_to_identity,
        "lowered_shape": lowered_shape,
        "normalized_scatter_nd_reduction": normalized_scatter_nd_reduction,
        "normalized_cast": normalized_cast,
        "shape_nodes_lowered": lower_shape_nodes,
        "spurious_complex_casts_demoted_to_f32": complex_cast_demoted,
        "cumsum_bool_inputs_normalized_to_int64": cumsum_bool_normalized,
        "f64_demoted_to_f32": bool(demote_f64_island and f64_values),
        "f64_demoted_nodes": f64_demoted_nodes,
        "f64_demoted_values": len(f64_values) if demote_f64_island else 0,
        "f64_values_remaining": remaining_f64,
        "requires_numerical_verification": bool(
            (demote_f64_island and f64_values)
            or (promote_bfloat16_to_f32 and bfloat16_values)
        ),
        "bfloat16_promoted_to_f32": bool(
            promote_bfloat16_to_f32 and bfloat16_values
        ),
        "bfloat16_promoted_nodes": bfloat16_promoted_nodes,
        "bfloat16_promoted_values": (
            len(bfloat16_values) if promote_bfloat16_to_f32 else 0
        ),
        "bfloat16_values_remaining": remaining_bfloat16,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "removed_value_info": removed_value_info,
        "materialized_contiguous_constants": 0,
        "external_data": external_data,
        "external_tensor_count": len(external_tensors),
        "external_data_policy": "streamed-byte-preserving-copy",
        "external_scalar_control_bytes_read": external_control_bytes,
        "staged_external_scalar_controls_verified": len(external_controls),
        "scalar_controls_resolved": len(controls),
        "weights_materialized": False,
        "checker": "filename",
        "typed_ports_preserved": source_ports == (
            [value.SerializeToString() for value in model.graph.input],
            [value.SerializeToString() for value in model.graph.output],
        ),
        "shape_inference": True,
        "full_model_output_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--demote-f64-island",
        action="store_true",
        help=(
            "Demote graph-internal F64 values to F32; this requires "
            "independent numerical verification."
        ),
    )
    parser.add_argument(
        "--promote-bfloat16-to-f32",
        action="store_true",
        help=(
            "Promote graph-internal BF16 ABI and constants to F32 for tools "
            "without BF16 calibration support; requires numerical verification."
        ),
    )
    parser.add_argument(
        "--skip-shape-lowering",
        action="store_true",
        help="Preserve dynamic Shape nodes instead of requiring static inputs.",
    )
    parser.add_argument(
        "--demote-spurious-complex-casts",
        action="store_true",
        help=(
            "Rewrite real-to-complex Cast nodes that arise from a legacy "
            "TorchScript exporter dtype bug back to real F32 Cast nodes."
        ),
    )
    parser.add_argument(
        "--normalize-cumsum-bool",
        action="store_true",
        help="Insert the implicit bool-to-int64 promotion required by ONNX CumSum.",
    )
    args = parser.parse_args()
    if os.path.lexists(args.report):
        raise FileExistsError(args.report)
    if args.report.absolute() == args.output.absolute():
        raise ValueError("report location aliases the ONNX graph")
    result = lower(
        args.input,
        args.output,
        demote_f64_island=args.demote_f64_island,
        promote_bfloat16_to_f32=args.promote_bfloat16_to_f32,
        lower_shape_nodes=not args.skip_shape_lowering,
        demote_spurious_complex_casts=args.demote_spurious_complex_casts,
        normalize_cumsum_bool=args.normalize_cumsum_bool,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
