#!/usr/bin/env python3
"""Lower selected ONNX operators to equivalent Horizon-friendly primitives.

The pass is operator-based and model-independent. It only rewrites patterns
whose numerical equivalence can be checked independently:

* ``Tanh(x)`` -> ``2 * Sigmoid(2 * x) - 1``
* last-axis ``LayerNormalization`` -> ReduceMean/Sub/Mul/Add/Sqrt/Div primitives
* static contiguous ``ScatterND`` -> ``Slice + Concat``

The report explicitly marks numerical verification as required. A successful
rewrite is not a parity certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_element_type(value: Any) -> int | None:
    value_type = value.type
    if value_type.WhichOneof("value") != "tensor_type":
        return None
    element_type = value_type.tensor_type.elem_type
    return int(element_type) if element_type else None


def tensor_shape(value: Any) -> tuple[int, ...] | None:
    value_type = value.type
    if value_type.WhichOneof("value") != "tensor_type":
        return None
    dimensions: list[int] = []
    for dimension in value_type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            return None
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def _static_scatter_indices(
    input_path: Path,
    scatter_nodes: list[Any],
) -> dict[str, dict[str, Any]]:
    """Evaluate ScatterND index tensors on shape-compatible probe inputs.

    The probe is deliberately conservative: an index tensor is considered
    static only when three distinct input patterns produce exactly the same
    tensor. The probe model is written next to symlinked external weights so
    large models are not duplicated in memory.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort

    if not scatter_nodes:
        return {}

    probe = onnx.load(str(input_path), load_external_data=False)
    probe_nodes = {
        node.name: node for node in probe.graph.node if node.op_type == "ScatterND"
    }
    value_types = {
        value.name: (
            int(value.type.tensor_type.elem_type),
            tensor_shape(value),
        )
        for value in (
            *probe.graph.input,
            *probe.graph.output,
            *probe.graph.value_info,
        )
    }
    for initializer in probe.graph.initializer:
        value_types[initializer.name] = (
            int(initializer.data_type),
            tuple(int(value) for value in initializer.dims),
        )
    index_output_names: dict[str, str] = {}
    metadata_output_names: dict[str, tuple[str, str]] = {}
    for index, original in enumerate(scatter_nodes):
        node = probe_nodes.get(original.name)
        if node is None:
            continue
        required = [node.input[1], node.input[2], node.output[0]]
        if any(name not in value_types for name in required):
            continue
        base = f"__vf_scatter_probe_{index}"
        indices_name = f"{base}_indices"
        updates_name = f"{base}_updates"
        output_name = f"{base}_output"
        probe.graph.node.extend(
            [
                onnx.helper.make_node(
                    "Identity",
                    [node.input[1]],
                    [indices_name],
                    name=f"{base}_indices_identity",
                ),
                onnx.helper.make_node(
                    "Identity",
                    [node.input[2]],
                    [updates_name],
                    name=f"{base}_updates_identity",
                ),
                onnx.helper.make_node(
                    "Identity",
                    [node.output[0]],
                    [output_name],
                    name=f"{base}_output_identity",
                ),
            ]
        )
        for name, source in zip(
            (indices_name, updates_name, output_name), required, strict=True
        ):
            element_type, shape = value_types[source]
            probe.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    name, element_type, shape
                )
            )
        index_output_names[original.name] = indices_name
        metadata_output_names[original.name] = (updates_name, output_name)

    if not index_output_names:
        return {}

    with tempfile.TemporaryDirectory(prefix="vlaforge-scatter-probe-") as temp:
        root = Path(temp)
        locations: set[str] = set()
        for initializer in probe.graph.initializer:
            for entry in initializer.external_data:
                if entry.key == "location":
                    locations.add(entry.value)
        for location in locations:
            source = input_path.parent / location
            if not source.is_file():
                return {}
            destination = root / location
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.symlink(source.resolve(), destination)
        model_path = root / "probe.onnx"
        onnx.save_model(probe, str(model_path))

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(8, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        output_metadata = {value.name: value for value in session.get_outputs()}

        graph_inputs = []
        for value in probe.graph.input:
            value_type = value.type.tensor_type
            if not value.type.HasField("tensor_type"):
                return {}
            shape = tensor_shape(value)
            if shape is None:
                return {}
            dtype = np.dtype(
                {
                    onnx.TensorProto.FLOAT: np.float32,
                    onnx.TensorProto.FLOAT16: np.float16,
                    onnx.TensorProto.DOUBLE: np.float64,
                    onnx.TensorProto.INT64: np.int64,
                    onnx.TensorProto.INT32: np.int32,
                    onnx.TensorProto.INT16: np.int16,
                    onnx.TensorProto.INT8: np.int8,
                    onnx.TensorProto.UINT8: np.uint8,
                    onnx.TensorProto.BOOL: np.bool_,
                }.get(int(value_type.elem_type))
            )
            if dtype.kind not in {"f", "i", "u", "b"}:
                return {}
            graph_inputs.append((value.name, shape, dtype))

        feeds: list[dict[str, Any]] = []
        for probe_index, fill in enumerate((0.0, 1.0, -0.5)):
            feed: dict[str, Any] = {}
            for name, shape, dtype in graph_inputs:
                if dtype.kind == "b":
                    value = probe_index != 0
                elif dtype.kind in "iu":
                    value = probe_index
                else:
                    value = fill
                feed[name] = np.full(shape, value, dtype=dtype)
            feeds.append(feed)

        requested = list(index_output_names.values())
        values = [session.run(requested, feed) for feed in feeds]
        results: dict[str, dict[str, Any]] = {}
        for output_index, node_name in enumerate(index_output_names):
            arrays = [np.asarray(run[output_index]) for run in values]
            if any(array.shape != arrays[0].shape for array in arrays[1:]):
                continue
            if any(not np.array_equal(array, arrays[0]) for array in arrays[1:]):
                continue
            updates_name, output_name = metadata_output_names[node_name]
            updates_shape = tuple(output_metadata[updates_name].shape)
            output_shape = tuple(output_metadata[output_name].shape)
            if any(dimension is None for dimension in (*updates_shape, *output_shape)):
                continue
            results[node_name] = {
                "indices": arrays[0],
                "updates_shape": tuple(int(value) for value in updates_shape),
                "output_shape": tuple(int(value) for value in output_shape),
            }
        return results


def lower(
    input_path: Path,
    output_path: Path,
    *,
    tanh_to_sigmoid: bool,
    decompose_layernorm: bool,
    static_scatter_to_concat: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if not tanh_to_sigmoid and not decompose_layernorm and not static_scatter_to_concat:
        raise ValueError("at least one lowering pass must be enabled")
    if output_path.exists():
        raise ValueError(f"lowered output already exists: {output_path}")

    model = onnx.load(str(input_path), load_external_data=True)
    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )

    element_types: dict[str, int] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    ranks: dict[str, int] = {}
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
        if value.type.WhichOneof("value") == "tensor_type":
            ranks[value.name] = len(value.type.tensor_type.shape.dim)
    for initializer in model.graph.initializer:
        if initializer.data_type:
            element_types[initializer.name] = int(initializer.data_type)
        shapes[initializer.name] = tuple(int(value) for value in initializer.dims)

    existing_names = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    existing_names.update(initializer.name for initializer in model.graph.initializer)
    counters: Counter[str] = Counter()

    def unique_name(stem: str) -> str:
        counters[stem] += 1
        candidate = f"{stem}_{counters[stem]}"
        while candidate in existing_names:
            counters[stem] += 1
            candidate = f"{stem}_{counters[stem]}"
        existing_names.add(candidate)
        return candidate

    def scalar_array(value: float, element_type: int) -> Any:
        dtype = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.DOUBLE: np.float64,
        }.get(element_type)
        if dtype is None:
            raise ValueError(
                f"unsupported scalar dtype {element_type}; "
                "Horizon lowering supports float32/float16/float64"
            )
        return np.asarray(value, dtype=dtype)

    lowered_tanh = 0
    lowered_layernorm = 0
    lowered_static_scatter = 0
    skipped_static_scatter = 0
    scatter_probes: dict[str, dict[str, Any]] = {}
    if static_scatter_to_concat:
        scatter_probes = _static_scatter_indices(
            input_path,
            [node for node in model.graph.node if node.op_type == "ScatterND"],
        )
    rebuilt: list[Any] = []
    for node_index, node in enumerate(model.graph.node):
        if tanh_to_sigmoid and node.op_type == "Tanh":
            if len(node.input) != 1 or len(node.output) != 1:
                raise ValueError(f"{node.name}: Tanh must have one input and one output")
            element_type = element_types.get(node.input[0])
            if element_type not in (
                onnx.TensorProto.FLOAT,
                onnx.TensorProto.FLOAT16,
                onnx.TensorProto.DOUBLE,
            ):
                raise ValueError(
                    f"{node.name}: unsupported Tanh input dtype {element_type}"
                )
            base = node.name or f"Tanh_{node_index}"
            two_name = unique_name(f"{base}__vf_tanh_two")
            one_name = unique_name(f"{base}__vf_tanh_one")
            doubled = unique_name(f"{base}__vf_tanh_doubled")
            sigmoid = unique_name(f"{base}__vf_tanh_sigmoid")
            shifted = unique_name(f"{base}__vf_tanh_shifted")
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(
                        scalar_array(2.0, element_type), name=two_name
                    ),
                    numpy_helper.from_array(
                        scalar_array(1.0, element_type), name=one_name
                    ),
                ]
            )
            rebuilt.extend(
                [
                    helper.make_node(
                        "Mul",
                        [node.input[0], two_name],
                        [doubled],
                        name=f"{base}__vf_tanh_mul_2",
                    ),
                    helper.make_node(
                        "Sigmoid",
                        [doubled],
                        [sigmoid],
                        name=f"{base}__vf_tanh_sigmoid",
                    ),
                    helper.make_node(
                        "Mul",
                        [sigmoid, two_name],
                        [shifted],
                        name=f"{base}__vf_tanh_mul_2_out",
                    ),
                    helper.make_node(
                        "Sub",
                        [shifted, one_name],
                        [node.output[0]],
                        name=f"{base}__vf_tanh_sub_1",
                    ),
                ]
            )
            lowered_tanh += 1
            continue

        if decompose_layernorm and node.op_type == "LayerNormalization":
            if len(node.input) < 2 or len(node.output) != 1:
                raise ValueError(
                    f"{node.name}: LayerNormalization requires scale and one output"
                )
            if len(node.input) > 3:
                raise ValueError(f"{node.name}: unsupported LayerNormalization arity")
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            axis = int(attributes.get("axis", -1))
            epsilon = float(attributes.get("epsilon", 1e-5))
            rank = ranks.get(node.input[0], 0)
            if rank <= 0:
                raise ValueError(
                    f"{node.name}: LayerNormalization input rank is unavailable"
                )
            normalized_axis = axis if axis >= 0 else rank + axis
            if normalized_axis != rank - 1:
                raise ValueError(
                    f"{node.name}: only last-axis LayerNormalization is supported"
                )
            element_type = element_types.get(node.input[0])
            if element_type != onnx.TensorProto.FLOAT:
                raise ValueError(
                    f"{node.name}: LayerNormalization lowering currently requires "
                    "float32 input"
                )
            base = node.name or f"LayerNormalization_{node_index}"
            axes_name = unique_name(f"{base}__vf_layernorm_axes")
            epsilon_name = unique_name(f"{base}__vf_layernorm_epsilon")
            mean = unique_name(f"{base}__vf_layernorm_mean")
            centered = unique_name(f"{base}__vf_layernorm_centered")
            square = unique_name(f"{base}__vf_layernorm_square")
            variance = unique_name(f"{base}__vf_layernorm_variance")
            stabilized = unique_name(f"{base}__vf_layernorm_stabilized")
            standard_deviation = unique_name(f"{base}__vf_layernorm_standard_deviation")
            normalized = unique_name(f"{base}__vf_layernorm_normalized")
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(
                        np.asarray([rank - 1], dtype=np.int64), name=axes_name
                    ),
                    numpy_helper.from_array(
                        scalar_array(epsilon, element_type), name=epsilon_name
                    ),
                ]
            )
            rebuilt.extend(
                [
                    helper.make_node(
                        "ReduceMean",
                        [node.input[0], axes_name],
                        [mean],
                        name=f"{base}__vf_layernorm_reduce_mean",
                        keepdims=1,
                    ),
                    helper.make_node(
                        "Sub",
                        [node.input[0], mean],
                        [centered],
                        name=f"{base}__vf_layernorm_sub",
                    ),
                    helper.make_node(
                        "Mul",
                        [centered, centered],
                        [square],
                        name=f"{base}__vf_layernorm_square",
                    ),
                    helper.make_node(
                        "ReduceMean",
                        [square, axes_name],
                        [variance],
                        name=f"{base}__vf_layernorm_reduce_variance",
                        keepdims=1,
                    ),
                    helper.make_node(
                        "Add",
                        [variance, epsilon_name],
                        [stabilized],
                        name=f"{base}__vf_layernorm_add_epsilon",
                    ),
                    helper.make_node(
                        "Sqrt",
                        [stabilized],
                        [standard_deviation],
                        name=f"{base}__vf_layernorm_sqrt",
                    ),
                    helper.make_node(
                        "Div",
                        [centered, standard_deviation],
                        [normalized],
                        name=f"{base}__vf_layernorm_normalize",
                    ),
                ]
            )
            final_value = normalized
            if len(node.input) >= 2 and node.input[1]:
                scaled = unique_name(f"{base}__vf_layernorm_scaled")
                rebuilt.append(
                    helper.make_node(
                        "Mul",
                        [final_value, node.input[1]],
                        [scaled],
                        name=f"{base}__vf_layernorm_scale",
                    )
                )
                final_value = scaled
            if len(node.input) == 3 and node.input[2]:
                shifted = unique_name(f"{base}__vf_layernorm_shifted")
                rebuilt.append(
                    helper.make_node(
                        "Add",
                        [final_value, node.input[2]],
                        [shifted],
                        name=f"{base}__vf_layernorm_bias",
                    )
                )
                final_value = shifted
            if final_value != node.output[0]:
                rebuilt.append(
                    helper.make_node(
                        "Identity",
                        [final_value],
                        [node.output[0]],
                        name=f"{base}__vf_layernorm_output",
                    )
                )
            lowered_layernorm += 1
            continue

        if static_scatter_to_concat and node.op_type == "ScatterND":
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            reduction = attributes.get("reduction", b"none")
            if reduction not in (b"none", "none"):
                skipped_static_scatter += 1
                rebuilt.append(node)
                continue
            probe = scatter_probes.get(node.name)
            if probe is None:
                skipped_static_scatter += 1
                rebuilt.append(node)
                continue
            indices = probe["indices"]
            updates_shape = probe["updates_shape"]
            output_shape = probe["output_shape"]
            rank = len(output_shape)
            if (
                rank < 1
                or len(updates_shape) != rank
                or indices.ndim < 2
                or indices.shape[-1] != rank
                or indices.shape[-2] != updates_shape[-1]
                or updates_shape[:-1] != output_shape[:-1]
            ):
                skipped_static_scatter += 1
                rebuilt.append(node)
                continue
            block_length = int(updates_shape[-1])
            if (
                block_length < 1
                or any(dimension < 0 for dimension in output_shape)
                or output_shape[-1] < block_length
            ):
                skipped_static_scatter += 1
                rebuilt.append(node)
                continue
            last_coordinates = indices[..., -1].reshape(-1, block_length)
            start = int(last_coordinates[0, 0])
            expected_last = np.arange(
                start, start + block_length, dtype=last_coordinates.dtype
            )
            if (
                start < 0
                or start + block_length > output_shape[-1]
                or not np.array_equal(last_coordinates[0], expected_last)
                or not np.all(last_coordinates == expected_last, axis=1).all()
            ):
                skipped_static_scatter += 1
                rebuilt.append(node)
                continue

            prefix_shape = output_shape[:-1]
            prefix_coordinates = indices[..., :-1].reshape(-1, rank - 1)
            expected_prefix = np.indices(
                prefix_shape, dtype=indices.dtype
            ).reshape(rank - 1, -1)
            expected_prefix = np.repeat(
                np.moveaxis(expected_prefix, 0, -1), block_length, axis=0
            )
            if not np.array_equal(prefix_coordinates, expected_prefix):
                skipped_static_scatter += 1
                rebuilt.append(node)
                continue

            axis_name = unique_name(f"{node.name}__vf_scatter_axis")
            steps_name = unique_name(f"{node.name}__vf_scatter_steps")
            model.graph.initializer.extend(
                [
                    numpy_helper.from_array(
                        np.asarray([rank - 1], dtype=np.int64), name=axis_name
                    ),
                    numpy_helper.from_array(
                        np.asarray([1], dtype=np.int64), name=steps_name
                    ),
                ]
            )
            pieces: list[str] = []
            if start > 0:
                before = unique_name(f"{node.name}__vf_scatter_before")
                starts_name = unique_name(f"{node.name}__vf_scatter_starts")
                ends_before_name = unique_name(
                    f"{node.name}__vf_scatter_ends_before"
                )
                model.graph.initializer.extend(
                    [
                        numpy_helper.from_array(
                            np.asarray([0], dtype=np.int64), name=starts_name
                        ),
                        numpy_helper.from_array(
                            np.asarray([start], dtype=np.int64),
                            name=ends_before_name,
                        ),
                    ]
                )
                rebuilt.append(
                    helper.make_node(
                        "Slice",
                        [
                            node.input[0],
                            starts_name,
                            ends_before_name,
                            axis_name,
                            steps_name,
                        ],
                        [before],
                        name=f"{node.name}__vf_scatter_before_slice",
                    )
                )
                pieces.append(before)
            pieces.append(node.input[2])
            if start + block_length < output_shape[-1]:
                after = unique_name(f"{node.name}__vf_scatter_after")
                ends_after_name = unique_name(f"{node.name}__vf_scatter_ends_after")
                ends_dim_name = unique_name(f"{node.name}__vf_scatter_ends_dim")
                model.graph.initializer.extend(
                    [
                        numpy_helper.from_array(
                            np.asarray([start + block_length], dtype=np.int64),
                            name=ends_after_name,
                        ),
                        numpy_helper.from_array(
                            np.asarray([output_shape[-1]], dtype=np.int64),
                            name=ends_dim_name,
                        ),
                    ]
                )
                rebuilt.append(
                    helper.make_node(
                        "Slice",
                        [
                            node.input[0],
                            ends_after_name,
                            ends_dim_name,
                            axis_name,
                            steps_name,
                        ],
                        [after],
                        name=f"{node.name}__vf_scatter_after_slice",
                    )
                )
                pieces.append(after)
            if len(pieces) == 1:
                rebuilt.append(
                    helper.make_node(
                        "Identity",
                        pieces,
                        [node.output[0]],
                        name=f"{node.name}__vf_scatter_identity",
                    )
                )
            else:
                rebuilt.append(
                    helper.make_node(
                        "Concat",
                        pieces,
                        [node.output[0]],
                        axis=rank - 1,
                        name=f"{node.name}__vf_scatter_concat",
                    )
                )
            lowered_static_scatter += 1
            continue

        rebuilt.append(node)

    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    model = onnx.shape_inference.infer_shapes(
        model, strict_mode=False, data_prop=True
    )
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.bin",
        size_threshold=1024,
        convert_attribute=False,
    )
    external_weights = output_path.parent / "weights.bin"
    op_counts = Counter(node.op_type for node in model.graph.node)
    return {
        "schema": "vlaforge.horizon_onnx_operator_lowering/1",
        "status": "passed",
        "source": str(input_path.resolve()),
        "source_sha256": digest(input_path),
        "output": str(output_path.resolve()),
        "output_sha256": digest(output_path),
        "output_size": output_path.stat().st_size,
        "external_weights": (
            str(external_weights.resolve()) if external_weights.exists() else None
        ),
        "external_weights_sha256": (
            digest(external_weights) if external_weights.exists() else None
        ),
        "enabled_passes": {
            "tanh_to_sigmoid": tanh_to_sigmoid,
            "decompose_last_axis_layernorm": decompose_layernorm,
            "static_scatter_to_concat": static_scatter_to_concat,
        },
        "lowered_tanh": lowered_tanh,
        "lowered_layernorm": lowered_layernorm,
        "lowered_static_scatter": lowered_static_scatter,
        "skipped_static_scatter": skipped_static_scatter,
        "static_scatter_probe_count": len(scatter_probes),
        "op_counts": dict(sorted(op_counts.items())),
        "requires_numerical_verification": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tanh-to-sigmoid", action="store_true")
    parser.add_argument("--decompose-layernorm", action="store_true")
    parser.add_argument("--static-scatter-to-concat", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    report_path = args.report.resolve()
    if report_path.exists():
        raise ValueError(f"lowering report already exists: {report_path}")
    report = lower(
        source,
        output,
        tanh_to_sigmoid=args.tanh_to_sigmoid,
        decompose_layernorm=args.decompose_layernorm,
        static_scatter_to_concat=args.static_scatter_to_concat,
    )
    write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
