import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")

SPEC = importlib.util.spec_from_file_location(
    "lower_onnx_split_sequences",
    Path(__file__).resolve().parents[2] / "tools/lower_onnx_split_sequences.py",
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def tensor(name, element_type, shape):
    return onnx.helper.make_tensor_value_info(name, element_type, shape)


def make_model(tmp_path):
    split_size = onnx.helper.make_tensor(
        "split_size", onnx.TensorProto.INT64, [], np.asarray(2, dtype=np.int64)
    )
    first = onnx.helper.make_tensor(
        "first", onnx.TensorProto.INT64, [], np.asarray(0, dtype=np.int64)
    )
    second = onnx.helper.make_tensor(
        "second", onnx.TensorProto.INT64, [], np.asarray(1, dtype=np.int64)
    )
    nodes = [
        onnx.helper.make_node(
            "SplitToSequence", ["value", "split_size"], ["parts"], axis=0
        ),
        onnx.helper.make_node("SequenceAt", ["parts", "first"], ["part0"]),
        onnx.helper.make_node("SequenceAt", ["parts", "second"], ["part1"]),
        onnx.helper.make_node("CastLike", ["part0", "other_float"], ["same"]),
        onnx.helper.make_node("CastLike", ["part1", "other_int"], ["cast"]),
        onnx.helper.make_node("Shape", ["part0"], ["shape"], start=0),
        onnx.helper.make_node(
            "ScatterND",
            ["data", "indices", "updates"],
            ["scattered"],
            reduction="none",
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "lowering_fixture",
        [
            tensor("value", onnx.TensorProto.FLOAT, [4]),
            tensor("other_float", onnx.TensorProto.FLOAT, [2]),
            tensor("other_int", onnx.TensorProto.INT64, [2]),
            tensor("data", onnx.TensorProto.FLOAT, [2, 2]),
            tensor("indices", onnx.TensorProto.INT64, [1, 1]),
            tensor("updates", onnx.TensorProto.FLOAT, [1, 2]),
        ],
        [
            tensor("same", onnx.TensorProto.FLOAT, [2]),
            tensor("cast", onnx.TensorProto.INT64, [2]),
            tensor("shape", onnx.TensorProto.INT64, [1]),
            tensor("scattered", onnx.TensorProto.FLOAT, [2, 2]),
        ],
        [split_size, first, second],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    source = tmp_path / "source.onnx"
    onnx.save_model(model, source)
    return source


def test_lowers_sequence_access_and_cast_like_generically(tmp_path):
    source = make_model(tmp_path)
    output = tmp_path / "lowered.onnx"
    report = TOOL.lower(source, output)
    model = onnx.load_model(output)

    assert report["lowered_split_to_sequence"] == 1
    assert report["lowered_slices"] == 2
    assert report["lowered_cast_like"] == 2
    assert report["cast_like_to_cast"] == 1
    assert report["cast_like_to_identity"] == 1
    assert report["lowered_shape"] == 1
    assert report["normalized_scatter_nd_reduction"] == 1
    assert not any(node.op_type in {"SplitToSequence", "SequenceAt", "CastLike", "Shape"}
                   for node in model.graph.node)
    assert sum(node.op_type == "Slice" for node in model.graph.node) == 2
    assert sum(node.op_type == "Identity" for node in model.graph.node) == 1
    cast = next(node for node in model.graph.node if node.op_type == "Cast")
    assert onnx.helper.get_attribute_value(cast.attribute[0]) == onnx.TensorProto.INT64
    shape_constant = next(
        node for node in model.graph.node
        if node.op_type == "Constant" and node.output[0] == "shape"
    )
    shape_value = onnx.numpy_helper.to_array(
        onnx.helper.get_attribute_value(shape_constant.attribute[0])
    )
    assert shape_value.tolist() == [2]
    scatter = next(node for node in model.graph.node if node.op_type == "ScatterND")
    assert not any(attribute.name == "reduction" for attribute in scatter.attribute)
    assert [value.name for value in model.graph.output] == [
        "same",
        "cast",
        "shape",
        "scattered",
    ]


def test_lowering_is_idempotent(tmp_path):
    source = make_model(tmp_path)
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    TOOL.lower(source, first)
    report = TOOL.lower(first, second)

    assert report["lowered_split_to_sequence"] == 0
    assert report["lowered_slices"] == 0
    assert report["lowered_cast_like"] == 0
    assert report["cast_like_to_cast"] == 0
    assert report["cast_like_to_identity"] == 0
    assert report["lowered_shape"] == 0
    assert report["normalized_scatter_nd_reduction"] == 0


def test_f64_island_demotion_is_explicit(tmp_path):
    one = onnx.helper.make_tensor(
        "one", onnx.TensorProto.DOUBLE, [1], np.asarray([1.0])
    )
    nodes = [
        onnx.helper.make_node(
            "Cast", ["value"], ["wide"], to=onnx.TensorProto.DOUBLE
        ),
        onnx.helper.make_node("Add", ["wide", "one"], ["sum"]),
        onnx.helper.make_node("Sin", ["sum"], ["sine"]),
        onnx.helper.make_node(
            "Cast", ["sine"], ["output"], to=onnx.TensorProto.FLOAT
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "f64_island",
        [tensor("value", onnx.TensorProto.FLOAT, [1])],
        [tensor("output", onnx.TensorProto.FLOAT, [1])],
        [one],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    source = tmp_path / "f64.onnx"
    onnx.save_model(model, source)

    preserved = tmp_path / "preserved.onnx"
    preserved_report = TOOL.lower(source, preserved)
    assert preserved_report["f64_demoted_to_f32"] is False
    assert preserved_report["f64_values_remaining"] > 0

    demoted = tmp_path / "demoted.onnx"
    demoted_report = TOOL.lower(
        source, demoted, demote_f64_island=True
    )
    lowered_model = onnx.load_model(demoted)
    assert demoted_report["f64_demoted_to_f32"] is True
    assert demoted_report["f64_values_remaining"] == 0
    assert demoted_report["requires_numerical_verification"] is True
    assert not any(
        value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        for value in (
            *lowered_model.graph.input,
            *lowered_model.graph.output,
            *lowered_model.graph.value_info,
        )
    )
    assert all(
        initializer.data_type != onnx.TensorProto.DOUBLE
        for initializer in lowered_model.graph.initializer
    )


def test_bfloat16_promotion_is_explicit(tmp_path):
    one = onnx.helper.make_tensor(
        "one", onnx.TensorProto.BFLOAT16, [1], [16256]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["value"], ["output"])],
        "bfloat16_abi",
        [tensor("value", onnx.TensorProto.BFLOAT16, [1])],
        [tensor("output", onnx.TensorProto.BFLOAT16, [1])],
        [one],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    source = tmp_path / "bfloat16.onnx"
    onnx.save_model(model, source)

    preserved = tmp_path / "preserved-bfloat16.onnx"
    preserved_report = TOOL.lower(source, preserved)
    assert preserved_report["bfloat16_promoted_to_f32"] is False
    assert preserved_report["bfloat16_values_remaining"] > 0

    promoted = tmp_path / "promoted-f32.onnx"
    promoted_report = TOOL.lower(
        source, promoted, promote_bfloat16_to_f32=True
    )
    lowered_model = onnx.load_model(promoted)
    assert promoted_report["bfloat16_promoted_to_f32"] is True
    assert promoted_report["bfloat16_values_remaining"] == 0
    assert promoted_report["requires_numerical_verification"] is True
    assert not any(
        value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
        for value in (
            *lowered_model.graph.input,
            *lowered_model.graph.output,
            *lowered_model.graph.value_info,
        )
    )
