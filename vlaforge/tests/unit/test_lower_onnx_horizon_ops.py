import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")

SPEC = importlib.util.spec_from_file_location(
    "lower_onnx_horizon_ops",
    Path(__file__).resolve().parents[2] / "tools/lower_onnx_horizon_ops.py",
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def tensor(name, element_type, shape):
    return onnx.helper.make_tensor_value_info(name, element_type, shape)


def test_lowers_tanh_and_last_axis_layernorm(tmp_path):
    scale = onnx.helper.make_tensor(
        "scale", onnx.TensorProto.FLOAT, [4], np.ones(4, dtype=np.float32)
    )
    bias = onnx.helper.make_tensor(
        "bias", onnx.TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32)
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Tanh", ["value"], ["tanh"], name="tanh"),
            onnx.helper.make_node(
                "LayerNormalization",
                ["value", "scale", "bias"],
                ["normalized"],
                name="norm",
                axis=-1,
                epsilon=1e-5,
            ),
        ],
        "operator_lowering",
        [tensor("value", onnx.TensorProto.FLOAT, [1, 4])],
        [
            tensor("tanh", onnx.TensorProto.FLOAT, [1, 4]),
            tensor("normalized", onnx.TensorProto.FLOAT, [1, 4]),
        ],
        [scale, bias],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    source = tmp_path / "source.onnx"
    output = tmp_path / "lowered.onnx"
    report = tmp_path / "report.json"
    onnx.save_model(model, source)

    result = TOOL.lower(
        source,
        output,
        tanh_to_sigmoid=True,
        decompose_layernorm=True,
    )
    lowered = onnx.load_model(output)
    ops = [node.op_type for node in lowered.graph.node]

    assert result["lowered_tanh"] == 1
    assert result["lowered_layernorm"] == 1
    assert "Tanh" not in ops
    assert "LayerNormalization" not in ops
    assert ops.count("Sigmoid") == 1
    assert ops.count("ReduceMean") == 2
    assert ops.count("Sqrt") == 1
    assert ops.count("Div") == 1
    assert [value.name for value in lowered.graph.output] == ["tanh", "normalized"]


def test_rewrites_are_idempotent(tmp_path):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Tanh", ["value"], ["output"], name="tanh")],
        "tanh_lowering",
        [tensor("value", onnx.TensorProto.FLOAT, [1, 4])],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 4])],
    )
    source = tmp_path / "source.onnx"
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    onnx.save_model(
        onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        ),
        source,
    )

    first_report = TOOL.lower(
        source,
        first,
        tanh_to_sigmoid=True,
        decompose_layernorm=False,
    )
    second_report = TOOL.lower(
        first,
        second,
        tanh_to_sigmoid=True,
        decompose_layernorm=False,
    )

    assert first_report["lowered_tanh"] == 1
    assert second_report["lowered_tanh"] == 0


def test_rejects_non_last_axis_layernorm(tmp_path):
    scale = onnx.helper.make_tensor(
        "scale", onnx.TensorProto.FLOAT, [2], np.ones(2, dtype=np.float32)
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "LayerNormalization",
                ["value", "scale"],
                ["output"],
                axis=0,
            )
        ],
        "unsupported_layernorm",
        [tensor("value", onnx.TensorProto.FLOAT, [2, 2])],
        [tensor("output", onnx.TensorProto.FLOAT, [2, 2])],
        [scale],
    )
    source = tmp_path / "source.onnx"
    onnx.save_model(
        onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        ),
        source,
    )

    with pytest.raises(ValueError, match="last-axis"):
        TOOL.lower(
            source,
            tmp_path / "output.onnx",
            tanh_to_sigmoid=False,
            decompose_layernorm=True,
        )


def test_rewrites_static_contiguous_scatternd(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    indices = np.asarray(
        [[[0, index] for index in range(32)]],
        dtype=np.int64,
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ScatterND",
                ["data", "indices", "updates"],
                ["output"],
                name="scatter",
            )
        ],
        "static_scatter",
        [
            tensor("data", onnx.TensorProto.FLOAT, [1, 64]),
            tensor("updates", onnx.TensorProto.FLOAT, [1, 32]),
        ],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 64])],
        [onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [1, 32, 2], indices)],
    )
    source = tmp_path / "source.onnx"
    output = tmp_path / "lowered.onnx"
    onnx.save_model(
        onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        ),
        source,
    )

    result = TOOL.lower(
        source,
        output,
        tanh_to_sigmoid=False,
        decompose_layernorm=False,
        static_scatter_to_concat=True,
    )
    lowered = onnx.load_model(output)
    ops = [node.op_type for node in lowered.graph.node]

    assert result["lowered_static_scatter"] == 1
    assert result["skipped_static_scatter"] == 0
    assert "ScatterND" not in ops
    assert ops.count("Slice") == 1
    assert ops.count("Concat") == 1

    feeds = {
        "data": np.arange(64, dtype=np.float32).reshape(1, 64),
        "updates": np.arange(32, dtype=np.float32).reshape(1, 32) + 100,
    }
    expected = feeds["data"].copy()
    expected[:, :32] = feeds["updates"]
    actual = ort.InferenceSession(
        str(output), providers=["CPUExecutionProvider"]
    ).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)


def test_keeps_noncontiguous_scatternd(tmp_path):
    indices = np.asarray(
        [[[0, index] for index in range(0, 64, 2)]],
        dtype=np.int64,
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ScatterND",
                ["data", "indices", "updates"],
                ["output"],
                name="scatter",
            )
        ],
        "noncontiguous_scatter",
        [
            tensor("data", onnx.TensorProto.FLOAT, [1, 64]),
            tensor("updates", onnx.TensorProto.FLOAT, [1, 32]),
        ],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 64])],
        [onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [1, 32, 2], indices)],
    )
    source = tmp_path / "source.onnx"
    output = tmp_path / "lowered.onnx"
    onnx.save_model(
        onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        ),
        source,
    )

    result = TOOL.lower(
        source,
        output,
        tanh_to_sigmoid=False,
        decompose_layernorm=False,
        static_scatter_to_concat=True,
    )
    lowered = onnx.load_model(output)

    assert result["lowered_static_scatter"] == 0
    assert result["skipped_static_scatter"] == 1
    assert [node.op_type for node in lowered.graph.node] == ["ScatterND"]


def test_keeps_scatternd_with_reduction(tmp_path):
    indices = np.asarray(
        [[[0, index] for index in range(32)]],
        dtype=np.int64,
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ScatterND",
                ["data", "indices", "updates"],
                ["output"],
                name="scatter",
                reduction="add",
            )
        ],
        "reduced_scatter",
        [
            tensor("data", onnx.TensorProto.FLOAT, [1, 64]),
            tensor("updates", onnx.TensorProto.FLOAT, [1, 32]),
        ],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 64])],
        [onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [1, 32, 2], indices)],
    )
    source = tmp_path / "source.onnx"
    output = tmp_path / "lowered.onnx"
    onnx.save_model(
        onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 18)],
            ir_version=9,
        ),
        source,
    )

    result = TOOL.lower(
        source,
        output,
        tanh_to_sigmoid=False,
        decompose_layernorm=False,
        static_scatter_to_concat=True,
    )
    lowered = onnx.load_model(output)

    assert result["lowered_static_scatter"] == 0
    assert result["skipped_static_scatter"] == 1
    assert [node.op_type for node in lowered.graph.node] == ["ScatterND"]


def test_data_dependent_indices_cannot_be_proved_static_by_three_probe_patterns(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    indices = np.asarray([[[0, index] for index in range(32)]], dtype=np.int64)
    constants = [onnx.numpy_helper.from_array(value, name) for name, value in (
        ("base_indices", indices), ("one", np.asarray(1, dtype=np.float32)),
        ("half", np.asarray(.5, dtype=np.float32)), ("axes", np.asarray([0], dtype=np.int64)),
        ("zero_prefix", np.asarray([0], dtype=np.int64)))]
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("Sub", ["x", "one"], ["minus_one"]),
        onnx.helper.make_node("Add", ["x", "half"], ["plus_half"]),
        onnx.helper.make_node("Mul", ["x", "minus_one"], ["product"]),
        onnx.helper.make_node("Mul", ["product", "plus_half"], ["varying"]),
        onnx.helper.make_node("Cast", ["varying"], ["shift"], to=onnx.TensorProto.INT64),
        onnx.helper.make_node("Unsqueeze", ["shift", "axes"], ["shift_vector"]),
        onnx.helper.make_node("Concat", ["zero_prefix", "shift_vector"], ["offset"], axis=0),
        onnx.helper.make_node("Add", ["base_indices", "offset"], ["indices"]),
        onnx.helper.make_node("ScatterND", ["data", "indices", "updates"], ["output"], name="scatter")],
        "three_probe_false_positive", [tensor("x", onnx.TensorProto.FLOAT, []),
            tensor("data", onnx.TensorProto.FLOAT, [1, 64]), tensor("updates", onnx.TensorProto.FLOAT, [1, 32])],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 64])], constants)
    source, output = tmp_path / "source.onnx", tmp_path / "lowered.onnx"
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9)
    onnx.save(onnx.shape_inference.infer_shapes(model), source)
    result = TOOL.lower(source, output, tanh_to_sigmoid=False, decompose_layernorm=False, static_scatter_to_concat=True)
    assert result["lowered_static_scatter"] == 0
    feeds = {"x": np.asarray(2, dtype=np.float32), "data": np.arange(64, dtype=np.float32).reshape(1, 64),
             "updates": np.arange(32, dtype=np.float32).reshape(1, 32) + 100}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("names", [("a", "b"), ("same", "same"), ("", "")])
def test_scatter_probe_identity_is_ssa_output_not_optional_node_name(tmp_path, names):
    ort = pytest.importorskip("onnxruntime")
    constants = [onnx.numpy_helper.from_array(np.asarray([[[0, value] for value in range(start, start + 32)]], dtype=np.int64), name)
                 for name, start in (("left_indices", 0), ("right_indices", 32))]
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("ScatterND", ["data", "left_indices", "left"], ["middle"], name=names[0]),
        onnx.helper.make_node("ScatterND", ["middle", "right_indices", "right"], ["output"], name=names[1])],
        "optional_node_names", [tensor("data", onnx.TensorProto.FLOAT, [1, 64]),
            tensor("left", onnx.TensorProto.FLOAT, [1, 32]), tensor("right", onnx.TensorProto.FLOAT, [1, 32])],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 64])], constants)
    source, output = tmp_path / "source.onnx", tmp_path / "lowered.onnx"
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9)
    onnx.save(onnx.shape_inference.infer_shapes(model), source)
    result = TOOL.lower(source, output, tanh_to_sigmoid=False, decompose_layernorm=False, static_scatter_to_concat=True)
    assert result["lowered_static_scatter"] == 2
    feeds = {"data": np.arange(64, dtype=np.float32).reshape(1, 64), "left": np.full((1, 32), 100, dtype=np.float32),
             "right": np.full((1, 32), 200, dtype=np.float32)}
    normalized = onnx.load(source)
    for index, node in enumerate(normalized.graph.node):
        node.name = f"source_debug_name_{index}"
    expected = ort.InferenceSession(normalized.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)


def test_shape_derived_indices_remain_supported_without_depending_on_tensor_values(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    constants = [onnx.numpy_helper.from_array(value, name) for name, value in (
        ("base", np.asarray([[[0, index] for index in range(32)]], dtype=np.int64)),
        ("axis", np.asarray(1, dtype=np.int64)), ("width", np.asarray(64, dtype=np.int64)),
        ("axes", np.asarray([0], dtype=np.int64)), ("zero", np.asarray([0], dtype=np.int64)))]
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("Shape", ["data"], ["shape"]),
        onnx.helper.make_node("Gather", ["shape", "axis"], ["actual_width"]),
        onnx.helper.make_node("Sub", ["actual_width", "width"], ["offset_scalar"]),
        onnx.helper.make_node("Unsqueeze", ["offset_scalar", "axes"], ["offset_vector"]),
        onnx.helper.make_node("Concat", ["zero", "offset_vector"], ["offset"], axis=0),
        onnx.helper.make_node("Add", ["base", "offset"], ["indices"]),
        onnx.helper.make_node("ScatterND", ["data", "indices", "updates"], ["output"], name="scatter")],
        "shape_only_indices", [tensor("data", onnx.TensorProto.FLOAT, [1, 64]), tensor("updates", onnx.TensorProto.FLOAT, [1, 32])],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 64])], constants)
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9)
    source, output = tmp_path / "source.onnx", tmp_path / "lowered.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(model), source)
    result = TOOL.lower(source, output, tanh_to_sigmoid=False, decompose_layernorm=False, static_scatter_to_concat=True)
    assert result["lowered_static_scatter"] == 1 and result["static_scatter_value_independence_required"]
    feeds = {"data": np.arange(64, dtype=np.float32).reshape(1, 64), "updates": np.ones((1, 32), dtype=np.float32) * 100}
    expected = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    np.testing.assert_array_equal(actual, expected)


def test_initializer_with_a_runtime_override_is_not_frozen_into_indices(tmp_path):
    indices = onnx.numpy_helper.from_array(np.asarray([[[0, index] for index in range(32)]], dtype=np.int64), "indices")
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("ScatterND", ["data", "indices", "updates"], ["output"], name="scatter")],
        "overridable_initializer", [tensor("data", onnx.TensorProto.FLOAT, [1, 64]), tensor("updates", onnx.TensorProto.FLOAT, [1, 32]),
            tensor("indices", onnx.TensorProto.INT64, [1, 32, 2])], [tensor("output", onnx.TensorProto.FLOAT, [1, 64])], [indices])
    source, output = tmp_path / "source.onnx", tmp_path / "lowered.onnx"
    onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9), source)
    result = TOOL.lower(source, output, tanh_to_sigmoid=False, decompose_layernorm=False, static_scatter_to_concat=True)
    assert result["lowered_static_scatter"] == 0 and result["skipped_static_scatter"] == 1


def test_constantness_does_not_cross_random_or_custom_operations_or_unknown_rank():
    scalar = onnx.numpy_helper.from_array(np.asarray(1, dtype=np.int64), "one")
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("RandomUniform", [], ["random"], shape=[1]),
        onnx.helper.make_node("Cast", ["random"], ["random_index"], to=onnx.TensorProto.INT64),
        onnx.helper.make_node("Add", ["one", "one"], ["custom_index"], domain="private.fixture"),
        onnx.helper.make_node("Shape", ["unknown_rank"], ["unknown_shape"])], "not_proven",
        [tensor("unknown_rank", onnx.TensorProto.FLOAT, None)], [], [scalar])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("private.fixture", 1)])
    constant = TOOL._constant_index_values(model)
    assert "one" in constant
    assert not {"random", "random_index", "custom_index", "unknown_shape"} & constant


def test_literal_scatter_does_not_execute_an_unrelated_backend_plugin(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    indices = onnx.numpy_helper.from_array(np.asarray([[[0, index] for index in range(32)]], dtype=np.int64), "indices")
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("Opaque", ["data"], ["opaque"], domain="private.backend", name="unrelated_plugin"),
        onnx.helper.make_node("ScatterND", ["data", "indices", "updates"], ["output"], name="scatter")],
        "literal_no_runtime_probe", [tensor("data", onnx.TensorProto.FLOAT, [1, 64]), tensor("updates", onnx.TensorProto.FLOAT, [1, 32])],
        [tensor("output", onnx.TensorProto.FLOAT, [1, 64]), tensor("opaque", onnx.TensorProto.FLOAT, [1, 64])], [indices])
    source, output = tmp_path / "source.onnx", tmp_path / "lowered.onnx"
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18), onnx.helper.make_opsetid("private.backend", 1)], ir_version=9)
    onnx.save(model, source)
    def unexpected(*_args, **_kwargs):
        raise AssertionError("literal indices require no whole-model inference")
    monkeypatch.setattr(ort, "InferenceSession", unexpected)
    result = TOOL.lower(source, output, tanh_to_sigmoid=False, decompose_layernorm=False, static_scatter_to_concat=True)
    assert result["lowered_static_scatter"] == 1 and result["static_scatter_initializer_tensor_count"] == 1
    assert result["static_scatter_numeric_probe_tensor_count"] == 0
    assert result["requires_numerical_verification"]
    assert any(node.domain == "private.backend" and node.op_type == "Opaque" for node in onnx.load(output).graph.node)


def test_intermediate_shape_annotation_is_not_a_shape_independence_proof():
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("NonZero", ["data"], ["varying_length"]),
        onnx.helper.make_node("Shape", ["varying_length"], ["not_static"]),
        onnx.helper.make_node("Constant", [], ["literal"], value=onnx.numpy_helper.from_array(np.asarray([3], dtype=np.int64))),
        onnx.helper.make_node("Shape", ["literal"], ["constant_shape"])],
        "unproven_intermediate_shape", [tensor("data", onnx.TensorProto.FLOAT, [8])], [],
        value_info=[tensor("varying_length", onnx.TensorProto.INT64, [1, 8])])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)])
    constant = TOOL._constant_index_values(model)
    assert "not_static" not in constant
    assert "constant_shape" in constant
