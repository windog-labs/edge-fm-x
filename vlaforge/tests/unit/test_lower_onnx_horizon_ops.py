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
