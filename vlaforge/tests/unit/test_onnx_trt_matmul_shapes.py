import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import helper, TensorProto


@pytest.fixture
def tool():
    file = Path(__file__).resolve().parents[2] / "tools/onnx_trt_plugin_shapes.py"
    spec = importlib.util.spec_from_file_location("plugin_shapes", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph(left, right):
    return helper.make_model(helper.make_graph([helper.make_node(
        "VLAForgeBFloat16MatMul", ["a", "b"], ["c"], name="dense", domain="vlaforge.trt",
        plugin_version="1", plugin_namespace="")], "dense",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, left), helper.make_tensor_value_info("b", TensorProto.FLOAT, right)],
        [helper.make_tensor_value_info("c", TensorProto.FLOAT, None)]),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("vlaforge.trt", 1)], ir_version=8)


@pytest.mark.parametrize("left,right,expected", [([3, 8], [8, 4], [3, 4]),
    ([2, 1, 3, 8], [1, 4, 8, 5], [2, 4, 3, 5])])
def test_native_matmul_shape_shadow_never_replaces_the_deployed_operator(tool, left, right, expected):
    model = graph(left, right)
    original = model.SerializeToString()
    result, report = tool.infer_reviewed_shapes(model, ["bf16-matmul/1"])
    assert model.SerializeToString() == original
    assert result.graph.node[0].SerializeToString() == model.graph.node[0].SerializeToString()
    assert [dim.dim_value for dim in result.graph.output[0].type.tensor_type.shape.dim] == expected
    assert report["plugin_counts"] == {"bf16-matmul/1": 1}
    assert not report["deployment_nodes_rewritten"] and not report["numerical_equivalence_verified"]


def test_native_matmul_requires_explicit_contract_selection(tool):
    with pytest.raises(ValueError, match="reviewed plugin"):
        tool.infer_reviewed_shapes(graph([3, 8], [8, 4]), ["bf16-round/1"])


def test_unknown_matmul_dimensions_are_not_filled_from_internal_annotations(tool):
    model = graph([None, 8], [8, 4])
    model.graph.value_info.append(helper.make_tensor_value_info("c", TensorProto.FLOAT, [3, 4]))
    with pytest.raises(ValueError, match="static positive"):
        tool.infer_reviewed_shapes(model, ["bf16-matmul/1"])
