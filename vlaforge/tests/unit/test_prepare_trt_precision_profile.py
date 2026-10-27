import importlib.util
from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

SPEC = importlib.util.spec_from_file_location("prepare_trt_precision_profile",
    Path(__file__).resolve().parents[2] / "tools/prepare_trt_precision_profile.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def test_native_layer_normalization_declares_float_opmath():
    import numpy as np
    import onnx

    h, t = onnx.helper, onnx.TensorProto
    graph = h.make_graph([
        h.make_node("MatMul", ["input", "weight"], ["dense_result"], name="dense"),
        h.make_node("LayerNormalization", ["dense_result", "scale", "bias"], ["output"],
                    name="native_norm", axis=-1, epsilon=1e-6)
    ], "native_normalization", [h.make_tensor_value_info("input", t.FLOAT, [1, 4])],
        [h.make_tensor_value_info("output", t.FLOAT, [1, 4])], [
            onnx.numpy_helper.from_array(np.eye(4, dtype=np.float32), "weight"),
            onnx.numpy_helper.from_array(np.ones(4, dtype=np.float32), "scale"),
            onnx.numpy_helper.from_array(np.zeros(4, dtype=np.float32), "bias")])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    profile = TOOL.build_profile(model)
    assert profile["layer_precisions"]["dense"] == "fp16"
    assert profile["layer_precisions"]["native_norm"] == "fp32"
    assert profile["layer_output_types"]["native_norm"] == "fp32"
    assert profile["correctness_verified"] is False


def test_float_mask_selection_is_not_left_to_half_precision():
    import numpy as np
    import onnx

    h, t = onnx.helper, onnx.TensorProto
    graph = h.make_graph([
        h.make_node("Where", ["mask", "values", "minimum"], ["selected"], name="mask_select"),
        h.make_node("Neg", ["selected"], ["output"], name="negate")
    ], "mask", [h.make_tensor_value_info("mask", t.BOOL, [1]),
                h.make_tensor_value_info("values", t.FLOAT, [1])],
        [h.make_tensor_value_info("output", t.FLOAT, [1])],
        [onnx.numpy_helper.from_array(np.array([np.finfo(np.float32).min], dtype=np.float32), "minimum")])
    profile = TOOL.build_profile(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]))
    assert profile["layer_precisions"] == {"mask_select": "fp32", "negate": "fp32"}
    assert profile["layer_output_types"] == {"mask_select": "fp32", "negate": "fp32"}


@pytest.mark.parametrize("op_type", ["Exp", "Softplus"])
def test_float_nonlinear_opmath_does_not_inherit_half_precision(op_type):
    node = helper.make_node(op_type, ["x"], ["y"], name="nonlinear")
    model = helper.make_model(helper.make_graph([node], "nonlinear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    before = model.SerializeToString()
    profile = TOOL.build_profile(model, dense_output_type="fp16")
    assert profile["layer_precisions"] == {"nonlinear": "fp32"}
    assert profile["layer_output_types"] == {"nonlinear": "fp32"}
    assert model.SerializeToString() == before and profile["correctness_verified"] is False


@pytest.mark.parametrize("op_type", ["Exp", "Softplus"])
def test_nonfloat_nonlinear_tensor_contract_is_not_retyped(op_type):
    model = helper.make_model(helper.make_graph([
        helper.make_node(op_type, ["x"], ["y"], name="half_nonlinear")], "half_nonlinear",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4])]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    profile = TOOL.build_profile(model)
    assert not profile["layer_precisions"] and not profile["layer_output_types"]


def graph():
    inputs = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
        ("x", TensorProto.FLOAT, [1, 2]), ("weight", TensorProto.FLOAT, [2, 2]),
        ("bias", TensorProto.FLOAT, [1, 2]), ("index", TensorProto.INT32, [1]))]
    nodes = [helper.make_node("MatMul", ["x", "weight"], ["dense_out"], name="dense"),
             helper.make_node("Add", ["dense_out", "bias"], ["output"], name="residual"),
             helper.make_node("LessOrEqual", ["output", "bias"], ["mask"], name="finite_compare"),
             helper.make_node("Add", ["index", "index"], ["next_index"], name="index_add")]
    outputs = [helper.make_tensor_value_info(name, dtype, shape) for name, dtype, shape in (
        ("output", TensorProto.FLOAT, [1, 2]), ("mask", TensorProto.BOOL, [1, 2]),
        ("next_index", TensorProto.INT32, [1]))]
    return helper.make_model(helper.make_graph(nodes, "precision-contract", inputs, outputs),
                             opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def test_dense_compute_stays_half_but_float_value_boundary_is_explicit():
    model = graph()
    before = model.SerializeToString()
    profile = TOOL.build_profile(model)
    assert profile["layer_precisions"]["dense"] == "fp16"
    assert profile["layer_output_types"]["dense"] == "fp32"
    assert model.SerializeToString() == before


def test_integer_and_bool_outputs_are_not_retyped_to_float():
    profile = TOOL.build_profile(graph())
    assert profile["layer_precisions"]["residual"] == "fp32"
    assert profile["layer_precisions"]["finite_compare"] == "fp32"
    assert "finite_compare" not in profile["layer_output_types"]
    assert "index_add" not in profile["layer_precisions"]
    assert profile["skipped_nonfloating_or_non_float32"] == [
        {"name": "index_add", "op": "Add", "output_types": [TensorProto.INT32]}]


@pytest.mark.parametrize("name", ["", "dense:0", "dense,*", "dense?", "dense\n"])
def test_ambiguous_layer_selection_is_rejected(name):
    model = graph()
    model.graph.node[0].name = name
    with pytest.raises(ValueError, match="exact names"):
        TOOL.build_profile(model)


def test_duplicate_node_names_are_not_silently_overwritten():
    model = graph()
    model.graph.node[1].name = "dense"
    with pytest.raises(ValueError, match="unique"):
        TOOL.build_profile(model)


def test_half_dense_outputs_require_explicit_opt_in_and_record_extra_rounding():
    profile = TOOL.build_profile(graph(), dense_output_type="fp16")
    assert profile["layer_output_types"]["dense"] == "fp16"
    assert profile["layer_output_types"]["residual"] == "fp32"
    assert profile["layer_precisions"]["dense"] == "fp16"
    assert "additional FP16 result rounding" in profile["dense_result_rounding"]
    assert not profile["correctness_verified"]


def test_unknown_dense_output_type_is_rejected():
    with pytest.raises(ValueError, match="output type"):
        TOOL.build_profile(graph(), dense_output_type="bf16")


def test_reviewed_layernorm_plugin_is_explicitly_fp32():
    model = helper.make_model(helper.make_graph([
        helper.make_node("LayerNorm", ["x", "gamma", "beta"], ["y"], name="normalization",
                         domain="trt.plugins", epsilon=1e-6, plugin_version="1", plugin_namespace="")],
        "normalization", [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
                          for name, shape in (("x", [1, 3, 4]), ("gamma", [4]), ("beta", [4]))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])]),
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("trt.plugins", 1)], ir_version=8)
    profile = TOOL.build_profile(model, dense_output_type="fp16")
    assert profile["layer_precisions"] == {"normalization": "fp32"}
    assert profile["layer_output_types"] == {"normalization": "fp32"}
    next(attribute for attribute in model.graph.node[0].attribute if attribute.name == "plugin_version").s = b"2"
    with pytest.raises(ValueError, match="unreviewed"):
        TOOL.build_profile(model)


def test_observed_dense_exception_keeps_full_float_result_without_changing_other_arithmetic():
    source = graph()
    before = source.SerializeToString()
    profile = TOOL.build_profile(source, dense_output_type="fp16", fp32_dense_nodes=("dense",))
    assert profile["layer_precisions"]["dense"] == "fp32"
    assert profile["layer_output_types"]["dense"] == "fp32"
    assert profile["fp32_dense_exceptions"] == ["dense"]
    assert profile["layer_precisions"]["residual"] == "fp32"
    assert source.SerializeToString() == before and not profile["correctness_verified"]
    for unknown in ("missing", "residual"):
        with pytest.raises(ValueError, match="exact existing dense"):
            TOOL.build_profile(source, fp32_dense_nodes=(unknown,))
