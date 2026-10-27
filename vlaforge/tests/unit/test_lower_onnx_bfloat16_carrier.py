import importlib.util
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper


@pytest.fixture
def tool(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("bf16_carrier", folder / "lower_onnx_bfloat16_carrier.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def graph():
    ml = pytest.importorskip("ml_dtypes")
    weight = np.linspace(-.4, .7, 8).reshape(2, 4).astype(ml.bfloat16)
    return helper.make_model(helper.make_graph([
        helper.make_node("Add", ["x", "weight"], ["sum"], name="add"),
        helper.make_node("Cast", ["sum"], ["float_sum"], name="to_float", to=TensorProto.FLOAT),
        helper.make_node("Mul", ["float_sum", "factor"], ["scaled"], name="scale"),
        helper.make_node("Cast", ["scaled"], ["rounded"], name="to_bf16", to=TensorProto.BFLOAT16),
        helper.make_node("Cast", ["rounded"], ["output"], name="result", to=TensorProto.FLOAT),
    ], "bf16_arithmetic", [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 4])],
        [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(np.array(.998671, np.float32), "factor")]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def test_rounding_boundaries_survive_and_source_is_not_modified(tool):
    model = graph()
    before = model.SerializeToString()
    lowered, report = tool.lower(model)
    assert model.SerializeToString() == before
    assert lowered.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert all(t.data_type != TensorProto.BFLOAT16 for t in lowered.graph.initializer)
    assert [x["operator"] for x in report["rounding_boundaries"]] == ["Add", "Cast"]
    assert len(report["input_rounding"]) == 1
    plugins = [node for node in lowered.graph.node if node.domain == "vlaforge.trt"]
    assert len(plugins) == 3
    assert report["compute_precision_certified"] is report["source_equivalence_verified"] is False
    onnx.checker.check_model(lowered)


def test_portable_reference_matches_actual_torch_bf16_arithmetic(tool):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    model = graph()
    lowered, _ = tool.lower(model, plugin=False)
    onnx.checker.check_model(lowered)
    session = ort.InferenceSession(lowered.SerializeToString(), providers=["CPUExecutionProvider"])
    weights = torch.from_numpy(numpy_helper.to_array(model.graph.initializer[0]).astype(np.float32)).bfloat16()
    for seed in range(8):
        data = np.random.default_rng(seed).normal(size=(2, 4)).astype(np.float32)
        expected = ((torch.from_numpy(data).bfloat16() + weights).float() * .998671).bfloat16().float().numpy()
        actual = session.run(None, {"x": data})[0]
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_copy_only_nodes_do_not_insert_new_arithmetic_rounding(tool):
    model = graph()
    model.graph.node.insert(1, helper.make_node("Identity", ["sum"], ["copied"], name="copy"))
    model.graph.node[2].input[0] = "copied"
    lowered, report = tool.lower(model)
    assert len(report["rounding_boundaries"]) == 2
    assert len([node for node in lowered.graph.node if node.op_type == "Identity"]) == 1


@pytest.mark.parametrize("operator", ["Cast", "CastLike"])
def test_double_to_bf16_is_not_silently_double_rounded(tool, operator):
    node = (helper.make_node("Cast", ["x"], ["y"], to=TensorProto.BFLOAT16) if operator == "Cast"
            else helper.make_node("CastLike", ["x", "like"], ["y"]))
    inputs = [helper.make_tensor_value_info("x", TensorProto.DOUBLE, [2])]
    if operator == "CastLike":
        inputs.append(helper.make_tensor_value_info("like", TensorProto.BFLOAT16, [2]))
    model = helper.make_model(helper.make_graph(
        [node], "double", inputs,
        [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [2])]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    with pytest.raises(ValueError, match="DOUBLE-to-BF16"):
        tool.lower(model)


@pytest.mark.parametrize("operator", ["RandomNormalLike", "Dropout", "QuantizeLinear"])
def test_random_or_quantized_graphs_are_not_assumed_pure(tool, operator):
    model = graph()
    model.graph.node[0].op_type = operator
    with pytest.raises(ValueError, match="functional"):
        tool.lower(model)


def test_shared_bf16_input_output_name_requires_explicit_adapter(tool):
    model = helper.make_model(helper.make_graph([], "alias",
        [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2])],
        [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [2])]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    with pytest.raises(ValueError, match="aliases"):
        tool.lower(model)


def test_integer_ports_and_constants_keep_their_original_dtype(tool):
    model = graph()
    model.graph.input.append(helper.make_tensor_value_info("count", TensorProto.INT64, [1]))
    model.graph.initializer.append(numpy_helper.from_array(np.array([1], np.int64), "one"))
    model.graph.node.append(helper.make_node("Add", ["count", "one"], ["next_count"], name="count_add"))
    model.graph.output.append(helper.make_tensor_value_info("next_count", TensorProto.INT64, [1]))
    lowered, report = tool.lower(model)
    assert lowered.graph.input[1].type.tensor_type.elem_type == TensorProto.INT64
    assert lowered.graph.output[1].type.tensor_type.elem_type == TensorProto.INT64
    assert next(t for t in lowered.graph.initializer if t.name == "one").data_type == TensorProto.INT64
    assert not report["integer_dtypes_changed"]


def test_external_bf16_and_float_weights_are_materialized_only_after_shape_inference(tool, tmp_path):
    source = tmp_path / "source.onnx"
    onnx.save_model(graph(), str(source), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    metadata = onnx.load(str(source), load_external_data=False)
    before = metadata.SerializeToString()
    with pytest.raises(ValueError, match="base directory"):
        tool.lower(metadata)
    lowered, _ = tool.lower(metadata, external_base_dir=tmp_path)
    assert metadata.SerializeToString() == before
    assert all(t.data_location == TensorProto.DEFAULT and t.HasField("raw_data")
               for t in lowered.graph.initializer)
    assert all(t.data_type == TensorProto.FLOAT for t in lowered.graph.initializer)
    onnx.checker.check_model(lowered)


def test_portable_carrier_shadow_keeps_shared_external_weight_locations(tool, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    lowered, _ = tool.lower(graph())
    path = tmp_path / "carrier.onnx"
    onnx.save_model(lowered, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    metadata = onnx.load(str(path), load_external_data=False)
    before = metadata.SerializeToString()
    shadow = tool.reference_from_carrier(metadata)
    assert metadata.SerializeToString() == before
    assert all(not node.domain for node in shadow.graph.node)
    assert [tensor.SerializeToString() for tensor in shadow.graph.initializer] == [
        tensor.SerializeToString() for tensor in metadata.graph.initializer]
    onnx.save_model(shadow, str(tmp_path / "reference.onnx"))
    onnx.checker.check_model(str(tmp_path / "reference.onnx"))
    expected_model, _ = tool.lower(graph(), plugin=False)
    expected = ort.InferenceSession(expected_model.SerializeToString(), providers=["CPUExecutionProvider"])
    actual = ort.InferenceSession(str(tmp_path / "reference.onnx"), providers=["CPUExecutionProvider"])
    inputs = {"x": np.random.default_rng(77).normal(size=(2, 4)).astype(np.float32)}
    np.testing.assert_array_equal(actual.run(None, inputs)[0], expected.run(None, inputs)[0])
    assert (tmp_path / "weights.bin").stat().st_nlink == 1


def test_portable_shadow_does_not_guess_unknown_plugin_semantics(tool):
    model, _ = tool.lower(graph())
    node = next(node for node in model.graph.node if node.domain)
    node.op_type = "OtherProvider"
    with pytest.raises(ValueError, match="unreviewed"):
        tool.reference_from_carrier(model)


def matmul_graph(left, right, *, dtype=TensorProto.BFLOAT16):
    return helper.make_model(helper.make_graph([
        helper.make_node("MatMul", ["a", "b"], ["y"], name="product"),
    ], "dense", [helper.make_tensor_value_info("a", dtype, left),
                   helper.make_tensor_value_info("b", dtype, right)],
        [helper.make_tensor_value_info("y", dtype, None)]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


@pytest.mark.parametrize("left,right", [([3, 8], [8, 4]), ([2, 3, 8], [8, 4]),
    ([2, 1, 3, 8], [1, 4, 8, 5]), ([3, 8], [2, 8, 5])])
def test_native_dense_lowering_requires_proven_bf16_and_retains_portable_oracle(tool, left, right):
    model = matmul_graph(left, right)
    original = model.SerializeToString()
    lowered, report = tool.lower(model, native_bf16_matmul=True)
    assert model.SerializeToString() == original
    nodes = [node for node in lowered.graph.node if node.op_type == "VLAForgeBFloat16MatMul"]
    assert len(nodes) == len(report["native_bf16_matmul"]) == 1
    assert nodes[0].domain == "vlaforge.trt" and nodes[0].name == "product"
    assert not report["compute_precision_certified"] and not report["source_equivalence_verified"]
    onnx.checker.check_model(lowered)
    portable = tool.reference_from_carrier(lowered)
    onnx.checker.check_model(portable)
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(portable.SerializeToString(), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(9137)
    # Binary rational inputs keep all products/sums exact in FLOAT.
    arrays = [rng.integers(-32, 33, size=shape).astype(np.float32) / 32 for shape in (left, right)]
    ml = pytest.importorskip("ml_dtypes")
    expected = np.matmul(*arrays).astype(ml.bfloat16).astype(np.float32)
    actual = session.run(None, dict(zip(("a", "b"), arrays, strict=True)))[0]
    np.testing.assert_array_equal(actual, expected)


def test_float_dense_nodes_are_not_reinterpreted_as_original_bf16(tool):
    lowered, report = tool.lower(matmul_graph([3, 8], [8, 4], dtype=TensorProto.FLOAT), native_bf16_matmul=True)
    assert not report["native_bf16_matmul"]
    assert next(node for node in lowered.graph.node if node.name == "product").op_type == "MatMul"


@pytest.mark.parametrize("left,right", [([8], [8, 4]), ([None, 8], [8, 4]), ([0, 8], [8, 4])])
def test_dense_provider_does_not_guess_vector_dynamic_or_empty_contracts(tool, left, right):
    with pytest.raises(ValueError, match="static positive rank"):
        tool.lower(matmul_graph(left, right), native_bf16_matmul=True)


def test_native_dense_policy_is_not_silently_enabled_for_portable_export(tool):
    with pytest.raises(ValueError, match="requires deployment plugins"):
        tool.lower(matmul_graph([3, 8], [8, 4]), plugin=False, native_bf16_matmul=True)


def test_dense_shape_proof_propagates_static_shape_tensor_values_without_guessing(tool):
    model = matmul_graph([2, 3, 8], [8, 4])
    model.graph.input[0].name = "x"
    model.graph.node[0].input[0] = "a"
    nodes = [
        helper.make_node("Shape", ["x"], ["shape"]),
        helper.make_node("Gather", ["shape", "zero"], ["batch"], axis=0),
        helper.make_node("Concat", ["batch", "tail"], ["target"], axis=0),
        helper.make_node("Reshape", ["x", "target"], ["a"]),
    ] + list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([0], np.int64), "zero"),
        numpy_helper.from_array(np.array([3, 8], np.int64), "tail")])
    lowered, report = tool.lower(model, native_bf16_matmul=True)
    assert report["native_bf16_matmul"][0]["input_shapes"] == [(2, 3, 8), (8, 4)]
    onnx.checker.check_model(lowered)
