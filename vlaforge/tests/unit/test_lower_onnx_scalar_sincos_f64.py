import copy
import importlib.util
from pathlib import Path
import struct

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")
from onnx.reference import ReferenceEvaluator

SPEC = importlib.util.spec_from_file_location(
    "scalar_sincos_f64", Path(__file__).resolve().parents[2] / "tools/lower_onnx_scalar_sincos_f64.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def fixture(*, dtype=onnx.TensorProto.DOUBLE, count=4, batch=1):
    helper = onnx.helper
    coefficient_dtype = np.float64 if dtype == onnx.TensorProto.DOUBLE else np.float32
    coefficients = np.linspace(-3, 7, count, dtype=coefficient_dtype).reshape(1, count)
    graph = helper.make_graph([
        helper.make_node("Cast", ["time"], ["wide"], to=dtype),
        helper.make_node("Mul", ["frequency", "wide"], ["phase"]),
        helper.make_node("Sin", ["phase"], ["sin"]),
        helper.make_node("Cos", ["phase"], ["cos"]),
        helper.make_node("Concat", ["sin", "cos"], ["joined"], axis=1),
        helper.make_node("Cast", ["joined"], ["output"], to=onnx.TensorProto.FLOAT),
    ], "fixture", [helper.make_tensor_value_info("time", onnx.TensorProto.FLOAT, [batch, 1])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [batch, 2 * count])],
        [onnx.numpy_helper.from_array(coefficients, "frequency")])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)


def test_plugin_contract_keeps_source_and_binary64_coefficients_exact():
    model = fixture()
    before = model.SerializeToString()
    candidate, report = TOOL.lower(model, "output")
    assert model.SerializeToString() == before
    onnx.checker.check_model(candidate)
    plugin = next(node for node in candidate.graph.node if node.domain == TOOL.DOMAIN)
    assert plugin.op_type == TOOL.PLUGIN and list(plugin.input) == ["time"]
    values = onnx.numpy_helper.to_array(model.graph.initializer[0])
    assert struct.pack("<8i", *report["frequency_words"]) == values.astype("<f8").tobytes()
    assert report["f64_arithmetic_demoted"] is False
    assert not report["target_execution_verified"]
    assert not report["source_numerical_parity_verified"]


def test_portable_shadow_matches_full_source_bytes_for_scalar_times():
    source = fixture()
    _, report = TOOL.lower(source, "output")
    shadow = TOOL.portable_shadow(source.graph.input[0], source.graph.output[0], report["frequency_words"])
    onnx.checker.check_model(shadow)
    original, translated = ReferenceEvaluator(source), ReferenceEvaluator(shadow)
    time = np.float32(1)
    for scalar in [np.float32(0), np.float32(-0.), np.float32(1e-20), *[
            np.float32(time - np.float32(.1) * np.float32(index)) for index in range(10)]]:
        feeds = {"time": np.array([[scalar]], dtype=np.float32)}
        assert original.run(None, feeds)[0].tobytes() == translated.run(None, feeds)[0].tobytes()


@pytest.mark.parametrize("kind", ["runtime-frequency", "float-phase", "observer", "wrong-order",
                                  "non-scalar", "too-wide", "nan-frequency"])
def test_rejects_unproved_or_different_sinusoidal_contracts(kind):
    model = fixture(dtype=onnx.TensorProto.FLOAT if kind == "float-phase" else onnx.TensorProto.DOUBLE,
                    count=TOOL.MAX_FREQUENCIES + 1 if kind == "too-wide" else 4,
                    batch=2 if kind == "non-scalar" else 1)
    if kind == "runtime-frequency":
        model.graph.input.append(onnx.helper.make_tensor_value_info("frequency", onnx.TensorProto.DOUBLE, [1, 4]))
    elif kind == "observer":
        model.graph.output.append(onnx.helper.make_tensor_value_info("sin", onnx.TensorProto.DOUBLE, [1, 4]))
    elif kind == "wrong-order":
        model.graph.node[4].input[:] = ["cos", "sin"]
    elif kind == "nan-frequency":
        model.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(np.full((1, 4), np.nan), "frequency"))
    with pytest.raises(ValueError):
        TOOL.lower(model, "output")


def test_preserves_unrelated_learned_nodes_and_weights():
    model = fixture()
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([[2]], dtype=np.float32), "weight"))
    extra = onnx.helper.make_node("MatMul", ["time", "weight"], ["other"])
    model.graph.node.append(extra)
    model.graph.output.append(onnx.helper.make_tensor_value_info("other", onnx.TensorProto.FLOAT, [1, 1]))
    candidate, _ = TOOL.lower(model, "output")
    assert any(node.SerializeToString() == extra.SerializeToString() for node in candidate.graph.node)
    assert next(item for item in candidate.graph.initializer if item.name == "weight").SerializeToString() == model.graph.initializer[1].SerializeToString()


def test_constant_range_budget_is_checked_before_allocation():
    h = onnx.helper
    model = h.make_model(h.make_graph([
        h.make_node("Range", ["start", "end", "step"], ["frequency"])
    ], "large", [], [h.make_tensor_value_info("frequency", onnx.TensorProto.INT64, [10 ** 12])], [
        onnx.numpy_helper.from_array(np.array(value, dtype=np.int64), name)
        for name, value in [("start", 0), ("end", 10 ** 12), ("step", 1)]]),
        opset_imports=[h.make_opsetid("", 18)], ir_version=8)
    with pytest.raises(ValueError, match="budget"):
        TOOL.constant_value(model, "frequency")


def test_external_coefficients_are_bounded_and_source_metadata_is_unchanged(tmp_path):
    model = fixture()
    path = tmp_path / "source.onnx"
    onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location="weights.bin", size_threshold=0)
    model = onnx.load(path, load_external_data=False)
    before = model.SerializeToString()
    candidate, report = TOOL.lower(model, "output", external_base_dir=tmp_path)
    assert report["frequency_count"] == 4 and model.SerializeToString() == before
    assert not candidate.graph.initializer
    poisoned = copy.deepcopy(model)
    entry = next(item for item in poisoned.graph.initializer[0].external_data if item.key == "length")
    entry.value = str(10 ** 12)
    with pytest.raises(ValueError):
        TOOL.lower(poisoned, "output", external_base_dir=tmp_path)


def test_shadow_internal_names_do_not_collide_with_public_ports():
    model = fixture()
    _, report = TOOL.lower(model, "output")
    source = onnx.helper.make_tensor_value_info("frequencies", onnx.TensorProto.FLOAT, [1, 1])
    output = onnx.helper.make_tensor_value_info("time_f64", onnx.TensorProto.FLOAT, [1, 8])
    onnx.checker.check_model(TOOL.portable_shadow(source, output, report["frequency_words"]))
