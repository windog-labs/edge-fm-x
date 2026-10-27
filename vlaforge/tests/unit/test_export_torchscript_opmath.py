import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.fixture
def exporter(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("export_composite", directory / "export_torchscript_region_onnx.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind", ["gelu-tanh", "gelu-erf", "silu"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_composite_preserves_float_opmath_before_16bit_result_rounding(exporter, monkeypatch, tmp_path, kind, dtype):
    activation = torch.nn.SiLU() if kind == "silu" else torch.nn.GELU(approximate="tanh" if kind == "gelu-tanh" else "none")
    inputs = (torch.tensor([-3.125, -2.125, -1.203125, -.109375, .125, .953125, 1.5, 3.125], dtype=dtype),)
    archive, examples = tmp_path / "activation.pt", tmp_path / "inputs.pt"
    torch.jit.trace(activation.eval(), inputs).save(str(archive))
    torch.save(inputs, examples)
    target = tmp_path / "activation.onnx"
    monkeypatch.setattr("sys.argv", ["export", "--torchscript", str(archive), "--inputs", str(examples),
        "--output", str(target), "--device", "cpu", "--opset", "17", "--skip-runtime-validation"])
    assert exporter.main() == 0
    from lower_onnx_bfloat16_carrier import lower
    portable, _ = lower(onnx.load(str(target)), plugin=False)
    session = ort.InferenceSession(portable.SerializeToString(), providers=["CPUExecutionProvider"])
    supplied = inputs[0].float().numpy() if dtype == torch.bfloat16 else inputs[0].numpy()
    actual = session.run(None, {"input_0": supplied})[0].astype(np.float32)
    with torch.inference_mode():
        expected = activation(*inputs).float().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-5)


def test_opmath_registration_restores_on_failure_and_never_overwrites_a_custom_policy(exporter):
    try:
        from torch.onnx._internal.torchscript_exporter import registration
    except ImportError:
        from torch.onnx._internal import registration
    names = ("aten::gelu", "aten::silu", "aten::var", "aten::var_mean", "aten::std")
    original = {name: registration.registry.get_function_group(name).get(17) for name in names}
    with pytest.raises(RuntimeError, match="injected"):
        with exporter.composite_opmath(17) as policy:
            assert not policy["registration_restored"]
            raise RuntimeError("injected export failure")
    assert policy["registration_restored"]
    assert all(registration.registry.get_function_group(name).get(17) is original[name] for name in names)

    def custom(graph, value, *args):
        return graph.op("Identity", value)

    torch.onnx.register_custom_op_symbolic("aten::gelu", custom, 17)
    try:
        with pytest.raises(ValueError, match="must not be overwritten"):
            with exporter.composite_opmath(17):
                pass
        assert registration.registry.get_function_group("aten::gelu").get(17) is custom
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::gelu", 17)


@pytest.mark.parametrize("kind", ["var", "var_mean", "std"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("correction", [0, 1, 2])
def test_real_variance_has_no_complex_cast_and_retains_native_opmath(exporter, tmp_path, kind, dtype, correction):
    class Reduction(torch.nn.Module):
        def forward(self, value):
            return getattr(torch, kind)(value, dim=-1, correction=correction, keepdim=True)

    value = torch.tensor([[1.125, 1.25, 1.5, 1.625, 1.75, 2.0, 2.125, 2.25],
                          [-3.125, -2.125, -1.203125, -.109375, .125, .953125, 1.5, 3.125]], dtype=dtype)
    module = torch.jit.trace(Reduction().eval(), (value,))
    target = tmp_path / "reduction.onnx"
    with exporter.composite_opmath(17):
        torch.onnx.export(module, (value,), str(target), dynamo=False, opset_version=17,
                          input_names=["value"])
    model = onnx.load(str(target))
    assert not any(node.op_type == "Cast" and any(
        attr.name == "to" and attr.i in (onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128)
        for attr in node.attribute) for node in model.graph.node)
    from lower_onnx_bfloat16_carrier import lower
    portable, _ = lower(model, plugin=False)
    session = ort.InferenceSession(portable.SerializeToString(), providers=["CPUExecutionProvider"])
    supplied = value.float().numpy() if dtype == torch.bfloat16 else value.numpy()
    actual = session.run(None, {"value": supplied})
    expected = module(value)
    expected = (expected,) if isinstance(expected, torch.Tensor) else expected
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a.astype(np.float32), b.float().numpy(), rtol=0, atol=3e-5)
