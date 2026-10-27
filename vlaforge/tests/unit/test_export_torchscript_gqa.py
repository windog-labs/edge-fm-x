import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.fixture
def exporter():
    file = Path(__file__).resolve().parents[2] / "tools/export_torchscript_region_onnx.py"
    spec = importlib.util.spec_from_file_location("export_gqa", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kv_heads,q_heads", [(1, 4), (2, 8), (2, 2)])
@pytest.mark.parametrize("causal", [False, True])
def test_expanded_head_order_matches_real_sdpa(exporter, tmp_path, kv_heads, q_heads, causal):
    class Attention(torch.nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=causal, enable_gqa=True)

    generator = torch.Generator().manual_seed(901)
    examples = tuple(torch.randn(shape, generator=generator) for shape in
                     ((1, q_heads, 5, 16), (1, kv_heads, 7, 16), (1, kv_heads, 7, 12)))
    module = torch.jit.trace(Attention(), examples)
    target = tmp_path / "gqa.onnx"
    with exporter.static_grouped_query_attention(17) as record:
        torch.onnx.export(module, examples, str(target), dynamo=False, opset_version=17,
                          input_names=["query", "key", "value"])
    onnx.checker.check_model(str(target))
    actual = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"]).run(
        None, dict(zip(("query", "key", "value"), [item.numpy() for item in examples], strict=True)))[0]
    np.testing.assert_allclose(actual, module(*examples).numpy(), atol=6e-7, rtol=2e-6)
    assert record["registration_restored"]
    assert record["rewritten"] == [{"rank": 4, "query_heads": q_heads, "key_value_heads": kv_heads,
                                    "repeat_interleave": q_heads // kv_heads}]


def test_dynamic_head_contract_is_not_guessed(exporter, tmp_path):
    class Attention(torch.nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)

    args = (torch.ones(1, 4, 2, 8), torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8))
    with exporter.static_grouped_query_attention(17):
        with pytest.raises(ValueError, match="static Q/K/V head"):
            torch.onnx.export(torch.jit.trace(Attention(), args), args, str(tmp_path / "unknown.onnx"),
                              dynamo=False, opset_version=17, input_names=["query", "key", "value"],
                              dynamic_axes={name: {1: name + "_heads"} for name in ("query", "key", "value")})


def test_registration_restores_and_never_overwrites_an_existing_policy(exporter):
    from torch.onnx import symbolic_opset14  # noqa: F401
    try:
        from torch.onnx._internal.torchscript_exporter import registration
    except ImportError:
        from torch.onnx._internal import registration
    name = "aten::scaled_dot_product_attention"
    original = registration.registry.get_function_group(name).get(17)
    with pytest.raises(RuntimeError, match="injected"):
        with exporter.static_grouped_query_attention(17) as record:
            raise RuntimeError("injected export failure")
    assert record["registration_restored"]
    assert registration.registry.get_function_group(name).get(17) is original
    torch.onnx.register_custom_op_symbolic(name, lambda graph, *args: graph.op("Identity", args[0]), 17)
    try:
        with pytest.raises(ValueError, match="must not be overwritten"):
            with exporter.static_grouped_query_attention(17):
                pass
    finally:
        torch.onnx.unregister_custom_op_symbolic(name, 17)


def test_observed_profile_binds_archive_input_bytes_and_valid_shapes(exporter, tmp_path):
    archive, inputs = tmp_path / "archive.pt", tmp_path / "inputs.pt"
    archive.write_bytes(b"frozen source")
    inputs.write_bytes(b"frozen inputs")
    def pinned(file):
        return {"bytes": file.stat().st_size, "sha256": exporter.digest(file)}
    profile = {"schema": "edgefm.frozen_JIT_observed_attention_shapes/1", "all_complete_CPU_outputs_byte_exact": True,
               "archive": pinned(archive), "inputs": pinned(inputs), "sites": [
                   {"source_range": "actual source site", "shapes": [[1, 8, 5, 16], [1, 2, 7, 16], [1, 2, 7, 12]]}]}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert list(exporter.load_attention_shape_profile(path, archive, inputs)) == ["actual source site"]
    inputs.write_bytes(b"other inputs")
    with pytest.raises(ValueError, match="does not match export inputs"):
        exporter.load_attention_shape_profile(path, archive, inputs)


def test_observed_profile_cannot_replace_a_different_source_site(exporter, tmp_path):
    class Attention(torch.nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    args = (torch.ones(1, 4, 2, 8), torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8))
    with pytest.raises(ValueError, match="no matching source site"):
        with exporter.static_grouped_query_attention(17, {"not the real site": [[1, 4, 2, 8], [1, 2, 2, 8], [1, 2, 2, 8]]}):
            torch.onnx.export(torch.jit.trace(Attention(), args), args, str(tmp_path / "mismatch.onnx"),
                              dynamo=False, opset_version=17)


@pytest.mark.parametrize("conflict", [False, True])
def test_observed_profile_has_matching_site_and_cannot_override_known_dimensions(exporter, tmp_path, conflict):
    class Attention(torch.nn.Module):
        def forward(self, query, key, value):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value, enable_gqa=True)
    args = (torch.ones(1, 4, 2, 8), torch.ones(1, 2, 2, 8), torch.ones(1, 2, 2, 8))
    module = torch.jit.trace(Attention(), args)
    site = next(node for node in module.inlined_graph.nodes() if node.kind() == "aten::scaled_dot_product_attention")
    shapes = [list(value.shape) for value in args]
    if conflict:
        shapes[0][1] = 8
    target = tmp_path / "observed.onnx"
    if conflict:
        with pytest.raises(ValueError, match="conflicts with a known graph dimension"):
            with exporter.static_grouped_query_attention(17, {str(site.sourceRange()): shapes}):
                torch.onnx.export(module, args, str(target), dynamo=False, opset_version=17)
    else:
        with exporter.static_grouped_query_attention(17, {str(site.sourceRange()): shapes}) as record:
            torch.onnx.export(module, args, str(target), dynamo=False, opset_version=17)
        onnx.checker.check_model(str(target))
        assert len(record["observed_specializations"]) == 1


def test_unused_observed_profile_is_rejected(exporter):
    with pytest.raises(ValueError, match="unused source sites"):
        with exporter.static_grouped_query_attention(17, {"unused": [[1, 4, 2, 8], [1, 2, 2, 8], [1, 2, 2, 8]]}):
            pass


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_explicit_scale_and_16bit_opmath_have_no_complex_scalar_cast(exporter, tmp_path, monkeypatch, dtype, masked):
    monkeypatch.syspath_prepend(str(Path(exporter.__file__).parent))
    from lower_onnx_bfloat16_carrier import lower
    class Attention(torch.nn.Module):
        def forward(self, query, key, value, mask):
            return torch.nn.functional.scaled_dot_product_attention(query, key, value,
                attn_mask=mask if masked else None, scale=0.125, enable_gqa=True)
    generator = torch.Generator().manual_seed(111)
    args = [torch.randn(shape, generator=generator).to(dtype) for shape in
            ((1, 4, 5, 16), (1, 2, 7, 16), (1, 2, 7, 12))]
    mask = torch.ones(5, 7, dtype=torch.bool)
    mask[0] = False
    args.append(mask)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        expected = Attention()(*args).float().numpy()
        module = torch.jit.trace(Attention(), tuple(args))
    target = tmp_path / "explicit.onnx"
    with exporter.static_grouped_query_attention(17) as record:
        torch.onnx.export(module, tuple(args), str(target), dynamo=False, opset_version=17,
                          input_names=["q", "k", "v", "mask"])
    model = onnx.load(str(target))
    assert not any(node.op_type == "Cast" and any(attr.name == "to" and attr.i in (
        onnx.TensorProto.COMPLEX64, onnx.TensorProto.COMPLEX128) for attr in node.attribute) for node in model.graph.node)
    portable, _ = lower(model, plugin=False)
    session = ort.InferenceSession(portable.SerializeToString(), providers=["CPUExecutionProvider"])
    values = {name: value.float().numpy() if dtype == torch.bfloat16 and name != "mask" else value.numpy()
              for name, value in zip(("q", "k", "v", "mask"), args, strict=True)}
    actual = session.run(None, {row.name: values[row.name] for row in session.get_inputs()})[0].astype(np.float32)
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    assert np.mean(delta * delta) < 1e-5
    assert np.vdot(actual.ravel(), expected.ravel()) / (np.linalg.norm(actual) * np.linalg.norm(expected)) > .9999
    assert bool(record["float_opmath_sites"]) == (dtype != torch.float32)


@pytest.mark.parametrize("mask_kind", ["bool", "float"])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("opset", [17, 18])
def test_fully_masked_sdpa_rows_are_zero_in_onnx_reference(exporter, tmp_path, mask_kind, gqa, opset):
    from onnx.reference import ReferenceEvaluator

    class Attention(torch.nn.Module):
        def forward(self, query, key, value, mask):
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, enable_gqa=gqa, scale=0.125)

    generator = torch.Generator().manual_seed(273)
    kv_heads = 2 if gqa else 4
    values = [torch.randn(shape, generator=generator) for shape in (
        (1, 4, 3, 8), (1, kv_heads, 5, 8), (1, kv_heads, 5, 6))]
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[0] = False
    mask[1, 2:] = False
    if mask_kind == "float":
        mask = torch.where(mask, 0.0, float("-inf"))
    values.append(mask)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        expected = Attention()(*values).numpy()
        module = torch.jit.trace(Attention(), tuple(values))
    target = tmp_path / "fully-masked.onnx"
    with exporter.static_grouped_query_attention(opset):
        torch.onnx.export(module, tuple(values), str(target), dynamo=False,
                          opset_version=opset, input_names=["q", "k", "v", "mask"])
    model = onnx.load(target)
    onnx.checker.check_model(model)
    actual = ReferenceEvaluator(model).run(
        None, dict(zip(("q", "k", "v", "mask"), [value.numpy() for value in values], strict=True)))[0]
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:, :, 0], np.zeros_like(actual[:, :, 0]))
    np.testing.assert_allclose(actual, expected, atol=6e-7, rtol=2e-6)
