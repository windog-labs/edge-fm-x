"""Synthetic output lowering contracts; real 16-observation evidence is separate."""

import pytest
torch = pytest.importorskip("torch")

from vlaforge.adapters.smolvla.smolvla_output import mean_std_output_module
from vlaforge.adapters.smolvla.smolvla_processing import resolve_smolvla_statistics


def selection():
    return resolve_smolvla_statistics({"action.mean": torch.tensor([2.0, -3.0]),
                                     "action.std": torch.tensor([0.5, 2.0])},
        namespace=None, robot_type="explicit-fixture", feature_shapes={"action": (2,)}, source_sha256="a" * 64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
def test_inverse_formula_and_arbitrary_horizon_keep_dtype(dtype):
    module = mean_std_output_module(selection(), feature="action")
    value = torch.tensor([[[1.0, -2.0]] * 7], dtype=dtype)
    result, accepted = module(value, torch.tensor([True]))
    expected = value * torch.tensor([0.5, 2.0], dtype=dtype) + torch.tensor([2.0, -3.0], dtype=dtype)
    assert result.dtype == dtype and result.shape == (1, 7, 2)
    assert torch.equal(result, expected)
    assert accepted.dtype == torch.bool and accepted.shape == (1,) and accepted.item()


def test_selected_statistics_are_copied_not_aliased():
    stats = selection()
    module = mean_std_output_module(stats, feature="action")
    exposed = stats.stats
    exposed["action"]["mean"].zero_()
    assert torch.equal(module.mean, torch.tensor([2.0, -3.0]))


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_nonfinite_normalized_output_rejects_publication(value):
    _, accepted = mean_std_output_module(selection(), feature="action")(torch.full((1, 3, 2), value), torch.tensor([True]))
    assert not accepted.item()


def test_wrong_shape_dtype_or_feature_rejected():
    module = mean_std_output_module(selection(), feature="action")
    with pytest.raises(ValueError, match="shape"):
        module(torch.zeros((1, 7, 4)), torch.tensor([True]))
    with pytest.raises(TypeError, match="floating"):
        module(torch.zeros((1, 7, 2), dtype=torch.int64), torch.tensor([True]))
    with pytest.raises(ValueError, match="declared"):
        mean_std_output_module(selection(), feature="different")


def test_export_save_reload_complete_output_and_predicate(tmp_path):
    module = mean_std_output_module(selection(), feature="action")
    value = torch.tensor([[[1.0, -2.0]] * 7])
    incoming = torch.tensor([True])
    program = torch.export.export(module, (value, incoming), strict=True)
    path = tmp_path / "output.pt2"
    torch.export.save(program, path)
    result = torch.export.load(path).module()(value, incoming)
    assert all(torch.equal(left, right) for left, right in zip(module(value, incoming), result, strict=True))
    assert not torch.export.load(path).module()(value, torch.tensor([False]))[1].item()


def test_original_rejection_survives_finite_native_output():
    output, accepted = mean_std_output_module(selection(), feature="action")(
        torch.zeros((1, 7, 2)), torch.tensor([False]))
    assert torch.isfinite(output).all() and not accepted.item()
