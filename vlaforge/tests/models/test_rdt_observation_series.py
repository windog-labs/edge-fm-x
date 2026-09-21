import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tool(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("rdt_observation_series", folder / "validate_rdt_observation_series.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_histories_are_distinct_and_include_legal_boundaries(tool):
    steps = tool.history_steps(146, 16)
    assert len(steps) == len(set(steps)) == 16
    assert steps[0] == 1 and steps[-1] == 145
    assert all(0 <= step - 1 < step < 146 for step in steps)


def test_nonfinite_scheduler_configuration_is_tagged_without_mutating_original(tool):
    value = {"scheduler": {"lambda_min_clipped": float("-inf")}, "other": [float("inf"), float("nan"), 5]}
    converted = tool.json_provenance(value)
    assert converted["scheduler"]["lambda_min_clipped"] == {"encoding": "nonfinite-number", "value": "-inf"}
    assert converted["other"][0]["value"] == "inf"
    assert converted["other"][1]["value"] == "nan"
    assert value["scheduler"]["lambda_min_clipped"] == float("-inf")
    assert json.loads(json.dumps(converted, allow_nan=False)) == converted


@pytest.mark.parametrize("frames,count", [(8, 8), (146, 7), (146, 17), (146.0, 16)])
def test_invalid_series_shape_is_rejected(tool, frames, count):
    with pytest.raises(ValueError, match="distinct valid"):
        tool.history_steps(frames, count)


def test_official_boundary_copies_saved_noise_and_keeps_masks_separate():
    torch = pytest.importorskip("torch")
    from vlaforge.adapters.rdt.rdt_fresh import inputs_from_official_reference

    combined = torch.tensor([[[1., 2., 3., 1., 0., 1.]]], dtype=torch.bfloat16)
    noise = torch.ones(1, 4, 3, dtype=torch.bfloat16)
    traces = {"state_adaptor_inputs": [combined], "noise": noise,
              "token_ids": torch.tensor([[1, 2]]), "text_attention_mask": torch.ones(1, 2, dtype=torch.int64),
              "pixel_values": torch.zeros(6, 3, 8, 8, dtype=torch.bfloat16),
              "denoiser_inputs": [(None, torch.tensor([25]))]}
    result = inputs_from_official_reference(SimpleNamespace(policy=SimpleNamespace(action_dim=3)), traces)
    assert torch.equal(result["unified_state"], combined[..., :3])
    assert torch.equal(result["action_mask"], combined[..., 3:])
    result["noise"].zero_()
    assert torch.equal(noise, torch.ones_like(noise))
    assert all(value.is_contiguous() for value in result.values())


@pytest.mark.parametrize("mutation", [None, "dtype", "shape", "finite", "extra"])
def test_saved_model_tensors_require_declared_bf16_abi(tool, mutation):
    torch = pytest.importorskip("torch")
    from vlaforge.ir.program import InputPort
    from vlaforge.ir.types import TensorType

    module = SimpleNamespace(inputs=(InputPort("noise", TensorType((1, 4, 3), "bf16"), device="cuda:0"),))
    values = {"noise": torch.ones(1, 4, 3, dtype=torch.bfloat16)}
    if mutation == "dtype":
        values["noise"] = values["noise"].float()
    elif mutation == "shape":
        values["noise"] = values["noise"].reshape(1, 3, 4)
    elif mutation == "finite":
        values["noise"][0, 0, 0] = float("nan")
    elif mutation == "extra":
        values["unbound"] = torch.ones(1)
    if mutation is None:
        tool.validate_model_inputs(values, module)
    else:
        with pytest.raises(ValueError, match="ports|static ABI"):
            tool.validate_model_inputs(values, module)
