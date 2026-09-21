from types import SimpleNamespace

import pytest
import torch

from vlaforge.adapters.rdt.rdt_reference import (
    check_environment,
    official_agilex_reference,
    strict_parameter_subset,
    partition_reference,
)


def test_strict_subnetwork_load_keeps_explicit_unused_parameter_evidence():
    model = torch.nn.Linear(3, 2)
    state = {"weight": torch.arange(6.0).reshape(2, 3), "bias": torch.ones(2),
             "decoder.weight": torch.ones(4, 3)}
    report = strict_parameter_subset(model, state, unused_prefixes=("decoder.",))
    assert report["strict"]
    assert report["explicitly_unused_keys"] == ["decoder.weight"]
    assert report["active_parameter_elements"] == 8
    assert report["checkpoint_state_elements_including_aliases"] == 20
    assert torch.equal(model.weight, state["weight"])


@pytest.mark.parametrize("kind", ["missing", "unexpected", "non_tensor", "empty"])
def test_partial_random_or_unknown_checkpoint_rejected(kind):
    model = torch.nn.Linear(3, 2)
    state = dict(model.state_dict())
    if kind == "missing":
        del state["bias"]
    elif kind == "unexpected":
        state["unknown.weight"] = torch.ones(2)
    elif kind == "non_tensor":
        state["bias"] = 3
    else:
        state = {}
    with pytest.raises(ValueError):
        strict_parameter_subset(model, state)


def test_unrecognized_environment_never_defaults_to_official():
    with pytest.raises(ValueError, match="explicit"):
        check_environment("latest")


@pytest.mark.parametrize("overrides", [
    {"instruction": ""}, {"images": [None] * 6}, {"images": [object()] * 5},
    {"proprio": torch.zeros(14)}, {"proprio": torch.full((1, 14), float("nan"))},
    {"control_frequency": 0}, {"control_frequency": float("inf")}, {"seed": -1},
])
def test_raw_reference_rejects_incomplete_or_invalid_inputs_before_model_use(overrides):
    inputs = {"instruction": "Move the object", "images": [object()] * 6,
              "proprio": torch.zeros(1, 14), "control_frequency": 25, "seed": 1}
    inputs.update(overrides)
    with pytest.raises(ValueError):
        official_agilex_reference(SimpleNamespace(), **inputs)


def test_partition_preserves_actual_diffusers_multistep_state():
    diffusers = pytest.importorskip("diffusers")
    scheduler = diffusers.DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample",
    )
    scheduler.set_timesteps(5)
    noise = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4) / 10
    outputs = [torch.full_like(noise, (index + 1) / 11) for index in range(5)]
    expected_samples = []
    current = noise.clone()
    for timestep, output in zip(scheduler.timesteps, outputs, strict=True):
        current = scheduler.step(output, timestep, current).prev_sample
        expected_samples.append(current.clone())
    pending = iter(outputs)
    policy = SimpleNamespace(
        pred_horizon=2, action_dim=4, noise_scheduler_sample=scheduler, num_inference_timesteps=5,
        adapt_conditions=lambda lang, image, state: (lang, image, state),
        state_adaptor=torch.nn.Identity(), model=lambda *args, **kwargs: next(pending),
    )
    reference = {
        "noise": noise, "state_adaptor_inputs": [torch.ones(1, 1, 8)],
        "language_tokens": torch.ones(1, 2, 8), "language_mask": torch.ones(1, 2, dtype=torch.bool),
        "image_tokens": torch.ones(6, 2, 3), "scheduler_timesteps": scheduler.timesteps,
        "scheduler_sigmas": scheduler.sigmas, "denoiser_inputs": [(None, torch.tensor([25]))],
        "model_outputs": outputs, "unified_actions": current,
    }
    result = partition_reference(SimpleNamespace(policy=policy), reference)
    assert result["complete_chunk_exact"]
    assert not result["scheduler_state_lowered_to_cpp"]
    assert all(torch.equal(expected, actual) for expected, actual in zip(expected_samples, result["samples"], strict=True))
    assert [item["step_index"] for item in result["scheduler_history"]] == [1, 2, 3, 4, 5]
    assert [item["lower_order_nums"] for item in result["scheduler_history"]] == [1, 2, 2, 2, 2]
    assert result["scheduler_history"][0]["valid_model_outputs"] == [False, True]
    assert result["scheduler_history"][-1]["valid_model_outputs"] == [True, True]
