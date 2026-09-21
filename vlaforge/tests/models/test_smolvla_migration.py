import json

import pytest
import torch
from vlaforge.adapters.smolvla.smolvla_migration import (
    materialize_profile,
    recover_statistics,
    sha256,
    verify_recovered_profile,
)


def checkpoint_pair(tmp_path):
    save_file = pytest.importorskip("safetensors.torch").save_file
    policy = tmp_path / "policy"
    policy.mkdir()
    weight = torch.tensor([[1.0, 2.0]])
    published = {
        "so100.buffer.action.mean": torch.tensor([3.0, 4.0]),
        "so100.buffer.action.std": torch.tensor([5.0, 6.0]),
    }
    historical = {
        "model._orig_mod.weight": weight,
        "model._orig_mod.scalar": torch.tensor(0.0),
    }
    for operation, feature in (
        ("normalize_inputs", "observation_state"),
        ("normalize_targets", "action"),
        ("unnormalize_outputs", "action"),
    ):
        for stat in ("mean", "std"):
            historical[f"{operation}.so100_buffer_{feature}.{stat}"] = (
                torch.tensor([11.0, 12.0])
                if feature == "observation_state"
                else published[f"so100.buffer.action.{stat}"].clone()
            )
    old = tmp_path / "historical.safetensors"
    current = policy / "model.safetensors"
    stats = policy / "published.safetensors"
    save_file(historical, old)
    save_file({"model.weight": weight, "model.scalar": torch.tensor(0.0)}, current)
    save_file(published, stats)
    arguments = {
        "historical_checkpoint": old,
        "historical_sha256": sha256(old),
        "current_checkpoint": current,
        "current_sha256": sha256(current),
        "published_statistics": stats,
        "published_sha256": sha256(stats),
        "namespace": "so100",
        "robot_type": "so100",
        "state_shape": (2,),
        "action_shape": (2,),
    }
    return arguments, historical, policy


def test_recovery_verifies_every_model_tensor_and_both_action_statistics(tmp_path):
    arguments, _, policy = checkpoint_pair(tmp_path)
    statistics, report = recover_statistics(**arguments)
    assert report["model_tensor_count"] == 2
    assert report["model_tensor_elements"] == 3
    assert report["model_tensors_bitwise_equal"]
    assert report["all_published_action_statistics_bitwise_equal"]
    assert statistics["observation.state.mean"].tolist() == [11.0, 12.0]
    assert report["physical_action_units_verified"] is False
    (policy / "config.json").write_text("{}")
    for name, registry in (
        ("policy_preprocessor.json", "normalizer_processor"),
        ("policy_postprocessor.json", "unnormalizer_processor"),
    ):
        (policy / name).write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "registry_name": registry,
                            "state_file": "published.safetensors",
                        }
                    ]
                }
            )
        )
    output = tmp_path / "recovered"
    result = materialize_profile(
        policy_path=policy, output=output, statistics=statistics, report=report
    )
    assert result["output_files"]["model.safetensors"] == arguments["current_sha256"]
    assert (output / "model.safetensors").resolve() == (policy / "model.safetensors")
    load_file = pytest.importorskip("safetensors.torch").load_file
    assert set(load_file(output / "unnormalizer_processor.safetensors")) == {
        "action.mean",
        "action.std",
    }
    assert set(load_file(output / "normalizer_processor.safetensors")) == set(
        statistics
    )
    assert verify_recovered_profile(policy, robot_type="so100") is None
    assert verify_recovered_profile(output, robot_type="so100")[
        "model_tensors_bitwise_equal"
    ]
    with pytest.raises(ValueError, match="provenance"):
        verify_recovered_profile(output, robot_type="other-robot")
    assert (
        json.loads((policy / "policy_preprocessor.json").read_text())["steps"][0][
            "state_file"
        ]
        == "published.safetensors"
    )
    with pytest.raises(ValueError, match="must be new"):
        materialize_profile(
            policy_path=policy, output=output, statistics=statistics, report=report
        )
    (output / "config.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_recovered_profile(output, robot_type="so100")


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("weight", "model tensor differs"),
        ("signed_zero", "model tensor differs"),
        ("missing_weight", "coverage"),
        ("unexpected_stat", "coverage"),
        ("missing_state", "coverage"),
        ("action", "action statistics differ"),
        ("negative_state_std", "must be nonnegative"),
    ],
)
def test_recovery_rejects_changes_not_explained_by_the_migration(
    tmp_path, mutation, match
):
    arguments, historical, _ = checkpoint_pair(tmp_path)
    if mutation == "weight":
        historical["model._orig_mod.weight"][0, 0] = 9.0
    elif mutation == "signed_zero":
        historical["model._orig_mod.scalar"] = torch.tensor(-0.0)
    elif mutation == "missing_weight":
        historical.pop("model._orig_mod.weight")
    elif mutation == "unexpected_stat":
        historical["unrecognized"] = torch.ones(1)
    elif mutation == "missing_state":
        historical.pop("normalize_inputs.so100_buffer_observation_state.mean")
    elif mutation == "action":
        historical["normalize_targets.so100_buffer_action.mean"][0] = 99.0
    elif mutation == "negative_state_std":
        historical["normalize_inputs.so100_buffer_observation_state.std"][0] = -1.0
    pytest.importorskip("safetensors.torch").save_file(
        historical, arguments["historical_checkpoint"]
    )
    arguments["historical_sha256"] = sha256(arguments["historical_checkpoint"])
    with pytest.raises(ValueError, match=match):
        recover_statistics(**arguments)


def test_recovery_rejects_unverified_source_and_wrong_robot(tmp_path):
    arguments, _, _ = checkpoint_pair(tmp_path)
    with pytest.raises(ValueError, match="digest mismatch"):
        recover_statistics(**{**arguments, "historical_sha256": "0" * 64})
    with pytest.raises(ValueError, match="robot_type"):
        recover_statistics(**{**arguments, "robot_type": "other-robot"})
