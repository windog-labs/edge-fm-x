from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vlaforge.adapters.smolvla.smolvla_processing import resolve_smolvla_statistics


def flat(namespace="so100"):
    prefix = f"{namespace}.buffer." if namespace is not None else ""
    return {
        f"{prefix}action.mean": torch.arange(1, 7, dtype=torch.float32),
        f"{prefix}action.std": torch.arange(2, 8, dtype=torch.float32),
    }


def resolve(state=None, **kwargs):
    options = dict(
        namespace="so100",
        robot_type="so100",
        feature_shapes={"action": (6,)},
        source_sha256="a" * 64,
    )
    options.update(kwargs)
    return resolve_smolvla_statistics(flat() if state is None else state, **options)


class ProcessorStep:
    def __init__(self, profile):
        self.values = profile.stats
        self.features = {
            key: SimpleNamespace(shape=shape, type=key)
            for key, shape in profile.feature_shapes.items()
        }
        self.norm_map = {key: "MEAN_STD" for key in self.features}
        self.normalize_observation_keys = None

    def state_dict(self):
        return {
            f"{feature}.{statistic}": value
            for feature, statistics in self.values.items()
            for statistic, value in statistics.items()
        }


def test_explicit_legacy_selection_does_not_choose_first_robot_variant():
    state = {**flat("so100-blue"), **flat("so100-red"), **flat()}
    state["so100-blue.buffer.action.mean"].fill_(100)
    profile = resolve(state)
    assert profile.unselected_namespaces == ("so100-blue", "so100-red")
    assert profile.stats["action"]["mean"].tolist() == [1, 2, 3, 4, 5, 6]
    assert (
        profile.to_dict()["source_to_processor_keys"]["so100.buffer.action.mean"]
        == "action.mean"
    )
    assert profile.to_dict()["robot_calibration_and_physical_units_verified"] is False
    profile.require_consumed(ProcessorStep(profile))


def test_modern_unscoped_stats_need_no_legacy_mapping():
    profile = resolve(flat(None), namespace=None)
    profile.require_consumed(ProcessorStep(profile))
    assert profile.namespace is None


def test_statistics_do_not_alias_source_or_override_consumers():
    state = flat()
    profile = resolve(state)
    state["so100.buffer.action.mean"].zero_()
    profile.stats["action"]["mean"].zero_()
    assert profile.stats["action"]["mean"].tolist() == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("namespace", [None, "", "absent"])
def test_namespace_is_never_guessed(namespace):
    with pytest.raises(ValueError):
        resolve({**flat(), **flat("so100-red")}, namespace=namespace)


def test_namespace_must_match_recorded_robot_not_just_available_variant():
    with pytest.raises(ValueError, match="robot_type"):
        resolve(flat("so100-blue"), namespace="so100-blue")


def test_mixed_modern_and_legacy_keys_are_not_silently_merged():
    with pytest.raises(ValueError, match="mixed"):
        resolve({**flat(), **flat(None)})


def test_missing_state_stats_cannot_be_replaced_by_action_stats():
    with pytest.raises(ValueError, match="missing.*observation.state"):
        resolve(feature_shapes={"observation.state": (6,), "action": (6,)})


def test_selected_but_unconsumed_stats_are_rejected():
    state = {
        **flat(),
        "so100.buffer.observation.state.mean": torch.zeros(6),
        "so100.buffer.observation.state.std": torch.ones(6),
    }
    with pytest.raises(ValueError, match="unconsumed"):
        resolve(state)


@pytest.mark.parametrize("key", ["mean", "std"])
def test_mean_and_std_are_both_required(key):
    state = flat()
    del state[f"so100.buffer.action.{key}"]
    with pytest.raises(ValueError, match="both mean and std"):
        resolve(state)


@pytest.mark.parametrize(
    "replacement",
    [
        torch.zeros(7),
        torch.full((6,), float("nan")),
        torch.full((6,), float("inf")),
        torch.full((6,), -1.0),
    ],
)
def test_invalid_statistics_fail_closed(replacement):
    state = flat()
    state["so100.buffer.action.std"] = replacement
    with pytest.raises(ValueError):
        resolve(state)


def test_generic_processor_identity_due_to_legacy_keys_fails_consumption_gate():
    profile = resolve()
    step = ProcessorStep(profile)
    step.state_dict = lambda: flat()
    with pytest.raises(ValueError, match="not consumed"):
        profile.require_consumed(step)


def test_action_only_profile_cannot_approve_missing_state_in_preprocessor():
    profile = resolve()
    step = ProcessorStep(profile)
    step.features["observation.state"] = SimpleNamespace(shape=(6,), type="state")
    step.norm_map["state"] = "MEAN_STD"
    with pytest.raises(ValueError, match="unverified.*observation.state"):
        profile.require_consumed(step)


@pytest.mark.parametrize(
    "change", ["identity", "wrong_stats", "wrong_shape", "missing_feature"]
)
def test_processor_cannot_ignore_or_change_selected_statistics(change):
    profile = resolve()
    step = ProcessorStep(profile)
    if change == "identity":
        step.norm_map["action"] = "IDENTITY"
    elif change == "wrong_stats":
        step.values["action"]["mean"].zero_()
    elif change == "wrong_shape":
        step.features["action"].shape = (7,)
    else:
        step.features.clear()
    with pytest.raises(ValueError):
        profile.require_consumed(step)


def test_normalization_and_native_scale_match_hand_calculated_full_chunk():
    profile = resolve()
    before = torch.tensor(
        [[[1, 2, 3, 4, 5, 6], [-1, -2, -3, -4, -5, -6]]], dtype=torch.float32
    )
    native = torch.tensor(
        [[[3, 8, 15, 24, 35, 48], [-1, -4, -9, -16, -25, -36]]], dtype=torch.float32
    )
    result = profile.verify_transform("action", before, native, inverse=True)
    assert result["count"] == 12 and result["changed_element_count"] == 11
    assert profile.verify_transform("action", native, before, inverse=False)[
        "exact_formula_match"
    ]


def test_identity_output_is_not_accepted_as_native_scale():
    profile = resolve()
    before = torch.ones(1, 50, 6)
    with pytest.raises(ValueError, match="differs"):
        profile.verify_transform("action", before, before, inverse=True)


@pytest.mark.parametrize(
    "after",
    [torch.ones(1, 49, 6), torch.ones(1, 50, 7), torch.full((1, 50, 6), float("nan"))],
)
def test_transform_checks_shape_and_every_finite_value(after):
    with pytest.raises(ValueError):
        resolve().verify_transform("action", torch.ones(1, 50, 6), after, inverse=True)
