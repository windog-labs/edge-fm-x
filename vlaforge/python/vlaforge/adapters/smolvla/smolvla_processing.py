"""Explicit statistics selection and consumption gates for SmolVLA processors.

Legacy checkpoint migrations can retain ``robot.buffer.feature.stat`` keys.
The upstream generic processor accepts them but silently returns identity for
an absent feature. Resolve only a caller-selected robot profile, then verify
that the upstream processor consumed exactly those statistics. No action to
state substitution, dataset refitting or default robot selection occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


def prepare_smolvla_statistics(policy_path, *, mode, namespace, robot_type):
    """Load the explicitly selected checkpoint statistics for any host consumer."""
    if mode not in {"strict-statistics", "published-compatibility"}:
        raise ValueError("unknown processing mode")
    if mode == "published-compatibility":
        if namespace is not None:
            raise ValueError("published compatibility cannot select replacement statistics")
        return None, {"mode": mode, "normalization_statistics_verified": False,
            "physical_action_units_verified": False, "formal_preprocessing_acceptance": False,
            "limitation": "published processor behavior only; missing or ignored statistics are not repaired"}
    from safetensors.torch import load_file

    policy_path = Path(policy_path)
    configuration = json.loads((policy_path / "policy_preprocessor.json").read_text())
    steps = [step for step in configuration["steps"] if step["registry_name"] == "normalizer_processor"]
    if len(steps) != 1:
        raise ValueError("strict preprocessing requires exactly one declared normalizer")
    step = steps[0]
    shapes = {name: tuple(feature["shape"]) for name, feature in step["config"]["features"].items()
              if step["config"]["norm_map"].get(feature["type"]) == "MEAN_STD"}
    if "observation.state" not in shapes:
        raise ValueError("this strict profile requires declared state MEAN_STD statistics")
    path = (policy_path / step["state_file"]).resolve(strict=True)
    if not path.is_relative_to(policy_path.resolve()):
        raise ValueError("processor statistics must belong to the checkpoint")
    profile = resolve_smolvla_statistics(load_file(str(path), device="cpu"), namespace=namespace,
        robot_type=robot_type, feature_shapes=shapes, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return profile, {"mode": mode, "selection": profile.to_dict(), "normalization_statistics_verified": False,
                     "physical_action_units_verified": False, "formal_preprocessing_acceptance": False}


@dataclass(frozen=True, slots=True)
class SmolVLAStatistics:
    namespace: str | None
    robot_type: str
    source_sha256: str
    feature_shapes: Mapping[str, tuple[int, ...]]
    _stats: Mapping[str, Mapping[str, Any]]
    _source_keys: Mapping[str, str]
    unselected_namespaces: tuple[str, ...]

    @property
    def stats(self) -> dict[str, dict[str, Any]]:
        """Fresh tensors for the upstream processor's explicit stats override."""
        return {
            feature: {name: value.clone() for name, value in statistics.items()}
            for feature, statistics in self._stats.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "vlaforge.smolvla_statistics_selection/1",
            "namespace": self.namespace,
            "robot_type": self.robot_type,
            "source_sha256": self.source_sha256,
            "feature_shapes": {
                key: list(value) for key, value in self.feature_shapes.items()
            },
            "source_to_processor_keys": dict(self._source_keys),
            "unselected_namespaces": list(self.unselected_namespaces),
            "stats": {
                feature: {name: value.tolist() for name, value in statistics.items()}
                for feature, statistics in self._stats.items()
            },
            "transforms": {
                "normalize": "(value - mean) / (std + eps)",
                "unnormalize": "value * std + mean",
            },
            "robot_calibration_and_physical_units_verified": False,
        }

    def require_consumed(self, processor_step: Any) -> None:
        """Reject missing, ignored, extra or changed statistics in a loaded step."""
        import torch

        expected = {
            f"{feature}.{statistic}": value
            for feature, statistics in self._stats.items()
            for statistic, value in statistics.items()
        }
        actual = processor_step.state_dict()
        if set(actual) != set(expected):
            raise ValueError(
                "processor statistics were not consumed under the selected feature names"
            )
        for key, value in expected.items():
            candidate = actual[key]
            if not isinstance(candidate, torch.Tensor) or not torch.equal(
                value, candidate.detach().to(device="cpu", dtype=torch.float32)
            ):
                raise ValueError(f"processor changed the selected statistic {key}")
        for name, feature in processor_step.features.items():
            mode = processor_step.norm_map.get(feature.type, "IDENTITY")
            if (
                getattr(mode, "value", mode) != "IDENTITY"
                and name not in self.feature_shapes
            ):
                raise ValueError(f"processor requires unverified statistics for {name}")
        for name, shape in self.feature_shapes.items():
            feature = processor_step.features.get(name)
            if feature is None or tuple(feature.shape) != shape:
                raise ValueError(
                    f"processor does not consume the declared feature {name}"
                )
            mode = processor_step.norm_map.get(feature.type)
            if getattr(mode, "value", mode) != "MEAN_STD":
                raise ValueError(f"processor does not use MEAN_STD for {name}")
            selected = getattr(processor_step, "normalize_observation_keys", None)
            if selected is not None and name != "action" and name not in selected:
                raise ValueError(f"processor excludes the selected feature {name}")

    def verify_transform(
        self,
        feature: str,
        before: Any,
        after: Any,
        *,
        inverse: bool,
        eps: float = 1e-8,
    ) -> dict[str, Any]:
        """Check every real before/after value against the upstream formula."""
        import torch

        if feature not in self.feature_shapes:
            raise ValueError(f"feature {feature} is not in the selected profile")
        if not math.isfinite(eps) or eps < 0:
            raise ValueError("normalization epsilon must be finite and nonnegative")
        shape = self.feature_shapes[feature]
        for name, value in (("before", before), ("after", after)):
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_floating_point()
                or value.numel() == 0
                or tuple(value.shape[-len(shape) :]) != shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"{name} must be a finite floating tensor with the selected feature shape"
                )
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError("processor changed the tensor shape or dtype")
        statistics = {
            key: value.to(before) for key, value in self._stats[feature].items()
        }
        expected = (
            before * statistics["std"] + statistics["mean"]
            if inverse
            else (before - statistics["mean"]) / (statistics["std"] + eps)
        )
        if not bool(torch.isfinite(expected).all()) or not torch.equal(
            expected.cpu(), after.cpu()
        ):
            raise ValueError(
                "processor result differs from the selected statistics transform"
            )
        return {
            "feature": feature,
            "inverse": inverse,
            "eps": eps,
            "count": before.numel(),
            "exact_formula_match": True,
            "changed_element_count": int(
                torch.count_nonzero(before.cpu() != after.cpu())
            ),
        }


def resolve_smolvla_statistics(
    flat_state: Mapping[str, Any],
    *,
    namespace: str | None,
    robot_type: str,
    feature_shapes: Mapping[str, tuple[int, ...]],
    source_sha256: str,
) -> SmolVLAStatistics:
    """Resolve flat MEAN_STD stats without guessing a robot or absent feature.

    ``namespace=None`` is allowed only for modern unscoped ``feature.mean/std``
    files. Legacy namespaces must be explicitly selected and must match the
    dataset's recorded robot_type. ``feature_shapes`` lists every feature that
    must consume stats; selected but unconsumed stats are errors, not warnings.
    This verifies numerical scale, not robot calibration or metric units.
    """
    import torch

    if not isinstance(robot_type, str) or not robot_type:
        raise ValueError("robot_type must come from explicit input provenance")
    if namespace is not None and (not isinstance(namespace, str) or not namespace):
        raise ValueError("namespace must be an explicit nonempty name or None")
    if len(source_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in source_sha256
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    if not feature_shapes:
        raise ValueError("declare at least one feature that consumes statistics")
    shapes = {}
    for name, shape in feature_shapes.items():
        if (
            not isinstance(name, str)
            or not name
            or not shape
            or any(type(size) is not int or size < 1 for size in shape)
        ):
            raise ValueError("feature names and fixed positive shapes are required")
        shapes[name] = tuple(shape)
    namespaces = {}
    for key, tensor in flat_state.items():
        if not isinstance(key, str) or "." not in key:
            raise ValueError(f"malformed statistic key {key!r}")
        stem, statistic = key.rsplit(".", 1)
        if statistic not in ("mean", "std"):
            raise ValueError(
                f"unsupported statistic {key}; this profile requires MEAN_STD"
            )
        scope, marker, feature = stem.partition(".buffer.")
        if not marker:
            scope, feature = None, stem
        if not feature or (marker and not scope):
            raise ValueError(f"malformed statistic feature {key!r}")
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"statistic {key} must be a floating tensor")
        namespaces.setdefault(scope, {}).setdefault(feature, {})[statistic] = (
            key,
            tensor,
        )
    if not namespaces:
        raise ValueError("statistics file contains no features")
    if None in namespaces and len(namespaces) > 1:
        raise ValueError("mixed unscoped and legacy namespaces are ambiguous")
    if namespace is None and set(namespaces) != {None}:
        raise ValueError(
            f"select an explicit statistics namespace from {sorted(namespaces)}"
        )
    if namespace not in namespaces:
        raise ValueError(f"statistics namespace {namespace!r} is absent")
    if namespace is not None and namespace != robot_type:
        raise ValueError("statistics namespace does not match the recorded robot_type")
    selected = namespaces[namespace]
    missing = set(shapes) - set(selected)
    if missing:
        raise ValueError(f"missing required feature statistics: {sorted(missing)}")
    if set(selected) != set(shapes):
        raise ValueError(
            f"unconsumed selected feature statistics: {sorted(set(selected) - set(shapes))}"
        )
    stats, source_keys = {}, {}
    for feature, shape in shapes.items():
        if set(selected[feature]) != {"mean", "std"}:
            raise ValueError(f"feature {feature} requires both mean and std")
        stats[feature] = {}
        for statistic, (key, value) in selected[feature].items():
            value = value.detach().to(device="cpu", dtype=torch.float32).clone()
            if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"statistic {key} must be finite with shape {shape}")
            if statistic == "std" and bool((value < 0).any()):
                raise ValueError(f"standard deviation {key} must be nonnegative")
            stats[feature][statistic] = value
            source_keys[key] = f"{feature}.{statistic}"
    return SmolVLAStatistics(
        namespace,
        robot_type,
        source_sha256,
        MappingProxyType(shapes),
        MappingProxyType(
            {key: MappingProxyType(value) for key, value in stats.items()}
        ),
        MappingProxyType(source_keys),
        tuple(sorted(key for key in namespaces if key != namespace)),
    )
