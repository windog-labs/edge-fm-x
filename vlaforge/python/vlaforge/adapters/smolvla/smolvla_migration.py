"""Recover migrated processor statistics only from an identical verified model.

This creates a new, explicitly selected robot profile. It never rewrites the
published checkpoint or substitutes observation-dataset statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from vlaforge.adapters.smolvla.smolvla_processing import resolve_smolvla_statistics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recover_statistics(
    *,
    historical_checkpoint: Path,
    historical_sha256: str,
    current_checkpoint: Path,
    current_sha256: str,
    published_statistics: Path,
    published_sha256: str,
    namespace: str,
    robot_type: str,
    state_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
) -> tuple[dict, dict]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    sources = {}
    for name, path, expected in (
        ("historical_checkpoint", historical_checkpoint, historical_sha256),
        ("current_checkpoint", current_checkpoint, current_sha256),
        ("published_statistics", published_statistics, published_sha256),
    ):
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"{name} requires a pinned SHA-256 digest")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{name} digest mismatch")
        sources[name] = {"path": str(path.resolve()), "sha256": actual}

    published = load_file(str(published_statistics), device="cpu")
    selected_action = resolve_smolvla_statistics(
        published,
        namespace=namespace,
        robot_type=robot_type,
        feature_shapes={"action": action_shape},
        source_sha256=published_sha256,
    )
    scopes = {key.partition(".buffer.")[0] for key in published}
    expected_published = {
        f"{scope}.buffer.action.{stat}" for scope in scopes for stat in ("mean", "std")
    }
    if set(published) != expected_published:
        raise ValueError(
            "published migration profile must contain only scoped action mean/std"
        )

    def exact(left, right):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        )

    old_stat_keys = {
        f"{operation}.{scope}_buffer_{feature}.{stat}"
        for scope in scopes
        for operation, feature in (
            ("normalize_inputs", "observation_state"),
            ("normalize_targets", "action"),
            ("unnormalize_outputs", "action"),
        )
        for stat in ("mean", "std")
    }
    verified_numel = 0
    with (
        safe_open(historical_checkpoint, framework="pt", device="cpu") as old,
        safe_open(current_checkpoint, framework="pt", device="cpu") as current,
    ):
        mapping = {
            key: key.replace("model._orig_mod.", "model.", 1)
            for key in old.keys()  # noqa: SIM118 - safe_open is not iterable.
            if key.startswith("model._orig_mod.")
        }
        if (
            not mapping
            or set(mapping.values()) != set(current.keys())
            or len(mapping) != len(set(mapping.values()))
            or set(old.keys()) != set(mapping) | old_stat_keys
        ):
            raise ValueError(
                "historical/current tensor coverage is not an exact known migration"
            )
        for old_key, current_key in mapping.items():
            before, after = old.get_tensor(old_key), current.get_tensor(current_key)
            if not exact(before, after):
                raise ValueError(
                    f"model tensor differs across migration: {current_key}"
                )
            verified_numel += after.numel()
        for scope in scopes:
            for stat in ("mean", "std"):
                expected = published[f"{scope}.buffer.action.{stat}"]
                for operation in ("normalize_targets", "unnormalize_outputs"):
                    key = f"{operation}.{scope}_buffer_action.{stat}"
                    if not exact(old.get_tensor(key), expected):
                        raise ValueError(
                            f"action statistics differ across migration: {key}"
                        )
        state = {
            stat: old.get_tensor(
                f"normalize_inputs.{namespace}_buffer_observation_state.{stat}"
            ).clone()
            for stat in ("mean", "std")
        }
    flat = {
        **{f"observation.state.{stat}": value for stat, value in state.items()},
        **{
            f"action.{stat}": value
            for stat, value in selected_action.stats["action"].items()
        },
    }
    profile = resolve_smolvla_statistics(
        flat,
        namespace=None,
        robot_type=robot_type,
        feature_shapes={"observation.state": state_shape, "action": action_shape},
        source_sha256=historical_sha256,
    )
    return flat, {
        "schema": "vlaforge.smolvla_statistics_recovery/1",
        "status": "verified_weights_and_recovered_statistics",
        "sources": sources,
        "selected_legacy_namespace": namespace,
        "robot_type": robot_type,
        "model_tensor_count": len(mapping),
        "model_tensor_elements": verified_numel,
        "model_tensors_bitwise_equal": True,
        "all_published_action_statistics_bitwise_equal": True,
        "selection": profile.to_dict(),
        "physical_action_units_verified": False,
        "robot_calibration_verified": False,
    }


def materialize_profile(
    *, policy_path: Path, output: Path, statistics: dict, report: dict
) -> dict:
    """Publish canonical upstream-readable stats without changing model weights."""
    from safetensors.torch import save_file

    if output.exists() or output.is_symlink():
        raise ValueError("recovered profile output must be new")
    if (
        sha256(policy_path / "model.safetensors")
        != report["sources"]["current_checkpoint"]["sha256"]
    ):
        raise ValueError(
            "processor source does not belong to the verified current checkpoint"
        )
    configurations = {}
    for name, registry in (
        ("policy_preprocessor.json", "normalizer_processor"),
        ("policy_postprocessor.json", "unnormalizer_processor"),
    ):
        configuration = json.loads((policy_path / name).read_text())
        steps = [
            step for step in configuration["steps"] if step["registry_name"] == registry
        ]
        if len(steps) != 1:
            raise ValueError(
                "processor structure is not the supported checkpoint migration"
            )
        steps[0]["state_file"] = f"{registry}.safetensors"
        configurations[name] = configuration
    output.mkdir(parents=True)
    shutil.copy2(policy_path / "config.json", output / "config.json")
    (output / "model.safetensors").symlink_to(
        (policy_path / "model.safetensors").resolve()
    )
    save_file(statistics, output / "normalizer_processor.safetensors")
    save_file(
        {key: value for key, value in statistics.items() if key.startswith("action.")},
        output / "unnormalizer_processor.safetensors",
    )
    for name, configuration in configurations.items():
        (output / name).write_text(json.dumps(configuration, indent=2) + "\n")
    result = {
        **report,
        "processor_source_files": {
            name: sha256(policy_path / name)
            for name in (
                "config.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            )
        },
        "output_files": {path.name: sha256(path) for path in output.iterdir()},
        "limitations": [
            "recovered checkpoint profile, not the as-published pipeline",
            "physical robot calibration and action units not verified",
        ],
    }
    (output / "recovery.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_recovered_profile(policy_path: Path, *, robot_type: str) -> dict | None:
    """Verify the recorded recovery and every generated processor file on reuse."""
    path = policy_path / "recovery.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "vlaforge.smolvla_statistics_recovery/1"
        or report.get("status") != "verified_weights_and_recovered_statistics"
        or report.get("robot_type") != robot_type
        or report.get("selected_legacy_namespace") != robot_type
        or report.get("model_tensors_bitwise_equal") is not True
        or report.get("all_published_action_statistics_bitwise_equal") is not True
    ):
        raise ValueError("invalid or mismatched statistics recovery provenance")
    expected_files = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "normalizer_processor.safetensors",
        "unnormalizer_processor.safetensors",
    }
    if set(report["output_files"]) != expected_files:
        raise ValueError("recovery output file coverage is incomplete")
    for name, expected in report["output_files"].items():
        if sha256(policy_path / name) != expected:
            raise ValueError(f"recovered profile file digest mismatch: {name}")
    if (
        report["output_files"]["model.safetensors"]
        != report["sources"]["current_checkpoint"]["sha256"]
    ):
        raise ValueError("recovered model differs from verified checkpoint")
    return {"report_sha256": sha256(path), **report}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--historical-sha256", required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--current-sha256", required=True)
    parser.add_argument("--published-statistics", type=Path, required=True)
    parser.add_argument("--published-sha256", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--robot-type", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.policy_path / "config.json").read_text())
    statistics, report = recover_statistics(
        historical_checkpoint=args.historical_checkpoint,
        historical_sha256=args.historical_sha256,
        current_checkpoint=args.policy_path / "model.safetensors",
        current_sha256=args.current_sha256,
        published_statistics=args.published_statistics,
        published_sha256=args.published_sha256,
        namespace=args.namespace,
        robot_type=args.robot_type,
        state_shape=tuple(config["input_features"]["observation.state"]["shape"]),
        action_shape=tuple(config["output_features"]["action"]["shape"]),
    )
    print(
        json.dumps(
            materialize_profile(
                policy_path=args.policy_path,
                output=args.output,
                statistics=statistics,
                report=report,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
