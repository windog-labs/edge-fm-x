#!/usr/bin/env python3
"""Execute the full captured MindDrive pipeline over two real frames."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_POSITION_BACKEND_MAX_ABS,
    MINDDRIVE_POSITION_BACKEND_NRMSE,
    MINDDRIVE_PIPELINE_ACTION_MAX_ABS,
    MINDDRIVE_PIPELINE_ACTION_NRMSE,
    MINDDRIVE_PIPELINE_CONTRACT_VERSION,
    MINDDRIVE_PIPELINE_DECISION_MAX_ABS,
    MINDDRIVE_PIPELINE_DECISION_NRMSE,
    MINDDRIVE_PIPELINE_DETECTION_BOX_P99,
    MINDDRIVE_PIPELINE_DETECTION_CENTER_P95_METERS,
    MINDDRIVE_PIPELINE_DETECTION_COUNT_DELTA,
    MINDDRIVE_PIPELINE_DETECTION_MATCH_FRACTION,
    MINDDRIVE_PIPELINE_DETECTION_MATCH_RADIUS_METERS,
    MINDDRIVE_PIPELINE_DETECTION_MOTION_P99,
    MINDDRIVE_PIPELINE_DETECTION_SCORE_FLOOR,
    MINDDRIVE_PIPELINE_DETECTION_SCORE_P99,
    MINDDRIVE_PIPELINE_DETECTION_TOKEN_MAX_ABS,
    MINDDRIVE_PIPELINE_DETECTION_TOKEN_NRMSE,
    MINDDRIVE_PIPELINE_MAP_MAX_ABS,
    MINDDRIVE_PIPELINE_MAP_NRMSE,
    MINDDRIVE_PIPELINE_TRAJECTORY_MAX_ABS,
    MINDDRIVE_PIPELINE_TRAJECTORY_NRMSE,
    MINDDRIVE_PIPELINE_VISION_MAX_ABS,
    MINDDRIVE_PIPELINE_VISION_NRMSE,
    MINDDRIVE_STATE_TYPES,
    load_real_minddrive_model,
    make_minddrive_flash_vision_encoder,
    make_minddrive_torch_initial_state,
)
from vlaforge.frontend import load_exported_region


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tensor_file(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _image_features(value: Any) -> Any:
    if isinstance(value, dict):
        return value["image_features"]
    return value


def _to_cpu(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    return value


def _to_device(value: Any, device: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _run_artifact(
    artifact: Path,
    invocations: tuple[tuple[Any, ...], ...],
    *,
    device: str,
) -> tuple[tuple[Any, ...], ...]:
    import torch

    exported = load_exported_region(artifact.resolve())
    implementation = exported.module()
    outputs = []
    with torch.inference_mode():
        for arguments in invocations:
            value = implementation(
                *tuple(_to_device(item, device) for item in arguments)
            )
            if not isinstance(value, tuple):
                value = (value,)
            outputs.append(_to_cpu(value))
    del implementation
    del exported
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(outputs)


def _run_flash_vision_plugin(
    source_root: Path,
    release_root: Path,
    invocations: tuple[tuple[Any, ...], ...],
    *,
    device: str,
) -> tuple[tuple[Any, ...], ...]:
    import torch

    model = load_real_minddrive_model(
        source_root,
        release_root,
        device=device,
    )
    implementation = make_minddrive_flash_vision_encoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    outputs = []
    with torch.inference_mode():
        for arguments in invocations:
            value = implementation(
                *tuple(_to_device(item, device) for item in arguments)
            )
            outputs.append((_to_cpu(value),))
    del implementation
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(outputs)


def _run_stateful_artifact(
    artifact: Path,
    frame_prefixes: tuple[tuple[Any, ...], ...],
    initial_state: tuple[Any, ...],
    *,
    state_output_offset: int,
    device: str,
) -> tuple[tuple[Any, ...], ...]:
    import torch

    exported = load_exported_region(artifact.resolve())
    implementation = exported.module()
    state = _to_device(initial_state, device)
    outputs = []
    with torch.inference_mode():
        for prefix in frame_prefixes:
            value = implementation(
                *tuple(_to_device(item, device) for item in prefix),
                *state,
            )
            if not isinstance(value, tuple):
                value = (value,)
            outputs.append(_to_cpu(value))
            state = tuple(item.clone() for item in value[state_output_offset:])
    del implementation
    del exported
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(outputs)


def _equivalence(
    reference: Any,
    candidate: Any,
    *,
    maximum_absolute_error: float,
    normalized_root_mean_square_error: float,
    enforce: bool = True,
) -> dict[str, object]:
    import torch

    reference_fp64 = reference.detach().to(torch.float64)
    candidate_fp64 = candidate.detach().to(torch.float64)
    difference = candidate_fp64 - reference_fp64
    max_abs = float(difference.abs().max().item())
    rmse = float(torch.sqrt(difference.square().mean()).item())
    reference_abs_max = float(reference_fp64.abs().max().item())
    nrmse = rmse / max(reference_abs_max, 1.0e-12)
    within = (
        max_abs <= maximum_absolute_error
        and nrmse <= normalized_root_mean_square_error
    )
    return {
        "maximum_absolute_error": max_abs,
        "root_mean_square_error": rmse,
        "normalized_root_mean_square_error": nrmse,
        "reference_absolute_maximum": reference_abs_max,
        "thresholds": {
            "maximum_absolute_error": maximum_absolute_error,
            "normalized_root_mean_square_error": (
                normalized_root_mean_square_error
            ),
        },
        "threshold_enforced": enforce,
        "passed": within if enforce else True,
        "within_provisional_threshold": within,
    }


def _exact(reference: Any, candidate: Any) -> dict[str, object]:
    import torch

    exact = bool(torch.equal(reference, candidate))
    return {"exact": exact, "passed": exact}


def _detection_set_equivalence(
    reference: tuple[Any, ...],
    candidate: tuple[Any, ...],
    *,
    enforce: bool,
) -> dict[str, object]:
    """Compare unordered detections after a locked confidence floor."""

    import torch
    from scipy.optimize import linear_sum_assignment

    reference_indices = torch.where(
        reference[4]
        & (reference[0] >= MINDDRIVE_PIPELINE_DETECTION_SCORE_FLOOR)
    )[0]
    candidate_indices = torch.where(
        candidate[4]
        & (candidate[0] >= MINDDRIVE_PIPELINE_DETECTION_SCORE_FLOOR)
    )[0]
    reference_rows = []
    candidate_rows = []
    per_label_counts = {}
    labels = sorted(
        set(reference[1][reference_indices].tolist())
        | set(candidate[1][candidate_indices].tolist())
    )
    maximum_label_delta = 0
    for label in labels:
        reference_label = reference_indices[
            reference[1][reference_indices] == label
        ]
        candidate_label = candidate_indices[
            candidate[1][candidate_indices] == label
        ]
        delta = abs(len(reference_label) - len(candidate_label))
        maximum_label_delta = max(maximum_label_delta, delta)
        per_label_counts[str(label)] = {
            "reference": len(reference_label),
            "candidate": len(candidate_label),
            "delta": delta,
        }
        if not len(reference_label) or not len(candidate_label):
            continue
        cost = torch.cdist(
            reference[3][reference_label, :2],
            candidate[3][candidate_label, :2],
        ).numpy()
        reference_assignment, candidate_assignment = (
            linear_sum_assignment(cost)
        )
        reference_rows.append(
            reference_label[torch.from_numpy(reference_assignment)]
        )
        candidate_rows.append(
            candidate_label[torch.from_numpy(candidate_assignment)]
        )
    if not reference_rows:
        return {
            "passed": False if enforce else True,
            "threshold_enforced": enforce,
            "reason": "no matched detections",
        }
    reference_rows_tensor = torch.cat(reference_rows)
    candidate_rows_tensor = torch.cat(candidate_rows)
    center_distance = torch.linalg.vector_norm(
        reference[3][reference_rows_tensor, :2]
        - candidate[3][candidate_rows_tensor, :2],
        dim=1,
    )
    matched_fraction = float(
        (
            center_distance
            <= MINDDRIVE_PIPELINE_DETECTION_MATCH_RADIUS_METERS
        )
        .to(torch.float32)
        .mean()
        .item()
    )
    center_p95 = float(torch.quantile(center_distance, 0.95).item())

    def p99(index: int) -> float:
        difference = (
            reference[index][reference_rows_tensor]
            - candidate[index][candidate_rows_tensor]
        ).abs().flatten()
        return float(torch.quantile(difference, 0.99).item())

    score_p99 = p99(0)
    motion_p99 = p99(2)
    box_p99 = p99(3)
    count_delta = abs(len(reference_indices) - len(candidate_indices))
    within = (
        count_delta <= MINDDRIVE_PIPELINE_DETECTION_COUNT_DELTA
        and maximum_label_delta
        <= MINDDRIVE_PIPELINE_DETECTION_COUNT_DELTA
        and center_p95
        <= MINDDRIVE_PIPELINE_DETECTION_CENTER_P95_METERS
        and matched_fraction
        >= MINDDRIVE_PIPELINE_DETECTION_MATCH_FRACTION
        and score_p99 <= MINDDRIVE_PIPELINE_DETECTION_SCORE_P99
        and box_p99 <= MINDDRIVE_PIPELINE_DETECTION_BOX_P99
        and motion_p99 <= MINDDRIVE_PIPELINE_DETECTION_MOTION_P99
    )
    return {
        "comparison": "per-class-hungarian-xy-center",
        "score_floor": MINDDRIVE_PIPELINE_DETECTION_SCORE_FLOOR,
        "reference_count": len(reference_indices),
        "candidate_count": len(candidate_indices),
        "count_delta": count_delta,
        "maximum_per_label_count_delta": maximum_label_delta,
        "per_label_counts": per_label_counts,
        "matched_count": len(reference_rows_tensor),
        "center_distance_p95_meters": center_p95,
        "fraction_within_match_radius": matched_fraction,
        "score_absolute_error_p99": score_p99,
        "box_absolute_error_p99": box_p99,
        "motion_absolute_error_p99": motion_p99,
        "thresholds": {
            "count_delta": MINDDRIVE_PIPELINE_DETECTION_COUNT_DELTA,
            "maximum_per_label_count_delta": (
                MINDDRIVE_PIPELINE_DETECTION_COUNT_DELTA
            ),
            "center_distance_p95_meters": (
                MINDDRIVE_PIPELINE_DETECTION_CENTER_P95_METERS
            ),
            "match_radius_meters": (
                MINDDRIVE_PIPELINE_DETECTION_MATCH_RADIUS_METERS
            ),
            "fraction_within_match_radius": (
                MINDDRIVE_PIPELINE_DETECTION_MATCH_FRACTION
            ),
            "score_absolute_error_p99": (
                MINDDRIVE_PIPELINE_DETECTION_SCORE_P99
            ),
            "box_absolute_error_p99": (
                MINDDRIVE_PIPELINE_DETECTION_BOX_P99
            ),
            "motion_absolute_error_p99": (
                MINDDRIVE_PIPELINE_DETECTION_MOTION_P99
            ),
        },
        "threshold_enforced": enforce,
        "within_provisional_threshold": within,
        "passed": within if enforce else True,
    }


def _raw_map(intermediates: dict[str, Any]) -> tuple[Any, ...]:
    raw = intermediates["map_head"][0]
    return (
        raw["all_lane_cls_one2one"],
        raw["all_lane_preds_one2one"],
        raw["outs_dec_one2one"],
        intermediates["map_head"][1],
    )


def _raw_detection(intermediates: dict[str, Any]) -> tuple[Any, ...]:
    raw = intermediates["detection_head"][0]
    return (
        raw["all_cls_scores"],
        raw["all_bbox_preds"],
        raw["all_traj_preds"],
        raw["all_traj_cls_scores"],
        raw["all_traffic_states"],
        intermediates["detection_head"][1],
    )


def _record(
    target: dict[str, object],
    name: str,
    reference: Any,
    candidate: Any,
    maximum_absolute_error: float,
    normalized_root_mean_square_error: float,
    *,
    enforce: bool = True,
) -> None:
    target[name] = _equivalence(
        reference,
        candidate,
        maximum_absolute_error=maximum_absolute_error,
        normalized_root_mean_square_error=(
            normalized_root_mean_square_error
        ),
        enforce=enforce,
    )


def _command_references(
    intermediates: dict[str, Any],
    invocation: dict[str, Any],
) -> tuple[Any, Any]:
    import torch

    speed = torch.argmax(intermediates["decision_expert"][0][0], dim=-1)
    raw_path = torch.argmax(
        invocation["ego_route_command"][0, 0, 0], dim=-1
    )
    mapping = torch.tensor((2, 4, 1, 0, 3, 5), dtype=torch.int64)
    path = torch.gather(mapping, 0, raw_path.reshape(1))[0]
    return speed, path


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {
        "vision_encoder": root / "vision" / "vision_encoder.pt2e",
        "position_encoder": (
            root / "perception" / "position_encoder.pt2e"
        ),
        "map_encoder": root / "perception" / "map_encoder.pt2e",
        "detection_encoder": (
            root / "perception" / "detection_encoder.pt2e"
        ),
        "detection_decoder": (
            root / "perception" / "detection_decoder.pt2e"
        ),
        "decision_expert": root / "language" / "decision_expert.pt2e",
        "action_expert": root / "language" / "action_expert.pt2e",
        "trajectory_decoder": (
            root / "trajectory" / "trajectory_decoder.pt2e"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-invocation", type=Path, required=True)
    parser.add_argument("--first-image-features", type=Path, required=True)
    parser.add_argument(
        "--middle-invocation", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--middle-image-features",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--second-invocation", type=Path, required=True)
    parser.add_argument("--second-image-features", type=Path, required=True)
    parser.add_argument("--second-intermediates", type=Path, required=True)
    parser.add_argument("--second-outputs", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--vision-provider",
        choices=("artifact", "flash-plugin"),
        default="artifact",
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tensors", type=Path)
    parser.add_argument(
        "--evidence-role",
        choices=("development", "heldout"),
        required=True,
    )
    parser.add_argument("--first-frame", required=True)
    parser.add_argument("--middle-frame", action="append", default=[])
    parser.add_argument("--second-frame", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    first = _load_tensor_file(args.first_invocation)
    second = _load_tensor_file(args.second_invocation)
    second_features_reference = _image_features(
        _load_tensor_file(args.second_image_features)
    )
    frame_inputs = [first]
    frame_names = [args.first_frame]
    if not (
        len(args.middle_invocation)
        == len(args.middle_image_features)
        == len(args.middle_frame)
    ):
        raise ValueError(
            "middle invocation, image features, and frame counts must match"
        )
    for middle_invocation, middle_frame in zip(
        args.middle_invocation,
        args.middle_frame,
        strict=True,
    ):
        frame_inputs.append(_load_tensor_file(middle_invocation))
        frame_names.append(middle_frame)
    frame_inputs.append(second)
    frame_names.append(args.second_frame)
    second_intermediates = _load_tensor_file(args.second_intermediates)
    second_outputs = _load_tensor_file(args.second_outputs)
    artifacts = _artifact_paths(args.artifact_root.resolve())
    for name, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} artifact is missing: {path}")

    # Vision is run from the real camera tensor; saved official features are
    # references only and never substituted into the captured pipeline.
    vision_invocations = tuple(
        (frame["camera_images"],) for frame in frame_inputs
    )
    if args.vision_provider == "flash-plugin":
        if args.source_root is None or args.release_root is None:
            raise ValueError(
                "flash-plugin requires --source-root and --release-root"
            )
        vision = _run_flash_vision_plugin(
            args.source_root.resolve(),
            args.release_root.resolve(),
            vision_invocations,
            device=args.device,
        )
    else:
        vision = _run_artifact(
            artifacts["vision_encoder"],
            vision_invocations,
            device=args.device,
        )
    position = _run_artifact(
        artifacts["position_encoder"],
        tuple(
            (
                vision[index][0],
                frame["lidar2img"],
                frame["camera_intrinsics"],
            )
            for index, frame in enumerate(frame_inputs)
        ),
        device=args.device,
    )
    initial = make_minddrive_torch_initial_state(torch, device="cpu")
    map_outputs = _run_stateful_artifact(
        artifacts["map_encoder"],
        tuple(
            (
                vision[index][0],
                position[index][0],
                frame["timestamp"],
                frame["ego_pose"],
                frame["ego_pose_inverse"],
            )
            for index, frame in enumerate(frame_inputs)
        ),
        tuple(initial[name] for name, _ in MINDDRIVE_STATE_TYPES[10:]),
        state_output_offset=4,
        device=args.device,
    )
    detection_outputs = _run_stateful_artifact(
        artifacts["detection_encoder"],
        tuple(
            (
                vision[index][0],
                position[index][0],
                *map_outputs[index][:3],
                frame["timestamp"],
                frame["ego_pose"],
                frame["ego_pose_inverse"],
                frame["can_bus"],
                frame["route_command_index"],
            )
            for index, frame in enumerate(frame_inputs)
        ),
        tuple(initial[name] for name, _ in MINDDRIVE_STATE_TYPES[:10]),
        state_output_offset=6,
        device=args.device,
    )
    decision = _run_artifact(
        artifacts["decision_expert"],
        (
            (
                second["decision_input_ids"],
                detection_outputs[-1][5],
                map_outputs[-1][3],
            ),
        ),
        device=args.device,
    )[0][0]
    action = _run_artifact(
        artifacts["action_expert"],
        (
            (
                second["planning_input_ids"],
                detection_outputs[-1][5],
                map_outputs[-1][3],
            ),
        ),
        device=args.device,
    )[0][0]
    trajectory = _run_artifact(
        artifacts["trajectory_decoder"],
        (
            (
                action,
                decision,
                second["ego_route_command"],
                second["trajectory_noise"],
                second["path_noise"],
            ),
        ),
        device=args.device,
    )[0]
    decoded = _run_artifact(
        artifacts["detection_decoder"],
        (
            detection_outputs[-1][:3],
            _raw_detection(second_intermediates)[:3],
        ),
        device=args.device,
    )
    decoded_candidate, decoded_reference = decoded

    evidence: dict[str, object] = {}
    enforce_numerical = args.evidence_role == "heldout"
    _record(
        evidence,
        "vision_features",
        second_features_reference,
        vision[-1][0],
        MINDDRIVE_PIPELINE_VISION_MAX_ABS,
        MINDDRIVE_PIPELINE_VISION_NRMSE,
        enforce=enforce_numerical,
    )
    _record(
        evidence,
        "position_embedding",
        second_intermediates["position_embedding"],
        position[-1][0],
        MINDDRIVE_POSITION_BACKEND_MAX_ABS,
        MINDDRIVE_POSITION_BACKEND_NRMSE,
        enforce=enforce_numerical,
    )
    map_reference = _raw_map(second_intermediates)
    for index, name in enumerate(
        ("map_classes", "map_coordinates", "map_queries", "map_tokens")
    ):
        _record(
            evidence,
            name,
            map_reference[index],
            map_outputs[-1][index],
            MINDDRIVE_PIPELINE_MAP_MAX_ABS,
            MINDDRIVE_PIPELINE_MAP_NRMSE,
            enforce=enforce_numerical,
        )
    _record(
        evidence,
        "detection_tokens",
        _raw_detection(second_intermediates)[5],
        detection_outputs[-1][5],
        MINDDRIVE_PIPELINE_DETECTION_TOKEN_MAX_ABS,
        MINDDRIVE_PIPELINE_DETECTION_TOKEN_NRMSE,
        enforce=enforce_numerical,
    )
    _record(
        evidence,
        "decision_logits",
        second_intermediates["decision_expert"][0],
        decision,
        MINDDRIVE_PIPELINE_DECISION_MAX_ABS,
        MINDDRIVE_PIPELINE_DECISION_NRMSE,
        enforce=enforce_numerical,
    )
    _record(
        evidence,
        "action_hidden",
        second_intermediates["action_expert"],
        action,
        MINDDRIVE_PIPELINE_ACTION_MAX_ABS,
        MINDDRIVE_PIPELINE_ACTION_NRMSE,
        enforce=enforce_numerical,
    )
    _record(
        evidence,
        "trajectory",
        second_outputs["ego_fut_preds"],
        trajectory[0],
        MINDDRIVE_PIPELINE_TRAJECTORY_MAX_ABS,
        MINDDRIVE_PIPELINE_TRAJECTORY_NRMSE,
        enforce=enforce_numerical,
    )
    _record(
        evidence,
        "path_trajectory",
        second_outputs["pw_ego_fut_pred"],
        trajectory[1],
        MINDDRIVE_PIPELINE_TRAJECTORY_MAX_ABS,
        MINDDRIVE_PIPELINE_TRAJECTORY_NRMSE,
        enforce=enforce_numerical,
    )
    speed_reference, path_reference = _command_references(
        second_intermediates, second
    )
    evidence["speed_command"] = _exact(speed_reference, trajectory[2])
    evidence["path_command"] = _exact(path_reference, trajectory[3])
    evidence["detection_set"] = _detection_set_equivalence(
        decoded_reference,
        decoded_candidate,
        enforce=enforce_numerical,
    )
    for name, index in (
        ("detection_valid_mask", 4),
        ("detection_valid_count", 5),
    ):
        evidence[name] = _exact(
            decoded_reference[index], decoded_candidate[index]
        )

    passed = all(bool(item["passed"]) for item in evidence.values())
    report = {
        "schema": "vlaforge.minddrive_real_pipeline_validation/1",
        "numerical_contract_version": (
            MINDDRIVE_PIPELINE_CONTRACT_VERSION
        ),
        "passed": passed,
        "evidence_level": (
            "real-L2-development-sequence"
            if args.evidence_role == "development"
            else "real-L2-held-out-end-to-end"
        ),
        "evidence_role": args.evidence_role,
        "sequence": frame_names,
        "state_semantics": (
            "zero-reset-read-latest-stage-write-commit-per-invocation"
        ),
        "vision_provider": {
            "kind": args.vision_provider,
            "abi": (
                "static-tensor-region-plugin"
                if args.vision_provider == "flash-plugin"
                else "torch-export"
            ),
            "source_root": (
                str(args.source_root.resolve())
                if args.source_root is not None
                else None
            ),
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in artifacts.items()
        },
        "inputs": {
            "first_invocation_sha256": _sha256(args.first_invocation),
            "first_image_features_sha256": _sha256(
                args.first_image_features
            ),
            "middle_invocation_sha256": [
                _sha256(path) for path in args.middle_invocation
            ],
            "middle_image_features_sha256": [
                _sha256(path) for path in args.middle_image_features
            ],
            "second_invocation_sha256": _sha256(args.second_invocation),
            "second_image_features_sha256": _sha256(
                args.second_image_features
            ),
            "second_intermediates_sha256": _sha256(
                args.second_intermediates
            ),
            "second_outputs_sha256": _sha256(args.second_outputs),
        },
        "outputs": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_tensors is not None:
        args.output_tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "map_candidate": map_outputs[-1][:4],
                "map_reference": _raw_map(second_intermediates),
                "detection_candidate_raw": detection_outputs[-1][:6],
                "detection_reference_raw": _raw_detection(
                    second_intermediates
                ),
                "detection_candidate_decoded": decoded_candidate,
                "detection_reference_decoded": decoded_reference,
                "decision_candidate": decision,
                "action_candidate": action,
                "trajectory_candidate": trajectory,
            },
            args.output_tensors,
        )
        report["output_tensors"] = {
            "path": str(args.output_tensors.resolve()),
            "sha256": _sha256(args.output_tensors),
        }
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not passed:
        raise ValueError(f"MindDrive pipeline validation failed: {report}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
