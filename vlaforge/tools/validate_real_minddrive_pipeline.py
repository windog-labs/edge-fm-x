#!/usr/bin/env python3
"""Execute the full captured MindDrive pipeline over a real frame sequence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
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
    MINDDRIVE_PIPELINE_DETECTION_STATE_ASSIGNMENT_MAX,
    MINDDRIVE_PIPELINE_DETECTION_STATE_MAX_ABS,
    MINDDRIVE_PIPELINE_DETECTION_STATE_NRMSE,
    MINDDRIVE_PIPELINE_DETECTION_TOKEN_MAX_ABS,
    MINDDRIVE_PIPELINE_DETECTION_TOKEN_NRMSE,
    MINDDRIVE_PIPELINE_MAP_MAX_ABS,
    MINDDRIVE_PIPELINE_MAP_NRMSE,
    MINDDRIVE_PIPELINE_MAP_STATE_ASSIGNMENT_MAX,
    MINDDRIVE_PIPELINE_TRAJECTORY_MAX_ABS,
    MINDDRIVE_PIPELINE_TRAJECTORY_NRMSE,
    MINDDRIVE_PIPELINE_VISION_MAX_ABS,
    MINDDRIVE_PIPELINE_VISION_NRMSE,
    MINDDRIVE_STATE_TYPES,
    MINDDRIVE_UPSTREAM_STATE_KEYS,
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


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple | list):
        return tuple(value)
    return (value,)


def _aoti_arguments(
    arguments: tuple[Any, ...], device: str
) -> tuple[Any, ...]:
    import torch

    values = []
    for value in arguments:
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "MindDrive AOTI physical Region accepts tensors only"
            )
        values.append(value.to(device).contiguous())
    return tuple(values)


def _aoti_call(
    runner: Any,
    arguments: tuple[Any, ...],
    *,
    device: str,
) -> tuple[Any, ...]:
    # AOTI may return tensors backed by storage owned by the loaded runner.
    # Physical Regions are deliberately loaded and released independently, so
    # merely calling contiguous() is insufficient when the result already has
    # contiguous strides: it can leave the next Region holding runner-owned
    # storage after `del runner`.  Clone every boundary value to transfer
    # ownership to the invocation before the provider is released.
    return tuple(
        value.contiguous().clone()
        for value in _as_tuple(
            runner(*_aoti_arguments(arguments, device))
        )
    )


def _run_aoti_artifact(
    artifact: Path,
    invocations: tuple[tuple[Any, ...], ...],
    *,
    device: str,
) -> tuple[tuple[Any, ...], ...]:
    import torch

    runner = torch._export.aot_load(str(artifact.resolve()), device)
    outputs = []
    with torch.inference_mode():
        for arguments in invocations:
            outputs.append(
                _to_cpu(
                    _aoti_call(runner, arguments, device=device)
                )
            )
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(outputs)


def _run_stateful_aoti_artifact(
    artifact: Path,
    frame_prefixes: tuple[tuple[Any, ...], ...],
    initial_state: tuple[Any, ...],
    *,
    state_output_offset: int,
    device: str,
) -> tuple[tuple[Any, ...], ...]:
    import torch

    runner = torch._export.aot_load(str(artifact.resolve()), device)
    state = _aoti_arguments(initial_state, device)
    outputs = []
    with torch.inference_mode():
        for prefix in frame_prefixes:
            value = _aoti_call(
                runner,
                (*prefix, *state),
                device=device,
            )
            outputs.append(_to_cpu(value))
            state = tuple(
                item.clone()
                for item in value[state_output_offset:]
            )
    del runner
    del state
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(outputs)


def _run_partitioned_aoti_vision(
    source_root: Path,
    artifact_root: Path,
    invocations: tuple[tuple[Any, ...], ...],
    *,
    device: str,
) -> tuple[tuple[Any, ...], ...]:
    """Execute 50 AOTI Regions around the upstream FlashAttention kernel."""

    import torch

    source = str(source_root.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from mmcv.models.utils.attention import FlashAttention

    flash = FlashAttention(attention_dropout=0.0).eval()
    features = [
        _aoti_arguments(arguments, device)[0]
        for arguments in invocations
    ]
    with torch.inference_mode():
        runner = torch._export.aot_load(
            str((artifact_root / "vision_stem.so").resolve()),
            device,
        )
        features = [
            _aoti_call(runner, (value,), device=device)[0]
            for value in features
        ]
        del runner
        for index in range(24):
            runner = torch._export.aot_load(
                str(
                    (
                        artifact_root
                        / f"vision_block_{index:02d}_pre.so"
                    ).resolve()
                ),
                device,
            )
            prepared = [
                _aoti_call(runner, (value,), device=device)
                for value in features
            ]
            del runner
            attention = [
                flash(
                    query.contiguous(),
                    key_value.contiguous(),
                    key_padding_mask=None,
                    causal=False,
                )[0].contiguous()
                for _, query, key_value in prepared
            ]
            runner = torch._export.aot_load(
                str(
                    (
                        artifact_root
                        / f"vision_block_{index:02d}_post.so"
                    ).resolve()
                ),
                device,
            )
            features = [
                _aoti_call(
                    runner,
                    (prepared[frame][0], attention[frame]),
                    device=device,
                )[0]
                for frame in range(len(features))
            ]
            del runner
            del prepared
            del attention
        runner = torch._export.aot_load(
            str((artifact_root / "vision_finish.so").resolve()),
            device,
        )
        outputs = [
            _to_cpu(
                _aoti_call(runner, (value,), device=device)
            )
            for value in features
        ]
        del runner
    del flash
    del features
    gc.collect()
    torch.cuda.empty_cache()
    return tuple(outputs)


def _run_partitioned_aoti_map(
    artifact_root: Path,
    frame_prefixes: tuple[tuple[Any, ...], ...],
    initial_state: tuple[Any, ...],
    *,
    device: str,
) -> tuple[tuple[Any, ...], ...]:
    """Execute the backend-only front/layer/finish map decomposition."""

    import torch

    names = (
        "map_front",
        *(f"map_decoder_layer_{index:02d}" for index in range(6)),
        "map_finish",
    )
    runners = {
        name: torch._export.aot_load(
            str((artifact_root / f"{name}.so").resolve()),
            device,
        )
        for name in names
    }
    state = _aoti_arguments(initial_state, device)
    outputs = []
    with torch.inference_mode():
        for prefix in frame_prefixes:
            (
                image_features,
                position_embedding,
                timestamp,
                ego_pose,
                ego_pose_inverse,
            ) = prefix
            front = _aoti_call(
                runners["map_front"],
                (
                    image_features,
                    timestamp,
                    ego_pose_inverse,
                    *state,
                ),
                device=device,
            )
            (
                query,
                image_memory,
                query_position,
                temporal_attention_mask,
                temporal_memory,
                temporal_position,
                reference_points,
                rec_ego_pose,
                memory_reference_point,
                memory_timestamp,
                memory_egopose,
                memory_mask,
            ) = front
            decoded = []
            for index in range(6):
                query = _aoti_call(
                    runners[f"map_decoder_layer_{index:02d}"],
                    (
                        query,
                        image_memory,
                        query_position,
                        position_embedding,
                        temporal_attention_mask,
                        temporal_memory,
                        temporal_position,
                    ),
                    device=device,
                )[0]
                decoded.append(query)
            value = _aoti_call(
                runners["map_finish"],
                (
                    *decoded,
                    reference_points,
                    timestamp,
                    ego_pose,
                    rec_ego_pose,
                    temporal_memory,
                    memory_reference_point,
                    memory_timestamp,
                    memory_egopose,
                    memory_mask,
                ),
                device=device,
            )
            outputs.append(_to_cpu(value))
            state = tuple(item.clone() for item in value[4:])
    del runners
    del state
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


def _proposal_state_equivalence(
    reference: tuple[Any, ...],
    candidate: tuple[Any, ...],
    *,
    state_names: tuple[str, ...],
    identity_index: int,
    proposal_field_count: int,
    maximum_absolute_error: float,
    normalized_root_mean_square_error: float,
    maximum_assignment_distance: float,
    enforce: bool,
) -> dict[str, object]:
    """Compare authoritative proposal state as age-partitioned row bundles.

    AOTI may exchange near-tied top-k rows. One geometry-derived assignment is
    therefore applied to every tensor in the proposal bundle; fields never
    receive independent best-case permutations. The two 300-row age
    partitions are matched separately so current and previous invocation
    state cannot be interchanged.
    """

    import torch
    from scipy.optimize import linear_sum_assignment

    if len(reference) != len(candidate) or len(reference) != len(state_names):
        raise ValueError("MindDrive authoritative state arity changed")
    proposal_count = int(reference[identity_index].shape[1])
    if proposal_count != 600:
        raise ValueError(
            "MindDrive proposal state profile changed: "
            f"{proposal_count} != 600"
        )
    reference_rows = []
    candidate_rows = []
    partitions = []
    for age, start in enumerate((0, 300)):
        stop = start + 300
        reference_identity = (
            reference[identity_index][0, start:stop]
            .to(torch.float64)
            .flatten(1)
        )
        candidate_identity = (
            candidate[identity_index][0, start:stop]
            .to(torch.float64)
            .flatten(1)
        )
        cost = torch.cdist(reference_identity, candidate_identity).numpy()
        reference_assignment, candidate_assignment = (
            linear_sum_assignment(cost)
        )
        reference_local = torch.from_numpy(reference_assignment)
        candidate_local = torch.from_numpy(candidate_assignment)
        reference_rows.append(reference_local + start)
        candidate_rows.append(candidate_local + start)
        assigned_cost = torch.from_numpy(
            cost[reference_assignment, candidate_assignment]
        )
        displacement = abs(reference_assignment - candidate_assignment)
        partitions.append(
            {
                "age": age,
                "start": start,
                "stop": stop,
                "permuted_row_count": int((displacement != 0).sum()),
                "maximum_rank_displacement": int(displacement.max()),
                "mean_identity_distance": float(
                    assigned_cost.mean().item()
                ),
                "p95_identity_distance": float(
                    torch.quantile(assigned_cost, 0.95).item()
                ),
                "maximum_identity_distance": float(
                    assigned_cost.max().item()
                ),
                "maximum_identity_distance_threshold": (
                    maximum_assignment_distance
                ),
                "assignment_within_threshold": bool(
                    assigned_cost.max().item()
                    <= maximum_assignment_distance
                ),
            }
        )
    reference_order = torch.cat(reference_rows)
    candidate_order = torch.cat(candidate_rows)
    if not torch.equal(reference_order, torch.arange(proposal_count)):
        raise ValueError("proposal assignment did not cover reference rows")

    fields: dict[str, object] = {}
    for index, name in enumerate(state_names):
        reference_value = reference[index]
        candidate_value = candidate[index]
        if index < proposal_field_count:
            reference_value = reference_value[:, reference_order]
            candidate_value = candidate_value[:, candidate_order]
        if (
            reference_value.dtype == torch.bool
            or not reference_value.is_floating_point()
        ):
            fields[name] = _exact(reference_value, candidate_value)
        else:
            fields[name] = _equivalence(
                reference_value,
                candidate_value,
                maximum_absolute_error=maximum_absolute_error,
                normalized_root_mean_square_error=(
                    normalized_root_mean_square_error
                ),
                enforce=enforce,
            )
    assignments_within = all(
        bool(item["assignment_within_threshold"])
        for item in partitions
    )
    fields_within = all(bool(item["passed"]) for item in fields.values())
    within = assignments_within and fields_within
    return {
        "comparison": (
            "two-age-partition geometry assignment with one shared "
            "permutation for every proposal-state field"
        ),
        "identity_state": state_names[identity_index],
        "proposal_rows": proposal_count,
        "partitions": partitions,
        "fields": fields,
        "thresholds": {
            "maximum_identity_distance": maximum_assignment_distance,
            "maximum_absolute_error": maximum_absolute_error,
            "normalized_root_mean_square_error": (
                normalized_root_mean_square_error
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
    parser.add_argument(
        "--second-persistent-state",
        type=Path,
        required=True,
        help=(
            "Upstream authoritative state after the final real frame; "
            "required for L2/L3 state-carry parity."
        ),
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--vision-provider",
        choices=("artifact", "flash-plugin", "partitioned-aoti24"),
        default="artifact",
    )
    parser.add_argument(
        "--execution-provider",
        choices=("export", "aoti24"),
        default="export",
    )
    parser.add_argument("--aoti24-root", type=Path)
    parser.add_argument(
        "--aoti24-artifact-manifest",
        "--aoti24-physical-abi-manifest",
        dest="aoti24_artifact_manifest",
        type=Path,
        help=(
            "Required for AOTI execution; binds all 64 Region captures, "
            "compile reports, physical tensor contracts, and shared objects."
        ),
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
    second_persistent_state = _load_tensor_file(
        args.second_persistent_state
    )
    artifacts = _artifact_paths(args.artifact_root.resolve())
    for name, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} artifact is missing: {path}")
    aoti_artifacts: dict[str, Path] = {}
    artifact_manifest: dict[str, Any] | None = None
    if args.execution_provider == "aoti24":
        if args.aoti24_root is None:
            raise ValueError("aoti24 execution requires --aoti24-root")
        if args.aoti24_artifact_manifest is None:
            raise ValueError(
                "aoti24 execution requires "
                "--aoti24-artifact-manifest"
            )
        aoti_root = args.aoti24_root.resolve()
        names = (
            "position_encoder",
            "detection_encoder",
            "detection_decoder",
            "decision_expert",
            "action_expert",
            "trajectory_decoder",
            "vision_stem",
            "vision_finish",
            "map_front",
            "map_finish",
            *(f"vision_block_{index:02d}_{part}"
              for index in range(24)
              for part in ("pre", "post")),
            *(f"map_decoder_layer_{index:02d}"
              for index in range(6)),
        )
        aoti_artifacts = {
            name: aoti_root / f"{name}.so" for name in names
        }
        for name, path in aoti_artifacts.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"{name} AOTI artifact is missing: {path}"
                )
        artifact_manifest = json.loads(
            args.aoti24_artifact_manifest.resolve().read_text(
                encoding="utf-8"
            )
        )
        if (
            artifact_manifest.get("schema")
            != "vlaforge.minddrive_aoti_artifact_manifest/1"
            or not artifact_manifest.get("passed")
        ):
            raise ValueError(
                "MindDrive AOTI artifact manifest is invalid or failed"
            )
        manifest_root = Path(
            artifact_manifest["artifact_root"]
        ).resolve()
        if manifest_root != aoti_root:
            raise ValueError(
                "MindDrive AOTI artifact root does not match the physical "
                f"ABI manifest: {aoti_root} != {manifest_root}"
            )
        manifest_regions = {
            str(item["name"]): item
            for item in artifact_manifest["regions"]
        }
        execution_names = set(names)
        if set(manifest_regions) != execution_names:
            raise ValueError(
                "MindDrive artifact manifest Region coverage changed"
            )
        for name in sorted(execution_names):
            path = aoti_artifacts[name]
            expected_hash = manifest_regions[name]["artifact"]["sha256"]
            if _sha256(path) != expected_hash:
                raise ValueError(
                    f"{name}: AOTI artifact hash does not match its capture "
                    "contract"
                )
        if args.vision_provider != "partitioned-aoti24":
            raise ValueError(
                "aoti24 execution requires partitioned-aoti24 vision"
            )

    # Vision is run from the real camera tensor; saved official features are
    # references only and never substituted into the captured pipeline.
    vision_invocations = tuple(
        (frame["camera_images"],) for frame in frame_inputs
    )
    if args.vision_provider == "partitioned-aoti24":
        if args.source_root is None:
            raise ValueError(
                "partitioned-aoti24 vision requires --source-root"
            )
        vision = _run_partitioned_aoti_vision(
            args.source_root.resolve(),
            args.aoti24_root.resolve(),
            vision_invocations,
            device=args.device,
        )
    elif args.vision_provider == "flash-plugin":
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
    run_artifact = (
        _run_aoti_artifact
        if args.execution_provider == "aoti24"
        else _run_artifact
    )
    execution_artifacts = (
        aoti_artifacts
        if args.execution_provider == "aoti24"
        else artifacts
    )
    position = run_artifact(
        execution_artifacts["position_encoder"],
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
    map_prefixes = tuple(
        (
            vision[index][0],
            position[index][0],
            frame["timestamp"],
            frame["ego_pose"],
            frame["ego_pose_inverse"],
        )
        for index, frame in enumerate(frame_inputs)
    )
    if args.execution_provider == "aoti24":
        map_outputs = _run_partitioned_aoti_map(
            args.aoti24_root.resolve(),
            map_prefixes,
            tuple(
                initial[name]
                for name, _ in MINDDRIVE_STATE_TYPES[10:]
            ),
            device=args.device,
        )
    else:
        map_outputs = _run_stateful_artifact(
            artifacts["map_encoder"],
            map_prefixes,
            tuple(
                initial[name]
                for name, _ in MINDDRIVE_STATE_TYPES[10:]
            ),
            state_output_offset=4,
            device=args.device,
        )
    detection_prefixes = tuple(
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
    )
    if args.execution_provider == "aoti24":
        detection_outputs = _run_stateful_aoti_artifact(
            aoti_artifacts["detection_encoder"],
            detection_prefixes,
            tuple(
                initial[name]
                for name, _ in MINDDRIVE_STATE_TYPES[:10]
            ),
            state_output_offset=6,
            device=args.device,
        )
    else:
        detection_outputs = _run_stateful_artifact(
            artifacts["detection_encoder"],
            detection_prefixes,
            tuple(
                initial[name]
                for name, _ in MINDDRIVE_STATE_TYPES[:10]
            ),
            state_output_offset=6,
            device=args.device,
        )
    decision = run_artifact(
        execution_artifacts["decision_expert"],
        (
            (
                second["decision_input_ids"],
                detection_outputs[-1][5],
                map_outputs[-1][3],
            ),
        ),
        device=args.device,
    )[0][0]
    action = run_artifact(
        execution_artifacts["action_expert"],
        (
            (
                second["planning_input_ids"],
                detection_outputs[-1][5],
                map_outputs[-1][3],
            ),
        ),
        device=args.device,
    )[0][0]
    trajectory = run_artifact(
        execution_artifacts["trajectory_decoder"],
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
    decoded = run_artifact(
        execution_artifacts["detection_decoder"],
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
    map_state_names = tuple(
        name for name, _ in MINDDRIVE_STATE_TYPES[10:]
    )
    map_state_reference = tuple(
        second_persistent_state[MINDDRIVE_UPSTREAM_STATE_KEYS[name]]
        for name in map_state_names
    )
    map_state_candidate = map_outputs[-1][4:]
    evidence["map_authoritative_state"] = _proposal_state_equivalence(
        map_state_reference,
        map_state_candidate,
        state_names=map_state_names,
        identity_index=1,
        proposal_field_count=4,
        maximum_absolute_error=MINDDRIVE_PIPELINE_MAP_MAX_ABS,
        normalized_root_mean_square_error=MINDDRIVE_PIPELINE_MAP_NRMSE,
        maximum_assignment_distance=(
            MINDDRIVE_PIPELINE_MAP_STATE_ASSIGNMENT_MAX
        ),
        enforce=enforce_numerical,
    )
    detection_state_names = tuple(
        name for name, _ in MINDDRIVE_STATE_TYPES[:10]
    )
    detection_state_reference = tuple(
        second_persistent_state[MINDDRIVE_UPSTREAM_STATE_KEYS[name]]
        for name in detection_state_names
    )
    detection_state_candidate = detection_outputs[-1][6:]
    evidence[
        "detection_authoritative_state"
    ] = _proposal_state_equivalence(
        detection_state_reference,
        detection_state_candidate,
        state_names=detection_state_names,
        identity_index=1,
        proposal_field_count=5,
        maximum_absolute_error=(
            MINDDRIVE_PIPELINE_DETECTION_STATE_MAX_ABS
        ),
        normalized_root_mean_square_error=(
            MINDDRIVE_PIPELINE_DETECTION_STATE_NRMSE
        ),
        maximum_assignment_distance=(
            MINDDRIVE_PIPELINE_DETECTION_STATE_ASSIGNMENT_MAX
        ),
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
            "real-L3-compiled-development-sequence"
            if args.execution_provider == "aoti24"
            and args.evidence_role == "development"
            else "real-L3-compiled-held-out-end-to-end"
            if args.execution_provider == "aoti24"
            else "real-L2-development-sequence"
            if args.evidence_role == "development"
            else "real-L2-held-out-end-to-end"
        ),
        "evidence_role": args.evidence_role,
        "execution_provider": args.execution_provider,
        "artifact_manifest": (
            {
                "path": str(
                    args.aoti24_artifact_manifest.resolve()
                ),
                "sha256": _sha256(
                    args.aoti24_artifact_manifest.resolve()
                ),
                "schema": artifact_manifest["schema"],
                "artifact_set_sha256": (
                    artifact_manifest["artifact_set_sha256"]
                ),
                "region_count": len(artifact_manifest["regions"]),
            }
            if artifact_manifest is not None
            else None
        ),
        "sequence": frame_names,
        "state_semantics": (
            "zero-reset-read-latest-stage-write-commit-per-invocation"
        ),
        "vision_provider": {
            "kind": args.vision_provider,
            "abi": (
                "static-tensor-region-plugin"
                if args.vision_provider == "flash-plugin"
                else (
                    "aoti24-contiguous-physical-regions"
                    "+compiled-flash-cuda"
                )
                if args.vision_provider == "partitioned-aoti24"
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
            for name, path in execution_artifacts.items()
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
            "second_persistent_state_sha256": _sha256(
                args.second_persistent_state
            ),
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
                "map_state_candidate": map_state_candidate,
                "map_state_reference": map_state_reference,
                "detection_candidate_raw": detection_outputs[-1][:6],
                "detection_reference_raw": _raw_detection(
                    second_intermediates
                ),
                "detection_state_candidate": detection_state_candidate,
                "detection_state_reference": detection_state_reference,
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
