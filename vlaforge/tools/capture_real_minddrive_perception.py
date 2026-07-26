#!/usr/bin/env python3
"""Strict-export MindDrive's real stateful perception Regions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_DECODED_BOX_MAX_ABS,
    MINDDRIVE_DECODED_BOX_NRMSE,
    MINDDRIVE_DECODED_MOTION_MAX_ABS,
    MINDDRIVE_DECODED_MOTION_NRMSE,
    MINDDRIVE_DECODED_SCORE_MAX_ABS,
    MINDDRIVE_DECODED_SCORE_NRMSE,
    MINDDRIVE_DETECTION_BACKEND_MAX_ABS,
    MINDDRIVE_DETECTION_BACKEND_NRMSE,
    MINDDRIVE_MAP_BACKEND_MAX_ABS,
    MINDDRIVE_MAP_BACKEND_NRMSE,
    MINDDRIVE_POSITION_BACKEND_MAX_ABS,
    MINDDRIVE_POSITION_BACKEND_NRMSE,
    MINDDRIVE_STATE_TYPES,
    build_real_minddrive_program,
    load_real_minddrive_model,
    make_minddrive_detection_decoder,
    make_minddrive_detection_encoder,
    make_minddrive_map_encoder,
    make_minddrive_position_encoder,
    make_minddrive_torch_initial_state,
)
from vlaforge.frontend import capture_region, save_exported_region


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equivalence(
    reference: Any,
    candidate: Any,
    *,
    maximum_absolute_error: float,
    normalized_root_mean_square_error: float,
) -> dict[str, object]:
    import torch

    reference_fp64 = reference.detach().to(
        device="cpu", dtype=torch.float64
    )
    candidate_fp64 = candidate.detach().to(
        device="cpu", dtype=torch.float64
    )
    difference = candidate_fp64 - reference_fp64
    max_abs = float(difference.abs().max().item())
    rmse = float(torch.sqrt(difference.square().mean()).item())
    reference_abs_max = float(reference_fp64.abs().max().item())
    nrmse = rmse / max(reference_abs_max, 1.0e-12)
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
        "passed": (
            max_abs <= maximum_absolute_error
            and nrmse <= normalized_root_mean_square_error
        ),
    }


def _require_all(
    section: str,
    names: tuple[str, ...],
    references: tuple[Any, ...],
    candidates: tuple[Any, ...],
    *,
    maximum_absolute_error: float,
    normalized_root_mean_square_error: float,
) -> dict[str, object]:
    evidence = {}
    for name, reference, candidate in zip(
        names, references, candidates, strict=True
    ):
        item = _equivalence(
            reference,
            candidate.detach().cpu(),
            maximum_absolute_error=maximum_absolute_error,
            normalized_root_mean_square_error=(
                normalized_root_mean_square_error
            ),
        )
        evidence[name] = item
        if not item["passed"]:
            raise ValueError(
                f"{section}.{name} failed locked source equivalence: {item}"
            )
    return evidence


def _save_capture(
    output: Path,
    program: Any,
    name: str,
    implementation: Any,
    region_inputs: tuple[Any, ...],
) -> tuple[dict[str, object], Any]:
    import torch

    with torch.inference_mode():
        eager = implementation(*region_inputs)
    capture = capture_region(
        program.region(name),
        implementation,
        region_inputs,
        strict=True,
        absolute_tolerance=1.0e-5,
        relative_tolerance=1.0e-5,
    )
    capture.require_supported()
    artifact = output / f"{name}.pt2e"
    evidence = output / f"{name}.capture.json"
    save_exported_region(
        capture,
        program_path=artifact,
        evidence_path=evidence,
    )
    return (
        {
            "strict_export": capture.evidence.to_dict(),
            "artifact": {
                "path": str(artifact.resolve()),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256(artifact),
            },
            "capture_evidence": {
                "path": str(evidence.resolve()),
                "sha256": _sha256(evidence),
            },
        },
        eager,
    )


def _raw_map_references(intermediates: dict[str, Any]) -> tuple[Any, ...]:
    raw = intermediates["map_head"][0]
    return (
        raw["all_lane_cls_one2one"],
        raw["all_lane_preds_one2one"],
        raw["outs_dec_one2one"],
        intermediates["map_head"][1],
    )


def _raw_detection_references(
    intermediates: dict[str, Any],
) -> tuple[Any, ...]:
    raw = intermediates["detection_head"][0]
    return (
        raw["all_cls_scores"],
        raw["all_bbox_preds"],
        raw["all_traj_preds"],
        raw["all_traj_cls_scores"],
        raw["all_traffic_states"],
        intermediates["detection_head"][1],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--invocation-inputs", type=Path, required=True)
    parser.add_argument("--image-features", type=Path, required=True)
    parser.add_argument("--upstream-intermediates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-release-hashes", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation = torch.load(
        args.invocation_inputs, map_location="cpu", weights_only=True
    )
    image_features_cpu = torch.load(
        args.image_features, map_location="cpu", weights_only=True
    )
    if isinstance(image_features_cpu, dict):
        image_features_cpu = image_features_cpu["image_features"]
    intermediates = torch.load(
        args.upstream_intermediates,
        map_location="cpu",
        weights_only=True,
    )
    image_features = image_features_cpu.to(args.device)
    state = make_minddrive_torch_initial_state(
        torch, device=args.device
    )
    program = build_real_minddrive_program(device=args.device)
    report: dict[str, object] = {
        "schema": "vlaforge.minddrive_real_perception_capture/1",
        "passed": True,
        "evidence_level": "real-L2-region-capture",
        "backend_substitution": "flash-attn-2-to-aten-sdpa",
        "calibration_frame": "00400",
        "inputs": {
            "invocation_inputs_sha256": _sha256(args.invocation_inputs),
            "image_features_sha256": _sha256(args.image_features),
            "upstream_intermediates_sha256": _sha256(
                args.upstream_intermediates
            ),
        },
        "regions": {},
    }

    started = time.perf_counter()
    model = load_real_minddrive_model(
        args.source_root,
        args.release_root,
        device=args.device,
        verify_hashes=not args.skip_release_hashes,
    )
    implementation = make_minddrive_position_encoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    position_inputs = (
        image_features,
        invocation["lidar2img"].to(args.device),
        invocation["camera_intrinsics"].to(args.device),
    )
    capture_record, position = _save_capture(
        output,
        program,
        "position_encoder",
        implementation,
        position_inputs,
    )
    capture_record["source_equivalence"] = _require_all(
        "position_encoder",
        ("position_embedding",),
        (intermediates["position_embedding"],),
        (position,),
        maximum_absolute_error=MINDDRIVE_POSITION_BACKEND_MAX_ABS,
        normalized_root_mean_square_error=MINDDRIVE_POSITION_BACKEND_NRMSE,
    )
    capture_record["elapsed_seconds"] = time.perf_counter() - started
    report["regions"]["position_encoder"] = capture_record
    del implementation
    gc.collect()
    torch.cuda.empty_cache()

    started = time.perf_counter()
    model = load_real_minddrive_model(
        args.source_root,
        args.release_root,
        device=args.device,
        verify_hashes=not args.skip_release_hashes,
    )
    implementation = make_minddrive_map_encoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    map_inputs = (
        image_features,
        position,
        invocation["timestamp"].to(args.device),
        invocation["ego_pose"].to(args.device),
        invocation["ego_pose_inverse"].to(args.device),
        *(
            state[name]
            for name, _ in MINDDRIVE_STATE_TYPES[10:]
        ),
    )
    capture_record, map_outputs = _save_capture(
        output,
        program,
        "map_encoder",
        implementation,
        map_inputs,
    )
    capture_record["source_equivalence"] = _require_all(
        "map_encoder",
        ("classes", "coordinates", "queries", "tokens"),
        _raw_map_references(intermediates),
        map_outputs[:4],
        maximum_absolute_error=MINDDRIVE_MAP_BACKEND_MAX_ABS,
        normalized_root_mean_square_error=MINDDRIVE_MAP_BACKEND_NRMSE,
    )
    capture_record["elapsed_seconds"] = time.perf_counter() - started
    report["regions"]["map_encoder"] = capture_record
    # ``torch.inference_mode`` marks outputs with inference-only storage.
    # Detection export is an independent Region boundary and therefore needs
    # ordinary borrowed input tensors, just as a runtime Session would bind.
    map_outputs = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in map_outputs
    )
    del implementation
    gc.collect()
    torch.cuda.empty_cache()

    started = time.perf_counter()
    model = load_real_minddrive_model(
        args.source_root,
        args.release_root,
        device=args.device,
        verify_hashes=not args.skip_release_hashes,
    )
    implementation = make_minddrive_detection_encoder(model)
    decoder = make_minddrive_detection_decoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    detection_inputs = (
        image_features,
        position,
        *map_outputs[:3],
        invocation["timestamp"].to(args.device),
        invocation["ego_pose"].to(args.device),
        invocation["ego_pose_inverse"].to(args.device),
        invocation["can_bus"].to(args.device),
        invocation["route_command_index"].to(args.device),
        *(
            state[name]
            for name, _ in MINDDRIVE_STATE_TYPES[:10]
        ),
    )
    capture_record, detection_outputs = _save_capture(
        output,
        program,
        "detection_encoder",
        implementation,
        detection_inputs,
    )
    capture_record["source_equivalence"] = _require_all(
        "detection_encoder",
        (
            "classes",
            "boxes",
            "trajectories",
            "trajectory_classes",
            "traffic_states",
            "tokens",
        ),
        _raw_detection_references(intermediates),
        detection_outputs[:6],
        maximum_absolute_error=MINDDRIVE_DETECTION_BACKEND_MAX_ABS,
        normalized_root_mean_square_error=(
            MINDDRIVE_DETECTION_BACKEND_NRMSE
        ),
    )
    capture_record["elapsed_seconds"] = time.perf_counter() - started
    report["regions"]["detection_encoder"] = capture_record
    del implementation
    gc.collect()
    torch.cuda.empty_cache()

    raw_detection = tuple(
        value.to(args.device)
        for value in _raw_detection_references(intermediates)
    )
    with torch.inference_mode():
        decoded_reference = decoder(*raw_detection[:3])
    started = time.perf_counter()
    capture_record, decoded = _save_capture(
        output,
        program,
        "detection_decoder",
        decoder,
        detection_outputs[:3],
    )
    decoded_evidence: dict[str, object] = {}
    float_contracts = (
        (
            "detection_scores",
            MINDDRIVE_DECODED_SCORE_MAX_ABS,
            MINDDRIVE_DECODED_SCORE_NRMSE,
        ),
        (
            "motion_trajectories",
            MINDDRIVE_DECODED_MOTION_MAX_ABS,
            MINDDRIVE_DECODED_MOTION_NRMSE,
        ),
        (
            "detection_boxes",
            MINDDRIVE_DECODED_BOX_MAX_ABS,
            MINDDRIVE_DECODED_BOX_NRMSE,
        ),
    )
    for name, reference_index, candidate_index, max_abs, nrmse in (
        (float_contracts[0][0], 0, 0, *float_contracts[0][1:]),
        (float_contracts[1][0], 2, 2, *float_contracts[1][1:]),
        (float_contracts[2][0], 3, 3, *float_contracts[2][1:]),
    ):
        decoded_evidence[name] = _require_all(
            "detection_decoder",
            (name,),
            (decoded_reference[reference_index],),
            (decoded[candidate_index],),
            maximum_absolute_error=max_abs,
            normalized_root_mean_square_error=nrmse,
        )[name]
    for name, index in (
        ("detection_labels", 1),
        ("detection_valid_mask", 4),
        ("detection_valid_count", 5),
    ):
        exact = bool(
            torch.equal(
                decoded_reference[index].detach().cpu(),
                decoded[index].detach().cpu(),
            )
        )
        decoded_evidence[name] = {"exact": exact, "passed": exact}
        if not exact:
            raise ValueError(
                f"detection_decoder.{name} failed exact equivalence"
            )
    capture_record["source_equivalence"] = decoded_evidence
    capture_record["elapsed_seconds"] = time.perf_counter() - started
    report["regions"]["detection_decoder"] = capture_record

    args_report = output / "perception_capture_report.json"
    args_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
