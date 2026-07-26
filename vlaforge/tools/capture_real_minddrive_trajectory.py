#!/usr/bin/env python3
"""Strict-export MindDrive's real probabilistic trajectory decoder."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_TRAJECTORY_DECODER_MAX_ABS,
    MINDDRIVE_TRAJECTORY_DECODER_NRMSE,
    build_real_minddrive_program,
    load_real_minddrive_model,
    make_minddrive_trajectory_decoder,
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

    reference_fp64 = reference.detach().to(torch.float64)
    candidate_fp64 = candidate.detach().to(torch.float64)
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


def _command_references(
    intermediates: dict[str, Any],
    invocation_inputs: dict[str, Any],
) -> tuple[Any, Any]:
    import torch

    speed = torch.argmax(intermediates["decision_expert"][0][0], dim=-1)
    raw_path = torch.argmax(
        invocation_inputs["ego_route_command"][0, 0, 0], dim=-1
    )
    mapping = torch.tensor((2, 4, 1, 0, 3, 5), dtype=torch.int64)
    return speed, torch.gather(mapping, 0, raw_path.reshape(1))[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--invocation-inputs", type=Path, required=True)
    parser.add_argument("--upstream-intermediates", type=Path, required=True)
    parser.add_argument("--upstream-outputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-release-hashes", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    invocation_inputs = torch.load(
        args.invocation_inputs, map_location="cpu", weights_only=True
    )
    intermediates = torch.load(
        args.upstream_intermediates,
        map_location="cpu",
        weights_only=True,
    )
    upstream_outputs = torch.load(
        args.upstream_outputs, map_location="cpu", weights_only=True
    )
    speed_reference, path_reference = _command_references(
        intermediates, invocation_inputs
    )
    references = (
        upstream_outputs["ego_fut_preds"],
        upstream_outputs["pw_ego_fut_pred"],
        speed_reference,
        path_reference,
    )
    region_inputs = (
        intermediates["action_expert"].to(args.device),
        intermediates["decision_expert"][0].to(args.device),
        invocation_inputs["ego_route_command"].to(args.device),
        invocation_inputs["trajectory_noise"].to(args.device),
        invocation_inputs["path_noise"].to(args.device),
    )

    started = time.perf_counter()
    model = load_real_minddrive_model(
        args.source_root,
        args.release_root,
        device=args.device,
        verify_hashes=not args.skip_release_hashes,
    )
    implementation = make_minddrive_trajectory_decoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    with torch.inference_mode():
        eager = implementation(*region_inputs)
    source_equivalence = {}
    for name, reference, candidate in zip(
        ("trajectory", "path_trajectory", "speed_command", "path_command"),
        references,
        eager,
        strict=True,
    ):
        if candidate.is_floating_point():
            evidence = _equivalence(
                reference,
                candidate.cpu(),
                maximum_absolute_error=(
                    MINDDRIVE_TRAJECTORY_DECODER_MAX_ABS
                ),
                normalized_root_mean_square_error=(
                    MINDDRIVE_TRAJECTORY_DECODER_NRMSE
                ),
            )
        else:
            exact = bool(torch.equal(reference, candidate.cpu()))
            evidence = {"exact": exact, "passed": exact}
        source_equivalence[name] = evidence
        if not evidence["passed"]:
            raise ValueError(
                f"{name} failed locked source equivalence: {evidence}"
            )

    program = build_real_minddrive_program(device=args.device)
    capture = capture_region(
        program.region("trajectory_decoder"),
        implementation,
        region_inputs,
        strict=True,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-6,
    )
    capture.require_supported()
    artifact = output / "trajectory_decoder.pt2e"
    capture_evidence = output / "trajectory_decoder.capture.json"
    save_exported_region(
        capture,
        program_path=artifact,
        evidence_path=capture_evidence,
    )
    report = {
        "schema": "vlaforge.minddrive_real_trajectory_capture/1",
        "passed": True,
        "evidence_level": "real-L2-region-capture",
        "rng_semantics": "two-explicit-gaussian-tensor-inputs",
        "compiler_transform": (
            "named-GRU-parameters-to-equivalent-ATen-GRU-without-"
            "flat-weight-storage-mutation"
        ),
        "inputs": {
            "invocation_inputs_sha256": _sha256(args.invocation_inputs),
            "upstream_intermediates_sha256": _sha256(
                args.upstream_intermediates
            ),
            "upstream_outputs_sha256": _sha256(args.upstream_outputs),
        },
        "source_equivalence": source_equivalence,
        "strict_export": capture.evidence.to_dict(),
        "artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        },
        "capture_evidence": {
            "path": str(capture_evidence),
            "sha256": _sha256(capture_evidence),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / "trajectory_capture_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
