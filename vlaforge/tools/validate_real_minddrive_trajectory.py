#!/usr/bin/env python3
"""Execute the captured MindDrive trajectory Region on a held-out frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_TRAJECTORY_DECODER_MAX_ABS,
    MINDDRIVE_TRAJECTORY_DECODER_NRMSE,
)
from vlaforge.frontend import load_exported_region


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equivalence(reference: Any, candidate: Any) -> dict[str, object]:
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
        "normalized_root_mean_square_error": nrmse,
        "thresholds": {
            "maximum_absolute_error": (
                MINDDRIVE_TRAJECTORY_DECODER_MAX_ABS
            ),
            "normalized_root_mean_square_error": (
                MINDDRIVE_TRAJECTORY_DECODER_NRMSE
            ),
        },
        "passed": (
            max_abs <= MINDDRIVE_TRAJECTORY_DECODER_MAX_ABS
            and nrmse <= MINDDRIVE_TRAJECTORY_DECODER_NRMSE
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--invocation-inputs", type=Path, required=True)
    parser.add_argument("--upstream-intermediates", type=Path, required=True)
    parser.add_argument("--upstream-outputs", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

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
    raw_path = torch.argmax(
        invocation_inputs["ego_route_command"][0, 0, 0], dim=-1
    )
    mapping = torch.tensor((2, 4, 1, 0, 3, 5), dtype=torch.int64)
    references = (
        upstream_outputs["ego_fut_preds"],
        upstream_outputs["pw_ego_fut_pred"],
        torch.argmax(intermediates["decision_expert"][0][0], dim=-1),
        torch.gather(mapping, 0, raw_path.reshape(1))[0],
    )
    region_inputs = (
        intermediates["action_expert"].to(args.device),
        intermediates["decision_expert"][0].to(args.device),
        invocation_inputs["ego_route_command"].to(args.device),
        invocation_inputs["trajectory_noise"].to(args.device),
        invocation_inputs["path_noise"].to(args.device),
    )
    exported = load_exported_region(args.artifact.resolve())
    with torch.inference_mode():
        candidates = exported.module()(*region_inputs)
    outputs = {}
    passed = True
    for name, reference, candidate in zip(
        ("trajectory", "path_trajectory", "speed_command", "path_command"),
        references,
        candidates,
        strict=True,
    ):
        if candidate.is_floating_point():
            evidence = _equivalence(reference, candidate.cpu())
        else:
            exact = bool(torch.equal(reference, candidate.cpu()))
            evidence = {"exact": exact, "passed": exact}
        outputs[name] = evidence
        passed = passed and bool(evidence["passed"])
    report = {
        "schema": "vlaforge.minddrive_real_trajectory_heldout/1",
        "passed": passed,
        "evidence_level": "real-L2-held-out-export-execution",
        "frame": args.frame,
        "artifact": {
            "path": str(args.artifact.resolve()),
            "sha256": _sha256(args.artifact),
        },
        "inputs": {
            "invocation_inputs_sha256": _sha256(args.invocation_inputs),
            "upstream_intermediates_sha256": _sha256(
                args.upstream_intermediates
            ),
            "upstream_outputs_sha256": _sha256(args.upstream_outputs),
        },
        "outputs": outputs,
    }
    if not passed:
        raise ValueError(f"held-out trajectory parity failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
