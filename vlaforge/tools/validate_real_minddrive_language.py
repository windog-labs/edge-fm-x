#!/usr/bin/env python3
"""Execute strict-captured MindDrive Qwen experts on a held-out real frame."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_DECISION_DCE_MAX_ABS,
    MINDDRIVE_DECISION_DCE_NRMSE,
)
from vlaforge.frontend import load_exported_region


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
    max_abs_threshold: float,
    nrmse_threshold: float,
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
        "normalized_root_mean_square_error": nrmse,
        "thresholds": {
            "maximum_absolute_error": max_abs_threshold,
            "normalized_root_mean_square_error": nrmse_threshold,
        },
        "passed": max_abs <= max_abs_threshold and nrmse <= nrmse_threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-inputs", type=Path, required=True)
    parser.add_argument("--upstream-intermediates", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    inputs = torch.load(
        args.preprocessed_inputs,
        map_location="cpu",
        weights_only=True,
    )
    intermediates = torch.load(
        args.upstream_intermediates,
        map_location="cpu",
        weights_only=True,
    )
    vision_tokens = torch.cat(
        (
            intermediates["detection_head"][1],
            intermediates["map_head"][1],
        ),
        dim=1,
    ).to(args.device)
    declarations = (
        (
            "decision_expert",
            inputs["decision_input_ids"],
            intermediates["decision_expert"][0],
            MINDDRIVE_DECISION_DCE_MAX_ABS,
            MINDDRIVE_DECISION_DCE_NRMSE,
        ),
        (
            "action_expert",
            inputs["planning_input_ids"],
            intermediates["action_expert"],
            1.0e-6,
            1.0e-7,
        ),
    )
    report: dict[str, object] = {
        "schema": "vlaforge.minddrive_real_language_heldout/1",
        "passed": True,
        "evidence_level": "real-L2-held-out-export-execution",
        "frame": args.frame,
        "inputs": {
            "preprocessed_sha256": _sha256(args.preprocessed_inputs),
            "upstream_intermediates_sha256": _sha256(
                args.upstream_intermediates
            ),
        },
        "regions": {},
    }
    for name, prompt_cpu, reference, max_abs, nrmse in declarations:
        artifact = (args.artifact_dir / f"{name}.pt2e").resolve()
        exported = load_exported_region(artifact)
        prompt = prompt_cpu.to(args.device)
        with torch.inference_mode():
            candidate = exported.module()(prompt, vision_tokens)
        equivalence = _equivalence(
            reference,
            candidate.cpu(),
            max_abs_threshold=max_abs,
            nrmse_threshold=nrmse,
        )
        if not equivalence["passed"]:
            report["passed"] = False
        report["regions"][name] = {
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            "source_equivalence": equivalence,
        }
        del candidate
        del exported
        del prompt
        gc.collect()
        torch.cuda.empty_cache()
    if not report["passed"]:
        raise ValueError(f"held-out language parity failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
