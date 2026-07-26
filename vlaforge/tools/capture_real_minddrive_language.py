#!/usr/bin/env python3
"""Strict-export MindDrive's real Qwen2 decision and action experts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_DECISION_DCE_MAX_ABS,
    MINDDRIVE_DECISION_DCE_NRMSE,
    build_real_minddrive_program,
    load_real_minddrive_model,
    make_minddrive_action_expert,
    make_minddrive_decision_expert,
)
from vlaforge.frontend import capture_region, save_exported_region


_ACTION_MAX_ABS = 1.0e-6
_ACTION_NRMSE = 1.0e-7


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--preprocessed-inputs", type=Path, required=True)
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
    detection_tokens_cpu = intermediates["detection_head"][1]
    map_tokens_cpu = intermediates["map_head"][1]
    declarations = (
        (
            "decision_expert",
            make_minddrive_decision_expert,
            inputs["decision_input_ids"],
            intermediates["decision_expert"][0],
            MINDDRIVE_DECISION_DCE_MAX_ABS,
            MINDDRIVE_DECISION_DCE_NRMSE,
            "full-vocabulary-projection-to-seven-exact-rows",
        ),
        (
            "action_expert",
            make_minddrive_action_expert,
            inputs["planning_input_ids"],
            intermediates["action_expert"],
            _ACTION_MAX_ABS,
            _ACTION_NRMSE,
            "static-waypoint-token-hidden-selection",
        ),
    )
    program = build_real_minddrive_program(device=args.device)
    report: dict[str, object] = {
        "schema": "vlaforge.minddrive_real_language_capture/1",
        "passed": True,
        "evidence_level": "real-L2-region-capture",
        "inputs": {
            "preprocessed": str(args.preprocessed_inputs.resolve()),
            "preprocessed_sha256": _sha256(args.preprocessed_inputs),
            "upstream_intermediates": str(
                args.upstream_intermediates.resolve()
            ),
            "upstream_intermediates_sha256": _sha256(
                args.upstream_intermediates
            ),
        },
        "regions": {},
    }
    for (
        name,
        factory,
        prompt_cpu,
        reference_cpu,
        max_abs,
        nrmse,
        compiler_transform,
    ) in declarations:
        started = time.perf_counter()
        torch.cuda.empty_cache()
        model = load_real_minddrive_model(
            args.source_root,
            args.release_root,
            device=args.device,
            verify_hashes=not args.skip_release_hashes,
        )
        implementation = factory(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        prompt = prompt_cpu.to(args.device)
        detection_tokens = detection_tokens_cpu.to(args.device)
        map_tokens = map_tokens_cpu.to(args.device)
        with torch.inference_mode():
            eager = implementation(
                prompt, detection_tokens, map_tokens
            )
        equivalence = _equivalence(
            reference_cpu,
            eager.cpu(),
            maximum_absolute_error=max_abs,
            normalized_root_mean_square_error=nrmse,
        )
        if not equivalence["passed"]:
            raise ValueError(
                f"{name} failed locked source-equivalence thresholds: "
                f"{equivalence}"
            )
        del eager
        torch.cuda.empty_cache()
        capture = capture_region(
            program.region(name),
            implementation,
            (prompt, detection_tokens, map_tokens),
            strict=True,
            absolute_tolerance=1.0e-5,
            relative_tolerance=1.0e-5,
        )
        capture.require_supported()
        program_path = output / f"{name}.pt2e"
        evidence_path = output / f"{name}.capture.json"
        save_exported_region(
            capture,
            program_path=program_path,
            evidence_path=evidence_path,
        )
        report["regions"][name] = {
            "compiler_transform": compiler_transform,
            "source_equivalence": equivalence,
            "strict_export": capture.evidence.to_dict(),
            "artifact": {
                "path": str(program_path),
                "size_bytes": program_path.stat().st_size,
                "sha256": _sha256(program_path),
            },
            "capture_evidence": {
                "path": str(evidence_path),
                "sha256": _sha256(evidence_path),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        del capture
        del implementation
        del prompt
        del detection_tokens
        del map_tokens
        gc.collect()
        torch.cuda.empty_cache()

    report_path = output / "language_capture_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
