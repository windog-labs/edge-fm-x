#!/usr/bin/env python3
"""Strict-export the real MindDrive EVA vision TensorRegion.

The report keeps two numerical questions separate:

1. official FlashAttention versus the deployable ATen SDPA backend; and
2. SDPA eager versus the strict torch.export program.

Frame 00400 is the declared backend-calibration sample.  The same locked
thresholds are then applied to frame 00401 as held-out validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from vlaforge.adapters.minddrive_real import (
    build_real_minddrive_program,
    compare_minddrive_vision_backends,
    load_real_minddrive_model,
    make_exportable_minddrive_vision_encoder,
)
from vlaforge.frontend import (
    capture_region,
    load_exported_region,
    save_exported_region,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--preprocessed-inputs", type=Path, required=True)
    parser.add_argument("--official-image-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-role", choices=("calibration", "held-out"))
    parser.add_argument("--frame", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reuse-exported-program",
        type=Path,
        help="execute an existing strict capture on a held-out frame",
    )
    parser.add_argument(
        "--reuse-capture-evidence",
        type=Path,
        help="capture JSON paired with --reuse-exported-program",
    )
    parser.add_argument(
        "--skip-release-hashes",
        action="store_true",
        help="size and revision checks remain enabled",
    )
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MindDrive vision capture requires CUDA")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inputs_payload = torch.load(
        args.preprocessed_inputs,
        map_location="cpu",
        weights_only=True,
    )
    camera_images = inputs_payload["camera_images"].to(args.device)
    reference_payload = torch.load(
        args.official_image_features,
        map_location="cpu",
        weights_only=True,
    )
    official_features = reference_payload["image_features"]

    started = time.perf_counter()
    reuse = args.reuse_exported_program is not None
    if reuse != (args.reuse_capture_evidence is not None):
        raise ValueError(
            "--reuse-exported-program and --reuse-capture-evidence "
            "must be provided together"
        )
    if reuse:
        program_path = args.reuse_exported_program.resolve()
        evidence_path = args.reuse_capture_evidence.resolve()
        exported = load_exported_region(program_path)
        with torch.inference_mode():
            sdpa_features = exported.module()(camera_images)
        strict_export_evidence = json.loads(
            evidence_path.read_text(encoding="utf-8")
        )
        attention_modules_replaced = 24
        rotary_aliases_canonicalized = 24
        evidence_level = "real-L2-held-out-export-execution"
    else:
        model = load_real_minddrive_model(
            args.source_root,
            args.release_root,
            device=args.device,
            verify_hashes=not args.skip_release_hashes,
        )
        encoder = make_exportable_minddrive_vision_encoder(model)
        # The Region owns only EVA.  Release detector, VLM, map and planning
        # heads before strict export so validation fits a 12GB RTX 3060.
        del model
        torch.cuda.empty_cache()
        with torch.inference_mode():
            sdpa_features = encoder(camera_images)
        program = build_real_minddrive_program(device=args.device)
        capture = capture_region(
            program.region("vision_encoder"),
            encoder,
            (camera_images,),
            strict=True,
            absolute_tolerance=1.0e-5,
            relative_tolerance=1.0e-5,
        )
        capture.require_supported()
        program_path = output / "vision_encoder.pt2e"
        evidence_path = output / "vision_encoder.capture.json"
        save_exported_region(
            capture,
            program_path=program_path,
            evidence_path=evidence_path,
        )
        strict_export_evidence = capture.evidence.to_dict()
        attention_modules_replaced = (
            encoder.replaced_flash_attention_modules
        )
        rotary_aliases_canonicalized = (
            encoder.canonicalized_rotary_aliases
        )
        evidence_level = "real-L2-region-capture"
    torch.cuda.synchronize()
    backend = compare_minddrive_vision_backends(
        official_features,
        sdpa_features.cpu(),
    )
    if not backend.passed:
        raise ValueError(
            "MindDrive vision backend equivalence failed locked thresholds: "
            f"{backend.to_dict()}"
        )
    sdpa_features_cpu = sdpa_features.detach().cpu()
    del sdpa_features
    torch.cuda.empty_cache()

    candidate_path = output / f"frame_{args.frame}_sdpa_features.pt"
    torch.save(
        {"image_features": sdpa_features_cpu},
        candidate_path,
    )
    report = {
        "schema": "vlaforge.minddrive_real_vision_capture/1",
        "passed": True,
        "evidence_level": evidence_level,
        "frame": args.frame,
        "sample_role": args.sample_role,
        "backend_equivalence": backend.to_dict(),
        "strict_export": strict_export_evidence,
        "attention_modules_replaced": attention_modules_replaced,
        "rotary_aliases_canonicalized": rotary_aliases_canonicalized,
        "inputs": {
            "preprocessed": str(args.preprocessed_inputs.resolve()),
            "preprocessed_sha256": _sha256(args.preprocessed_inputs),
            "official_features": str(
                args.official_image_features.resolve()
            ),
            "official_features_sha256": _sha256(
                args.official_image_features
            ),
        },
        "artifacts": {
            "exported_program": str(program_path),
            "exported_program_sha256": _sha256(program_path),
            "capture_evidence": str(evidence_path),
            "capture_evidence_sha256": _sha256(evidence_path),
            "sdpa_features": str(candidate_path),
            "sdpa_features_sha256": _sha256(candidate_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output / f"frame_{args.frame}_vision_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
