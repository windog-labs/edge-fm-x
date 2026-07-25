#!/usr/bin/env python3
"""Capture and audit the pinned real DiffusionDrive checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.diffusiondrive_real import (  # noqa: E402
    DIFFUSIONDRIVE_CHECKPOINT_SIZE,
    DIFFUSIONDRIVE_HF_REVISION,
    DIFFUSIONDRIVE_UPSTREAM_REVISION,
    RealDiffusionDriveConfig,
    build_real_diffusiondrive_program,
    capture_real_diffusiondrive_regions,
    load_real_diffusiondrive_regions,
    run_real_diffusiondrive_chain,
    run_upstream_diffusiondrive_with_explicit_noise,
)
from vlaforge.analysis import verify  # noqa: E402
from vlaforge.compiler import compile_module  # noqa: E402
from vlaforge.ir.serializer import io_schema_digest  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(actual: Any, expected: Any) -> dict[str, object]:
    import torch

    actual_cpu = actual.detach().to(torch.float64).cpu()
    expected_cpu = expected.detach().to(torch.float64).cpu()
    difference = (actual_cpu - expected_cpu).abs()
    denominator = torch.linalg.vector_norm(expected_cpu)
    numerator = torch.linalg.vector_norm(actual_cpu - expected_cpu)
    nrmse = float(
        numerator / denominator
        if denominator != 0
        else numerator
    )
    return {
        "shape": list(actual.shape),
        "exact": bool(torch.equal(actual_cpu, expected_cpu)),
        "maximum_absolute_error": float(difference.max().item()),
        "mean_absolute_error": float(difference.mean().item()),
        "normalized_root_mean_square_error": nrmse,
    }


def _repository_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=_REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "source_dirty": dirty}


def _git_object_sha256(
    repository: Path, revision: str, relative_path: Path
) -> str:
    payload = subprocess.run(
        ["git", "show", f"{revision}:{relative_path.as_posix()}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--upstream-revision",
        default=DIFFUSIONDRIVE_UPSTREAM_REVISION,
    )
    parser.add_argument(
        "--checkpoint-revision",
        default=DIFFUSIONDRIVE_HF_REVISION,
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--relative-tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    import torch

    module = build_real_diffusiondrive_program(device=args.device)
    diagnostics = verify(module, raise_on_error=False)
    if diagnostics:
        raise ValueError(f"DiffusionDrive Semantic IR invalid: {diagnostics}")
    compilation = compile_module(
        module,
        default_device=args.device,
        state_device=args.device,
    )

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    regions = load_real_diffusiondrive_regions(
        RealDiffusionDriveConfig(
            source_root=args.source_root,
            checkpoint=args.checkpoint,
            device=args.device,
            upstream_revision=args.upstream_revision,
            checkpoint_revision=args.checkpoint_revision,
        )
    )
    load_seconds = time.perf_counter() - load_started
    eager_started = time.perf_counter()
    explicit = run_real_diffusiondrive_chain(regions)
    upstream = run_upstream_diffusiondrive_with_explicit_noise(regions)
    torch.cuda.synchronize()
    eager_seconds = time.perf_counter() - eager_started
    peak_memory = int(torch.cuda.max_memory_allocated())

    parity = {
        name: _metrics(explicit[name], upstream[name])
        for name in (
            "candidate_trajectories",
            "candidate_scores",
            "trajectory",
            "bev_semantic_map",
            "agent_states",
            "agent_labels",
        )
    }
    for name, record in parity.items():
        if (
            record["maximum_absolute_error"] > args.absolute_tolerance
            and record["normalized_root_mean_square_error"]
            > args.relative_tolerance
        ):
            raise ValueError(
                f"DiffusionDrive upstream parity failed for {name}: {record}"
            )

    capture_started = time.perf_counter()
    captures = capture_real_diffusiondrive_regions(
        regions,
        args.export_dir,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    capture_seconds = time.perf_counter() - capture_started
    capture_records = []
    for capture in captures:
        assert capture.evidence is not None
        program = args.export_dir / f"{capture.region.name}.pt2e"
        evidence = args.export_dir / f"{capture.region.name}.capture.json"
        capture_records.append(
            {
                "name": capture.region.name,
                "program": str(program.resolve()),
                "program_sha256": _sha256(program),
                "program_size_bytes": program.stat().st_size,
                "evidence": str(evidence.resolve()),
                "graph_sha256": capture.evidence.graph_digest,
                "export_seconds": capture.evidence.export_seconds,
                "maximum_export_error": (
                    capture.evidence.maximum_absolute_error
                ),
                "effect_audit": capture.evidence.effect_audit.to_dict(),
            }
        )

    source_relative_files = (
        Path("navsim/agents/diffusiondrive/transfuser_model_v2.py"),
        Path("navsim/agents/diffusiondrive/transfuser_backbone.py"),
        Path("navsim/agents/diffusiondrive/modules/blocks.py"),
        Path(
            "navsim/agents/diffusiondrive/modules/conditional_unet1d.py"
        ),
    )
    source_files = []
    for relative_path in source_relative_files:
        path = args.source_root / relative_path
        worktree_sha256 = _sha256(path)
        pinned_sha256 = _git_object_sha256(
            args.source_root,
            args.upstream_revision,
            relative_path,
        )
        if worktree_sha256 != pinned_sha256:
            raise ValueError(
                "DiffusionDrive source file differs from pinned revision: "
                f"{relative_path}"
            )
        source_files.append(
            {
                "path": str(path.resolve()),
                "relative_path": relative_path.as_posix(),
                "sha256": worktree_sha256,
                "pinned_git_object_sha256": pinned_sha256,
                "matches_pinned_revision": True,
            }
        )
    report = {
        "schema": "vlaforge.diffusiondrive_real_frontend/1",
        "status": "passed",
        "passed": True,
        "evidence_kind": "real-checkpoint-frontend-capture",
        "evidence_level": "L2",
        "model": "DiffusionDrive NAVSIM 88.1 PDMS",
        "upstream": {
            "repository": "https://github.com/hustvl/DiffusionDrive",
            "revision": args.upstream_revision,
            "source_files": source_files,
            "code_license": "MIT",
        },
        "checkpoint": {
            "repository": "hustvl/DiffusionDrive",
            "revision": args.checkpoint_revision,
            "path": str(args.checkpoint.resolve()),
            "sha256": regions.checkpoint_sha256,
            "size_bytes": args.checkpoint.stat().st_size,
            "expected_size_bytes": DIFFUSIONDRIVE_CHECKPOINT_SIZE,
            "license": "non-commercial per Hugging Face model card",
            "missing_keys": list(regions.missing_keys),
            "unexpected_keys": list(regions.unexpected_keys),
        },
        "input_contract": {
            "camera_feature": [1, 3, 256, 1024],
            "lidar_feature": [1, 1, 256, 256],
            "status_feature": [1, 8],
            "noise": [1, 20, 8, 2],
            "note": (
                "Sensor synchronization and feature construction remain "
                "external; noise is explicit for deterministic deployment."
            ),
        },
        "output_contract": {
            name: list(value.shape)
            for name, value in explicit.items()
            if name != "planner_state"
        },
        "semantic_ir": {
            "io_schema_digest": io_schema_digest(module),
            "regions": [region.name for region in module.regions],
            "core_op_delta": 0,
            "cache_input_ids": list(
                compilation.certificate.caches[0].input_ids
            ),
            "cache_state_ids": list(
                compilation.certificate.caches[0].state_ids
            ),
            "bounded_denoise_steps": 2,
            "loop_carried_state_shape": [1, 820],
        },
        "correctness": {
            "upstream_forward_vs_region_chain": parity,
            "candidate_trajectories_finite": bool(
                torch.isfinite(
                    explicit["candidate_trajectories"]
                ).all()
            ),
            "candidate_scores_finite": bool(
                torch.isfinite(explicit["candidate_scores"]).all()
            ),
        },
        "captures": capture_records,
        "timing": {
            "checkpoint_load_seconds": load_seconds,
            "eager_chain_seconds": eager_seconds,
            "capture_total_seconds": capture_seconds,
            "peak_cuda_allocated_bytes": peak_memory,
        },
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "device": args.device,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "diffusers",
                    "numpy",
                    "timm",
                    "torchvision",
                )
            },
        },
        "repository": _repository_state(),
        "reproduction": {
            "command": [
                sys.executable,
                str(Path(__file__).relative_to(_REPOSITORY_ROOT)),
                "--source-root",
                str(args.source_root.resolve()),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--export-dir",
                str(args.export_dir.resolve()),
                "--report",
                "<report.json>",
                "--device",
                args.device,
            ],
            "environment": {
                "PYTHONPATH": "vlaforge/python",
                "CUDA_VISIBLE_DEVICES": os.getenv(
                    "CUDA_VISIBLE_DEVICES", "<unset>"
                ),
            },
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
