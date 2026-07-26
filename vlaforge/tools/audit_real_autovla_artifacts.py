#!/usr/bin/env python3
"""Audit the partitioned real AutoVLA frontend against CUDA AOTI artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.autovla_real import (  # noqa: E402
    AUTOVLA_DECODE_STEPS,
    AUTOVLA_HIDDEN_SIZE,
)
from vlaforge.frontend import load_exported_region  # noqa: E402


REPORT_SCHEMA = "vlaforge.autovla_real_artifact/1"
REGIONS = (
    "autovla_decoder_mlp",
    "autovla_action_projection",
    "autovla_trajectory_decode",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _metrics(expected: Any, actual: Any) -> dict[str, object]:
    import torch

    if tuple(expected.shape) != tuple(actual.shape):
        raise ValueError(
            "AutoVLA artifact shape mismatch: "
            f"{tuple(expected.shape)} != {tuple(actual.shape)}"
        )
    exact = bool(expected.dtype == actual.dtype and expected.equal(actual))
    if not (expected.is_floating_point() or actual.is_floating_point()):
        maximum = 0.0 if exact else 1.0
        mean = maximum
        nrmse = maximum
    else:
        left = expected.detach().to(device="cpu", dtype=torch.float64)
        right = actual.detach().to(device="cpu", dtype=torch.float64)
        difference = (left - right).abs()
        maximum = float(difference.max().item()) if difference.numel() else 0.0
        mean = float(difference.mean().item()) if difference.numel() else 0.0
        rmse = (
            math.sqrt(float(difference.square().mean().item()))
            if difference.numel()
            else 0.0
        )
        reference_rms = (
            math.sqrt(float(left.square().mean().item()))
            if left.numel()
            else 0.0
        )
        nrmse = rmse / reference_rms if reference_rms else rmse
    return {
        "shape": [int(item) for item in expected.shape],
        "dtype": str(expected.dtype),
        "exact": exact,
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
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
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "revision": revision,
        "source_dirty": bool(status),
        "tracked_status": status.splitlines() if status else [],
    }


def _validate_chain(
    *,
    export_dir: Path,
    artifact_dir: Path,
    frontend_path: Path,
    compile_path: Path,
) -> tuple[dict[str, Any], list[dict[str, object]], list[dict[str, object]]]:
    frontend = _json(frontend_path)
    if (
        frontend.get("schema") != "vlaforge.autovla_real_frontend/1"
        or frontend.get("status") != "passed"
        or not str(frontend.get("evidence_level", "")).startswith("L2")
    ):
        raise ValueError("AutoVLA frontend report is not passing real L2")
    if int(frontend.get("semantic_ir", {}).get("core_op_delta", -1)) != 0:
        raise ValueError("AutoVLA frontend changed the frozen core op set")

    compile_report = _json(compile_path)
    if compile_report.get("schema") != "vlaforge.real_aoti_compile/1":
        raise ValueError("invalid AutoVLA AOTI compile report")
    captures = {
        str(item["name"]): item for item in frontend.get("captures", ())
    }
    compiled = {
        str(item["region"]): item
        for item in compile_report.get("regions", ())
    }
    if set(captures) != set(REGIONS) or set(compiled) != set(REGIONS):
        raise ValueError("AutoVLA Region set changed across frontend/compile")

    exports = []
    artifacts = []
    for name in REGIONS:
        export = export_dir / f"{name}.pt2e"
        artifact = artifact_dir / f"{name}.pt2"
        for path in (export, artifact):
            if not path.is_file():
                raise FileNotFoundError(path)
        export_sha = _sha256(export)
        artifact_sha = _sha256(artifact)
        capture = captures[name]
        compile_record = compiled[name]
        if (
            capture.get("program_sha256") != export_sha
            or compile_record.get("export_sha256") != export_sha
            or compile_record.get("package_sha256") != artifact_sha
            or int(compile_record.get("package_size_bytes", -1))
            != artifact.stat().st_size
        ):
            raise ValueError(f"{name}: AutoVLA artifact digest chain mismatch")
        exports.append(
            {
                "region": name,
                "path": str(export.resolve()),
                "sha256": export_sha,
                "size_bytes": export.stat().st_size,
                "graph_sha256": capture["graph_sha256"],
            }
        )
        artifacts.append(
            {
                "region": name,
                "path": str(artifact.resolve()),
                "sha256": artifact_sha,
                "size_bytes": artifact.stat().st_size,
                "compile_seconds": float(compile_record["compile_seconds"]),
            }
        )
    return frontend, exports, artifacts


def _run_pipeline(
    torch: Any,
    callables: Mapping[str, Any],
    hidden: Any,
    *,
    device: str,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    with torch.inference_mode():
        decoded = _as_tuple(callables[REGIONS[0]](hidden))[0]
        logits = _as_tuple(callables[REGIONS[1]](decoded))[0]
        trajectory, tokens = _as_tuple(callables[REGIONS[2]](logits))
    torch.cuda.synchronize(device)
    return (
        {
            "decoded_hidden": decoded,
            "action_logits": logits,
            "trajectory": trajectory,
            "action_tokens": tokens,
        },
        time.perf_counter() - started,
    )


def audit(
    *,
    export_dir: Path,
    artifact_dir: Path,
    frontend_path: Path,
    compile_path: Path,
    device: str,
    region_nrmse_tolerance: float,
    trajectory_max_abs_tolerance: float,
) -> dict[str, Any]:
    import torch
    import torch._inductor.codecache  # noqa: F401

    if not torch.cuda.is_available():
        raise RuntimeError("AutoVLA real artifact audit requires CUDA")
    major, minor = torch.cuda.get_device_capability(device)
    target = f"sm_{major}{minor}"
    if target != "sm_86":
        raise RuntimeError(
            f"AutoVLA artifact evidence requires sm_86, got {target}"
        )
    frontend, export_records, artifact_records = _validate_chain(
        export_dir=export_dir,
        artifact_dir=artifact_dir,
        frontend_path=frontend_path,
        compile_path=compile_path,
    )
    exported = {
        name: load_exported_region(export_dir / f"{name}.pt2e").module()
        for name in REGIONS
    }
    artifacts = {
        name: torch._inductor.aoti_load_package(
            str(artifact_dir / f"{name}.pt2")
        )
        for name in REGIONS
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(20260726)
    hidden = torch.randn(
        (1, AUTOVLA_DECODE_STEPS, AUTOVLA_HIDDEN_SIZE),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.cuda.reset_peak_memory_stats(device)
    exported_outputs, exported_seconds = _run_pipeline(
        torch, exported, hidden, device=device
    )
    artifact_outputs, artifact_seconds = _run_pipeline(
        torch, artifacts, hidden, device=device
    )
    repeated_outputs, repeated_seconds = _run_pipeline(
        torch, artifacts, hidden, device=device
    )
    artifact_metrics = {
        name: _metrics(exported_outputs[name], artifact_outputs[name])
        for name in exported_outputs
    }
    repeat_metrics = {
        name: _metrics(artifact_outputs[name], repeated_outputs[name])
        for name in artifact_outputs
    }
    region_passed = all(
        artifact_metrics[name]["normalized_root_mean_square_error"]
        <= region_nrmse_tolerance
        for name in ("decoded_hidden", "action_logits")
    )
    trajectory_passed = (
        artifact_metrics["trajectory"]["maximum_absolute_error"]
        <= trajectory_max_abs_tolerance
    )
    tokens_passed = artifact_metrics["action_tokens"]["exact"]
    repeat_passed = all(item["exact"] for item in repeat_metrics.values())
    passed = bool(
        region_passed and trajectory_passed and tokens_passed and repeat_passed
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "model": frontend.get("model"),
        "evidence_kind": "real-checkpoint-compiled-artifact-partition",
        "evidence_level": (
            "L3-partitioned-real-checkpoint-artifact"
            if passed
            else "L3-candidate-partitioned-real-checkpoint-artifact"
        ),
        "scope": frontend["scope"],
        "target": target,
        "frontend_report": {
            "path": str(frontend_path.resolve()),
            "sha256": _sha256(frontend_path),
            "evidence_level": frontend["evidence_level"],
            "repository": frontend["repository"],
        },
        "compile_report": {
            "path": str(compile_path.resolve()),
            "sha256": _sha256(compile_path),
        },
        "regions": list(REGIONS),
        "exported_programs": export_records,
        "artifacts": artifact_records,
        "semantic_ir": {
            "core_op_delta": 0,
            "source": frontend["semantic_ir"],
        },
        "correctness": {
            "artifact_vs_exported": artifact_metrics,
            "artifact_repeatability": repeat_metrics,
            "regions_within_nrmse": region_passed,
            "trajectory_within_max_abs": trajectory_passed,
            "action_tokens_exact": tokens_passed,
            "repeatability_exact": repeat_passed,
        },
        "tolerances": {
            "region_nrmse": region_nrmse_tolerance,
            "trajectory_max_abs": trajectory_max_abs_tolerance,
            "action_tokens": "exact",
            "repeatability": "exact",
        },
        "timing_and_memory": {
            "exported_seconds": exported_seconds,
            "artifact_seconds": artifact_seconds,
            "artifact_repeat_seconds": repeated_seconds,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "note": "single-run correctness audit; not a paper benchmark",
        },
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "device": device,
        },
        "repository": _repository_state(),
        "claim_boundary": {
            "partitioned_real_checkpoint": True,
            "full_end_to_end_autovla": False,
            "generated_cpp_session": False,
            "host_cuda_sm86_only": True,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--frontend-report", type=Path, required=True)
    parser.add_argument("--compile-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--region-nrmse-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--trajectory-max-abs-tolerance",
        type=float,
        default=2e-3,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit(
        export_dir=args.export_dir.resolve(),
        artifact_dir=args.artifact_dir.resolve(),
        frontend_path=args.frontend_report.resolve(),
        compile_path=args.compile_report.resolve(),
        device=args.device,
        region_nrmse_tolerance=args.region_nrmse_tolerance,
        trajectory_max_abs_tolerance=args.trajectory_max_abs_tolerance,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
