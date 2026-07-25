"""Real-checkpoint DiffusionDrive AOTInductor artifact parity audit.

This remains Adapter-owned: VLAForge core sees only five flat TensorRegions,
while this module knows how the pinned driving model partitions condition
encoding, explicit-noise initialization, bounded denoising, and named outputs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

from vlaforge.adapters.diffusiondrive_real import (
    DIFFUSIONDRIVE_CHECKPOINT_SHA256,
    DIFFUSIONDRIVE_HF_REVISION,
    DIFFUSIONDRIVE_UPSTREAM_REVISION,
    RealDiffusionDriveConfig,
    load_real_diffusiondrive_regions,
)


DIFFUSIONDRIVE_ARTIFACT_EVIDENCE_SCHEMA = (
    "vlaforge.diffusiondrive_artifact_evidence/1"
)
DIFFUSIONDRIVE_REGIONS = (
    "condition_encoder",
    "initialize_planner_state",
    "make_denoise_timestep",
    "denoise_planner_step",
    "decode_planner_outputs",
)
_OUTPUT_NAMES = (
    "candidate_trajectories",
    "candidate_scores",
    "trajectory",
    "bev_semantic_map",
    "agent_states",
    "agent_labels",
)
_CONDITION_NAMES = (
    "ego_query",
    "agents_query",
    "cross_bev",
    "status_encoding",
    "bev_semantic_map",
    "agent_states",
    "agent_labels",
)


def audit_real_diffusiondrive_artifacts(
    config: RealDiffusionDriveConfig,
    *,
    export_dir: str | Path,
    artifact_dir: str | Path,
    frontend_report: str | Path,
    region_nrmse_tolerance: float = 1e-3,
    trajectory_max_abs_tolerance: float = 2e-3,
    trajectory_mean_abs_tolerance: float = 5e-4,
) -> dict[str, object]:
    """Run eager, exported, and AOTI five-Region pipelines and compare them."""

    import torch
    import torch._inductor.codecache  # noqa: F401

    for value, name in (
        (region_nrmse_tolerance, "Region NRMSE tolerance"),
        (trajectory_max_abs_tolerance, "trajectory max-abs tolerance"),
        (trajectory_mean_abs_tolerance, "trajectory mean-abs tolerance"),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if not torch.cuda.is_available() and config.device.startswith("cuda"):
        raise RuntimeError("DiffusionDrive artifact audit requires CUDA")
    major, minor = torch.cuda.get_device_capability(config.device)
    target = f"sm_{major}{minor}"
    if target != "sm_86":
        raise RuntimeError(
            f"DiffusionDrive evidence requires sm_86, current target is {target}"
        )

    exports = Path(export_dir)
    artifacts = Path(artifact_dir)
    frontend_path = Path(frontend_report)
    frontend = _load_frontend_report(frontend_path, config)
    export_records, artifact_records = _verify_artifact_chain(
        exports,
        artifacts,
        frontend,
        target=target,
    )

    torch.cuda.reset_peak_memory_stats(config.device)
    regions = load_real_diffusiondrive_regions(config)
    inputs = regions.example_inputs
    eager_callables = {
        name: getattr(regions, name) for name in DIFFUSIONDRIVE_REGIONS
    }
    exported_callables = {
        name: _load_exported(torch, exports / f"{name}.pt2e")
        for name in DIFFUSIONDRIVE_REGIONS
    }
    artifact_callables = {
        name: torch._inductor.aoti_load_package(
            str(artifacts / f"{name}.pt2")
        )
        for name in DIFFUSIONDRIVE_REGIONS
    }

    eager, eager_seconds = _run_pipeline(
        torch,
        eager_callables,
        inputs,
        device=config.device,
    )
    exported, exported_seconds = _run_pipeline(
        torch,
        exported_callables,
        inputs,
        device=config.device,
    )
    artifact, artifact_seconds = _run_pipeline(
        torch,
        artifact_callables,
        inputs,
        device=config.device,
    )
    repeated, repeated_seconds = _run_pipeline(
        torch,
        artifact_callables,
        inputs,
        device=config.device,
    )

    exported_condition = _metric_map(
        _CONDITION_NAMES,
        eager["condition"],
        exported["condition"],
    )
    artifact_condition = _metric_map(
        _CONDITION_NAMES,
        eager["condition"],
        artifact["condition"],
    )
    exported_steps = tuple(
        _metrics(expected, actual)
        for expected, actual in zip(
            eager["planner_steps"],
            exported["planner_steps"],
            strict=True,
        )
    )
    artifact_steps = tuple(
        _metrics(expected, actual)
        for expected, actual in zip(
            eager["planner_steps"],
            artifact["planner_steps"],
            strict=True,
        )
    )
    exported_outputs = {
        name: _metrics(eager["outputs"][name], exported["outputs"][name])
        for name in _OUTPUT_NAMES
    }
    artifact_outputs = {
        name: _metrics(eager["outputs"][name], artifact["outputs"][name])
        for name in _OUTPUT_NAMES
    }
    repeatability = {
        name: _metrics(
            artifact["outputs"][name],
            repeated["outputs"][name],
        )
        for name in _OUTPUT_NAMES
    }
    exported_init = _metrics(
        eager["planner_initial"],
        exported["planner_initial"],
    )
    artifact_init = _metrics(
        eager["planner_initial"],
        artifact["planner_initial"],
    )
    exported_timesteps = tuple(
        _metrics(expected, actual)
        for expected, actual in zip(
            eager["timesteps"],
            exported["timesteps"],
            strict=True,
        )
    )
    artifact_timesteps = tuple(
        _metrics(expected, actual)
        for expected, actual in zip(
            eager["timesteps"],
            artifact["timesteps"],
            strict=True,
        )
    )

    exported_exact = all(
        item["exact"]
        for item in (
            *exported_condition.values(),
            exported_init,
            *exported_timesteps,
            *exported_steps,
            *exported_outputs.values(),
        )
    )
    artifact_region_metrics = (
        *artifact_condition.values(),
        artifact_init,
        *artifact_timesteps,
        *artifact_steps,
        *artifact_outputs.values(),
    )
    artifact_regions_passed = all(
        item["normalized_root_mean_square_error"]
        <= region_nrmse_tolerance
        for item in artifact_region_metrics
    )
    trajectory_metrics = artifact_outputs["trajectory"]
    trajectory_passed = (
        trajectory_metrics["maximum_absolute_error"]
        <= trajectory_max_abs_tolerance
        and trajectory_metrics["mean_absolute_error"]
        <= trajectory_mean_abs_tolerance
    )
    repeatability_passed = all(
        item["exact"] for item in repeatability.values()
    )
    passed = (
        exported_exact
        and artifact_regions_passed
        and trajectory_passed
        and repeatability_passed
    )

    return {
        "schema": DIFFUSIONDRIVE_ARTIFACT_EVIDENCE_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "evidence_kind": "real_checkpoint_compiled_artifact",
        "evidence_level": "L3" if passed else "L3-candidate",
        "model": "DiffusionDrive NAVSIM 88.1 PDMS",
        "upstream_revision": config.upstream_revision,
        "checkpoint": {
            "revision": config.checkpoint_revision,
            "path": str(config.checkpoint.resolve()),
            "sha256": regions.checkpoint_sha256,
            "expected_sha256": DIFFUSIONDRIVE_CHECKPOINT_SHA256,
        },
        "frontend_report": {
            "path": str(frontend_path.resolve()),
            "sha256": _sha256(frontend_path),
            "repository": frontend["repository"],
        },
        "target": target,
        "regions": list(DIFFUSIONDRIVE_REGIONS),
        "exported_programs": export_records,
        "artifacts": artifact_records,
        "correctness": {
            "exported_vs_eager": {
                "condition": exported_condition,
                "planner_initial": exported_init,
                "timesteps": list(exported_timesteps),
                "planner_steps": list(exported_steps),
                "outputs": exported_outputs,
                "all_exact": exported_exact,
            },
            "artifact_vs_eager": {
                "condition": artifact_condition,
                "planner_initial": artifact_init,
                "timesteps": list(artifact_timesteps),
                "planner_steps": list(artifact_steps),
                "outputs": artifact_outputs,
                "all_regions_within_nrmse": artifact_regions_passed,
                "trajectory_within_absolute_tolerance": trajectory_passed,
            },
            "artifact_repeatability": {
                "outputs": repeatability,
                "all_exact": repeatability_passed,
            },
        },
        "timing": {
            "eager_seconds": eager_seconds,
            "exported_seconds": exported_seconds,
            "artifact_seconds": artifact_seconds,
            "artifact_repeat_seconds": repeated_seconds,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(config.device)
            ),
            "note": "single-run audit metadata, not paper benchmark",
        },
        "tolerances": {
            "region_nrmse": region_nrmse_tolerance,
            "trajectory_max_abs": trajectory_max_abs_tolerance,
            "trajectory_mean_abs": trajectory_mean_abs_tolerance,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(config.device),
            "device": config.device,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("diffusers", "numpy", "timm", "torchvision")
            },
        },
    }


def _run_pipeline(
    torch: Any,
    callables: dict[str, Callable[..., Any]],
    inputs: dict[str, Any],
    *,
    device: str,
) -> tuple[dict[str, object], float]:
    started = time.perf_counter()
    with torch.inference_mode():
        condition = _as_tuple(
            callables["condition_encoder"](
                inputs["camera_feature"],
                inputs["lidar_feature"],
                inputs["status_feature"],
            )
        )
        planner_initial = _single(
            callables["initialize_planner_state"](inputs["noise"])
        )
        planner_state = planner_initial
        timesteps = []
        planner_steps = []
        for step in range(2):
            timestep = _single(
                callables["make_denoise_timestep"](
                    torch.tensor(step, dtype=torch.int64)
                )
            )
            timesteps.append(timestep)
            planner_state = _single(
                callables["denoise_planner_step"](
                    planner_state,
                    timestep,
                    condition[0],
                    condition[1],
                    condition[2],
                    condition[3],
                )
            )
            planner_steps.append(planner_state)
        candidates, scores, trajectory = _as_tuple(
            callables["decode_planner_outputs"](planner_state)
        )
    _synchronize(torch, device)
    elapsed = time.perf_counter() - started
    return (
        {
            "condition": condition,
            "planner_initial": planner_initial,
            "timesteps": tuple(timesteps),
            "planner_steps": tuple(planner_steps),
            "outputs": {
                "candidate_trajectories": candidates,
                "candidate_scores": scores,
                "trajectory": trajectory,
                "bev_semantic_map": condition[4],
                "agent_states": condition[5],
                "agent_labels": condition[6],
            },
        },
        elapsed,
    )


def _load_frontend_report(
    path: Path,
    config: RealDiffusionDriveConfig,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "vlaforge.diffusiondrive_real_frontend/1"
        or value.get("status") != "passed"
        or value.get("evidence_level") != "L2"
    ):
        raise ValueError("frontend report is not passing DiffusionDrive L2")
    if value.get("upstream", {}).get("revision") != (
        DIFFUSIONDRIVE_UPSTREAM_REVISION
    ):
        raise ValueError("frontend upstream revision mismatch")
    checkpoint = value.get("checkpoint", {})
    if (
        checkpoint.get("revision") != DIFFUSIONDRIVE_HF_REVISION
        or checkpoint.get("sha256") != DIFFUSIONDRIVE_CHECKPOINT_SHA256
    ):
        raise ValueError("frontend checkpoint identity mismatch")
    if config.upstream_revision != DIFFUSIONDRIVE_UPSTREAM_REVISION:
        raise ValueError("requested upstream revision is not pinned")
    if config.checkpoint_revision != DIFFUSIONDRIVE_HF_REVISION:
        raise ValueError("requested checkpoint revision is not pinned")
    return value


def _verify_artifact_chain(
    exports: Path,
    artifacts: Path,
    frontend: dict[str, Any],
    *,
    target: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    captures = {
        str(item["name"]): item for item in frontend.get("captures", ())
    }
    export_records = []
    artifact_records = []
    for name in DIFFUSIONDRIVE_REGIONS:
        export_path = exports / f"{name}.pt2e"
        artifact_path = artifacts / f"{name}.pt2"
        manifest_path = artifacts / f"{name}.compile.json"
        capture_path = exports / f"{name}.capture.json"
        for path in (
            export_path,
            artifact_path,
            manifest_path,
            capture_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        export_sha256 = _sha256(export_path)
        artifact_sha256 = _sha256(artifact_path)
        frontend_capture = captures.get(name)
        if frontend_capture is None:
            raise ValueError(f"{name}: frontend capture record missing")
        if (
            manifest.get("schema")
            != "vlaforge.compile_artifact_result/1"
            or manifest.get("status") != "passed"
            or manifest.get("backend") != "aoti"
            or manifest.get("target") != target
        ):
            raise ValueError(f"{name}: invalid compile manifest")
        if (
            manifest["exported_program"]["sha256"] != export_sha256
            or frontend_capture.get("program_sha256") != export_sha256
            or manifest["artifact"]["sha256"] != artifact_sha256
            or int(manifest["artifact"]["size_bytes"])
            != artifact_path.stat().st_size
        ):
            raise ValueError(f"{name}: artifact chain digest mismatch")
        if (
            capture.get("region_name") != name
            or not capture.get("effect_audit", {}).get("passed", False)
            or capture.get("graph_digest")
            != frontend_capture.get("graph_sha256")
        ):
            raise ValueError(f"{name}: capture contract mismatch")
        export_records.append(
            {
                "region": name,
                "path": str(export_path.resolve()),
                "sha256": export_sha256,
                "size_bytes": export_path.stat().st_size,
                "graph_sha256": capture["graph_digest"],
                "graph_nodes": int(manifest["graph_nodes"]),
            }
        )
        artifact_records.append(
            {
                "region": name,
                "path": str(artifact_path.resolve()),
                "sha256": artifact_sha256,
                "size_bytes": artifact_path.stat().st_size,
                "compile_seconds": float(manifest["compile_seconds"]),
                "target": str(manifest["target"]),
            }
        )
    return export_records, artifact_records


def _load_exported(torch: Any, path: Path) -> Any:
    with path.open("rb") as handle:
        return torch.export.load(handle).module()


def _metric_map(
    names: tuple[str, ...],
    expected: tuple[Any, ...],
    actual: tuple[Any, ...],
) -> dict[str, dict[str, object]]:
    if len(names) != len(expected) or len(expected) != len(actual):
        raise ValueError("metric map arity mismatch")
    return {
        name: _metrics(left, right)
        for name, left, right in zip(
            names,
            expected,
            actual,
            strict=True,
        )
    }


def _metrics(expected: Any, actual: Any) -> dict[str, object]:
    import torch

    if tuple(expected.shape) != tuple(actual.shape):
        raise ValueError(
            "numerical comparison shape mismatch: "
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
        nrmse = rmse / max(reference_rms, 1e-12)
    return {
        "shape": [int(item) for item in expected.shape],
        "dtype": str(expected.dtype).removeprefix("torch."),
        "maximum_absolute_error": maximum,
        "mean_absolute_error": mean,
        "normalized_root_mean_square_error": nrmse,
        "exact": exact,
    }


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _single(value: Any) -> Any:
    values = _as_tuple(value)
    if len(values) != 1:
        raise ValueError(f"expected one Region output, got {len(values)}")
    return values[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
