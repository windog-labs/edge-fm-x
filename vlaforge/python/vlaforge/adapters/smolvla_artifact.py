"""Real-checkpoint SmolVLA AOTInductor artifact parity audit.

This module is deliberately Adapter-owned.  It knows the upstream SmolVLA
region partition and deterministic audit fixture, while the VLAForge core only
sees flat TensorRegion values and compiled artifact contracts.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vlaforge.adapters.smolvla_real import RealSmolVLAConfig


SMOLVLA_ARTIFACT_EVIDENCE_SCHEMA = "vlaforge.smolvla_artifact_evidence/1"
_REGIONS = ("prepare_prefix", "solver_step", "trim_action_chunk")


@dataclass(frozen=True, slots=True)
class NumericalMetrics:
    shape: tuple[int, ...]
    dtype: str
    maximum_absolute_error: float
    mean_absolute_error: float
    normalized_root_mean_square_error: float
    exact: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SmolVLAArtifactEvidence:
    schema: str
    evidence_kind: str
    evidence_level: str
    checkpoint_path: str
    checkpoint_sha256: str
    upstream_revision: str
    torch_version: str
    transformers_version: str
    cuda_version: str | None
    device: str
    gpu_name: str | None
    num_steps: int
    frontend_report_sha256: str
    exported_programs: tuple[dict[str, object], ...]
    artifacts: tuple[dict[str, object], ...]
    prefix_metrics: tuple[NumericalMetrics, ...]
    solver_step_metrics: tuple[NumericalMetrics, ...]
    exported_final_vs_eager: NumericalMetrics
    artifact_final_vs_exported: NumericalMetrics
    artifact_final_vs_eager: NumericalMetrics
    artifact_repeatability: NumericalMetrics
    eager_seconds: float
    exported_seconds: float
    artifact_seconds: float
    peak_cuda_memory_mb: float | None
    tolerances: dict[str, float]
    passed: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return result

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def audit_real_smolvla_artifacts(
    config: RealSmolVLAConfig,
    *,
    export_dir: str | Path,
    artifact_dir: str | Path,
    frontend_report: str | Path,
    report_path: str | Path | None = None,
    final_max_abs_tolerance: float = 5e-2,
    final_mean_abs_tolerance: float = 1e-2,
    region_nrmse_tolerance: float = 2e-2,
) -> SmolVLAArtifactEvidence:
    """Compare eager, exported, and packaged AOTI SmolVLA regions.

    The audit executes the real checkpoint once through upstream eager code,
    then executes the exact captured prefix -> bounded solver -> trim partition
    through both ``torch.export`` and packaged AOTInductor artifacts.
    """

    import torch
    import torch._inductor.codecache  # noqa: F401
    import transformers
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    if config.num_steps < 1:
        raise ValueError("num_steps must be positive")
    for value, name in (
        (final_max_abs_tolerance, "final max-abs tolerance"),
        (final_mean_abs_tolerance, "final mean-abs tolerance"),
        (region_nrmse_tolerance, "Region NRMSE tolerance"),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    exports = Path(export_dir)
    artifacts = Path(artifact_dir)
    frontend_path = Path(frontend_report)
    checkpoint = config.policy_path / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.vlm_path.is_dir():
        raise FileNotFoundError(config.vlm_path)
    frontend = json.loads(frontend_path.read_text(encoding="utf-8"))
    if frontend.get("model") != "SmolVLA" or not frontend.get("passed"):
        raise ValueError("frontend report is not a passing SmolVLA capture")
    if frontend.get("checkpoint_revision") != config.lerobot_revision:
        raise ValueError("frontend and requested upstream revisions differ")
    checkpoint_sha256 = _sha256(checkpoint)
    frontend_digests = {
        str(item["path"]): str(item["sha256"])
        for item in frontend.get("checkpoint_digests", ())
    }
    if frontend_digests.get(checkpoint.name) != checkpoint_sha256:
        raise ValueError("frontend report checkpoint digest mismatch")

    export_records, artifact_records = _verify_artifact_chain(
        exports, artifacts
    )
    cuda_device = (
        torch.device(config.device)
        if config.device.startswith("cuda")
        else None
    )
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats()

    policy_config = PreTrainedConfig.from_pretrained(
        config.policy_path, local_files_only=True
    )
    policy_config.vlm_model_name = str(config.vlm_path)
    policy_config.device = config.device
    policy_config.num_steps = config.num_steps
    policy = SmolVLAPolicy.from_pretrained(
        config.policy_path,
        config=policy_config,
        local_files_only=True,
        strict=False,
    ).eval()
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    tokens = tokenizer(
        "pick up the block\n",
        padding="max_length",
        max_length=policy_config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    image_key = next(iter(policy_config.image_features))
    image = torch.linspace(
        0,
        1,
        3 * 256 * 256,
        device=config.device,
        dtype=torch.float32,
    ).reshape(1, 3, 256, 256)
    state = torch.linspace(
        -0.2,
        0.3,
        6,
        device=config.device,
        dtype=torch.float32,
    ).reshape(1, 6)
    language_tokens = tokens["input_ids"].to(config.device)
    language_masks = tokens["attention_mask"].to(
        config.device, dtype=torch.bool
    )
    batch = {
        image_key: image,
        OBS_STATE: state,
        OBS_LANGUAGE_TOKENS: language_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: language_masks,
    }
    noise = torch.linspace(
        -1,
        1,
        policy_config.chunk_size * policy_config.max_action_dim,
        device=config.device,
        dtype=torch.float32,
    ).reshape(1, policy_config.chunk_size, policy_config.max_action_dim)

    started = time.perf_counter()
    with torch.inference_mode():
        eager_action = policy.predict_action_chunk(batch, noise=noise)
    _synchronize(torch, config.device)
    eager_seconds = time.perf_counter() - started
    eager_action_cpu = eager_action.detach().cpu()
    del eager_action, policy
    gc.collect()
    if cuda_device is not None:
        torch.cuda.empty_cache()

    prefix_export = _load_exported(torch, exports / "prepare_prefix.pt2e")
    prefix_artifact = torch._inductor.aoti_load_package(
        str(artifacts / "prepare_prefix.pt2")
    )
    started = time.perf_counter()
    with torch.inference_mode():
        exported_prefix = _as_tuple(
            prefix_export(
                image,
                state,
                language_tokens,
                language_masks,
            )
        )
    _synchronize(torch, config.device)
    exported_prefix_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        artifact_prefix = _as_tuple(
            prefix_artifact(
                image,
                state,
                language_tokens,
                language_masks,
            )
        )
    _synchronize(torch, config.device)
    artifact_prefix_seconds = time.perf_counter() - started
    if len(exported_prefix) != len(artifact_prefix):
        raise RuntimeError("prefix artifact output arity mismatch")
    prefix_metrics = tuple(
        _metrics(expected, actual)
        for expected, actual in zip(
            exported_prefix, artifact_prefix, strict=True
        )
    )

    solver_export = _load_exported(torch, exports / "solver_step.pt2e")
    solver_artifact = torch._inductor.aoti_load_package(
        str(artifacts / "solver_step.pt2")
    )
    exported_sample = noise.clone()
    artifact_sample = noise.clone()
    solver_metrics: list[NumericalMetrics] = []
    exported_solver_seconds = 0.0
    artifact_solver_seconds = 0.0
    with torch.inference_mode():
        for step in range(config.num_steps):
            timestep = torch.tensor(
                1.0 - step / config.num_steps,
                dtype=torch.float32,
                device=config.device,
            ).expand(exported_sample.shape[0])
            started = time.perf_counter()
            exported_sample = _as_tuple(
                solver_export(
                    exported_prefix[0],
                    exported_sample,
                    timestep,
                    *exported_prefix[1:],
                )
            )[0]
            _synchronize(torch, config.device)
            exported_solver_seconds += time.perf_counter() - started
            started = time.perf_counter()
            artifact_sample = _as_tuple(
                solver_artifact(
                    artifact_prefix[0],
                    artifact_sample,
                    timestep,
                    *artifact_prefix[1:],
                )
            )[0]
            _synchronize(torch, config.device)
            artifact_solver_seconds += time.perf_counter() - started
            solver_metrics.append(
                _metrics(exported_sample, artifact_sample)
            )

    trim_export = _load_exported(
        torch, exports / "trim_action_chunk.pt2e"
    )
    trim_artifact = torch._inductor.aoti_load_package(
        str(artifacts / "trim_action_chunk.pt2")
    )
    started = time.perf_counter()
    with torch.inference_mode():
        exported_action = _as_tuple(trim_export(exported_sample))[0]
    _synchronize(torch, config.device)
    exported_trim_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with torch.inference_mode():
        artifact_action = _as_tuple(trim_artifact(artifact_sample))[0]
    _synchronize(torch, config.device)
    artifact_trim_seconds = time.perf_counter() - started

    with torch.inference_mode():
        repeated_prefix = _as_tuple(
            prefix_artifact(
                image,
                state,
                language_tokens,
                language_masks,
            )
        )
        repeated_sample = noise.clone()
        for step in range(config.num_steps):
            timestep = torch.tensor(
                1.0 - step / config.num_steps,
                dtype=torch.float32,
                device=config.device,
            ).expand(repeated_sample.shape[0])
            repeated_sample = _as_tuple(
                solver_artifact(
                    repeated_prefix[0],
                    repeated_sample,
                    timestep,
                    *repeated_prefix[1:],
                )
            )[0]
        repeated_action = _as_tuple(trim_artifact(repeated_sample))[0]
    _synchronize(torch, config.device)

    exported_vs_eager = _metrics(
        eager_action_cpu, exported_action.detach().cpu()
    )
    artifact_vs_exported = _metrics(exported_action, artifact_action)
    artifact_vs_eager = _metrics(
        eager_action_cpu, artifact_action.detach().cpu()
    )
    repeatability = _metrics(artifact_action, repeated_action)
    prefix_passed = all(
        item.normalized_root_mean_square_error
        <= region_nrmse_tolerance
        for item in prefix_metrics
    )
    solver_passed = all(
        item.normalized_root_mean_square_error
        <= region_nrmse_tolerance
        for item in solver_metrics
    )
    final_passed = (
        artifact_vs_eager.maximum_absolute_error
        <= final_max_abs_tolerance
        and artifact_vs_eager.mean_absolute_error
        <= final_mean_abs_tolerance
    )
    passed = (
        prefix_passed
        and solver_passed
        and final_passed
        and repeatability.exact
    )
    peak_memory = (
        float(torch.cuda.max_memory_allocated() / 2**20)
        if cuda_device is not None
        else None
    )
    result = SmolVLAArtifactEvidence(
        schema=SMOLVLA_ARTIFACT_EVIDENCE_SCHEMA,
        evidence_kind="real_checkpoint_compiled_artifact",
        evidence_level="L3" if passed else "L3-candidate",
        checkpoint_path=str(checkpoint.resolve()),
        checkpoint_sha256=checkpoint_sha256,
        upstream_revision=config.lerobot_revision,
        torch_version=torch.__version__,
        transformers_version=transformers.__version__,
        cuda_version=torch.version.cuda,
        device=config.device,
        gpu_name=(
            torch.cuda.get_device_name()
            if cuda_device is not None
            else None
        ),
        num_steps=config.num_steps,
        frontend_report_sha256=_sha256(frontend_path),
        exported_programs=export_records,
        artifacts=artifact_records,
        prefix_metrics=prefix_metrics,
        solver_step_metrics=tuple(solver_metrics),
        exported_final_vs_eager=exported_vs_eager,
        artifact_final_vs_exported=artifact_vs_exported,
        artifact_final_vs_eager=artifact_vs_eager,
        artifact_repeatability=repeatability,
        eager_seconds=eager_seconds,
        exported_seconds=(
            exported_prefix_seconds
            + exported_solver_seconds
            + exported_trim_seconds
        ),
        artifact_seconds=(
            artifact_prefix_seconds
            + artifact_solver_seconds
            + artifact_trim_seconds
        ),
        peak_cuda_memory_mb=peak_memory,
        tolerances={
            "final_max_abs": final_max_abs_tolerance,
            "final_mean_abs": final_mean_abs_tolerance,
            "region_nrmse": region_nrmse_tolerance,
        },
        passed=passed,
    )
    if report_path is not None:
        result.write(report_path)
    return result


def _verify_artifact_chain(
    exports: Path, artifacts: Path
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    export_records: list[dict[str, object]] = []
    artifact_records: list[dict[str, object]] = []
    for region in _REGIONS:
        exported = exports / f"{region}.pt2e"
        artifact = artifacts / f"{region}.pt2"
        manifest_path = artifacts / f"{region}.compile.json"
        if not exported.is_file():
            raise FileNotFoundError(exported)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "vlaforge.compile_artifact_result/1"
            or manifest.get("status") != "passed"
            or manifest.get("target") != "sm_86"
        ):
            raise ValueError(f"{region}: invalid compile manifest")
        export_sha256 = _sha256(exported)
        artifact_sha256 = _sha256(artifact)
        if (
            manifest["exported_program"]["sha256"] != export_sha256
            or manifest["artifact"]["sha256"] != artifact_sha256
            or int(manifest["artifact"]["size_bytes"])
            != artifact.stat().st_size
        ):
            raise ValueError(f"{region}: compile manifest digest mismatch")
        export_records.append(
            {
                "region": region,
                "path": exported.name,
                "sha256": export_sha256,
                "size_bytes": exported.stat().st_size,
                "graph_nodes": int(manifest["graph_nodes"]),
            }
        )
        artifact_records.append(
            {
                "region": region,
                "path": artifact.name,
                "sha256": artifact_sha256,
                "size_bytes": artifact.stat().st_size,
                "compile_seconds": float(manifest["compile_seconds"]),
                "target": str(manifest["target"]),
            }
        )
    return tuple(export_records), tuple(artifact_records)


def _load_exported(torch: Any, path: Path) -> Any:
    with path.open("rb") as handle:
        return torch.export.load(handle).module()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _metrics(expected: Any, actual: Any) -> NumericalMetrics:
    import torch

    if tuple(expected.shape) != tuple(actual.shape):
        raise ValueError(
            f"numerical comparison shape mismatch: "
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
            float(torch_sqrt_mean_square(difference))
            if difference.numel()
            else 0.0
        )
        reference_rms = (
            float(torch_sqrt_mean_square(left.abs()))
            if left.numel()
            else 0.0
        )
        nrmse = rmse / max(reference_rms, 1e-12)
    return NumericalMetrics(
        shape=tuple(int(item) for item in expected.shape),
        dtype=str(expected.dtype).removeprefix("torch."),
        maximum_absolute_error=maximum,
        mean_absolute_error=mean,
        normalized_root_mean_square_error=nrmse,
        exact=exact,
    )


def torch_sqrt_mean_square(value: Any) -> float:
    return math.sqrt(float(value.square().mean().item()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
