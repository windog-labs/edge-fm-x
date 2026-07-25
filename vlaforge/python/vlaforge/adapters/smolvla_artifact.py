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
from vlaforge.frontend.builder import ModuleBuilder
from vlaforge.ir import ops
from vlaforge.ir.program import (
    Block,
    InputPort,
    Invocation,
    OutputPort,
    StateSlot,
    TensorRegion,
    Value,
)
from vlaforge.ir.types import PendingOutputType, ScalarType, TensorType


SMOLVLA_ARTIFACT_EVIDENCE_SCHEMA = "vlaforge.smolvla_artifact_evidence/1"
_REGIONS = ("prepare_prefix", "solver_step", "trim_action_chunk")


def build_compiled_smolvla_action_program(
    *,
    chunk_size: int = 50,
    max_action_dim: int = 32,
    action_dim: int = 6,
    token_length: int = 48,
    prefix_length: int = 113,
    cache_layers: int = 16,
    cache_heads: int = 5,
    cache_head_dim: int = 64,
    num_steps: int = 10,
    device: str = "cuda:0",
) -> Any:
    """Build the flat-Tensor real SmolVLA L4 deployment program."""

    dimensions = (
        chunk_size,
        max_action_dim,
        action_dim,
        token_length,
        prefix_length,
        cache_layers,
        cache_heads,
        cache_head_dim,
        num_steps,
    )
    if min(dimensions) < 1:
        raise ValueError("compiled SmolVLA dimensions must be positive")
    image = TensorType((1, 3, 256, 256), "f32")
    robot_state = TensorType((1, action_dim), "f32")
    language = TensorType((1, token_length), "i64")
    language_mask = TensorType((1, token_length), "bool")
    sample = TensorType((1, chunk_size, max_action_dim), "f32")
    action_chunk = TensorType((1, chunk_size, action_dim), "f32")
    action = TensorType((1, action_dim), "f32")
    pad_mask = TensorType((1, prefix_length), "bool")
    cache = TensorType(
        (1, prefix_length, cache_heads, cache_head_dim), "bf16"
    )
    timestep = TensorType((1,), "f32")
    cursor = ScalarType("i32")
    index = ScalarType("index")
    boolean = ScalarType("bool")

    builder = ModuleBuilder("smolvla_real_cuda_l4")
    for name, payload in (
        ("image", image),
        ("state", robot_state),
        ("instruction_tokens", language),
        ("instruction_mask", language_mask),
        ("noise", sample),
    ):
        builder.add_input(
            InputPort(name, payload, device=device, alignment=64)
        )
    builder.add_output(
        OutputPort(
            "action",
            action,
            group="manipulation",
            device=device,
            alignment=64,
        )
    )
    builder.add_state(StateSlot("action_queue", action_chunk, retention=2))
    builder.add_state(StateSlot("queue_cursor", cursor, retention=2))

    prefix_outputs = (pad_mask,) + (cache,) * (cache_layers * 2)
    builder.add_region(
        TensorRegion(
            "prepare_prefix",
            (
                Value("image_arg", image),
                Value("state_arg", robot_state),
                Value("tokens_arg", language),
                Value("mask_arg", language_mask),
            ),
            prefix_outputs,
            metadata={
                "memoize": True,
                "cache_input_ports": [
                    "image",
                    "state",
                    "instruction_tokens",
                    "instruction_mask",
                ],
                "cache_state_slots": [],
                "loop_invariant": True,
            },
        )
    )
    builder.add_region(
        TensorRegion(
            "make_timestep",
            (Value("step_arg", index),),
            (timestep,),
        )
    )
    builder.add_region(
        TensorRegion(
            "solver_step",
            (
                Value("pad_mask_arg", pad_mask),
                Value("sample_arg", sample),
                Value("timestep_arg", timestep),
                *tuple(
                    Value(f"cache_arg_{item}", cache)
                    for item in range(cache_layers * 2)
                ),
            ),
            (sample,),
        )
    )
    builder.add_region(
        TensorRegion(
            "trim_action_chunk",
            (Value("sample_arg", sample),),
            (action_chunk,),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_is_empty",
            (Value("cursor_arg", cursor),),
            (boolean,),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_select",
            (
                Value("queue_arg", action_chunk),
                Value("cursor_arg", cursor),
            ),
            (action,),
        )
    )
    builder.add_region(
        TensorRegion(
            "queue_advance",
            (Value("cursor_arg", cursor),),
            (cursor,),
        )
    )
    builder.add_region(TensorRegion("queue_zero", (), (cursor,)))

    prefix_names = ("prefix_pad_mask",) + tuple(
        f"prefix_cache_{item}" for item in range(cache_layers * 2)
    )
    loop = Block.of(
        (
            ops.invoke(
                ("timestep_value",),
                (timestep,),
                "make_timestep",
                ("solver_index",),
            ),
            ops.invoke(
                ("sample_next",),
                (sample,),
                "solver_step",
                (
                    prefix_names[0],
                    "sample_iter",
                    "timestep_value",
                    *prefix_names[1:],
                ),
            ),
            ops.yield_values("sample_next"),
        )
    )
    refill = Block.of(
        (
            ops.invoke(
                prefix_names,
                prefix_outputs,
                "prepare_prefix",
                (
                    "image_value",
                    "state_value",
                    "instruction_value",
                    "instruction_mask_value",
                ),
            ),
            ops.for_loop(
                Value("sample_final", sample),
                "noise_value",
                Value("solver_index", index),
                Value("sample_iter", sample),
                loop,
                lower=0,
                upper=num_steps,
            ),
            ops.invoke(
                ("refilled_queue",),
                (action_chunk,),
                "trim_action_chunk",
                ("sample_final",),
            ),
            ops.invoke(
                ("refilled_action",),
                (action,),
                "queue_select",
                ("refilled_queue", "zero_cursor"),
            ),
            ops.invoke(
                ("cursor_after_refill",),
                (cursor,),
                "queue_advance",
                ("zero_cursor",),
            ),
            ops.yield_values(
                "refilled_action",
                "refilled_queue",
                "cursor_after_refill",
            ),
        )
    )
    reuse = Block.of(
        (
            ops.invoke(
                ("queued_action",),
                (action,),
                "queue_select",
                ("queue_value", "cursor_value"),
            ),
            ops.invoke(
                ("cursor_after_reuse",),
                (cursor,),
                "queue_advance",
                ("cursor_value",),
            ),
            ops.yield_values(
                "queued_action", "queue_value", "cursor_after_reuse"
            ),
        )
    )
    body = Block.of(
        (
            ops.input_read("image_value", "image_revision", image, "image"),
            ops.input_read(
                "state_value", "state_revision", robot_state, "state"
            ),
            ops.input_read(
                "instruction_value",
                "instruction_revision",
                language,
                "instruction_tokens",
            ),
            ops.input_read(
                "instruction_mask_value",
                "instruction_mask_revision",
                language_mask,
                "instruction_mask",
            ),
            ops.input_read(
                "noise_value", "noise_revision", sample, "noise"
            ),
            ops.transaction_begin("txn"),
            ops.state_read_latest(
                "queue_snapshot",
                action_chunk,
                "action_queue",
                "txn",
            ),
            ops.snapshot_value(
                "queue_value", action_chunk, "queue_snapshot"
            ),
            ops.state_read_latest(
                "cursor_snapshot", cursor, "queue_cursor", "txn"
            ),
            ops.snapshot_value(
                "cursor_value", cursor, "cursor_snapshot"
            ),
            ops.invoke(
                ("queue_empty",),
                (boolean,),
                "queue_is_empty",
                ("cursor_value",),
            ),
            ops.invoke(("zero_cursor",), (cursor,), "queue_zero", ()),
            ops.if_op(
                (
                    Value("selected_action", action),
                    Value("queue_next", action_chunk),
                    Value("cursor_next", cursor),
                ),
                "queue_empty",
                refill,
                reuse,
            ),
            ops.stage_write(
                "queue_pending",
                action_chunk,
                "action_queue",
                "txn",
                "queue_next",
            ),
            ops.stage_write(
                "cursor_pending",
                cursor,
                "queue_cursor",
                "txn",
                "cursor_next",
            ),
            ops.validate(
                "action_valid", "selected_action", "finite_action"
            ),
            ops.output_create(
                "pending_action",
                "selected_action",
                action,
                "action",
            ),
            ops.output_group(
                "pending_outputs",
                "manipulation",
                (
                    (
                        "pending_action",
                        PendingOutputType("action", action),
                    ),
                ),
            ),
            ops.transaction_commit(
                "committed_outputs",
                (PendingOutputType("action", action),),
                "manipulation",
                "txn",
                "pending_outputs",
                "action_valid",
            ),
            ops.return_values("committed_outputs"),
        )
    )
    builder.add_invocation(
        Invocation(
            "act",
            body,
            metadata={
                "template": "ChunkedAction",
                "persistent_state": "action_queue,queue_cursor",
                "derived_cache": "flattened_prefix_kv",
                "solver": "bounded_flow_matching",
            },
        )
    )
    return builder.build()


def capture_smolvla_support_regions(
    export_dir: str | Path,
    *,
    device: str = "cuda:0",
    chunk_size: int = 50,
    action_dim: int = 6,
    num_steps: int = 10,
) -> tuple[Any, ...]:
    """Capture the small Adapter/control TensorRegions required by L4."""

    import torch
    from vlaforge.frontend import capture_region, save_exported_region

    if min(chunk_size, action_dim, num_steps) < 1:
        raise ValueError("support Region dimensions must be positive")
    output = Path(export_dir)
    output.mkdir(parents=True, exist_ok=True)
    cursor_type = ScalarType("i32")
    index_type = ScalarType("index")
    queue_type = TensorType((1, chunk_size, action_dim), "f32")
    action_type = TensorType((1, action_dim), "f32")
    timestep_type = TensorType((1,), "f32")

    class MakeTimestep(torch.nn.Module):
        def forward(self, step: Any) -> Any:
            return (
                1.0 - step.to(dtype=torch.float32) / num_steps
            ).reshape(1).to(device=device)

    class QueueIsEmpty(torch.nn.Module):
        def forward(self, cursor: Any) -> Any:
            return cursor >= chunk_size

    class QueueSelect(torch.nn.Module):
        def forward(self, queue: Any, cursor: Any) -> Any:
            indices = (
                cursor.to(torch.int64)
                .reshape(1, 1, 1)
                .expand(queue.shape[0], 1, queue.shape[2])
            )
            return torch.gather(queue, 1, indices).squeeze(1)

    class QueueAdvance(torch.nn.Module):
        def forward(self, cursor: Any) -> Any:
            return cursor + 1

    class QueueZero(torch.nn.Module):
        def forward(self) -> Any:
            return torch.zeros((), device=device, dtype=torch.int32)

    step = torch.tensor(2, dtype=torch.int64)
    cursor = torch.tensor(3, device=device, dtype=torch.int32)
    queue = torch.arange(
        chunk_size * action_dim,
        device=device,
        dtype=torch.float32,
    ).reshape(1, chunk_size, action_dim)
    declarations = (
        (
            TensorRegion(
                "make_timestep",
                (Value("step", index_type),),
                (timestep_type,),
            ),
            MakeTimestep(),
            (step,),
        ),
        (
            TensorRegion(
                "queue_is_empty",
                (Value("cursor", cursor_type),),
                (ScalarType("bool"),),
            ),
            QueueIsEmpty(),
            (cursor,),
        ),
        (
            TensorRegion(
                "queue_select",
                (
                    Value("queue", queue_type),
                    Value("cursor", cursor_type),
                ),
                (action_type,),
            ),
            QueueSelect(),
            (queue, cursor),
        ),
        (
            TensorRegion(
                "queue_advance",
                (Value("cursor", cursor_type),),
                (cursor_type,),
            ),
            QueueAdvance(),
            (cursor,),
        ),
        (
            TensorRegion("queue_zero", (), (cursor_type,)),
            QueueZero(),
            (),
        ),
    )
    captures = []
    for region, implementation, arguments in declarations:
        capture = capture_region(
            region,
            implementation,
            arguments,
            strict=True,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )
        capture.require_supported()
        save_exported_region(
            capture,
            program_path=output / f"{region.name}.pt2e",
            evidence_path=output / f"{region.name}.capture.json",
        )
        captures.append(capture)
    return tuple(captures)


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
