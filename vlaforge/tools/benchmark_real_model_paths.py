#!/usr/bin/env python3
"""Benchmark real eager and direct-AOTI model paths on host CUDA.

Each process owns exactly one model and one execution path.  Setup, input
upload, output probes, allocator inspection, and report generation are outside
the timed interval.  The measured interval contains one full action-chunk or
planning invocation plus ``torch.cuda.synchronize()``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SOURCE_ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


_SCHEMA = "vlaforge.real_model_path_benchmark/1"
_MODELS = ("smolvla", "diffusiondrive", "minddrive")
_PATHS = ("eager", "direct_artifact")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_kib() -> int:
    for line in Path("/proc/self/status").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        raise ValueError("latency samples must be non-empty")
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _latency_summary(values: list[int]) -> dict[str, float | int]:
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ns": mean,
        "p50_ns": _nearest_rank(values, 0.50),
        "p90_ns": _nearest_rank(values, 0.90),
        "p99_ns": _nearest_rank(values, 0.99),
        "minimum_ns": min(values),
        "maximum_ns": max(values),
        "throughput_runs_per_second": 1e9 / mean,
    }


def _read_tensor(
    torch: Any,
    root: Path,
    name: str,
    dtype: Any,
    shape: tuple[int, ...],
) -> Any:
    path = root / f"{name}.bin"
    payload = bytearray(path.read_bytes())
    tensor = torch.frombuffer(payload, dtype=dtype).clone().reshape(shape)
    return tensor.to("cuda:0")


def _as_outputs(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _minddrive_meta_action_replay(
    torch: Any,
    candidate: Any,
    reference: Any,
    *,
    allowed_token_ids: set[int],
) -> dict[str, object] | None:
    """Validate and describe replay of the frontend's sole stochastic token."""

    if torch.equal(candidate, reference):
        return None
    mismatch = torch.nonzero(candidate != reference).reshape(-1)
    valid = (
        tuple(candidate.shape) == tuple(reference.shape)
        and mismatch.tolist() == [candidate.numel() - 1]
        and int(candidate[-1]) in allowed_token_ids
        and int(reference[-1]) in allowed_token_ids
    )
    if not valid:
        raise ValueError(
            "upstream decision prompt differs from the captured input "
            "outside its stochastic meta-action token"
        )
    return {
        "sampled_token_id": int(candidate[-1]),
        "replayed_token_id": int(reference[-1]),
        "token_offset": int(mismatch[0]),
        "allowed_token_ids": sorted(allowed_token_ids),
    }


def _normalize_minddrive_upstream_state(
    model: Any,
    state_types: tuple[tuple[str, Any], ...],
    upstream_keys: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    """Project upstream over-retained tensors onto the fixed deployment state."""

    normalized = {}
    heads = {
        "detection": model.pts_bbox_head,
        "map": model.map_head,
    }
    for name, payload in state_types:
        prefix, attribute = upstream_keys[name].split(".", maxsplit=1)
        value = getattr(heads[prefix], attribute)
        expected = tuple(payload.shape)
        actual = tuple(value.shape)
        if len(actual) != len(expected):
            raise RuntimeError(
                f"{name}: upstream state rank changed: {actual} != {expected}"
            )
        if len(expected) == 1:
            if actual != expected:
                raise RuntimeError(
                    f"{name}: upstream scalar state changed: "
                    f"{actual} != {expected}"
                )
            projected = value
        else:
            if (
                actual[0] != expected[0]
                or actual[1] < expected[1]
                or actual[2:] != expected[2:]
            ):
                raise RuntimeError(
                    f"{name}: upstream state cannot project to fixed "
                    f"shape: {actual} -> {expected}"
                )
            projected = value[:, : expected[1]].contiguous()
            setattr(heads[prefix], attribute, projected)
        normalized[name] = {
            "observed_shape": list(actual),
            "committed_shape": list(projected.shape),
            "truncated": actual != tuple(projected.shape),
        }
    return normalized


def _clone_minddrive_invocation(value: Any) -> Any:
    """Clone mutable frontend views without copying bound tensor storage."""

    if isinstance(value, dict):
        return {
            key: _clone_minddrive_invocation(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_minddrive_invocation(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_minddrive_invocation(item) for item in value)
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(value):
        # MindDrive's image frontend calls squeeze_() on its argument. Give
        # upstream an independent TensorImpl while retaining the Session
        # contract's zero-copy borrowed storage.
        return value.as_strided(
            value.size(),
            value.stride(),
            value.storage_offset(),
        )
    return value


@dataclass(frozen=True, slots=True)
class _SequenceValue:
    role: str
    binding: int | None
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SequenceArtifact:
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _SequenceNode:
    artifact_id: int
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _SequenceManifest:
    region: str
    target: str
    device: str
    values: tuple[_SequenceValue, ...]
    artifacts: tuple[_SequenceArtifact, ...]
    nodes: tuple[_SequenceNode, ...]

    @property
    def input_count(self) -> int:
        return sum(value.role == "input" for value in self.values)

    @property
    def output_count(self) -> int:
        return sum(value.role == "output" for value in self.values)


def _parse_aoti_sequence(path: Path) -> _SequenceManifest:
    """Parse the bounded deployment sequence used by the C++ AOTI backend."""

    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0

    def fields(expected: str) -> list[str]:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError(f"{path}: expected {expected}, reached EOF")
        result = lines[cursor].split()
        cursor += 1
        if not result or result[0] != expected:
            raise ValueError(
                f"{path}:{cursor}: expected {expected}, got {result}"
            )
        return result

    header = fields("VLAFORGE_AOTI_SEQUENCE")
    if header != ["VLAFORGE_AOTI_SEQUENCE", "1"]:
        raise ValueError(f"{path}: unsupported AOTI sequence header")
    region_line = fields("region")
    target_line = fields("target")
    device_line = fields("device")
    if any(len(item) != 2 for item in (region_line, target_line, device_line)):
        raise ValueError(f"{path}: malformed sequence identity")

    value_count_line = fields("values")
    if len(value_count_line) != 2:
        raise ValueError(f"{path}: malformed value count")
    values = []
    input_bindings = []
    output_bindings = []
    for expected_id in range(int(value_count_line[1])):
        declaration = fields("value")
        if len(declaration) < 6 or int(declaration[1]) != expected_id:
            raise ValueError(f"{path}: non-dense sequence value ids")
        role = declaration[2]
        raw_binding = int(declaration[3])
        rank = int(declaration[5])
        shape = tuple(int(item) for item in declaration[6:])
        if (
            role not in {"input", "output", "temporary"}
            or len(shape) != rank
            or any(item < 0 for item in shape)
        ):
            raise ValueError(f"{path}: invalid value {expected_id}")
        if role == "temporary":
            if raw_binding != -1:
                raise ValueError(f"{path}: temporary has binding")
            binding = None
        else:
            if raw_binding < 0:
                raise ValueError(f"{path}: external value has no binding")
            binding = raw_binding
            (input_bindings if role == "input" else output_bindings).append(
                binding
            )
        values.append(
            _SequenceValue(role, binding, declaration[4], shape)
        )
    for name, bindings in (
        ("input", input_bindings),
        ("output", output_bindings),
    ):
        if sorted(bindings) != list(range(len(bindings))):
            raise ValueError(f"{path}: {name} bindings are not dense")

    artifact_count_line = fields("artifacts")
    if len(artifact_count_line) != 2:
        raise ValueError(f"{path}: malformed artifact count")
    artifacts = []
    for expected_id in range(int(artifact_count_line[1])):
        declaration = fields("artifact")
        if len(declaration) != 5 or int(declaration[1]) != expected_id:
            raise ValueError(f"{path}: non-dense artifact ids")
        artifacts.append(
            _SequenceArtifact(
                declaration[2],
                declaration[3],
                int(declaration[4]),
            )
        )

    node_count_line = fields("nodes")
    if len(node_count_line) != 2:
        raise ValueError(f"{path}: malformed node count")
    nodes = []
    defined = [value.role == "input" for value in values]
    uses = [0 for _ in values]
    for _ in range(int(node_count_line[1])):
        declaration = fields("node")
        if len(declaration) < 5:
            raise ValueError(f"{path}: malformed node")
        index = 1
        artifact_id = int(declaration[index])
        index += 1
        input_count = int(declaration[index])
        index += 1
        inputs = tuple(
            int(item)
            for item in declaration[index : index + input_count]
        )
        index += input_count
        if index >= len(declaration):
            raise ValueError(f"{path}: node has no output count")
        output_count = int(declaration[index])
        index += 1
        outputs = tuple(
            int(item)
            for item in declaration[index : index + output_count]
        )
        index += output_count
        if (
            index != len(declaration)
            or artifact_id < 0
            or artifact_id >= len(artifacts)
            or not inputs
            or not outputs
        ):
            raise ValueError(f"{path}: invalid node declaration")
        for value_id in inputs:
            if value_id < 0 or value_id >= len(values) or not defined[value_id]:
                raise ValueError(f"{path}: node reads undefined value")
            uses[value_id] += 1
        for value_id in outputs:
            if (
                value_id < 0
                or value_id >= len(values)
                or values[value_id].role == "input"
                or defined[value_id]
            ):
                raise ValueError(f"{path}: node redefines value")
            defined[value_id] = True
        nodes.append(_SequenceNode(artifact_id, inputs, outputs))
    if fields("end") != ["end"] or cursor != len(lines):
        raise ValueError(f"{path}: sequence has trailing content")
    for index, value in enumerate(values):
        if value.role == "output" and not defined[index]:
            raise ValueError(f"{path}: output is undefined")
        if value.role == "temporary" and (
            not defined[index] or uses[index] == 0
        ):
            raise ValueError(f"{path}: temporary is unused")
    if set(node.artifact_id for node in nodes) != set(range(len(artifacts))):
        raise ValueError(f"{path}: sequence contains unused artifact")
    return _SequenceManifest(
        region_line[1],
        target_line[1],
        device_line[1],
        tuple(values),
        tuple(artifacts),
        tuple(nodes),
    )


class _DirectAotiSequence:
    """Persistent Python control for the exact C++ sequence artifact set."""

    def __init__(self, torch: Any, manifest_path: Path) -> None:
        self._torch = torch
        self.path = manifest_path.resolve()
        self.manifest = _parse_aoti_sequence(self.path)
        if self.manifest.device != "cuda:0":
            raise ValueError(
                f"{self.path}: benchmark expects cuda:0 sequence"
            )
        self._artifact_paths = []
        self._runners = []
        for artifact in self.manifest.artifacts:
            artifact_path = (self.path.parent / artifact.relative_path).resolve()
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != artifact.size_bytes
                or _sha256(artifact_path) != artifact.sha256
            ):
                raise ValueError(
                    f"{self.path}: physical artifact verification failed: "
                    f"{artifact_path}"
                )
            self._artifact_paths.append(artifact_path)
            self._runners.append(
                torch._export.aot_load(str(artifact_path), "cuda:0")
            )
        self._uses = [0 for _ in self.manifest.values]
        for node in self.manifest.nodes:
            for value_id in node.inputs:
                self._uses[value_id] += 1

    def _matches(self, tensor: Any, value: _SequenceValue) -> bool:
        dtype = {
            "bool": self._torch.bool,
            "i32": self._torch.int32,
            "i64": self._torch.int64,
            "f16": self._torch.float16,
            "bf16": self._torch.bfloat16,
            "f32": self._torch.float32,
            "f64": self._torch.float64,
            "u8": self._torch.uint8,
        }.get(value.dtype)
        return bool(
            dtype is not None
            and tensor.is_cuda
            and tensor.device.index == 0
            and tensor.is_contiguous()
            and tensor.dtype == dtype
            and tuple(tensor.shape) == value.shape
        )

    def run(self, *inputs: Any) -> tuple[Any, ...]:
        if len(inputs) != self.manifest.input_count:
            raise ValueError(f"{self.path}: sequence input arity mismatch")
        values: list[Any | None] = [None for _ in self.manifest.values]
        for value_id, value in enumerate(self.manifest.values):
            if value.role != "input":
                continue
            assert value.binding is not None
            tensor = inputs[value.binding]
            if not self._matches(tensor, value):
                raise ValueError(
                    f"{self.path}: input {value.binding} metadata mismatch"
                )
            values[value_id] = tensor
        remaining = list(self._uses)
        with self._torch.inference_mode():
            for node_index, node in enumerate(self.manifest.nodes):
                arguments = tuple(values[index] for index in node.inputs)
                if any(item is None for item in arguments):
                    raise RuntimeError(
                        f"{self.path}: node {node_index} input is undefined"
                    )
                outputs = _as_outputs(
                    self._runners[node.artifact_id](*arguments)
                )
                if len(outputs) != len(node.outputs):
                    raise RuntimeError(
                        f"{self.path}: node {node_index} output arity changed"
                    )
                for output_index, (value_id, tensor) in enumerate(
                    zip(node.outputs, outputs, strict=True)
                ):
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                    if not self._matches(
                        tensor, self.manifest.values[value_id]
                    ):
                        raise RuntimeError(
                            f"{self.path}: node {node_index} output "
                            f"{output_index} metadata mismatch"
                        )
                    values[value_id] = tensor
                for value_id in node.inputs:
                    remaining[value_id] -= 1
                    if remaining[value_id] < 0:
                        raise RuntimeError(
                            f"{self.path}: value liveness underflow"
                        )
                    if (
                        remaining[value_id] == 0
                        and self.manifest.values[value_id].role
                        == "temporary"
                    ):
                        values[value_id] = None
        result: list[Any | None] = [None] * self.manifest.output_count
        for value_id, value in enumerate(self.manifest.values):
            if value.role == "output":
                assert value.binding is not None
                result[value.binding] = values[value_id]
        if any(item is None for item in result):
            raise RuntimeError(f"{self.path}: output is undefined")
        return tuple(result)

    def metadata(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": _sha256(self.path),
            "region": self.manifest.region,
            "target": self.manifest.target,
            "physical_artifact_count": len(self._artifact_paths),
            "physical_artifact_bytes": sum(
                path.stat().st_size for path in self._artifact_paths
            ),
            "node_count": len(self.manifest.nodes),
            "value_count": len(self.manifest.values),
        }

    def close(self) -> None:
        self._runners.clear()
        self._artifact_paths.clear()


def _load_smolvla_eager(args: argparse.Namespace) -> tuple[
    Callable[[], Any],
    Callable[[], None],
    dict[str, object],
]:
    import torch
    import transformers
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    if args.checkpoint is None or args.vlm_path is None:
        raise ValueError("SmolVLA eager requires --checkpoint and --vlm-path")
    policy_root = args.checkpoint.resolve()
    checkpoint = policy_root / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not args.vlm_path.is_dir():
        raise FileNotFoundError(args.vlm_path)
    config = PreTrainedConfig.from_pretrained(
        policy_root,
        local_files_only=True,
    )
    config.vlm_model_name = str(args.vlm_path.resolve())
    config.device = "cuda:0"
    config.num_steps = 10
    policy = SmolVLAPolicy.from_pretrained(
        policy_root,
        config=config,
        local_files_only=True,
        strict=False,
    ).eval()
    image_key = next(iter(config.image_features))
    batch = {
        image_key: _read_tensor(
            torch, args.input_root, "image", torch.float32, (1, 3, 256, 256)
        ),
        OBS_STATE: _read_tensor(
            torch, args.input_root, "state", torch.float32, (1, 6)
        ),
        OBS_LANGUAGE_TOKENS: _read_tensor(
            torch, args.input_root, "tokens", torch.int64, (1, 48)
        ),
        OBS_LANGUAGE_ATTENTION_MASK: _read_tensor(
            torch, args.input_root, "mask", torch.bool, (1, 48)
        ),
    }
    noise = _read_tensor(
        torch, args.input_root, "noise", torch.float32, (1, 50, 32)
    )

    def run() -> Any:
        with torch.inference_mode():
            return policy.predict_action_chunk(batch, noise=noise)

    metadata: dict[str, object] = {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "vlm_path": str(args.vlm_path.resolve()),
        "upstream_revision": args.upstream_revision,
        "transformers": transformers.__version__,
        "execution_contract": "eager explicit-noise full action chunk",
    }
    return run, lambda: None, metadata


def _load_smolvla_direct(args: argparse.Namespace) -> tuple[
    Callable[[], Any],
    Callable[[], None],
    dict[str, object],
]:
    import torch
    import torch._inductor.codecache  # noqa: F401

    if args.l3_root is None or args.support_root is None:
        raise ValueError(
            "SmolVLA direct artifact requires --l3-root and --support-root"
        )
    artifact_paths = {
        "prepare_prefix": (
            args.l3_root / "artifacts" / "prepare_prefix.pt2"
        ),
        "solver_step": args.l3_root / "artifacts" / "solver_step.pt2",
        "trim_action_chunk": (
            args.l3_root / "artifacts" / "trim_action_chunk.pt2"
        ),
        "make_timestep": (
            args.support_root / "artifacts" / "make_timestep.pt2"
        ),
    }
    for path in artifact_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    callables = {
        name: torch._inductor.aoti_load_package(str(path))
        for name, path in artifact_paths.items()
    }
    image = _read_tensor(
        torch, args.input_root, "image", torch.float32, (1, 3, 256, 256)
    )
    state = _read_tensor(
        torch, args.input_root, "state", torch.float32, (1, 6)
    )
    tokens = _read_tensor(
        torch, args.input_root, "tokens", torch.int64, (1, 48)
    )
    mask = _read_tensor(
        torch, args.input_root, "mask", torch.bool, (1, 48)
    )
    noise = _read_tensor(
        torch, args.input_root, "noise", torch.float32, (1, 50, 32)
    )
    steps = tuple(
        torch.tensor(step, dtype=torch.int64) for step in range(10)
    )

    def run() -> Any:
        with torch.inference_mode():
            prefix = _as_outputs(
                callables["prepare_prefix"](image, state, tokens, mask)
            )
            sample = noise
            for step in steps:
                timestep = callables["make_timestep"](step)
                sample = callables["solver_step"](
                    prefix[0],
                    sample,
                    timestep,
                    *prefix[1:],
                )
            return callables["trim_action_chunk"](sample)

    metadata = {
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
        "execution_contract": (
            "direct AOTI full prefix plus 10-step flow plus trim"
        ),
    }
    return run, lambda: None, metadata


def _load_diffusiondrive_eager(args: argparse.Namespace) -> tuple[
    Callable[[], Any],
    Callable[[], None],
    dict[str, object],
]:
    import torch
    from vlaforge.adapters.diffusiondrive_real import (
        RealDiffusionDriveConfig,
        load_real_diffusiondrive_regions,
    )

    if args.source_root is None or args.checkpoint is None:
        raise ValueError(
            "DiffusionDrive eager requires --source-root and --checkpoint"
        )
    regions = load_real_diffusiondrive_regions(
        RealDiffusionDriveConfig(
            source_root=args.source_root,
            checkpoint=args.checkpoint,
            device="cuda:0",
        )
    )
    inputs = {
        "camera_feature": _read_tensor(
            torch,
            args.input_root,
            "camera_feature",
            torch.float32,
            (1, 3, 256, 1024),
        ),
        "lidar_feature": _read_tensor(
            torch,
            args.input_root,
            "lidar_feature",
            torch.float32,
            (1, 1, 256, 256),
        ),
        "status_feature": _read_tensor(
            torch,
            args.input_root,
            "status_feature",
            torch.float32,
            (1, 8),
        ),
        "noise": _read_tensor(
            torch,
            args.input_root,
            "noise",
            torch.float32,
            (1, 20, 8, 2),
        ),
    }
    noise = inputs["noise"]
    original_randn = torch.randn

    def explicit_randn(*shape: object, **kwargs: object) -> Any:
        requested = (
            tuple(shape[0])
            if len(shape) == 1 and hasattr(shape[0], "__iter__")
            else tuple(int(item) for item in shape)
        )
        if requested == tuple(noise.shape):
            return noise.to(
                device=kwargs.get("device", noise.device),
                dtype=kwargs.get("dtype", noise.dtype),
            )
        return original_randn(*shape, **kwargs)

    torch.randn = explicit_randn

    def run() -> Any:
        with torch.inference_mode():
            outputs = regions.model(
                {
                    "camera_feature": inputs["camera_feature"],
                    "lidar_feature": inputs["lidar_feature"],
                    "status_feature": inputs["status_feature"],
                }
            )
        return outputs["trajectory"]

    def cleanup() -> None:
        torch.randn = original_randn

    checkpoint = args.checkpoint.resolve()
    metadata: dict[str, object] = {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": regions.checkpoint_sha256,
            "size_bytes": checkpoint.stat().st_size,
        },
        "source_root": str(args.source_root.resolve()),
        "execution_contract": (
            "pinned upstream eager forward with explicit deterministic noise"
        ),
    }
    return run, cleanup, metadata


def _load_diffusiondrive_direct(args: argparse.Namespace) -> tuple[
    Callable[[], Any],
    Callable[[], None],
    dict[str, object],
]:
    import torch
    import torch._inductor.codecache  # noqa: F401
    from vlaforge.adapters.diffusiondrive_artifact import (
        DIFFUSIONDRIVE_REGIONS,
    )

    if args.l3_root is None:
        raise ValueError("DiffusionDrive direct artifact requires --l3-root")
    artifact_paths = {
        name: args.l3_root / "artifacts" / f"{name}.pt2"
        for name in DIFFUSIONDRIVE_REGIONS
    }
    for path in artifact_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    callables = {
        name: torch._inductor.aoti_load_package(str(path))
        for name, path in artifact_paths.items()
    }
    camera = _read_tensor(
        torch,
        args.input_root,
        "camera_feature",
        torch.float32,
        (1, 3, 256, 1024),
    )
    lidar = _read_tensor(
        torch,
        args.input_root,
        "lidar_feature",
        torch.float32,
        (1, 1, 256, 256),
    )
    status = _read_tensor(
        torch, args.input_root, "status_feature", torch.float32, (1, 8)
    )
    noise = _read_tensor(
        torch, args.input_root, "noise", torch.float32, (1, 20, 8, 2)
    )
    steps = tuple(torch.tensor(step, dtype=torch.int64) for step in range(2))

    def run() -> Any:
        with torch.inference_mode():
            condition = _as_outputs(
                callables["condition_encoder"](camera, lidar, status)
            )
            planner_state = callables["initialize_planner_state"](noise)
            for step in steps:
                timestep = callables["make_denoise_timestep"](step)
                planner_state = callables["denoise_planner_step"](
                    planner_state,
                    timestep,
                    condition[0],
                    condition[1],
                    condition[2],
                    condition[3],
                )
            outputs = _as_outputs(
                callables["decode_planner_outputs"](planner_state)
            )
            return outputs[2]

    metadata = {
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in artifact_paths.items()
        },
        "execution_contract": (
            "direct AOTI condition plus initialize plus 2 denoise steps "
            "plus decode"
        ),
    }
    return run, lambda: None, metadata


_MINDDRIVE_FRAMES = (
    "frame_00400",
    "frame_00401",
    "frame_00402",
    "frame_00403",
    "frame_00404",
)


def _load_minddrive_inputs(
    torch: Any,
    *,
    bundle_root: Path,
    input_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    schema_path = bundle_root / "metadata" / "input_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("schema") != "vlaforge.input_schema/2":
        raise ValueError("MindDrive input schema version changed")
    declarations = schema.get("inputs")
    if not isinstance(declarations, list) or len(declarations) != 13:
        raise ValueError("MindDrive input schema coverage changed")
    dtype_map = {
        "bool": torch.bool,
        "f32": torch.float32,
        "f64": torch.float64,
        "i32": torch.int32,
        "i64": torch.int64,
        "u8": torch.uint8,
    }
    frames = []
    input_hashes: dict[str, dict[str, str]] = {}
    for frame_name in _MINDDRIVE_FRAMES:
        frame_root = input_root / frame_name
        if not frame_root.is_dir():
            raise FileNotFoundError(frame_root)
        frame: dict[str, Any] = {}
        hashes = {}
        for declaration in declarations:
            payload = declaration["payload"]
            if payload.get("kind") != "tensor":
                raise ValueError("MindDrive benchmark inputs must be tensors")
            name = str(declaration["name"])
            path = frame_root / f"{name}.bin"
            dtype_name = str(payload["dtype"])
            shape = tuple(int(item) for item in payload["shape"])
            if dtype_name not in dtype_map:
                raise ValueError(f"unsupported MindDrive dtype: {dtype_name}")
            raw = bytearray(path.read_bytes())
            tensor = torch.frombuffer(raw, dtype=dtype_map[dtype_name]).clone()
            expected_elements = math.prod(shape)
            if tensor.numel() != expected_elements:
                raise ValueError(
                    f"{path}: {tensor.numel()} elements != "
                    f"{expected_elements}"
                )
            frame[name] = tensor.reshape(shape).to("cuda:0")
            hashes[name] = _sha256(path)
        frames.append(frame)
        input_hashes[frame_name] = hashes
    return frames, {
        "schema_path": str(schema_path.resolve()),
        "schema_sha256": _sha256(schema_path),
        "io_schema_digest": schema["io_schema_digest"],
        "frames": list(_MINDDRIVE_FRAMES),
        "input_hashes": input_hashes,
    }


def _load_verified_minddrive_direct_artifacts(
    torch: Any, bundle_root: Path
) -> tuple[
    _DirectAotiSequence,
    _DirectAotiSequence,
    dict[str, Any],
    dict[str, object],
]:
    bundle_path = bundle_root / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema") != "vlaforge.compile_bundle/4":
        raise ValueError("MindDrive compile bundle version changed")
    records = {
        str(item["region_name"]): item
        for item in bundle["region_artifacts"]
    }
    expected = {
        "vision_encoder",
        "position_encoder",
        "map_encoder",
        "detection_encoder",
        "decision_expert",
        "action_expert",
        "trajectory_decoder",
        "detection_decoder",
    }
    if set(records) != expected:
        raise ValueError("MindDrive logical Region coverage changed")

    vision = _DirectAotiSequence(
        torch, bundle_root / records["vision_encoder"]["artifact_path"]
    )
    map_sequence = _DirectAotiSequence(
        torch, bundle_root / records["map_encoder"]["artifact_path"]
    )
    direct_names = expected - {"vision_encoder", "map_encoder"}
    runners = {}
    artifact_metadata = {}
    for name in sorted(direct_names):
        record = records[name]
        path = (bundle_root / record["artifact_path"]).resolve()
        expected_size = int(record["artifact_size_bytes"])
        expected_hash = str(record["artifact_sha256"])
        if (
            record.get("artifact_kind") != "shared_library"
            or not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_hash
        ):
            raise ValueError(
                f"MindDrive direct artifact verification failed: {name}"
            )
        runners[name] = torch._export.aot_load(str(path), "cuda:0")
        artifact_metadata[name] = {
            "path": str(path),
            "sha256": expected_hash,
            "size_bytes": expected_size,
        }
    return vision, map_sequence, runners, {
        "bundle": {
            "path": str(bundle_path.resolve()),
            "sha256": _sha256(bundle_path),
            "io_schema_digest": bundle["io_schema_digest"],
        },
        "sequences": {
            "vision_encoder": vision.metadata(),
            "map_encoder": map_sequence.metadata(),
        },
        "direct_artifacts": artifact_metadata,
    }


def _load_minddrive_direct(args: argparse.Namespace) -> tuple[
    Callable[[], Any],
    Callable[[], None],
    dict[str, object],
]:
    import gc

    import torch
    import torch._inductor.codecache  # noqa: F401
    from vlaforge.adapters.minddrive_real import (
        MINDDRIVE_STATE_TYPES,
        make_minddrive_torch_initial_state,
    )

    if args.bundle_root is None:
        raise ValueError("MindDrive direct artifact requires --bundle-root")
    bundle_root = args.bundle_root.resolve()
    frames, input_metadata = _load_minddrive_inputs(
        torch,
        bundle_root=bundle_root,
        input_root=args.input_root.resolve(),
    )
    vision, map_sequence, runners, artifact_metadata = (
        _load_verified_minddrive_direct_artifacts(torch, bundle_root)
    )
    initial = make_minddrive_torch_initial_state(torch, device="cuda:0")
    state = {
        name: initial[name] for name, _ in MINDDRIVE_STATE_TYPES
    }
    detection_names = tuple(
        name for name, _ in MINDDRIVE_STATE_TYPES[:10]
    )
    map_names = tuple(name for name, _ in MINDDRIVE_STATE_TYPES[10:])
    frame_index = 0

    def call(name: str, *inputs: Any) -> tuple[Any, ...]:
        outputs = []
        for tensor in _as_outputs(runners[name](*inputs)):
            outputs.append(
                tensor if tensor.is_contiguous() else tensor.contiguous()
            )
        return tuple(outputs)

    def run() -> Any:
        nonlocal frame_index, state
        frame = frames[frame_index % len(frames)]
        frame_index += 1
        with torch.inference_mode():
            image_features = vision.run(frame["camera_images"])[0]
            position = call(
                "position_encoder",
                image_features,
                frame["lidar2img"],
                frame["camera_intrinsics"],
            )[0]
            map_outputs = map_sequence.run(
                image_features,
                position,
                frame["timestamp"],
                frame["ego_pose"],
                frame["ego_pose_inverse"],
                *(state[name] for name in map_names),
            )
            detection_outputs = call(
                "detection_encoder",
                image_features,
                position,
                *map_outputs[:3],
                frame["timestamp"],
                frame["ego_pose"],
                frame["ego_pose_inverse"],
                frame["can_bus"],
                frame["route_command_index"],
                *(state[name] for name in detection_names),
            )
            decision = call(
                "decision_expert",
                frame["decision_input_ids"],
                detection_outputs[5],
                map_outputs[3],
            )[0]
            action = call(
                "action_expert",
                frame["planning_input_ids"],
                detection_outputs[5],
                map_outputs[3],
            )[0]
            trajectory = call(
                "trajectory_decoder",
                action,
                decision,
                frame["ego_route_command"],
                frame["trajectory_noise"],
                frame["path_noise"],
            )
            call(
                "detection_decoder",
                detection_outputs[0],
                detection_outputs[1],
                detection_outputs[2],
            )
            state = {
                **{
                    name: value.contiguous().clone()
                    for name, value in zip(
                        detection_names,
                        detection_outputs[6:],
                        strict=True,
                    )
                },
                **{
                    name: value.contiguous().clone()
                    for name, value in zip(
                        map_names,
                        map_outputs[4:],
                        strict=True,
                    )
                },
            }
            return trajectory[0]

    def cleanup() -> None:
        nonlocal state
        state = {}
        frames.clear()
        runners.clear()
        vision.close()
        map_sequence.close()
        gc.collect()
        torch.cuda.empty_cache()

    return run, cleanup, {
        **artifact_metadata,
        "inputs": input_metadata,
        "execution_contract": (
            "direct persistent AOTI execution of two verified sequences and "
            "six shared-library Regions with explicit 16-state carry"
        ),
        "state_count": len(MINDDRIVE_STATE_TYPES),
    }


def _load_minddrive_eager(args: argparse.Namespace) -> tuple[
    Callable[[], Any],
    Callable[[], None],
    dict[str, object],
]:
    import gc

    import torch
    from vlaforge.adapters.minddrive_real import (
        MINDDRIVE_STATE_TYPES,
        MINDDRIVE_UPSTREAM_STATE_KEYS,
        load_real_minddrive_model,
    )

    if (
        args.source_root is None
        or args.release_root is None
        or args.frame_root is None
        or args.bundle_root is None
    ):
        raise ValueError(
            "MindDrive eager requires --source-root, --release-root, "
            "--frame-root, and --bundle-root"
        )
    source_root = args.source_root.resolve()
    release_root = args.release_root.resolve()
    frame_root = args.frame_root.resolve()
    bundle_root = args.bundle_root.resolve()
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    # MindDrive's vendored MMCV imports its training-only Carla environment
    # from package initializers even for a model-only offline build.  Install
    # a narrow inert module for that unused dependency rather than importing
    # simulator, leaderboard, sensor, or control code.
    import types

    training_runner_module = "team_code.carla_env.carla_env_scenario"
    if training_runner_module not in sys.modules:
        unused_runner = types.ModuleType(training_runner_module)

        class _UnusedCarlaScenarioEnv:
            def __init__(self, *_values: object, **_options: object) -> None:
                raise RuntimeError(
                    "CarlaScenarioEnv is outside the offline benchmark"
                )

        unused_runner.CarlaScenarioEnv = _UnusedCarlaScenarioEnv
        unused_runner.TickRuntimeError = RuntimeError
        sys.modules[training_runner_module] = unused_runner
    unused_iou_module = "mmcv.ops.iou3d_det.iou3d_cuda"
    if unused_iou_module not in sys.modules:
        unused_iou = types.ModuleType(unused_iou_module)

        def unavailable_iou(*_values: object, **_options: object) -> None:
            raise RuntimeError(
                "iou3d_cuda is outside the frozen MindDrive inference path"
            )

        unused_iou.boxes_iou_bev_gpu = unavailable_iou
        unused_iou.nms_gpu = unavailable_iou
        unused_iou.nms_normal_gpu = unavailable_iou
        sys.modules[unused_iou_module] = unused_iou
    unused_roiaware_module = (
        "mmcv.ops.roiaware_pool3d.roiaware_pool3d_ext"
    )
    if unused_roiaware_module not in sys.modules:
        unused_roiaware = types.ModuleType(unused_roiaware_module)

        def unavailable_roiaware(
            *_values: object, **_options: object
        ) -> None:
            raise RuntimeError(
                "roiaware_pool3d is outside the frozen MindDrive inference "
                "path"
            )

        unused_roiaware.points_in_boxes_gpu = unavailable_roiaware
        unused_roiaware.points_in_boxes_cpu = unavailable_roiaware
        unused_roiaware.points_in_boxes_batch = unavailable_roiaware
        unused_roiaware.forward = unavailable_roiaware
        unused_roiaware.backward = unavailable_roiaware
        sys.modules[unused_roiaware_module] = unused_roiaware
    from mmcv import Config
    from mmcv.datasets.pipelines import Compose
    from mmcv.models.utils import attention as minddrive_attention
    from mmcv.parallel.collate import collate
    from probe_real_minddrive_eager import (
        _build_raw_invocation,
        _flatten_preprocessed_inputs,
        _move_batch_to_device,
        _replace_tokenizer_paths,
        _reset_model_state,
    )
    original_unpad_input = minddrive_attention.unpad_input

    def compatible_unpad_input(*values: Any, **options: Any) -> Any:
        outputs = original_unpad_input(*values, **options)
        if len(outputs) == 4:
            # The pinned source ignores its fifth value. flash-attn 2.6
            # removed that unused return while preserving the four tensor
            # results consumed by MindDrive.
            return (*outputs, None)
        return outputs

    minddrive_attention.unpad_input = compatible_unpad_input

    model = load_real_minddrive_model(
        source_root,
        release_root,
        device="cuda:0",
    )
    config_path = (
        source_root
        / "adzoo"
        / "minddrive"
        / "configs"
        / "minddrive_qwen2_05B_infer.py"
    )
    config = Config.fromfile(str(config_path))
    vlm_root = release_root / "llava-qwen2-0.5b"
    pipeline = Compose(
        _replace_tokenizer_paths(
            list(config.inference_only_pipeline), vlm_root
        )
    )
    direct_inputs, input_metadata = _load_minddrive_inputs(
        torch,
        bundle_root=bundle_root,
        input_root=args.input_root.resolve(),
    )
    batches = []
    frontend_exact = {}
    meta_action_replay = {}
    for frame_name, expected in zip(
        _MINDDRIVE_FRAMES, direct_inputs, strict=True
    ):
        raw = _build_raw_invocation(
            frame_root=frame_root,
            frame=frame_name.removeprefix("frame_"),
        )
        raw.pop("input_provenance")
        prepared = pipeline(raw)
        batch = collate([prepared], samples_per_gpu=1)
        flattened = _flatten_preprocessed_inputs(batch)
        decision_candidate = flattened["decision_input_ids"]
        decision_reference = expected["decision_input_ids"].cpu()
        if not torch.equal(decision_candidate, decision_reference):
            speed_token_ids = {
                int(transform.tokenizer.convert_tokens_to_ids(token))
                for transform in pipeline.transforms
                if hasattr(transform, "SPEED_SPT")
                for token in transform.SPEED_SPT
            }
            replay = _minddrive_meta_action_replay(
                torch,
                decision_candidate,
                decision_reference,
                allowed_token_ids=speed_token_ids,
            )
            if replay is None:
                raise AssertionError("mismatched prompt requires replay")
            # The official pipeline samples its speed meta-action with
            # random.choice. Replay the captured token so all three paths
            # consume the same declared frontend input.
            batch["input_ids"][0][0][0] = decision_reference.clone()
            flattened = _flatten_preprocessed_inputs(batch)
            meta_action_replay[frame_name] = replay
        exact = {}
        for name, candidate in flattened.items():
            reference = expected[name].cpu()
            exact[name] = bool(torch.equal(candidate, reference))
        if not all(exact.values()):
            mismatched = [name for name, passed in exact.items() if not passed]
            raise ValueError(
                f"{frame_name}: upstream frontend differs from bundle "
                f"inputs: {mismatched}"
            )
        frontend_exact[frame_name] = exact
        _move_batch_to_device(batch, "cuda:0")
        batches.append(batch)
    _reset_model_state(model)
    frame_index = 0
    original_randn_like = torch.randn_like
    state_normalization = {
        "calls": 0,
        "truncation_counts": {
            name: 0 for name, _ in MINDDRIVE_STATE_TYPES
        },
        "last": {},
    }

    def run() -> Any:
        nonlocal frame_index
        index = frame_index % len(batches)
        frame_index += 1
        noises = (
            direct_inputs[index]["trajectory_noise"],
            direct_inputs[index]["path_noise"],
        )
        noise_index = 0

        def explicit_randn_like(reference: Any, *values: Any, **kwargs: Any) -> Any:
            nonlocal noise_index
            if noise_index >= len(noises):
                raise RuntimeError(
                    "MindDrive eager produced unexpected hidden RNG"
                )
            noise = noises[noise_index]
            noise_index += 1
            if tuple(noise.shape) != tuple(reference.shape):
                raise RuntimeError(
                    "MindDrive eager planner-noise shape changed"
                )
            return noise.to(
                device=kwargs.get("device", reference.device),
                dtype=kwargs.get("dtype", reference.dtype),
            )

        torch.randn_like = explicit_randn_like
        try:
            with torch.inference_mode():
                # The upstream forward normalizes nested MMCV containers in
                # place. A Session Run receives a fresh binding view, so
                # preserve that contract while borrowing the tensor storage.
                invocation = _clone_minddrive_invocation(batches[index])
                output = model(invocation, return_loss=False)
        finally:
            torch.randn_like = original_randn_like
        if noise_index != 2:
            raise RuntimeError(
                "MindDrive eager did not consume both explicit noises"
            )
        if not isinstance(output, list) or len(output) != 1:
            raise RuntimeError("MindDrive eager output contract changed")
        normalized = _normalize_minddrive_upstream_state(
            model,
            MINDDRIVE_STATE_TYPES,
            MINDDRIVE_UPSTREAM_STATE_KEYS,
        )
        state_normalization["calls"] += 1
        for name, record in normalized.items():
            state_normalization["truncation_counts"][name] += int(
                record["truncated"]
            )
        state_normalization["last"] = normalized
        return output[0]["pts_bbox"]["ego_fut_preds"]

    def cleanup() -> None:
        nonlocal model
        torch.randn_like = original_randn_like
        minddrive_attention.unpad_input = original_unpad_input
        batches.clear()
        direct_inputs.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    checkpoint = release_root / "minddrive_rltrain.pth"
    return run, cleanup, {
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "source_root": str(source_root),
        "release_root": str(release_root),
        "real_frame_root": str(frame_root),
        "unused_training_runner_import_shim": training_runner_module,
        "unused_iou_import_shim": unused_iou_module,
        "unused_roiaware_import_shim": unused_roiaware_module,
        "flash_attn_unpad_compatibility": {
            "upstream_expected_returns": 5,
            "installed_returns": 4,
            "added_value": None,
            "upstream_uses_added_value": False,
        },
        "inputs": input_metadata,
        "frontend_exact_to_bundle": frontend_exact,
        "stochastic_meta_action_replay": meta_action_replay,
        "authoritative_state_projection": state_normalization,
        "execution_contract": (
            "pinned upstream full eager forward with official offline "
            "frontend, five real frames, explicit planner noise, and "
            "upstream authoritative state"
        ),
    }


def _load_path(
    args: argparse.Namespace,
) -> tuple[Callable[[], Any], Callable[[], None], dict[str, object]]:
    if args.model == "smolvla":
        if args.path == "eager":
            return _load_smolvla_eager(args)
        return _load_smolvla_direct(args)
    if args.model == "minddrive":
        if args.path == "eager":
            return _load_minddrive_eager(args)
        return _load_minddrive_direct(args)
    if args.path == "eager":
        return _load_diffusiondrive_eager(args)
    return _load_diffusiondrive_direct(args)


def _device_used(torch: Any) -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(total_bytes - free_bytes)


def _benchmark(
    torch: Any,
    run: Callable[[], Any],
    *,
    warmup: int,
    samples: int,
    first_run_counts_as_warmup: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if first_run_counts_as_warmup and warmup < 1:
        raise ValueError(
            "first Run can count as warmup only when warmup is positive"
        )
    rss_initialized = _rss_kib()
    cuda_initialized = _device_used(torch)
    first_started = time.perf_counter_ns()
    first_output = run()
    torch.cuda.synchronize()
    first_run_ns = time.perf_counter_ns() - first_started
    first_probe = float(first_output.reshape(-1)[0].item())
    if not math.isfinite(first_probe):
        raise RuntimeError("first model Run produced a non-finite output")
    remaining_warmups = warmup - int(first_run_counts_as_warmup)
    for _ in range(remaining_warmups):
        run()
        torch.cuda.synchronize()
    rss_start = _rss_kib()
    cuda_used_start = _device_used(torch)
    allocated_start = int(torch.cuda.memory_allocated())
    reserved_start = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, object]] = []
    checksum = 0.0
    for index in range(samples):
        started = time.perf_counter_ns()
        output = run()
        torch.cuda.synchronize()
        latency = time.perf_counter_ns() - started
        probe = float(output.reshape(-1)[0].item())
        if not math.isfinite(probe):
            raise RuntimeError("benchmark produced a non-finite output")
        checksum += probe
        records.append(
            {
                "index": index,
                "latency_ns": latency,
                "output_probe": probe,
            }
        )
    runtime = {
        "checksum": checksum,
        "first_run_ns": first_run_ns,
        "first_run_output_probe": first_probe,
        "rss_initialized_kib": rss_initialized,
        "rss_start_kib": rss_start,
        "rss_end_kib": _rss_kib(),
        "maximum_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_used_initialized_bytes": cuda_initialized,
        "cuda_used_start_bytes": cuda_used_start,
        "cuda_used_end_bytes": _device_used(torch),
        "torch_allocated_start_bytes": allocated_start,
        "torch_allocated_end_bytes": int(torch.cuda.memory_allocated()),
        "torch_allocated_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_reserved_start_bytes": reserved_start,
        "torch_reserved_end_bytes": int(torch.cuda.memory_reserved()),
        "torch_reserved_peak_bytes": int(torch.cuda.max_memory_reserved()),
    }
    return records, runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=_MODELS, required=True)
    parser.add_argument("--path", choices=_PATHS, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--l3-root", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--frame-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm-path", type=Path)
    parser.add_argument("--upstream-revision", default="unknown")
    parser.add_argument(
        "--first-run-counts-as-warmup",
        action="store_true",
        help=(
            "include the separately reported first Run in --warmup; used "
            "to align a stateful five-frame sequence with generated C++"
        ),
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be non-negative and samples positive")
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("real-model path benchmark requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    initialized = time.perf_counter()
    run, cleanup, metadata = _load_path(args)
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - initialized
    try:
        records, runtime = _benchmark(
            torch,
            run,
            warmup=args.warmup,
            samples=args.samples,
            first_run_counts_as_warmup=(
                args.first_run_counts_as_warmup
            ),
        )
    finally:
        cleanup()
    latencies = [int(item["latency_ns"]) for item in records]
    memory = {
        "warmup_rss_residency_kib": (
            int(runtime["rss_start_kib"])
            - int(runtime["rss_initialized_kib"])
        ),
        "rss_drift_kib": (
            int(runtime["rss_end_kib"]) - int(runtime["rss_start_kib"])
        ),
        "warmup_cuda_residency_bytes": (
            int(runtime["cuda_used_start_bytes"])
            - int(runtime["cuda_used_initialized_bytes"])
        ),
        "cuda_used_drift_bytes": (
            int(runtime["cuda_used_end_bytes"])
            - int(runtime["cuda_used_start_bytes"])
        ),
        "torch_allocated_drift_bytes": (
            int(runtime["torch_allocated_end_bytes"])
            - int(runtime["torch_allocated_start_bytes"])
        ),
        "torch_reserved_drift_bytes": (
            int(runtime["torch_reserved_end_bytes"])
            - int(runtime["torch_reserved_start_bytes"])
        ),
    }
    report = {
        "schema": _SCHEMA,
        "status": "passed",
        "model": args.model,
        "path": args.path,
        "seed": args.seed,
        "warmup": args.warmup,
        "warmup_semantics": (
            "first-run-inclusive"
            if args.first_run_counts_as_warmup
            else "additional-after-first-run"
        ),
        "samples": args.samples,
        "initialization_seconds": initialization_seconds,
        "latency": _latency_summary(latencies),
        "runtime": runtime,
        "memory": memory,
        "model_path": metadata,
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "driver": subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
        "reproduction": {
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                env=dict(os.environ),
            ).stdout.strip(),
            "source_dirty": bool(
                subprocess.run(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=dict(os.environ),
                ).stdout.strip()
            ),
        },
        "classification": (
            "Host-CUDA model-path benchmark; no Orin claim and no "
            "model-kernel optimization attribution"
        ),
        "samples_raw": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
