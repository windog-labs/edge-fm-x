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
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SOURCE_ROOT / "python"))


_SCHEMA = "vlaforge.real_model_path_benchmark/1"
_MODELS = ("smolvla", "diffusiondrive")
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


def _load_path(
    args: argparse.Namespace,
) -> tuple[Callable[[], Any], Callable[[], None], dict[str, object]]:
    if args.model == "smolvla":
        if args.path == "eager":
            return _load_smolvla_eager(args)
        return _load_smolvla_direct(args)
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
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rss_initialized = _rss_kib()
    cuda_initialized = _device_used(torch)
    for _ in range(warmup):
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
    parser.add_argument("--l3-root", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm-path", type=Path)
    parser.add_argument("--upstream-revision", default="unknown")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.samples < 1:
        parser.error("warmup must be non-negative and samples positive")
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("real-model path benchmark requires CUDA")
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
        "warmup": args.warmup,
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
