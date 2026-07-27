#!/usr/bin/env python3
"""Run the portable AutoVLA high-memory handoff gates.

This entry point intentionally covers the already implemented real-checkpoint
partition.  It gives an A100/H20 developer a trustworthy baseline before the
full camera-to-trajectory adapter is implemented:

1. verify source/checkpoint identity and destination capacity;
2. reproduce the partitioned real L2 frontend;
3. compile destination-native AOTI packages in a fresh process;
4. audit exported-vs-compiled parity without reusing another GPU's packages.

Every stage is a separate process so model and compiler allocations do not
silently accumulate across the handoff workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.autovla_real import (  # noqa: E402
    AUTOVLA_CHECKPOINT_REVISION,
    AUTOVLA_CHECKPOINT_SHA256,
    AUTOVLA_CHECKPOINT_SIZE,
    AUTOVLA_CODEBOOK_SHA256,
    AUTOVLA_QWEN_CONFIG_SHA256,
    AUTOVLA_QWEN_REVISION,
    AUTOVLA_SOURCE_SHA256,
    AUTOVLA_UPSTREAM_REVISION,
)


REPORT_SCHEMA = "vlaforge.high_memory_autovla_handoff/1"
REGIONS = (
    "autovla_decoder_mlp",
    "autovla_action_projection",
    "autovla_trajectory_decode",
)
STAGE_ORDER = ("partition-l2", "partition-compile", "partition-l3")


@dataclass(frozen=True, slots=True)
class RunConfig:
    source_root: Path
    checkpoint: Path
    codebook: Path
    qwen_config: Path
    output_root: Path
    device: str
    target: str
    inductor_profile: str
    region_nrmse_tolerance: float
    trajectory_max_abs_tolerance: float


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    command: tuple[str, ...]
    report: Path


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(
    root: Path,
    *arguments: str,
) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository_state(root: Path) -> dict[str, Any]:
    status = _git(root, "status", "--short", "--untracked-files=no")
    return {
        "root": str(root.resolve()),
        "revision": _git(root, "rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "tracked_status": status.splitlines() if status else [],
    }


def _target_from_capability(major: int, minor: int) -> str:
    if major < 1 or minor < 0:
        raise ValueError(f"invalid CUDA capability: {(major, minor)}")
    return f"sm_{major}{minor}"


def _torch_arch_list(target: str) -> str:
    if not target.startswith("sm_") or not target[3:].isdigit():
        raise ValueError(f"invalid CUDA target: {target}")
    digits = target[3:]
    if len(digits) < 2:
        raise ValueError(f"invalid CUDA target: {target}")
    return f"{digits[:-1]}.{digits[-1]}"


def _host_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * pages)


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(path)
        candidate = candidate.parent
    return candidate


def _gpu_probe(device: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    index = torch.device(device)
    properties = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "device": device,
        "gpu": torch.cuda.get_device_name(index),
        "target": _target_from_capability(major, minor),
        "compute_capability": [major, minor],
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(major >= 8),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": _nvidia_driver(),
    }


def _nvidia_driver() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    result = subprocess.run(
        [
            executable,
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    values = [item.strip() for item in result.stdout.splitlines() if item]
    return values[0] if values else None


def _artifact_record(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_size: int | None,
    verify_hash: bool,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "expected_sha256": expected_sha256,
        "expected_size_bytes": expected_size,
    }
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return record, errors
    actual_size = path.stat().st_size
    record["size_bytes"] = actual_size
    if expected_size is not None and actual_size != expected_size:
        errors.append(
            f"size mismatch for {path}: {actual_size} != {expected_size}"
        )
    if verify_hash:
        actual_hash = _sha256(path)
        record["sha256"] = actual_hash
        if expected_sha256 is not None and actual_hash != expected_sha256:
            errors.append(
                f"SHA256 mismatch for {path}: "
                f"{actual_hash} != {expected_sha256}"
            )
    else:
        record["sha256"] = None
        record["hash_verification_skipped"] = True
    return record, errors


def _source_record(source_root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if not source_root.is_dir():
        return (
            {"root": str(source_root.resolve()), "exists": False},
            [f"missing AutoVLA source root: {source_root}"],
        )
    try:
        repository = _repository_state(source_root)
    except (OSError, subprocess.CalledProcessError) as error:
        return (
            {"root": str(source_root.resolve()), "exists": True},
            [f"AutoVLA source is not a Git checkout: {error}"],
        )
    if repository["revision"] != AUTOVLA_UPSTREAM_REVISION:
        errors.append(
            "AutoVLA source revision mismatch: "
            f"{repository['revision']} != {AUTOVLA_UPSTREAM_REVISION}"
        )
    files: list[dict[str, Any]] = []
    for relative, expected in AUTOVLA_SOURCE_SHA256.items():
        path = source_root / relative
        record, item_errors = _artifact_record(
            path,
            expected_sha256=expected,
            expected_size=None,
            verify_hash=True,
        )
        record["relative_path"] = relative
        files.append(record)
        errors.extend(item_errors)
    return {
        "root": str(source_root.resolve()),
        "exists": True,
        "repository": repository,
        "files": files,
    }, errors


def _preflight(
    config: RunConfig,
    *,
    minimum_vram_gib: float,
    minimum_host_ram_gib: float,
    minimum_free_disk_gib: float,
    verify_checkpoint_hash: bool,
    allow_dirty: bool,
    qwen_model_root: Path | None,
) -> dict[str, Any]:
    errors: list[str] = []
    repository = _repository_state(_REPOSITORY_ROOT)
    if repository["source_dirty"] and not allow_dirty:
        errors.append("VLAForge repository is dirty; commit before evidence")

    source, source_errors = _source_record(config.source_root)
    errors.extend(source_errors)
    checkpoint, checkpoint_errors = _artifact_record(
        config.checkpoint,
        expected_sha256=AUTOVLA_CHECKPOINT_SHA256,
        expected_size=AUTOVLA_CHECKPOINT_SIZE,
        verify_hash=verify_checkpoint_hash,
    )
    errors.extend(checkpoint_errors)
    codebook, codebook_errors = _artifact_record(
        config.codebook,
        expected_sha256=AUTOVLA_CODEBOOK_SHA256,
        expected_size=None,
        verify_hash=True,
    )
    errors.extend(codebook_errors)
    qwen_config, config_errors = _artifact_record(
        config.qwen_config,
        expected_sha256=AUTOVLA_QWEN_CONFIG_SHA256,
        expected_size=None,
        verify_hash=True,
    )
    errors.extend(config_errors)

    gpu = _gpu_probe(config.device)
    if gpu["target"] != config.target:
        errors.append(
            f"requested target {config.target} does not match "
            f"destination GPU {gpu['target']}"
        )
    gib = 1024**3
    if gpu["total_memory_bytes"] < int(minimum_vram_gib * gib):
        errors.append(
            "insufficient GPU memory: "
            f"{gpu['total_memory_bytes'] / gib:.2f} GiB "
            f"< {minimum_vram_gib:.2f} GiB"
        )
    if not gpu["bf16_supported"]:
        errors.append("destination GPU does not support the BF16 profile")
    host_ram = _host_ram_bytes()
    if host_ram < int(minimum_host_ram_gib * gib):
        errors.append(
            f"insufficient Host RAM: {host_ram / gib:.2f} GiB "
            f"< {minimum_host_ram_gib:.2f} GiB"
        )
    disk_root = _existing_parent(config.output_root)
    disk = shutil.disk_usage(disk_root)
    if disk.free < int(minimum_free_disk_gib * gib):
        errors.append(
            f"insufficient free disk: {disk.free / gib:.2f} GiB "
            f"< {minimum_free_disk_gib:.2f} GiB"
        )

    qwen_model: dict[str, Any] | None = None
    if qwen_model_root is not None:
        required = qwen_model_root / "config.json"
        weights = tuple(qwen_model_root.glob("*.safetensors"))
        qwen_model = {
            "root": str(qwen_model_root.resolve()),
            "config_exists": required.is_file(),
            "weight_shard_count": len(weights),
            "weight_bytes": sum(item.stat().st_size for item in weights),
            "revision": AUTOVLA_QWEN_REVISION,
        }
        if not required.is_file() or not weights:
            errors.append(
                "Qwen model root must contain config.json and safetensor "
                f"weight shards: {qwen_model_root}"
            )

    return {
        "schema": REPORT_SCHEMA,
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "platform": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "host_ram_bytes": host_ram,
            "disk_probe_root": str(disk_root),
            "disk_free_bytes": disk.free,
            "gpu": gpu,
        },
        "requirements": {
            "minimum_vram_gib": minimum_vram_gib,
            "minimum_host_ram_gib": minimum_host_ram_gib,
            "minimum_free_disk_gib": minimum_free_disk_gib,
            "destination_native_aoti_required": True,
        },
        "inputs": {
            "source": source,
            "checkpoint": checkpoint,
            "codebook": codebook,
            "qwen_config": qwen_config,
            "qwen_model": qwen_model,
        },
        "pinned_revisions": {
            "autovla": AUTOVLA_UPSTREAM_REVISION,
            "checkpoint": AUTOVLA_CHECKPOINT_REVISION,
            "qwen": AUTOVLA_QWEN_REVISION,
        },
        "errors": errors,
        "claim_boundary": {
            "preflight_and_partition_baseline_only": True,
            "full_camera_to_trajectory_evidence": False,
            "cross_gpu_performance_claim": False,
        },
    }


def _stages(config: RunConfig) -> tuple[Stage, ...]:
    l2_root = config.output_root / "partition_l2"
    l3_root = config.output_root / "partition_l3"
    frontend_report = l2_root / "autovla_frontend_l2.json"
    compile_report = l3_root / "aoti_compile.json"
    l3_report = l3_root / "autovla_artifact_l3.json"
    python = sys.executable
    return (
        Stage(
            "partition-l2",
            (
                python,
                str(_SOURCE_ROOT / "tools/audit_real_autovla_frontend.py"),
                "--source-root",
                str(config.source_root),
                "--checkpoint",
                str(config.checkpoint),
                "--codebook",
                str(config.codebook),
                "--qwen-config",
                str(config.qwen_config),
                "--export-dir",
                str(l2_root / "exports"),
                "--report",
                str(frontend_report),
                "--device",
                config.device,
            ),
            frontend_report,
        ),
        Stage(
            "partition-compile",
            (
                python,
                str(_SOURCE_ROOT / "tools/compile_real_aoti_exports.py"),
                "--export-dir",
                str(l2_root / "exports"),
                "--output-dir",
                str(l3_root / "artifacts"),
                "--report",
                str(compile_report),
                "--device",
                config.device,
                "--inductor-profile",
                config.inductor_profile,
                *REGIONS,
            ),
            compile_report,
        ),
        Stage(
            "partition-l3",
            (
                python,
                str(_SOURCE_ROOT / "tools/audit_real_autovla_artifacts.py"),
                "--export-dir",
                str(l2_root / "exports"),
                "--artifact-dir",
                str(l3_root / "artifacts"),
                "--frontend-report",
                str(frontend_report),
                "--compile-report",
                str(compile_report),
                "--report",
                str(l3_report),
                "--device",
                config.device,
                "--expected-target",
                config.target,
                "--region-nrmse-tolerance",
                str(config.region_nrmse_tolerance),
                "--trajectory-max-abs-tolerance",
                str(config.trajectory_max_abs_tolerance),
            ),
            l3_report,
        ),
    )


def _stage_complete(stage: Stage, target: str) -> bool:
    if not stage.report.is_file():
        return False
    payload = _json(stage.report)
    if stage.name == "partition-compile":
        return (
            payload.get("schema") == "vlaforge.real_aoti_compile/1"
            and payload.get("environment", {}).get("target") == target
            and {
                str(item.get("region"))
                for item in payload.get("regions", ())
            }
            == set(REGIONS)
        )
    return bool(payload.get("passed") or payload.get("status") == "passed")


def _run_stage(
    stage: Stage,
    *,
    environment: Mapping[str, str],
    state: dict[str, Any],
    state_path: Path,
) -> int:
    log = state_path.parent / "logs" / f"{stage.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    record = {
        "name": stage.name,
        "status": "running",
        "started_at": started,
        "command": list(stage.command),
        "log": str(log.resolve()),
        "report": str(stage.report.resolve()),
    }
    state["stages"].append(record)
    _write_json(state_path, state)
    monotonic = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            stage.command,
            cwd=_REPOSITORY_ROOT,
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    record.update(
        {
            "status": "passed" if process.returncode == 0 else "failed",
            "returncode": process.returncode,
            "elapsed_seconds": time.monotonic() - monotonic,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json(state_path, state)
    return process.returncode


def _path_default(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_path_default("VLAFORGE_AUTOVLA_SOURCE_ROOT"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_path_default("VLAFORGE_AUTOVLA_CHECKPOINT"),
    )
    parser.add_argument(
        "--codebook",
        type=Path,
        default=_path_default("VLAFORGE_AUTOVLA_CODEBOOK"),
    )
    parser.add_argument(
        "--qwen-config",
        type=Path,
        default=_path_default("VLAFORGE_AUTOVLA_QWEN_CONFIG"),
    )
    parser.add_argument(
        "--qwen-model-root",
        type=Path,
        default=_path_default("VLAFORGE_AUTOVLA_QWEN_MODEL_ROOT"),
        help="optional full Qwen snapshot probe for subsequent full frontend work",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_path_default("VLAFORGE_HIGH_MEMORY_OUTPUT_ROOT"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--target",
        default="auto",
        help="auto or an explicit destination target such as sm_80/sm_90",
    )
    parser.add_argument(
        "--through",
        choices=("preflight", *STAGE_ORDER),
        default="partition-l3",
    )
    parser.add_argument(
        "--inductor-profile",
        choices=("default", "conservative"),
        default="conservative",
    )
    parser.add_argument("--region-nrmse-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--trajectory-max-abs-tolerance",
        type=float,
        default=2e-3,
    )
    parser.add_argument("--minimum-vram-gib", type=float, default=39.0)
    parser.add_argument("--minimum-host-ram-gib", type=float, default=60.0)
    parser.add_argument("--minimum-free-disk-gib", type=float, default=80.0)
    parser.add_argument(
        "--verify-checkpoint-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="print destination commands without executing or probing inputs",
    )
    return parser


def _required_paths(args: argparse.Namespace) -> None:
    missing = [
        option
        for option in (
            "source_root",
            "checkpoint",
            "codebook",
            "qwen_config",
            "output_root",
        )
        if getattr(args, option) is None
    ]
    if missing:
        names = ", ".join(item.replace("_", "-") for item in missing)
        raise ValueError(f"missing required paths: {names}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _required_paths(args)
    except ValueError as error:
        _parser().error(str(error))
    assert args.source_root is not None
    assert args.checkpoint is not None
    assert args.codebook is not None
    assert args.qwen_config is not None
    assert args.output_root is not None

    explicit_target = None if args.target == "auto" else args.target
    if args.print_plan and explicit_target is None:
        raise ValueError("--print-plan requires an explicit --target")
    if explicit_target is not None:
        _torch_arch_list(explicit_target)
    target = explicit_target
    if target is None:
        target = str(_gpu_probe(args.device)["target"])

    config = RunConfig(
        source_root=args.source_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        codebook=args.codebook.resolve(),
        qwen_config=args.qwen_config.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
        target=target,
        inductor_profile=args.inductor_profile,
        region_nrmse_tolerance=args.region_nrmse_tolerance,
        trajectory_max_abs_tolerance=args.trajectory_max_abs_tolerance,
    )
    stages = _stages(config)
    selected_count = (
        0
        if args.through == "preflight"
        else STAGE_ORDER.index(args.through) + 1
    )
    selected = stages[:selected_count]
    if args.print_plan:
        print(
            json.dumps(
                {
                    "schema": REPORT_SCHEMA,
                    "target": target,
                    "torch_cuda_arch_list": _torch_arch_list(target),
                    "stages": [
                        {
                            "name": stage.name,
                            "command": list(stage.command),
                            "report": str(stage.report),
                        }
                        for stage in selected
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    config.output_root.mkdir(parents=True, exist_ok=True)
    preflight = _preflight(
        config,
        minimum_vram_gib=args.minimum_vram_gib,
        minimum_host_ram_gib=args.minimum_host_ram_gib,
        minimum_free_disk_gib=args.minimum_free_disk_gib,
        verify_checkpoint_hash=args.verify_checkpoint_hash,
        allow_dirty=args.allow_dirty,
        qwen_model_root=(
            args.qwen_model_root.resolve()
            if args.qwen_model_root is not None
            else None
        ),
    )
    preflight_path = config.output_root / "preflight.json"
    _write_json(preflight_path, preflight)
    print(
        json.dumps(
            {
                "status": preflight["status"],
                "target": target,
                "report": str(preflight_path),
                "errors": preflight["errors"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not preflight["passed"]:
        return 2
    if args.through == "preflight":
        return 0

    state_path = config.output_root / "run_state.json"
    state: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "running",
        "target": target,
        "torch_cuda_arch_list": _torch_arch_list(target),
        "preflight": str(preflight_path.resolve()),
        "stages": [],
    }
    environment = os.environ.copy()
    environment["TORCH_CUDA_ARCH_LIST"] = _torch_arch_list(target)
    environment["PYTHONPATH"] = str(_SOURCE_ROOT / "python")
    for stage in selected:
        if args.resume and _stage_complete(stage, target):
            state["stages"].append(
                {
                    "name": stage.name,
                    "status": "resumed",
                    "report": str(stage.report.resolve()),
                }
            )
            _write_json(state_path, state)
            continue
        returncode = _run_stage(
            stage,
            environment=environment,
            state=state,
            state_path=state_path,
        )
        if returncode != 0:
            state["status"] = "failed"
            _write_json(state_path, state)
            print(
                f"{stage.name} failed; inspect "
                f"{state_path.parent / 'logs' / (stage.name + '.log')}",
                file=sys.stderr,
            )
            return returncode
    state["status"] = "passed"
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
