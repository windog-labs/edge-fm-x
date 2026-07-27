#!/usr/bin/env python3
"""Probe the official full AutoVLA camera-to-trajectory eager path.

The output is an L2 *candidate* capture, not Semantic IR evidence.  It proves
that the pinned source/checkpoint can execute the real processor, vision/VLM
frontend, generation loop, action-token extraction, and trajectory decoder on
an externally assembled three-camera history.  A later adapter must still
partition and verify the captured path before the evidence can be promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import platform
import resource
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.autovla_real import (  # noqa: E402
    AUTOVLA_ACTION_START_ID,
    AUTOVLA_ACTION_VOCAB_SIZE,
    AUTOVLA_CHECKPOINT_REVISION,
    AUTOVLA_CHECKPOINT_SHA256,
    AUTOVLA_CHECKPOINT_SIZE,
    AUTOVLA_CODEBOOK_SHA256,
    AUTOVLA_QWEN_REVISION,
    AUTOVLA_SOURCE_SHA256,
    AUTOVLA_UPSTREAM_REVISION,
)


REPORT_SCHEMA = "vlaforge.autovla_full_eager_probe/1"
INPUT_SCHEMA = "vlaforge.autovla_full_input/1"
CAMERAS = ("front_camera", "front_left_camera", "front_right_camera")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _numeric_vector(
    value: object,
    *,
    name: str,
) -> list[float]:
    if not isinstance(value, list) or len(value) not in (2, 3):
        raise ValueError(f"{name} must be a 2- or 3-element list")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(number)
    return result


def load_input_manifest(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve an already assembled camera history without doing sync."""

    manifest = _load_json(path)
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError(
            f"input schema must be {INPUT_SCHEMA!r}: {path}"
        )
    sample_id = manifest.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    images = manifest.get("images")
    if not isinstance(images, Mapping) or set(images) != set(CAMERAS):
        raise ValueError(
            "images must contain exactly front/front-left/front-right cameras"
        )
    resolved_images: dict[str, list[str]] = {}
    image_records = []
    for camera in CAMERAS:
        frames = images[camera]
        if not isinstance(frames, list) or len(frames) != 4:
            raise ValueError(f"{camera} must contain exactly four frames")
        resolved_images[camera] = []
        for index, value in enumerate(frames):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{camera}[{index}] must be a path")
            frame = Path(value).expanduser()
            if not frame.is_absolute():
                frame = path.parent / frame
            frame = frame.resolve()
            if not frame.is_file():
                raise FileNotFoundError(frame)
            resolved_images[camera].append(str(frame))
            image_records.append(
                {
                    "camera": camera,
                    "history_index": index,
                    "path": str(frame),
                    "size_bytes": frame.stat().st_size,
                    "sha256": _sha256(frame),
                }
            )
    command = manifest.get("driving_command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("driving_command must be a non-empty string")
    revision = manifest.get("revision")
    if revision is not None and (
        isinstance(revision, bool) or not isinstance(revision, int)
    ):
        raise ValueError("revision must be an integer when provided")
    features = {
        "images": resolved_images,
        # All paths are resolved above, so the upstream adapter must not join
        # another dataset root.
        "sensor_data_path": "",
        "vehicle_velocity": _numeric_vector(
            manifest.get("vehicle_velocity"),
            name="vehicle_velocity",
        ),
        "vehicle_acceleration": _numeric_vector(
            manifest.get("vehicle_acceleration"),
            name="vehicle_acceleration",
        ),
        "driving_command": command,
    }
    evidence = {
        "schema": INPUT_SCHEMA,
        "sample_id": sample_id,
        "revision": revision,
        "manifest": str(path.resolve()),
        "manifest_sha256": _sha256(path),
        "images": image_records,
        "history_assembled_externally": True,
        "sensor_sync_performed_by_vlaforge": False,
    }
    return features, evidence


def _load_config(
    *,
    config_path: Path,
    qwen_model_root: Path,
    codebook: Path,
) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"AutoVLA config is not a mapping: {config_path}")
    model = value.get("model")
    if not isinstance(model, dict):
        raise ValueError("AutoVLA config has no model mapping")
    model["pretrained_model_path"] = str(qwen_model_root.resolve())
    model["codebook_cache_path"] = str(codebook.resolve())
    return value


def _install_inference_only_score_shim() -> None:
    """Isolate upstream training-only nuPlan imports from AutoVLA inference."""

    module_name = "models.utils.score"
    if module_name in sys.modules:
        return

    class TrainingOnlyDependency:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError(
                "nuPlan scoring is unavailable in the inference-only "
                "AutoVLA environment"
            )

    module = types.ModuleType(module_name)
    module.PDM_Reward = TrainingOnlyDependency
    module.TrajectorySampling = TrainingOnlyDependency
    module.Trajectory = TrainingOnlyDependency
    sys.modules[module_name] = module


def _verify_source(source_root: Path) -> dict[str, Any]:
    try:
        repository = _repository_state(source_root)
    except (OSError, subprocess.CalledProcessError):
        repository = {
            "root": str(source_root.resolve()),
            "revision": AUTOVLA_UPSTREAM_REVISION,
            "source_dirty": False,
            "tracked_status": [],
            "identity_mode": "pinned-content-sha256",
            "git_checkout": False,
        }
    else:
        repository["identity_mode"] = "git-revision-and-content-sha256"
        repository["git_checkout"] = True
    if repository["revision"] != AUTOVLA_UPSTREAM_REVISION:
        raise ValueError(
            "AutoVLA revision mismatch: "
            f"{repository['revision']} != {AUTOVLA_UPSTREAM_REVISION}"
        )
    files = []
    for relative, expected in AUTOVLA_SOURCE_SHA256.items():
        path = source_root / relative
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"AutoVLA source digest mismatch: {relative}"
            )
        files.append(
            {
                "relative_path": relative,
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    return {"repository": repository, "files": files}


def _strict_load_checkpoint(
    model: Any,
    checkpoint: Path,
    *,
    verify_hash: bool,
) -> dict[str, Any]:
    import torch

    if checkpoint.stat().st_size != AUTOVLA_CHECKPOINT_SIZE:
        raise ValueError("AutoVLA checkpoint size mismatch")
    checkpoint_hash = _sha256(checkpoint) if verify_hash else None
    if verify_hash and checkpoint_hash != AUTOVLA_CHECKPOINT_SHA256:
        raise ValueError("AutoVLA checkpoint SHA256 mismatch")
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("AutoVLA checkpoint root is not a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise ValueError("AutoVLA checkpoint state_dict is not a mapping")
    prefix = "autovla."
    stripped = {
        str(key).removeprefix(prefix): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(stripped, strict=True)
    return {
        "path": str(checkpoint.resolve()),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": checkpoint_hash,
        "hash_verified": verify_hash,
        "tensor_count": len(stripped),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict": True,
    }


def _seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capturable_predict(
    model: Any,
    input_features: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    inputs = model.get_prompt(input_features)
    model_inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    generated = model.vlm.generate(
        **model_inputs,
        max_length=model.gen_conf["max_length"],
        do_sample=True,
        temperature=model.gen_conf["temperature"],
        top_k=model.gen_conf["top_k"],
        top_p=model.gen_conf["top_p"],
    )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated)
    ][0][:-1]
    decoded_text = model.processor.decode(trimmed)
    action_tokens = trimmed[trimmed >= model.action_start_id]
    trajectory = model.action_tokenizer.decode_token_ids_to_trajectory(
        action_tokens.cpu()
    )[0, 1:]
    return {
        "model_inputs": {
            key: value.detach().cpu() for key, value in model_inputs.items()
        },
        "generated_ids": generated.detach().cpu(),
        "trimmed_ids": trimmed.detach().cpu(),
        "action_tokens": action_tokens.detach().cpu(),
        "trajectory": trajectory.detach().cpu(),
        "decoded_text": decoded_text,
    }


def _metrics(expected: Any, actual: Any) -> dict[str, Any]:
    import torch

    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    difference = (expected.to(torch.float64) - actual.to(torch.float64)).abs()
    return {
        "shape": [int(item) for item in actual.shape],
        "exact": bool(torch.equal(expected, actual)),
        "maximum_absolute_error": (
            float(difference.max().item()) if difference.numel() else 0.0
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--qwen-model-root", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--verify-checkpoint-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("full AutoVLA eager probe requires CUDA")
    source_root = args.source_root.resolve()
    checkpoint = args.checkpoint.resolve()
    codebook = args.codebook.resolve()
    qwen_model_root = args.qwen_model_root.resolve()
    if _sha256(codebook) != AUTOVLA_CODEBOOK_SHA256:
        raise ValueError("AutoVLA codebook SHA256 mismatch")
    qwen_config = qwen_model_root / "config.json"
    if not qwen_config.is_file():
        raise FileNotFoundError(qwen_config)
    source = _verify_source(source_root)
    features, input_evidence = load_input_manifest(
        args.input_manifest.resolve()
    )
    config = _load_config(
        config_path=args.config.resolve(),
        qwen_model_root=qwen_model_root,
        codebook=codebook,
    )

    sys.path.insert(0, str(source_root))
    _install_inference_only_score_shim()
    upstream = importlib.import_module("models.autovla")
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    load_started = time.perf_counter()
    model = upstream.AutoVLA(
        config,
        inference=True,
        device=args.device,
    )
    checkpoint_record = _strict_load_checkpoint(
        model,
        checkpoint,
        verify_hash=args.verify_checkpoint_hash,
    )
    model.eval()
    torch.cuda.synchronize(args.device)
    load_seconds = time.perf_counter() - load_started

    _seed(torch, args.seed)
    official_started = time.perf_counter()
    with torch.inference_mode():
        official_trajectory, official_text = model.predict(features)
    torch.cuda.synchronize(args.device)
    official_seconds = time.perf_counter() - official_started

    _seed(torch, args.seed)
    captured_started = time.perf_counter()
    with torch.inference_mode():
        captured = _capturable_predict(model, features)
    torch.cuda.synchronize(args.device)
    captured_seconds = time.perf_counter() - captured_started
    trajectory_metrics = _metrics(
        official_trajectory,
        captured["trajectory"],
    )
    text_exact = official_text == captured["decoded_text"]
    tokens = captured["action_tokens"]
    token_range = bool(
        tokens.numel() > 0
        and torch.all(tokens >= AUTOVLA_ACTION_START_ID)
        and torch.all(
            tokens
            < AUTOVLA_ACTION_START_ID + AUTOVLA_ACTION_VOCAB_SIZE
        )
    )
    finite = bool(torch.isfinite(captured["trajectory"]).all())
    passed = bool(
        trajectory_metrics["exact"]
        and text_exact
        and token_range
        and finite
        and not checkpoint_record["missing_keys"]
        and not checkpoint_record["unexpected_keys"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = args.output_dir / "autovla_full_eager_capture.pt"
    torch.save(
        {
            "schema": REPORT_SCHEMA,
            "input": input_evidence,
            "model_inputs": captured["model_inputs"],
            "generated_ids": captured["generated_ids"],
            "trimmed_ids": captured["trimmed_ids"],
            "action_tokens": tokens,
            "trajectory": captured["trajectory"],
            "official_trajectory": official_trajectory.detach().cpu(),
        },
        capture_path,
    )
    major, minor = torch.cuda.get_device_capability(args.device)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "evidence_level": "L2-candidate-full-real-checkpoint-eager",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "AutoVLA",
        "source": source,
        "inference_environment": {
            "training_only_nuplan_score_import_shim": True,
            "model_inference_implementation_modified": False,
        },
        "checkpoint": checkpoint_record,
        "pinned_revisions": {
            "autovla": AUTOVLA_UPSTREAM_REVISION,
            "checkpoint": AUTOVLA_CHECKPOINT_REVISION,
            "qwen": AUTOVLA_QWEN_REVISION,
        },
        "input": input_evidence,
        "capture": {
            "path": str(capture_path.resolve()),
            "sha256": _sha256(capture_path),
            "size_bytes": capture_path.stat().st_size,
        },
        "outputs": {
            "trajectory": trajectory_metrics,
            "action_token_count": int(tokens.numel()),
            "action_tokens_in_range": token_range,
            "trajectory_finite": finite,
            "decoded_text_exact": text_exact,
            "decoded_text_sha256": hashlib.sha256(
                captured["decoded_text"].encode("utf-8")
            ).hexdigest(),
        },
        "timing_and_memory": {
            "load_seconds": load_seconds,
            "official_predict_seconds": official_seconds,
            "capturable_predict_seconds": captured_seconds,
            "peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(args.device)
            ),
            "peak_cuda_reserved_bytes": int(
                torch.cuda.max_memory_reserved(args.device)
            ),
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        },
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device),
            "device": args.device,
            "target": f"sm_{major}{minor}",
        },
        "repository": _repository_state(_REPOSITORY_ROOT),
        "claim_boundary": {
            "official_full_camera_to_trajectory_eager": True,
            "semantic_ir_or_plan_parity": False,
            "compiled_artifact": False,
            "generated_cpp_session": False,
            "sensor_sync_or_collection": False,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
