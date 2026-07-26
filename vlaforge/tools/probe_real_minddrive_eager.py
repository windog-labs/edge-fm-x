#!/usr/bin/env python3
"""Run the pinned MindDrive 0.5B offline frontend without CARLA.

This is an upstream-compatibility probe, not an L2 claim generator.  It keeps
the official model, checkpoint, image/VQA pipeline, and model forward path,
while reconstructing one invocation from an archived real Bench2Drive frame.
Sensor acquisition, synchronization, route planning, PID, and VehicleControl
remain outside the process.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MINDDRIVE_REVISION = "1a4085dab1c20895a0c8d2b67b4f8e65712fa8de"
MINDDRIVE_CHECKPOINT_REVISION = (
    "5cf1eafc7f6d1028006f2d97d083d8e9aa4c0b12"
)
MINDDRIVE_CHECKPOINT_SHA256 = (
    "39c86eddeaf57b15b9aeb54beb9f698539f6ee1e83529724b1bfc8c5e11b4ba0"
)
MINDDRIVE_CHECKPOINT_SIZE = 6_593_355_869
MINDDRIVE_VLM_SHA256 = (
    "6fc9882475867279ee66e505ded47b5d722fc09b0d34bc7684a26080d662825f"
)
MINDDRIVE_VLM_SIZE = 1_892_090_688

CAMERAS = (
    ("CAM_FRONT", "rgb_front"),
    ("CAM_FRONT_LEFT", "rgb_front_left"),
    ("CAM_FRONT_RIGHT", "rgb_front_right"),
    ("CAM_BACK", "rgb_back"),
    ("CAM_BACK_LEFT", "rgb_back_left"),
    ("CAM_BACK_RIGHT", "rgb_back_right"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object in {path}")
    return payload


def _source_revision(source_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _key_groups(keys: list[str]) -> dict[str, int]:
    groups: dict[str, int] = {}
    for key in keys:
        parts = key.split(".")
        group = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        groups[group] = groups.get(group, 0) + 1
    return dict(sorted(groups.items()))


def _key_audit(keys: list[str]) -> dict[str, Any]:
    return {
        "count": len(keys),
        "groups": _key_groups(keys),
        "sample": keys[:32],
        "sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }


def _is_allowed_training_extra(key: str) -> bool:
    return (
        key.startswith("value_net.")
        or key.startswith("value_net_pro.")
        or (
            key.startswith("lm_head.")
            and key.endswith("rotary_emb.inv_freq")
        )
    )


def _quaternion_z(yaw: float) -> list[float]:
    return [
        math.cos(yaw / 2.0),
        0.0,
        0.0,
        math.sin(yaw / 2.0),
    ]


def _build_raw_invocation(
    *,
    frame_root: Path,
    frame: str,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    annotation_path = frame_root / "anno" / f"{frame}.json.gz"
    if not annotation_path.is_file():
        annotation_path = (
            frame_root / "appended_anno" / f"{frame}.json.gz"
        )
    annotation = _load_gzip_json(annotation_path)
    measurement = _load_gzip_json(
        frame_root / "measurements" / f"{frame}.json.gz"
    )
    sensors = annotation["sensors"]
    lidar = sensors["LIDAR_TOP"]
    lidar2ego = np.asarray(lidar["lidar2ego"], dtype=np.float32)
    world2lidar = np.asarray(lidar["world2lidar"], dtype=np.float32)
    lidar2global = np.linalg.inv(world2lidar).astype(np.float32)

    images = []
    lidar2cam = []
    lidar2img = []
    intrinsics = []
    filenames = []
    for camera, folder in CAMERAS:
        filename = frame_root / "camera" / folder / f"{frame}.jpg"
        image = cv2.imread(str(filename), cv2.IMREAD_COLOR)
        if image is None or image.shape != (900, 1600, 3):
            raise ValueError(
                f"unexpected {camera} image shape: "
                f"{None if image is None else image.shape}"
            )
        camera_info = sensors[camera]
        cam2ego = np.asarray(camera_info["cam2ego"], dtype=np.float32)
        intrinsic = np.eye(4, dtype=np.float32)
        intrinsic[:3, :3] = np.asarray(
            camera_info["intrinsic"], dtype=np.float32
        )
        camera_lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego
        images.append(image)
        lidar2cam.append(camera_lidar2cam.astype(np.float32))
        intrinsics.append(intrinsic)
        lidar2img.append((intrinsic @ camera_lidar2cam).astype(np.float32))
        filenames.append(str(filename.resolve()))

    yaw = float(
        math.atan2(lidar2global[1, 0], lidar2global[0, 0])
    )
    if yaw < 0:
        yaw += 2 * math.pi
    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[:3] = lidar2global[:3, 3]
    can_bus[3:7] = np.asarray(_quaternion_z(yaw), dtype=np.float32)
    can_bus[7] = float(annotation["speed"])
    can_bus[10:13] = np.asarray(
        annotation["acceleration"], dtype=np.float32
    )
    can_bus[13:16] = np.asarray(
        annotation["angular_velocity"], dtype=np.float32
    )
    can_bus[16] = yaw
    can_bus[17] = math.degrees(yaw)

    raw_command = int(
        annotation.get("command_near", measurement.get("command", 4))
    )
    if raw_command < 1 or raw_command > 6:
        raw_command = 4
    command = raw_command - 1
    ego_fut_cmd = np.zeros(6, dtype=np.float32)
    ego_fut_cmd[command] = 1.0
    stacked_shape = np.stack(images, axis=-1).shape
    return {
        "folder": frame_root.name,
        "scene_token": frame_root.name,
        "frame_idx": int(frame),
        "timestamp": int(frame) / 20.0,
        "img": images,
        "filename": filenames,
        "lidar2img": lidar2img,
        "lidar2cam": lidar2cam,
        "cam_intrinsic": intrinsics,
        "lidar2ego": lidar2ego,
        "world2lidar": world2lidar,
        "ego_pose": lidar2global,
        "ego_pose_inv": world2lidar,
        "l2g_r_mat": lidar2global[:3, :3],
        "l2g_t": lidar2global[:3, 3],
        "can_bus": can_bus,
        "command": command,
        "ego_fut_cmd": ego_fut_cmd,
        "img_shape": stacked_shape,
        "ori_shape": stacked_shape,
        "pad_shape": stacked_shape,
        "input_provenance": {
            "annotation": str(annotation_path.resolve()),
            "annotation_speed": float(annotation["speed"]),
            "raw_route_command": raw_command,
            "route_command_index": command,
            "measurement_target_point": measurement.get("target_point"),
        },
    }


def _replace_tokenizer_paths(
    pipeline: list[dict[str, Any]],
    vlm_root: Path,
) -> list[dict[str, Any]]:
    result = []
    for declaration in pipeline:
        updated = dict(declaration)
        if "tokenizer" in updated:
            updated["tokenizer"] = str(vlm_root)
        transforms = updated.get("transforms")
        if isinstance(transforms, list):
            updated["transforms"] = _replace_tokenizer_paths(
                transforms, vlm_root
            )
        result.append(updated)
    return result


def _move_batch_to_device(batch: dict[str, Any], device: str) -> None:
    import torch

    for key, value in batch.items():
        if key == "img_metas":
            continue
        if isinstance(value, list) and value and torch.is_tensor(value[0]):
            value[0] = value[0].to(device)
        if key == "input_ids" and isinstance(value, list):
            for outer in value[0]:
                for index, item in enumerate(outer):
                    if torch.is_tensor(item):
                        outer[index] = item.to(device)


def _flatten_preprocessed_inputs(
    batch: dict[str, Any],
) -> dict[str, Any]:
    return {
        "camera_images": batch["img"][0].detach().cpu(),
        "decision_input_ids": batch["input_ids"][0][0][0]
        .detach()
        .cpu(),
        "planning_input_ids": batch["input_ids"][0][0][1]
        .detach()
        .cpu(),
        "ego_route_command": batch["ego_fut_cmd"][0].detach().cpu(),
        "can_bus": batch["can_bus"][0].detach().cpu(),
        "lidar2img": batch["lidar2img"][0].detach().cpu(),
        "camera_intrinsics": batch["cam_intrinsic"][0].detach().cpu(),
        "timestamp": batch["timestamp"][0].detach().cpu(),
        "ego_pose": batch["ego_pose"][0].detach().cpu(),
        "ego_pose_inverse": batch["ego_pose_inv"][0].detach().cpu(),
        "route_command_index": batch["command"][0].detach().cpu(),
    }


def _input_records(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        name: _tensor_record(tensor) for name, tensor in inputs.items()
    }


def _extract_persistent_state(model: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    heads = (
        ("detection", model.pts_bbox_head),
        ("map", model.map_head),
    )
    for prefix, head in heads:
        limits = {
            "memory_embedding": head.memory_len,
            "memory_reference_point": head.memory_len,
            "memory_timestamp": head.memory_len,
            "memory_egopose": head.memory_len,
            "memory_velo": head.memory_len,
            "memory_mask": head.memory_len,
            "memory_canbus": getattr(head, "can_bus_len", None),
            "memory_scene_query": getattr(
                head, "scence_memory_len", None
            ),
            "scene_memory_timestamp": getattr(
                head, "scence_memory_len", None
            ),
        }
        for name in (
            "memory_embedding",
            "memory_reference_point",
            "memory_timestamp",
            "memory_egopose",
            "memory_velo",
            "sample_time",
            "memory_mask",
            "memory_canbus",
            "his_memory_canbus_len",
            "memory_scene_query",
            "scene_memory_timestamp",
            "his_state_counter",
        ):
            value = getattr(head, name, None)
            if value is None:
                continue
            limit = limits.get(name)
            if limit is not None and value.ndim >= 2:
                value = value[:, : int(limit)]
            state[f"{prefix}.{name}"] = value.detach().cpu()
    return state


def _reset_model_state(model: Any) -> None:
    if getattr(model, "with_pts_bbox", False):
        model.pts_bbox_head.reset_memory()
    if getattr(model, "with_map_head", False):
        model.map_head.reset_memory()
    model.test_flag = True


def _tensor_record(value: Any) -> dict[str, Any]:
    import torch

    tensor = value.detach()
    floating = tensor if tensor.is_floating_point() else tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": bool(torch.isfinite(floating).all()),
        "minimum": float(floating.min().item()) if tensor.numel() else None,
        "maximum": float(floating.max().item()) if tensor.numel() else None,
        "mean": float(floating.mean().item()) if tensor.numel() else None,
    }


def _detach_tensor_tree(value: Any) -> Any:
    """Keep only tensor-bearing parts of an upstream intermediate tree."""

    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {
            str(key): converted
            for key, item in value.items()
            if (converted := _detach_tensor_tree(item)) is not None
        }
    if isinstance(value, (tuple, list)):
        converted = [
            item
            for value_item in value
            if (item := _detach_tensor_tree(value_item)) is not None
        ]
        return tuple(converted) if isinstance(value, tuple) else converted
    return None


def _collect_named_outputs(result: dict[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    import torch

    pts = result["pts_bbox"]
    tensors: dict[str, Any] = {}
    for name in (
        "ego_fut_preds",
        "pw_ego_fut_pred",
        "scores_3d",
        "labels_3d",
        "trajs_3d",
    ):
        value = pts.get(name)
        if torch.is_tensor(value):
            tensors[name] = value
    boxes = pts.get("boxes_3d")
    if boxes is not None and torch.is_tensor(getattr(boxes, "tensor", None)):
        tensors["boxes_3d"] = boxes.tensor
    scalars = {
        name: pts[name]
        for name in ("speed_value", "path_value")
        if name in pts
    }
    records = {name: _tensor_record(value) for name, value in tensors.items()}
    records.update(
        {
            name: {
                "type": type(value).__name__,
                "value": int(value)
                if isinstance(value, (int, bool))
                else float(value),
            }
            for name, value in scalars.items()
        }
    )
    return tensors, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--frame", default="00400")
    parser.add_argument(
        "--warmup-frame",
        action="append",
        default=[],
        help=(
            "run this frame before --frame without resetting memory; repeat "
            "the option to reconstruct a longer stateful sequence"
        ),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-tensors", type=Path)
    parser.add_argument("--image-features", type=Path)
    parser.add_argument("--preprocessed-inputs", type=Path)
    parser.add_argument("--persistent-state", type=Path)
    parser.add_argument("--persistent-state-before-run", type=Path)
    parser.add_argument(
        "--intermediates",
        type=Path,
        help="save tensor-only source boundaries for Region adaptation",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16"), default="fp32"
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="stop after strict checkpoint load",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    release_root = args.release_root.resolve()
    checkpoint = release_root / "minddrive_rltrain.pth"
    vlm_root = release_root / "llava-qwen2-0.5b"
    vlm_weights = vlm_root / "model.safetensors"
    revision = _source_revision(source_root)
    if revision != MINDDRIVE_REVISION:
        raise ValueError(
            f"MindDrive revision mismatch: {revision} != {MINDDRIVE_REVISION}"
        )
    if checkpoint.stat().st_size != MINDDRIVE_CHECKPOINT_SIZE:
        raise ValueError("MindDrive checkpoint size mismatch")
    if _sha256(checkpoint) != MINDDRIVE_CHECKPOINT_SHA256:
        raise ValueError("MindDrive checkpoint SHA256 mismatch")
    if vlm_weights.stat().st_size != MINDDRIVE_VLM_SIZE:
        raise ValueError("MindDrive VLM size mismatch")
    if _sha256(vlm_weights) != MINDDRIVE_VLM_SHA256:
        raise ValueError("MindDrive VLM SHA256 mismatch")

    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import numpy as np
    import torch
    from transformers.utils import logging as transformers_logging
    from mmcv import Config
    from mmcv.datasets.pipelines import Compose
    from mmcv.models import build_model
    from mmcv.parallel.collate import collate

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    transformers_logging.set_verbosity_error()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("MindDrive CUDA probe requires CUDA")

    config_path = (
        source_root
        / "adzoo"
        / "minddrive"
        / "configs"
        / "minddrive_qwen2_05B_infer.py"
    )
    config = Config.fromfile(str(config_path))
    config.model.tokenizer = str(vlm_root)
    config.model.lm_head = str(vlm_root)
    if args.precision == "fp16":
        config.model.fp32_infer = False
        config.model.fp16_infer = True

    host_load_started = time.perf_counter()
    model = build_model(
        config.model,
        train_cfg=config.get("train_cfg"),
        test_cfg=config.get("test_cfg"),
    )
    checkpoint_payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = checkpoint_payload["state_dict"]
    model_keys = set(model.state_dict())
    checkpoint_keys = set(state)
    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    allowed_training_extras = [
        key for key in unexpected_keys if _is_allowed_training_extra(key)
    ]
    disallowed_unexpected_keys = sorted(
        set(unexpected_keys) - set(allowed_training_extras)
    )
    projected_state = {
        key: value for key, value in state.items() if key in model_keys
    }
    if not missing_keys and not disallowed_unexpected_keys:
        model.load_state_dict(projected_state, strict=True)
    host_load_seconds = time.perf_counter() - host_load_started
    report: dict[str, Any] = {
        "schema": "vlaforge.minddrive_real_eager_probe/1",
        "status": "load-audited",
        "passed": False,
        "evidence_kind": "real-checkpoint-upstream-eager-probe",
        "evidence_level": "L2-prerequisite-only",
        "claim_boundary": (
            "A passing report proves an exact inference-state projection and "
            "an upstream eager reference only; Semantic IR/Plan parity is "
            "still required for L2. Training-only value-network tensors and "
            "non-persistent rotary buffers are audited but not deployed."
        ),
        "model": "MindDrive 0.5B",
        "upstream_revision": revision,
        "checkpoint_revision": MINDDRIVE_CHECKPOINT_REVISION,
        "checkpoint_sha256": MINDDRIVE_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "precision": args.precision,
        "device": args.device,
        "seed": args.seed,
        "load_audit": {
            "policy": "exact-inference-projection-v1",
            "model_state_key_count": len(model_keys),
            "checkpoint_state_key_count": len(state),
            "matched_key_count": len(projected_state),
            "missing": _key_audit(missing_keys),
            "unexpected": _key_audit(unexpected_keys),
            "allowed_training_extras": _key_audit(
                allowed_training_extras
            ),
            "disallowed_unexpected": _key_audit(
                disallowed_unexpected_keys
            ),
        },
        "timing": {"host_model_and_checkpoint_load_seconds": host_load_seconds},
        "memory": {
            "host_peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            )
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if missing_keys or disallowed_unexpected_keys:
        raise ValueError(
            "MindDrive checkpoint/model mismatch: "
            f"missing={len(missing_keys)} "
            f"(sha256={report['load_audit']['missing']['sha256']}), "
            f"disallowed_unexpected={len(disallowed_unexpected_keys)} "
            "(sha256="
            f"{report['load_audit']['disallowed_unexpected']['sha256']}); "
            f"see {args.report}"
        )
    report["status"] = "strict-inference-projection-passed"
    del state
    del projected_state
    del checkpoint_payload
    if args.load_only:
        report["passed"] = True
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        return 0

    pipeline = Compose(
        _replace_tokenizer_paths(
            list(config.inference_only_pipeline), vlm_root
        )
    )
    raw = _build_raw_invocation(
        frame_root=args.frame_root.resolve(),
        frame=args.frame,
    )
    input_provenance = raw.pop("input_provenance")
    prepared = pipeline(raw)
    batch = collate([prepared], samples_per_gpu=1)
    flat_inputs = _flatten_preprocessed_inputs(batch)
    _move_batch_to_device(batch, args.device)
    warmup_batches = []
    for warmup_frame in args.warmup_frame:
        warmup_raw = _build_raw_invocation(
            frame_root=args.frame_root.resolve(),
            frame=warmup_frame,
        )
        warmup_raw.pop("input_provenance")
        warmup_prepared = pipeline(warmup_raw)
        warmup_batch = collate(
            [warmup_prepared], samples_per_gpu=1
        )
        _move_batch_to_device(warmup_batch, args.device)
        warmup_batches.append(warmup_batch)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_load_started = time.perf_counter()
    model = model.eval().to(args.device)
    torch.cuda.synchronize()
    cuda_load_seconds = time.perf_counter() - cuda_load_started
    cuda_after_load = int(torch.cuda.memory_allocated())

    _reset_model_state(model)
    persistent_state_before_run = None
    if warmup_batches:
        with torch.inference_mode():
            for warmup_batch in warmup_batches:
                model(warmup_batch, return_loss=False)
        persistent_state_before_run = _extract_persistent_state(model)
    captured_image_features: dict[str, Any] = {}
    captured_intermediates: dict[str, Any] = {}
    original_extract_feat = model.extract_feat
    original_prepare_location = model.prepare_location
    original_position_embedding = model.position_embeding
    original_decision = model.lm_head.inference_action_distribution
    original_action = model.lm_head.inference_waypoints
    lm_implementation = original_decision.__self__
    original_multimodal = (
        lm_implementation.prepare_inputs_labels_for_multimodal
    )
    multimodal_call_index = 0

    def capture_image_features(image: Any) -> Any:
        features = original_extract_feat(image)
        captured_image_features["value"] = features.detach().cpu()
        return features

    def capture_prepare_location(*capture_args: Any, **capture_kwargs: Any) -> Any:
        value = original_prepare_location(*capture_args, **capture_kwargs)
        captured_intermediates["location"] = value.detach().cpu()
        return value

    def capture_position_embedding(
        *capture_args: Any, **capture_kwargs: Any
    ) -> Any:
        value = original_position_embedding(*capture_args, **capture_kwargs)
        captured_intermediates["position_embedding"] = (
            value.detach().cpu()
        )
        return value

    def capture_decision(*capture_args: Any, **capture_kwargs: Any) -> Any:
        value = original_decision(*capture_args, **capture_kwargs)
        captured_intermediates["decision_expert"] = _detach_tensor_tree(
            value
        )
        return value

    def capture_action(*capture_args: Any, **capture_kwargs: Any) -> Any:
        value = original_action(*capture_args, **capture_kwargs)
        captured_intermediates["action_expert"] = _detach_tensor_tree(value)
        return value

    def capture_multimodal(
        *capture_args: Any, **capture_kwargs: Any
    ) -> Any:
        nonlocal multimodal_call_index
        value = original_multimodal(*capture_args, **capture_kwargs)
        captured_intermediates[
            f"multimodal_preparation_{multimodal_call_index}"
        ] = _detach_tensor_tree(value)
        multimodal_call_index += 1
        return value

    def capture_map_head(
        _module: Any, _inputs: Any, output: Any
    ) -> None:
        captured_intermediates["map_head"] = _detach_tensor_tree(output)

    def capture_detection_head(
        _module: Any, _inputs: Any, output: Any
    ) -> None:
        captured_intermediates["detection_head"] = _detach_tensor_tree(
            output
        )

    hooks = [
        model.map_head.register_forward_hook(capture_map_head),
        model.pts_bbox_head.register_forward_hook(capture_detection_head),
    ]
    model.extract_feat = capture_image_features
    model.prepare_location = capture_prepare_location
    model.position_embeding = capture_position_embedding
    model.lm_head.inference_action_distribution = capture_decision
    model.lm_head.inference_waypoints = capture_action
    lm_implementation.prepare_inputs_labels_for_multimodal = (
        capture_multimodal
    )
    original_randn_like = torch.randn_like
    planner_noises: list[Any] = []

    def capture_randn_like(
        *capture_args: Any, **capture_kwargs: Any
    ) -> Any:
        value = original_randn_like(*capture_args, **capture_kwargs)
        planner_noises.append(value.detach().cpu())
        return value

    torch.randn_like = capture_randn_like
    eager_started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model(batch, return_loss=False)
    finally:
        torch.randn_like = original_randn_like
        model.extract_feat = original_extract_feat
        model.prepare_location = original_prepare_location
        model.position_embeding = original_position_embedding
        model.lm_head.inference_action_distribution = original_decision
        model.lm_head.inference_waypoints = original_action
        lm_implementation.prepare_inputs_labels_for_multimodal = (
            original_multimodal
        )
        for hook in hooks:
            hook.remove()
    captured_intermediates["planner_noises"] = tuple(planner_noises)
    if len(planner_noises) != 2:
        raise ValueError(
            "MindDrive eager run did not produce exactly two planner noises"
        )
    flat_inputs["trajectory_noise"] = planner_noises[0]
    flat_inputs["path_noise"] = planner_noises[1]
    if args.preprocessed_inputs is not None:
        args.preprocessed_inputs.parent.mkdir(parents=True, exist_ok=True)
        torch.save(flat_inputs, args.preprocessed_inputs)
    torch.cuda.synchronize()
    eager_seconds = time.perf_counter() - eager_started
    if not isinstance(output, list) or len(output) != 1:
        raise ValueError("MindDrive eager output is not a batch-one result")
    tensors, output_records = _collect_named_outputs(output[0])
    persistent_state = _extract_persistent_state(model)
    required = {"ego_fut_preds", "pw_ego_fut_pred"}
    if not required.issubset(tensors):
        raise ValueError(
            f"MindDrive eager output missing required tensors: "
            f"{sorted(required - tensors.keys())}"
        )
    if not all(record.get("finite", True) for record in output_records.values()):
        raise ValueError("MindDrive eager output contains non-finite values")

    report.update(
        {
            "status": "passed",
            "passed": True,
            "input": {
                "frame_root": str(args.frame_root.resolve()),
                "frame": args.frame,
                "warmup_frame": args.warmup_frame,
                "camera_count": 6,
                "raw_camera_shape_hwc": [900, 1600, 3],
                "provenance": input_provenance,
                "preprocessed": _input_records(flat_inputs),
            },
            "outputs": output_records,
            "persistent_state": _input_records(persistent_state),
            "timing": {
                **report["timing"],
                "cuda_model_load_seconds": cuda_load_seconds,
                "eager_seconds": eager_seconds,
            },
            "memory": {
                **report["memory"],
                "cuda_allocated_after_load_bytes": cuda_after_load,
                "cuda_peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "cuda_peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved()
                ),
            },
        }
    )
    if args.output_tensors is not None:
        args.output_tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {name: value.detach().cpu() for name, value in tensors.items()},
            args.output_tensors,
        )
        report["output_tensors"] = str(args.output_tensors.resolve())
        report["output_tensors_sha256"] = _sha256(args.output_tensors)
    if args.image_features is not None:
        if "value" not in captured_image_features:
            raise ValueError("MindDrive eager run captured no image features")
        args.image_features.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"image_features": captured_image_features["value"]},
            args.image_features,
        )
        report["image_features"] = {
            **_tensor_record(captured_image_features["value"]),
            "path": str(args.image_features.resolve()),
            "sha256": _sha256(args.image_features),
        }
    if args.preprocessed_inputs is not None:
        report["preprocessed_inputs"] = str(
            args.preprocessed_inputs.resolve()
        )
        report["preprocessed_inputs_sha256"] = _sha256(
            args.preprocessed_inputs
        )
    if args.persistent_state is not None:
        args.persistent_state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(persistent_state, args.persistent_state)
        report["persistent_state_artifact"] = str(
            args.persistent_state.resolve()
        )
        report["persistent_state_sha256"] = _sha256(
            args.persistent_state
        )
    if args.persistent_state_before_run is not None:
        if persistent_state_before_run is None:
            raise ValueError(
                "--persistent-state-before-run requires --warmup-frame"
            )
        args.persistent_state_before_run.parent.mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            persistent_state_before_run,
            args.persistent_state_before_run,
        )
        report["persistent_state_before_run"] = str(
            args.persistent_state_before_run.resolve()
        )
        report["persistent_state_before_run_sha256"] = _sha256(
            args.persistent_state_before_run
        )
    if args.intermediates is not None:
        args.intermediates.parent.mkdir(parents=True, exist_ok=True)
        torch.save(captured_intermediates, args.intermediates)
        report["intermediates"] = {
            "path": str(args.intermediates.resolve()),
            "sha256": _sha256(args.intermediates),
            "keys": sorted(captured_intermediates),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
