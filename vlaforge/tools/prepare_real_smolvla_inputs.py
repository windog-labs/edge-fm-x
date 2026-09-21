#!/usr/bin/env python3
"""Materialize real LeRobot observations with checkpoint-owned preprocessing.

This input pack contains no deployment or accuracy claim. Reference actions
must be generated separately from the exact saved noise and processed inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path, *, git_blob: bool = False) -> str:
    result = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        result.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_dataset_source(
    root: Path, info_path: Path, tree_path: Path, revision: str
) -> dict:
    info = json.loads(info_path.read_text())
    tree = json.loads(tree_path.read_text())
    if info.get("sha") != revision:
        raise ValueError("dataset revision does not match the saved Hub metadata")
    files = [item for item in tree if item.get("type") == "file"]
    names = {item["path"] for item in files}
    if len(names) != len(files) or names != {
        item["rfilename"] for item in info["siblings"]
    }:
        raise ValueError("dataset tree is duplicated or incomplete")
    result = {}
    for item in files:
        path = (root / item["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(root.resolve())
            or path.stat().st_size != item["size"]
        ):
            raise ValueError(f"dataset path or size mismatch: {item['path']}")
        if "lfs" in item:
            actual = digest(path)
            expected = item["lfs"]["oid"]
        else:
            actual = digest(path, git_blob=True)
            expected = item["oid"]
        if actual != expected:
            raise ValueError(f"dataset object digest mismatch: {item['path']}")
        result[item["path"]] = digest(path)
    return result


def camera_mapping(entries: list[str]) -> dict[str, str]:
    result = {}
    for entry in entries:
        source, separator, destination = entry.partition("=")
        if not separator or not source or not destination:
            raise ValueError("camera mappings must be SOURCE=DESTINATION")
        if source in result or destination in result.values():
            raise ValueError("camera mapping must be one-to-one")
        result[source] = destination
    if not result:
        raise ValueError("at least one explicit real-camera mapping is required")
    return result


def prepare_statistics(
    policy_path: Path, *, mode: str, namespace: str | None, robot_type: str
):
    """A published processor is not evidence that its statistics were consumed."""
    from vlaforge.adapters.smolvla.smolvla_processing import prepare_smolvla_statistics

    return prepare_smolvla_statistics(policy_path, mode=mode, namespace=namespace, robot_type=robot_type)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-info", type=Path, required=True)
    parser.add_argument("--dataset-tree", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--processing-mode",
        choices=("strict-statistics", "published-compatibility"),
        default="strict-statistics",
    )
    parser.add_argument("--statistics-namespace")
    args = parser.parse_args()
    if args.count < 1 or args.stride < 1 or args.start < 0:
        raise ValueError("invalid observation interval")
    mapping = camera_mapping(args.camera)
    if args.output.exists():
        raise ValueError(
            "input pack output must be new; incomplete packs are not reusable"
        )
    source_files = verify_dataset_source(
        args.dataset_root, args.dataset_info, args.dataset_tree, args.dataset_revision
    )

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla import processor_smolvla  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import batch_to_transition, transition_to_batch

    config = PreTrainedConfig.from_pretrained(args.policy_path, local_files_only=True)
    if not set(mapping.values()).issubset(config.image_features):
        raise ValueError("mapped camera is absent from the checkpoint profile")
    dataset = LeRobotDataset(
        args.dataset_repo,
        root=args.dataset_root,
        revision=args.dataset_revision,
        video_backend="pyav",
    )
    indices = range(args.start, args.start + args.count * args.stride, args.stride)
    if indices[-1] >= len(dataset):
        raise ValueError("requested observations exceed dataset length")
    if not set(mapping).issubset(dataset.meta.camera_keys):
        raise ValueError("source camera is absent from the real dataset")
    from vlaforge.adapters.smolvla.smolvla_migration import verify_recovered_profile

    recovery = verify_recovered_profile(
        args.policy_path, robot_type=dataset.meta.info["robot_type"]
    )
    if recovery is not None and args.processing_mode != "strict-statistics":
        raise ValueError("a recovered profile is not the as-published processor")
    profile, normalization = prepare_statistics(
        args.policy_path,
        mode=args.processing_mode,
        namespace=args.statistics_namespace,
        robot_type=dataset.meta.info["robot_type"],
    )
    overrides = {
        "rename_observations_processor": {"rename_map": mapping},
        "tokenizer_processor": {"tokenizer_name": str(args.vlm_path.resolve())},
        "device_processor": {"device": "cpu"},
    }
    if profile is not None:
        overrides["normalizer_processor"] = {"stats": profile.stats}
    processor = PolicyProcessorPipeline.from_pretrained(
        str(args.policy_path),
        config_filename="policy_preprocessor.json",
        overrides=overrides,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    normalizer = None
    if profile is not None:
        from lerobot.processor import NormalizerProcessorStep

        normalizers = [
            step
            for step in processor.steps
            if isinstance(step, NormalizerProcessorStep)
        ]
        if len(normalizers) != 1:
            raise ValueError(
                "loaded processor changed the declared normalizer structure"
            )
        normalizer = normalizers[0]
        profile.require_consumed(normalizer)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    args.output.mkdir(parents=True)
    records = []
    for index in indices:
        raw = dataset[index]
        observation = {name: raw[name] for name in mapping}
        observation.update(
            {"observation.state": raw["observation.state"], "task": raw["task"]}
        )
        prepared = processor(observation)
        state_verification = None
        if profile is not None:
            state_verification = profile.verify_transform(
                "observation.state",
                raw["observation.state"].unsqueeze(0),
                prepared["observation.state"],
                inverse=False,
                eps=normalizer.eps,
            )
        values = {
            name: value.detach().cpu().contiguous().numpy()
            for name, value in prepared.items()
            if isinstance(value, torch.Tensor)
        }
        for name in mapping.values():
            values[f"{name}_padding_mask"] = np.ones((1,), dtype=np.bool_)
        values["noise"] = torch.randn(
            (1, config.chunk_size, config.max_action_dim),
            generator=generator,
            dtype=torch.float32,
        ).numpy()
        path = args.output / f"observation-{index:06d}.npz"
        np.savez_compressed(path, **values)
        records.append(
            {
                "dataset_index": index,
                "episode_index": int(raw["episode_index"]),
                "frame_index": int(raw["frame_index"]),
                "task": raw["task"],
                "path": path.name,
                "sha256": digest(path),
                "tensors": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in values.items()
                },
                "noise_sha256": hashlib.sha256(values["noise"].tobytes()).hexdigest(),
                "state_normalization": state_verification,
            }
        )
        print(f"saved index={index} path={path}", flush=True)
    if profile is not None:
        normalization["normalization_statistics_verified"] = True
    manifest = {
        "schema": "vlaforge.smolvla_observations/2",
        "status": "materialized_tensor_boundary_only",
        "dataset_repo": args.dataset_repo,
        "dataset_revision": args.dataset_revision,
        "dataset_files": source_files,
        "camera_mapping": mapping,
        "dataset_metadata_sha256": {
            "info": digest(args.dataset_info),
            "tree": digest(args.dataset_tree),
        },
        "missing_checkpoint_cameras": sorted(
            set(config.image_features) - set(mapping.values())
        ),
        "processor_overrides": {
            name: value
            for name, value in overrides.items()
            if name != "normalizer_processor"
        },
        "checkpoint_sha256": digest(args.policy_path / "model.safetensors"),
        "processor_files": {
            path.name: digest(path) for path in args.policy_path.glob("policy_*.*")
        },
        "seed": args.seed,
        "noise_semantics": "saved_float32_tensor_is_authoritative",
        "measurement_boundary": "checkpoint-processed-tensors; Python preprocessing excluded",
        "normalization": normalization,
        "statistics_recovery": recovery,
        "records": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
