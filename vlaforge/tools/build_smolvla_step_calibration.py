#!/usr/bin/env python3
"""Materialize real SmolVLA solver-step calibration and held-out samples.

The source handoff contains one observation per sample and the verified final
action chunk.  This tool replays the captured prefix and solver-step regions to
record the actual 35-input tuple at every solver step.  Calibration and
held-out observations are written to separate directories so they cannot be
silently pooled by a backend conversion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE / "python") not in sys.path:
    sys.path.insert(0, str(SOURCE / "python"))


_NUMPY_DTYPES = {
    "bool": np.bool_,
    "f32": np.float32,
    "float32": np.float32,
    "i64": np.int64,
    "int64": np.int64,
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _safe_member(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"handoff path escapes its root: {relative}")
    return path


def verify_handoff(root: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema") != "vlaforge.board_input_handoff/1"
        or manifest.get("scope") != "data-and-reference-only"
    ):
        raise ValueError("expected the verified SmolVLA data/reference handoff")
    files = manifest.get("files")
    samples = manifest.get("samples")
    if not isinstance(files, list) or not files:
        raise ValueError("handoff must declare its files")
    if not isinstance(samples, list) or not samples:
        raise ValueError("handoff must declare its samples")
    for entry in files:
        path = _safe_member(root, entry["path"])
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"handoff size changed: {entry['path']}")
        if digest(path) != entry["sha256"]:
            raise ValueError(f"handoff digest changed: {entry['path']}")


def verify_capture(root: Path, manifest: dict[str, Any]) -> None:
    if (
        manifest.get("schema") != "vlaforge.smolvla_fresh_capture/1"
        or manifest.get("status") != "captured"
    ):
        raise ValueError("expected a completed SmolVLA fresh capture")
    for region in manifest.get("regions", []):
        path = root / "exports" / f"{region['name']}.pt2e"
        if digest(path) != region["export_sha256"]:
            raise ValueError(f"captured region changed: {region['name']}")


def _dtype(specification: dict[str, Any]) -> np.dtype:
    try:
        return np.dtype(_NUMPY_DTYPES[specification["dtype"]])
    except KeyError as error:
        raise ValueError(f"unsupported handoff dtype: {specification['dtype']}") from error


def load_binary(path: Path, specification: dict[str, Any]) -> np.ndarray:
    dtype = _dtype(specification)
    value = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(specification["shape"], dtype=np.int64))
    if value.size != expected:
        raise ValueError(
            f"{path.name} contains {value.size} values, expected {expected}"
        )
    return value.reshape(specification["shape"])


def load_tensor(path: Path, specification: dict[str, Any], device: str):
    import torch

    value = load_binary(path, specification)
    return torch.from_numpy(value.copy()).to(device)


def region_name(capture: dict[str, Any], *, suffix: str) -> str:
    matches = [
        region["name"]
        for region in capture["regions"]
        if region["name"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one captured region ending in {suffix}")
    return matches[0]


def _check_example(
    arguments: tuple[Any, ...],
    expected: tuple[Any, ...],
    name: str,
    *,
    check_dtype: bool = True,
):
    if len(arguments) != len(expected):
        raise ValueError(f"{name} runtime example has the wrong argument count")
    for index, (actual, declared) in enumerate(zip(arguments, expected, strict=True)):
        if tuple(actual.shape) != tuple(declared.shape) or (
            check_dtype and actual.dtype != declared.dtype
        ):
            raise ValueError(f"{name} runtime example {index} changed profile")


def _action_metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = float(
        np.linalg.norm(actual.astype(np.float64))
        * np.linalg.norm(reference.astype(np.float64))
    )
    cosine = (
        float(np.vdot(actual.astype(np.float64), reference.astype(np.float64)) / denominator)
        if denominator
        else None
    )
    return {
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "mean_squared_error": float(np.mean(delta * delta)),
        "cosine_similarity": cosine,
        "bitwise_equal": np.array_equal(reference, actual),
    }


def build(args: argparse.Namespace) -> int:
    import torch

    handoff = args.handoff.resolve(strict=True)
    capture_root = args.capture_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("calibration output must be new")

    handoff_manifest = read_json(handoff / "manifest.json")
    verify_handoff(handoff, handoff_manifest)
    capture = read_json(capture_root / "capture.json")
    verify_capture(capture_root, capture)

    input_specs = {item["name"]: item for item in handoff_manifest["inputs"]}
    sample_input_order = tuple(
        name for name in input_specs if name != "noise"
    )
    if not sample_input_order or "noise" not in input_specs:
        raise ValueError("handoff does not expose the expected model inputs")

    prefix_name = region_name(capture, suffix="prefix_51")
    step_name = region_name(capture, suffix="_step")
    prefix_path = capture_root / "exports" / f"{prefix_name}.pt2e"
    step_path = capture_root / "exports" / f"{step_name}.pt2e"

    prefix = torch.export.load(prefix_path).module().to(args.device)
    step = torch.export.load(step_path).module().to(args.device)

    sample_id = int(handoff_manifest["samples"][0]["sample_id"])
    first = handoff_manifest["samples"][0]
    first_prefix_args = tuple(
        load_tensor(
            _safe_member(handoff, first["inputs"][name]),
            input_specs[name],
            args.device,
        )
        for name in sample_input_order
    )
    prefix_example = torch.load(
        capture_root / "exports" / f"{prefix_name}.runtime-inputs.pt",
        map_location=args.device,
        weights_only=True,
    )
    step_example = torch.load(
        capture_root / "exports" / f"{step_name}.runtime-inputs.pt",
        map_location=args.device,
        weights_only=True,
    )
    _check_example(first_prefix_args, tuple(prefix_example), "prefix")

    step_names = (
        "pad_masks",
        "sample",
        "step_index",
        *(f"flat_cache_{index}" for index in range(32)),
    )
    if len(step_example) != len(step_names):
        raise ValueError("captured solver step does not expose the expected 35 inputs")

    count = len(handoff_manifest["samples"])
    if not 1 <= args.calibration_count < count:
        raise ValueError("calibration count must leave at least one held-out sample")
    split_for_index = {
        index: "calibration" if index < args.calibration_count else "held-out"
        for index in range(count)
    }

    records: list[dict[str, Any]] = []
    step_values: dict[str, dict[int, dict[str, list[np.ndarray]]]] = {
        "calibration": {step_index: {} for step_index in range(args.steps)},
        "held-out": {step_index: {} for step_index in range(args.steps)},
    }
    final_references: dict[str, list[np.ndarray]] = {
        "calibration": [],
        "held-out": [],
    }

    for index, sample in enumerate(handoff_manifest["samples"]):
        split = split_for_index[index]
        prefix_args = tuple(
            load_tensor(
                _safe_member(handoff, sample["inputs"][name]),
                input_specs[name],
                args.device,
            )
            for name in sample_input_order
        )
        with torch.inference_mode():
            prefix_outputs = prefix(*prefix_args)
            sample_value = load_tensor(
                _safe_member(handoff, sample["inputs"]["noise"]),
                input_specs["noise"],
                args.device,
            )
            step_index = torch.zeros(
                (1,), dtype=torch.int64, device=args.device
            )
            prefix_cache = (
                prefix_outputs[0],
                *prefix_outputs[1:],
            )
            for solver_step in range(args.steps):
                # The captured step region keeps BF16 KV caches internally.
                # The Horizon ABI consumes FP32, so save the faithful FP32
                # representation while executing the native captured profile.
                arguments = (
                    prefix_cache[0],
                    sample_value,
                    step_index,
                    *(value.to(torch.float32) for value in prefix_cache[1:]),
                )
                _check_example(
                    arguments,
                    tuple(step_example),
                    "step",
                    check_dtype=False,
                )
                arrays = tuple(
                    value.detach().cpu().contiguous().numpy()
                    for value in arguments
                )
                for name, value in zip(step_names, arrays, strict=True):
                    step_values[split][solver_step].setdefault(name, []).append(value)
                sample_value, step_index = step(
                    prefix_cache[0],
                    sample_value,
                    step_index,
                    *prefix_cache[1:],
                )
            if int(step_index.item()) != args.steps:
                raise ValueError("solver did not advance through every declared step")

        reference = load_binary(
            _safe_member(
                handoff,
                sample["outputs"]["action_chunk"]["eager"]["storage"],
            ),
            handoff_manifest["outputs"][0],
        )
        actual = sample_value[..., : reference.shape[-1]].detach().cpu().numpy()
        metrics = _action_metrics(reference, actual)
        if not metrics["bitwise_equal"]:
            raise ValueError(
                f"captured solver replay differs from sample {sample_id + index}"
            )
        final_references[split].append(reference.copy())
        records.append(
            {
                "sample_id": sample["sample_id"],
                "split": split,
                "action_reference": metrics,
            }
        )
        print(
            f"replayed sample={sample['sample_id']} split={split}",
            flush=True,
        )

    output.mkdir(parents=True)
    split_evidence: dict[str, Any] = {}
    for split in ("calibration", "held-out"):
        split_root = output / split
        split_root.mkdir()
        step_records = []
        for solver_step in range(args.steps):
            path = split_root / f"step-{solver_step:02d}.npz"
            arrays = {
                name: np.stack(samples, axis=0)
                for name, samples in step_values[split][solver_step].items()
            }
            np.savez(path, **arrays)
            step_records.append(
                {
                    "step": solver_step,
                    "path": str(path.relative_to(output)),
                    "sha256": digest(path),
                    "sample_count": len(next(iter(arrays.values()))),
                    "inputs": {
                        name: {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                        }
                        for name, value in arrays.items()
                    },
                }
            )
        references_path = split_root / "final-action-chunks.npz"
        references = np.stack(final_references[split], axis=0)
        np.savez(references_path, action_chunk=references)
        split_evidence[split] = {
            "steps": step_records,
            "final_action_chunks": {
                "path": str(references_path.relative_to(output)),
                "sha256": digest(references_path),
                "shape": list(references.shape),
                "dtype": str(references.dtype),
            },
        }

    result = {
        "schema": "vlaforge.smolvla_step_calibration/1",
        "status": "materialized",
        "capture_root": str(capture_root),
        "capture_sha256": digest(capture_root / "capture.json"),
        "handoff_root": str(handoff),
        "handoff_manifest_sha256": digest(handoff / "manifest.json"),
        "prefix_region": prefix_name,
        "step_region": step_name,
        "step_count": args.steps,
        "step_input_names": list(step_names),
        "calibration_count": args.calibration_count,
        "held_out_count": count - args.calibration_count,
        "splits": split_evidence,
        "samples": records,
        "all_source_replay_bitwise_equal": True,
        "physical_action_units_verified": False,
        "full_paper_acceptance": False,
    }
    write_json(output / "manifest.json", result)
    print(json.dumps({"output": str(output), "splits": split_evidence}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--calibration-count", type=int, default=8)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("steps must be positive")
    try:
        return build(args)
    except Exception as error:
        if args.output.exists():
            write_json(
                args.output / "failure.json",
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
