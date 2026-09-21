#!/usr/bin/env python3
"""Evaluate a quantized SmolVLA solver-step ONNX before HBDK compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    reference = reference.astype(np.float64)
    actual = actual.astype(np.float64)
    delta = actual - reference
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(actual))
    return {
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "mean_squared_error": float(np.mean(delta * delta)),
        "root_mean_squared_error": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": (
            float(np.vdot(actual, reference) / denominator)
            if denominator
            else None
        ),
        "bitwise_equal": np.array_equal(reference, actual),
    }


def output_values(outputs: dict[str, np.ndarray], sample_shape: tuple[int, ...]):
    matching = [
        (name, value)
        for name, value in outputs.items()
        if tuple(value.shape) == sample_shape
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one sample output with shape {sample_shape}, got {matching}"
        )
    step_values = [value for value in outputs.values() if value.size == 1]
    if len(step_values) != 1:
        raise ValueError("expected exactly one scalar step-index output")
    return matching[0][1], step_values[0].reshape(1)


def infer_series_correct(executor, step_paths: list[Path], sample_index: int):
    sample = None
    step_index = None
    states = []
    step_outputs = []
    for path in step_paths:
        with np.load(path, allow_pickle=False) as pack:
            inputs = {
                name: np.ascontiguousarray(pack[name][sample_index])
                for name in pack.files
            }
        if sample is not None:
            inputs["sample"] = sample
        if step_index is not None:
            inputs["step_index"] = step_index
        outputs = executor.inference(inputs)
        sample, step_index = output_values(outputs, tuple(inputs["sample"].shape))
        states.append(np.array(sample, copy=True))
        step_outputs.append(np.array(step_index, copy=True))
    return states, step_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    from hmct.executor import ORTExecutor
    from hmct.ir import load_model

    calibration_root = args.calibration_root.resolve(strict=True)
    manifest = load_json(calibration_root / "manifest.json")
    if manifest.get("schema") != "vlaforge.smolvla_step_calibration/1":
        raise ValueError("unexpected calibration manifest schema")
    step_records = manifest["splits"]["held-out"]["steps"]
    step_paths = [calibration_root / record["path"] for record in step_records]
    reference_path = (
        calibration_root
        / manifest["splits"]["held-out"]["final_action_chunks"]["path"]
    )
    with np.load(reference_path, allow_pickle=False) as pack:
        references = pack["action_chunk"]
    count = references.shape[0]
    if args.max_samples is not None:
        count = min(count, args.max_samples)
    if count < 1:
        raise ValueError("held-out evaluation requires at least one sample")

    original = ORTExecutor(load_model(str(args.original))).create_session()
    quantized = ORTExecutor(load_model(str(args.quantized))).create_session()
    samples = []
    for index in range(count):
        original_states, original_steps = infer_series_correct(
            original, step_paths, index
        )
        quantized_states, quantized_steps = infer_series_correct(
            quantized, step_paths, index
        )
        original_final = original_states[-1][..., : references.shape[-1]]
        quantized_final = quantized_states[-1][..., : references.shape[-1]]
        samples.append(
            {
                "held_out_index": index,
                "reference_original": metrics(references[index], original_final),
                "quantized_vs_reference": metrics(references[index], quantized_final),
                "quantized_vs_original": metrics(original_final, quantized_final),
                "per_step_quantized_vs_original": [
                    metrics(left, right)
                    for left, right in zip(
                        original_states, quantized_states, strict=True
                    )
                ],
                "step_index_exact": bool(
                    np.array_equal(
                        np.stack(original_steps), np.stack(quantized_steps)
                    )
                ),
            }
        )
        print(
            "evaluated "
            f"held-out={index} "
            f"cosine={samples[-1]['quantized_vs_reference']['cosine_similarity']:.9f} "
            f"mse={samples[-1]['quantized_vs_reference']['mean_squared_error']:.6g}",
            flush=True,
        )

    report = {
        "schema": "vlaforge.smolvla_step_onnx_evaluation/1",
        "status": "passed",
        "original": str(args.original.resolve()),
        "original_sha256": digest(args.original),
        "quantized": str(args.quantized.resolve()),
        "quantized_sha256": digest(args.quantized),
        "calibration_root": str(calibration_root),
        "held_out_samples": count,
        "all_step_indices_exact": all(item["step_index_exact"] for item in samples),
        "samples": samples,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
