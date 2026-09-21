#!/usr/bin/env python3
"""Quantize a fixed SmolVLA solver-step ONNX from step-indexed real samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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


def initializer_count(graph: Any) -> int | None:
    for attribute in ("initializers", "initializer"):
        initializers = getattr(graph, attribute, None)
        if initializers is not None:
            return len(initializers)
    return None


def sample_paths(root: Path, manifest: dict[str, Any], split: str) -> list[Path]:
    try:
        records = manifest["splits"][split]["steps"]
    except KeyError as error:
        raise ValueError(f"calibration manifest has no {split} split") from error
    paths = []
    for record in records:
        path = root / record["path"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"calibration step changed: {path}")
        paths.append(path)
    return paths


def calibration_data(
    root: Path, manifest: dict[str, Any], *, limit: int = 0
) -> dict[str, list[np.ndarray]]:
    paths = sample_paths(root, manifest, "calibration")
    values: dict[str, list[np.ndarray]] = {}
    expected_names: tuple[str, ...] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as pack:
            names = tuple(sorted(pack.files))
            if expected_names is None:
                expected_names = names
            elif names != expected_names:
                raise ValueError("calibration step inputs changed between steps")
            for name in names:
                value = pack[name]
                if value.ndim < 1:
                    raise ValueError(f"{name} must include a sample axis")
                samples = [np.ascontiguousarray(sample) for sample in value]
                values.setdefault(name, []).extend(samples)
        if limit:
            values = {
                name: samples[:limit]
                for name, samples in values.items()
            }
            if all(len(samples) >= limit for samples in values.values()):
                break
    if not values:
        raise ValueError("calibration split is empty")
    counts = {len(samples) for samples in values.values()}
    if len(counts) != 1:
        raise ValueError("calibration inputs have inconsistent sample counts")
    return values


def recipe(name: str, *, topk: int) -> dict[str, Any]:
    if name == "int8-calibrated":
        return {
            "model_config": {
                "all_node_type": "int8",
                "model_output_type": "float32",
            }
        }
    if name == "int16-calibrated":
        return {
            "model_config": {
                "all_node_type": "int16",
                "model_output_type": "float32",
            }
        }
    if name == "mixed-int8-int16":
        if topk < 1:
            raise ValueError("mixed precision requires a positive sensitivity topk")
        return {
            "model_config": {
                "all_node_type": "int8",
                "model_output_type": "float32",
                "search": {
                    "layer": {
                        "topk": topk,
                        "qtype": "int16",
                        "metric": "cosine-similarity",
                    }
                },
            }
        }
    if name == "float16-calibrated":
        return {
            "model_config": {
                "all_node_type": "float16",
                "model_output_type": "float32",
            }
        }
    if name == "float16-lut-fp32":
        return {
            "model_config": {
                "all_node_type": "float16",
                "model_output_type": "float32",
            },
            "op_config": {
                "Lut": {
                    "qtype": "float32",
                }
            },
        }
    raise ValueError(f"unknown recipe: {name}")


def parse_op_qtype(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("operator qtype must be OP=TYPE")
    op_type, qtype = (part.strip() for part in value.split("=", 1))
    if not op_type or not qtype:
        raise argparse.ArgumentTypeError("operator qtype must be OP=TYPE")
    return op_type, qtype


def apply_op_qtypes(
    config: dict[str, Any], overrides: list[tuple[str, str]]
) -> dict[str, dict[str, str]]:
    applied: dict[str, dict[str, str]] = {}
    op_config = config.setdefault("op_config", {})
    for op_type, qtype in overrides:
        op_config[op_type] = {"qtype": qtype}
        applied[op_type] = {"qtype": qtype}
    return applied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--march", default="nash-m")
    parser.add_argument(
        "--recipe",
        choices=(
            "int8-calibrated",
            "int16-calibrated",
            "mixed-int8-int16",
            "float16-calibrated",
            "float16-lut-fp32",
        ),
        required=True,
    )
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument(
        "--calibration-limit",
        type=int,
        default=0,
        help="Use at most this many samples across the calibration split.",
    )
    parser.add_argument(
        "--optimization",
        action="append",
        default=None,
        help=(
            "HMCT optimization directive, for example "
            "skip_fuse_to_hzswish. Repeat to provide multiple directives."
        ),
    )
    parser.add_argument(
        "--op-qtype",
        action="append",
        type=parse_op_qtype,
        default=[],
        metavar="OP=TYPE",
        help=(
            "Override the quantization type for an ONNX operator type. "
            "Repeat to override multiple operators."
        ),
    )
    args = parser.parse_args()

    root = args.calibration_root.resolve(strict=True)
    onnx = args.onnx.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("quantization output must be new")
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema") != "vlaforge.smolvla_step_calibration/1":
        raise ValueError("unexpected calibration manifest schema")
    config = recipe(args.recipe, topk=args.topk)
    op_qtype_overrides = apply_op_qtypes(config, args.op_qtype)
    if args.calibration_limit < 0:
        raise ValueError("calibration limit must be non-negative")
    data = calibration_data(root, manifest, limit=args.calibration_limit)

    from hmct.builder import build_onnx
    from hmct.ir import save_model

    started = time.time_ns()
    output.mkdir(parents=True)
    quantized_path = output / f"{onnx.stem}-{args.recipe}.onnx"
    report = {
        "schema": "vlaforge.smolvla_hmct_quantization/1",
        "status": "started",
        "source": str(onnx),
        "source_sha256": digest(onnx),
        "calibration_root": str(root),
        "calibration_manifest_sha256": digest(root / "manifest.json"),
        "recipe": args.recipe,
        "quant_config": config,
        "march": args.march,
        "calibration_limit": args.calibration_limit or None,
        "optimization": args.optimization,
        "op_qtype_overrides": op_qtype_overrides,
        "calibration_samples": {
            name: len(samples) for name, samples in sorted(data.items())
        },
        "started_ns": started,
    }
    write_json(output / "report.json", report)
    try:
        builder = build_onnx(
            str(onnx),
            march=args.march,
            cali_dict={
                "calibration_data": data,
                "calibration_type": "max",
            },
            quant_config=config,
            optimization=args.optimization,
            save_model=False,
            verbose=True,
            return_builder=True,
        )
        if builder.quantized_model is None:
            raise RuntimeError("HMCT did not produce a quantized model")
        save_model(builder.quantized_model, str(quantized_path))
        report.update(
            {
                "status": "passed",
                "quantized_onnx": str(quantized_path),
                "quantized_sha256": digest(quantized_path),
                "quantized_size": quantized_path.stat().st_size,
                "node_count": len(builder.quantized_model.graph.nodes),
                "initializer_count": initializer_count(builder.quantized_model.graph),
                "op_types": sorted(
                    {node.op_type for node in builder.quantized_model.graph.nodes}
                ),
                "elapsed_ns": time.time_ns() - started,
            }
        )
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_ns": time.time_ns() - started,
            }
        )
        write_json(output / "report.json", report)
        raise
    write_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
