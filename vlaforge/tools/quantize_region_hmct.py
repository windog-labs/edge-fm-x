#!/usr/bin/env python3
"""Quantize a fixed ONNX region from a verified region-input calibration pack."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import contextmanager
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


def load_calibration(
    root: Path,
    manifest: dict[str, Any],
    *,
    split: str,
    limit: int,
) -> dict[str, list[np.ndarray]]:
    try:
        record = manifest["splits"][split]
    except KeyError as error:
        raise ValueError(f"calibration manifest has no {split} split") from error
    path = root / record["path"]
    if digest(path) != record["sha256"]:
        raise ValueError(f"calibration pack changed: {path}")
    expected_names = set(manifest["input_names"])
    data: dict[str, list[np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as pack:
        if set(pack.files) != expected_names:
            raise ValueError("calibration pack input names do not match its manifest")
        sample_counts = {pack[name].shape[0] for name in pack.files}
        if len(sample_counts) != 1:
            raise ValueError("calibration inputs have inconsistent sample counts")
        count = next(iter(sample_counts))
        if limit:
            count = min(count, limit)
        if count < 1:
            raise ValueError("calibration split is empty")
        for name in manifest["input_names"]:
            data[name] = [
                np.ascontiguousarray(pack[name][index])
                for index in range(count)
            ]
    return data


_ONNX_ELEMENT_DTYPES = {
    1: np.dtype("float32"),
    7: np.dtype("int64"),
    9: np.dtype("bool"),
    11: np.dtype("float64"),
}


def graph_input_specs(onnx_path: Path) -> list[dict[str, Any]]:
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    specs: list[dict[str, Any]] = []
    for value in model.graph.input:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("elem_type"):
            raise ValueError(f"ONNX input has no element type: {value.name}")
        if not tensor_type.HasField("shape"):
            raise ValueError(f"ONNX input has no shape: {value.name}")
        shape = []
        for dimension in tensor_type.shape.dim:
            if dimension.HasField("dim_value"):
                shape.append(int(dimension.dim_value))
            elif dimension.dim_param:
                shape.append(dimension.dim_param)
            else:
                raise ValueError(f"ONNX input has an unknown dimension: {value.name}")
        specs.append(
            {
                "name": value.name,
                "element_type": int(tensor_type.elem_type),
                "shape": shape,
            }
        )
    if not specs:
        raise ValueError("ONNX graph has no runtime inputs")
    names = [spec["name"] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("ONNX graph input names must be unique")
    return specs


def map_calibration_inputs(
    data: dict[str, list[np.ndarray]],
    calibration_names: list[str],
    input_specs: list[dict[str, Any]],
) -> tuple[dict[str, list[np.ndarray]], dict[str, str]]:
    if len(calibration_names) != len(input_specs):
        raise ValueError(
            "calibration input count does not match the ONNX graph: "
            f"{len(calibration_names)} versus {len(input_specs)}"
        )
    if list(data) != calibration_names:
        raise ValueError("calibration data order does not match its manifest")

    graph_names = [spec["name"] for spec in input_specs]
    overlap = set(calibration_names) & set(graph_names)
    if overlap and overlap != set(calibration_names):
        raise ValueError("calibration and ONNX input names partially overlap")

    mapped: dict[str, list[np.ndarray]] = {}
    mapping: dict[str, str] = {}
    for calibration_name, specification in zip(
        calibration_names, input_specs, strict=True
    ):
        graph_name = specification["name"]
        samples = data[calibration_name]
        if not samples:
            raise ValueError(f"calibration input is empty: {calibration_name}")
        expected_dtype = _ONNX_ELEMENT_DTYPES.get(specification["element_type"])
        if expected_dtype is None:
            raise ValueError(
                f"unsupported ONNX input dtype {specification['element_type']} "
                f"for {graph_name}"
            )
        expected_shape = specification["shape"]
        for sample in samples:
            if sample.dtype != expected_dtype:
                raise ValueError(
                    f"calibration dtype for {calibration_name} does not match "
                    f"{graph_name}: {sample.dtype} versus {expected_dtype}"
                )
            if len(sample.shape) != len(expected_shape):
                raise ValueError(
                    f"calibration rank for {calibration_name} does not match "
                    f"{graph_name}: {sample.shape} versus {expected_shape}"
                )
            for actual, expected in zip(sample.shape, expected_shape, strict=True):
                if isinstance(expected, int) and expected > 0 and actual != expected:
                    raise ValueError(
                        f"calibration shape for {calibration_name} does not match "
                        f"{graph_name}: {sample.shape} versus {expected_shape}"
                    )
        mapped[graph_name] = samples
        mapping[calibration_name] = graph_name
    return mapped, mapping


@contextmanager
def calibration_batch_size(batch_size: int | None):
    if batch_size is None:
        yield
        return

    from hmct.quantizer.calibrater.activation import base as activation_base

    original_executor = activation_base.ORTExecutor

    class FixedBatchExecutor(original_executor):
        def forward_with_batch(
            self,
            input_data,
            batch_size=1,  # noqa: ARG002
            output_names=None,
            progressbar=None,
        ):
            return super().forward_with_batch(
                input_data,
                batch_size=forced_batch_size,
                output_names=output_names,
                progressbar=progressbar,
            )

    forced_batch_size = batch_size
    activation_base.ORTExecutor = FixedBatchExecutor
    try:
        yield
    finally:
        activation_base.ORTExecutor = original_executor


def initializer_count(graph: Any) -> int | None:
    for attribute in ("initializers", "initializer"):
        initializers = getattr(graph, attribute, None)
        if initializers is not None:
            return len(initializers)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-split", default="calibration")
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
    parser.add_argument("--calibration-limit", type=int, default=0)
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=0,
        help=(
            "Force the HMCT calibration batch size. Zero keeps HMCT's automatic "
            "batch selection."
        ),
    )
    parser.add_argument("--optimization", action="append", default=None)
    parser.add_argument("--op-qtype", action="append", type=parse_op_qtype, default=[])
    args = parser.parse_args()

    root = args.calibration_root.resolve(strict=True)
    onnx = args.onnx.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ValueError("quantization output must be new")
    if args.calibration_limit < 0:
        raise ValueError("calibration limit must be non-negative")
    if args.calibration_batch_size < 0:
        raise ValueError("calibration batch size must be non-negative")
    manifest = load_json(root / "manifest.json")
    if manifest.get("schema") != "vlaforge.region_input_calibration/1":
        raise ValueError("unexpected region calibration manifest schema")
    calibration_names = manifest.get("input_names")
    if (
        not isinstance(calibration_names, list)
        or not calibration_names
        or any(not isinstance(name, str) or not name for name in calibration_names)
    ):
        raise ValueError("calibration manifest must declare named inputs")
    config = recipe(args.recipe, topk=args.topk)
    op_qtype_overrides = apply_op_qtypes(config, args.op_qtype)
    data = load_calibration(
        root,
        manifest,
        split=args.calibration_split,
        limit=args.calibration_limit,
    )
    input_specs = graph_input_specs(onnx)
    data, input_mapping = map_calibration_inputs(
        data,
        calibration_names,
        input_specs,
    )

    from hmct.builder import build_onnx
    from hmct.ir import save_model

    started = time.time_ns()
    output.mkdir(parents=True)
    quantized_path = output / f"{onnx.stem}-{args.recipe}.onnx"
    report = {
        "schema": "vlaforge.region_hmct_quantization/1",
        "status": "started",
        "source": str(onnx),
        "source_sha256": digest(onnx),
        "calibration_root": str(root),
        "calibration_manifest_sha256": digest(root / "manifest.json"),
        "calibration_split": args.calibration_split,
        "recipe": args.recipe,
        "quant_config": config,
        "march": args.march,
        "calibration_limit": args.calibration_limit or None,
        "calibration_batch_size": args.calibration_batch_size or None,
        "optimization": args.optimization,
        "op_qtype_overrides": op_qtype_overrides,
        "calibration_input_names": calibration_names,
        "onnx_input_names": [spec["name"] for spec in input_specs],
        "input_mapping": input_mapping,
        "calibration_samples": {
            name: len(samples) for name, samples in sorted(data.items())
        },
        "started_ns": started,
    }
    write_json(output / "report.json", report)
    try:
        with calibration_batch_size(args.calibration_batch_size or None):
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
