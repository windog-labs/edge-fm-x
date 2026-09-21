#!/usr/bin/env python3
"""Export a validated TorchScript tensor region to ONNX with parity evidence.

The tool is model-agnostic. It accepts a TorchScript archive and a tuple of
runtime inputs, exports a fixed-shape ONNX graph, checks the ONNX model, and
records TorchScript/ONNX Runtime parity when the installed runtime supports
the emitted operators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_metadata(value) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def tensor_outputs(value):
    import torch

    values = (value,) if isinstance(value, torch.Tensor) else tuple(value)
    if not values or any(not isinstance(item, torch.Tensor) for item in values):
        raise ValueError("TorchScript region must return a flat tuple of tensors")
    return values


def numpy_outputs(value) -> tuple:
    import numpy as np

    if isinstance(value, np.ndarray):
        return (value,)
    return tuple(value)


def tensor_metrics(reference, actual) -> dict[str, Any]:
    import numpy as np

    reference = np.asarray(reference)
    actual = np.asarray(actual)
    if reference.shape != actual.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    if reference.dtype == np.bool_ or actual.dtype == np.bool_:
        delta = np.logical_xor(reference, actual)
        return {
            "shape_match": True,
            "exact": bool(np.array_equal(reference, actual)),
            "mismatch_count": int(np.count_nonzero(delta)),
        }
    left = reference.astype(np.float64)
    right = actual.astype(np.float64)
    delta = right - left
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return {
        "shape_match": True,
        "exact": bool(np.array_equal(reference, actual)),
        "maximum_absolute_error": float(np.max(np.abs(delta))),
        "mean_squared_error": float(np.mean(delta * delta)),
        "root_mean_squared_error": float(np.sqrt(np.mean(delta * delta))),
        "cosine_similarity": (
            float(np.vdot(right, left) / denominator)
            if denominator
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="Disable the legacy exporter constant-folding pass.",
    )
    parser.add_argument(
        "--skip-runtime-validation",
        action="store_true",
        help="Keep ONNX checker evidence without claiming runtime parity.",
    )
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    torchscript = args.torchscript.resolve(strict=True)
    inputs_path = args.inputs.resolve(strict=True)
    output = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output.with_suffix(".report.json")
    )
    if output.exists():
        raise FileExistsError(output)
    if report_path.exists():
        raise FileExistsError(report_path)
    if args.opset < 17:
        raise ValueError("opset must be at least 17")

    started_ns = time.time_ns()
    module = torch.jit.load(str(torchscript), map_location=args.device).to(args.device)
    examples = torch.load(
        inputs_path,
        map_location=args.device,
        weights_only=True,
    )
    if not isinstance(examples, tuple) or not examples:
        raise ValueError("runtime inputs must be a non-empty tuple")
    if any(not isinstance(value, torch.Tensor) for value in examples):
        raise ValueError("runtime inputs must contain only tensors")
    examples = tuple(value.to(args.device).contiguous() for value in examples)

    with torch.inference_mode():
        reference = tensor_outputs(module(*examples))

    input_names = [f"input_{index}" for index in range(len(examples))]
    output_names = [f"output_{index}" for index in range(len(reference))]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".torchscript-onnx-", dir=output.parent)
    )
    staged = temporary_root / output.name
    report: dict[str, Any] = {
        "schema": "vlaforge.torchscript_onnx_export/1",
        "status": "started",
        "torchscript": str(torchscript),
        "torchscript_sha256": digest(torchscript),
        "inputs": str(inputs_path),
        "inputs_sha256": digest(inputs_path),
        "output": str(output),
        "opset": args.opset,
        "dynamo": False,
        "device": args.device,
        "torch_version": str(torch.__version__),
        "onnx_version": str(onnx.__version__),
        "onnxruntime_version": str(ort.__version__),
        "input_names": input_names,
        "output_names": output_names,
        "input_contract": [tensor_metadata(value) for value in examples],
        "output_contract": [tensor_metadata(value) for value in reference],
        "started_ns": started_ns,
    }
    try:
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            torch.onnx.export(
                module,
                examples,
                str(staged),
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=not args.no_constant_folding,
                input_names=input_names,
                output_names=output_names,
                training=torch.onnx.TrainingMode.EVAL,
                dynamo=False,
                external_data=False,
            )
        model = onnx.load(str(staged), load_external_data=True)
        onnx.checker.check_model(model)
        onnx_input_names = [value.name for value in model.graph.input]
        onnx_output_names = [value.name for value in model.graph.output]
        report["onnx_checker"] = "passed"
        report["onnx_input_names"] = onnx_input_names
        report["onnx_output_names"] = onnx_output_names
        report["onnx_nodes"] = len(model.graph.node)
        report["onnx_initializers"] = len(model.graph.initializer)
        report["onnx_op_types"] = sorted(
            {node.op_type for node in model.graph.node}
        )

        if args.skip_runtime_validation:
            report["runtime_parity"] = {
                "status": "skipped",
                "reason": "explicit --skip-runtime-validation",
            }
        else:
            session = ort.InferenceSession(
                str(staged),
                providers=["CPUExecutionProvider"],
            )
            if len(onnx_input_names) not in (0, len(examples)):
                raise ValueError(
                    "ONNX input count differs from the TorchScript runtime "
                    f"contract: {len(onnx_input_names)} != {len(examples)}"
                )
            feeds = (
                {
                    name: value.detach().cpu().numpy()
                    for name, value in zip(
                        onnx_input_names,
                        examples,
                        strict=True,
                    )
                }
                if onnx_input_names
                else {}
            )
            actual = numpy_outputs(session.run(None, feeds))
            expected = numpy_outputs(
                tuple(value.detach().cpu().numpy() for value in reference)
            )
            parity = [
                tensor_metrics(left, right)
                for left, right in zip(expected, actual, strict=True)
            ]
            report["runtime_parity"] = {
                "status": "passed",
                "outputs": parity,
                "all_shapes_match": all(item["shape_match"] for item in parity),
                "constant_folded_inputs": len(examples) - len(onnx_input_names),
            }

        os.link(staged, output)
        report.update(
            {
                "status": "passed",
                "output_sha256": digest(output),
                "output_size": output.stat().st_size,
                "elapsed_ns": time.time_ns() - started_ns,
            }
        )
        write_json(report_path, report)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
                "elapsed_ns": time.time_ns() - started_ns,
            }
        )
        write_json(report_path, report)
        if output.exists():
            output.unlink()
        raise
    finally:
        for path in sorted(temporary_root.glob("**/*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        temporary_root.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
