#!/usr/bin/env python3
"""Capture the logical MindDrive map Region as physical backend Regions."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    load_real_minddrive_model,
    make_partitioned_minddrive_map_encoder,
)
from vlaforge.frontend import (
    capture_region,
    load_exported_region,
    save_exported_region,
)
from vlaforge.ir.program import TensorRegion, Value
from vlaforge.ir.types import TensorType


_DTYPES = {
    "torch.float16": "f16",
    "torch.float32": "f32",
    "torch.float64": "f64",
    "torch.int32": "i32",
    "torch.int64": "i64",
    "torch.bool": "bool",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple | list):
        return tuple(value)
    return (value,)


def _tensor_type(tensor: Any) -> TensorType:
    dtype = _DTYPES.get(str(tensor.dtype))
    if dtype is None:
        raise ValueError(f"unsupported physical Region dtype: {tensor.dtype}")
    return TensorType(
        tuple(int(item) for item in tensor.shape),
        dtype,
    )


def _tensor_contract(tensor: Any) -> dict[str, object]:
    return {
        "shape": [int(item) for item in tensor.shape],
        "dtype": _DTYPES[str(tensor.dtype)],
        "device": str(tensor.device),
        "strides": [int(item) for item in tensor.stride()],
        "storage_offset": int(tensor.storage_offset()),
        "contiguous": bool(tensor.is_contiguous()),
        "layout": "contiguous" if tensor.is_contiguous() else "strided",
    }


def _require_contiguous(
    name: str,
    role: str,
    values: tuple[Any, ...],
) -> None:
    violations = [
        {"index": index, **_tensor_contract(value)}
        for index, value in enumerate(values)
        if not value.is_contiguous()
    ]
    if violations:
        raise ValueError(
            f"{name}: {role} violates the required contiguous physical "
            f"Region ABI: {violations}"
        )


def _capture(
    output: Path,
    name: str,
    implementation: Any,
    inputs: tuple[Any, ...],
) -> tuple[tuple[Any, ...], dict[str, object]]:
    import torch

    with torch.no_grad():
        eager = _as_tuple(implementation(*inputs))
    _require_contiguous(name, "inputs", inputs)
    _require_contiguous(name, "outputs", eager)
    region = TensorRegion(
        name,
        tuple(
            Value(f"input_{index}", _tensor_type(value))
            for index, value in enumerate(inputs)
        ),
        tuple(_tensor_type(value) for value in eager),
        metadata={
            "backend_physical_partition": True,
            "logical_region": "map_encoder",
        },
    )
    started = time.perf_counter()
    capture = capture_region(
        region,
        implementation,
        inputs,
        strict=True,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    capture.require_supported()
    artifact = output / f"{name}.pt2e"
    evidence = output / f"{name}.capture.json"
    save_exported_region(
        capture,
        program_path=artifact,
        evidence_path=evidence,
    )
    record = {
        "name": name,
        "artifact": str(artifact.resolve()),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": _sha256(artifact),
        "capture_evidence": str(evidence.resolve()),
        "capture_evidence_sha256": _sha256(evidence),
        "graph_digest": capture.evidence.graph_digest,
        "strict_export": True,
        "effect_audit_passed": capture.evidence.effect_audit.passed,
        "eager_export_maximum_absolute_error": (
            capture.evidence.maximum_absolute_error
        ),
        "capture_seconds": time.perf_counter() - started,
        "inputs": [_tensor_contract(value) for value in inputs],
        "outputs": [_tensor_contract(value) for value in eager],
    }
    del capture
    gc.collect()
    torch.cuda.empty_cache()
    return eager, record


def _comparison(reference: Any, candidate: Any) -> dict[str, object]:
    import torch

    reference = reference.detach().cpu()
    candidate = candidate.detach().cpu()
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "passed": False,
            "shape_match": reference.shape == candidate.shape,
            "dtype_match": reference.dtype == candidate.dtype,
        }
    if not reference.is_floating_point():
        exact = bool(torch.equal(reference, candidate))
        return {
            "passed": exact,
            "exact": exact,
            "shape_match": True,
            "dtype_match": True,
        }
    difference = candidate.to(torch.float64) - reference.to(torch.float64)
    maximum = (
        float(difference.abs().max().item())
        if difference.numel()
        else 0.0
    )
    rmse = (
        float(torch.sqrt(difference.square().mean()).item())
        if difference.numel()
        else 0.0
    )
    reference_maximum = (
        float(reference.abs().max().item())
        if reference.numel()
        else 0.0
    )
    exact = bool(torch.equal(reference, candidate))
    return {
        "passed": exact,
        "exact": exact,
        "shape_match": True,
        "dtype_match": True,
        "maximum_absolute_error": maximum,
        "root_mean_square_error": rmse,
        "normalized_root_mean_square_error": (
            rmse / max(reference_maximum, 1.0e-12)
        ),
        "reference_absolute_maximum": reference_maximum,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--logical-export", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-release-hashes", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    if not torch.cuda.is_available():
        raise RuntimeError("partitioned MindDrive map capture requires CUDA")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    logical_export = args.logical_export.resolve()
    logical_program = load_exported_region(logical_export)
    map_inputs, map_kwargs = logical_program.example_inputs
    if map_kwargs:
        raise ValueError("MindDrive map profile unexpectedly uses kwargs")
    with torch.no_grad():
        reference = _as_tuple(
            logical_program.module()(*map_inputs, **map_kwargs)
        )
    del logical_program
    gc.collect()
    torch.cuda.empty_cache()

    model = load_real_minddrive_model(
        args.source_root,
        args.release_root,
        device=args.device,
        verify_hashes=not args.skip_release_hashes,
    )
    encoder = make_partitioned_minddrive_map_encoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    records: list[dict[str, object]] = []
    front_inputs = (
        map_inputs[0],
        map_inputs[2],
        map_inputs[4],
        *map_inputs[5:],
    )
    front, record = _capture(
        output,
        "map_front",
        encoder.front,
        front_inputs,
    )
    records.append(record)
    (
        query,
        image_memory,
        query_position,
        temporal_attention_mask,
        temporal_memory,
        temporal_position,
        reference_points,
        rec_ego_pose,
        memory_reference_point,
        memory_timestamp,
        memory_egopose,
        memory_mask,
    ) = front

    decoded = []
    for index, layer in enumerate(encoder.layers):
        layer_inputs = (
            query,
            image_memory,
            query_position,
            map_inputs[1],
            temporal_attention_mask,
            temporal_memory,
            temporal_position,
        )
        layer_outputs, record = _capture(
            output,
            f"map_decoder_layer_{index:02d}",
            layer,
            layer_inputs,
        )
        records.append(record)
        (query,) = layer_outputs
        decoded.append(query)

    finish_inputs = (
        *decoded,
        reference_points,
        map_inputs[2],
        map_inputs[3],
        rec_ego_pose,
        temporal_memory,
        memory_reference_point,
        memory_timestamp,
        memory_egopose,
        memory_mask,
    )
    candidate, record = _capture(
        output,
        "map_finish",
        encoder.finish,
        finish_inputs,
    )
    records.append(record)
    torch.cuda.synchronize()

    comparisons = [
        _comparison(expected, actual)
        for expected, actual in zip(reference, candidate, strict=True)
    ]
    source_exact = all(item["passed"] for item in comparisons)
    if not source_exact:
        raise ValueError(
            "partitioned map changed source outputs: "
            f"{comparisons}"
        )
    report = {
        "schema": "vlaforge.minddrive_partitioned_map_capture/2",
        "passed": True,
        "evidence_role": "real-L3-physical-partition-prerequisite",
        "logical_region": "map_encoder",
        "physical_export_region_count": len(records),
        "decoder_layer_count": len(encoder.layers),
        "core_op_delta": 0,
        "physical_tensor_abi": {
            "layout": "contiguous",
            "validated_at_capture": True,
            "arbitrary_strides_in_semantic_ir": False,
        },
        "source_exact": source_exact,
        "source_equivalence": comparisons,
        "regions": records,
        "logical_export": {
            "path": str(logical_export),
            "size_bytes": logical_export.stat().st_size,
            "sha256": _sha256(logical_export),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "regions": len(records),
                "decoder_layers": len(encoder.layers),
                "source_exact": source_exact,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
