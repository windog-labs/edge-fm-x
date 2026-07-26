#!/usr/bin/env python3
"""Capture source-exact MindDrive EVA around FlashAttention CUDA calls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from vlaforge.adapters.minddrive_real import (
    MINDDRIVE_INPUT_TYPES,
    load_real_minddrive_model,
    make_partitioned_minddrive_flash_vision_encoder,
)
from vlaforge.frontend import capture_region, save_exported_region
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
    return value if isinstance(value, tuple) else (value,)


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
        {
            "index": index,
            **_tensor_contract(value),
        }
        for index, value in enumerate(values)
        if not value.is_contiguous()
    ]
    if violations:
        raise ValueError(
            f"{name}: physical Region {role} violates the contiguous ABI: "
            f"{violations}"
        )


def _capture(
    output: Path,
    name: str,
    implementation: Any,
    inputs: tuple[Any, ...],
) -> tuple[Any, dict[str, object]]:
    import torch

    with torch.no_grad():
        eager = implementation(*inputs)
    eager_values = _as_tuple(eager)
    _require_contiguous(name, "inputs", inputs)
    _require_contiguous(name, "outputs", eager_values)
    region = TensorRegion(
        name,
        tuple(
            Value(f"input_{index}", _tensor_type(value))
            for index, value in enumerate(inputs)
        ),
        tuple(_tensor_type(value) for value in eager_values),
        metadata={
            "backend_physical_partition": True,
            "logical_region": "vision_encoder",
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
        "inputs": [
            _tensor_contract(value)
            for value in inputs
        ],
        "outputs": [
            _tensor_contract(value)
            for value in eager_values
        ],
    }
    del capture
    gc.collect()
    torch.cuda.empty_cache()
    return eager, record


def _equivalence(reference: Any, candidate: Any) -> dict[str, object]:
    import torch

    reference = reference.detach().cpu()
    candidate = candidate.detach().cpu()
    difference = candidate.to(torch.float64) - reference.to(torch.float64)
    maximum = float(difference.abs().max().item())
    rmse = float(torch.sqrt(difference.square().mean()).item())
    reference_maximum = float(reference.abs().max().item())
    return {
        "exact": bool(torch.equal(reference, candidate)),
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
    parser.add_argument("--invocation-inputs", type=Path, required=True)
    parser.add_argument("--official-image-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    import flash_attn_2_cuda
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    if not torch.cuda.is_available():
        raise RuntimeError("partitioned MindDrive vision capture requires CUDA")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = load_real_minddrive_model(
        args.source_root,
        args.release_root,
        device=args.device,
    )
    encoder = make_partitioned_minddrive_flash_vision_encoder(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    invocation = torch.load(
        args.invocation_inputs,
        map_location="cpu",
        weights_only=True,
    )
    camera_images = invocation["camera_images"].to(args.device)
    expected_camera = dict(MINDDRIVE_INPUT_TYPES)["camera_images"]
    if tuple(camera_images.shape) != expected_camera.shape:
        raise ValueError("MindDrive partition capture input profile changed")
    reference_payload = torch.load(
        args.official_image_features,
        map_location="cpu",
        weights_only=True,
    )
    reference = reference_payload["image_features"]

    records: list[dict[str, object]] = []
    features, record = _capture(
        output,
        "vision_stem",
        encoder.stem,
        (camera_images,),
    )
    records.append(record)
    flash_calls = []
    for index, (pre, flash, post) in enumerate(
        zip(
            encoder.block_pre,
            encoder.flash_attention,
            encoder.block_post,
            strict=True,
        )
    ):
        pre_name = f"vision_block_{index:02d}_pre"
        pre_values, pre_record = _capture(
            output,
            pre_name,
            pre,
            (features,),
        )
        records.append(pre_record)
        shortcut, query, key_value = pre_values
        with torch.no_grad():
            attention, _ = flash(
                query,
                key_value,
                key_padding_mask=None,
                causal=False,
            )
        attention = attention.contiguous()
        flash_calls.append(
            {
                "block": index,
                "query_shape": list(query.shape),
                "key_value_shape": list(key_value.shape),
                "dtype": _DTYPES[str(query.dtype)],
                "causal": False,
                "dropout": 0.0,
            }
        )
        post_name = f"vision_block_{index:02d}_post"
        features, post_record = _capture(
            output,
            post_name,
            post,
            (shortcut, attention),
        )
        records.append(post_record)
        del pre_values, shortcut, query, key_value, attention
        gc.collect()
        torch.cuda.empty_cache()
    candidate, finish_record = _capture(
        output,
        "vision_finish",
        encoder.finish,
        (features,),
    )
    records.append(finish_record)
    torch.cuda.synchronize()

    comparison = _equivalence(reference, candidate)
    if not comparison["exact"]:
        raise ValueError(
            "partitioned source-exact vision changed outputs: "
            f"{comparison}"
        )
    flash_binary = Path(flash_attn_2_cuda.__file__).resolve()
    dynamic_dependencies = subprocess.run(
        ["ldd", str(flash_binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    report = {
        "schema": "vlaforge.minddrive_partitioned_vision_capture/2",
        "passed": True,
        "evidence_role": "real-L3-physical-partition-prerequisite",
        "logical_region": "vision_encoder",
        "physical_export_region_count": len(records),
        "flash_attention_call_count": len(flash_calls),
        "core_op_delta": 0,
        "physical_tensor_abi": {
            "layout": "contiguous",
            "arbitrary_strides_in_semantic_ir": False,
            "validated_at_capture": True,
        },
        "source_equivalence": comparison,
        "regions": records,
        "flash_attention_provider": {
            "path": str(flash_binary),
            "size_bytes": flash_binary.stat().st_size,
            "sha256": _sha256(flash_binary),
            "version": "2.6.3",
            "compiled_cuda_binary": True,
            "stable_tensor_abi": True,
            "links_libtorch_python": "libtorch_python" in dynamic_dependencies,
            "no_python_l4_compatible": False,
            "calls": flash_calls,
        },
        "inputs": {
            "invocation": str(args.invocation_inputs.resolve()),
            "invocation_sha256": _sha256(args.invocation_inputs),
            "reference": str(args.official_image_features.resolve()),
            "reference_sha256": _sha256(args.official_image_features),
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
                "flash_calls": len(flash_calls),
                "source_exact": comparison["exact"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
