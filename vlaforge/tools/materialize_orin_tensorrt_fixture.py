#!/usr/bin/env python3
"""Materialize a compile-only SM87 TensorRT generated Session fixture.

The payloads are deliberately not TensorRT engines. They exist only so the
generated source embeds realistic artifact paths, hashes, sizes, target, and
backend variants. Driverless JetPack containers compile the complete arm64
Session; a real engine is required before any runtime evidence can be claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vlaforge.adapters import build_driving_diffusion_fixture
from vlaforge.codegen import (
    CppArtifactRegionDefinition,
    driving_diffusion_runner_source,
    driving_diffusion_validators,
    generate_cpp_session,
)
from vlaforge.compiler import CompilerProfile, compile_module


def materialize(output: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output}")
    generated = output / "generated"
    artifacts = output / "artifacts"
    generated.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    fixture = build_driving_diffusion_fixture()
    compilation = compile_module(
        fixture.module,
        profile=CompilerProfile.OFF,
        default_device="cuda:0",
        state_device="cuda:0",
    )
    definitions = {}
    artifact_rows = []
    for index, region in enumerate(compilation.module.regions):
        payload = (
            "VLAForge TensorRT compile-only placeholder\n"
            f"region={region.name}\n"
            "target=sm_87\n"
        ).encode()
        relative = f"artifacts/{region.name}.engine"
        destination = output / relative
        destination.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        definitions[region.name] = CppArtifactRegionDefinition(
            region_name=region.name,
            backend="tensorrt",
            artifact_path=relative,
            artifact_sha256=digest,
            artifact_size_bytes=len(payload),
            io_schema_digest=compilation.plan.io_schema_digest,
            target="sm_87",
            device="cuda:0",
            backend_variant="tensorrt-10.3-cu126-jetpack-r36.4",
        )
        artifact_rows.append(
            {
                "region_id": index,
                "region_name": region.name,
                "path": relative,
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )

    sources = generate_cpp_session(
        compilation.plan,
        compilation.module,
        artifact_regions=definitions,
        validators=driving_diffusion_validators(),
        runner_source=driving_diffusion_runner_source(),
    )
    sources.write(generated)
    report = {
        "schema": "vlaforge.orin_tensorrt_compile_fixture/1",
        "status": "compile_only",
        "target": "sm_87",
        "backend": "tensorrt",
        "backend_variant": "tensorrt-10.3-cu126-jetpack-r36.4",
        "io_schema_digest": compilation.plan.io_schema_digest,
        "generated_source_digest": sources.digest(),
        "artifacts": artifact_rows,
        "claim_boundary": {
            "serialized_tensorrt_engines": False,
            "orin_gpu_execution": False,
            "real_model": False,
            "purpose": (
                "arm64 backend/codegen/link portability before Orin access"
            ),
        },
    }
    (output / "fixture_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = materialize(args.output.resolve())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
