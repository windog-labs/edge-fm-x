#!/usr/bin/env python3
"""Compile saved real-model torch.export regions into AOTI packages."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch


_INDUCTOR_PROFILES: dict[str, dict[str, object]] = {
    "default": {
        "aot_inductor.force_mmap_weights": True,
    },
    # Keep GEMMs on ATen and disable epilogue fusion when a deployment audit
    # needs the closest practical match to exported eager BF16 numerics.
    # `emulate_precision_casts` is deliberately excluded: PyTorch 2.6 fails
    # while lowering the real SmolVLA prefix graph with that option.
    "conservative": {
        "aot_inductor.force_mmap_weights": True,
        "force_same_precision": True,
        "max_autotune_gemm_backends": "ATEN",
        "mixed_mm_choice": "aten",
        "epilogue_fusion": False,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--inductor-profile",
        choices=tuple(_INDUCTOR_PROFILES),
        default="default",
    )
    parser.add_argument("regions", nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name in args.regions:
        export_path = args.export_dir / f"{name}.pt2e"
        package_path = args.output_dir / f"{name}.pt2"
        program = torch.export.load(export_path)
        started = time.perf_counter()
        actual = Path(
            torch._inductor.aoti_compile_and_package(
                program,
                package_path=str(package_path),
                inductor_configs=_INDUCTOR_PROFILES[
                    args.inductor_profile
                ],
            )
        )
        elapsed = time.perf_counter() - started
        if actual != package_path or not package_path.is_file():
            raise RuntimeError(
                f"AOTI package path mismatch: {actual} != {package_path}"
            )
        records.append(
            {
                "region": name,
                "export_path": str(export_path.resolve()),
                "export_sha256": _sha256(export_path),
                "package_path": str(package_path.resolve()),
                "package_sha256": _sha256(package_path),
                "package_size_bytes": package_path.stat().st_size,
                "compile_seconds": elapsed,
                "graph_nodes": len(tuple(program.graph_module.graph.nodes)),
            }
        )
        print(
            f"compiled region={name} seconds={elapsed:.3f} "
            f"bytes={package_path.stat().st_size}",
            flush=True,
        )
        del program
        gc.collect()

    report = {
        "schema": "vlaforge.real_aoti_compile/1",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "inductor_profile": args.inductor_profile,
        "inductor_configs": _INDUCTOR_PROFILES[args.inductor_profile],
        "regions": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
