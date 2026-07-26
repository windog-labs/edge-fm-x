#!/usr/bin/env python3
"""Measure whether every verified MindDrive raw AOTI artifact can stay loaded.

The audit intentionally runs in a fresh process and does not execute the
model.  It answers the narrower provider-design question of whether the
complete physical artifact set can be Session-resident on the target device.
CUDA allocator counters alone do not include memory owned by raw AOTI model
containers, so the report also records the driver-visible free-memory delta.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path
from typing import Any


_SCHEMA = "vlaforge.minddrive_aoti_residency/1"
_MANIFEST_SCHEMA = "vlaforge.minddrive_aoti_artifact_manifest/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _current_rss_bytes() -> int:
    with Path("/proc/self/status").open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/self/status has no VmRSS")


def _maximum_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch

    if not args.device.startswith("cuda:"):
        raise ValueError("MindDrive residency audit requires explicit cuda:N")
    device_ordinal = int(args.device.split(":", 1)[1])
    artifact_root = args.artifact_root.resolve()
    manifest_path = args.artifact_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != _MANIFEST_SCHEMA
        or manifest.get("passed") is not True
    ):
        raise ValueError("MindDrive artifact manifest is invalid or failed")
    if Path(manifest["artifact_root"]).resolve() != artifact_root:
        raise ValueError("artifact root disagrees with aggregate manifest")

    regions = sorted(
        manifest["regions"],
        key=lambda item: (
            int(item["artifact"]["size_bytes"]),
            str(item["name"]),
        ),
    )
    if not regions:
        raise ValueError("artifact manifest contains no Regions")

    verified = []
    for item in regions:
        name = str(item["name"])
        artifact = item["artifact"]
        path = artifact_root / f"{name}.so"
        if path.resolve() != Path(artifact["path"]).resolve():
            raise ValueError(f"{name}: artifact path changed")
        size_bytes = path.stat().st_size
        if size_bytes != int(artifact["size_bytes"]):
            raise ValueError(f"{name}: artifact size changed")
        digest = _sha256(path)
        if digest != str(artifact["sha256"]):
            raise ValueError(f"{name}: artifact SHA-256 changed")
        verified.append(
            {
                "name": name,
                "path": str(path),
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )

    torch.cuda.set_device(device_ordinal)
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device_ordinal)
    free_before, total = torch.cuda.mem_get_info(device_ordinal)
    rss_before = _current_rss_bytes()
    runners = []
    loads = []
    started = time.perf_counter()
    status = "passed"
    failure: dict[str, str] | None = None
    try:
        for item in verified:
            load_started = time.perf_counter()
            runner = torch._export.aot_load(item["path"], args.device)
            runners.append(runner)
            free_now, _ = torch.cuda.mem_get_info(device_ordinal)
            loads.append(
                {
                    "name": item["name"],
                    "load_seconds": time.perf_counter() - load_started,
                    "resident_count": len(runners),
                    "current_rss_bytes": _current_rss_bytes(),
                    "maximum_rss_bytes": _maximum_rss_bytes(),
                    "cuda_driver_free_bytes": free_now,
                    "cuda_driver_used_delta_bytes": free_before - free_now,
                }
            )
    except BaseException as error:
        status = "failed"
        failure = {
            "type": type(error).__name__,
            "message": str(error),
        }

    torch.cuda.synchronize(device_ordinal)
    free_loaded, _ = torch.cuda.mem_get_info(device_ordinal)
    loaded_summary = {
        "requested_count": len(verified),
        "resident_count": len(runners),
        "all_resident": len(runners) == len(verified),
        "load_seconds": time.perf_counter() - started,
        "current_rss_bytes": _current_rss_bytes(),
        "maximum_rss_bytes": _maximum_rss_bytes(),
        "cuda_driver_free_bytes": free_loaded,
        "cuda_driver_used_delta_bytes": free_before - free_loaded,
    }

    runners.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device_ordinal)
    free_after_destroy, _ = torch.cuda.mem_get_info(device_ordinal)
    report = {
        "schema": _SCHEMA,
        "status": status,
        "passed": status == "passed" and loaded_summary["all_resident"],
        "pid": os.getpid(),
        "artifact_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "artifact_set_sha256": manifest["artifact_set_sha256"],
            "region_count": len(verified),
            "artifact_bytes": sum(
                int(item["size_bytes"]) for item in verified
            ),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(device_ordinal),
            "cuda_total_bytes": total,
            "cuda_driver_free_before_bytes": free_before,
            "rss_before_bytes": rss_before,
        },
        "loaded": loaded_summary,
        "after_destroy": {
            "current_rss_bytes": _current_rss_bytes(),
            "maximum_rss_bytes": _maximum_rss_bytes(),
            "cuda_driver_free_bytes": free_after_destroy,
            "cuda_driver_unrecovered_bytes": (
                free_before - free_after_destroy
            ),
        },
        "loads": loads,
        "failure": failure,
        "scope": (
            "fresh-process load-only Session-residency feasibility; "
            "not numerical or generated-C++ L4 evidence"
        ),
    }
    _write_report(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
