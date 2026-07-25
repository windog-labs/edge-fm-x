#!/usr/bin/env python3
"""Compile and audit memory-bounded real OpenVLA CUDA artifacts.

The filename intentionally avoids the repository-wide ``build_*`` ignore
pattern so this production evidence tool cannot be omitted from a commit.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.openvla_partitioned import (  # noqa: E402
    OPENVLA_ACTION_DIM,
    OPENVLA_CHUNK_SIZE,
    OPENVLA_HEADS,
    OPENVLA_HEAD_DIM,
    OPENVLA_LAYER_COUNT,
    OPENVLA_MAX_CACHE_LENGTH,
    OPENVLA_PARTITION_CAPTURE_SCHEMA,
    OPENVLA_PREFIX_LENGTH,
    OPENVLA_UPSTREAM_REVISION,
    artifact_region_names,
    decode_chunk_names,
    prefill_chunk_names,
)


_REPORT_SCHEMA = "vlaforge.openvla_real_l3/1"
_COMPILE_SCHEMA = "vlaforge.real_aoti_compile/1"
_BF16_NRMSE_TOLERANCE = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPOSITORY_ROOT,
    ).stdout.strip()


def _verify_capture(
    capture_root: Path,
) -> dict[str, Any]:
    report_path = capture_root / "capture.json"
    report = _json(report_path)
    if (
        report.get("schema") != OPENVLA_PARTITION_CAPTURE_SCHEMA
        or report.get("status") != "passed"
        or not report.get("passed", False)
        or report.get("checkpoint", {}).get("revision")
        != OPENVLA_UPSTREAM_REVISION
        or report.get("correctness", {}).get("token_ids_equal")
        is not True
        or report.get("correctness", {}).get(
            "all_export_replays_exact"
        )
        is not True
    ):
        raise ValueError("OpenVLA partition capture is not passing")
    names = tuple(
        str(item["name"]) for item in report.get("regions", ())
    )
    if (
        len(names) != len(artifact_region_names())
        or frozenset(names) != frozenset(artifact_region_names())
    ):
        raise ValueError("OpenVLA partition capture region set mismatch")
    for name in names:
        for suffix in (".pt2e", ".capture.json"):
            path = capture_root / "source_exports" / f"{name}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)
    return report


def _verified_compile_record(
    report_path: Path,
    *,
    normalized_root: Path,
    artifact_root: Path,
    name: str,
) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    payload = _json(report_path)
    records = payload.get("regions")
    if (
        payload.get("schema") != _COMPILE_SCHEMA
        or not isinstance(records, list)
        or len(records) != 1
        or records[0].get("region") != name
    ):
        return None
    record = records[0]
    normalized = normalized_root / f"{name}.pt2"
    artifact = artifact_root / f"{name}.pt2"
    if (
        not normalized.is_file()
        or not artifact.is_file()
        or record.get("normalize_exact") is not True
        or record.get("normalized_export_sha256")
        != _sha256(normalized)
        or record.get("package_sha256") != _sha256(artifact)
        or record.get("package_size_bytes") != artifact.stat().st_size
    ):
        return None
    return record


def _compile_regions(
    *,
    capture_root: Path,
    output_root: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    source_exports = capture_root / "source_exports"
    normalized_root = output_root / "normalized_exports"
    artifact_root = output_root / "artifacts"
    report_root = output_root / "compile_reports"
    log_root = output_root / "compile_logs"
    for path in (
        normalized_root,
        artifact_root,
        report_root,
        log_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for index, name in enumerate(artifact_region_names(), start=1):
        report_path = report_root / f"{name}.json"
        record = (
            _verified_compile_record(
                report_path,
                normalized_root=normalized_root,
                artifact_root=artifact_root,
                name=name,
            )
            if resume
            else None
        )
        if record is not None:
            print(
                f"[{index:02d}/{len(artifact_region_names()):02d}] "
                f"reuse {name}",
                flush=True,
            )
            records.append(record)
            continue

        for path in (
            normalized_root / f"{name}.pt2",
            artifact_root / f"{name}.pt2",
            report_path,
        ):
            path.unlink(missing_ok=True)
        command = [
            sys.executable,
            str(_SOURCE_ROOT / "tools" / "compile_real_aoti_exports.py"),
            "--export-dir",
            str(source_exports),
            "--normalized-export-dir",
            str(normalized_root),
            "--output-dir",
            str(artifact_root),
            "--report",
            str(report_path),
            "--inductor-profile",
            "conservative",
            name,
        ]
        environment = {
            **dict(os.environ),
            "PYTHONPATH": str(_SOURCE_ROOT / "python"),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "MAX_JOBS": "1",
        }
        print(
            f"[{index:02d}/{len(artifact_region_names()):02d}] "
            f"compile {name}",
            flush=True,
        )
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            cwd=_REPOSITORY_ROOT,
        )
        (log_root / f"{name}.stdout").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (log_root / f"{name}.stderr").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{name}: AOTI compile failed with "
                f"{completed.returncode}\n{completed.stderr[-8000:]}"
            )
        record = _verified_compile_record(
            report_path,
            normalized_root=normalized_root,
            artifact_root=artifact_root,
            name=name,
        )
        if record is None:
            raise RuntimeError(f"{name}: compile evidence verification failed")
        records.append(record)
    return records


def _metrics(
    expected: tuple[Any, ...],
    actual: tuple[Any, ...],
) -> list[dict[str, object]]:
    import torch

    if len(expected) != len(actual):
        raise ValueError("artifact output arity mismatch")
    results: list[dict[str, object]] = []
    for index, (reference, observed) in enumerate(
        zip(expected, actual, strict=True)
    ):
        if (
            reference.dtype != observed.dtype
            or tuple(reference.shape) != tuple(observed.shape)
        ):
            raise ValueError(f"artifact output {index} contract mismatch")
        difference = (reference - observed).abs().float()
        maximum = (
            float(difference.max().item()) if difference.numel() else 0.0
        )
        mean = (
            float(difference.mean().item()) if difference.numel() else 0.0
        )
        reference_rms = (
            reference.float().square().mean().sqrt()
            if reference.numel()
            else torch.tensor(0.0)
        )
        error_rms = (
            difference.square().mean().sqrt()
            if difference.numel()
            else torch.tensor(0.0)
        )
        denominator = float(reference_rms.item())
        nrmse = (
            float(error_rms.item()) / denominator
            if denominator > 0.0
            else (0.0 if float(error_rms.item()) == 0.0 else float("inf"))
        )
        exact = bool(torch.equal(reference, observed))
        floating = reference.is_floating_point()
        tolerance = _BF16_NRMSE_TOLERANCE if floating else 0.0
        passed = nrmse <= tolerance if floating else exact
        results.append(
            {
                "output": index,
                "shape": list(reference.shape),
                "dtype": str(reference.dtype),
                "exact": exact,
                "maximum_absolute_error": maximum,
                "mean_absolute_error": mean,
                "nrmse": nrmse,
                "nrmse_tolerance": tolerance,
                "passed": passed,
            }
        )
    return results


def _run_region_pair(
    torch: Any,
    *,
    normalized_root: Path,
    extracted_root: Path,
    name: str,
) -> list[dict[str, object]]:
    exported = torch.export.load(normalized_root / f"{name}.pt2")
    arguments, keyword_arguments = exported.example_inputs
    with torch.inference_mode():
        expected = _as_tuple(
            exported.module()(
                *arguments,
                **keyword_arguments,
            )
        )
    artifact = _load_extracted_runner(torch, extracted_root, name)
    with torch.inference_mode():
        actual = tuple(artifact.run(list(arguments)))
    torch.cuda.synchronize()
    result = _metrics(expected, actual)
    del exported, artifact, expected, actual, arguments
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_artifact(
    torch: Any,
    extracted_root: Path,
    name: str,
    arguments: tuple[Any, ...],
) -> tuple[Any, ...]:
    artifact = _load_extracted_runner(torch, extracted_root, name)
    with torch.inference_mode():
        raw = tuple(artifact.run(list(arguments)))
        outputs = tuple(value.clone() for value in raw)
    torch.cuda.synchronize()
    del artifact, raw
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def _artifact_pipeline(
    torch: Any,
    *,
    normalized_root: Path,
    extracted_root: Path,
) -> dict[str, object]:
    prepare_export = torch.export.load(
        normalized_root / "prepare_multimodal_prefix.pt2"
    )
    prepare_arguments, _ = prepare_export.example_inputs
    del prepare_export
    hidden, causal_mask, position_ids, cache_position = _run_artifact(
        torch,
        extracted_root,
        "prepare_multimodal_prefix",
        prepare_arguments,
    )

    fixed_cache: list[Any] = []
    for name in prefill_chunk_names():
        outputs = _run_artifact(
            torch,
            extracted_root,
            name,
            (hidden, causal_mask, position_ids, cache_position),
        )
        hidden = outputs[0]
        fixed_cache.extend(
            torch.nn.functional.pad(
                value,
                (0, 0, 0, OPENVLA_MAX_CACHE_LENGTH - value.shape[2]),
            )
            for value in outputs[1:]
        )

    logits, token = _run_artifact(
        torch,
        extracted_root,
        "token_logits_head",
        (hidden[:, -1:, :],),
    )
    del logits
    tokens = [int(token.item())]
    for step in range(OPENVLA_ACTION_DIM - 1):
        (hidden,) = _run_artifact(
            torch,
            extracted_root,
            "decode_token_embedding",
            (token,),
        )
        decode_position = OPENVLA_PREFIX_LENGTH + step
        position_ids = torch.tensor(
            [[decode_position]],
            dtype=torch.int64,
            device="cuda:0",
        )
        cache_position = position_ids[0]
        for chunk_index, name in enumerate(decode_chunk_names()):
            offset = chunk_index * 2 * OPENVLA_CHUNK_SIZE
            outputs = _run_artifact(
                torch,
                extracted_root,
                name,
                (
                    hidden,
                    position_ids,
                    cache_position,
                    *fixed_cache[
                        offset : offset + 2 * OPENVLA_CHUNK_SIZE
                    ],
                ),
            )
            hidden = outputs[0]
            fixed_cache[
                offset : offset + 2 * OPENVLA_CHUNK_SIZE
            ] = outputs[1:]
        logits, token = _run_artifact(
            torch,
            extracted_root,
            "token_logits_head",
            (hidden,),
        )
        del logits
        tokens.append(int(token.item()))

    action_tokens = torch.tensor(
        [tokens],
        dtype=torch.int64,
        device="cuda:0",
    )
    (action,) = _run_artifact(
        torch,
        extracted_root,
        "detokenize_action",
        (action_tokens,),
    )
    return {
        "tokens": tokens,
        "action": [float(value) for value in action.cpu().tolist()],
        "fixed_cache_tensors": len(fixed_cache),
        "fixed_cache_bytes": sum(
            value.numel() * value.element_size()
            for value in fixed_cache
        ),
    }


def _audit(
    *,
    capture_root: Path,
    output_root: Path,
    compile_records: list[dict[str, Any]],
    capture: dict[str, Any],
    pipeline_repeats: int,
    extracted_root: Path,
) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA artifact audit requires CUDA")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) != (8, 6):
        raise RuntimeError("OpenVLA artifact audit is pinned to sm_86")
    normalized_root = output_root / "normalized_exports"
    artifact_root = output_root / "artifacts"
    extracted = _prepare_extracted_artifacts(
        artifact_root=artifact_root,
        extracted_root=extracted_root,
        compile_records=compile_records,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    parity: list[dict[str, object]] = []
    for index, name in enumerate(artifact_region_names(), start=1):
        print(
            f"[audit {index:02d}/{len(artifact_region_names()):02d}] "
            f"{name}",
            flush=True,
        )
        outputs = _run_region_pair(
            torch,
            normalized_root=normalized_root,
            extracted_root=extracted_root,
            name=name,
        )
        parity.append({"region": name, "outputs": outputs})
        if not all(bool(item["passed"]) for item in outputs):
            raise RuntimeError(f"{name}: artifact parity tolerance failed")

    pipelines: list[dict[str, object]] = []
    for repeat in range(pipeline_repeats):
        print(
            f"[pipeline {repeat + 1}/{pipeline_repeats}]",
            flush=True,
        )
        pipelines.append(
            _artifact_pipeline(
                torch,
                normalized_root=normalized_root,
                extracted_root=extracted_root,
            )
        )
    reference_tokens = capture["correctness"]["reference_tokens"]
    reference_action = capture["correctness"]["action"]
    for pipeline in pipelines:
        if pipeline["tokens"] != reference_tokens:
            raise RuntimeError(
                "OpenVLA artifact pipeline token mismatch: "
                f"{pipeline['tokens']} != {reference_tokens}"
            )
        maximum_action_error = max(
            abs(expected - actual)
            for expected, actual in zip(
                reference_action,
                pipeline["action"],
                strict=True,
            )
        )
        pipeline["action_maximum_absolute_error"] = maximum_action_error
        if maximum_action_error > 1e-12:
            raise RuntimeError(
                "OpenVLA artifact pipeline action mismatch: "
                f"{maximum_action_error}"
            )
    deterministic = all(
        pipeline["tokens"] == pipelines[0]["tokens"]
        and pipeline["action"] == pipelines[0]["action"]
        for pipeline in pipelines[1:]
    )
    if not deterministic:
        raise RuntimeError("OpenVLA artifact pipeline is not deterministic")

    revision = _git(["rev-parse", "HEAD"])
    dirty = bool(
        _git(["status", "--porcelain", "--untracked-files=no"])
    )
    return {
        "schema": _REPORT_SCHEMA,
        "status": "passed",
        "passed": True,
        "evidence_level": "L3",
        "evidence_kind": "real-checkpoint-partitioned-aoti-parity",
        "model": "OpenVLA-7B",
        "checkpoint": capture["checkpoint"],
        "repository": {
            "revision": revision,
            "source_dirty": dirty,
        },
        "environment": {
            "host": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "target": "sm_86",
        },
        "partition": capture["partition"],
        "compile": {
            "profile": "conservative",
            "regions": compile_records,
            "total_seconds": sum(
                float(record["compile_seconds"])
                for record in compile_records
            ),
            "total_artifact_bytes": sum(
                int(record["package_size_bytes"])
                for record in compile_records
            ),
            "maximum_peak_host_rss_kib": max(
                int(record["compile_peak_rss_kib"])
                for record in compile_records
            ),
            "all_active_version_normalizations_exact": all(
                record.get("normalize_exact") is True
                for record in compile_records
            ),
            "stable_extracted_runner": {
                "root": str(extracted_root),
                "files": extracted["files"],
                "size_bytes": extracted["size_bytes"],
                "reason": (
                    "reuse stable wrapper.so paths across bounded decode "
                    "steps instead of repeatedly extracting PT2 packages"
                ),
            },
        },
        "correctness": {
            "bf16_nrmse_tolerance": _BF16_NRMSE_TOLERANCE,
            "region_parity": parity,
            "all_region_outputs_within_tolerance": True,
            "pipelines": pipelines,
            "pipeline_repeats": pipeline_repeats,
            "repeated_pipeline_exact": deterministic,
            "action_tokens_equal": True,
            "final_action_equal": True,
        },
        "memory": {
            "capture_peak_cuda_allocated_bytes": capture["memory"][
                "peak_cuda_allocated_bytes"
            ],
            "audit_peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated()
            ),
            "authoritative_state_bytes": 0,
            "derived_fixed_kv_bytes": pipelines[0][
                "fixed_cache_bytes"
            ],
        },
        "unsupported_items": [
            {
                "item": "generated no-Python C++ L4",
                "status": "not claimed by this report",
                "reason": (
                    "L3 uses backend-owned weight-paged subartifacts; "
                    "generated Session scheduling is a separate gate"
                ),
            }
        ],
        "source_capture": {
            "path": str(capture_root / "capture.json"),
            "sha256": _sha256(capture_root / "capture.json"),
            "torch": capture["environment"]["torch"],
            "all_export_replays_exact": capture["correctness"][
                "all_export_replays_exact"
            ],
        },
    }


def _as_tuple(value: object) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _prepare_extracted_artifacts(
    *,
    artifact_root: Path,
    extracted_root: Path,
    compile_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Materialize stable AOTI shared-object paths exactly once.

    ``aoti_load_package`` extracts every PT2 load to a fresh temporary path.
    OpenVLA reuses each decode artifact six times, so that behavior retains
    many deleted-but-mapped multi-hundred-MiB shared objects in one process.
    Stable paths let the dynamic loader reuse mappings while runner
    destruction still releases each Region's CUDA constants.
    """

    extracted_root.mkdir(parents=True, exist_ok=True)
    expected_hashes = {
        str(record["region"]): str(record["package_sha256"])
        for record in compile_records
    }
    file_count = 0
    size_bytes = 0
    for index, name in enumerate(artifact_region_names(), start=1):
        package = artifact_root / f"{name}.pt2"
        destination = extracted_root / name
        marker = destination / ".package-sha256"
        expected_hash = expected_hashes[name]
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != expected_hash
            or not tuple(destination.rglob("*.wrapper.so"))
        ):
            import shutil

            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            with zipfile.ZipFile(package) as archive:
                members = tuple(
                    member
                    for member in archive.infolist()
                    if "/data/aotinductor/model/" in member.filename
                )
                if not members:
                    raise ValueError(f"{name}: package has no AOTI model")
                for member in members:
                    relative = Path(member.filename)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(
                            f"{name}: unsafe package member "
                            f"{member.filename!r}"
                        )
                    archive.extract(member, destination)
            marker.write_text(expected_hash + "\n", encoding="utf-8")
            print(
                f"[extract {index:02d}/{len(artifact_region_names()):02d}] "
                f"{name}",
                flush=True,
            )
        for path in destination.rglob("*"):
            if path.is_file():
                file_count += 1
                size_bytes += path.stat().st_size
    return {"files": file_count, "size_bytes": size_bytes}


def _load_extracted_runner(
    torch: Any,
    extracted_root: Path,
    name: str,
) -> Any:
    candidates = tuple(
        (extracted_root / name).rglob("*.wrapper.so")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{name}: expected one extracted wrapper.so, got "
            f"{len(candidates)}"
        )
    shared_object = candidates[0]
    return torch._C._aoti.AOTIModelContainerRunnerCuda(
        str(shared_object),
        1,
        "cuda:0",
        str(shared_object.parent),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--extracted-root",
        type=Path,
        help=(
            "stable wrapper.so cache; defaults to "
            "<output-root>/extracted_artifacts"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--pipeline-repeats", type=int, default=2)
    args = parser.parse_args(argv)
    if args.compile_only and args.audit_only:
        parser.error("--compile-only and --audit-only are mutually exclusive")
    if args.pipeline_repeats < 1:
        parser.error("--pipeline-repeats must be positive")

    capture = _verify_capture(args.capture_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        compile_records = []
        for name in artifact_region_names():
            record = _verified_compile_record(
                args.output_root / "compile_reports" / f"{name}.json",
                normalized_root=args.output_root / "normalized_exports",
                artifact_root=args.output_root / "artifacts",
                name=name,
            )
            if record is None:
                raise RuntimeError(f"{name}: missing verified compile record")
            compile_records.append(record)
    else:
        compile_records = _compile_regions(
            capture_root=args.capture_root,
            output_root=args.output_root,
            resume=args.resume,
        )
    if args.compile_only:
        return 0

    report = _audit(
        capture_root=args.capture_root,
        output_root=args.output_root,
        compile_records=compile_records,
        capture=capture,
        pipeline_repeats=args.pipeline_repeats,
        extracted_root=(
            args.extracted_root
            or args.output_root / "extracted_artifacts"
        ),
    )
    destination = args.report or args.output_root / "artifact-l3.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
