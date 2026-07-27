#!/usr/bin/env python3
"""Create a durable, checkpoint-light A100/H20 handoff directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
sys.path.insert(0, str(_SOURCE_ROOT / "python"))

from vlaforge.adapters.autovla_real import (  # noqa: E402
    AUTOVLA_CODEBOOK_SHA256,
    AUTOVLA_SOURCE_SHA256,
    AUTOVLA_UPSTREAM_REVISION,
)


AUTOVLA_INFERENCE_FILES = (
    "LICENSE",
    "README.md",
    "models/__init__.py",
    "models/autovla.py",
    "models/action_tokenizer.py",
    "config/eval/qwen2.5-vl-3B-nusc-sft-eval.yaml",
    "codebook_cache/agent_vocab.pkl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _archive_directory(source: Path, output: Path) -> None:
    with tarfile.open(output, "w:gz") as archive:
        archive.add(source, arcname=source.name, recursive=True)


def _archive_autovla_inference_source(
    source: Path,
    output: Path,
    *,
    revision: str,
) -> str:
    if revision != AUTOVLA_UPSTREAM_REVISION:
        raise ValueError(
            f"unexpected AutoVLA source revision: {revision}"
        )
    root_name = f"autovla-inference-{revision[:8]}"
    for relative, expected in AUTOVLA_SOURCE_SHA256.items():
        path = source / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(
                f"AutoVLA pinned source mismatch: {relative}"
            )
    codebook = source / "codebook_cache/agent_vocab.pkl"
    if not codebook.is_file() or _sha256(codebook) != AUTOVLA_CODEBOOK_SHA256:
        raise ValueError("AutoVLA codebook digest mismatch")
    with tarfile.open(output, "w:gz") as archive:
        for relative in AUTOVLA_INFERENCE_FILES:
            path = source / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.add(path, arcname=f"{root_name}/{relative}")
    return root_name


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--autovla-source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-l2-root", type=Path)
    parser.add_argument(
        "--verify-checkpoint-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"handoff directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    status = _git(
        _REPOSITORY_ROOT,
        "status",
        "--short",
        "--untracked-files=no",
    )
    if status:
        raise RuntimeError("commit tracked VLAForge changes before handoff")
    branch = _git(_REPOSITORY_ROOT, "symbolic-ref", "--short", "HEAD")
    revision = _git(_REPOSITORY_ROOT, "rev-parse", "HEAD")

    source_root = args.autovla_source_root.resolve()
    source_revision = _git(source_root, "rev-parse", "HEAD")
    source_status = _git(
        source_root,
        "status",
        "--short",
        "--untracked-files=no",
    )

    repository_bundle = output / "edge-fm-x.bundle"
    subprocess.run(
        [
            "git",
            "bundle",
            "create",
            str(repository_bundle),
            branch,
        ],
        cwd=_REPOSITORY_ROOT,
        check=True,
    )
    subprocess.run(
        ["git", "bundle", "verify", str(repository_bundle)],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    autovla_archive = (
        output / f"autovla-inference-source-{source_revision[:8]}.tar.gz"
    )
    autovla_archive_root = _archive_autovla_inference_source(
        source_root,
        autovla_archive,
        revision=source_revision,
    )

    packaged = {
        "repository_bundle": _file(repository_bundle),
        "autovla_source": _file(autovla_archive),
    }
    if args.baseline_l2_root is not None:
        baseline_root = args.baseline_l2_root.resolve()
        if not baseline_root.is_dir():
            raise FileNotFoundError(baseline_root)
        baseline_archive = output / "autovla-partition-l2-baseline.tar.gz"
        _archive_directory(baseline_root, baseline_archive)
        packaged["partition_l2_baseline"] = _file(baseline_archive)

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_record = {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": (
            _sha256(checkpoint)
            if args.verify_checkpoint_hash
            else None
        ),
        "hash_verified": args.verify_checkpoint_hash,
        "packaged": False,
        "transfer_separately": True,
    }
    manifest = {
        "schema": "vlaforge.high_memory_handoff/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "vlaforge": {
            "branch": branch,
            "revision": revision,
        },
        "autovla": {
            "revision": source_revision,
            "source_checkout_dirty": bool(source_status),
            "source_checkout_status_line_count": (
                len(source_status.splitlines()) if source_status else 0
            ),
            "archive_source": (
                "hash-verified inference allowlist from pinned checkout"
            ),
            "archive_root": autovla_archive_root,
        },
        "packaged": packaged,
        "external": {
            "checkpoint": checkpoint_record,
            "qwen_snapshot": {
                "repository": "Qwen/Qwen2.5-VL-3B-Instruct",
                "revision": (
                    "66285546d2b821cf421d4f5eb2576359d3770cd3"
                ),
                "packaged": False,
                "download_or_transfer_separately": True,
            },
            "real_camera_input": {
                "packaged": False,
                "reason": "user-provided licensed offline sample",
            },
        },
        "claim_boundary": {
            "source_handoff_only": True,
            "contains_no_new_gpu_evidence": True,
        },
    }
    manifest_path = output / "handoff_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksums = [
        f"{record['sha256']}  {Path(record['path']).name}"
        for record in packaged.values()
    ]
    checksums.append(f"{_sha256(manifest_path)}  {manifest_path.name}")
    (output / "SHA256SUMS").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "\n".join(
            (
                "# VLAForge high-memory handoff",
                "",
                f"- VLAForge revision: `{revision}`",
                f"- Branch: `{branch}`",
                f"- AutoVLA revision: `{source_revision}`",
                "",
                "Restore:",
                "",
                "```bash",
                f"git clone --branch {branch} "
                "edge-fm-x.bundle edge-fm-x",
                "cd edge-fm-x",
                "cd ..",
                f"tar -xzf {autovla_archive.name}",
                "sha256sum -c SHA256SUMS",
                "```",
                "",
                "Transfer the checkpoint, pinned Qwen snapshot, and licensed "
                "real camera sample separately. Then follow "
                "`doc/vlaforge_high_memory_handoff.md`.",
                "",
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
