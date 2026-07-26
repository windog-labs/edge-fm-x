#!/usr/bin/env python3
"""Inventory the durable and external evidence for the paper artifact.

Large real-model checkpoints, AOTI packages, captures, and profiler databases
are intentionally excluded from Git.  This audit makes that boundary explicit:
it hashes the committed reports/raw summaries, inventories every absolute
``/tmp`` reference in the current formal reports, and records the external
roots that a user must archive to retain byte-for-byte reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "vlaforge.reproducibility_manifest/1"
_FORMAL_REPORT_GLOBS = (
    "doc/reports/vlaforge_ablations_v01/paper_ablations.json",
    "doc/reports/vlaforge_architecture_v01/*.json",
    "doc/reports/vlaforge_autovla_v01/autovla_frontend_l2.json",
    "doc/reports/vlaforge_cuda_matrix_v01/cuda_paper_matrix.json",
    "doc/reports/vlaforge_heldout_v01/*.json",
    "doc/reports/vlaforge_minddrive_v01/*.json",
    "doc/reports/vlaforge_release_v01/*.json",
    "doc/reports/vlaforge_real_v03/*.json",
)
_PAPER_ARTIFACTS = (
    "doc/vlaforge_paper_draft.md",
    "doc/vlaforge_paper_design.md",
    "doc/vlaforge_claim_evidence_map.md",
    "doc/model_cards/README.md",
    "doc/model_cards/autovla.md",
    "doc/model_cards/minddrive.md",
    "doc/figures/vlaforge_paper/architecture.svg",
    "doc/figures/vlaforge_paper/performance.svg",
    "doc/figures/vlaforge_paper/ablations.svg",
    "doc/figures/vlaforge_paper/figures_manifest.json",
)
_ARCHIVE_ROOTS = (
    (
        "smolvla_l3_capture_artifacts",
        "/tmp/vlaforge-smolvla-l3.hr4TVE",
        True,
        "real SmolVLA export and AOTI artifact reproduction",
    ),
    (
        "smolvla_l4_support",
        "/tmp/vlaforge-smolvla-l4.oYi5dQ",
        True,
        "real SmolVLA generated-session support inputs and artifacts",
    ),
    (
        "smolvla_l4_aligned_bundle",
        "/tmp/vlaforge-smolvla-l4-aligned-0cf3d12",
        True,
        "final aligned SmolVLA Compile Bundle",
    ),
    (
        "diffusiondrive_checkpoint",
        "/tmp/vlaforge-diffusiondrive-ckpt",
        True,
        "pinned DiffusionDrive checkpoint",
    ),
    (
        "diffusiondrive_l3",
        "/tmp/vlaforge-diffusiondrive-l3-clean",
        True,
        "real DiffusionDrive exports and AOTI artifacts",
    ),
    (
        "diffusiondrive_l4",
        "/tmp/vlaforge-diffusiondrive-l4-clean",
        True,
        "real DiffusionDrive Compile Bundle and deterministic inputs",
    ),
    (
        "openvla_l3_capture",
        "/tmp/vlaforge-openvla-l3-capture",
        True,
        "real OpenVLA source exports and deterministic inputs",
    ),
    (
        "openvla_l3_artifacts",
        "/tmp/vlaforge-openvla-l3-artifacts",
        True,
        "36 normalized OpenVLA exports and sm_86 AOTI packages",
    ),
    (
        "autovla_checkpoint",
        "/tmp/vlaforge-autovla-checkpoint-a7d7ba3",
        True,
        "pinned 16.29 GB AutoVLA PDMS 89 checkpoint",
    ),
    (
        "autovla_l2_frontend",
        "/tmp/vlaforge-autovla-l2-f750e54",
        True,
        "real AutoVLA partitioned L2 exports and capture evidence",
    ),
    (
        "autovla_l3_candidate",
        "/tmp/vlaforge-autovla-l3-conservative-f750e54",
        False,
        "optional conservative AOTI L3-candidate artifacts",
    ),
    (
        "minddrive_complete_l2_l3_l4",
        "/home/zhangzimo/Archives/vlaforge-minddrive-0.5b-20260726",
        True,
        (
            "complete MindDrive checkpoint, frontend captures, AOTI "
            "artifacts, verified bundle, and generated L4 evidence"
        ),
    ),
    (
        "nsight_binary_profiles",
        "/tmp/vlaforge-nsight-v2",
        False,
        (
            "optional timeline reinspection; parsed NSYS/NCU summaries and "
            "profile hashes are committed"
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(_REPOSITORY_ROOT)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _walk_strings(
    value: object,
    *,
    pointer: str = "$",
) -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_strings(
                item,
                pointer=f"{pointer}.{key}",
            )
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _walk_strings(
                item,
                pointer=f"{pointer}[{index}]",
            )
    elif isinstance(value, str):
        yield pointer, value


def _absolute_references(
    reports: Iterable[tuple[Path, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    references: dict[str, list[dict[str, str]]] = {}
    for report_path, report in reports:
        for pointer, value in _walk_strings(report):
            known_external_roots = tuple(
                path for _, path, _, _ in _ARCHIVE_ROOTS
            )
            if not (
                value.startswith(("/tmp/", str(_REPOSITORY_ROOT) + "/"))
                or any(
                    value == root or value.startswith(root + "/")
                    for root in known_external_roots
                )
            ):
                continue
            references.setdefault(value, []).append(
                {
                    "report": str(report_path.relative_to(_REPOSITORY_ROOT)),
                    "json_pointer": pointer,
                }
            )
    result = []
    archive_roots = tuple(
        (name, Path(path)) for name, path, _, _ in _ARCHIVE_ROOTS
    )
    for value, referrers in sorted(references.items()):
        path = Path(value)
        archive = next(
            (
                name
                for name, root in archive_roots
                if path == root or root in path.parents
            ),
            None,
        )
        inside_repository = (
            path == _REPOSITORY_ROOT
            or _REPOSITORY_ROOT in path.parents
        )
        result.append(
            {
                "path": value,
                "exists": path.exists(),
                "kind": (
                    "directory"
                    if path.is_dir()
                    else "file" if path.is_file() else "missing"
                ),
                "inside_repository": inside_repository,
                "covered_by_archive_root": archive,
                "referenced_by": referrers,
            }
        )
    return result


def _reproduction_commands(
    reports: Iterable[tuple[Path, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    commands = []

    def visit(
        value: object,
        *,
        report: Path,
        pointer: str = "$",
    ) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{pointer}.{key}"
                if (
                    key in {"command", "orchestrator"}
                    and isinstance(item, list)
                    and all(isinstance(part, str) for part in item)
                ):
                    commands.append(
                        {
                            "report": str(
                                report.relative_to(_REPOSITORY_ROOT)
                            ),
                            "json_pointer": child,
                            "command": item,
                        }
                    )
                visit(item, report=report, pointer=child)
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                visit(
                    item,
                    report=report,
                    pointer=f"{pointer}[{index}]",
                )

    for report_path, report in reports:
        visit(report, report=report_path)
    return commands


def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    # Evidence is often moved from an ephemeral experiment path into durable
    # storage while the original path is retained as a compatibility symlink.
    # GNU du measures the symlink inode unless the root is dereferenced, which
    # would silently turn a multi-GiB archive into a byte-sized inventory row.
    measured_path = path.resolve(strict=True)
    completed = subprocess.run(
        ["du", "-sb", str(measured_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.split()[0])


def _archive_inventory() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "path": path,
            "exists": Path(path).exists(),
            "size_bytes": _du_bytes(Path(path)),
            "must_archive": must_archive,
            "reason": reason,
        }
        for name, path, must_archive, reason in _ARCHIVE_ROOTS
    ]


def _environment() -> dict[str, Any]:
    import torch

    def version(command: list[str]) -> str:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        text = (completed.stdout + completed.stderr).strip()
        return text if completed.returncode == 0 else "unavailable"

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total,"
            "power.limit",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "host": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "nvcc": version(["nvcc", "--version"]).splitlines()[-1],
        "cmake": version(["cmake", "--version"]).splitlines()[0],
        "compiler": version(["c++", "--version"]).splitlines()[0],
        "nsys": version(["nsys", "--version"]).splitlines()[0],
        "ncu": version(["ncu", "--version"]).splitlines()[-1],
    }


def _formal_reports() -> list[Path]:
    paths = {
        path
        for pattern in _FORMAL_REPORT_GLOBS
        for path in _REPOSITORY_ROOT.glob(pattern)
    }
    return sorted(paths)


def _formal_status_is_acceptable(report: Mapping[str, Any]) -> bool:
    return (
        report.get("status")
        in {"passed", "blocked", "resource_blocked"}
        or report.get("passed") is True
    )


def audit(
    artifact_evaluation_path: Path,
    *,
    baseline_revision: str,
) -> dict[str, Any]:
    artifact_evaluation = _json(artifact_evaluation_path)
    if (
        artifact_evaluation.get("status") != "passed"
        or artifact_evaluation.get("artifact_evaluation", {}).get("target")
        != "sm_86"
        or artifact_evaluation.get("claim_boundary", {}).get("orin")
    ):
        raise ValueError("installed-wheel artifact evaluation is incomplete")
    current_revision = _git(["rev-parse", "HEAD"])
    tracked_status = _git(
        ["status", "--short", "--untracked-files=no"]
    )
    if tracked_status:
        raise RuntimeError(
            "reproducibility audit requires a clean tracked worktree"
        )
    if (
        artifact_evaluation["repository"]["evaluated_revision"]
        != current_revision
    ):
        raise ValueError(
            "artifact evaluation revision does not match current revision"
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline_revision, "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("baseline revision is not an ancestor of HEAD")

    formal_paths = _formal_reports()
    reports = [(path, _json(path)) for path in formal_paths]
    required_reports = {
        "architecture_surface.json",
        "autovla_frontend_l2.json",
        "cuda_paper_matrix.json",
        "paper_ablations.json",
        "heldout_audit.json",
        "release_gate.json",
        "smolvla_artifact_l4.json",
        "diffusiondrive_artifact_l4.json",
        "openvla_artifact_l3.json",
        "real_cuda_evidence.json",
        "minddrive_l3.json",
        "minddrive_l4.json",
    }
    if not required_reports.issubset({path.name for path in formal_paths}):
        raise ValueError("formal report inventory is incomplete")
    if any(
        not _formal_status_is_acceptable(report)
        for _, report in reports
    ):
        raise ValueError("formal report has an unexpected status")

    references = _absolute_references(reports)
    commands = _reproduction_commands(reports)
    archives = _archive_inventory()
    committed_raw_roots = (
        _REPOSITORY_ROOT
        / "doc/reports/vlaforge_real_v03/real_cuda_raw",
        _REPOSITORY_ROOT
        / "doc/reports/vlaforge_cuda_matrix_v01/raw",
        _REPOSITORY_ROOT
        / "doc/reports/vlaforge_ablations_v01/raw",
    )
    committed_raw = [
        _file_record(path)
        for root in committed_raw_roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    paper_artifacts = [
        _file_record(_REPOSITORY_ROOT / relative)
        for relative in _PAPER_ARTIFACTS
    ]
    frozen = next(
        report
        for path, report in reports
        if path.name == "heldout_audit.json"
    )["frozen_core"]
    if not frozen["matches"]:
        raise ValueError("frozen core fingerprint no longer matches")

    must_archive = [item for item in archives if item["must_archive"]]
    report = {
        "schema": _SCHEMA,
        "status": "passed",
        "passed": True,
        "repository": {
            "baseline_revision": baseline_revision,
            "audited_revision": current_revision,
            "source_dirty": False,
            "frozen_core_revision": frozen["frozen"]["revision"],
            "frozen_core_sha256": frozen["frozen"]["combined_sha256"],
            "frozen_core_matches": frozen["matches"],
        },
        "environment": _environment(),
        "installed_wheel_artifact_evaluation": artifact_evaluation,
        "formal_reports": [_file_record(path) for path in formal_paths],
        "committed_raw_evidence": committed_raw,
        "paper_artifacts": paper_artifacts,
        "external_path_references": references,
        "reproduction_commands": commands,
        "external_archive_roots": archives,
        "summary": {
            "formal_report_count": len(formal_paths),
            "committed_raw_file_count": len(committed_raw),
            "paper_artifact_count": len(paper_artifacts),
            "reproduction_command_count": len(commands),
            "absolute_reference_count": len(references),
            "external_reference_count": sum(
                not item["inside_repository"] for item in references
            ),
            "missing_external_reference_count": sum(
                not item["inside_repository"] and not item["exists"]
                for item in references
            ),
            "must_archive_root_count": len(must_archive),
            "must_archive_missing_count": sum(
                not item["exists"] for item in must_archive
            ),
            "must_archive_total_bytes": sum(
                int(item["size_bytes"]) for item in must_archive
            ),
            "optional_archive_total_bytes": sum(
                int(item["size_bytes"])
                for item in archives
                if not item["must_archive"]
            ),
        },
        "claim_boundary": {
            "large_checkpoints_or_artifacts_committed": False,
            "binary_nsight_profiles_committed": False,
            "parsed_nsight_summaries_committed": True,
            "host_cuda": True,
            "orin": False,
        },
    }
    return report


def _gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def render_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    wheel = report["installed_wheel_artifact_evaluation"]
    lines = [
        "# VLAForge reproducibility manifest",
        "",
        f"Status: **{report['status']}**.",
        "",
        "## Frozen baseline",
        "",
        f"- Baseline revision: `{report['repository']['baseline_revision']}`",
        f"- Audited revision: `{report['repository']['audited_revision']}`",
        (
            "- Frozen core SHA256: "
            f"`{report['repository']['frozen_core_sha256']}`"
        ),
        "- Frozen core still matches: yes",
        (
            "- Installed-wheel CUDA target: "
            f"`{wheel['artifact_evaluation']['target']}`"
        ),
        (
            "- Installed-wheel package import: "
            f"`{wheel['isolation']['package_import']}`"
        ),
        "",
        "## Durable evidence in Git",
        "",
        f"- Formal JSON reports: {summary['formal_report_count']}",
        (
            "- Committed raw JSON/CSV/Nsight text summaries: "
            f"{summary['committed_raw_file_count']}"
        ),
        (
            "- Paper, Model Card and figure artifacts: "
            f"{summary.get('paper_artifact_count', 0)}"
        ),
        (
            "- Extracted reproduction commands: "
            f"{summary['reproduction_command_count']}"
        ),
        "- Large checkpoints, AOTI packages and binary profiler databases: "
        "not committed",
        "",
        "## External archive roots",
        "",
        "| Evidence root | Size | Archive? | Status | Reason |",
        "|---|---:|---|---|---|",
    ]
    for item in report["external_archive_roots"]:
        lines.append(
            f"| `{item['path']}` | {_gib(int(item['size_bytes']))} | "
            f"{'required' if item['must_archive'] else 'optional'} | "
            f"{'present' if item['exists'] else 'missing'} | "
            f"{item['reason']} |"
        )
    lines.extend(
        (
            "",
            (
                "Required external roots currently total "
                f"**{_gib(summary['must_archive_total_bytes'])}**. "
                "They must be archived outside Git to retain byte-for-byte "
                "real-model reproduction."
            ),
            (
                f"The audit found {summary['missing_external_reference_count']} "
                "missing ephemeral references. They remain disclosed in the "
                "JSON manifest; committed hashes and summaries are still "
                "available, while byte-for-byte reruns require regeneration "
                "or the external archive roots above."
            ),
            "",
            "## Claim boundary",
            "",
            "- This manifest covers Host-CUDA artifact reproducibility.",
            "- It does not contain Orin evidence.",
            "- The installed-wheel smoke uses a synthetic tensor Region and "
            "is not real-model support evidence.",
            "- Model kernels remain upstream AOTI/cuDNN/CUTLASS/Triton work.",
            "",
        )
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-evaluation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--baseline-revision",
        default="f0fc1be",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)

    report = audit(
        args.artifact_evaluation.resolve(),
        baseline_revision=args.baseline_revision,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
