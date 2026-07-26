#!/usr/bin/env python3
"""Mechanically audit the Host-CUDA paper completion boundary.

This audit deliberately does not require Orin, a real vehicle, sensor
middleware, OpenVLA L4, cross-GPU measurements, or a second-machine rerun.
Those are useful extensions, but they are outside the submission gate agreed
for the RTX 3060 Host-CUDA paper artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_SCHEMA = "vlaforge.paper_completion_audit/1"
REQUIRED_PAPER_SECTIONS = (
    "## Abstract",
    "## 1. Introduction",
    "## 2. Motivation and Problem Definition",
    "## 3. Design",
    "## 4. Implementation",
    "## 5. Evaluation Methodology",
    "## 6. Results",
    "## 8. Related Work",
    "## 9. Limitations",
    "## 10. Reproducibility and Artifact Evaluation",
    "## 11. Claim Boundary",
)
REQUIRED_FIGURES = (
    "architecture.svg",
    "performance.svg",
    "ablations.svg",
)
OPTIONAL_NOT_REQUIRED = (
    "Orin latency, power, thermal, SM87, or JetPack evidence",
    "real-vehicle or sensor closed-loop integration",
    "ROS/Cyber, periodic scheduling, dropped-frame, or publish logic",
    "OpenVLA real L4",
    "cross-GPU performance",
    "second-machine independent artifact reproduction",
    "legacy EdgeFM CUDA kernel compilation or optimization",
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _evidence_record(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema": report.get("schema"),
        "status": report.get("status"),
    }


def _audit_matrix(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    cells = report.get("cells", ())
    _require(report.get("passed") is True, "CUDA matrix did not pass")
    _require(
        int(summary.get("cell_count", 0)) >= 30,
        "CUDA matrix must contain at least 30 path/model/workload cells",
    )
    _require(
        int(summary.get("task_count", 0)) >= 150,
        "CUDA matrix must contain at least 150 fresh-process tasks",
    )
    _require(
        int(summary.get("minimum_workloads_per_model", 0)) >= 5,
        "CUDA matrix must cover at least five workloads per model",
    )
    _require(
        int(summary.get("minimum_processes_per_cell", 0)) >= 5,
        "CUDA matrix must use at least five processes per cell",
    )
    _require(
        summary.get("all_output_parity_passed") is True,
        "CUDA matrix output parity failed",
    )
    _require(bool(cells), "CUDA matrix contains no statistic cells")
    steady = cells[0].get("steady_latency", {})
    required_statistics = {
        "mean_ns",
        "p50_ns",
        "p90_ns",
        "p99_ns",
        "process_mean_stddev_ns",
    }
    _require(
        required_statistics.issubset(steady)
        and all(
            "ci95" in steady[name]
            for name in ("mean_ns", "p50_ns", "p90_ns", "p99_ns")
        ),
        "CUDA matrix is missing required statistics",
    )
    return {
        "passed": True,
        "cells": int(summary["cell_count"]),
        "tasks": int(summary["task_count"]),
        "statistics": [
            "mean",
            "p50",
            "p90",
            "p99",
            "process-mean standard deviation",
            "process-cluster bootstrap 95% CI",
        ],
    }


def _audit_ablations(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    _require(report.get("passed") is True, "paper ablations did not pass")
    _require(
        int(summary.get("ablation_count", 0)) >= 4,
        "fewer than four formal ablations were recorded",
    )
    _require(
        int(summary.get("exact_reuse_task_count", 0)) >= 40,
        "exact-reuse ablation has insufficient process coverage",
    )
    _require(
        summary.get("all_transaction_failure_retry_passed") is True,
        "transaction failure/retry ablation failed",
    )
    _require(
        summary.get("clean_wheel_boundary_passed") is True,
        "clean-wheel deployment-boundary ablation failed",
    )
    for key in (
        "exact_reuse",
        "static_arena",
        "transaction_failure_retry",
        "deployment_boundary",
    ):
        _require(key in report, f"missing formal ablation: {key}")
    return {
        "passed": True,
        "formal_ablations": [
            "exact reuse",
            "static arena",
            "transaction/failure/retry",
            "deployment boundary",
        ],
    }


def _audit_heldout(report: Mapping[str, Any]) -> dict[str, Any]:
    level = str(report.get("evidence_level", ""))
    semantic_ir = report.get("semantic_ir", {})
    repository = report.get("repository", {})
    _require(report.get("passed") is True, "held-out model audit did not pass")
    _require(
        level.startswith(("L2", "L3", "L4")),
        "held-out real model did not reach L2",
    )
    _require(
        int(semantic_ir.get("core_op_delta", -1)) == 0,
        "held-out model changed the frozen core op set",
    )
    _require(
        repository.get("source_dirty") is False,
        "held-out formal report was produced from a dirty source tree",
    )
    exact_reuse = report.get("exact_reuse", {})
    semantic = exact_reuse.get("semantic", {})
    _require(
        int(semantic.get("hits", 0)) >= 1
        and int(semantic.get("misses", 0)) >= 2,
        "held-out InputRevision hit/miss evidence is incomplete",
    )
    correctness = report.get("correctness", {})
    _require(
        correctness.get("semantic_plan_trace_exact") is True,
        "held-out Semantic/Plan trace parity failed",
    )
    return {
        "passed": True,
        "model": report.get("model"),
        "evidence_level": level,
        "core_op_delta": 0,
        "partitioned": "partition" in level.lower(),
    }


def _audit_release_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {})
    clean = report.get("clean_build", {})
    _require(
        report.get("passed") is True
        and summary.get("release_gate_passed") is True,
        "final Host-CUDA release gate did not pass",
    )
    _require(
        int(summary.get("python_passed", 0)) >= 200,
        "final Python test suite is incomplete",
    )
    _require(
        int(summary.get("cpu_ctest_passed", 0)) == 7,
        "CPU CTest gate is not 7/7",
    )
    _require(
        int(summary.get("cuda_ctest_passed", 0)) == 8,
        "CUDA CTest gate is not 8/8",
    )
    _require(
        int(summary.get("cuda_aoti_opt_in_passed", 0)) == 1,
        "live CUDA AOTI opt-in gate did not pass",
    )
    _require(
        summary.get("wheel_install_bundle_passed") is True
        and summary.get("no_python_runner") is True,
        "installed-wheel/no-Python artifact gate did not pass",
    )
    _require(
        clean.get("old_edgefm_or_custom_cuda_sources_compiled") is False,
        "final gate compiled an old EdgeFM/custom CUDA source",
    )
    return {
        "passed": True,
        "python_passed": int(summary["python_passed"]),
        "cpu_ctest": "7/7",
        "cuda_ctest": "8/8",
        "live_cuda_aoti": True,
        "clean_wheel_no_python": True,
        "legacy_cuda_kernels_compiled": False,
    }


def _audit_paper(
    paper: Path,
    model_card: Path,
    claim_map: Path,
    artifact_readme: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    paper_text = paper.read_text(encoding="utf-8")
    missing_sections = [
        section
        for section in REQUIRED_PAPER_SECTIONS
        if section not in paper_text
    ]
    _require(
        not missing_sections,
        f"paper draft is missing sections: {missing_sections}",
    )
    for path in (model_card, claim_map, artifact_readme):
        _require(path.stat().st_size >= 512, f"incomplete paper artifact: {path}")
    figure_records = []
    for name in REQUIRED_FIGURES:
        path = figures_dir / name
        _require(path.is_file(), f"missing required paper figure: {path}")
        text = path.read_text(encoding="utf-8")
        _require("<svg" in text, f"invalid SVG paper figure: {path}")
        figure_records.append(
            {"name": name, "sha256": _sha256(path), "size_bytes": path.stat().st_size}
        )
    return {
        "passed": True,
        "paper": str(paper),
        "model_card": str(model_card),
        "claim_evidence_map": str(claim_map),
        "artifact_readme": str(artifact_readme),
        "figures": figure_records,
    }


def audit(
    *,
    matrix_path: Path,
    ablations_path: Path,
    heldout_path: Path,
    release_gate_path: Path,
    paper_path: Path,
    model_card_path: Path,
    claim_map_path: Path,
    artifact_readme_path: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    matrix = _json(matrix_path)
    ablations = _json(ablations_path)
    heldout = _json(heldout_path)
    release_gate = _json(release_gate_path)
    evidence = {
        "cuda_matrix": _evidence_record(matrix_path, matrix),
        "formal_ablations": _evidence_record(ablations_path, ablations),
        "heldout_real_model": _evidence_record(heldout_path, heldout),
        "final_release_gate": _evidence_record(release_gate_path, release_gate),
    }
    gates = {
        "performance_matrix": _audit_matrix(matrix),
        "formal_ablations": _audit_ablations(ablations),
        "heldout_real_model": _audit_heldout(heldout),
        "final_python_cpp_cuda": _audit_release_gate(release_gate),
        "paper_artifact": _audit_paper(
            paper_path,
            model_card_path,
            claim_map_path,
            artifact_readme_path,
            figures_dir,
        ),
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "passed": True,
        "submission_ready": True,
        "paper_positioning": (
            "Stateful Invocation Whole-Program Compilation for VLA Deployment"
        ),
        "required_host_cuda_gates": gates,
        "evidence": evidence,
        "claim_boundary": {
            "measured_platform": "RTX 3060 sm_86 / CUDA 12.8",
            "host_cuda_only": True,
            "orin_required": False,
            "real_vehicle_or_sensor_stack_required": False,
            "openvla_l4_required": False,
            "cross_gpu_required": False,
            "second_machine_required": False,
            "legacy_edgefm_cuda_kernel_work_required": False,
        },
        "optional_not_required": list(OPTIONAL_NOT_REQUIRED),
        "summary": {
            "submission_ready": True,
            "required_gate_count": len(gates),
            "required_gate_passed": len(gates),
            "heldout_evidence_level": gates["heldout_real_model"][
                "evidence_level"
            ],
            "core_op_delta": gates["heldout_real_model"]["core_op_delta"],
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    gates = report["required_host_cuda_gates"]
    heldout = gates["heldout_real_model"]
    release = gates["final_python_cpp_cuda"]
    lines = [
        "# VLAForge paper completion audit",
        "",
        "Status: **submission-ready**.",
        "",
        "## Required Host-CUDA gates",
        "",
        "| Gate | Result |",
        "|---|---|",
        "| 5-workload × 5-process CUDA matrix | passed |",
        "| Four formal contribution ablations | passed |",
        (
            f"| Held-out real model | {heldout['model']}, "
            f"`{heldout['evidence_level']}`, core op delta 0 |"
        ),
        (
            f"| Final Python/C++/CUDA gate | "
            f"{release['python_passed']} Python tests; CPU 7/7; "
            "CUDA 8/8; live AOTI passed |"
        ),
        "| Clean installed-wheel no-Python artifact | passed |",
        "| Paper, figures, Model Card, claim map, artifact README | passed |",
        "",
        "## Completion boundary",
        "",
        (
            "The current paper is complete for the measured RTX 3060 `sm_86` "
            "and CUDA 12.8 Host-CUDA scope. The following remain optional "
            "extensions, not submission blockers:"
        ),
        "",
    ]
    lines.extend(f"- {item}" for item in report["optional_not_required"])
    lines.extend(
        [
            "",
            "The paper must not generalize the measured performance beyond "
            "this platform.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--ablations", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--paper", type=Path, required=True)
    parser.add_argument("--model-card", type=Path, required=True)
    parser.add_argument("--claim-map", type=Path, required=True)
    parser.add_argument("--artifact-readme", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit(
        matrix_path=args.matrix.resolve(),
        ablations_path=args.ablations.resolve(),
        heldout_path=args.heldout.resolve(),
        release_gate_path=args.release_gate.resolve(),
        paper_path=args.paper.resolve(),
        model_card_path=args.model_card.resolve(),
        claim_map_path=args.claim_map.resolve(),
        artifact_readme_path=args.artifact_readme.resolve(),
        figures_dir=args.figures_dir.resolve(),
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
