from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_paper_completion.py"
    )
    specification = importlib.util.spec_from_file_location(
        "audit_paper_completion",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    matrix = _write_json(
        tmp_path / "matrix.json",
        {
            "schema": "matrix/1",
            "status": "passed",
            "passed": True,
            "cells": [
                {
                    "steady_latency": {
                        "mean_ns": {"estimate": 1.0, "ci95": [0.9, 1.1]},
                        "p50_ns": {"estimate": 1.0, "ci95": [0.9, 1.1]},
                        "p90_ns": {"estimate": 1.1, "ci95": [1.0, 1.2]},
                        "p99_ns": {"estimate": 1.2, "ci95": [1.1, 1.3]},
                        "process_mean_stddev_ns": 0.1,
                    }
                }
            ],
            "summary": {
                "cell_count": 30,
                "task_count": 150,
                "minimum_workloads_per_model": 5,
                "minimum_processes_per_cell": 5,
                "all_output_parity_passed": True,
            },
        },
    )
    ablations = _write_json(
        tmp_path / "ablations.json",
        {
            "schema": "ablations/1",
            "status": "passed",
            "passed": True,
            "summary": {
                "ablation_count": 4,
                "exact_reuse_task_count": 40,
                "all_transaction_failure_retry_passed": True,
                "clean_wheel_boundary_passed": True,
            },
            "exact_reuse": {},
            "static_arena": {},
            "transaction_failure_retry": {},
            "deployment_boundary": {},
        },
    )
    heldout = _write_json(
        tmp_path / "heldout.json",
        {
            "schema": "heldout/1",
            "status": "passed",
            "passed": True,
            "model": "AutoVLA",
            "evidence_level": "L2-partitioned-real-checkpoint-frontend",
            "semantic_ir": {"core_op_delta": 0},
            "repository": {"source_dirty": False},
            "exact_reuse": {
                "semantic": {"hits": 1, "misses": 2},
            },
            "correctness": {"semantic_plan_trace_exact": True},
        },
    )
    release = _write_json(
        tmp_path / "release.json",
        {
            "schema": "release/1",
            "status": "passed",
            "passed": True,
            "summary": {
                "release_gate_passed": True,
                "python_passed": 235,
                "cpu_ctest_passed": 8,
                "cuda_ctest_passed": 9,
                "cuda_aoti_opt_in_passed": 1,
                "wheel_install_bundle_passed": True,
                "no_python_runner": True,
            },
            "clean_build": {
                "old_edgefm_or_custom_cuda_sources_compiled": False,
            },
        },
    )
    paper = tmp_path / "paper.md"
    module = _module()
    paper.write_text(
        "\n\n".join(
            f"{section}\n\nEvidence-backed text."
            for section in module.REQUIRED_PAPER_SECTIONS
        ),
        encoding="utf-8",
    )
    large_text = "evidence\n" * 100
    model_card = tmp_path / "model-card.md"
    claim_map = tmp_path / "claim-map.md"
    artifact_readme = tmp_path / "artifact-readme.md"
    for path in (model_card, claim_map, artifact_readme):
        path.write_text(large_text, encoding="utf-8")
    figures = tmp_path / "figures"
    figures.mkdir()
    for name in module.REQUIRED_FIGURES:
        (figures / name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
            encoding="utf-8",
        )
    return {
        "matrix_path": matrix,
        "ablations_path": ablations,
        "heldout_path": heldout,
        "release_gate_path": release,
        "paper_path": paper,
        "model_card_path": model_card,
        "claim_map_path": claim_map,
        "artifact_readme_path": artifact_readme,
        "figures_dir": figures,
    }


def test_submission_gate_excludes_optional_hardware_and_vehicle_work(
    tmp_path: Path,
) -> None:
    module = _module()
    report = module.audit(**_fixtures(tmp_path))

    assert report["submission_ready"] is True
    assert report["summary"]["heldout_evidence_level"].startswith("L2")
    assert report["claim_boundary"] == {
        "measured_platform": "RTX 3060 sm_86 / CUDA 12.8",
        "host_cuda_only": True,
        "orin_required": False,
        "real_vehicle_or_sensor_stack_required": False,
        "openvla_l4_required": False,
        "cross_gpu_required": False,
        "second_machine_required": False,
        "legacy_edgefm_cuda_kernel_work_required": False,
    }


def test_submission_gate_rejects_fixture_only_heldout(
    tmp_path: Path,
) -> None:
    module = _module()
    paths = _fixtures(tmp_path)
    heldout = json.loads(paths["heldout_path"].read_text(encoding="utf-8"))
    heldout["evidence_level"] = "L1-fixture"
    paths["heldout_path"].write_text(json.dumps(heldout), encoding="utf-8")

    with pytest.raises(ValueError, match="did not reach L2"):
        module.audit(**paths)
