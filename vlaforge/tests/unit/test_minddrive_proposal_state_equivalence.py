from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch


def _load_validation_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "validate_real_minddrive_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_real_minddrive_pipeline",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state() -> tuple[torch.Tensor, ...]:
    row = torch.arange(600, dtype=torch.float32).reshape(1, 600, 1)
    return (
        torch.cat((row, -row), dim=-1),
        row.clone(),
        row.to(torch.float64),
        torch.tensor([7.0]),
    )


def test_proposal_state_equivalence_accepts_one_shared_row_permutation() -> None:
    module = _load_validation_module()
    reference = _state()
    order = torch.arange(600)
    order[1], order[2] = order[2].clone(), order[1].clone()
    candidate = (
        reference[0][:, order],
        reference[1][:, order],
        reference[2][:, order],
        reference[3].clone(),
    )

    evidence = module._proposal_state_equivalence(
        reference,
        candidate,
        state_names=("embedding", "reference", "timestamp", "sample_time"),
        identity_index=1,
        proposal_field_count=3,
        maximum_absolute_error=0.0,
        normalized_root_mean_square_error=0.0,
        maximum_assignment_distance=0.0,
        enforce=True,
    )

    assert evidence["passed"]
    assert evidence["partitions"][0]["permuted_row_count"] == 2
    assert evidence["partitions"][0]["maximum_rank_displacement"] == 1


def test_proposal_state_equivalence_rejects_inconsistent_bundle_field() -> None:
    module = _load_validation_module()
    reference = _state()
    order = torch.arange(600)
    order[1], order[2] = order[2].clone(), order[1].clone()
    bad_embedding = reference[0][:, order].clone()
    bad_embedding[0, 1, 0] += 1.0
    candidate = (
        bad_embedding,
        reference[1][:, order],
        reference[2][:, order],
        reference[3].clone(),
    )

    evidence = module._proposal_state_equivalence(
        reference,
        candidate,
        state_names=("embedding", "reference", "timestamp", "sample_time"),
        identity_index=1,
        proposal_field_count=3,
        maximum_absolute_error=0.0,
        normalized_root_mean_square_error=0.0,
        maximum_assignment_distance=0.0,
        enforce=True,
    )

    assert not evidence["passed"]
    assert not evidence["fields"]["embedding"]["passed"]
    assert evidence["fields"]["reference"]["passed"]


def test_proposal_state_equivalence_rejects_distant_assignment() -> None:
    module = _load_validation_module()
    reference = _state()
    candidate = tuple(value.clone() for value in reference)
    candidate[1][:, :300] += 0.25

    evidence = module._proposal_state_equivalence(
        reference,
        candidate,
        state_names=("embedding", "reference", "timestamp", "sample_time"),
        identity_index=1,
        proposal_field_count=3,
        maximum_absolute_error=1.0,
        normalized_root_mean_square_error=1.0,
        maximum_assignment_distance=0.1,
        enforce=True,
    )

    assert not evidence["passed"]
    assert not evidence["partitions"][0]["assignment_within_threshold"]
    assert evidence["partitions"][1]["assignment_within_threshold"]
