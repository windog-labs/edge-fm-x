import hashlib
import json

import pytest
import torch

from vlaforge.adapters.smolvla_artifact import (
    _metrics,
    _verify_artifact_chain,
)


def test_smolvla_artifact_metrics_distinguish_exact_and_tolerant_parity():
    expected = torch.tensor([1.0, 2.0], dtype=torch.float32)
    exact = _metrics(expected, expected.clone())
    approximate = _metrics(
        expected, torch.tensor([1.0, 2.01], dtype=torch.float32)
    )
    assert exact.exact
    assert exact.maximum_absolute_error == 0.0
    assert not approximate.exact
    assert approximate.maximum_absolute_error == pytest.approx(0.01)
    assert approximate.normalized_root_mean_square_error > 0.0


def test_smolvla_artifact_chain_authenticates_exports_and_packages(tmp_path):
    exports = tmp_path / "exports"
    artifacts = tmp_path / "artifacts"
    exports.mkdir()
    artifacts.mkdir()
    for index, region in enumerate(
        ("prepare_prefix", "solver_step", "trim_action_chunk")
    ):
        exported = exports / f"{region}.pt2e"
        artifact = artifacts / f"{region}.pt2"
        exported.write_bytes(f"export-{region}".encode())
        artifact.write_bytes(f"artifact-{region}".encode())
        manifest = {
            "schema": "vlaforge.compile_artifact_result/1",
            "status": "passed",
            "target": "sm_86",
            "graph_nodes": index + 1,
            "compile_seconds": float(index + 1),
            "exported_program": {
                "sha256": hashlib.sha256(exported.read_bytes()).hexdigest()
            },
            "artifact": {
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "size_bytes": artifact.stat().st_size,
            },
        }
        (artifacts / f"{region}.compile.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    export_records, artifact_records = _verify_artifact_chain(
        exports, artifacts
    )
    assert len(export_records) == len(artifact_records) == 3
    assert artifact_records[0]["target"] == "sm_86"

    (artifacts / "solver_step.pt2").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="digest mismatch"):
        _verify_artifact_chain(exports, artifacts)
