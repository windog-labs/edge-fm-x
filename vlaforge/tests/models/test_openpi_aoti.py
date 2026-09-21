"""Artifact provenance and tensor-metric unit tests, not real model execution."""

from argparse import Namespace
import json
from pathlib import Path

import pytest

from vlaforge.adapters.openpi.openpi_aoti import (
    _canonical,
    _region_result,
    _tensor_metrics,
    compile_regions,
    verify_compiled_region,
)
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.deployment.aoti_profile import aoti_configs


def test_policy_comparison_preserves_scalar_types():
    assert _canonical({"flag": True}) != _canonical({"flag": 1})


def test_single_region_result_is_a_tensor_not_a_nested_tuple():
    torch = pytest.importorskip("torch")
    value = torch.ones(())
    assert _region_result((value,)) is value
    assert _region_result((value, value)) == (value, value)


def test_missing_native_tensor_is_a_contract_error():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="must return a Tensor"):
        _tensor_metrics(torch.ones(()), None)


def test_compile_failure_is_reported_before_allocating_model(tmp_path):
    pytest.importorskip("torch")
    source = tmp_path / "capture.json"
    source.write_text("{}")
    args = Namespace(
        output_dir=tmp_path / "compile",
        capture_report=source,
        capture_sha256="0" * 64,
        regions=["region"],
        profile="eager-numerics",
    )
    with pytest.raises(ValueError, match="report SHA256"):
        compile_regions(args)
    report = json.loads((args.output_dir / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["numeric_parity_verified"] is False
    assert report["source_files"]["cli.py"]["size_bytes"] > 0
    assert report["orchestrator_source"]["size_bytes"] > 0


@pytest.mark.parametrize("dtype_name", ["bool", "float32", "bfloat16", "int64"])
def test_artifact_metrics_keep_dtype_and_zero_error(dtype_name):
    torch = pytest.importorskip("torch")
    value = torch.ones(2, dtype=getattr(torch, dtype_name))
    result = _tensor_metrics(value, value.clone())
    assert result["exact"] and result["finite"]
    assert result["mse"] == result["max_abs"] == 0
    with pytest.raises(ValueError, match="shape or dtype"):
        _tensor_metrics(value, value.to(torch.float64))


def _record(tmp_path):
    artifact = tmp_path / "weights.pt2"
    artifact.write_bytes(b"transport-unit-test-not-an-AOTI-package")
    digest = file_digest(artifact)
    source_sha = "1" * 64
    record = {
        "status": "passed",
        "target": "sm_90",
        "inductor_profile": "eager-numerics",
        "inductor_configs": aoti_configs("eager-numerics"),
        "backend_graph_passes": [],
        "backend_program_audit": {"passes": [], "rewrites": []},
        "backend_package_audit": {"passes": []},
        "exported_program": {"sha256": source_sha},
        "artifact": {
            "path": str(artifact),
            "sha256": digest["sha256"],
            "size_bytes": digest["size_bytes"],
        },
    }
    path = tmp_path / "compile.json"
    path.write_text(json.dumps(record))
    return path, record, {"archive": {"sha256": source_sha}}


@pytest.mark.parametrize(
    "field",
    [
        "target",
        "source",
        "size",
        "bytes",
        "config_type",
        "pass",
        "missing_pre_audit",
        "pre_pass",
        "pre_rewrites",
        "pre_audit_type",
    ],
)
def test_artifact_chain_rejects_tampering(tmp_path, monkeypatch, field):
    import vlaforge.adapters.openpi.openpi_aoti as audit

    path, record, region = _record(tmp_path)
    monkeypatch.setattr(audit, "verify_package_audit", lambda *a, **kw: None)
    verify_compiled_region(path, region, target="sm_90", profile="eager-numerics")
    if field == "target":
        record["target"] = "sm_80"
    elif field == "source":
        record["exported_program"]["sha256"] = "2" * 64
    elif field == "size":
        record["artifact"]["size_bytes"] += 1
    elif field == "bytes":
        Path(record["artifact"]["path"]).write_bytes(b"changed")
    elif field == "config_type":
        record["inductor_configs"]["force_same_precision"] = 1
    elif field == "missing_pre_audit":
        record.pop("backend_program_audit")
    elif field == "pre_pass":
        record["backend_program_audit"]["passes"] = [{"name": "unrecorded_pass"}]
    elif field == "pre_rewrites":
        record["backend_program_audit"]["rewrites"] = None
    elif field == "pre_audit_type":
        record["backend_program_audit"] = []
    else:
        record["backend_graph_passes"] = [{"name": "unrecorded_pass"}]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="evidence mismatch"):
        verify_compiled_region(path, region, target="sm_90", profile="eager-numerics")


def test_aten_preserving_requires_current_pre_aot_ledger(tmp_path, monkeypatch):
    import vlaforge.adapters.openpi.openpi_aoti as audit
    from vlaforge.deployment.aoti_export import (
        backend_pass_records,
        backend_program_pass_records,
    )
    from vlaforge.deployment.aoti_package import package_pass_records

    path, record, region = _record(tmp_path)
    configs = aoti_configs("aten-preserving")
    record.update(
        inductor_profile="aten-preserving",
        inductor_configs=configs,
        backend_graph_passes=backend_pass_records(configs),
        backend_program_audit={
            "passes": backend_program_pass_records(configs),
            "rewrites": [],
        },
        backend_package_audit={"passes": package_pass_records(configs)},
    )
    assert record["backend_program_audit"]["passes"]
    monkeypatch.setattr(audit, "verify_package_audit", lambda *a, **kw: None)
    path.write_text(json.dumps(record))
    verify_compiled_region(path, region, target="sm_90", profile="aten-preserving")
    record["backend_program_audit"]["passes"] = []
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="evidence mismatch"):
        verify_compiled_region(path, region, target="sm_90", profile="aten-preserving")
