import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def tool(monkeypatch):
    folder = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("rdt_torchscript_tool", folder / "build_real_rdt_torchscript.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mutation", [None, "artifact", "source", "variant", "jit", "device", "case", "missing_cases", "helper", "status"])
def test_torchscript_candidate_binds_export_profile_helper_and_actual_validation(tool, tmp_path, mutation):
    from vlaforge.deployment import torchscript_export

    folder = tmp_path / "artifacts"
    folder.mkdir()
    artifact = folder / "region.pt"
    artifact.write_bytes(b"CPU metadata fixture, not an executable")
    identity = {"sha256": tool.digest(artifact), "size_bytes": artifact.stat().st_size}
    audit = {"status": "region_cases_passed", "backend_variant": "torchscript-aten/1",
             "jit_optimization": False, "archive_device_policy": "preserve",
             "implementation_sha256": tool.digest(Path(torchscript_export.__file__)),
             "artifact_sha256": identity["sha256"], "validation_cases": [{"bitwise_equal": True}]}
    report = {"status": "region_cases_passed", "backend": "torchscript", "region_validation": audit,
              "exported_program": {"sha256": "captured"}, "artifact": identity}
    if mutation == "artifact":
        artifact.write_bytes(b"changed")
    elif mutation == "source":
        report["exported_program"]["sha256"] = "other-export"
    elif mutation == "variant":
        audit["backend_variant"] = "unverified-variant"
    elif mutation == "jit":
        audit["jit_optimization"] = True
    elif mutation == "device":
        audit["archive_device_policy"] = "relocate"
    elif mutation == "case":
        audit["validation_cases"][0]["bitwise_equal"] = False
    elif mutation == "missing_cases":
        audit["validation_cases"] = []
    elif mutation == "helper":
        audit["implementation_sha256"] = "older-helper"
    elif mutation == "status":
        audit["status"] = "traced-not-validated"
    tool.write(folder / "region.compile.json", report)
    captured = {"name": "region", "export": {"sha256": "captured"}}
    if mutation is None:
        assert tool.verify_archive(tmp_path, captured) == (report, artifact)
    else:
        with pytest.raises(ValueError, match="export/profile/implementation"):
            tool.verify_archive(tmp_path, captured)
