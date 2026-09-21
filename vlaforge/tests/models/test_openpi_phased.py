"""Small tensor-fixture regressions; these are not pretrained OpenPI evidence."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
from vlaforge.adapters.openpi import openpi_phased as phased
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_frontend import OpenPIFrontend
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType
from vlaforge.numerical_context import snapshot
from vlaforge.validation.contracts import NumericContract


@pytest.fixture
def persisted(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import numpy as np

    value = TensorType((1, 2, 2), "f64")
    flag = TensorType((1,), "bool")
    state_type = TensorType((1, 2), "f32")

    class Identity(torch.nn.Module):
        def forward(self, noise):
            return noise.clone()

    region = tensor_region(
        "identity", inputs=(Value("noise", value),), outputs=(value,)
    )(Identity())
    builder = InvocationBuilder(
        "phased-serialization-fixture",
        inputs=(
            InputPort("noise", value),
            InputPort("accepted", flag),
            InputPort("state", state_type),
        ),
        outputs=(OutputPort("normalized_action_chunk", value),),
    )
    (result,) = builder.call(region, builder.input("noise"))
    program = builder.finish(
        {"normalized_action_chunk": result}, accepted=builder.input("accepted")
    )
    noise = torch.arange(4, dtype=torch.float64).reshape(1, 2, 2)
    frontend = OpenPIFrontend(program, {"identity": (noise,)}, noise, noise)
    original_load = torch.export.load
    monkeypatch.setattr(
        torch.export,
        "load",
        lambda *_a, **_kw: pytest.fail("persistence must never reload a weight owner"),
    )
    capture = phased.capture_and_persist_openpi(
        frontend,
        tmp_path / "exported_regions",
        contract=NumericContract(),
        on_progress=lambda _: None,
    )
    monkeypatch.setattr(torch.export, "load", original_load)
    np.savez(
        tmp_path / "prepared_inputs.npz",
        noise=noise.numpy(),
        accepted=np.ones(1, dtype=bool),
        state=np.zeros((1, 2), dtype=np.float32),
    )
    np.savez(
        tmp_path / "actions.npz",
        normalized_reference=noise.numpy(),
        physical_reference=noise.numpy()[0],
    )
    identity = {"pid": 2147483647, "start_ticks": 1}
    assert not Path(f"/proc/{identity['pid']}").exists()
    source = {
        "status": "persisted",
        "pid": identity["pid"],
        "process_identity": identity,
        "device": "cpu",
        "capture": capture,
        "prepared_inputs": file_digest(tmp_path / "prepared_inputs.npz"),
        "actions": file_digest(tmp_path / "actions.npz"),
        "numerical_context": snapshot().to_dict(),
        "tolerances": asdict(NumericContract()),
        "evidence_level": "fixture-regression-only",
        "processor_config": {"fixture": True},
        "checkpoint_provenance": {"fixture": True},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(source))
    monkeypatch.setattr(
        phased,
        "_native_output_transform",
        lambda _: lambda values: {"actions": values["actions"]},
    )
    return path, source


def test_persistence_is_not_reload_and_join_requires_independent_full_outputs(
    persisted,
):
    path, source = persisted
    original_bytes = path.read_bytes()
    assert source["capture"]["saved_reload_verified"] is False
    assert source["capture"]["regions"][0]["saved_reload_region_parity"] == "not-run"
    report = phased.validate_persisted_openpi(
        path, path.parent / "reload", capture_sha256=file_digest(path)["sha256"]
    )
    assert report["status"] == "passed"
    assert report["openpi_model_constructed"] is False
    assert len(report["fidelity"]) == 2
    assert all(
        item["metrics"]["maximum_absolute_error"] == 0 for item in report["fidelity"]
    )
    assert path.read_bytes() == original_bytes
    joined_path = path.parent / "validated_capture_report.json"
    joined = json.loads(joined_path.read_text())
    phased.verify_phased_capture_join(joined, joined_path)
    assert joined["source_evidence_level"] == "fixture-regression-only"
    (path.parent / "reload/actions.npz").write_bytes(b"corrupt output")
    with pytest.raises(ValueError, match="output hash"):
        phased.verify_phased_capture_join(joined, joined_path)


def test_compiler_does_not_admit_persisted_only_evidence(persisted):
    from vlaforge.adapters.openpi.openpi_aoti import _capture_source

    path, _ = persisted
    with pytest.raises(ValueError, match="passed real capture"):
        _capture_source(path, file_digest(path)["sha256"])


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "capture_status",
        "context",
        "archive",
        "examples",
        "metadata",
        "inputs",
        "actions",
        "duplicate",
        "active_worker",
        "missing_worker",
    ],
)
def test_independent_reloader_fails_closed(persisted, change):
    path, source = persisted
    if change == "status":
        source["status"] = "passed"
    elif change == "capture_status":
        source["capture"]["status"] = "passed"
    elif change == "context":
        source["capture"]["numerical_context"]["float32_matmul_precision"] = "medium"
    elif change in ("archive", "examples"):
        source["capture"]["regions"][0][change]["sha256"] = "0" * 64
    elif change == "metadata":
        source["capture"]["regions"][0]["example_metadata"]["items"]["args"]["items"][
            0
        ]["stride"] = [99, 2, 1]
    elif change in ("inputs", "actions"):
        source["prepared_inputs" if change == "inputs" else "actions"]["sha256"] = (
            "0" * 64
        )
    elif change == "duplicate":
        source["capture"]["regions"].append(source["capture"]["regions"][0])
    elif change == "active_worker":
        source["process_identity"] = phased.process_identity()
    else:
        source.pop("process_identity")
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        phased.validate_persisted_openpi(
            path, path.parent / "rejected", capture_sha256=file_digest(path)["sha256"]
        )
    assert not (path.parent / "validated_capture_report.json").exists()


@pytest.mark.parametrize(
    "field", ["numerical_context", "prepared_inputs", "device", "checkpoint_provenance"]
)
def test_join_cannot_relabel_original_provenance(persisted, field):
    path, _ = persisted
    phased.validate_persisted_openpi(
        path, path.parent / "reload", capture_sha256=file_digest(path)["sha256"]
    )
    joined_path = path.parent / "validated_capture_report.json"
    joined = json.loads(joined_path.read_text())
    joined[field] = "fabricated"
    with pytest.raises(ValueError):
        phased.verify_phased_capture_join(joined, joined_path)


def test_tensor_evidence_preserves_noncontiguous_dtype_and_offset(tmp_path):
    torch = pytest.importorskip("torch")
    value = torch.arange(40, dtype=torch.bfloat16).reshape(5, 8)[1:4, 1::2]
    original = phased.tensor_tree_metadata(value)
    path = tmp_path / "tensor.pt"
    torch.save({"value": value}, path)
    restored = torch.load(path, weights_only=True)["value"]
    assert phased.tensor_tree_metadata(restored) == original
    assert original["storage_offset"] == 9
    assert original["stride"] == [8, 2]
    torch.testing.assert_close(value, restored, atol=0, rtol=0, check_stride=True)


def test_process_identity_detects_actual_live_worker_and_pid_reuse(monkeypatch):
    identity = phased.process_identity()
    with pytest.raises(ValueError, match="must exit"):
        phased._require_original_worker_exited(identity)
    monkeypatch.setattr(
        phased,
        "process_identity",
        lambda _: {**identity, "start_ticks": identity["start_ticks"] + 1},
    )
    phased._require_original_worker_exited(identity)
