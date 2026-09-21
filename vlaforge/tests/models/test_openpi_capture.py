"""Serialization/binding regression utilities, not pretrained VLA model evidence."""

from dataclasses import asdict
import json

import pytest

from vlaforge.adapters.openpi.openpi_capture import (
    _saved_device,
    capture_and_reload_openpi,
    replay_saved_capture,
)
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_frontend import OpenPIFrontend, OpenPIInput
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType
from vlaforge.validation.contracts import NumericContract
from vlaforge.numerical_context import snapshot


def test_saved_reload_uses_the_existing_typed_tensor_binding_contract(tmp_path):
    torch = pytest.importorskip("torch")
    value = TensorType((1, 2, 2), "f64")
    flag = TensorType((1,), "bool")

    class Identity(torch.nn.Module):
        def forward(self, noise):
            return noise.clone()

    identity = tensor_region(
        "identity", inputs=(Value("noise", value),), outputs=(value,)
    )(Identity())
    builder = InvocationBuilder(
        "serialization-regression",
        inputs=(InputPort("noise", value), InputPort("accepted", flag)),
        outputs=(OutputPort("normalized_action_chunk", value),),
    )
    (result,) = builder.call(identity, builder.input("noise"))
    program = builder.finish(
        {"normalized_action_chunk": result}, accepted=builder.input("accepted")
    )
    noise = torch.arange(4, dtype=torch.float64).reshape(1, 2, 2)
    prepared = OpenPIInput(
        None, {"noise": noise, "accepted": torch.ones(1, dtype=torch.bool)}, 0
    )
    frontend = OpenPIFrontend(program, {"identity": (noise,)}, noise, noise)
    reports = []
    report, output = capture_and_reload_openpi(
        frontend,
        prepared,
        tmp_path / "export",
        contract=NumericContract(),
        on_progress=reports.append,
    )
    assert report["status"] == "passed"
    assert report["regions"][0]["saved_reload_region_parity"] == "passed"
    assert report["full_chunk_fidelity"]["metrics"]["maximum_absolute_error"] == 0
    torch.testing.assert_close(output, noise, atol=0, rtol=0)
    import numpy as np

    np.savez(
        tmp_path / "prepared_inputs.npz",
        noise=noise.numpy(),
        accepted=prepared.tensors["accepted"].numpy(),
    )
    np.savez(tmp_path / "actions.npz", normalized_reference=noise.numpy())
    (tmp_path / "export").rename(tmp_path / "exported_regions")
    for entry in report["regions"]:
        entry["archive"]["path"] = str(
            tmp_path / "exported_regions" / (entry["region"] + ".pt2")
        )
    source = {
        "status": "fixture-capture-only",
        "device": "cpu",
        "capture": report,
        "prepared_inputs": file_digest(tmp_path / "prepared_inputs.npz"),
        "actions": file_digest(tmp_path / "actions.npz"),
        "tolerances": asdict(NumericContract()),
        "numerical_context": snapshot().to_dict(),
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source))
    digest = file_digest(tmp_path / "exported_regions/invocation_ir.json")["sha256"]
    restored = replay_saved_capture(path, tmp_path / "replay", ir_sha256=digest)
    assert restored["status"] == "passed"
    assert restored["openpi_constructed_in_replay"] is False
    assert restored["numerical_execution"]["caller_policy_restoration_verified"] is True
    assert (
        report["regions"][0]["capture_evidence"]["observed_numerical_context"]
        == snapshot().to_dict()
    )
    with pytest.raises(ValueError, match="IR hash"):
        replay_saved_capture(path, tmp_path / "rejected", ir_sha256="0" * 64)
    assert not (tmp_path / "rejected").exists()
    wrong = snapshot().to_dict()
    wrong["cudnn_benchmark"] = not wrong["cudnn_benchmark"]
    wrong_path = tmp_path / "wrong-context.json"
    wrong_path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="differs"):
        replay_saved_capture(
            path,
            tmp_path / "wrong-policy",
            ir_sha256=digest,
            numerical_context=wrong_path,
        )
    source.pop("numerical_context")
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="explicit"):
        replay_saved_capture(path, tmp_path / "missing-policy", ir_sha256=digest)
    correct_path = tmp_path / "context.json"
    correct_path.write_text(snapshot().to_json())
    with pytest.raises(ValueError, match="explicit"):
        replay_saved_capture(
            path,
            tmp_path / "unacknowledged-legacy",
            ir_sha256=digest,
            numerical_context=correct_path,
        )
    legacy = replay_saved_capture(
        path,
        tmp_path / "legacy-replay",
        ir_sha256=digest,
        numerical_context=correct_path,
        allow_legacy_context=True,
    )
    assert (
        legacy["numerical_execution"]["origin"]
        == "explicit-complete-context-for-legacy-report"
    )


@pytest.mark.parametrize("device", [None, "cuda", "cuda:-1", "mps", "cuda:0 extra", 0])
def test_saved_device_rejects_missing_or_ambiguous_binding(device):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="explicit"):
        _saved_device({"device": device})


def test_saved_device_rejects_unavailable_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        _saved_device({"device": "cuda:0"})
