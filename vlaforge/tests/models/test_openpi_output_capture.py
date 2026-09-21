"""Synthetic archive contract tests, not model or robot validation."""

import numpy as np
import pytest
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_output_capture import (
    checked_output_rejections,
    output_reference,
    processor_config_for_output,
)


def archives(root, *, state=None, normalized=None, native=None):
    np.savez(root / "actions.npz",
             normalized_reference=np.zeros((1, 3, 8), dtype=np.float32) if normalized is None else normalized,
             physical_reference=np.zeros((3, 4), dtype=np.float64) if native is None else native)
    np.savez(root / "prepared_inputs.npz", state=np.zeros((1, 8), dtype=np.float64) if state is None else state)
    return {"actions": file_digest(root / "actions.npz"), "prepared_inputs": file_digest(root / "prepared_inputs.npz")}


def test_preserves_complete_dtypes_and_horizon(tmp_path):
    source = archives(tmp_path)
    state, normalized, native = output_reference(source, tmp_path)
    assert state.dtype == native.dtype == np.float64
    assert normalized.dtype == np.float32
    assert native.shape == (1, 3, 4)
    assert normalized.shape == (1, 3, 8)


@pytest.mark.parametrize("field,value", [
    ("state", np.zeros((2, 8), dtype=np.float32)),
    ("state", np.zeros((8,), dtype=np.float32)),
    ("state", np.full((1, 8), np.inf, dtype=np.float32)),
    ("normalized", np.zeros((1, 2, 8), dtype=np.float32)),
    ("normalized", np.zeros((2, 3, 8), dtype=np.float32)),
    ("normalized", np.zeros((3, 8), dtype=np.float32)),
    ("native", np.zeros((1, 3, 4), dtype=np.float64)),
    ("native", np.full((3, 4), np.nan, dtype=np.float64)),
])
def test_rejects_incomplete_or_nonfinite_tensors(tmp_path, field, value):
    with pytest.raises(ValueError, match="dimensions or finiteness"):
        output_reference(archives(tmp_path, **{field: value}), tmp_path)


@pytest.mark.parametrize("field,shape", [("state", (1, 8)), ("normalized", (1, 3, 8)), ("native", (3, 4))])
def test_rejects_integer_tensor_reference(tmp_path, field, shape):
    with pytest.raises(TypeError, match="floating"):
        output_reference(archives(tmp_path, **{field: np.zeros(shape, dtype=np.int64)}), tmp_path)


@pytest.mark.parametrize("key", ["actions", "prepared_inputs"])
def test_rejects_wrong_archive_identity(tmp_path, key):
    source = archives(tmp_path)
    source[key]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="archive differs"):
        output_reference(source, tmp_path)


@pytest.mark.parametrize("broken", ["incoming", "source", None])
def test_rejection_probe_checks_discarded_source_values_and_original_predicate(broken):
    torch = pytest.importorskip("torch")

    class Output(torch.nn.Module):
        def forward(self, state, actions, incoming):
            native = actions[..., :2].contiguous()
            accepted = torch.isfinite(native).all().reshape(1)
            if broken != "incoming":
                accepted = accepted & incoming
            if broken != "source":
                accepted = accepted & torch.isfinite(actions).all()
            return native, accepted

    inputs = (torch.zeros(1, 4), torch.zeros(1, 3, 4), torch.tensor([True]))
    if broken is None:
        assert checked_output_rejections(Output(), inputs) == {
            "incoming_false_rejected": True, "complete_source_nonfinite_rejected": True}
    else:
        with pytest.raises(ValueError):
            checked_output_rejections(Output(), inputs)
    assert torch.isfinite(inputs[1]).all() and inputs[2].item()


def test_legacy_capture_processor_config_is_explicit_and_canonical(tmp_path):
    source = {
        "checkpoint_provenance": {
            "source": {"root": str(tmp_path / "upstream")}
        }
    }
    config = processor_config_for_output(source, tmp_path / "checkpoint")
    assert config == {
        "source_root": str((tmp_path / "upstream").resolve()),
        "checkpoint_dir": str((tmp_path / "checkpoint").resolve()),
    }


def test_captured_processor_config_is_preserved_and_rejects_conflict(tmp_path):
    source = {"processor_config": {
        "source_root": str(tmp_path / "upstream"),
        "checkpoint_dir": str(tmp_path / "checkpoint"),
    }}
    assert processor_config_for_output(source, None) == source["processor_config"]
    assert processor_config_for_output(source, tmp_path / "checkpoint") == source["processor_config"]
    with pytest.raises(ValueError, match="differs"):
        processor_config_for_output(source, tmp_path / "other")


def test_legacy_processor_config_requires_explicit_checkpoint(tmp_path):
    source = {"checkpoint_provenance": {"source": {"root": str(tmp_path)}}}
    with pytest.raises(ValueError, match="--processor-checkpoint-dir"):
        processor_config_for_output(source, None)
