"""Byte-contract diagnostic unit tests, not real model support evidence."""

import json

import pytest

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_runtime_compare import (
    compare_tensor_bytes,
    read_trace,
    runtime_region_ids,
)


def test_native_ids_follow_hash_bound_compiled_ir_not_input_order(tmp_path):
    ir = tmp_path / "semantic_ir.json"
    ir.write_text(json.dumps({"regions": [{"name": "a"}, {"name": "z"}]}))
    manifest = tmp_path / "bundle.json"
    manifest.write_text(
        json.dumps({"semantic_ir": {"path": ir.name, **file_digest(ir)}})
    )
    expected = file_digest(manifest)
    region_ids, evidence = runtime_region_ids(tmp_path, expected)
    assert region_ids == {"0": "a", "1": "z"}
    assert evidence["compiled_semantic_ir"]["sha256"] == file_digest(ir)["sha256"]
    ir.write_text(json.dumps({"regions": [{"name": "z"}, {"name": "a"}]}))
    with pytest.raises(ValueError, match="IR checksum"):
        runtime_region_ids(tmp_path, expected)


def test_bfloat16_metric_preserves_actual_encoded_values():
    np = pytest.importorskip("numpy")
    left = np.array([0x3F80, 0x4000], dtype="<u2").tobytes()
    right = np.array([0x3F80, 0x4040], dtype="<u2").tobytes()
    result = compare_tensor_bytes(left, right, "torch.bfloat16")
    assert not result["exact_bytes"]
    assert result["mismatch_count"] == 1
    assert result["mean_squared_error"] == 0.5
    assert result["maximum_absolute_error"] == 1.0


def test_signed_zero_bytes_are_distinct_from_equal_values():
    np = pytest.importorskip("numpy")
    result = compare_tensor_bytes(
        np.array([0.0], dtype="<f4").tobytes(),
        np.array([-0.0], dtype="<f4").tobytes(),
        "torch.float32",
    )
    assert result["exact_values"] and not result["exact_bytes"]


@pytest.mark.parametrize("change", ["digest", "shape", "escape", "duplicate"])
def test_trace_reader_rejects_incomplete_or_changed_boundaries(tmp_path, change):
    trace = tmp_path / "region-trace"
    trace.mkdir()
    data = trace / "call-0-input-0.bin"
    data.write_bytes(b"\x00" * 4)
    row = {
        "index": 0,
        "file": data.name,
        "size_bytes": 4,
        "dtype": "torch.float32",
        "shape": [1],
        "sha256": file_digest(data)["sha256"],
    }
    if change == "digest":
        row["sha256"] = "0" * 64
    elif change == "shape":
        row["shape"] = [2]
    elif change == "escape":
        row["file"] = "../report.json"
    report = {
        "schema": "vlaforge.openpi_same_artifact_runtime_probe/1",
        "status": "diagnostic-complete",
        "capture": {"sha256": "1" * 64},
        "execution_context": {},
        "calls": [
            {
                "call": 0,
                "region": "region",
                "inputs": [row] * (2 if change == "duplicate" else 1),
                "outputs": [],
            }
        ],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        read_trace(path, file_digest(path)["sha256"])
