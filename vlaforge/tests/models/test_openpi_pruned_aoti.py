"""CPU protocol helpers only; no pretrained model or CUDA claim."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vlaforge.adapters.openpi.openpi_pruned_aoti import TRACE_SCHEMA, _compare_tree, _cpu_tree, _record, _trace_source
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.numerical_context import snapshot


def test_scalar_and_zero_sign_complete_cpu_transport():
    values = (torch.tensor(1.), torch.tensor([-0.]), torch.ones(2, dtype=torch.bfloat16), torch.tensor(True))
    copy = _cpu_tree(values)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(values, copy, strict=True))
    assert all(item["bitwise_equal"] for item in _compare_tree(values, copy))
    copy[1].fill_(0)
    assert _compare_tree(values, copy)[1]["bitwise_equal"] is False


def test_complete_tree_and_dtype_are_required():
    with pytest.raises(ValueError, match="tree"):
        _compare_tree((torch.tensor(1),), [torch.tensor(1)])
    with pytest.raises(ValueError, match="shape/dtype"):
        _compare_tree((torch.tensor(1),), (torch.tensor(1.),))


@pytest.mark.parametrize("tamper", [None, "alive", "status", "context", "call_order", "specimen", "actions"])
def test_trace_proof_requires_exited_owner_and_all_files(tmp_path, tamper):
    from vlaforge.adapters.openpi.openpi_phased import process_identity
    context = snapshot()
    specimen = tmp_path / "call-000.pt"
    torch.save({"args": (torch.ones(1),), "expected": (torch.ones(1),)}, specimen)
    np.savez(tmp_path / "actions.npz", normalized=np.ones(1))
    capture = {"path": "/fixture/capture.json", "sha256": "1" * 64}
    row = {"schema": TRACE_SCHEMA, "status": "passed", "source_report": capture,
           "numerical_context": context.to_dict(), "complete_bitwise_equal": True,
           "caller_policy_restoration_verified": True, "process_identity": {"pid": 999999999, "start_ticks": 1},
           "calls": [{"index": 0, "region": "test", "specimen": _record(specimen)}],
           "actions": file_digest(tmp_path / "actions.npz")}
    if tamper == "alive":
        row["process_identity"] = process_identity()
    elif tamper == "status":
        row["status"] = "failed"
    elif tamper == "context":
        row["numerical_context"]["cudnn_enabled"] = 1
    elif tamper == "call_order":
        row["calls"][0]["index"] = 1
    elif tamper == "specimen":
        specimen.write_bytes(b"changed")
    elif tamper == "actions":
        (tmp_path / "actions.npz").write_bytes(b"changed")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(row))
    if tamper is None:
        assert _trace_source(report, file_digest(report)["sha256"], capture, context) == row
    else:
        with pytest.raises(ValueError):
            _trace_source(report, file_digest(report)["sha256"], capture, context)
