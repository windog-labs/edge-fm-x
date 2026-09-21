import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from vlaforge.validation.allocator_metrics import (
    COUNTS,
    GROUPS,
    PHASES,
    allocator_report,
)


@pytest.fixture
def tool():
    path = Path(__file__).resolve().parents[2] / "tools" / "benchmark_session.py"
    spec = importlib.util.spec_from_file_location("allocator_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path, tool):
    rows = []
    for index, phase in enumerate(PHASES):
        counters = {
            name: {
                "current": 10,
                "peak": 20,
                "allocated": 100 + index,
                "freed": 90 + index,
            }
            for name in GROUPS
        }
        counters.update(dict.fromkeys(COUNTS, 0))
        rows.append(
            {
                "schema": "vlaforge.libtorch_allocator_snapshot/1",
                "phase": phase,
                "torch_release": "2.10.0",
                "allocator": "native",
                "ordinal": 0,
                "device_free_bytes": 100,
                "device_total_bytes": 200,
                "initialized": True,
                "counters": counters,
            }
        )
    raw = tmp_path / "allocator-snapshots.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    execution = tmp_path / "execution.json"
    tool.write(execution, {"status": "executed", "pid": 123, "pilot": False})
    report = allocator_report(rows, ordinal=0)
    report.update(
        raw_snapshots_sha256=tool.sha(raw),
        execution_sha256=tool.sha(execution),
        pilot=False,
    )
    tool.write(tmp_path / "allocator-report.json", report)
    binding = tool.verify_allocator_evidence(tmp_path, ordinal=0, pilot=False)
    return tmp_path, binding


def test_report_binds_both_raw_and_recomputed_observations(tool, evidence):
    folder, binding = evidence
    assert set(binding) == {"raw_snapshots_sha256", "report_sha256"}
    assert (
        tool.verify_allocator_evidence(folder, ordinal=0, pilot=False, expected=binding)
        == binding
    )


@pytest.mark.parametrize(
    "name", ["allocator-snapshots.jsonl", "allocator-report.json", "execution.json"]
)
def test_deleted_evidence_is_not_accepted(tool, evidence, name):
    folder, binding = evidence
    (folder / name).unlink()
    with pytest.raises(FileNotFoundError):
        tool.verify_allocator_evidence(folder, ordinal=0, pilot=False, expected=binding)


@pytest.mark.parametrize(
    "name", ["allocator-snapshots.jsonl", "allocator-report.json", "execution.json"]
)
def test_changed_evidence_is_not_accepted(tool, evidence, name):
    folder, binding = evidence
    path = folder / name
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError):
        tool.verify_allocator_evidence(folder, ordinal=0, pilot=False, expected=binding)


@pytest.mark.parametrize(
    "mutation", ["requests", "raw_hash", "execution_hash", "pilot", "snapshot"]
)
def test_self_consistent_report_hash_cannot_bypass_recomputation(
    tool, evidence, mutation
):
    folder, _ = evidence
    path = folder / "allocator-report.json"
    report = tool.read(path)
    if mutation == "requests":
        report["steady_interval"]["allocator_requests"] = 0
    elif mutation == "snapshot":
        report["snapshots"][2]["device_free_bytes"] -= 1
    elif mutation == "pilot":
        report["pilot"] = 0
    else:
        report[
            {"raw_hash": "raw_snapshots_sha256", "execution_hash": "execution_sha256"}[
                mutation
            ]
        ] = "0" * 64
    tool.write(path, report)
    binding = {
        "raw_snapshots_sha256": tool.sha(folder / "allocator-snapshots.jsonl"),
        "report_sha256": tool.sha(path),
    }
    with pytest.raises(ValueError, match="disagrees"):
        tool.verify_allocator_evidence(folder, ordinal=0, pilot=False, expected=binding)


def aggregate_protocol():
    from vlaforge.validation.session_benchmark import (
        BOUNDARY,
        SCHEMA,
        validate_protocol,
    )

    value = {
        "schema": SCHEMA,
        "boundary": BOUNDARY,
        "processes": 5,
        "warmup": 128,
        "measured": 1024,
        "policies": ["off"],
        "bundles": {"off": "bundle"},
        "quality_gate": "passed",
        "samples": [{"inputs": {}, "direct": "direct.npy", "eager": "eager.npy"}],
        "evidence": ["source.json"],
        "allocator_observation": "libtorch-native/1",
        "gpu_ordinal": 0,
    }
    validate_protocol(value)
    return value


@pytest.mark.parametrize("binding", [{}, None, {"report_sha256": "0" * 64}, "absent"])
def test_aggregate_requires_complete_allocator_bindings(
    tool, tmp_path, monkeypatch, binding
):
    monkeypatch.setattr(tool, "verify_frozen", lambda _: None)
    tool.write(
        tmp_path / "protocol.json",
        aggregate_protocol(),
    )
    report = {"status": "passed", "pilot": False, "boundary": aggregate_protocol()["boundary"]}
    if binding != "absent":
        report["allocator_observation"] = binding
    tool.write(tmp_path / "runs" / "00-off" / "report.json", report)
    with pytest.raises(ValueError, match="bindings"):
        tool.aggregate(SimpleNamespace(output=tmp_path))


def test_aggregate_revalidates_bound_evidence(tool, tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "verify_frozen", lambda _: None)
    tool.write(
        tmp_path / "protocol.json",
        aggregate_protocol(),
    )
    binding = dict.fromkeys(("raw_snapshots_sha256", "report_sha256"), "0" * 64)
    tool.write(
        tmp_path / "runs" / "00-off" / "report.json",
        {
            "status": "passed",
            "pilot": False,
            "boundary": aggregate_protocol()["boundary"],
            "allocator_observation": binding,
        },
    )

    def reject(folder, *, ordinal, pilot, expected):
        assert folder == tmp_path / "runs" / "00-off"
        assert ordinal == 0 and pilot is False and expected == binding
        raise ValueError("actual verifier invoked")

    monkeypatch.setattr(tool, "verify_allocator_evidence", reject)
    with pytest.raises(ValueError, match="actual verifier invoked"):
        tool.aggregate(SimpleNamespace(output=tmp_path))
