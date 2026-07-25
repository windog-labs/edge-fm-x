from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "evaluate_ir_necessity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_ir_necessity",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ir_necessity_ablation_detects_all_adversarial_cases() -> None:
    result = _module().evaluate()
    assert result["gate_passed"]
    rows = {item["ablation"]: item for item in result["ablations"]}
    assert set(rows) == {
        "no_input_revision_fail_closed",
        "no_io_schema_digest",
        "no_atomic_state_output_commit",
        "no_transactional_output_boundary",
    }
    assert not any(
        item["unsafe_program_accepted"] for item in rows.values()
    )
    assert all(item["contract_detected_fault"] for item in rows.values())
