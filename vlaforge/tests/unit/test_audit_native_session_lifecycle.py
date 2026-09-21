"""CPU-only checks for the independent Session lifecycle auditor."""

import copy
import importlib.util
from pathlib import Path

import pytest


_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/audit_native_session_lifecycle.py"
)
_SPEC = importlib.util.spec_from_file_location("native_lifecycle_audit", _PATH)
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def _report() -> dict:
    baseline = {"tokens": "primary-tokens", "logits": "primary-logits"}
    alternate = {"tokens": "alternate-tokens", "logits": "alternate-logits"}
    lifecycle = {
        "schema": tool.LIFECYCLE_SCHEMA,
        "status": "passed",
        "primary_input_sha256": {"input_ids": "primary-input"},
        "alternate_input_sha256": {"input_ids": "alternate-input"},
        "baseline_outputs_sha256": baseline,
        "alternate_outputs_sha256": alternate,
        "repeated_primary_outputs_sha256": baseline,
        "isolated_primary_outputs_sha256": baseline,
        "isolated_alternate_outputs_sha256": alternate,
        "repeated_alternate_outputs_sha256": alternate,
        "outputs_after_rejected_reset": baseline,
        "outputs_after_accepted_reset": {},
        "accepted_reset_invalidated_outputs": True,
        "reset_primary_outputs_sha256": baseline,
        "session_b_outputs_after_session_a_reset": alternate,
        "reset_alternate_outputs_sha256": alternate,
        "fresh_session_primary_outputs_sha256": baseline,
        "same_session_repeat_stable": True,
        "two_session_interleaving_isolated": True,
        "destroy_recreate_stable": True,
        "rejected_reset": {"rejected": True, "code": 9},
        "accepted_reset": {"rejected": False, "code": 0},
        "second_session_reset": {"rejected": False, "code": 0},
    }
    return {
        "schema": tool.REPORT_SCHEMA,
        "status": "passed",
        "baseline_outputs_sha256": baseline,
        "final_outputs_sha256": baseline,
        "committed_outputs_after_rejection": baseline,
        "valid_outputs_unchanged_after_rejection": True,
        "shape_rejections": [{"rejected": True}],
        "accepted_false_rejection": {"rejected": True, "code": 7},
        "lifecycle": lifecycle,
    }


def test_audit_accepts_consistent_lifecycle_report():
    summary = tool.audit_report_data(_report())

    assert summary["lifecycle_checks"] == 20
    assert summary["accepted_false_rejected"] is True


def test_audit_rejects_cross_session_output_change():
    report = copy.deepcopy(_report())
    report["lifecycle"]["isolated_primary_outputs_sha256"] = {
        "tokens": "polluted",
        "logits": "polluted",
    }

    with pytest.raises(ValueError, match="second-Session primary"):
        tool.audit_report_data(report)
