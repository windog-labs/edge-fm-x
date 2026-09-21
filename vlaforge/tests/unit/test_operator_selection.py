import copy
import hashlib
import json

import pytest
from vlaforge.deployment.operator_selection import select_operator_candidate


def candidate(**updates):
    value = {
        "status": "measured",
        "microbenchmark_numeric_eligible": True,
        "correctness_passed": True,
        "speedup_percent": 12.5,
        "gpu": "NVIDIA H20",
        "artifact_sha256": "a" * 64,
    }
    value.update(updates)
    return value


def e2e(**updates):
    shared = {key: "b" * 64 for key in ("input_contract_sha256", "output_contract_sha256",
        "numerical_policy_sha256", "measurement_protocol_sha256")}
    baseline = {**shared, "gpu": "NVIDIA H20", "bundle_sha256": "c" * 64,
        "execution_audit_sha256": "d" * 64, "loaded_artifact_sha256": ["e" * 64],
        "measured_calls": 5120, "mean_latency_ns": 100}
    selected = {**shared, "gpu": "NVIDIA H20", "bundle_sha256": "f" * 64,
        "execution_audit_sha256": "1" * 64, "loaded_artifact_sha256": ["a" * 64],
        "measured_calls": 5120, "mean_latency_ns": 97}
    value = {
        "status": "completed",
        "end_to_end_integrated": True,
        "correctness_passed": True,
        "complete_output_bitwise_equal": True,
        "measured_calls": 5120,
        "speedup_percent": 3.0,
        "confidence_gate_passed": True,
        "operator_integration": {"schema": "vlaforge.operator_integration/1",
            "candidate_report_sha256": hashlib.sha256(json.dumps(candidate(), sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "baseline": baseline, "candidate": selected},
    }
    value.update(updates)
    return value


def test_selection_requires_complete_e2e_gate():
    result = select_operator_candidate(candidate(), e2e(), skill_key="rope:dtype=f16", hardware="h20")
    assert result.status == "selected"
    assert result.end_to_end_integrated is True
    assert len(result.candidate_digest) == len(result.e2e_digest) == 64


@pytest.mark.parametrize(
    "candidate_update,e2e_update,reason",
    [
        ({"correctness_passed": False}, {}, "candidate microbenchmark"),
        ({}, {"end_to_end_integrated": False}, "complete-model E2E"),
        ({}, {"complete_output_bitwise_equal": False}, "complete-model E2E"),
        ({}, {"speedup_percent": -1.0}, "complete-model E2E"),
        ({}, {"status": "failed"}, "complete-model E2E"),
    ],
)
def test_selection_keeps_failed_candidates_rejected(candidate_update, e2e_update, reason):
    result = select_operator_candidate(candidate(**candidate_update), e2e(**e2e_update), skill_key="gemm:n=4", hardware="h20")
    assert result.status == "rejected"
    assert reason in result.reason


def test_minimum_speedup_is_explicit():
    result = select_operator_candidate(candidate(), e2e(speedup_percent=0.2), skill_key="embedding:v=1", hardware="h20", minimum_e2e_speedup_percent=1.0)
    assert result.status == "rejected"
    with pytest.raises(ValueError):
        select_operator_candidate(candidate(), e2e(), skill_key="x", hardware="h20", minimum_e2e_speedup_percent=-1)


def test_explicit_e2e_confidence_gate_is_required_when_present():
    result = select_operator_candidate(candidate(), e2e(confidence_gate_passed=False), skill_key="embedding", hardware="h20")
    assert result.status == "rejected"
    assert result.reason == "complete-model E2E integration/performance gate failed"


def test_hash_binds_the_complete_evidence():
    one = select_operator_candidate(candidate(), e2e(), skill_key="attention:qk", hardware="h20")
    two = select_operator_candidate(candidate(speedup_percent=12.6), e2e(), skill_key="attention:qk", hardware="h20")
    assert one.candidate_digest != two.candidate_digest
    assert one.e2e_digest == two.e2e_digest


@pytest.mark.parametrize("report", [{"speedup_percent": float("nan")}, {"speedup_percent": "fast"}])
def test_nonfinite_or_untyped_speedup_is_rejected_before_hash(report):
    with pytest.raises((ValueError, TypeError)):
        select_operator_candidate(candidate(**report), e2e(), skill_key="norm", hardware="h20")


def test_integration_flag_without_candidate_and_platform_binding_is_rejected():
    result = select_operator_candidate(candidate(), e2e(operator_integration=None), skill_key="embedding", hardware="h20")
    assert result.status == "rejected"
    assert result.end_to_end_integrated is False


@pytest.mark.parametrize("lane,key,value", [
    ("candidate", "gpu", "NVIDIA GeForce RTX 3060"),
    ("baseline", "gpu", "NVIDIA H100"),
    ("candidate", "loaded_artifact_sha256", ["e" * 64]),
    ("baseline", "loaded_artifact_sha256", ["a" * 64]),
    ("candidate", "bundle_sha256", "c" * 64),
    ("candidate", "execution_audit_sha256", "d" * 64),
    ("candidate", "execution_audit_sha256", "not-an-audit"),
    ("candidate", "input_contract_sha256", "2" * 64),
    ("candidate", "output_contract_sha256", "2" * 64),
    ("candidate", "numerical_policy_sha256", "2" * 64),
    ("candidate", "measurement_protocol_sha256", "2" * 64),
    ("candidate", "measured_calls", True),
    ("candidate", "mean_latency_ns", 99),
    ("candidate", "mean_latency_ns", 0),
])
def test_unmatched_integration_evidence_rejected(lane, key, value):
    report = e2e()
    report["operator_integration"][lane][key] = value
    original = copy.deepcopy(report)
    result = select_operator_candidate(candidate(), report, skill_key="embedding", hardware="h20")
    assert result.status == "rejected" and not result.end_to_end_integrated
    assert original == report


def test_missing_confidence_does_not_default_to_pass():
    report = e2e()
    del report["confidence_gate_passed"]
    assert select_operator_candidate(candidate(), report, skill_key="x", hardware="h20").status == "rejected"


def test_changed_candidate_does_not_inherit_integration():
    result = select_operator_candidate(candidate(artifact_sha256="9" * 64), e2e(), skill_key="x", hardware="h20")
    assert result.status == "rejected" and not result.end_to_end_integrated


def test_unchanged_latency_is_not_a_deployment_gain():
    report = e2e(speedup_percent=0)
    report["operator_integration"]["candidate"]["mean_latency_ns"] = 100
    assert select_operator_candidate(candidate(), report, skill_key="x", hardware="h20").status == "rejected"
