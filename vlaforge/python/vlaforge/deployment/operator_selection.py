"""Model-independent gate for registering an optimized operator skill.

Selection is deliberately separate from candidate generation and benchmarking.
An isolated operator speedup cannot be promoted unless a complete-model E2E
measurement uses the same input/output and numerical contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class OperatorSelection:
    schema: str
    status: str
    skill_key: str
    hardware: str
    candidate_digest: str
    e2e_digest: str
    reason: str
    microbenchmark_speedup_percent: float | None
    e2e_speedup_percent: float | None
    correctness_passed: bool
    end_to_end_integrated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _digest(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("selection evidence must be finite JSON data") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _speedup(report: Mapping[str, Any], *, key: str) -> float | None:
    value = report.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{key} must be a finite number or null")
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return float(value)


def _sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _integration_bound(candidate: Mapping[str, Any], e2e: Mapping[str, Any], speedup: float | None) -> bool:
    """Check audited report identities, not just a caller's integration flag.

    This metadata gate does not authenticate measurements. The report producer
    must verify bundle payloads, execution and complete outputs against raw data.
    Initial support requires the measured artifact itself in the candidate
    bundle; recompiling or fusing it needs a separate derived-artifact contract.
    """
    binding = e2e.get("operator_integration")
    if (not isinstance(binding, Mapping)
            or binding.get("schema") != "vlaforge.operator_integration/1"
            or binding.get("candidate_report_sha256") != _digest(candidate)):
        return False
    artifact = candidate.get("artifact_sha256")
    gpu = candidate.get("gpu")
    if not _sha256(artifact) or not isinstance(gpu, str) or not gpu.strip():
        return False
    baseline, selected = binding.get("baseline"), binding.get("candidate")
    if not isinstance(baseline, Mapping) or not isinstance(selected, Mapping):
        return False
    shared = ("input_contract_sha256", "output_contract_sha256", "numerical_policy_sha256",
              "measurement_protocol_sha256")
    for lane in (baseline, selected):
        if lane.get("gpu") != gpu:
            return False
        if any(not _sha256(lane.get(key)) for key in (*shared, "bundle_sha256", "execution_audit_sha256")):
            return False
        payloads = lane.get("loaded_artifact_sha256")
        if not isinstance(payloads, list) or not payloads or not all(_sha256(value) for value in payloads):
            return False
        calls, mean = lane.get("measured_calls"), lane.get("mean_latency_ns")
        if type(calls) is not int or calls <= 0 or calls != e2e.get("measured_calls"):
            return False
        if (not isinstance(mean, (int, float)) or isinstance(mean, bool)
                or not math.isfinite(mean) or mean <= 0):
            return False
    if (any(baseline[key] != selected[key] for key in shared)
            or baseline["bundle_sha256"] == selected["bundle_sha256"]
            or baseline["execution_audit_sha256"] == selected["execution_audit_sha256"]
            or artifact in baseline["loaded_artifact_sha256"]
            or artifact not in selected["loaded_artifact_sha256"]):
        return False
    actual_speedup = 100 * (1 - selected["mean_latency_ns"] / baseline["mean_latency_ns"])
    return speedup is not None and math.isclose(speedup, actual_speedup, rel_tol=1e-12, abs_tol=1e-12)


def select_operator_candidate(
    candidate: Mapping[str, Any],
    e2e: Mapping[str, Any],
    *,
    skill_key: str,
    hardware: str,
    minimum_e2e_speedup_percent: float = 0.0,
) -> OperatorSelection:
    """Apply the complete selection chain and return an auditable decision.

    ``candidate`` is an operator microbenchmark report. ``e2e`` is a complete
    model campaign report with a paired baseline and candidate. Both reports are
    treated as immutable evidence and hashed into the decision. The E2E report
    must bind the candidate's canonical JSON SHA256 and actual loaded artifact
    to two audited, same-GPU-type lanes with identical measurement, input/output
    and numerical contracts. A rejected candidate remains visible to reporting.
    """
    if not isinstance(skill_key, str) or not skill_key.strip():
        raise ValueError("skill_key must be a non-empty operator signature")
    if not isinstance(hardware, str) or not hardware.strip():
        raise ValueError("hardware must be explicit")
    if (not isinstance(minimum_e2e_speedup_percent, (int, float))
            or isinstance(minimum_e2e_speedup_percent, bool)
            or minimum_e2e_speedup_percent < 0
            or not math.isfinite(minimum_e2e_speedup_percent)):
        raise ValueError("minimum E2E speedup must be finite and non-negative")
    if not isinstance(candidate, Mapping) or not isinstance(e2e, Mapping):
        raise TypeError("candidate and e2e reports must be mappings")
    micro = _speedup(candidate, key="speedup_percent")
    correctness = candidate.get("correctness_passed") is True and e2e.get("correctness_passed") is True
    candidate_ready = (
        candidate.get("status") == "measured"
        and candidate.get("microbenchmark_numeric_eligible") is True
        and correctness
    )
    e2e_speedup = _speedup(e2e, key="speedup_percent")
    integrated = e2e.get("end_to_end_integrated") is True and _integration_bound(candidate, e2e, e2e_speedup)
    e2e_ready = (
        e2e.get("status") in {"completed", "passed"}
        and integrated
        and e2e.get("complete_output_bitwise_equal") is True
        and type(e2e.get("measured_calls")) is int
        and e2e["measured_calls"] > 0
        and e2e_speedup is not None
        and e2e_speedup > 0
        and e2e_speedup >= float(minimum_e2e_speedup_percent)
        and e2e.get("confidence_gate_passed") is True
    )
    if not candidate_ready:
        reason = "candidate microbenchmark or correctness gate failed"
        status = "rejected"
    elif not e2e_ready:
        reason = "complete-model E2E integration/performance gate failed"
        status = "rejected"
    else:
        reason = "candidate and complete-model E2E gates passed"
        status = "selected"
    return OperatorSelection(
        schema="vlaforge.operator_selection/2",
        status=status,
        skill_key=skill_key,
        hardware=hardware,
        candidate_digest=_digest(candidate),
        e2e_digest=_digest(e2e),
        reason=reason,
        microbenchmark_speedup_percent=micro,
        e2e_speedup_percent=e2e_speedup,
        correctness_passed=correctness,
        end_to_end_integrated=integrated,
    )
