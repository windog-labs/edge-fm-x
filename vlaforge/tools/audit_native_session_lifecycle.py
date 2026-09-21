"""Independently audit native Session contract and lifecycle probe reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "vlaforge.native_session_contract_probe/2"
LIFECYCLE_SCHEMA = "vlaforge.native_session_lifecycle/1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_report_data(report: dict[str, Any]) -> dict[str, Any]:
    require(report.get("schema") == REPORT_SCHEMA, "report schema differs")
    require(report.get("status") == "passed", "report status differs")

    baseline = report.get("baseline_outputs_sha256")
    require(
        isinstance(baseline, dict) and bool(baseline),
        "baseline outputs are missing",
    )
    require(
        report.get("final_outputs_sha256") == baseline,
        "final outputs differ from baseline",
    )
    require(
        report.get("committed_outputs_after_rejection") == baseline,
        "rejected call changed committed outputs",
    )
    require(
        report.get("valid_outputs_unchanged_after_rejection") is True,
        "valid-output rejection flag differs",
    )

    rejections = report.get("shape_rejections")
    require(
        isinstance(rejections, list)
        and bool(rejections)
        and all(item.get("rejected") is True for item in rejections),
        "shape rejection coverage differs",
    )
    accepted_false = report.get("accepted_false_rejection")
    require(
        isinstance(accepted_false, dict)
        and accepted_false.get("rejected") is True,
        "accepted=false rejection evidence differs",
    )

    lifecycle = report.get("lifecycle")
    require(isinstance(lifecycle, dict), "lifecycle report is missing")
    require(
        lifecycle.get("schema") == LIFECYCLE_SCHEMA,
        "lifecycle schema differs",
    )
    require(
        lifecycle.get("status") == "passed",
        "lifecycle status differs",
    )
    primary_inputs = lifecycle.get("primary_input_sha256")
    alternate_inputs = lifecycle.get("alternate_input_sha256")
    require(
        isinstance(primary_inputs, dict)
        and bool(primary_inputs)
        and isinstance(alternate_inputs, dict)
        and bool(alternate_inputs)
        and primary_inputs != alternate_inputs,
        "lifecycle inputs are missing or identical",
    )

    baseline = lifecycle.get("baseline_outputs_sha256")
    alternate = lifecycle.get("alternate_outputs_sha256")
    require(
        isinstance(baseline, dict)
        and bool(baseline)
        and isinstance(alternate, dict)
        and bool(alternate)
        and baseline != alternate,
        "lifecycle primary and alternate outputs do not differ",
    )
    equalities = {
        "repeated primary": (
            lifecycle.get("repeated_primary_outputs_sha256"),
            baseline,
        ),
        "second-Session primary": (
            lifecycle.get("isolated_primary_outputs_sha256"),
            baseline,
        ),
        "second-Session alternate": (
            lifecycle.get("isolated_alternate_outputs_sha256"),
            alternate,
        ),
        "repeated alternate": (
            lifecycle.get("repeated_alternate_outputs_sha256"),
            alternate,
        ),
        "outputs after rejected reset": (
            lifecycle.get("outputs_after_rejected_reset"),
            baseline,
        ),
        "primary after accepted reset": (
            lifecycle.get("reset_primary_outputs_sha256"),
            baseline,
        ),
        "second Session after first reset": (
            lifecycle.get("session_b_outputs_after_session_a_reset"),
            alternate,
        ),
        "alternate after second reset": (
            lifecycle.get("reset_alternate_outputs_sha256"),
            alternate,
        ),
        "fresh Session primary": (
            lifecycle.get("fresh_session_primary_outputs_sha256"),
            baseline,
        ),
    }
    for name, (actual, expected) in equalities.items():
        require(actual == expected, f"lifecycle equality differs: {name}")

    require(
        lifecycle.get("outputs_after_accepted_reset") == {},
        "accepted reset did not invalidate committed outputs",
    )
    require(
        lifecycle.get("accepted_reset_invalidated_outputs") is True,
        "accepted-reset invalidation flag differs",
    )
    require(
        lifecycle.get("same_session_repeat_stable") is True,
        "same-Session repeat flag differs",
    )
    require(
        lifecycle.get("two_session_interleaving_isolated") is True,
        "two-Session isolation flag differs",
    )
    require(
        lifecycle.get("destroy_recreate_stable") is True,
        "destroy/recreate flag differs",
    )
    require(
        lifecycle.get("rejected_reset", {}).get("rejected") is True,
        "non-increasing reset was not rejected",
    )
    require(
        lifecycle.get("accepted_reset", {}).get("code") == 0,
        "accepted reset did not succeed",
    )
    require(
        lifecycle.get("second_session_reset", {}).get("code") == 0,
        "second-Session reset did not succeed",
    )

    return {
        "shape_rejections": len(rejections),
        "accepted_false_rejected": True,
        "baseline_outputs": baseline,
        "alternate_outputs": alternate,
        "lifecycle_checks": len(equalities) + 11,
    }


def audit_report_file(report_path: Path) -> dict[str, Any]:
    report_path = report_path.resolve(strict=True)
    report = json.loads(report_path.read_text())
    summary = audit_report_data(report)

    bundle = Path(report["bundle"]).resolve(strict=True)
    library = Path(report["library"]).resolve(strict=True)
    require(
        sha256(bundle / "bundle.json")
        == report.get("bundle_manifest_sha256"),
        "bundle manifest hash differs",
    )
    require(
        sha256(library) == report.get("library_sha256"),
        "library hash differs",
    )
    require(
        sha256(bundle / "metadata/input_schema.json")
        == report.get("input_schema_sha256"),
        "input schema hash differs",
    )
    require(
        sha256(bundle / "metadata/output_schema.json")
        == report.get("output_schema_sha256"),
        "output schema hash differs",
    )
    return {
        "schema": "vlaforge.native_session_lifecycle_audit/1",
        "status": "passed",
        "report": str(report_path),
        "report_sha256": sha256(report_path),
        "bundle": str(bundle),
        "bundle_manifest_sha256": report["bundle_manifest_sha256"],
        "library": str(library),
        "library_sha256": report["library_sha256"],
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError("output already exists")
    reports = [audit_report_file(path) for path in args.report]
    result = {
        "schema": "vlaforge.native_session_lifecycle_audit_set/1",
        "status": "passed",
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "reports": len(reports),
        "lifecycle_checks": sum(
            item["lifecycle_checks"] for item in reports
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
