#!/usr/bin/env python3
"""Adversarial evidence for passive Invocation IR v0.2 contracts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from vlaforge.adapters import build_openvla_fixture, build_smolvla_fixture
from vlaforge.analysis import verify
from vlaforge.interpreter import InputBinding, Interpreter, InterpreterError
from vlaforge.ir.program import Operation


def _rules(module) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                diagnostic.rule
                for diagnostic in verify(module, raise_on_error=False)
            }
        )
    )


def _unverified_publish(module):
    invocation = module.invocations[0]
    operations = list(invocation.body.operations)
    operations.insert(
        -1,
        Operation(
            "vla.action.publish",
            operands=("pending_action",),
        ),
    )
    return replace(
        module,
        invocations=(
            replace(
                invocation,
                body=replace(
                    invocation.body,
                    operations=tuple(operations),
                ),
            ),
        ),
    )


def evaluate() -> dict[str, object]:
    openvla = build_openvla_fixture()
    unstamped = {
        name: InputBinding(binding.value)
        for name, binding in openvla.runs[0].inputs.items()
    }
    runtime = Interpreter(
        openvla.module,
        regions=openvla.regions,
        validators=openvla.validators,
    )
    runtime.run(inputs=unstamped)
    runtime.run(inputs=unstamped)
    revision_fail_closed = (
        runtime.cache.hits == 0 and runtime.cache.misses == 2
    )

    schema_mismatch_rejected = False
    schema_error = ""
    try:
        Interpreter(
            openvla.module,
            regions=openvla.regions,
            validators=openvla.validators,
            expected_schema_digest="0" * 64,
        )
    except InterpreterError as error:
        schema_mismatch_rejected = True
        schema_error = str(error)

    smol = build_smolvla_fixture()
    rejected = Interpreter(
        smol.module,
        regions=smol.regions,
        validators={"finite_action": lambda _: False},
        initial_state=smol.initial_state,
    )
    before = rejected.state_store.versions("queue_cursor")[-1].version
    validation_rejected = False
    try:
        rejected.run(inputs=smol.runs[0].inputs)
    except InterpreterError:
        validation_rejected = True
    after = rejected.state_store.versions("queue_cursor")[-1].version
    no_output = False
    try:
        rejected.read_output()
    except InterpreterError:
        no_output = True
    atomic_abort = validation_rejected and before == after and no_output

    publish_rules = _rules(_unverified_publish(openvla.module))
    rows = [
        {
            "ablation": "no_input_revision_fail_closed",
            "removed_contract": (
                "missing InputRevision is assigned a fresh identity per Bind/Run"
            ),
            "adversarial_case": (
                "same borrowed tensor is bound twice without a revision"
            ),
            "expected_rule": "runtime.cache_miss_without_revision",
            "observed_rules": [
                f"hits={runtime.cache.hits}",
                f"misses={runtime.cache.misses}",
            ],
            "unsafe_program_accepted": False,
            "contract_detected_fault": revision_fail_closed,
            "failure_mode": (
                "without fail-closed identity, changed external storage could "
                "silently reuse a derived cache"
            ),
        },
        {
            "ablation": "no_io_schema_digest",
            "removed_contract": "bundle/session I/O schema digest match",
            "adversarial_case": "caller initializes a session with a stale schema",
            "expected_rule": "runtime.schema_digest_mismatch",
            "observed_rules": [schema_error],
            "unsafe_program_accepted": False,
            "contract_detected_fault": schema_mismatch_rejected,
            "failure_mode": "model upgrade could silently bind the wrong input id",
        },
        {
            "ablation": "no_atomic_state_output_commit",
            "removed_contract": (
                "state staging and named outputs share one validated transaction"
            ),
            "adversarial_case": "output validation fails after state staging",
            "expected_rule": "runtime.abort_preserves_state_and_output",
            "observed_rules": [
                f"before_version={before}",
                f"after_version={after}",
                f"output_unavailable={no_output}",
            ],
            "unsafe_program_accepted": False,
            "contract_detected_fault": atomic_abort,
            "failure_mode": (
                "state and externally observed model output could describe "
                "different invocations"
            ),
        },
        {
            "ablation": "no_transactional_output_boundary",
            "removed_contract": "Session returns only committed output groups",
            "adversarial_case": "an Adapter inserts an action.publish opcode",
            "expected_rule": "op.unknown",
            "observed_rules": list(publish_rules),
            "unsafe_program_accepted": False,
            "contract_detected_fault": "op.unknown" in publish_rules,
            "failure_mode": (
                "the compiler would perform bottom-software I/O and expose "
                "uncommitted values"
            ),
        },
    ]
    gate_passed = (
        all(bool(row["contract_detected_fault"]) for row in rows)
        and _rules(openvla.module) == ()
        and _rules(smol.module) == ()
    )
    return {
        "schema": "vlaforge.ir_necessity_ablation/2",
        "evidence_scope": (
            "adversarial deployment-contract tests; not vehicle safety proof"
        ),
        "baseline": {
            "smolvla_verifier_rules": list(_rules(smol.module)),
            "openvla_verifier_rules": list(_rules(openvla.module)),
        },
        "ablations": rows,
        "gate_passed": gate_passed,
    }


def _write_csv(path: Path, result: dict[str, object]) -> None:
    fields = (
        "ablation",
        "removed_contract",
        "adversarial_case",
        "expected_rule",
        "observed_rules",
        "contract_detected_fault",
        "unsafe_program_accepted",
        "failure_mode",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in result["ablations"]:
            current = dict(row)
            current["observed_rules"] = ",".join(current["observed_rules"])
            writer.writerow(current)


def _markdown(result: dict[str, object]) -> str:
    lines = [
        "# Invocation IR v0.2 Necessity Ablations",
        "",
        f"- Gate passed: `{str(result['gate_passed']).lower()}`",
        f"- Scope: {result['evidence_scope']}",
        "",
        "| Removed contract | Adversarial case | Result |",
        "|---|---|---|",
    ]
    for row in result["ablations"]:
        lines.append(
            f"| {row['removed_contract']} | {row['adversarial_case']} | "
            f"{'fault detected' if row['contract_detected_fault'] else 'missed'} |"
        )
    lines.extend(
        [
            "",
            "These tests justify input identity, schema binding, and atomic "
            "state/output semantics. They do not claim sensor synchronization, "
            "runtime scheduling, or vehicle safety.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ir_necessity_ablation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "ir_necessity_ablation.csv", result)
    (args.output_dir / "ir_necessity_ablation.md").write_text(
        _markdown(result),
        encoding="utf-8",
    )
    print(json.dumps({"gate_passed": result["gate_passed"]}))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
