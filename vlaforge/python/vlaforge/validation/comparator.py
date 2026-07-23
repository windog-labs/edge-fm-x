"""State/solver/action trace comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from vlaforge.interpreter.trace import Trace
from vlaforge.validation.contracts import NumericContract


@dataclass(frozen=True, slots=True)
class ComparisonIssue:
    path: str
    expected: Any
    actual: Any
    reason: str


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    equal: bool
    compared_events: int
    issues: tuple[ComparisonIssue, ...]

    def format(self) -> str:
        if self.equal:
            return f"trace comparison passed: {self.compared_events} events"
        lines = [
            f"trace comparison failed: {len(self.issues)} issue(s), "
            f"{self.compared_events} aligned events"
        ]
        lines.extend(
            f"- {issue.path}: {issue.reason}; expected={issue.expected!r}, "
            f"actual={issue.actual!r}"
            for issue in self.issues
        )
        return "\n".join(lines)


def compare_traces(
    expected: Trace,
    actual: Trace,
    contract: NumericContract = NumericContract(),
) -> ComparisonReport:
    issues: list[ComparisonIssue] = []
    expected_events = expected.to_data()["events"]
    actual_events = actual.to_data()["events"]
    if len(expected_events) != len(actual_events):
        issues.append(
            ComparisonIssue(
                "events.length",
                len(expected_events),
                len(actual_events),
                "event count differs",
            )
        )
    aligned = min(len(expected_events), len(actual_events))
    for index in range(aligned):
        _compare(
            expected_events[index],
            actual_events[index],
            f"events[{index}]",
            contract,
            issues,
        )
    return ComparisonReport(not issues, aligned, tuple(issues))


def _compare(
    expected: Any,
    actual: Any,
    path: str,
    contract: NumericContract,
    issues: list[ComparisonIssue],
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected is not actual:
            issues.append(ComparisonIssue(path, expected, actual, "boolean mismatch"))
        return
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        if not math.isclose(
            float(expected),
            float(actual),
            rel_tol=contract.relative_tolerance,
            abs_tol=contract.absolute_tolerance,
        ):
            issues.append(ComparisonIssue(path, expected, actual, "numeric mismatch"))
        return
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if bool(expected.get("tensor")) and bool(actual.get("tensor")):
            for key in ("shape", "dtype"):
                _compare(
                    expected.get(key),
                    actual.get(key),
                    f"{path}.{key}",
                    contract,
                    issues,
                )
            if "values" in expected and "values" in actual:
                _compare(
                    expected["values"],
                    actual["values"],
                    f"{path}.values",
                    contract,
                    issues,
                )
            elif contract.compare_tensor_hashes:
                _compare(
                    expected.get("sha256"),
                    actual.get("sha256"),
                    f"{path}.sha256",
                    contract,
                    issues,
                )
            return
        keys = set(expected) | set(actual)
        for key in sorted(keys, key=str):
            if key not in expected or key not in actual:
                issues.append(
                    ComparisonIssue(
                        f"{path}.{key}",
                        expected.get(key),
                        actual.get(key),
                        "mapping key missing",
                    )
                )
            else:
                _compare(
                    expected[key],
                    actual[key],
                    f"{path}.{key}",
                    contract,
                    issues,
                )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            issues.append(
                ComparisonIssue(path, len(expected), len(actual), "sequence length differs")
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]", contract, issues)
        return
    if expected != actual:
        issues.append(ComparisonIssue(path, expected, actual, "value mismatch"))

