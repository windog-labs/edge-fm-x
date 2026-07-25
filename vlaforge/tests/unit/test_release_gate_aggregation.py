from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "aggregate_release_gate.py"
    )
    specification = importlib.util.spec_from_file_location(
        "aggregate_release_gate",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_junit_summary_supports_testsuites_and_skips(
    tmp_path: Path,
) -> None:
    aggregation = _module()
    report = tmp_path / "junit.xml"
    report.write_text(
        """
<testsuites>
  <testsuite tests="3" failures="0" errors="0" skipped="1" time="1.5">
    <testcase name="one"/>
    <testcase name="two"><skipped/></testcase>
    <testcase name="three"/>
  </testsuite>
</testsuites>
""".strip(),
        encoding="utf-8",
    )

    assert aggregation.junit_summary(report) == {
        "tests": 3,
        "passed": 2,
        "skipped": 1,
        "failures": 0,
        "errors": 0,
        "seconds": 1.5,
        "test_names": ["one", "two", "three"],
    }
