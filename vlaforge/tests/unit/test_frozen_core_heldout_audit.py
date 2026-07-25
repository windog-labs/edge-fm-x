from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from vlaforge.adapters import (
    build_driving_ar_fixture,
    build_groot_n1_like_fixture,
    build_octo_like_fixture,
)


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_frozen_core_heldouts.py"
    )
    specification = importlib.util.spec_from_file_location(
        "audit_frozen_core_heldouts",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_literal_matches_record_first_line_and_missing_literals() -> None:
    audit = _module()

    matches, missing = audit._literal_matches(
        "alpha\nneedle one\nbeta needle two\n",
        ("needle one", "needle two", "absent"),
    )

    assert matches == {"needle one": 2, "needle two": 3}
    assert missing == ("absent",)


@pytest.mark.parametrize(
    ("name", "factory", "template"),
    (
        ("Octo", build_octo_like_fixture, "DiffusionPolicy"),
        ("GR00T N1.7", build_groot_n1_like_fixture, "MultiEmbodimentDiT"),
        ("AutoVLA", build_driving_ar_fixture, "AutoregressiveTrajectory"),
    ),
)
def test_heldout_fixtures_compile_and_match_verified_plan(
    name, factory, template
) -> None:
    audit = _module()

    record = audit.audit_fixture(name, factory)

    assert record["passed"] is True
    assert record["adapter_template"] == template
    assert record["core_op_delta"] == 0
    assert record["unknown_opcodes"] == []
    assert record["semantic_plan_output_state_parity"] is True
    assert record["semantic_plan_trace_parity"] is True
    assert record["compilation_certificate"]["profile"] == "verified"
    assert record["compilation_certificate"]["arena"]["compiled_bytes"] > 0
