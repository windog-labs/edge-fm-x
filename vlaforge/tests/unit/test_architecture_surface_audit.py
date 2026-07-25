from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_architecture_surface.py"
    )
    specification = importlib.util.spec_from_file_location(
        "audit_architecture_surface",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_architecture_surface_is_passive_and_build_isolated() -> None:
    audit = _module()

    report = audit.audit_repository()

    assert report["passed"] is True
    assert report["production_surface"]["forbidden_findings"] == []
    assert report["semantic_ir"]["old_control_or_publish_opcodes"] == []
    assert report["semantic_ir"]["core_action_queue"] is False
    assert report["build_graph"]["tracked_cuda_sources"] == []
    assert report["build_graph"]["invalid_edges"] == []
    assert (
        report["build_graph"]["root_edgefm_build_drives_vlaforge"] is False
    )


def test_forbidden_scanner_reports_category_path_and_line(
    tmp_path: Path,
) -> None:
    audit = _module()
    source = tmp_path / "legacy.cpp"
    source.write_text("safe\nRunTick();\n", encoding="utf-8")

    original_relative = audit._relative
    audit._relative = lambda path: path.name
    try:
        findings = audit.scan_forbidden_text(
            (source,),
            {"tick": re.compile(r"\bRunTick\b")},
        )
    finally:
        audit._relative = original_relative

    assert findings == [
        {
            "category": "tick",
            "path": "legacy.cpp",
            "line": 2,
            "match": "RunTick",
        }
    ]
