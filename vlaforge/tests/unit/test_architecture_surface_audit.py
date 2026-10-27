from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest


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


def test_architecture_surface_is_passive_and_root_is_vlaforge_only() -> None:
    audit = _module()

    report = audit.audit_repository()

    assert report["passed"] is True
    assert report["production_surface"]["forbidden_findings"] == []
    assert report["semantic_ir"]["old_control_or_publish_opcodes"] == []
    assert report["semantic_ir"]["core_action_queue"] is False
    assert set(report["build_graph"]["tracked_cuda_sources"]) <= set(report["build_graph"]["optional_cuda_providers"])
    assert "vlaforge/backends/tensorrt_precision_plugins.cu" in report["build_graph"]["optional_cuda_providers"]
    assert report["build_graph"]["invalid_edges"] == []
    assert report["build_graph"]["root_build_drives_vlaforge"] is True
    assert report["build_graph"]["root_legacy_build_references"] == []
    assert report["build_graph"]["retired_tracked_files"] == []


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


def test_unapproved_cuda_source_is_still_forbidden(monkeypatch):
    audit = _module()
    original = audit._tracked_files
    extra = audit._SOURCE_ROOT / "runtime/unapproved.cu"
    monkeypatch.setattr(audit, "_tracked_files", lambda roots: (*original(roots), extra)
                        if tuple(roots) == ("vlaforge",) else original(roots))
    with pytest.raises(ValueError, match="build surface"):
        audit._audit_cmake()


def test_approved_provider_source_cannot_link_into_core(monkeypatch):
    audit = _module()
    original = Path.read_text
    main = audit._SOURCE_ROOT / "CMakeLists.txt"

    def read(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        return value + "\ntarget_sources(vlaforge_runtime PRIVATE backends/tensorrt_precision_plugins.cu)\n" if path == main else value

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(ValueError, match="build surface"):
        audit._audit_cmake()


def test_precision_provider_cannot_be_enabled_by_default(monkeypatch):
    audit = _module()
    original = Path.read_text
    main = audit._SOURCE_ROOT / "CMakeLists.txt"

    def read(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        return value.replace('BF16 rounding plugin" OFF)', 'BF16 rounding plugin" ON)') if path == main else value

    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(ValueError, match="build surface"):
        audit._audit_cmake()
