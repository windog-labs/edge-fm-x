from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "audit_reproducibility_baseline.py"
    )
    specification = importlib.util.spec_from_file_location(path.stem, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_absolute_reference_and_command_inventory() -> None:
    audit = _module()
    report_path = audit._REPOSITORY_ROOT / "doc" / "fake.json"
    report = {
        "reproduction": {
            "command": ["python", "tool.py", "--root", "/tmp/model"],
            "nested": [{"path": "/tmp/model/artifact.pt2"}],
        },
        "committed": str(audit._REPOSITORY_ROOT / "doc" / "report.json"),
    }
    references = audit._absolute_references([(report_path, report)])
    indexed = {item["path"]: item for item in references}
    assert indexed["/tmp/model"]["inside_repository"] is False
    assert indexed["/tmp/model/artifact.pt2"]["inside_repository"] is False
    assert indexed[str(audit._REPOSITORY_ROOT / "doc" / "report.json")][
        "inside_repository"
    ]

    commands = audit._reproduction_commands([(report_path, report)])
    assert commands == [
        {
            "report": "doc/fake.json",
            "json_pointer": "$.reproduction.command",
            "command": ["python", "tool.py", "--root", "/tmp/model"],
        }
    ]


def test_reproducibility_markdown_discloses_external_archives() -> None:
    audit = _module()
    report = {
        "status": "passed",
        "repository": {
            "baseline_revision": "base",
            "audited_revision": "head",
            "frozen_core_sha256": "1" * 64,
        },
        "installed_wheel_artifact_evaluation": {
            "artifact_evaluation": {"target": "sm_86"},
            "isolation": {"package_import": "/venv/site-packages/vlaforge"},
        },
        "summary": {
            "formal_report_count": 7,
            "committed_raw_file_count": 42,
            "reproduction_command_count": 8,
            "must_archive_total_bytes": 2 * 1024**3,
            "missing_external_reference_count": 3,
        },
        "external_archive_roots": [
            {
                "path": "/tmp/real-model",
                "size_bytes": 2 * 1024**3,
                "must_archive": True,
                "exists": True,
                "reason": "real model",
            }
        ],
    }
    markdown = audit.render_markdown(report)
    assert "2.00 GiB" in markdown
    assert "required" in markdown
    assert "does not contain Orin evidence" in markdown
    assert "not real-model support evidence" in markdown
