import importlib.util
import json
from pathlib import Path

import pytest


def load_tool():
    path = Path(__file__).parents[2] / "tools" / "diagnose_exported_numerics.py"
    spec = importlib.util.spec_from_file_location("numerical_probe_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def capture(tmp_path):
    tool = load_tool()
    program, evidence, manifest = [tmp_path / name for name in ("p.pt2", "e.json", "m.json")]
    program.write_bytes(b"serialized program")
    evidence.write_text('{"graph_digest": "before-serialization"}')
    record = {
        "export_sha256": tool.digest(program),
        "capture_sha256": tool.digest(evidence),
    }
    manifest.write_text(json.dumps({"regions": [record]}))
    return tool, program, evidence, manifest


def test_sources_are_bound_by_serialized_bytes(capture):
    tool, program, evidence, manifest = capture
    result = tool.verify_capture_sources(manifest, program, evidence)
    assert result == {
        "capture_manifest": tool.digest(manifest),
        "exported_program": tool.digest(program),
        "capture_evidence": tool.digest(evidence),
    }


@pytest.mark.parametrize("changed", ["program", "evidence", "crossed", "duplicate"])
def test_rejects_mutated_or_ambiguous_source_pair(capture, changed):
    tool, program, evidence, manifest = capture
    if changed == "program":
        program.write_bytes(b"changed")
    elif changed == "evidence":
        evidence.write_text("{}")
    else:
        record = json.loads(manifest.read_text())["regions"][0]
        regions = [record, record] if changed == "duplicate" else [
            {**record, "export_sha256": "another-program"},
            {**record, "capture_sha256": "another-evidence"},
        ]
        manifest.write_text(json.dumps({"regions": regions}))
    with pytest.raises(ValueError, match="bind exactly"):
        tool.verify_capture_sources(manifest, program, evidence)
