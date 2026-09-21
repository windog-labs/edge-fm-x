import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).parents[2] / "tools/upgrade_session_protocol_v1.py"
SPEC = importlib.util.spec_from_file_location("upgrade_session_protocol_v1", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
migrate = MODULE.migrate


def test_migration_preserves_input_and_reference_bindings(tmp_path):
    protocol = {
        "schema": "vlaforge.session_latency_protocol/1", "bundles": {"off": "/bundle"},
        "active_dimensions": [0, 1], "evidence": ["old.json"],
        "samples": [{"sample_id": "s0", "inputs": {"x": "/x.bin"}, "direct": "/d.npy", "eager": "/e.npy"}],
    }
    module = {"outputs": [{"name": "action_chunk"}]}
    protocol_path, module_path = tmp_path / "protocol.json", tmp_path / "module.json"
    protocol_path.write_text(json.dumps(protocol)); module_path.write_text(json.dumps(module))
    result = migrate(protocol_path, module_path, tmp_path / "migrated.json")
    assert result["schema"] == "vlaforge.session_latency_protocol/2"
    assert result["samples"][0]["inputs"] == protocol["samples"][0]["inputs"]
    assert result["samples"][0]["outputs"] == {"action_chunk": {"direct": "/d.npy", "eager": "/e.npy"}}
    assert result["bundles"]["required"] == "/bundle"
    migration = json.loads((tmp_path / "migrated.migration.json").read_text())
    assert migration["input_bytes_unchanged"] is True


def test_migration_rejects_ambiguous_or_already_v2_protocol(tmp_path):
    module = tmp_path / "module.json"; module.write_text(json.dumps({"outputs": [{"name": "x"}]}))
    for value in ({"schema": "vlaforge.session_latency_protocol/2"},
                  {"schema": "vlaforge.session_latency_protocol/1", "samples": []},
                  {"schema": "vlaforge.session_latency_protocol/1", "samples": [{"sample_id": "s", "inputs": {}, "direct": "/a", "eager": "/b", "extra": 1}]}):
        path = tmp_path / "protocol.json"; path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            migrate(path, module, tmp_path / "out.json")
