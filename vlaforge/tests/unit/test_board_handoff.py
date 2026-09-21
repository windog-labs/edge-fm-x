"""Synthetic contract tests only; real model pack verification is archived separately."""

import json

import numpy as np
import pytest
from vlaforge.ir.program import InputPort, Module, OutputPort
from vlaforge.ir.serializer import module_to_data
from vlaforge.ir.types import TensorType
from vlaforge.validation.board_handoff import (
    DATA_SCHEMA,
    prepare,
    read_json,
    safe_path,
    sha256,
    stage,
    target_pending,
    target_template,
    validate,
    write_new,
)
from vlaforge.validation.session_benchmark import BOUNDARY, MULTI_SCHEMA, SCHEMA


def source(tmp_path, multi=False):
    root = tmp_path / "source"
    root.mkdir()
    tensor = TensorType((1, 2), "f32")
    integer = TensorType((1, 2), "i64")
    outputs = (OutputPort("actions", tensor, output_id=0, device="cuda:0"),)
    if multi:
        outputs += (OutputPort("tokens", integer, output_id=1, device="cuda:0"),)
    module = Module("typed_fixture", (InputPort("noise", tensor, input_id=0, device="cuda:0"),), outputs, (), (), ())
    write_new(root / "module.json", module_to_data(module))
    value = np.asarray([[1., 2.]], dtype=np.float32)
    (root / "noise.bin").write_bytes(value.tobytes())
    np.save(root / "eager.npy", value)
    np.save(root / "direct.npy", value)
    sample = {"inputs": {"noise": str(root / "noise.bin")},
              "eager": str(root / "eager.npy"), "direct": str(root / "direct.npy")}
    protocol = {"schema": SCHEMA, "boundary": BOUNDARY,
                "warmup": 128, "measured": 1024, "processes": 5,
                "policies": ["off"], "bundles": {"off": "/not-shipped/x86-bundle"},
                "quality_gate": "passed", "eager_validation": "bitwise",
                "checkpoint_sha256": "1" * 64, "samples": [sample],
                "evidence": [str(root / "module.json")]}
    if multi:
        np.save(root / "tokens.npy", np.asarray([[2**54, -(2**54) + 1]], dtype=np.int64))
        protocol.update(schema=MULTI_SCHEMA,
                        paper_gates={"mse_max": 1e-5, "cosine_min": .9999},
                        outputs=[{"name": "actions", "role": "primary-action"},
                                 {"name": "tokens", "role": "exact"}])
        protocol["samples"] = [{"sample_id": 0, "inputs": sample["inputs"], "outputs": {
            "actions": {key: sample[key] for key in ("eager", "direct")},
            "tokens": {key: str(root / "tokens.npy") for key in ("eager", "direct")}}}]
    write_new(root / "protocol.json", protocol)
    write_new(root / "selection.json", {
        "schema": "vlaforge.board_source_selection/1", "identity": {"kind": "synthetic-unit-test"},
        "limitations": ["not real model evidence"], "files": [
            {"path": str(root / "module.json"), "sha256": sha256(root / "module.json"), "role": "test-ir"}]})
    return root


def pack(tmp_path, multi=False):
    root = source(tmp_path, multi)
    destination = tmp_path / "pack"
    result = prepare(root / "protocol.json", root / "module.json", root / "selection.json", destination)
    return root, destination, result


def data_source(tmp_path, count=3):
    root = source(tmp_path, multi=True)
    original = read_json(root / "protocol.json")
    protocol = {key: original[key] for key in ("boundary", "checkpoint_sha256", "outputs")}
    protocol.update(schema=DATA_SCHEMA, samples=[dict(original["samples"][0], sample_id=index)
                                               for index in range(count)])
    (root / "protocol.json").write_text(json.dumps(protocol))
    return root, protocol


@pytest.mark.parametrize("count", [3, 115])
def test_data_only_source_needs_no_invented_benchmark_schedule(tmp_path, count):
    root, protocol = data_source(tmp_path, count)
    destination = tmp_path / "pack"
    result = prepare(root / "protocol.json", root / "module.json", root / "selection.json", destination)
    root.rename(tmp_path / "unavailable-original-source")
    assert result == validate(destination)
    assert result["sample_count"] == count and result["scope"] == "offline-data-integrity-only"
    assert len(result["comparisons"]) == 2 * count
    assert read_json(destination / "source/protocol.json") == protocol
    assert all(key not in protocol for key in ("warmup", "measured", "processes", "quality_gate", "bundles"))


@pytest.mark.parametrize("mutation", ["warmup", "measured", "claim", "duplicate", "bool-id", "empty-id",
                                      "empty-samples", "missing-output", "missing-reference", "bad-path",
                                      "bad-checkpoint", "blank-boundary", "unknown-schema"])
def test_data_only_source_rejects_invented_statistics_and_malformed_samples(tmp_path, mutation):
    root, protocol = data_source(tmp_path)
    if mutation in ("warmup", "measured"):
        protocol[mutation] = 1024
    elif mutation == "claim":
        protocol["board_verified"] = True
    elif mutation == "duplicate":
        protocol["samples"][1]["sample_id"] = 0
    elif mutation in ("bool-id", "empty-id"):
        protocol["samples"][0]["sample_id"] = True if mutation == "bool-id" else ""
    elif mutation == "empty-samples":
        protocol["samples"] = []
    elif mutation == "missing-output":
        protocol["samples"][0]["outputs"].pop("tokens")
    elif mutation == "missing-reference":
        protocol["samples"][0]["outputs"]["actions"].pop("direct")
    elif mutation == "bad-path":
        protocol["samples"][0]["inputs"]["noise"] = None
    elif mutation == "bad-checkpoint":
        protocol["checkpoint_sha256"] = "not-a-digest"
    elif mutation == "blank-boundary":
        protocol["boundary"] = " "
    else:
        protocol["schema"] = "vlaforge.board_data_protocol/99"
    (root / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        prepare(root / "protocol.json", root / "module.json", root / "selection.json", tmp_path / "failed")
    assert not (tmp_path / "failed").exists()


@pytest.mark.parametrize("multi", [False, True])
def test_portable_complete_typed_pack_survives_source_removal(tmp_path, multi):
    root, destination, result = pack(tmp_path, multi)
    root.rename(tmp_path / "unavailable-original-paths")
    moved = tmp_path / "moved"
    destination.rename(moved)
    assert validate(moved) == result
    assert result["board_executed"] is False and result["board_verified"] is False
    assert all(row["complete_bitwise_equal"] for row in result["comparisons"])
    if multi:
        assert len(result["comparisons"]) == 2
        assert (moved / "samples/0000/output-001-eager.bin").read_bytes() == np.asarray([[2**54, -(2**54) + 1]], dtype="<i8").tobytes()


@pytest.mark.parametrize("field", ["board_executed", "board_verified", "physical_units_verified", "robot_calibration_verified"])
@pytest.mark.parametrize("value", [True, 0, "false"])
def test_offline_claims_fail_closed(tmp_path, field, value):
    _, destination, _ = pack(tmp_path)
    manifest = read_json(destination / "manifest.json")
    manifest[field] = value
    with pytest.raises(ValueError, match="cannot assert"):
        validate(destination, manifest=manifest)


@pytest.mark.parametrize("path", ["../escape", "/tmp/out", "a/../b", "a//b", "./a", "a\\b"])
def test_unsafe_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_symlink_and_unlisted_file_rejected(tmp_path):
    _, destination, _ = pack(tmp_path)
    (destination / "extra").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="unlisted"):
        validate(destination)
    (destination / "extra").unlink()
    original = destination / "samples/0000/input-000.bin"
    original.rename(tmp_path / "external.bin")
    original.symlink_to(tmp_path / "external.bin")
    with pytest.raises(ValueError, match="symlink"):
        validate(destination)


def test_manifest_symlink_cannot_escape_pack(tmp_path):
    _, destination, _ = pack(tmp_path)
    manifest = destination / "manifest.json"
    manifest.rename(tmp_path / "external-manifest.json")
    manifest.symlink_to(tmp_path / "external-manifest.json")
    with pytest.raises(ValueError, match="symlink"):
        validate(destination)


@pytest.mark.parametrize("mutation", ["hash", "size", "duplicate", "sample", "dtype", "missing-output", "schema", "unknown"])
def test_manifest_corruption_rejected(tmp_path, mutation):
    _, destination, _ = pack(tmp_path)
    value = read_json(destination / "manifest.json")
    if mutation == "hash":
        value["files"][0]["sha256"] = "0" * 64
    elif mutation == "size":
        value["files"][0]["size_bytes"] += 1
    elif mutation == "duplicate":
        value["files"].append(value["files"][0])
    elif mutation == "sample":
        value["samples"][0]["sample_id"] = True
    elif mutation == "dtype":
        value["inputs"][0]["dtype"] = "f16"
    elif mutation == "missing-output":
        value["samples"][0]["outputs"] = {}
    elif mutation == "schema":
        value["schema"] = "vlaforge.board_input_handoff/99"
    else:
        value["extra"] = True
    with pytest.raises(ValueError):
        validate(destination, manifest=value)


def test_create_only_and_failed_prepare_does_not_publish_manifest(tmp_path):
    root, destination, _ = pack(tmp_path)
    with pytest.raises(FileExistsError):
        prepare(root / "protocol.json", root / "module.json", root / "selection.json", destination)
    (root / "noise.bin").write_bytes(b"bad")
    failed = tmp_path / "failed"
    with pytest.raises(ValueError):
        prepare(root / "protocol.json", root / "module.json", root / "selection.json", failed)
    assert not (failed / "manifest.json").exists()


def test_atomic_publication_failure_does_not_publish_manifest(tmp_path, monkeypatch):
    root = source(tmp_path)
    destination = tmp_path / "failed-publication"
    def denied(*args, **kwargs):
        raise OSError("injected publication failure")
    monkeypatch.setattr("os.link", denied)
    with pytest.raises(OSError, match="publication"):
        prepare(root / "protocol.json", root / "module.json", root / "selection.json", destination)
    assert not (destination / "manifest.json").exists()
    assert not list(tmp_path.glob(".board-manifest-*"))


@pytest.mark.parametrize("family", ["orin-cuda", "horizon-bpu"])
@pytest.mark.parametrize("phase", ["preflight", "build", "run", "collect"])
def test_unknown_target_stages_never_execute(tmp_path, monkeypatch, family, phase):
    _, destination, _ = pack(tmp_path)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    result = stage(destination, destination / f"targets/{family}.json", phase, tmp_path / "attempt")
    assert result["status"] == "pending"
    assert "sdk_versions" in result["pending"] and "driver" in result["pending"]
    assert result["board_executed"] is False and result["board_verified"] is False


@pytest.mark.parametrize("mutation", ["family", "provider_contract", "architecture", "memory_bytes", "sdk_versions", "extra"])
def test_target_unknown_wrong_typed_fields_rejected(mutation):
    value = target_template("orin-cuda")
    value[mutation] = {"family": "unknown", "provider_contract": "bpu-region-session",
                       "architecture": "x86_64", "memory_bytes": True,
                       "sdk_versions": {}, "extra": "unknown"}[mutation]
    with pytest.raises(ValueError):
        target_pending(value)


def test_duplicate_json_fields_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match="duplicate"):
        read_json(path)
