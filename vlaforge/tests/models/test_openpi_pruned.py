"""Small CPU derived-evidence contract tests, not pretrained OpenPI evidence."""

from argparse import Namespace
import gc
import json
from pathlib import Path
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_phased import tensor_tree_metadata
from vlaforge.adapters.openpi.openpi_pruned import _compare, _validated_regions, prune_region, reference
from vlaforge.ir.program import Module, InputPort, OutputPort, TensorRegion, Value
from vlaforge.ir.serializer import canonical_json
from vlaforge.ir.types import TensorType
from vlaforge.numerical_context import snapshot


def source(tmp_path):
    class Small(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Parameter(torch.ones(2))
            self.unused = torch.nn.Parameter(torch.ones(8))
        def forward(self, value):
            return value + self.used
    root = tmp_path / "original"
    folder = root / "exported_regions"
    folder.mkdir(parents=True)
    value = torch.tensor([2., 3.])
    original = Small().eval()
    ep = torch.export.export(original, (value,))
    torch.export.save(ep, folder / "prefix.pt2")
    specimen = {"args": (value,), "expected": original(value).detach()}
    torch.save(specimen, folder / "prefix.examples.pt")
    tensor = TensorType((2,), "f32")
    module = Module("prune_cpu_fixture", (InputPort("noise", tensor, input_id=0, device="cpu"),),
                    (OutputPort("normalized_action_chunk", tensor, output_id=0, device="cpu"),), (),
                    (TensorRegion("prefix", (Value("value", tensor),), (tensor,)),), ())
    (folder / "invocation_ir.json").write_text(canonical_json(module, indent=2) + "\n")
    context = snapshot().to_dict()
    report = {"status": "persisted", "device": "cpu", "process_identity": {"pid": 999999999, "start_ticks": 1},
              "numerical_context": context, "capture": {"schema": "vlaforge.openpi_persisted_capture/1",
              "status": "persisted_awaiting_independent_reload", "saved_reload_verified": False,
              "numerical_context": context, "invocation_ir": file_digest(folder / "invocation_ir.json"), "regions": [
                  {"region": "prefix", "supported": True, "saved_reload_region_parity": "not-run",
                   "archive": {"path": str(folder / "prefix.pt2"), **file_digest(folder / "prefix.pt2")},
                   "examples": {"path": str(folder / "prefix.examples.pt"), **file_digest(folder / "prefix.examples.pt")},
                   "capture_evidence": {"schema": "cpu-fixture-not-model", "graph_digest": "0" * 64, "inputs": [], "outputs": []},
                   "example_metadata": tensor_tree_metadata(specimen)}]}}
    path = root / "report.json"
    path.write_text(json.dumps(report))
    return path


def test_region_releases_original_before_serialized_pruned_reload(tmp_path, monkeypatch):
    path = source(tmp_path)
    reference_hash = file_digest(path)["sha256"]
    output = tmp_path / "regions/prefix"
    actual_load = torch.export.load
    owners = []
    def load(archive, *args, **kwargs):
        if owners:
            gc.collect()
            assert owners[0]() is None, "original EP still retained during pruned load"
        result = actual_load(archive, *args, **kwargs)
        owners.append(weakref.ref(result))
        return result
    monkeypatch.setattr(torch.export, "load", load)
    prune_region(Namespace(reference_report=path, reference_sha256=reference_hash, region="prefix", output=output))
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "passed" and report["source_unique_storage_bytes"] > report["pruned_unique_storage_bytes"]
    assert report["full_model_output_verified"] is False and report["no_python_deployment"] is False
    _, _, context, _, entries = reference(path, reference_hash)
    assert set(_validated_regions(output.parent, path, entries, context)) == {"prefix"}
    assert file_digest(path)["sha256"] == reference_hash


@pytest.mark.parametrize("mutation", ["schema", "source_hash", "serialized", "computations", "ports", "example"])
def test_derived_ledger_and_complete_example_gate_fail_closed(tmp_path, mutation):
    path = source(tmp_path)
    reference_hash = file_digest(path)["sha256"]
    output = tmp_path / "regions/prefix"
    prune_region(Namespace(reference_report=path, reference_sha256=reference_hash, region="prefix", output=output))
    _, _, context, _, entries = reference(path, reference_hash)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    ledger_path = output / "pruned/ledger.json"
    ledger = json.loads(ledger_path.read_text())
    if mutation == "schema":
        ledger["schema"] = "vlaforge.unused_lifted_state_pruning/99"
    elif mutation == "source_hash":
        ledger["source_artifact_sha256"] = "0" * 64
    elif mutation == "serialized":
        ledger["pre_and_post_serialization_state_verified"] = False
    elif mutation == "computations":
        ledger["computation_nodes_removed"] = 1
    elif mutation == "ports":
        ledger["user_ports_unchanged"] = False
    else:
        report["saved_pruned_vs_captured_example"][0]["bitwise_equal"] = False
    ledger_path.write_text(json.dumps(ledger))
    report["ledger"].update(file_digest(ledger_path))
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        _validated_regions(output.parent, path, entries, context)


def test_complete_bits_not_only_float_value_equality():
    with pytest.raises(ValueError, match="storage bits"):
        _compare(torch.tensor([0.]), torch.tensor([-0.]))


def _joined_fixture(tmp_path):
    from vlaforge.adapters.openpi.openpi_pruned_capture import prepare_pruned_capture

    path = source(tmp_path)
    row = json.loads(path.read_text())
    value = np.array([2., 3.], dtype=np.float32)
    np.savez(path.parent / "actions.npz", normalized_reference=value, physical_reference=value)
    np.savez(path.parent / "prepared_inputs.npz", noise=value)
    row.update(actions=file_digest(path.parent / "actions.npz"), prepared_inputs=file_digest(path.parent / "prepared_inputs.npz"))
    path.write_text(json.dumps(row))
    digest = file_digest(path)["sha256"]
    regions = tmp_path / "regions"
    prune_region(Namespace(reference_report=path, reference_sha256=digest, region="prefix", output=regions / "prefix"))
    _, _, context, _, entries = reference(path, digest)
    checked = _validated_regions(regions, path, entries, context)
    whole = tmp_path / "whole"
    whole.mkdir()
    np.savez(whole / "actions.npz", normalized=value, native=value)
    (whole / "invocation_trace.json").write_text('{"fixture":true}')
    report = {"schema": "vlaforge.openpi_pruned_whole_ir/1", "status": "passed", "complete_bitwise_equal": True,
              "original_reference": {"path": str(path), **file_digest(path)}, "regions": checked,
              "numerical_context": context.to_dict(), "no_python_deployment": False,
              "physical_units_verified": False, "robot_calibration_verified": False,
              "actions": file_digest(whole / "actions.npz"), "trace": file_digest(whole / "invocation_trace.json"),
              "fidelity": [{"fixture_only": True}]}
    proof = whole / "report.json"
    proof.write_text(json.dumps(report))
    output = tmp_path / "joined"
    result = prepare_pruned_capture(path, digest, regions, proof, file_digest(proof)["sha256"], output)
    return path, proof, output / "capture.json", result


def test_derived_join_preserves_original_and_changes_graph_archive_identity(tmp_path):
    from vlaforge.adapters.openpi.openpi_aoti import _capture_source
    from vlaforge.adapters.openpi.openpi_pruned_capture import verify_pruned_capture_join

    original, whole, joined, record = _joined_fixture(tmp_path)
    source = json.loads(joined.read_text())
    assert json.loads(original.read_text())["status"] == "persisted"
    assert source["capture"]["regions"][0]["capture_evidence"]["graph_digest"] != "0" * 64
    assert source["capture"]["regions"][0]["source_capture_evidence"]["graph_digest"] == "0" * 64
    verify_pruned_capture_join(source, joined)
    _capture_source(joined, record["sha256"])
    assert source["no_python_deployment"] is False


@pytest.mark.parametrize("mutation", ["schema", "numeric_type", "archive_identity", "graph", "port", "input", "outputs", "whole_status", "whole_bytes"])
def test_derived_join_rejects_false_or_tampered_proofs(tmp_path, mutation):
    from vlaforge.adapters.openpi.openpi_aoti import _capture_source

    original, whole, joined, _ = _joined_fixture(tmp_path)
    source = json.loads(joined.read_text())
    if mutation == "schema":
        source["schema"] = "vlaforge.openpi_validated_pruned_capture/99"
    elif mutation == "numeric_type":
        source["numerical_context"]["cudnn_enabled"] = 1
    elif mutation == "archive_identity":
        source["capture"]["regions"][0]["archive"] = json.loads(original.read_text())["capture"]["regions"][0]["archive"]
    elif mutation == "graph":
        source["capture"]["regions"][0]["capture_evidence"]["graph_digest"] = "0" * 64
    elif mutation == "port":
        source["capture"]["regions"][0]["capture_evidence"]["inputs"] = [{}]
    elif mutation in {"input", "outputs"}:
        (joined.parent / ("prepared_inputs.npz" if mutation == "input" else "actions.npz")).write_bytes(b"changed")
    else:
        row = json.loads(whole.read_text())
        if mutation == "whole_status":
            row["status"] = "failed"
        else:
            np.savez(whole.parent / "actions.npz", normalized=np.array([2., 4.], dtype=np.float32), native=np.array([2., 3.], dtype=np.float32))
            row["actions"] = file_digest(whole.parent / "actions.npz")
        whole.write_text(json.dumps(row))
        source["derivation"]["whole_ir"].update(file_digest(whole))
    joined.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        _capture_source(joined, file_digest(joined)["sha256"])


def test_failed_create_only_join_never_publishes_complete_capture(tmp_path, monkeypatch):
    import vlaforge.adapters.openpi.openpi_pruned_capture as join

    original, whole, joined, _ = _joined_fixture(tmp_path)
    def fail_copy(*args, **kwargs):
        raise OSError("injected copy failure")
    monkeypatch.setattr(join.shutil, "copyfile", fail_copy)
    failed = tmp_path / "failed"
    with pytest.raises(OSError):
        join.prepare_pruned_capture(original, file_digest(original)["sha256"], tmp_path / "regions", whole, file_digest(whole)["sha256"], failed)
    assert not (failed / "capture.json").exists()
