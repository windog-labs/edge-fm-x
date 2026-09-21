import csv
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit_tool():
    path = Path(__file__).resolve().parents[2] / "tools/audit_multimodal_baseline.py"
    spec = importlib.util.spec_from_file_location("multimodal_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evidence(tmp_path, audit_tool):
    np = pytest.importorskip("numpy")

    def write(path, data):
        path.write_text(json.dumps(data))

    protocol = {"schema": "vlaforge.multimodal_baseline_protocol/1", "mode": "pilot", "backend": "python",
                "workers": 1, "warmup": 1, "measured": 1, "new_tokens": 2, "gpu_uuid": "GPU-test"}
    write(tmp_path / "protocol.json", protocol)
    folder, monitor = tmp_path / "worker-00", tmp_path / "monitor-00"
    folder.mkdir()
    monitor.mkdir()
    np.save(folder / "reference_tokens.npy", np.array([1, 2], dtype=np.int64), allow_pickle=False)
    np.save(folder / "generated_tokens.npy", np.array([[1, 2], [1, 2]], dtype=np.int64), allow_pickle=False)
    np.savez(folder / "prepared_inputs.npz", pixel_values=np.ones((1, 4)))
    with (folder / "samples.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "phase", "prepare_h2d_ns", "ttft_ns", "resident_generate_ns", "input_to_tokens_ns", "decode_tokens_per_s", "new_tokens"])
        writer.writerows([[0, "warmup", 100, 200, 300, 400, 5e6, 2], [1, "measured", 100, 200, 300, 400, 5e6, 2]])
    write(monitor / "preflight-owners.json", [])
    write(monitor / "monitor.json", {"status": "exited", "exitcode": 0, "monitored_gpu": "GPU-test",
          "container_pid": 10, "nvml_pid": 100,
          "observations": [{"stage": stage, "owners": [] if stage == "reset" else [{"pid": 100}]}
                           for stage in ("registered-1", "reset", "registered-2", "running")]})

    def seal():
        write(folder / "execution.json", {"status": "passed", "pid": 10, "gpu": {"uuid": "GPU-test"},
              "protocol_sha256": audit_tool.digest(tmp_path / "protocol.json"),
              "files": {p.name: audit_tool.digest(p) for p in folder.iterdir() if p.name != "execution.json"}})
        write(tmp_path / "campaign.json", {"status": "passed", "mode": "pilot",
              "protocol_sha256": audit_tool.digest(tmp_path / "protocol.json"), "runs": [{"worker": 0,
              "exit_code": 0, "start_ns": 1, "end_ns": 2,
              "report_sha256": audit_tool.digest(folder / "execution.json"),
              "monitor_sha256": audit_tool.digest(monitor / "monitor.json")}]})

    seal()
    return tmp_path, seal


def test_full_pilot_recomputed_from_raw(audit_tool, evidence):
    root, _ = evidence
    result = audit_tool.audit(root)
    assert result["workers"][0]["complete_tokens_exact"] is True
    assert result["mode"] == "pilot" and result["full_goal_acceptance"] is False


def test_corrupt_tokens_rejected_even_with_updated_file_hashes(audit_tool, evidence):
    np = pytest.importorskip("numpy")
    root, seal = evidence
    np.save(root / "worker-00/generated_tokens.npy", np.array([[1, 2], [1, 3]], dtype=np.int64), allow_pickle=False)
    seal()
    with pytest.raises(ValueError, match="token output"):
        audit_tool.audit(root)


def test_foreign_owner_cannot_be_certified_by_pass_flag(audit_tool, evidence):
    root, seal = evidence
    path = root / "monitor-00/monitor.json"
    data = json.loads(path.read_text())
    data["observations"][-1]["owners"].append({"pid": 999})
    path.write_text(json.dumps(data))
    seal()
    with pytest.raises(ValueError, match="foreign GPU"):
        audit_tool.audit(root)
