import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tool():
    tools = Path(__file__).resolve().parents[2] / "tools"
    sys.path.insert(0, str(tools))
    try:
        spec = importlib.util.spec_from_file_location("multimodal_bench", tools / "benchmark_qwen35.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(tools))


def test_decode_rate_excludes_first_token(tool):
    assert tool.decode_rate(16, 1_000_000_000, 2_000_000_000) == 15.0
    assert tool.decode_rate(1, 10, 20) is None
    with pytest.raises(ValueError):
        tool.decode_rate(16, 20, 10)


def test_transformers_loading_sets_are_structured_json(tool):
    info = {"missing_keys": set(), "unexpected_keys": {"b", "a"}}
    assert json.loads(json.dumps(tool.loading_metadata(info))) == {
        "missing_keys": [], "unexpected_keys": ["a", "b"]}


def test_first_token_timestamp_follows_cpu_completion(tool):
    sequence = []

    class Tokens:
        def detach(self):
            return self

        def cpu(self):
            sequence.append("cpu")
            return self

        def reshape(self, *_):
            return self

        def tolist(self):
            return [10]

    stream = tool.TokenStream(clock=lambda: sequence.append("clock") or 99)
    stream.put(Tokens())  # Prompt is not a generated token.
    stream.put(Tokens())
    assert sequence == ["cpu", "clock"]
    assert stream.first_ns == 99
    assert stream.tokens == [10]


def test_unknown_gpu_uuid_is_rejected_before_launch(tool, monkeypatch):
    monkeypatch.setattr(tool.subprocess, "check_output", lambda *a, **kw: "GPU-other, NVIDIA H20, 535, 0, 0\n")
    with pytest.raises(ValueError, match="UUID"):
        tool.device_snapshot("GPU-requested")


def test_formal_threshold_is_not_silently_downgraded(tool):
    args = SimpleNamespace(workers=5, warmup=128, measured=1024, new_tokens=16, mode="formal")
    tool.validate_settings(args)
    args.workers = 1
    with pytest.raises(ValueError, match="formal"):
        tool.validate_settings(args)


def test_reference_check_rejects_changed_or_truncated_tokens(tool):
    np = pytest.importorskip("numpy")
    ref = np.array([1, 2, 3], dtype=np.int64)
    assert tool.check_tokens(ref.copy(), ref) is None
    with pytest.raises(ValueError, match="token"):
        tool.check_tokens(ref[:2], ref)
    with pytest.raises(ValueError, match="token"):
        tool.check_tokens(ref + 1, ref)


def test_controller_serializes_monitored_workers_and_binds_reports(tool, tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    image = tmp_path / "input.jpg"
    image.write_bytes(b"test")
    args = SimpleNamespace(model=model, image=image, output=tmp_path / "new", gpu="GPU-test",
                           prompt="describe", workers=3, warmup=1, measured=1, new_tokens=2,
                           mode="pilot", worker=None, vision_module=None)
    monkeypatch.setattr(tool, "device_snapshot", lambda gpu: {"uuid": gpu})
    calls = []

    def monitored(command, folder, **kwargs):
        index = len(calls)
        assert index == 0 or (args.output / f"worker-{index-1:02d}/execution.json").exists()
        assert kwargs["gpu"] == "GPU-test"
        assert all(isinstance(item, str) for item in command)
        folder.mkdir(parents=True)
        worker = args.output / f"worker-{index:02d}"
        worker.mkdir()
        tool.write(worker / "execution.json", {"status": "passed", "protocol_sha256": tool.sha(args.output / "protocol.json")})
        tool.write(folder / "monitor.json", {"exitcode": 0, "container_pid": 100 + index})
        calls.append(index)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tool, "run_monitored", monitored)
    assert tool.run_controller(args) == 0
    assert calls == [0, 1, 2]
    campaign = json.loads((args.output / "campaign.json").read_text())
    assert campaign["mode"] == "pilot"
    assert campaign["no_python_deployment"] is False
    assert all(row["report_sha256"] for row in campaign["runs"])


def test_failed_worker_blocks_subsequent_launches(tool, tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    image = tmp_path / "input.jpg"
    image.write_bytes(b"test")
    args = SimpleNamespace(model=model, image=image, output=tmp_path / "new", gpu="GPU-test",
                           prompt="describe", workers=3, warmup=1, measured=1, new_tokens=2,
                           mode="pilot", worker=None, vision_module=None)
    monkeypatch.setattr(tool, "device_snapshot", lambda gpu: {"uuid": gpu})
    calls = []

    def fail(command, folder, **kwargs):
        calls.append(command)
        raise RuntimeError("foreign owner")

    monkeypatch.setattr(tool, "run_monitored", fail)
    assert tool.run_controller(args) == 1
    assert len(calls) == 1


def test_existing_evidence_directory_is_never_overwritten(tool, tmp_path):
    with pytest.raises(FileExistsError):
        tool.run_controller(SimpleNamespace(output=tmp_path))
