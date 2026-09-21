import importlib.util
from pathlib import Path

import pytest


def monitor():
    path = Path(__file__).resolve().parents[2] / "tools/cogact_gpu_monitor.py"
    spec = importlib.util.spec_from_file_location("gpu_monitor_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("selector", ["0", "GPU-1234", "1"])
def test_owners_query_the_requested_physical_gpu(monkeypatch, selector):
    module = monitor()

    def query(args, **kwargs):
        assert args[args.index("-i") + 1] == selector
        return "123, python, 42\n"

    monkeypatch.setattr(module.subprocess, "check_output", query)
    assert module.owners(selector) == [
        {"pid": 123, "name": "python", "memory_mib": "42"}
    ]


def test_nondefault_gpu_owner_blocks_launch_without_touching_other_gpus(
    tmp_path, monkeypatch
):
    module = monitor()

    def occupied(gpu):
        assert gpu == "GPU-second"
        return [{"pid": 123}]

    monkeypatch.setattr(module, "owners", occupied)
    with pytest.raises(RuntimeError, match="GPU-second"):
        module.run_monitored(["never-run"], tmp_path / "run", gpu="GPU-second")
