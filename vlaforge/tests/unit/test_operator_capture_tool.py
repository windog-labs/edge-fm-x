import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def tool(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "operator_capture_tool", directory / "extract_operator_examples.py"
    )
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Model(torch.nn.Module):
    def forward(self, x):
        return x.sin()


@pytest.mark.parametrize("extension", ["npz", "pt"])
@pytest.mark.parametrize("mutation", [None, "dtype", "shape", "device"])
def test_input_formats_bind_actual_exported_tensor_contract(
    tool, tmp_path, extension, mutation
):
    x = torch.ones(2, 3)
    program = torch.export.export(Model(), (x,))
    evidence = {"inputs": [{"name": "x", "device": "cpu", "type": {"shape": [2, 3]}}]}
    value = x.double() if mutation == "dtype" else x[:1] if mutation == "shape" else x
    if mutation == "device":
        evidence["inputs"][0]["device"] = "cuda:0"
    path = tmp_path / f"inputs.{extension}"
    if extension == "npz":
        np.savez(path, x=value.numpy(), unused_noise=np.zeros(1))
    else:
        torch.save((value,), path)
    if mutation:
        with pytest.raises(ValueError, match="shape/dtype/device"):
            tool.load_runtime_inputs(path, evidence, program)
    else:
        (actual,) = tool.load_runtime_inputs(path, evidence, program)
        assert torch.equal(actual, x)
