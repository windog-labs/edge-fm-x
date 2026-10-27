import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper

SPEC = importlib.util.spec_from_file_location(
    "export_torchscript_region_onnx",
    Path(__file__).resolve().parents[2] / "tools/export_torchscript_region_onnx.py",
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def stage_model(root):
    root.mkdir()
    tensor = numpy_helper.from_array(np.array([1, 2], dtype=np.float32), "weight")
    model = helper.make_model(helper.make_graph(
        [helper.make_node("Identity", ["weight"], ["output"])], "external", [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])], [tensor],
    ), opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    staged = root / "model.onnx"
    onnx.save_model(model, staged, save_as_external_data=True,
                    all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
    return staged, onnx.load(staged, load_external_data=False)


def test_external_weights_are_published_and_load_after_staging_is_removed(tmp_path):
    staged, model = stage_model(tmp_path / "staged")
    output = tmp_path / "model.onnx"
    files = TOOL.external_files(model, staged.parent)
    assert files == [Path("weights.bin")]
    published = TOOL.publish_files(staged, output, files)
    assert published == [tmp_path / "weights.bin", output]
    for path in staged.parent.iterdir():
        path.unlink()
    staged.parent.rmdir()
    onnx.checker.check_model(str(output))
    restored = onnx.load(output, load_external_data=True)
    np.testing.assert_array_equal(numpy_helper.to_array(restored.graph.initializer[0]), [1, 2])


def test_existing_weight_is_not_overwritten(tmp_path):
    staged, model = stage_model(tmp_path / "staged")
    existing = tmp_path / "weights.bin"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        TOOL.publish_files(staged, tmp_path / "model.onnx", TOOL.external_files(model, staged.parent))
    assert existing.read_bytes() == b"keep"
    assert not (tmp_path / "model.onnx").exists()


def test_racing_graph_does_not_remove_someone_elses_output(tmp_path):
    staged, model = stage_model(tmp_path / "staged")
    output = tmp_path / "model.onnx"
    output.write_bytes(b"keep graph")
    with pytest.raises(FileExistsError):
        TOOL.publish_files(staged, output, TOOL.external_files(model, staged.parent))
    assert output.read_bytes() == b"keep graph"
    assert not (tmp_path / "weights.bin").exists()


@pytest.mark.parametrize("location", ["../outside", "/tmp/outside", ""])
def test_external_paths_cannot_escape_export_directory(tmp_path, location):
    staged, model = stage_model(tmp_path / "staged")
    for entry in model.graph.initializer[0].external_data:
        if entry.key == "location":
            entry.value = location
    with pytest.raises(ValueError, match="inside"):
        TOOL.external_files(model, staged.parent)


def test_external_weight_symlink_is_rejected(tmp_path):
    staged, model = stage_model(tmp_path / "staged")
    weight = staged.parent / "weights.bin"
    weight.unlink()
    weight.symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes"):
        TOOL.external_files(model, staged.parent)


def test_inline_model_needs_no_sidecar(tmp_path):
    staged, _ = stage_model(tmp_path / "staged")
    model = onnx.load(staged, load_external_data=True)
    assert TOOL.external_files(model, staged.parent) == []


def test_pruned_middle_input_keeps_original_named_binding():
    torch = pytest.importorskip("torch")
    examples = tuple(torch.tensor([index]) for index in range(3))
    result = TOOL.runtime_feeds(["input_0", "input_2"],
                                ["input_0", "input_1", "input_2"], examples)
    np.testing.assert_array_equal(result["input_2"], [2])
    assert TOOL.runtime_feeds([], ["input_0", "input_1", "input_2"], examples) == {}
    with pytest.raises(ValueError, match="names"):
        TOOL.runtime_feeds(["unrecognized"], ["input_0", "input_1", "input_2"], examples)


def test_explicit_device_relocation_changes_no_dtype_or_scalar():
    torch = pytest.importorskip("torch")

    class Factory(torch.nn.Module):
        def forward(self, value):
            return value + torch.arange(3, dtype=torch.float32, device="cpu")

    module = torch.jit.trace(Factory(), (torch.ones(3),))
    constants = [node for node in module.graph.nodes() if node.kind() == "prim::Constant"]
    device = next(node for node in constants if isinstance(node.output().toIValue(), torch.device))
    scalars = [(str(node.output().type()), node.output().toIValue()) for node in constants if node != device]
    device.s_("value", "cuda:0")
    changes = TOOL.relocate_device_constants(module, "cpu")
    assert changes == [{"module": "", "method": "forward", "from": "cuda:0", "to": "cpu"}]
    assert [(str(node.output().type()), node.output().toIValue()) for node in constants if node != device] == scalars
    torch.testing.assert_close(module(torch.ones(3)), torch.tensor([1., 2., 3.]))
    assert TOOL.relocate_device_constants(module, "cpu") == []


def test_nested_graph_attribute_external_tensor_is_discovered(tmp_path):
    staged, model = stage_model(tmp_path / "staged")
    nested = helper.make_graph(
        [helper.make_node("Constant", [], ["nested_output"], value=model.graph.initializer[0])],
        "nested", [], [helper.make_tensor_value_info("nested_output", TensorProto.FLOAT, [2])])
    del model.graph.initializer[:]
    model.graph.node.append(helper.make_node("If", ["condition"], ["branch"],
                                             then_branch=nested, else_branch=nested))
    assert TOOL.external_files(model, staged.parent) == [Path("weights.bin")]


def test_cli_validates_export_with_an_unused_middle_input(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    class Region(torch.nn.Module):
        def forward(self, first, unused, last):
            return first + last

    examples = tuple(torch.tensor([float(index)]) for index in range(3))
    archive, inputs = tmp_path / "region.pt", tmp_path / "inputs.pt"
    torch.jit.trace(Region(), examples).save(str(archive))
    torch.save(examples, inputs)
    output = tmp_path / "region.onnx"
    monkeypatch.setattr("sys.argv", ["export", "--torchscript", str(archive),
        "--inputs", str(inputs), "--output", str(output), "--device", "cpu", "--opset", "17"])
    assert TOOL.main() == 0
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["runtime_parity"]["status"] == "passed"
    assert report["onnx_input_names"] == ["input_0", "input_2"]
    assert report["runtime_parity"]["outputs"][0]["exact"]
    assert report["external_data"] == []


def test_reference_failure_writes_a_failure_receipt_without_a_graph(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    examples = (torch.tensor([1.]),)
    archive, inputs = tmp_path / "region.pt", tmp_path / "inputs.pt"
    torch.jit.trace(torch.nn.Identity(), examples).save(str(archive))
    torch.save({"not": "a tuple"}, inputs)
    output = tmp_path / "region.onnx"
    monkeypatch.setattr("sys.argv", ["export", "--torchscript", str(archive),
        "--inputs", str(inputs), "--output", str(output), "--device", "cpu"])
    with pytest.raises(ValueError, match="tuple"):
        TOOL.main()
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["status"] == "failed" and report["phase"] == "load-or-reference"
    assert not output.exists()
