import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from vlaforge.analysis.onnx_inventory import onnx_operator_inventory


def model(nodes=None, shape=(2, 4)):
    nodes = nodes or [helper.make_node("MatMul", ["x", "w"], ["a"], name="first"),
                      helper.make_node("MatMul", ["x", "w"], ["b"], name="second")]
    return helper.make_model(helper.make_graph(nodes, "fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
         helper.make_tensor_value_info("w", TensorProto.FLOAT, (4, 4))],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, shape),
         helper.make_tensor_value_info("b", TensorProto.FLOAT, shape)]), opset_imports=[helper.make_opsetid("", 17)])


def test_repeated_logical_signatures_group_without_inventing_layout_or_compute_precision():
    graph = model()
    original = graph.SerializeToString()
    result = onnx_operator_inventory(graph, region_name="stage")
    assert result["static_nodes"] == 2 and result["unique_signatures"] == 1
    item = result["operators"][0]
    assert item["family"] == "gemm" and item["static_nodes_per_region_call"] == 2
    assert item["signature"]["inputs"][0]["shape"] == [2, 4]
    assert item["signature"]["inputs"][0]["dtype"] == "FLOAT"
    assert item["signature"]["physical_layout"] == "not-known-from-ONNX"
    assert item["signature"]["target_compute_precision"] == "not-known-from-logical-IO"
    assert graph.SerializeToString() == original
    assert result["gpu_profiling_performed"] is False and result["measured_runtime_counts"] is False
    json.dumps(result, allow_nan=False)


def test_dynamic_dimensions_are_preserved_not_guessed_static():
    result = onnx_operator_inventory(model(shape=("sequence", 4)), region_name="dynamic")
    item = result["operators"][0]
    assert item["signature"]["inputs"][0]["shape"] == [{"symbolic": "sequence"}, 4]
    assert item["all_logical_tensor_ports_static_positive"] is False


def test_attributes_preserve_gemm_scalar_semantics():
    first = model([helper.make_node("Gemm", ["x", "w"], ["a"], alpha=1.0)])
    second = model([helper.make_node("Gemm", ["x", "w"], ["a"], alpha=2.0)])
    a = onnx_operator_inventory(first, region_name="a")["operators"][0]
    b = onnx_operator_inventory(second, region_name="b")["operators"][0]
    assert a["signature_sha256"] != b["signature_sha256"]


def test_external_initializer_payload_is_never_opened(tmp_path):
    graph = model()
    graph.graph.input.pop()
    weights = TensorProto(name="w", data_type=TensorProto.FLOAT, dims=[4, 4], data_location=TensorProto.EXTERNAL)
    weights.external_data.add(key="location", value=str(tmp_path / "intentionally-absent.bin"))
    graph.graph.initializer.append(weights)
    result = onnx_operator_inventory(graph, region_name="external")
    item = result["operators"][0]
    assert item["signature"]["input_roles"] == ["graph-input", "initializer"]
    assert item["signature"]["inputs"][1]["shape"] == [4, 4]
    assert result["external_weight_data_loading_performed"] is False
    assert not list(tmp_path.iterdir())


def test_indexing_is_not_silently_reclassified_as_embedding_or_vla():
    graph = helper.make_model(helper.make_graph([helper.make_node("Gather", ["x", "index"], ["a"])], "indexing",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 4]), helper.make_tensor_value_info("index", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 4])]), opset_imports=[helper.make_opsetid("", 17)])
    result = onnx_operator_inventory(graph, region_name="gather")
    assert result["operators"][0]["family"] == "indexing-not-verified-embedding"
    assert "unclassified is not N/A" in result["composite_semantics"]


def test_missing_output_metadata_stays_explicit_without_shape_inference():
    graph = model([helper.make_node("Unknown", ["x"], ["untyped"], domain="private.fixture")])
    graph.opset_import.append(helper.make_opsetid("private.fixture", 1))
    result = onnx_operator_inventory(graph, region_name="opaque", infer_shapes=False)
    item = result["operators"][0]
    assert item["signature"]["outputs"] == [{"missing_tensor_metadata": True}]
    assert not item["all_logical_tensor_ports_static_positive"]


@pytest.mark.parametrize("change", ["nested", "local-function", "undeclared-domain", "duplicate-attribute", "duplicate-opset", "empty-region"])
def test_unsupported_or_ambiguous_graphs_fail_explicitly(change):
    graph = model()
    region = "stage"
    if change == "nested":
        graph.graph.node[0].attribute.append(helper.make_attribute("body", copy.deepcopy(graph.graph)))
    elif change == "local-function":
        graph.functions.append(helper.make_function("fixture", "block", ["x"], ["a"], [], [helper.make_opsetid("", 17)]))
    elif change == "undeclared-domain":
        graph.graph.node[0].domain = "private.unreviewed"
    elif change == "duplicate-attribute":
        graph.graph.node[0].attribute.extend([helper.make_attribute("alpha", 1.0), helper.make_attribute("alpha", 2.0)])
    elif change == "duplicate-opset":
        graph.opset_import.append(helper.make_opsetid("ai.onnx", 18))
        graph.opset_import[0].domain = "ai.onnx"
    else:
        region = ""
    with pytest.raises(ValueError):
        onnx_operator_inventory(graph, region_name=region)


def test_standalone_cli_records_source_hash_and_never_overwrites_output(tmp_path):
    source = tmp_path / "fixture.onnx"
    source.write_bytes(model().SerializeToString())
    output = tmp_path / "inventory.json"
    script = Path(__file__).resolve().parents[2] / "tools/inspect_onnx_operator_signatures.py"
    command = [sys.executable, str(script), "--input", str(source), "--region", "standalone", "--output", str(output)]
    subprocess.run(command, capture_output=True, check=True)
    result = json.loads(output.read_text())
    assert result["source"]["bytes"] == source.stat().st_size
    saved = output.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == saved
