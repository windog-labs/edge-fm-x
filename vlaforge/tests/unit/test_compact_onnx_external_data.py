import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto as T, helper, numpy_helper

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("compact_onnx_external_data", TOOLS / "compact_onnx_external_data.py")
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def external(name, dtype, shape, location="weights.bin", offset=0, length=None):
    tensor = T(name=name, data_type=dtype, data_location=T.EXTERNAL)
    tensor.dims.extend(shape)
    tensor.external_data.add(key="location", value=location)
    if offset is not None:
        tensor.external_data.add(key="offset", value=str(offset))
    if length is not None:
        tensor.external_data.add(key="length", value=str(length))
    return tensor


def graph(tensors):
    nodes = [helper.make_node("Identity", [t.name], ["output_" + str(i)]) for i, t in enumerate(tensors)]
    outputs = [helper.make_tensor_value_info("output_" + str(i), t.data_type, list(t.dims)) for i, t in enumerate(tensors)]
    return helper.make_model(helper.make_graph(nodes, "parts", [], outputs, tensors),
                             opset_imports=[helper.make_opsetid("", 17)], ir_version=8)


def source(tmp_path, tensors=None):
    root = tmp_path / "source"
    root.mkdir()
    data = b"HEADER-UNUSED----" + np.arange(12, dtype=np.float32).tobytes() + b"UNUSED-TRAILER---"
    (root / "weights.bin").write_bytes(data)
    if tensors is None:
        tensors = [external("weight", T.FLOAT, [4], offset=16, length=16),
                   external("gamma", T.FLOAT, [4], offset=24, length=16),
                   external("bias", T.FLOAT, [4], offset=16, length=16),
                   external("beta", T.FLOAT, [2], offset=48, length=8)]
    model = graph(tensors)
    (root / "part.onnx").write_bytes(model.SerializeToString())
    return root, model


def assert_ranges(report, published):
    for entry in report["ranges"]:
        assert entry["source_sha256_before"] == entry["source_sha256_after"] == entry["output_sha256"]
        with (published / report["sidecar"]["location"]).open("rb") as stream:
            stream.seek(entry["output_offset"])
            assert hashlib.sha256(stream.read(entry["bytes"])).hexdigest() == entry["output_sha256"]


def test_overlap_and_exact_aliases_are_deduplicated_and_all_parameters_preserved(tmp_path):
    root, model = source(tmp_path)
    before = model.SerializeToString()
    weights_before = (root / "weights.bin").read_bytes()
    copied, report = TOOL.publish(model, root, tmp_path / "published", chunk_bytes=7)
    assert report["external_tensors"] == 4
    assert report["source_used_bytes"] == 56
    assert report["source_unique_range_bytes"] == 40
    assert report["source_union_bytes"] == report["copied_bytes"] == 32
    assert report["exact_duplicate_bytes"] == 16
    assert report["overlap_duplicate_bytes"] == 8
    assert report["total_duplicate_bytes"] == 24
    assert report["source_unreferenced_bytes"] == len(weights_before) - 32
    assert len(report["unions"]) == 2
    assert report["source_read_bytes"] == 96
    assert model.SerializeToString() == before
    assert (root / "weights.bin").read_bytes() == weights_before
    assert [t.name for t in copied.graph.initializer] == ["weight", "gamma", "bias", "beta"]
    offsets = [int(dict((e.key, e.value) for e in t.external_data)["offset"]) for t in copied.graph.initializer]
    assert offsets == [0, 8, 0, 24]
    assert (tmp_path / "published/model.onnx.data").read_bytes() == weights_before[16:40] + weights_before[48:56]
    assert_ranges(report, tmp_path / "published")
    for path in (tmp_path / "published").iterdir():
        assert not path.is_symlink() and path.stat().st_nlink == 1
    assert (root / "weights.bin").stat().st_ino != (tmp_path / "published/model.onnx.data").stat().st_ino


def test_tiny_ort_is_byte_exact_and_target_loads_after_source_unlink(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    root, model = source(tmp_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(str(root / "part.onnx"), options, providers=["CPUExecutionProvider"]).run(None, {})
    _, report = TOOL.publish(model, root, tmp_path / "standalone")
    for path in root.iterdir():
        path.unlink()
    root.rmdir()
    onnx.checker.check_model(str(tmp_path / "standalone/model.onnx"))
    loaded = onnx.load(tmp_path / "standalone/model.onnx", load_external_data=True)
    actual = ort.InferenceSession(str(tmp_path / "standalone/model.onnx"), options, providers=["CPUExecutionProvider"]).run(None, {})
    for a, b in zip(expected, actual, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert [numpy_helper.to_array(t).tobytes() for t in loaded.graph.initializer] == [v.tobytes() for v in expected]
    assert report["status"] == "published-byte-verified-not-runtime-validated"


def test_streams_only_referenced_source_unions_without_savers_or_array_decoding(tmp_path, monkeypatch):
    root, model = source(tmp_path)
    source_inode = (root / "weights.bin").stat().st_ino
    scans = []
    original = TOOL._scan
    def scan(stream, start, length, members, **kwargs):
        if os.fstat(stream.fileno()).st_ino == source_inode:
            scans.append((start, length))
        return original(stream, start, length, members, **kwargs)
    monkeypatch.setattr(TOOL, "_scan", scan)
    def forbidden(*args, **kwargs):
        raise AssertionError("tensor decoder/saver invoked")
    monkeypatch.setattr(onnx, "load", forbidden)
    monkeypatch.setattr(onnx, "save_model", forbidden)
    monkeypatch.setattr(numpy_helper, "to_array", forbidden)
    monkeypatch.setattr(os, "link", forbidden)
    TOOL.publish(model, root, tmp_path / "streamed", chunk_bytes=3)
    assert scans == [(16, 24), (48, 8)] * 3


def test_two_source_files_keep_independent_ranges(tmp_path):
    tensors = [external("one", T.FLOAT, [2], "weights.bin", 16, 8),
               external("two", T.FLOAT, [2], "second.bin", 4, 8)]
    root, model = source(tmp_path, tensors)
    (root / "second.bin").write_bytes(b"junk" + struct.pack("<ff", 91, 92) + b"unused")
    _, report = TOOL.publish(model, root, tmp_path / "two")
    assert report["source_union_bytes"] == 16 and len(report["source_files"]) == 2
    assert_ranges(report, tmp_path / "two")


@pytest.mark.parametrize("dtype,width", [(T.FLOAT, 4), (T.DOUBLE, 8), (T.FLOAT16, 2), (T.BFLOAT16, 2),
    (T.INT8, 1), (T.UINT8, 1), (T.INT16, 2), (T.UINT16, 2), (T.INT32, 4), (T.UINT32, 4),
    (T.INT64, 8), (T.UINT64, 8), (T.BOOL, 1), (T.COMPLEX64, 8), (T.COMPLEX128, 16),
    (T.FLOAT8E4M3FN, 1)])
def test_fixed_width_dtype_bytes_unchanged(tmp_path, dtype, width):
    root, model = source(tmp_path, [external("typed", dtype, [2], offset=1, length=2 * width)])
    _, report = TOOL.publish(model, root, tmp_path / "typed")
    assert (tmp_path / "typed/model.onnx.data").read_bytes() == (root / "weights.bin").read_bytes()[1:1 + 2 * width]
    assert report["tensors"][0]["shape"] == [2]


@pytest.mark.parametrize("dtype,bits", [(T.INT4, 4), (T.UINT4, 4), (T.FLOAT4E2M1, 4), (T.INT2, 2), (T.UINT2, 2)])
def test_fixed_subbyte_storage_uses_ceiling_extent_without_decoding(tmp_path, dtype, bits):
    size = (3 * bits + 7) // 8
    root, model = source(tmp_path, [external("packed", dtype, [3], offset=5, length=size)])
    _, report = TOOL.publish(model, root, tmp_path / "packed")
    assert report["source_union_bytes"] == size
    assert (tmp_path / "packed/model.onnx.data").read_bytes() == (root / "weights.bin").read_bytes()[5:5 + size]


@pytest.mark.parametrize("location", ["", "../outside", "/tmp/outside", "nested/../../outside", "C:\\outside", "nested\\outside"])
def test_path_escapes_are_rejected_before_output_creation(tmp_path, location):
    root, model = source(tmp_path, [external("bad", T.FLOAT, [2], location, 0, 8)])
    with pytest.raises(ValueError, match="inside"):
        TOOL.publish(model, root, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("key,value", [("offset", "-1"), ("length", "-1"), ("offset", "+1"), ("length", "8.0"),
    ("offset", " 1"), ("offset", str(1 << 63)), ("offset", "999999"), ("length", "999999"), ("length", "4")])
def test_invalid_offsets_lengths_and_dtype_extents_are_rejected(tmp_path, key, value):
    root, model = source(tmp_path, [external("bad", T.FLOAT, [2], offset=0, length=8)])
    next(e for e in model.graph.initializer[0].external_data if e.key == key).value = value
    with pytest.raises(ValueError):
        TOOL.publish(model, root, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_truncated_source_and_missing_source_are_rejected(tmp_path):
    root, model = source(tmp_path, [external("bad", T.FLOAT, [2], offset=0, length=8)])
    (root / "weights.bin").write_bytes(b"short")
    with pytest.raises(ValueError, match="truncated"):
        TOOL.publish(model, root, tmp_path / "bad")
    (root / "weights.bin").unlink()
    with pytest.raises(FileNotFoundError):
        TOOL.publish(model, root, tmp_path / "bad")


@pytest.mark.parametrize("dtype", [T.STRING, T.UNDEFINED])
def test_variable_or_unknown_dtype_rejected(tmp_path, dtype):
    root, model = source(tmp_path, [external("bad", dtype, [2], offset=0, length=8)])
    with pytest.raises(ValueError, match="fixed-width"):
        TOOL.publish(model, root, tmp_path / "bad")


@pytest.mark.parametrize("key", ["checksum", "basepath", "unknown", "location"])
def test_ambiguous_or_unsupported_metadata_rejected(tmp_path, key):
    root, model = source(tmp_path)
    model.graph.initializer[0].external_data.add(key=key, value="bad")
    with pytest.raises(ValueError, match="metadata key"):
        TOOL.publish(model, root, tmp_path / "bad")


def test_external_tensor_must_not_have_materialized_payload(tmp_path):
    root, model = source(tmp_path)
    model.graph.initializer[0].raw_data = b"a" * 16
    with pytest.raises(ValueError, match="inline payload"):
        TOOL.publish(model, root, tmp_path / "bad")


def test_inline_model_no_weights_and_optional_external_offset_length(tmp_path):
    root = tmp_path / "inline-source"
    root.mkdir()
    original = graph([numpy_helper.from_array(np.array([1., 2.], dtype=np.float32), "inline")])
    copied, report = TOOL.publish(original, root, tmp_path / "inline")
    assert copied.SerializeToString() == original.SerializeToString()
    assert report["source_union_bytes"] == 0 and (tmp_path / "inline/model.onnx.data").read_bytes() == b""
    (root / "weights.bin").write_bytes(struct.pack("<ff", 3, 4))
    model = graph([external("inferred", T.FLOAT, [2], offset=None)])
    _, report = TOOL.publish(model, root, tmp_path / "implicit")
    assert report["source_used_bytes"] == 8


def test_zero_sized_and_scalar_tensor_extents(tmp_path):
    root, model = source(tmp_path, [external("empty", T.FLOAT, [0], offset=0, length=0),
                                  external("scalar", T.FLOAT, [], offset=16, length=4)])
    copied, report = TOOL.publish(model, root, tmp_path / "zero")
    assert report["source_used_bytes"] == report["source_union_bytes"] == 4
    zero = dict((x.key, x.value) for x in copied.graph.initializer[0].external_data)
    assert zero["offset"] == "4" and zero["length"] == "0"
    loaded = onnx.load(tmp_path / "zero/model.onnx", load_external_data=True)
    assert numpy_helper.to_array(loaded.graph.initializer[0]).size == 0


def test_source_symlink_parent_symlink_and_hardlink_rejected(tmp_path):
    root, model = source(tmp_path)
    alias = tmp_path / "hardlink"
    os.link(root / "weights.bin", alias)
    with pytest.raises(ValueError, match="single-link"):
        TOOL.publish(model, root, tmp_path / "bad")
    alias.unlink()
    (root / "linked.bin").symlink_to(root / "weights.bin")
    changed = copy.deepcopy(model)
    next(x for x in changed.graph.initializer[0].external_data if x.key == "location").value = "linked.bin"
    with pytest.raises(ValueError, match="symlink"):
        TOOL.publish(changed, root, tmp_path / "bad")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        TOOL.publish(model, linked_root, tmp_path / "bad")


def test_existing_output_and_output_name_aliases_and_source_tree_are_rejected(tmp_path):
    root, model = source(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        TOOL.publish(model, root, output)
    assert (output / "keep").read_bytes() == b"keep"
    for options in ({"model_name": "same", "data_name": "same"}, {"data_name": "../outside"}, {"model_name": "compaction.json"}):
        with pytest.raises(ValueError):
            TOOL.publish(model, root, tmp_path / "bad", **options)
    with pytest.raises(ValueError, match="outside"):
        TOOL.publish(model, root, root / "never-write")


def test_nested_graph_tensor_and_sparse_storage_are_preserved(tmp_path):
    root, _ = source(tmp_path)
    nested = helper.make_graph([helper.make_node("Constant", [], ["nested"], value=external("gamma", T.FLOAT, [2], offset=16, length=8))],
                              "nested", [], [helper.make_tensor_value_info("nested", T.FLOAT, [2])])
    model = graph([])
    model.graph.node.append(helper.make_node("If", ["condition"], ["branch"], then_branch=nested, else_branch=nested))
    sparse = model.graph.sparse_initializer.add()
    sparse.values.CopyFrom(external("sparse", T.FLOAT, [2], offset=16, length=8))
    sparse.indices.CopyFrom(external("indices", T.INT64, [2], offset=32, length=16))
    sparse.dims.extend([4])
    copied, report = TOOL.publish(model, root, tmp_path / "nested")
    assert report["external_tensors"] == 4
    assert any("then_branch" in a.name for a in copied.graph.node[-1].attribute)
    assert copied.graph.sparse_initializer[0].values.data_type == T.FLOAT
    assert report["non_address_metadata_unchanged"]
    assert_ranges(report, tmp_path / "nested")


def test_source_mutation_and_failure_artifact_are_preserved(tmp_path, monkeypatch):
    root, model = source(tmp_path)
    original = TOOL._scan
    changed = False
    def mutate(stream, *args, **kwargs):
        nonlocal changed
        result = original(stream, *args, **kwargs)
        if kwargs.get("output") is not None and not changed:
            changed = True
            with (root / "weights.bin").open("r+b") as handle:
                handle.seek(16)
                handle.write(b"FAIL")
        return result
    monkeypatch.setattr(TOOL, "_scan", mutate)
    with pytest.raises(ValueError, match="changed"):
        TOOL.publish(model, root, tmp_path / "failed")
    assert not (tmp_path / "failed/model.onnx").exists()
    assert (tmp_path / "failed/model.onnx.data").exists()
    failure = json.loads((tmp_path / "failed/failure.json").read_text())
    assert failure["status"] == "failed" and not failure["valid_published_model"]


def test_source_metadata_mutation_is_rejected(tmp_path, monkeypatch):
    root, model = source(tmp_path)
    original = TOOL._scan
    def mutate(stream, *args, **kwargs):
        result = original(stream, *args, **kwargs)
        if kwargs.get("output") is not None:
            model.graph.doc_string = "changed by caller"
        return result
    monkeypatch.setattr(TOOL, "_scan", mutate)
    with pytest.raises(ValueError, match="metadata mutated"):
        TOOL.publish(model, root, tmp_path / "failed")
    assert not (tmp_path / "failed/model.onnx").exists()


def test_cli_uses_metadata_only_and_exclusive_outputs(tmp_path, monkeypatch):
    root, _ = source(tmp_path)
    monkeypatch.setattr(sys, "argv", ["compact", "--input", str(root / "part.onnx"), "--output-dir", str(tmp_path / "cli")])
    assert TOOL.main() == 0
    report = json.loads((tmp_path / "cli/compaction.json").read_text())
    assert report["copied_bytes"] == 32
    with pytest.raises(FileExistsError):
        TOOL.main()
