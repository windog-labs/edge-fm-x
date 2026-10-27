import importlib.util
import hashlib
from pathlib import Path
import shutil

import pytest

onnx = pytest.importorskip("onnx")
np = pytest.importorskip("numpy")

SPEC = importlib.util.spec_from_file_location(
    "lower_onnx_split_sequences",
    Path(__file__).resolve().parents[2] / "tools/lower_onnx_split_sequences.py",
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def tensor(name, element_type, shape):
    return onnx.helper.make_tensor_value_info(name, element_type, shape)


def make_model(tmp_path):
    split_size = onnx.helper.make_tensor(
        "split_size", onnx.TensorProto.INT64, [], np.asarray(2, dtype=np.int64)
    )
    first = onnx.helper.make_tensor(
        "first", onnx.TensorProto.INT64, [], np.asarray(0, dtype=np.int64)
    )
    second = onnx.helper.make_tensor(
        "second", onnx.TensorProto.INT64, [], np.asarray(1, dtype=np.int64)
    )
    nodes = [
        onnx.helper.make_node(
            "SplitToSequence", ["value", "split_size"], ["parts"], axis=0
        ),
        onnx.helper.make_node("SequenceAt", ["parts", "first"], ["part0"]),
        onnx.helper.make_node("SequenceAt", ["parts", "second"], ["part1"]),
        onnx.helper.make_node("CastLike", ["part0", "other_float"], ["same"]),
        onnx.helper.make_node("CastLike", ["part1", "other_int"], ["cast"]),
        onnx.helper.make_node("Shape", ["part0"], ["shape"], start=0),
        onnx.helper.make_node(
            "ScatterND",
            ["data", "indices", "updates"],
            ["scattered"],
            reduction="none",
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "lowering_fixture",
        [
            tensor("value", onnx.TensorProto.FLOAT, [4]),
            tensor("other_float", onnx.TensorProto.FLOAT, [2]),
            tensor("other_int", onnx.TensorProto.INT64, [2]),
            tensor("data", onnx.TensorProto.FLOAT, [2, 2]),
            tensor("indices", onnx.TensorProto.INT64, [1, 1]),
            tensor("updates", onnx.TensorProto.FLOAT, [1, 2]),
        ],
        [
            tensor("same", onnx.TensorProto.FLOAT, [2]),
            tensor("cast", onnx.TensorProto.INT64, [2]),
            tensor("shape", onnx.TensorProto.INT64, [1]),
            tensor("scattered", onnx.TensorProto.FLOAT, [2, 2]),
        ],
        [split_size, first, second],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    source = tmp_path / "source.onnx"
    onnx.save_model(model, source)
    return source


def test_lowers_sequence_access_and_cast_like_generically(tmp_path):
    source = make_model(tmp_path)
    output = tmp_path / "lowered.onnx"
    report = TOOL.lower(source, output)
    model = onnx.load_model(output)

    assert report["lowered_split_to_sequence"] == 1
    assert report["lowered_slices"] == 2
    assert report["lowered_cast_like"] == 2
    assert report["cast_like_to_cast"] == 1
    assert report["cast_like_to_identity"] == 1
    assert report["lowered_shape"] == 1
    assert report["normalized_scatter_nd_reduction"] == 1
    assert not any(node.op_type in {"SplitToSequence", "SequenceAt", "CastLike", "Shape"}
                   for node in model.graph.node)
    assert sum(node.op_type == "Slice" for node in model.graph.node) == 2
    assert sum(node.op_type == "Identity" for node in model.graph.node) == 1
    cast = next(node for node in model.graph.node if node.op_type == "Cast")
    assert onnx.helper.get_attribute_value(cast.attribute[0]) == onnx.TensorProto.INT64
    shape_constant = next(
        node for node in model.graph.node
        if node.op_type == "Constant" and node.output[0] == "shape"
    )
    shape_value = onnx.numpy_helper.to_array(
        onnx.helper.get_attribute_value(shape_constant.attribute[0])
    )
    assert shape_value.tolist() == [2]
    scatter = next(node for node in model.graph.node if node.op_type == "ScatterND")
    assert not any(attribute.name == "reduction" for attribute in scatter.attribute)
    assert [value.name for value in model.graph.output] == [
        "same",
        "cast",
        "shape",
        "scattered",
    ]


def test_lowering_is_idempotent(tmp_path):
    source = make_model(tmp_path)
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    TOOL.lower(source, first)
    report = TOOL.lower(first, second)

    assert report["lowered_split_to_sequence"] == 0
    assert report["lowered_slices"] == 0
    assert report["lowered_cast_like"] == 0
    assert report["cast_like_to_cast"] == 0
    assert report["cast_like_to_identity"] == 0
    assert report["lowered_shape"] == 0
    assert report["normalized_scatter_nd_reduction"] == 0


def test_f64_island_demotion_is_explicit(tmp_path):
    one = onnx.helper.make_tensor(
        "one", onnx.TensorProto.DOUBLE, [1], np.asarray([1.0])
    )
    nodes = [
        onnx.helper.make_node(
            "Cast", ["value"], ["wide"], to=onnx.TensorProto.DOUBLE
        ),
        onnx.helper.make_node("Add", ["wide", "one"], ["sum"]),
        onnx.helper.make_node("Sin", ["sum"], ["sine"]),
        onnx.helper.make_node(
            "Cast", ["sine"], ["output"], to=onnx.TensorProto.FLOAT
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "f64_island",
        [tensor("value", onnx.TensorProto.FLOAT, [1])],
        [tensor("output", onnx.TensorProto.FLOAT, [1])],
        [one],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    source = tmp_path / "f64.onnx"
    onnx.save_model(model, source)

    preserved = tmp_path / "preserved.onnx"
    preserved_report = TOOL.lower(source, preserved)
    assert preserved_report["f64_demoted_to_f32"] is False
    assert preserved_report["f64_values_remaining"] > 0

    demoted = tmp_path / "demoted.onnx"
    demoted_report = TOOL.lower(
        source, demoted, demote_f64_island=True
    )
    lowered_model = onnx.load_model(demoted)
    assert demoted_report["f64_demoted_to_f32"] is True
    assert demoted_report["f64_values_remaining"] == 0
    assert demoted_report["requires_numerical_verification"] is True
    assert not any(
        value.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE
        for value in (
            *lowered_model.graph.input,
            *lowered_model.graph.output,
            *lowered_model.graph.value_info,
        )
    )
    assert all(
        initializer.data_type != onnx.TensorProto.DOUBLE
        for initializer in lowered_model.graph.initializer
    )


def test_bfloat16_promotion_is_explicit(tmp_path):
    one = onnx.helper.make_tensor(
        "one", onnx.TensorProto.BFLOAT16, [1], [16256]
    )
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["value"], ["output"])],
        "bfloat16_abi",
        [tensor("value", onnx.TensorProto.BFLOAT16, [1])],
        [tensor("output", onnx.TensorProto.BFLOAT16, [1])],
        [one],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 18)],
        ir_version=9,
    )
    source = tmp_path / "bfloat16.onnx"
    onnx.save_model(model, source)

    preserved = tmp_path / "preserved-bfloat16.onnx"
    preserved_report = TOOL.lower(source, preserved)
    assert preserved_report["bfloat16_promoted_to_f32"] is False
    assert preserved_report["bfloat16_values_remaining"] > 0

    promoted = tmp_path / "promoted-f32.onnx"
    promoted_report = TOOL.lower(
        source, promoted, promote_bfloat16_to_f32=True
    )
    lowered_model = onnx.load_model(promoted)
    assert promoted_report["bfloat16_promoted_to_f32"] is True
    assert promoted_report["bfloat16_values_remaining"] == 0
    assert promoted_report["requires_numerical_verification"] is True
    assert not any(
        value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
        for value in (
            *lowered_model.graph.input,
            *lowered_model.graph.output,
            *lowered_model.graph.value_info,
        )
    )


def make_external_model(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source = make_model(root)
    model = onnx.load(source)
    model.graph.initializer.append(onnx.numpy_helper.from_array(
        np.array([0.5, -0.25, 1.25, -0.5], dtype=np.float32), "weight"))
    model.graph.node.insert(0, onnx.helper.make_node("Add", ["value", "weight"], ["weighted"]))
    model.graph.node[1].input[0] = "weighted"
    model.graph.input.append(tensor("unused_state", onnx.TensorProto.DOUBLE, [1]))
    relative = Path("tensors/weights.bin")
    weight_file = root / relative
    weight_file.parent.mkdir()
    with weight_file.open("xb") as stream:
        stream.write(b"preserved padding")
        for value in model.graph.initializer:
            replacement = onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(value), value.name)
            raw = replacement.raw_data
            onnx.external_data_helper.set_external_data(
                replacement, str(relative), offset=stream.tell(), length=len(raw))
            stream.write(raw)
            replacement.ClearField("raw_data")
            value.CopyFrom(replacement)
        stream.write(b"preserved unused trailing bytes")
    onnx.save_model(model, source)
    return source, relative


def fixture_feeds():
    return {"value": np.array([1.5, 2.5, -3.5, 4.5], dtype=np.float32),
            "other_float": np.zeros(2, dtype=np.float32),
            "other_int": np.zeros(2, dtype=np.int64),
            "data": np.zeros((2, 2), dtype=np.float32),
            "indices": np.array([[0]], dtype=np.int64),
            "updates": np.array([[3, 4]], dtype=np.float32),
            "unused_state": np.zeros(1, dtype=np.float64)}


def test_external_offsets_equivalence_and_exact_preservation(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    source, relative = make_external_model(tmp_path)
    original = onnx.load(source, load_external_data=False)
    before = {path: path.read_bytes() for path in (source, source.parent / relative)}
    output = tmp_path / "lowered/model.onnx"
    report = TOOL.lower(source, output)
    lowered = onnx.load(output, load_external_data=False)
    assert [value.SerializeToString() for value in lowered.graph.initializer] == [
        value.SerializeToString() for value in original.graph.initializer]
    assert report["typed_ports_preserved"]
    assert report["external_scalar_control_bytes_read"] == 24
    assert report["staged_external_scalar_controls_verified"] == 3
    assert report["weights_materialized"] is False
    assert report["external_data"][0]["source_sha256"] == hashlib.sha256(before[source.parent / relative]).hexdigest()
    assert (output.parent / relative).read_bytes() == before[source.parent / relative]
    assert (output.parent / relative).stat().st_ino != (source.parent / relative).stat().st_ino
    for path, contents in before.items():
        assert path.read_bytes() == contents
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    # ORT cannot infer SplitToSequence with external scalar controls; inline
    # this tiny reference fixture only, independently of the lowering path.
    reference = onnx.load(source, load_external_data=True).SerializeToString()
    expected = ort.InferenceSession(reference, options, providers=["CPUExecutionProvider"]).run(None, fixture_feeds())
    shutil.rmtree(source.parent)
    actual = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"]).run(None, fixture_feeds())
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)
    assert report["full_model_output_verified"] is False


def test_external_weights_are_never_loaded_or_converted(tmp_path, monkeypatch):
    source, _ = make_external_model(tmp_path)
    load = onnx.load
    checker = onnx.checker.check_model
    checked = []

    def metadata_only(path, **kwargs):
        assert kwargs.get("load_external_data") is False
        return load(path, **kwargs)

    def no_payload(*args, **kwargs):
        pytest.fail("external payload must not be loaded into protobuf or NumPy")

    def filename_checker(path, *args, **kwargs):
        assert isinstance(path, str)
        checked.append(path)
        return checker(path, *args, **kwargs)

    monkeypatch.setattr(onnx, "load", metadata_only)
    monkeypatch.setattr(onnx.numpy_helper, "to_array", no_payload)
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor", no_payload)
    monkeypatch.setattr(onnx.checker, "check_model", filename_checker)
    report = TOOL.lower(source, tmp_path / "output/model.onnx")
    assert len(checked) == 1
    assert report["lowered_slices"] == 2


def test_large_external_tensor_does_not_enter_protobuf(tmp_path, monkeypatch):
    source, relative = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    huge = onnx.TensorProto(name="huge_unused_weight", data_type=onnx.TensorProto.FLOAT,
                            dims=[(3 << 30) // 4], data_location=onnx.TensorProto.EXTERNAL)
    for key, value in {"location": "huge.bin", "offset": "0", "length": str(3 << 30)}.items():
        huge.external_data.add(key=key, value=value)
    model.graph.initializer.append(huge)
    with (source.parent / "huge.bin").open("xb") as stream:
        stream.truncate(3 << 30)
    onnx.save_model(model, source)
    copy = TOOL.copy_external

    def sparse_copy(src, dst, location):
        if location == Path("huge.bin"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("xb") as stream:
                stream.truncate(src.stat().st_size)
            return {"location": str(location), "size_bytes": src.stat().st_size}
        return copy(src, dst, location)

    def forbidden(*args, **kwargs):
        pytest.fail("large weight payload must remain external")

    monkeypatch.setattr(TOOL, "copy_external", sparse_copy)
    monkeypatch.setattr(onnx.numpy_helper, "to_array", forbidden)
    monkeypatch.setattr(onnx.external_data_helper, "load_external_data_for_tensor", forbidden)
    output = tmp_path / "large/model.onnx"
    report = TOOL.lower(source, output)
    assert report["external_tensor_count"] == 5
    assert output.stat().st_size < 20000
    assert (output.parent / "huge.bin").stat().st_size == 3 << 30
    assert report["external_scalar_control_bytes_read"] == 24


@pytest.mark.parametrize("flag", ["demote_f64_island", "promote_bfloat16_to_f32",
                                  "demote_spurious_complex_casts", "normalize_cumsum_bool"])
def test_external_precision_flags_reject_before_publication(tmp_path, flag):
    source, _ = make_external_model(tmp_path)
    with pytest.raises(ValueError, match="dtype-changing flags"):
        TOOL.lower(source, tmp_path / "output/model.onnx", **{flag: True})
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("location", ["../escape.bin", "/tmp/escape.bin", ""])
def test_external_unsafe_paths_are_rejected(tmp_path, location):
    source, _ = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    next(entry for entry in model.graph.initializer[0].external_data if entry.key == "location").value = location
    onnx.save_model(model, source)
    with pytest.raises(ValueError, match="inside"):
        TOOL.lower(source, tmp_path / "output/model.onnx")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("key,value", [("offset", "-1"), ("length", "-1"),
                                      ("offset", "9999999999"), ("length", "9999999999"),
                                      ("offset", "no"), ("length", "9")])
def test_invalid_external_control_ranges_are_rejected(tmp_path, key, value):
    source, _ = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    next(entry for entry in model.graph.initializer[0].external_data if entry.key == key).value = value
    onnx.save_model(model, source)
    with pytest.raises(ValueError, match="external"):
        TOOL.lower(source, tmp_path / "output/model.onnx")
    assert not (tmp_path / "output").exists()


def test_duplicate_external_metadata_is_rejected(tmp_path):
    source, _ = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    model.graph.initializer[0].external_data.add(key="offset", value="0")
    onnx.save_model(model, source)
    with pytest.raises(ValueError, match="duplicate"):
        TOOL.lower(source, tmp_path / "output/model.onnx")


@pytest.mark.parametrize("kind", ["graph", "sidecar", "graph-symlink", "directory-symlink"])
def test_external_publication_collisions_preserve_existing_files(tmp_path, kind):
    source, relative = make_external_model(tmp_path)
    output = tmp_path / "output/model.onnx"
    output.parent.mkdir()
    if kind == "directory-symlink":
        (output.parent / relative.parent).symlink_to(source.parent / relative.parent, target_is_directory=True)
        with pytest.raises((ValueError, FileExistsError)):
            TOOL.lower(source, output)
        return
    occupied = output if kind in ("graph", "graph-symlink") else output.parent / relative
    occupied.parent.mkdir(parents=True, exist_ok=True)
    if kind == "graph-symlink":
        occupied.symlink_to(tmp_path / "missing")
    else:
        occupied.write_bytes(b"preserve existing")
    with pytest.raises(FileExistsError):
        TOOL.lower(source, output)
    if kind == "graph-symlink":
        assert occupied.is_symlink()
    else:
        assert occupied.read_bytes() == b"preserve existing"


def test_source_external_symlink_is_rejected(tmp_path):
    source, relative = make_external_model(tmp_path)
    weight = source.parent / relative
    outside = tmp_path / "outside.bin"
    weight.rename(outside)
    weight.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes|symlinks"):
        TOOL.lower(source, tmp_path / "output/model.onnx")


def test_external_sidecar_cannot_alias_output_graph(tmp_path):
    source, relative = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    for value in model.graph.initializer:
        next(entry for entry in value.external_data if entry.key == "location").value = "model.onnx"
    (source.parent / relative).rename(source.parent / "model.onnx")
    onnx.save_model(model, source)
    with pytest.raises(ValueError, match="aliases"):
        TOOL.lower(source, tmp_path / "output/model.onnx")


def test_nested_external_constant_attributes_are_preserved(tmp_path):
    source, relative = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    weight = model.graph.initializer[-1]
    branch = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["branch_value"], value=weight)],
        "branch", [], [tensor("branch_value", onnx.TensorProto.FLOAT, [4])])
    model.graph.input.append(tensor("condition", onnx.TensorProto.BOOL, []))
    model.graph.node.append(onnx.helper.make_node("If", ["condition"], ["choice"],
                                                 then_branch=branch, else_branch=branch))
    model.graph.output.append(tensor("choice", onnx.TensorProto.FLOAT, [4]))
    onnx.save_model(model, source)
    descriptors = [value.SerializeToString() for value in TOOL.tensor_messages(model)]
    output = tmp_path / "nested/model.onnx"
    report = TOOL.lower(source, output)
    lowered = onnx.load(output, load_external_data=False)
    assert report["external_tensor_count"] == 6
    assert [value.SerializeToString() for value in TOOL.tensor_messages(lowered)
            if value.data_location == onnx.TensorProto.EXTERNAL] == descriptors
    assert (output.parent / relative).read_bytes() == (source.parent / relative).read_bytes()


def test_generated_value_names_avoid_existing_names(tmp_path):
    source = make_model(tmp_path)
    model = onnx.load(source)
    occupied = "vlaforge_slice_starts_0_0_1"
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([99], dtype=np.int64), occupied))
    onnx.save_model(model, source)
    output = tmp_path / "lowered.onnx"
    TOOL.lower(source, output)
    lowered = onnx.load(output)
    assert not any(occupied in node.output for node in lowered.graph.node)


@pytest.mark.parametrize("invalid", ["vector-split", "out-of-range", "exposed-sequence"])
def test_unsafe_sequence_rewrites_reject_before_publication(tmp_path, invalid):
    source = make_model(tmp_path)
    model = onnx.load(source)
    if invalid == "vector-split":
        model.graph.initializer[0].dims.append(1)
    elif invalid == "out-of-range":
        model.graph.initializer[1].int64_data[0] = 7
    else:
        model.graph.output.append(onnx.helper.make_tensor_sequence_value_info("parts", onnx.TensorProto.FLOAT, [2]))
    onnx.save_model(model, source)
    with pytest.raises(ValueError):
        TOOL.lower(source, tmp_path / "lowered.onnx")
    assert not (tmp_path / "lowered.onnx").exists()


def test_failed_filename_check_leaves_no_published_files(tmp_path, monkeypatch):
    source, _ = make_external_model(tmp_path)

    def reject(path):
        assert isinstance(path, str)
        raise ValueError("checker rejected fixture")

    monkeypatch.setattr(onnx.checker, "check_model", reject)
    with pytest.raises(ValueError, match="checker rejected"):
        TOOL.lower(source, tmp_path / "output/model.onnx")
    assert list((tmp_path / "output").iterdir()) == []


@pytest.mark.parametrize("has_default", [False, True])
def test_runtime_sequence_controls_are_never_frozen(tmp_path, has_default):
    source, _ = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    model.graph.input.append(tensor("first", onnx.TensorProto.INT64, []))
    if not has_default:
        del model.graph.initializer[1]
    onnx.save_model(model, source)
    with pytest.raises(ValueError, match="runtime graph input"):
        TOOL.lower(source, tmp_path / "output/model.onnx")
    assert not (tmp_path / "output").exists()


def test_external_constant_control_descriptors_are_restored_after_inference(tmp_path):
    source, _ = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    control = model.graph.initializer[0]
    model.graph.node.insert(0, onnx.helper.make_node("Constant", [], [control.name], value=control))
    del model.graph.initializer[0]
    onnx.save_model(model, source)
    before = sorted(value.SerializeToString() for value in TOOL.tensor_messages(model))
    output = tmp_path / "output/model.onnx"
    report = TOOL.lower(source, output)
    lowered = onnx.load(output, load_external_data=False)
    assert sorted(value.SerializeToString() for value in TOOL.tensor_messages(lowered)
                  if value.data_location == onnx.TensorProto.EXTERNAL) == before
    assert report["external_scalar_control_bytes_read"] == 24


def test_nested_external_attribute_cannot_escape_source_root(tmp_path):
    source, _ = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    weight = onnx.TensorProto()
    weight.CopyFrom(model.graph.initializer[-1])
    next(entry for entry in weight.external_data if entry.key == "location").value = "../outside.bin"
    branch = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["nested"], value=weight)],
        "branch", [], [tensor("nested", onnx.TensorProto.FLOAT, [4])])
    model.graph.node.append(onnx.helper.make_node("If", ["condition"], ["choice"],
                                                 then_branch=branch, else_branch=branch))
    onnx.save_model(model, source)
    with pytest.raises(ValueError, match="inside"):
        TOOL.lower(source, tmp_path / "output/model.onnx")


def test_racing_graph_publication_rolls_back_only_our_sidecars(tmp_path, monkeypatch):
    source, relative = make_external_model(tmp_path)
    output = tmp_path / "output/model.onnx"
    external_files, publish_files = TOOL.external_helpers()

    def race(staged, destination, sidecars):
        destination.write_bytes(b"someone else's graph")
        return publish_files(staged, destination, sidecars)

    monkeypatch.setattr(TOOL, "external_helpers", lambda: (external_files, race))
    with pytest.raises(FileExistsError):
        TOOL.lower(source, output)
    assert output.read_bytes() == b"someone else's graph"
    assert not (output.parent / relative).exists()


def test_nested_sequence_capture_cannot_be_removed(tmp_path):
    source = make_model(tmp_path)
    model = onnx.load(source)
    branch = onnx.helper.make_graph(
        [onnx.helper.make_node("SequenceLength", ["parts"], ["length"])],
        "branch", [], [tensor("length", onnx.TensorProto.INT64, [])])
    model.graph.input.append(tensor("condition", onnx.TensorProto.BOOL, []))
    model.graph.node.append(onnx.helper.make_node("If", ["condition"], ["choice"],
                                                 then_branch=branch, else_branch=branch))
    model.graph.output.append(tensor("choice", onnx.TensorProto.INT64, []))
    onnx.save_model(model, source)
    with pytest.raises(ValueError, match="nested sequence captures"):
        TOOL.lower(source, tmp_path / "lowered.onnx")


def test_unknown_rank_is_not_a_scalar_shape():
    assert TOOL.tensor_shape(tensor("unknown", onnx.TensorProto.FLOAT, None)) is None
    assert TOOL.tensor_shape(tensor("scalar", onnx.TensorProto.FLOAT, [])) == ()


def test_cli_report_collision_is_detected_before_lowering(tmp_path, monkeypatch):
    source, _ = make_external_model(tmp_path)
    report = tmp_path / "receipt.json"
    report.write_bytes(b"existing report")
    output = tmp_path / "output/model.onnx"
    monkeypatch.setattr("sys.argv", ["lower", "--input", str(source), "--output", str(output),
                                     "--report", str(report)])
    with pytest.raises(FileExistsError):
        TOOL.main()
    assert report.read_bytes() == b"existing report"
    assert not output.parent.exists()


@pytest.mark.parametrize("moment", ["before-load", "after-load"])
def test_source_graph_snapshot_is_pinned_across_load(tmp_path, monkeypatch, moment):
    source, relative = make_external_model(tmp_path)
    load = onnx.load
    replacement = load(source, load_external_data=False)
    replacement.graph.name += "_changed"
    replacement_bytes = replacement.SerializeToString()
    original_hash = TOOL.digest(source)
    weight_hash = TOOL.digest(source.parent / relative)
    observed_hashes = []
    digest = TOOL.digest

    def observed_digest(path):
        value = digest(path)
        if path == source:
            observed_hashes.append(value)
        return value

    def mutating_load(path, **kwargs):
        assert observed_hashes == [original_hash]
        if moment == "before-load":
            source.write_bytes(replacement_bytes)
        model = load(path, **kwargs)
        if moment == "after-load":
            source.write_bytes(replacement_bytes)
        return model

    monkeypatch.setattr(TOOL, "digest", observed_digest)
    monkeypatch.setattr(onnx, "load", mutating_load)
    output = tmp_path / "output/model.onnx"
    with pytest.raises(ValueError, match="source ONNX graph changed while loading"):
        TOOL.lower(source, output)
    assert observed_hashes == [original_hash, hashlib.sha256(replacement_bytes).hexdigest()]
    assert not output.parent.exists()
    assert digest(source.parent / relative) == weight_hash


def test_source_graph_snapshot_is_rechecked_before_publication(tmp_path, monkeypatch):
    source, relative = make_external_model(tmp_path)
    replacement = onnx.load(source, load_external_data=False)
    replacement.graph.name += "_changed_after_load"
    replacement_bytes = replacement.SerializeToString()
    weight_hash = TOOL.digest(source.parent / relative)
    checker = onnx.checker.check_model
    checks = []

    def mutate_after_check(path):
        checker(path)
        checks.append(path)
        source.write_bytes(replacement_bytes)

    monkeypatch.setattr(onnx.checker, "check_model", mutate_after_check)
    output = tmp_path / "output/model.onnx"
    with pytest.raises(ValueError, match="source ONNX graph changed during lowering"):
        TOOL.lower(source, output)
    assert len(checks) == 1
    assert list(output.parent.iterdir()) == []
    assert TOOL.digest(source.parent / relative) == weight_hash


@pytest.mark.parametrize("as_constant", [False, True])
@pytest.mark.parametrize("control_name", ["split_size", "first", "second"])
def test_resolved_external_controls_match_staged_copy(tmp_path, monkeypatch, as_constant, control_name):
    source, relative = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    position = next(index for index, value in enumerate(model.graph.initializer) if value.name == control_name)
    control = model.graph.initializer[position]
    metadata = {entry.key: entry.value for entry in control.external_data}
    offset, length = int(metadata["offset"]), int(metadata["length"])
    if as_constant:
        model.graph.node.insert(0, onnx.helper.make_node("Constant", [], [control_name], value=control))
        del model.graph.initializer[position]
        onnx.save_model(model, source)
    source_hash = TOOL.digest(source)
    weight_path = source.parent / relative
    with weight_path.open("rb") as stream:
        stream.seek(offset)
        original = stream.read(length)
    replacement = (0 if int.from_bytes(original, "little", signed=True) == 1 else 1).to_bytes(
        length, "little", signed=True)
    copy = TOOL.copy_external
    copies = []

    def mutate_before_copy(src, dst, location):
        assert location == relative
        with src.open("r+b") as stream:
            stream.seek(offset)
            stream.write(replacement)
        # copy_external sees a stable B snapshot and succeeds; the scalar
        # resolver has already used A for the structural specialization.
        receipt = copy(src, dst, location)
        assert receipt["source_sha256"] == receipt["output_sha256"]
        copies.append(receipt)
        return receipt

    def forbidden_check(*args, **kwargs):
        pytest.fail("mismatched staged controls must be rejected before the checker")

    monkeypatch.setattr(TOOL, "copy_external", mutate_before_copy)
    monkeypatch.setattr(onnx.checker, "check_model", forbidden_check)
    output = tmp_path / "output/model.onnx"
    with pytest.raises(ValueError, match=f"{control_name}: external scalar control changed"):
        TOOL.lower(source, output)
    assert len(copies) == 1
    assert list(output.parent.iterdir()) == []
    assert TOOL.digest(source) == source_hash


def test_external_control_validation_reads_staged_sidecar(tmp_path, monkeypatch):
    source, relative = make_external_model(tmp_path)
    model = onnx.load(source, load_external_data=False)
    control = model.graph.initializer[0]
    offset = int(next(entry.value for entry in control.external_data if entry.key == "offset"))
    source_hash = TOOL.digest(source)
    weight_hash = TOOL.digest(source.parent / relative)
    copy = TOOL.copy_external

    def mutate_staged_copy(src, dst, location):
        receipt = copy(src, dst, location)
        with dst.open("r+b") as stream:
            stream.seek(offset)
            stream.write((1).to_bytes(8, "little", signed=True))
        return receipt

    def forbidden_check(*args, **kwargs):
        pytest.fail("staged controls must be validated before the checker")

    monkeypatch.setattr(TOOL, "copy_external", mutate_staged_copy)
    monkeypatch.setattr(onnx.checker, "check_model", forbidden_check)
    output = tmp_path / "output/model.onnx"
    with pytest.raises(ValueError, match="split_size: external scalar control changed"):
        TOOL.lower(source, output)
    assert list(output.parent.iterdir()) == []
    assert TOOL.digest(source) == source_hash
    assert TOOL.digest(source.parent / relative) == weight_hash
