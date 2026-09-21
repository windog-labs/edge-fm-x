import hashlib
import json

import numpy as np
import pytest

from vlaforge.adapters.smolvla.smolvla_host_pipeline import load_raw_records
from vlaforge.validation.host_pipeline import digest


def pack(tmp_path, arrays=None):
    root = tmp_path / "recordings"
    root.mkdir()
    arrays = arrays if arrays is not None else {
        "observation.images.top": np.arange(18, dtype=np.uint8).reshape(3, 2, 3),
        "observation.state": np.arange(6, dtype=np.float32),
        "noise": np.zeros((1, 2, 6), dtype=np.float32),
    }
    data = root / "0.npz"
    np.savez(data, **arrays)
    manifest = {"schema": "edgefm.recorded_lerobot_inputs/1", "records": [{
        "index": 0, "path": "0.npz", "sha256": digest(data),
        "text": {"task": "Move the object"},
        "arrays": {name: {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
            for name, value in arrays.items()},
    }]}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, arrays


def rewrite(path, manifest):
    path.write_text(json.dumps(manifest))
    return digest(path)


def test_recorded_pixels_state_noise_and_text_survive_without_casting(tmp_path):
    path, manifest, expected = pack(tmp_path)
    actual_manifest, samples = load_raw_records(path, digest(path))
    assert actual_manifest == manifest
    arrays, text = samples[0]
    assert text == {"task": "Move the object"}
    for name, value in expected.items():
        assert arrays[name].dtype == value.dtype
        assert arrays[name].shape == value.shape
        assert arrays[name].tobytes() == value.tobytes()


def test_changed_manifest_is_rejected_before_loading(tmp_path):
    path, manifest, _ = pack(tmp_path)
    original = digest(path)
    manifest["records"][0]["text"]["task"] = "A different instruction"
    rewrite(path, manifest)
    with pytest.raises(ValueError, match="manifest identity"):
        load_raw_records(path, original)


def test_replaced_npz_is_rejected(tmp_path):
    path, _, arrays = pack(tmp_path)
    arrays["noise"].fill(1)
    np.savez(path.parent / "0.npz", **arrays)
    with pytest.raises(ValueError, match="file digest"):
        load_raw_records(path, digest(path))


@pytest.mark.parametrize("escape", ["relative", "symlink"])
def test_npz_cannot_escape_recording_directory_even_with_valid_digest(tmp_path, escape):
    path, manifest, arrays = pack(tmp_path)
    outside = tmp_path / "outside.npz"
    np.savez(outside, **arrays)
    if escape == "relative":
        manifest["records"][0]["path"] = "../outside.npz"
    else:
        link = path.parent / "link.npz"
        link.symlink_to(outside)
        manifest["records"][0]["path"] = link.name
    manifest["records"][0]["sha256"] = digest(outside)
    with pytest.raises(ValueError, match="path, order"):
        load_raw_records(path, rewrite(path, manifest))


@pytest.mark.parametrize("field,value", [("shape", [3, 3, 2]), ("dtype", "float32"), ("sha256", "0" * 64)])
def test_array_metadata_is_checked_beyond_npz_digest(tmp_path, field, value):
    path, manifest, _ = pack(tmp_path)
    manifest["records"][0]["arrays"]["observation.images.top"][field] = value
    with pytest.raises(ValueError, match="array identity"):
        load_raw_records(path, rewrite(path, manifest))


def test_undeclared_array_is_rejected(tmp_path):
    path, manifest, _ = pack(tmp_path)
    del manifest["records"][0]["arrays"]["noise"]
    with pytest.raises(ValueError, match="array set"):
        load_raw_records(path, rewrite(path, manifest))


def test_record_order_is_not_silently_changed(tmp_path):
    path, manifest, _ = pack(tmp_path)
    manifest["records"][0]["index"] = 1
    with pytest.raises(ValueError, match="order"):
        load_raw_records(path, rewrite(path, manifest))


@pytest.mark.parametrize("text", [{"task": [1, 2]}, "task"])
def test_text_is_not_coerced_from_another_type(tmp_path, text):
    path, manifest, _ = pack(tmp_path)
    manifest["records"][0]["text"] = text
    with pytest.raises(ValueError, match="explicit strings"):
        load_raw_records(path, rewrite(path, manifest))


@pytest.mark.parametrize("array", [np.array([float("nan")]), np.array([float("inf")]), np.array([1 + 2j]), np.array(["text"])])
def test_hash_bound_nonfinite_or_nonnumeric_arrays_are_rejected(tmp_path, array):
    path, _, _ = pack(tmp_path, {"noise": array})
    with pytest.raises(ValueError, match="array identity"):
        load_raw_records(path, digest(path))
