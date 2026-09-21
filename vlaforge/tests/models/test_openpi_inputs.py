"""Input serialization byte fixtures, not pretrained action-model evidence."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
from vlaforge.adapters.openpi.openpi_inputs import load_openpi_input_pack


SOURCE = Path(
    os.environ.get(
        "VLAFORGE_OPENPI_SOURCE_ROOT",
        Path(__file__).resolve().parents[3] / "third_party/openpi",
    )
)


def _pack(tmp_path, arrays=None):
    arrays = arrays or {
        "images/cam_high": np.zeros((3, 2, 2), dtype=np.uint8),
        "state": np.arange(14, dtype=np.float32),
        "prompt": np.asarray("test fixture"),
        "noise": np.ones((2, 3), dtype=np.float32),
    }
    np.savez(tmp_path / "input.npz", **arrays)
    manifest = {
        "schema": "vlaforge.openpi_input_pack/1",
        "input_npz": "input.npz",
        "input_digest": file_digest(tmp_path / "input.npz"),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for name, value in arrays.items()
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_pack_restores_actual_official_nested_observation_tree(tmp_path):
    pytest.importorskip("transformers", reason="actual pinned OpenPI dependency test")
    pytest.importorskip("jax", reason="actual pinned OpenPI dependency test")
    pytest.importorskip("flax", reason="actual pinned OpenPI dependency test")
    path, manifest = _pack(tmp_path)
    observation, noise, result = load_openpi_input_pack(path, source_root=SOURCE)
    assert observation["prompt"] == "test fixture"
    assert observation["images"]["cam_high"].shape == (3, 2, 2)
    np.testing.assert_array_equal(observation["state"], np.arange(14, dtype=np.float32))
    assert noise.dtype == np.float32
    assert "noise" not in observation
    assert result == manifest


@pytest.mark.parametrize("change", ["digest", "array_hash", "array_set", "path_escape"])
def test_pack_rejects_tampered_metadata_before_source_import(tmp_path, change):
    path, manifest = _pack(tmp_path)
    if change == "digest":
        manifest["input_digest"]["sha256"] = "0" * 64
    elif change == "array_hash":
        manifest["arrays"]["state"]["sha256"] = "0" * 64
    elif change == "array_set":
        del manifest["arrays"]["state"]
    else:
        outer = tmp_path.parent / "outside.npz"
        outer.write_bytes((tmp_path / "input.npz").read_bytes())
        manifest["input_npz"] = "../outside.npz"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_openpi_input_pack(path, source_root="/source-must-not-be-imported")


@pytest.mark.parametrize(
    "noise",
    [np.ones((2, 3), dtype=np.float64), np.full((2, 3), np.nan, dtype=np.float32)],
)
def test_pack_rejects_changed_noise_semantics(tmp_path, noise):
    path, _ = _pack(tmp_path, {"state": np.zeros(14), "noise": noise})
    with pytest.raises(ValueError, match="finite float32"):
        load_openpi_input_pack(path, source_root="/source-must-not-be-imported")


def test_pack_rejects_ambiguous_flattened_paths(tmp_path):
    path, _ = _pack(
        tmp_path,
        {
            "images": np.zeros(1),
            "images/cam_high": np.zeros(1),
            "noise": np.ones((2, 3), dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="ambiguous"):
        load_openpi_input_pack(path, source_root="/source-must-not-be-imported")


def test_pack_never_unpickles_object_arrays(tmp_path):
    path, _ = _pack(
        tmp_path,
        {
            "state": np.asarray([{"unsafe": "object"}], dtype=object),
            "noise": np.ones((2, 3), dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="Object arrays"):
        load_openpi_input_pack(path, source_root="/source-must-not-be-imported")
