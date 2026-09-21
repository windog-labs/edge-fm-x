import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "smolvla_input_pack_tool",
    Path(__file__).resolve().parents[2] / "tools/prepare_real_smolvla_inputs.py",
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.mark.parametrize(
    "entries", [[], ["x"], ["x="], ["=y"], ["x=y", "x=z"], ["x=z", "y=z"]]
)
def test_camera_mapping_requires_explicit_one_to_one_sources(entries):
    with pytest.raises(ValueError):
        tool.camera_mapping(entries)


def test_camera_mapping_retains_real_source_order():
    assert tool.camera_mapping(["top=camera1", "wrist=camera2"]) == {
        "top": "camera1",
        "wrist": "camera2",
    }


@pytest.mark.parametrize("lfs", [True, False])
def test_dataset_revision_size_and_object_identity_are_verified(tmp_path, lfs):
    root = tmp_path / "dataset"
    root.mkdir()
    path = root / "sample.bin"
    path.write_bytes(b"sample")
    item = {"path": "sample.bin", "type": "file", "size": 6}
    if lfs:
        item["lfs"] = {"oid": hashlib.sha256(b"sample").hexdigest()}
    else:
        item["oid"] = hashlib.sha1(b"blob 6\0sample").hexdigest()
    info, tree = tmp_path / "info.json", tmp_path / "tree.json"
    info.write_text(
        json.dumps({"sha": "revision", "siblings": [{"rfilename": "sample.bin"}]})
    )
    tree.write_text(json.dumps([item]))
    assert tool.verify_dataset_source(root, info, tree, "revision") == {
        "sample.bin": hashlib.sha256(b"sample").hexdigest()
    }
    with pytest.raises(ValueError, match="revision"):
        tool.verify_dataset_source(root, info, tree, "wrong")
    path.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="digest"):
        tool.verify_dataset_source(root, info, tree, "revision")
    tree.write_text("[]")
    with pytest.raises(ValueError, match="incomplete"):
        tool.verify_dataset_source(root, info, tree, "revision")


def test_published_compatibility_never_claims_statistics_acceptance(tmp_path):
    profile, record = tool.prepare_statistics(
        tmp_path, mode="published-compatibility", namespace=None, robot_type="so100"
    )
    assert profile is None
    assert record["normalization_statistics_verified"] is False
    assert record["physical_action_units_verified"] is False
    assert record["formal_preprocessing_acceptance"] is False
    with pytest.raises(ValueError, match="cannot select"):
        tool.prepare_statistics(
            tmp_path,
            mode="published-compatibility",
            namespace="so100",
            robot_type="so100",
        )


def test_strict_preprocessing_rejects_missing_state_in_legacy_checkpoint(tmp_path):
    import torch
    save_file = pytest.importorskip("safetensors.torch").save_file

    configuration = {
        "steps": [
            {
                "registry_name": "normalizer_processor",
                "state_file": "statistics.safetensors",
                "config": {
                    "features": {
                        "observation.state": {"type": "STATE", "shape": [6]},
                        "action": {"type": "ACTION", "shape": [6]},
                    },
                    "norm_map": {"STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
                },
            }
        ]
    }
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(configuration))
    statistics = {
        "so100.buffer.action.mean": torch.zeros(6),
        "so100.buffer.action.std": torch.ones(6),
    }
    path = tmp_path / "statistics.safetensors"
    save_file(statistics, path)
    with pytest.raises(
        ValueError, match="missing required feature statistics.*observation.state"
    ):
        tool.prepare_statistics(
            tmp_path, mode="strict-statistics", namespace="so100", robot_type="so100"
        )
    statistics.update(
        {
            "so100.buffer.observation.state.mean": torch.ones(6),
            "so100.buffer.observation.state.std": torch.full((6,), 2.0),
        }
    )
    save_file(statistics, path)
    profile, record = tool.prepare_statistics(
        tmp_path, mode="strict-statistics", namespace="so100", robot_type="so100"
    )
    assert set(profile.stats) == {"observation.state", "action"}
    assert record["normalization_statistics_verified"] is False
    assert record["selection"]["source_sha256"] == tool.digest(path)
    with pytest.raises(ValueError, match="does not match.*robot_type"):
        tool.prepare_statistics(
            tmp_path,
            mode="strict-statistics",
            namespace="so100",
            robot_type="different-robot",
        )
