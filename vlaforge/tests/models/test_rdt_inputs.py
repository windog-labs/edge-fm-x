import io
import json
import tarfile

import pytest

from vlaforge.adapters.rdt.rdt_assets import file_identity
from vlaforge.adapters.rdt.rdt_inputs import extract_verified_prefix_members


def make_archive(tmp_path, entries):
    path = tmp_path / "prefix.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, kind, payload in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    identity = file_identity(path)
    return path, {"size": identity["size"], "lfs": {"sha256": identity["sha256"]}}


def test_selected_members_do_not_claim_complete_dataset(tmp_path):
    path, record = make_archive(tmp_path, [("task/a.hdf5", tarfile.REGTYPE, b"recorded"),
                                           ("task/instruction.json", tarfile.REGTYPE, b"{}")])
    report = extract_verified_prefix_members(path, record, ("task/a.hdf5", "task/instruction.json"), tmp_path / "out")
    assert report["source_shard_verified"]
    assert not report["complete_archive_verified"]
    assert not report["complete_dataset_verified"]
    assert report["status"] == "selected_complete_members_verified"


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE),
    ("/absolute", tarfile.REGTYPE), ("link", tarfile.SYMTYPE), ("hard", tarfile.LNKTYPE), ("device", tarfile.CHRTYPE)])
def test_tar_path_and_link_rejections(tmp_path, name, kind):
    path, record = make_archive(tmp_path, [(name, kind, b"bad"), ("target", tarfile.REGTYPE, b"good")])
    with pytest.raises(ValueError):
        extract_verified_prefix_members(path, record, ("target",), tmp_path / "out")
    report = json.loads((tmp_path / "out/extraction.json").read_text())
    assert report["status"] == "failed"


def test_wrong_shard_hash_fails_before_creating_output(tmp_path):
    path, record = make_archive(tmp_path, [("target", tarfile.REGTYPE, b"good")])
    record["lfs"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        extract_verified_prefix_members(path, record, ("target",), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_or_truncated_member_never_passes(tmp_path):
    path, record = make_archive(tmp_path, [("different", tarfile.REGTYPE, b"data")])
    with pytest.raises(ValueError, match="not every"):
        extract_verified_prefix_members(path, record, ("target",), tmp_path / "out")
    assert json.loads((tmp_path / "out/extraction.json").read_text())["status"] == "failed"


@pytest.fixture
def recorded_fixture(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    import numpy as np
    from PIL import Image
    from vlaforge.adapters.rdt import rdt_inputs as inputs

    root = tmp_path / "episode"
    hdf5 = root / inputs.FIRST_EPISODE_MEMBERS[0]
    hdf5.parent.mkdir(parents=True)
    instruction = root / inputs.FIRST_EPISODE_MEMBERS[1]
    instruction.write_text(json.dumps({"instruction": "Recorded unit test instruction"}))
    encoded = io.BytesIO()
    Image.fromarray(np.full((3, 4, 3), (200, 100, 10), dtype=np.uint8)).save(encoded, format="JPEG")
    raw = encoded.getvalue()
    with h5py.File(hdf5, "w") as episode:
        episode.attrs["sim"] = False
        episode.attrs["compress"] = True
        episode.create_dataset("observations/qpos", data=np.arange(42, dtype=np.float32).reshape(3, 14))
        for camera in inputs.CAMERA_ORDER:
            episode.create_dataset(f"observations/images/{camera}", data=np.array([raw] * 3, dtype=f"S{len(raw)}"))
    monkeypatch.setattr(inputs, "FIRST_EPISODE_SHA256", (file_identity(hdf5)["sha256"], file_identity(instruction)["sha256"]))
    source = tmp_path / "source"
    source_records = {}
    for name in ("scripts/agilex_model.py", "data/hdf5_vla_dataset.py", "configs/dataset_control_freq.json"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"agilex":25}' if name.endswith(".json") else "# fixture source")
        source_records[name] = file_identity(path)
    monkeypatch.setattr(inputs, "verify_source", lambda path: {"files": source_records})
    return inputs, root, source


def test_pack_recomputes_recorded_tensors_not_only_manifest_hash(recorded_fixture, tmp_path):
    import numpy as np
    inputs, root, source = recorded_fixture
    output = tmp_path / "pack"
    manifest = inputs.prepare_recorded_input(episode_root=root, source_root=source, destination=output,
                                             step=1, seed=123, image_decoding="pil-rgb")
    arrays, loaded = inputs.load_recorded_input(output, source_root=source)
    assert loaded["instruction"] == "Recorded unit test instruction"
    assert arrays["proprio"].shape == (1, 14)
    assert [item["step"] for item in loaded["images"]] == [0, 0, 0, 1, 1, 1]
    arrays["proprio"] = arrays["proprio"] + 1
    np.savez(output / "inputs.npz", **arrays)
    manifest["inputs"] = file_identity(output / "inputs.npz")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="independently decoded"):
        inputs.load_recorded_input(output, source_root=source)


@pytest.mark.parametrize("step", [0, 3, -1, True])
def test_no_fabricated_or_unavailable_camera_history(recorded_fixture, tmp_path, step):
    inputs, root, source = recorded_fixture
    with pytest.raises(ValueError, match="history"):
        inputs.prepare_recorded_input(episode_root=root, source_root=source, destination=tmp_path / "bad-pack",
                                      step=step, seed=1, image_decoding="pil-rgb")
