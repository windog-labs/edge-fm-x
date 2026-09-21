from copy import deepcopy
import hashlib
import io
import json

import pytest

from vlaforge.adapters.rdt import rdt_assets as assets


def metadata(component):
    spec = assets.RDT_ASSETS[component]
    records = []
    for name in spec["files"]:
        item = {"rfilename": name, "size": 3,
                "blobId": hashlib.sha1(b"blob 3\0abc").hexdigest()}
        if name.endswith((".bin", ".safetensors")):
            item["lfs"] = {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
        records.append(item)
    return {"id": spec["repo"], "sha": spec["revision"],
            "cardData": {"license": spec["license"]}, "siblings": records}


@pytest.mark.parametrize("component", assets.RDT_ASSETS)
def test_selected_inventory_requires_complete_pinned_profile(component):
    value = metadata(component)
    assert len(assets.selected_records(value, component)) == len(assets.RDT_ASSETS[component]["files"])
    for key, replacement in (("id", "other/repo"), ("sha", "a" * 40), ("siblings", [])):
        broken = {**value, key: replacement}
        with pytest.raises(ValueError):
            assets.selected_records(broken, component)


@pytest.mark.parametrize("mutation", ["duplicate", "license", "lfs_size", "lfs_sha", "size", "blob", "no_lfs"])
def test_malformed_metadata_rejected(mutation):
    value = metadata("policy")
    weight = value["siblings"][-1]
    if mutation == "duplicate":
        value["siblings"].append(deepcopy(weight))
    elif mutation == "license":
        value["cardData"]["license"] = "unknown"
    elif mutation == "lfs_size":
        weight["lfs"]["size"] = 4
    elif mutation == "lfs_sha":
        weight["lfs"]["sha256"] = "a" * 63
    elif mutation == "size":
        weight["size"] = True
    elif mutation == "blob":
        weight["blobId"] = "not-a-digest"
    else:
        del weight["lfs"]
    with pytest.raises(ValueError):
        assets.selected_records(value, "policy")


def test_verify_all_components_and_detect_same_size_corruption(tmp_path):
    (tmp_path / "metadata").mkdir()
    for component in assets.RDT_ASSETS:
        value = metadata(component)
        (tmp_path / "metadata" / f"{component}.json").write_text(json.dumps(value))
        (tmp_path / component).mkdir()
        for item in value["siblings"]:
            (tmp_path / component / item["rfilename"]).write_bytes(b"abc")
    report = assets.verify_assets(tmp_path)
    assert report["all_required_assets_verified"]
    assert not report["real_model_execution_verified"]
    (tmp_path / "text" / "spiece.model").write_bytes(b"xyz")
    with pytest.raises(ValueError, match="digest"):
        assets.verify_assets(tmp_path)


@pytest.mark.parametrize("name", ["../other", "/other", "text/../other", "a//b"])
def test_unsafe_asset_paths_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        assets._target(tmp_path, name)


class Response(io.BytesIO):
    def __init__(self, payload, status, headers):
        super().__init__(payload)
        self.status = status
        self.headers = headers


def test_download_checks_bytes_and_reuses_only_verified_file(tmp_path, monkeypatch):
    record = metadata("policy")["siblings"][-1]
    monkeypatch.setattr(assets, "urlopen", lambda *a, **k: Response(b"abc", 200, {"Content-Length": "3"}))
    result = assets.download_file(tmp_path, "policy", record, endpoint=assets.TRANSPORT_ENDPOINTS[0])
    assert not result["reused_verified"]
    assert assets.download_file(tmp_path, "policy", record, endpoint=assets.TRANSPORT_ENDPOINTS[0])["reused_verified"]
    (tmp_path / "policy" / record["rfilename"]).write_bytes(b"xyz")
    with pytest.raises(ValueError, match="digest"):
        assets.download_file(tmp_path, "policy", record, endpoint=assets.TRANSPORT_ENDPOINTS[0])


@pytest.mark.parametrize("response_range", ["bytes 1-2/3", "bytes 0-1/3", None])
def test_owned_resume_requires_exact_content_range(tmp_path, monkeypatch, response_range):
    record = metadata("policy")["siblings"][-1]
    directory = tmp_path / "policy"
    directory.mkdir()
    (directory / "pytorch_model.bin.vf-part").write_bytes(b"a")
    (directory / "pytorch_model.bin.vf-part.json").write_text(json.dumps({
        "repo": assets.RDT_ASSETS["policy"]["repo"],
        "revision": assets.RDT_ASSETS["policy"]["revision"], "record": record,
    }))
    def response(request, **kwargs):
        assert request.headers["Range"] == "bytes=1-"
        return Response(b"bc", 206, {"Content-Length": "2", "Content-Range": response_range})
    monkeypatch.setattr(assets, "urlopen", response)
    if response_range == "bytes 1-2/3":
        assert assets.download_file(tmp_path, "policy", record, endpoint=assets.TRANSPORT_ENDPOINTS[1])["resumed_bytes"] == 1
    else:
        with pytest.raises(ValueError, match="Content-Range"):
            assets.download_file(tmp_path, "policy", record, endpoint=assets.TRANSPORT_ENDPOINTS[1])
        assert (directory / "pytorch_model.bin.vf-part").read_bytes() == b"a"


def test_unowned_partial_preserved(tmp_path):
    directory = tmp_path / "policy"
    directory.mkdir()
    partial = directory / "pytorch_model.bin.vf-part"
    partial.write_bytes(b"original")
    with pytest.raises(ValueError, match="unowned"):
        assets.download_file(tmp_path, "policy", metadata("policy")["siblings"][-1], endpoint=assets.TRANSPORT_ENDPOINTS[0])
    assert partial.read_bytes() == b"original"
