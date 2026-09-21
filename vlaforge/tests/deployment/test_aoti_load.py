"""Python extraction fail-closed/ownership tests; no CUDA execution."""

import hashlib
import json
import stat
import zipfile
from pathlib import Path

import pytest
from vlaforge.deployment.aoti_load import _extract, _verify, load_aoti_package


def archive(path, *, extra=(), version=b"0", metadata=b'{"AOTI_DEVICE_KEY":"cpu"}'):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, payload in [
            ("archive_format", b"pt2"), ("archive_version", version),
            ("data/aotinductor/model/x.wrapper.so", b"library"),
            ("data/aotinductor/model/x.wrapper_metadata.json", metadata),
            *extra,
        ]:
            output.writestr("archive/" + name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("extra,error", [
    ([("../escape", b"escape")], "unsafe"),
    ([("data/aotinductor/model/x.wrapper.so", b"duplicate")], "duplicate"),
    ([("data/constants/x.wrapper.so", b"collision")], "collision"),
    ([("data/aotinductor/model/y.so", b"duplicate")], "exactly one"),
    ([("data/aotinductor/model/a.blob", b"a"), ("data/constants/b.blob", b"b")], "multiple"),
    ([("data/aotinductor/model/sub", b"file"), ("data/aotinductor/model/sub/file", b"nested")], "File exists"),
])
def test_malformed_members_rejected_without_leftovers(tmp_path, extra, error):
    path = tmp_path / "model.pt2"
    archive(path, extra=extra)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with path.open("rb") as stream, pytest.raises((ValueError, FileExistsError), match=error):
        _extract(stream, root)
    assert not list(root.iterdir())


@pytest.mark.parametrize("payload", [b"{}", b'{"AOTI_DEVICE_KEY":1}', b'{"AOTI_DEVICE_KEY":"cpu","AOTI_DEVICE_KEY":"cuda"}'])
def test_metadata_or_device_rejected_before_native_load(tmp_path, payload):
    path = tmp_path / "model.pt2"
    digest = archive(path, metadata=payload)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="metadata"):
        load_aoti_package(path, extraction_root=root, sha256=digest, size_bytes=path.stat().st_size, device="cpu")
    assert not list(root.iterdir())


def test_actual_streaming_hashes_and_manual_owner_lifecycle(tmp_path):
    path = tmp_path / "model.pt2"
    digest = archive(path, extra=[("data/constants/w.bin", b"w" * (3 * 1024 * 1024 + 1))])
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with path.open("rb") as stream:
        _verify(stream, digest, path.stat().st_size)
        owner, ledger = _extract(stream, root)
        _verify(stream, digest, path.stat().st_size)
    for member in ledger["members"]:
        assert hashlib.sha256((Path(owner.name) / member["file"]).read_bytes()).hexdigest() == member["sha256"]
    assert len(list(root.iterdir())) == 1
    owner.cleanup()
    owner.cleanup()
    assert not list(root.iterdir())


def test_zip_symlink_rejected_not_followed(tmp_path):
    path = tmp_path / "model.pt2"
    archive(path)
    with zipfile.ZipFile(path, "a") as output:
        link = zipfile.ZipInfo("archive/data/aotinductor/model/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(link, b"/etc/passwd")
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with path.open("rb") as stream, pytest.raises(ValueError, match="links"):
        _extract(stream, root)
    assert not list(root.iterdir())


@pytest.mark.parametrize("change", ["hash", "size", "root-permission", "root-symlink", "version"])
def test_loader_rejects_before_runner(tmp_path, change):
    path = tmp_path / "model.pt2"
    digest = archive(path, version=b"99" if change == "version" else b"0")
    root = tmp_path / "private"
    root.mkdir(mode=0o755 if change == "root-permission" else 0o700)
    if change == "root-symlink":
        link = tmp_path / "link"
        link.symlink_to(root, target_is_directory=True)
        root = link
    with pytest.raises(ValueError):
        load_aoti_package(path, extraction_root=root, sha256="0" * 64 if change == "hash" else digest,
                          size_bytes=path.stat().st_size + (change == "size"), device="cpu")
    assert not list(root.iterdir())


def test_recorded_evidence_is_json_data(tmp_path):
    path = tmp_path / "model.pt2"
    archive(path)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    with path.open("rb") as stream:
        owner, ledger = _extract(stream, root)
    try:
        assert json.loads(json.dumps(ledger)) == ledger
    finally:
        owner.cleanup()
