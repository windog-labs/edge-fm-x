"""Materialized artifact schema/filesystem tests, not model numerical evidence."""

import hashlib
import zipfile
from dataclasses import replace

import pytest
from vlaforge.deployment.aoti_materialized import (
    MaterializedAotiPackage,
    MaterializedFile,
    materialize_aoti_package,
)


def package(tmp_path):
    path = tmp_path / "source.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in (("archive_format", b"pt2"), ("archive_version", b"0"),
            ("data/aotinductor/model/a.wrapper.so", b"not loaded in filesystem fixture"),
            ("data/aotinductor/model/a.wrapper_metadata.json", b'{"AOTI_DEVICE_KEY":"cpu"}'),
            ("data/aotinductor/model/a.wrapper.json", b'{"nodes":[]}'),
            ("data/constants/weight.blob", b"payload")):
            archive.writestr("archive/" + name, content)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = materialize_aoti_package(path, tmp_path / "published", sha256=digest, size_bytes=path.stat().st_size)
    return path, manifest, MaterializedAotiPackage.parse(manifest.read_text())


def test_materializes_complete_immutable_payload_and_roundtrips(tmp_path):
    source, manifest, result = package(tmp_path)
    assert result.package_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.weight_blob == "payload/weight.blob"
    assert result.library == "payload/a.wrapper.so"
    assert result.canonical_text() == manifest.read_text()
    assert result.verify(manifest.parent) == {"AOTI_DEVICE_KEY": "cpu"}
    assert sorted(path.name for path in manifest.parent.iterdir()) == ["model.vfaoti", "payload"]
    with pytest.raises(FileExistsError):
        materialize_aoti_package(source, manifest.parent, sha256=result.package_sha256, size_bytes=result.package_size_bytes)


@pytest.mark.parametrize("name", [".", "../x", "/x", "x//y", "x/../y", "a b", "a\\b", "-", "a/", "a/./b"])
def test_invalid_names_fail_closed(name):
    with pytest.raises(ValueError, match="path"):
        MaterializedFile(name, "a" * 64, 1)


@pytest.mark.parametrize("change", ["order", "duplicate", "missing-library", "missing-metadata", "missing-blob", "device", "version", "trailing", "leading-zero"])
def test_invalid_or_noncanonical_manifest_rejected(tmp_path, change):
    _, manifest, result = package(tmp_path)
    with pytest.raises(ValueError):
        if change == "order":
            replace(result, files=tuple(reversed(result.files)))
        elif change == "duplicate":
            replace(result, files=(*result.files, result.files[-1]))
        elif change == "missing-library":
            replace(result, library="payload/absent.so")
        elif change == "missing-metadata":
            replace(result, files=tuple(item for item in result.files if not item.path.endswith("_metadata.json")))
        elif change == "missing-blob":
            replace(result, weight_blob=None)
        elif change == "device":
            replace(result, device_type="unknown")
        else:
            text = manifest.read_text()
            if change == "version":
                text = text.replace("MATERIALIZED 1", "MATERIALIZED 2")
            elif change == "leading-zero":
                text = text.replace("files 4", "files 04")
            else:
                text += "trailing\n"
            MaterializedAotiPackage.parse(text)


@pytest.mark.parametrize("change", ["content", "missing", "symlink", "metadata-device"])
def test_signed_members_and_metadata_checked_before_load(tmp_path, change):
    _, manifest, result = package(tmp_path)
    library = manifest.parent / result.library
    if change == "content":
        library.write_bytes(b"changed")
    elif change == "missing":
        library.unlink()
    elif change == "symlink":
        outside = tmp_path / "outside"
        library.rename(outside)
        library.symlink_to(outside)
    else:
        result = replace(result, device_type="cuda")
    with pytest.raises(ValueError):
        result.verify(manifest.parent)
