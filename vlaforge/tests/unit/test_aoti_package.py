import copy
import hashlib
import json
import os
import zipfile

import pytest
from vlaforge.deployment import aoti_package as package


def document(value=7):
    return {
        "nodes": [
            {
                "name": "buf0",
                "node": {
                    "target": "aten::_assert_tensor_metadata",
                    "inputs": [
                        {"name": "dtype", "arg": {"as_scalar_type": value}},
                        {"name": "size", "arg": {"as_int": 7}},
                        {"name": "layout", "arg": {"as_layout": 7}},
                    ],
                },
            }
        ]
    }


def mapping():
    return {
        "as_scalar_type": {7: {"value": 6, "symbol": "dtype_float32"}},
        "as_layout": {7: {"value": 0, "symbol": "layout_strided"}},
    }


def test_native_enum_translation_does_not_change_ordinary_numbers_or_assertions():
    original = document()
    result, changes = package.translate_proxy_arguments(original, mapping())
    assert original == document()
    assert result["nodes"][0]["node"]["target"] == "aten::_assert_tensor_metadata"
    assert result["nodes"][0]["node"]["inputs"][1]["arg"] == {"as_int": 7}
    assert [(row["before"], row["after"]) for row in changes] == [(7, 6), (7, 0)]


@pytest.mark.parametrize("value", [0, 100, True, "7"])
def test_unknown_or_mistyped_enum_is_rejected(value):
    with pytest.raises(ValueError, match="unsupported serialized"):
        package.translate_proxy_arguments(document(value), mapping())


def test_package_translation_retains_source_code_weights_and_rejects_reapplication(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        package, "native_enum_maps", lambda: (mapping(), {"torch": "2.10.0+cpu-test"})
    )
    source, target = tmp_path / "original.pt2", tmp_path / "translated.pt2"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("model/weights.so", b"unaltered weights and machine code")
        archive.writestr("model/operator.wrapper.json", json.dumps(document()))
    original = source.read_bytes()
    report = package.normalize_proxy_enums(source, target)
    package.verify_package_audit(
        target,
        {"fallback_by_default": True},
        {
            "passes": package.package_pass_records({"fallback_by_default": True}),
            "translation": report,
        },
    )
    assert source.read_bytes() == original
    assert not report["numeric_parity_verified"]
    with zipfile.ZipFile(target) as archive:
        assert archive.read("model/weights.so") == b"unaltered weights and machine code"
    with pytest.raises(ValueError, match="already normalized"):
        package.normalize_proxy_enums(target, tmp_path / "twice.pt2")
    with pytest.raises(ValueError, match="new destination"):
        package.normalize_proxy_enums(source, target)


def test_exclusive_publish_does_not_delete_another_writer_file(tmp_path, monkeypatch):
    source, target = tmp_path / "original.pt2", tmp_path / "translated.pt2"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("model/code.so", b"source code")

    def concurrent_writer():
        target.write_bytes(b"other writer owns this")
        return mapping(), {"torch": "2.10.0"}

    monkeypatch.setattr(package, "native_enum_maps", concurrent_writer)
    with pytest.raises(FileExistsError):
        package.normalize_proxy_enums(source, target)
    assert target.read_bytes() == b"other writer owns this"
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_archive_write_removes_only_private_temporary(tmp_path, monkeypatch):
    source, target = tmp_path / "original.pt2", tmp_path / "translated.pt2"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("model/code.so", b"source code")
    monkeypatch.setattr(
        package, "native_enum_maps", lambda: (mapping(), {"torch": "2.10.0"})
    )
    monkeypatch.setattr(
        zipfile.ZipFile,
        "writestr",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write failed")),
    )
    with pytest.raises(OSError, match="write failed"):
        package.normalize_proxy_enums(source, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    with zipfile.ZipFile(source) as archive:
        assert archive.read("model/code.so") == b"source code"


@pytest.mark.parametrize("payload", [b'{"nodes":[],"nodes":[]}', b'{"nodes":NaN}'])
def test_duplicate_or_nonfinite_wrapper_json_is_rejected(
    tmp_path, monkeypatch, payload
):
    source, target = tmp_path / "original.pt2", tmp_path / "translated.pt2"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("model/op.wrapper.json", payload)
    monkeypatch.setattr(
        package, "native_enum_maps", lambda: (mapping(), {"torch": "2.10.0"})
    )
    with pytest.raises(ValueError, match="package JSON"):
        package.normalize_proxy_enums(source, target)
    assert source.exists() and not target.exists()


def test_finalize_backup_publication_cannot_replace_existing_backup(
    tmp_path, monkeypatch
):
    path = tmp_path / "candidate.pt2"
    path.write_bytes(b"original candidate")
    original = path.with_suffix(".unfinalized.pt2")
    link = os.link

    def competing_backup(source, destination):
        original.write_bytes(b"another original")
        return link(source, destination)

    monkeypatch.setattr(package.os, "link", competing_backup)
    with pytest.raises(FileExistsError):
        package.finalize_aoti_package(path, {"fallback_by_default": True})
    assert path.read_bytes() == b"original candidate"
    assert original.read_bytes() == b"another original"


@pytest.mark.parametrize(
    "change",
    [
        "missing_translation",
        "failed",
        "wrong_artifact_sha",
        "source_marker",
        "version",
        "parity_claim",
        "missing_marker",
        "unchanged_sha",
        "missing_member",
        "duplicate_member_record",
        "rewritten_code",
        "missing_rewrites",
        "passes",
    ],
)
def test_reuse_rejects_inconsistent_translation_evidence(tmp_path, monkeypatch, change):
    source, target = tmp_path / "original.pt2", tmp_path / "translated.pt2"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("model/code.so", b"preserved code and weights")
        archive.writestr("model/op.wrapper.json", json.dumps(document()))
    monkeypatch.setattr(
        package, "native_enum_maps", lambda: (mapping(), {"torch": "2.10.0"})
    )
    translation = package.normalize_proxy_enums(source, target)
    configs = {"fallback_by_default": True}
    audit = {
        "passes": package.package_pass_records(configs),
        "translation": copy.deepcopy(translation),
    }
    translated = audit["translation"]
    if change == "missing_translation":
        del audit["translation"]
    elif change == "failed":
        translated["status"] = "failed"
    elif change == "wrong_artifact_sha":
        translated["artifact_sha256"] = "f" * 64
    elif change == "source_marker":
        translated["source_sha256"] = "f" * 64
    elif change == "version":
        translated["torch"] = "2.11.0"
    elif change == "parity_claim":
        translated["numeric_parity_verified"] = True
    elif change == "missing_marker":
        target.write_bytes(source.read_bytes())
        translated["artifact_sha256"] = package.digest(target)
    elif change == "unchanged_sha":
        translated["unchanged_members"][0]["sha256"] = hashlib.sha256(
            b"different"
        ).hexdigest()
    elif change == "missing_member":
        translated["unchanged_members"] = []
    elif change == "duplicate_member_record":
        translated["unchanged_members"].append(translated["unchanged_members"][0])
    elif change == "rewritten_code":
        translated["rewrites"][0]["member"] = "model/code.so"
    elif change == "missing_rewrites":
        del translated["rewrites"]
    else:
        audit["passes"] = []
    with pytest.raises(ValueError):
        package.verify_package_audit(target, configs, audit)


def test_installed_serializer_maps_to_actual_native_abi():
    torch = pytest.importorskip("torch")
    if torch.__version__.split("+")[0] != "2.10.0":
        pytest.skip("version-specific C ABI")
    maps, _ = package.native_enum_maps()
    values = torch.ones(2)
    native = maps["as_scalar_type"][7]["value"]
    layout = maps["as_layout"][7]["value"]
    torch.ops.aten._assert_tensor_metadata.default(values, dtype=native, layout=layout)
