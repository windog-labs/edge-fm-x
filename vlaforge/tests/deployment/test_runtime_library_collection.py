from pathlib import Path

import pytest
from vlaforge.deployment.build import _collect_runtime_libraries


def manifest(build, values):
    path = build / "vlaforge_runtime/vlaforge-runtime-libraries-Release.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(str(value) + "\n" for value in values))


def test_actual_shared_targets_and_soname_are_copied_and_hashed(tmp_path):
    build, bundle = tmp_path / "build", tmp_path / "bundle"
    library = build / "vlaforge_runtime/libprovider.so.1.2"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"actual-build-output")
    soname = library.with_name("libprovider.so.1")
    soname.symlink_to(library.name)
    manifest(build, [library, soname, library])
    records = _collect_runtime_libraries(build, bundle)
    assert len(records) == 2
    assert {record.path for record in records} == {
        "lib/libprovider.so.1.2",
        "lib/libprovider.so.1",
    }
    for record in records:
        assert record.role == "runtime_shared_library"
        assert (bundle / record.path).read_bytes() == b"actual-build-output"
        assert not (bundle / record.path).is_symlink()


def test_static_only_runtime_emits_an_empty_manifest(tmp_path):
    manifest(tmp_path / "build", [])
    assert _collect_runtime_libraries(tmp_path / "build", tmp_path / "bundle") == ()


def test_missing_target_manifest_is_not_silently_accepted(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        _collect_runtime_libraries(tmp_path / "build", tmp_path / "bundle")


@pytest.mark.parametrize("kind", ["absolute", "relative", "symlink"])
def test_library_outside_isolated_build_is_rejected(tmp_path, kind):
    build = tmp_path / "build"
    outside = tmp_path / "vendor.so"
    outside.write_bytes(b"not-a-project-build-target")
    value = outside
    if kind == "relative":
        value = Path("../vendor.so")
    elif kind == "symlink":
        value = build / "vlaforge_runtime/link.so"
        value.parent.mkdir(parents=True)
        value.symlink_to(outside)
    manifest(build, [value])
    with pytest.raises(ValueError, match="outside"):
        _collect_runtime_libraries(build, tmp_path / "bundle")


def test_distinct_target_basename_collision_is_rejected(tmp_path):
    build = tmp_path / "build"
    paths = [build / name / "library.so" for name in ("one", "two")]
    for path in paths:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"target")
    manifest(build, paths)
    with pytest.raises(ValueError, match="basenames collide"):
        _collect_runtime_libraries(build, tmp_path / "bundle")


def test_bundle_auxiliary_file_is_not_overwritten(tmp_path):
    build, bundle = tmp_path / "build", tmp_path / "bundle"
    source = build / "libprovider.so"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"built")
    target = bundle / "lib/libprovider.so"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"existing")
    manifest(build, [source])
    with pytest.raises(ValueError, match="existing bundle"):
        _collect_runtime_libraries(build, bundle)
    assert target.read_bytes() == b"existing"
