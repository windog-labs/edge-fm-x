"""Explicit extraction configuration and opt-in actual LibTorch CPU checks."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from vlaforge.deployment.build import _aoti_package_extraction_configuration

RUNTIME = Path(__file__).resolve().parents[2]


def configuration(root, *, backend="aoti", version="2.10.0+cu128"):
    return _aoti_package_extraction_configuration(
        root, {"r": SimpleNamespace(capability=SimpleNamespace(backend=backend))},
        {backend: version}, runtime_root=RUNTIME,
    )


def test_default_has_no_new_configuration_or_source_reads():
    assert _aoti_package_extraction_configuration(None, {}, {}, runtime_root=Path("/absent")) is None


@pytest.mark.parametrize("value", [True, 1, "", "relative", "/a/../b", "/a/./b", "/a//b", "/a/", "/a;B=1", '/a"', "/a\\b", "/a\nb", "/a$B"])
def test_unsafe_build_root_rejected(value):
    with pytest.raises(ValueError, match="path"):
        configuration(value)


@pytest.mark.parametrize("version", [None, "unknown", "2.9.1", "2.11.0", "2.10", "2.10.0rc1"])
def test_unaudited_sdk_rejected(version):
    with pytest.raises(ValueError, match="LibTorch 2.10"):
        configuration("/nas/private", version=version)


def test_explicit_build_records_root_and_actual_sources():
    result = configuration("/nas/private root")
    assert result["root"] == "/nas/private root"
    assert not result["shared_cache"] and not result["environment_modified"]
    for name, digest in result["source_sha256"].items():
        assert digest == hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="AOTI Region"):
        configuration("/nas/private", backend="torchscript")


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if os.environ.get("VLAFORGE_RUN_AOTI_EXTRACTION_CPU") != "1":
        pytest.skip("opt-in actual LibTorch SDK build and CPU AOTI execution")
    import torch

    assert not torch.cuda.is_initialized()
    root = tmp_path_factory.mktemp("aoti-extraction")
    build = root / "build"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "TORCH_CUDA_ARCH_LIST": "8.6"}
    commands = []
    for command in [
        ["cmake", "-S", str(RUNTIME), "-B", str(build), "-DVLAFORGE_BUILD_AOTI_BACKEND=ON",
         "-DBUILD_TESTING=ON", "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}"],
        ["cmake", "--build", str(build), "--parallel", "2", "--target", "vlaforge_aoti_extraction_smoke"],
    ]:
        result = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
        commands.append({"command": command, "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode})
        (root / "build-commands.json").write_text(json.dumps(commands, indent=2))
        assert result.returncode == 0, result.stdout + result.stderr
    return root, build / "tests/cpp/vlaforge_aoti_extraction_smoke", environment


def package(path, records):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in [("archive_format", b"pt2"), ("archive_version", b"0"), (".data/version", b"6\n"), *records]:
            archive.writestr("fixture/" + name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


BASE = [("data/aotinductor/model/test.wrapper.so", b"not loaded in extract-only test"),
        ("data/aotinductor/model/test.wrapper_metadata.json", b'{"AOTI_DEVICE_KEY":"cpu"}')]


def invoke(native, tmp_path, records, *, expected=0, error=None, sha=None, root=None, mode="extract", reference=None):
    _, runner, environment = native
    archive = tmp_path / "package.pt2"
    digest = package(archive, records)
    if mode == "sequence":
        from vlaforge.deployment import (
            AotiSequenceArtifact,
            AotiSequenceManifest,
            AotiSequenceNode,
            AotiSequenceValue,
        )

        manifest = AotiSequenceManifest(
            region_name="composed", target="cpu", device="cpu",
            values=(AotiSequenceValue(0, "x", "input", 0, "f32", (4, 4)),
                    AotiSequenceValue(1, "y", "output", 0, "f32", (4,))),
            artifacts=(AotiSequenceArtifact(0, "histogram", archive.name, digest, archive.stat().st_size),),
            nodes=(AotiSequenceNode(0, (0,), (1,)),),
        )
        archive = tmp_path / "sequence.txt"
        archive.write_text(manifest.canonical_text())
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    destination = root or tmp_path / "private"
    destination.mkdir(mode=0o700, exist_ok=True)
    sentinel = destination / "foreign-sentinel"
    sentinel.write_bytes(b"untouched")
    command = [str(runner), mode, str(archive), str(destination), sha or digest]
    if reference:
        command.append(str(reference))
    result = subprocess.run(command, env=environment, text=True, capture_output=True, check=False)
    (tmp_path / "command.json").write_text(json.dumps({"command": command, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, indent=2))
    assert result.returncode == expected, result.stdout + result.stderr
    if error:
        assert error in result.stderr
    assert sentinel.read_bytes() == b"untouched"
    assert sorted(p.name for p in destination.iterdir()) == ["foreign-sentinel"]
    return result


def test_native_streaming_sidecars_and_constants_own_cleanup(native, tmp_path):
    records = BASE + [("data/constants/sub/weight.bin", b"x" * (3 * 1024 * 1024 + 3)),
                      ("data/aotinductor/model/test.wrapper.json", b'{"nodes":[]}'),
                      ("data/aotinductor/model/kernel.cubin", b"CUDA fixture bytes"),
                      ("data/aotinductor/model/weights.blob", b"blob"),
                      ("data/aotinductor/another/ignored.so", b"other model")]
    result = invoke(native, tmp_path, records)
    assert "MEMBER weight.bin 3145731" in result.stdout
    assert "MEMBER test.wrapper.json" in result.stdout
    assert "ignored.so" not in result.stdout


@pytest.mark.parametrize("extra,error", [
    ([("../escape", b"x")], "unsafe"),
    ([("/escape", b"x")], "unsafe"),
    ([("data/aotinductor/model/a/../escape", b"x")], "unsafe"),
    ([("data/aotinductor/model/a\\escape", b"x")], "unsafe"),
    ([BASE[0]], "duplicate"),
    ([("data/constants/test.wrapper.so", b"other")], "collision"),
    ([("data/aotinductor/model/second.so", b"other")], "exactly one"),
    ([("data/aotinductor/model/a.blob", b"a"), ("data/aotinductor/model/b.blob", b"b")], "multiple weight"),
    ([("data/aotinductor/model/a", b"file"), ("data/aotinductor/model/a/b", b"nested")], "Not a directory"),
])
def test_native_malformed_members_fail_closed(native, tmp_path, extra, error):
    invoke(native, tmp_path, BASE + extra, expected=42, error=error)


def test_native_missing_metadata_and_bad_sha(native, tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    invoke(native, first, BASE[:1], expected=42, error="metadata")
    second = tmp_path / "second"
    second.mkdir()
    invoke(native, second, BASE, sha="0" * 64, expected=42, error="SHA-256")


def test_native_source_only_package_rejected(native, tmp_path):
    invoke(native, tmp_path, BASE[1:] + [("data/aotinductor/model/test.wrapper.cpp", b"source")],
           expected=42, error="source-only")


def test_native_symlink_and_nonprivate_roots_rejected(native, tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    link = tmp_path / "linked"
    link.symlink_to(actual, target_is_directory=True)
    invoke(native, tmp_path, BASE, root=link, expected=42, error="canonical private")
    actual.chmod(0o755)
    invoke(native, tmp_path, BASE, root=actual, expected=42, error="canonical private")


@pytest.mark.parametrize("weights", ["embedded", "blob"])
def test_actual_cpu_aoti_proxy_and_weight_package(native, tmp_path, weights):
    import torch
    from vlaforge.deployment.aoti_profile import aoti_configs

    class Histogram(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16)

        def forward(self, value):
            return torch.histc(value @ self.weight, bins=4, min=0.0, max=100.0)

    inputs = (torch.arange(16, dtype=torch.float32).reshape(4, 4),)
    module = Histogram().eval()
    exported = torch.export.export(module, inputs, strict=True)
    path = tmp_path / "actual.pt2"
    options = aoti_configs("aten-preserving")
    if weights == "blob":
        options.update({"aot_inductor.force_mmap_weights": False,
                        "aot_inductor.package_constants_in_so": False,
                        "aot_inductor.package_constants_on_disk_format": "binary_blob"})
    torch._inductor.aoti_compile_and_package(exported, package_path=str(path), inductor_configs=options)
    with zipfile.ZipFile(path) as archive:
        records = [(name.split("/", 1)[1], archive.read(name)) for name in archive.namelist()
                   if name.split("/", 1)[1] not in {"archive_format", "archive_version", ".data/version"}]
    proxies = [name for name, _ in records if name.endswith(".wrapper.json")]
    assert len(proxies) == 1, "test must exercise a real OSSProxyExecutor, not only native kernels"
    if weights == "blob":
        assert any(name.endswith(".blob") for name, _ in records)
    reference = tmp_path / "reference.bin"
    reference.write_bytes(module(*inputs).numpy().tobytes())
    result = invoke(native, tmp_path, records, mode="region", reference=reference)
    assert "EXTRACTION passed region" in result.stdout
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    invoke(native, sequence, records, mode="sequence", reference=reference)
    missing = tmp_path / "missing-proxy"
    missing.mkdir()
    invoke(native, missing, [(n, p) for n, p in records if n not in proxies], mode="region",
           reference=reference, expected=42, error="proxy executor")
    from vlaforge.deployment.aoti_load import load_aoti_package

    private = tmp_path / "python-private"
    private.mkdir(mode=0o700)
    with load_aoti_package(path, extraction_root=private,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           size_bytes=path.stat().st_size, device="cpu") as loaded:
        for _ in range(3):
            assert loaded(*inputs).numpy().tobytes() == reference.read_bytes()
        with zipfile.ZipFile(path) as original:
            for member in loaded.extraction["members"]:
                assert hashlib.sha256(original.read(member["record"])).hexdigest() == member["sha256"]
        (tmp_path / "python-extraction.json").write_text(json.dumps(loaded.extraction, indent=2))
    assert not list(private.iterdir())
    with pytest.raises(RuntimeError, match="closed"):
        loaded(*inputs)
    from vlaforge.deployment.aoti_materialized import (
        load_materialized_aoti,
        materialize_aoti_package,
    )

    materialized = materialize_aoti_package(path, tmp_path / "materialized",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)
    materialized_sha = hashlib.sha256(materialized.read_bytes()).hexdigest()
    original_members = {str(p.relative_to(materialized.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in materialized.parent.rglob("*") if p.is_file()}
    with load_materialized_aoti(materialized, sha256=materialized_sha,
                                size_bytes=materialized.stat().st_size, device="cpu") as materialized_run:
        assert materialized_run(*inputs).numpy().tobytes() == reference.read_bytes()
    command = [str(native[1]), "materialized", str(materialized), str(private), materialized_sha, str(reference)]
    result = subprocess.run(command, env=native[2], capture_output=True, text=True, check=False)
    (tmp_path / "materialized-native.json").write_text(json.dumps({"argv": command, "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr}, indent=2))
    assert result.returncode == 0, result.stdout + result.stderr
    assert {str(p.relative_to(materialized.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in materialized.parent.rglob("*") if p.is_file()} == original_members
    assert not list(private.iterdir())
    if weights == "embedded":
        legacy = tmp_path / "legacy-default"
        legacy.mkdir()
        invoke(native, legacy, records, mode="default", reference=reference)
        configured = tmp_path / "configured-default"
        configured.mkdir(mode=0o700)
        build_root, _, environment = native
        commands = []
        for command in [
            ["cmake", "-S", str(RUNTIME), "-B", str(build_root / "build"),
             f"-DVLAFORGE_AOTI_PACKAGE_EXTRACTION_ROOT={configured}"],
            ["cmake", "--build", str(build_root / "build"), "--parallel", "2", "--target", "vlaforge_aoti_extraction_smoke"],
        ]:
            result = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
            commands.append({"command": command, "stdout": result.stdout, "stderr": result.stderr,
                             "exit_code": result.returncode})
            (tmp_path / "default-build-commands.json").write_text(json.dumps(commands, indent=2))
            assert result.returncode == 0, result.stdout + result.stderr
        built = tmp_path / "build-default"
        built.mkdir()
        invoke(native, built, records, mode="default", root=configured, reference=reference)
    assert not torch.cuda.is_initialized()


def test_native_descriptor_required_before_load(native, tmp_path):
    invoke(native, tmp_path, BASE, mode="nohash", expected=42, error="requires SHA-256")


@pytest.mark.parametrize("change", ["version", "device", "path", "member-bytes", "missing-member", "symlink", "manifest-sha"])
def test_native_materialized_validation_precedes_library_loading(native, tmp_path, change):
    from vlaforge.deployment.aoti_materialized import materialize_aoti_package
    source = tmp_path / "source.pt2"
    digest = package(source, BASE)
    manifest = materialize_aoti_package(source, tmp_path / "published", sha256=digest, size_bytes=source.stat().st_size)
    library = manifest.parent / "payload/test.wrapper.so"
    if change in ("version", "device", "path"):
        content = manifest.read_text()
        old, new = {"version": ("MATERIALIZED 1", "MATERIALIZED 2"),
                    "device": ("model cpu", "model cuda"),
                    "path": ("payload/test.wrapper.so", "../outside.so")}[change]
        manifest.write_text(content.replace(old, new))
    elif change == "member-bytes":
        library.write_bytes(b"changed")
    elif change == "missing-member":
        library.unlink()
    elif change == "symlink":
        outside = tmp_path / "outside.so"
        library.rename(outside)
        library.symlink_to(outside)
    sha = "0" * 64 if change == "manifest-sha" else hashlib.sha256(manifest.read_bytes()).hexdigest()
    command = [str(native[1]), "materialized", str(manifest), "/unused", sha]
    result = subprocess.run(command, env=native[2], capture_output=True, text=True, check=False)
    (tmp_path / "rejected-native.json").write_text(json.dumps({"argv": command, "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr}, indent=2))
    assert result.returncode == 42, result.stdout + result.stderr
    assert "file too short" not in result.stderr and "invalid ELF" not in result.stderr
    assert "missing expected output" not in result.stderr
