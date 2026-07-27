import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from vlaforge.adapters import build_hybrid_external_feature_fixture
from vlaforge.codegen import hybrid_external_feature_validators
from vlaforge.compiler import CompilerProfile, compile_module
from vlaforge.deployment import (
    ArtifactIdentity,
    ArtifactKind,
    ArtifactResidency,
    BackendCapability,
    EffectAudit,
    RegionArtifactContract,
    ValueContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_bev_shared_plugin_builds_verified_no_python_bundle(
    tmp_path: Path,
) -> None:
    runtime_root = Path(__file__).resolve().parents[2]
    example_root = runtime_root / "examples/external_bev_plugin"
    fixture = build_hybrid_external_feature_fixture()
    compilation = compile_module(
        fixture.module,
        profile=CompilerProfile.VERIFIED,
        default_device="cpu",
    )
    schema_digest = compilation.certificate.io_schema_digest

    plugin_build = tmp_path / "plugin-build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(example_root),
            "-B",
            str(plugin_build),
            "-G",
            "Ninja",
            f"-DVLAFORGE_INCLUDE_DIR={runtime_root / 'include'}",
            f"-DVLAFORGE_EXPECTED_SCHEMA_DIGEST={schema_digest}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(plugin_build), "--parallel", "4"],
        check=True,
        capture_output=True,
        text=True,
    )
    plugin = plugin_build / "libvlaforge_external_bev_plugin.so"
    plugin_digest = _sha256(plugin)
    supported_dtypes = ("f32", "i32", "i64")
    contracts = {}
    sources = {}
    for region_id, region in enumerate(compilation.module.regions):
        contracts[region.name] = RegionArtifactContract(
            region_id=region_id,
            region_name=region.name,
            inputs=tuple(
                ValueContract.from_ir(
                    value.name,
                    value.type,
                    device="cpu",
                )
                for value in region.inputs
            ),
            outputs=tuple(
                ValueContract.from_ir(
                    f"output_{index}",
                    value,
                    device="cpu",
                )
                for index, value in enumerate(region.outputs)
            ),
            io_schema_digest=schema_digest,
            identity=ArtifactIdentity(
                model_name=fixture.module.name,
                upstream_revision="source-faithful-external-bev/1",
                checkpoint_identity="fixture:no-checkpoint",
                graph_sha256=plugin_digest,
            ),
            artifact_kind=ArtifactKind.SHARED_LIBRARY,
            artifact_path=(
                f"artifacts/{region_id:03d}_{region.name}.so"
            ),
            artifact_sha256=plugin_digest,
            artifact_size_bytes=plugin.stat().st_size,
            residency=(
                ArtifactResidency.INVOCATION
                if region_id == 0
                else ArtifactResidency.SESSION
            ),
            workspace=WorkspaceContract(),
            capability=BackendCapability(
                backend="shared_plugin",
                target="cpu",
                supported_dtypes=supported_dtypes,
            ),
            effect_audit=EffectAudit(),
            backend_variant="shared-plugin/1",
        )
        sources[region.name] = plugin

    bundle = tmp_path / "bundle"
    manifest = build_artifact_compile_bundle(
        fixture.module,
        bundle,
        region_artifacts=contracts,
        artifact_sources=sources,
        validators=hybrid_external_feature_validators(),
        runner_source=(example_root / "runner.cpp").read_text(
            encoding="utf-8"
        ),
        runtime_root=runtime_root,
        cmake_prefix_path=Path(sys.prefix),
        backend_versions={"shared_plugin": "1"},
        profile=CompilerProfile.VERIFIED,
        source_revision="source-faithful-external-bev/1",
        source_dirty=False,
        default_device="cpu",
        state_device="cpu",
    )
    manifest.verify_files(bundle)

    environment = {
        **os.environ,
        "PYTHONHOME": "/definitely/not/a/python/home",
        "PYTHONPATH": "/definitely/not/a/python/path",
    }
    completed = subprocess.run(
        [str(bundle / "bin/vlaforge_generated_runner"), str(bundle)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("EXTERNAL_PLUGIN,")
    )
    result_fields = result_line.split(",")
    assert [float(value) for value in result_fields[1:5]] == pytest.approx(
        [0.006, 0.31, 0.5, 0.1],
        abs=1e-7,
    )
    assert int(result_fields[5]) == 58
    assert "PLUGIN_CACHE,1,5" in completed.stdout
    assert "PLUGIN_FAILURE_RETRY,1" in completed.stdout
    assert "PLUGIN_TYPED_GENERIC,1" in completed.stdout
    linked = subprocess.run(
        ["ldd", str(bundle / "bin/vlaforge_generated_runner")],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    assert "libpython" not in linked

    tampered_plugin = (
        bundle / manifest.region_artifacts[0].artifact_path
    )
    original_plugin = tampered_plugin.read_bytes()
    corrupted_plugin = bytearray(original_plugin)
    corrupted_plugin[-1] ^= 0x01
    try:
        tampered_plugin.write_bytes(corrupted_plugin)
        rejected = subprocess.run(
            [
                str(bundle / "bin/vlaforge_generated_runner"),
                str(bundle),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert rejected.returncode != 0
    finally:
        tampered_plugin.write_bytes(original_plugin)
    manifest.verify_files(bundle)

    generated = (bundle / "generated/session_generated.cpp").read_text(
        encoding="utf-8"
    )
    assert "vlaforge_external_region_plugin_open" in generated
    assert "region_plugins_[0u] == nullptr" in generated
    assert "kArtifactInvocationResident0 =\n    true" in generated
    assert "VLAFORGE_VALUE_SCALAR" in generated
    assert "shared plugin Region load failed" in generated
    assert all(
        contract.capability.backend == "shared_plugin"
        and contract.artifact_kind is ArtifactKind.SHARED_LIBRARY
        for contract in manifest.region_artifacts
    )
