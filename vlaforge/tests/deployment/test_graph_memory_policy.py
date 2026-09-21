"""Generic build-time policy validation; no model or GPU execution claims."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from vlaforge.deployment.build import _libtorch_graph_memory_configuration


def contracts(*backends, target="sm_86"):
    return {str(index): SimpleNamespace(capability=SimpleNamespace(backend=backend, target=target))
            for index, backend in enumerate(backends)}


@pytest.mark.parametrize("policy", [None, True, "", "on", "scoped", "SCOPED-RECLAIM"])
def test_unknown_policy_rejected(policy):
    with pytest.raises(ValueError, match="must be retain or scoped-reclaim"):
        _libtorch_graph_memory_configuration(policy, {}, {})


def test_default_retain_keeps_unknown_backend_versions_compatible():
    result = _libtorch_graph_memory_configuration("retain", contracts("aoti"), {"aoti": "unknown"})
    assert result["cmake_definition"].endswith("=OFF")
    assert not result["scoped_device_frees_possible"]


@pytest.mark.parametrize("version", [None, "", "unknown", "2.9.1", "2.11.0", "2.10", "2.10.0rc1", "2.10.0;injected"])
def test_scoped_rejects_unqualified_backend_versions(version):
    with pytest.raises(ValueError, match="audited LibTorch 2.10"):
        _libtorch_graph_memory_configuration("scoped-reclaim", contracts("torchscript"), {"torchscript": version})


@pytest.mark.parametrize("backends,target", [(["torchscript"], "cpu"), (["shared_plugin"], "sm_86"), ([], "sm_86")])
def test_scoped_requires_libtorch_cuda_provider(backends, target):
    with pytest.raises(ValueError, match="CUDA LibTorch Region"):
        _libtorch_graph_memory_configuration("scoped-reclaim", contracts(*backends, target=target), {})


def test_all_libtorch_cuda_backend_versions_must_be_qualified():
    with pytest.raises(ValueError, match="for aoti"):
        _libtorch_graph_memory_configuration("scoped-reclaim", contracts("torchscript", "aoti"),
                                            {"torchscript": "2.10.0+cu128", "aoti": "2.9.1"})
    result = _libtorch_graph_memory_configuration("scoped-reclaim", contracts("torchscript", "aoti"),
                                                 {"torchscript": "2.10.0+cu128", "aoti": "2.10.0+cu128"})
    assert result["cmake_definition"].endswith("=ON")
    assert result["actual_sdk_version_checked_by_cmake_and_header"]
    assert result["scoped_device_frees_possible"] and not result["global_cache_clear"]


@pytest.mark.skipif(os.environ.get("VLAFORGE_RUN_LIBTORCH_CUDA_BUILD") != "1",
                    reason="opt-in actual CUDA SDK compilation without GPU execution")
def test_public_scoped_argument_reaches_actual_native_build(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from vlaforge.compiler import compile_module
    from vlaforge.deployment import (
        ArtifactIdentity,
        ArtifactKind,
        BackendCapability,
        EffectAudit,
        RegionArtifactContract,
        ValueContract,
        WorkspaceContract,
        build_artifact_compile_bundle,
    )
    from vlaforge.deployment import build as build_implementation
    from vlaforge.frontend import InvocationBuilder, tensor_region
    from vlaforge.ir.program import InputPort, OutputPort, Value
    from vlaforge.ir.types import TensorType

    assert not torch.cuda.is_initialized()
    runtime = Path(__file__).resolve().parents[2]
    tensor = TensorType((4,), "f32")

    class Double(torch.nn.Module):
        def forward(self, value):
            return value * 2

    @tensor_region("twice", inputs=(Value("x", tensor),), outputs=(tensor,))
    def twice(value):
        return value * 2

    builder = InvocationBuilder("graph_policy_build_test", inputs=(InputPort("x", tensor, device="cuda:0"),
                               InputPort("accepted", TensorType((1,), "bool"), device="cuda:0")),
                               outputs=(OutputPort("result", tensor, device="cuda:0"),))
    (result,) = builder.call(twice, builder.input("x"))
    invocation = builder.finish({"result": result}, accepted=builder.input("accepted"))
    region = invocation.module.regions[0]
    archive = tmp_path / "twice.pt"
    # Device-neutral tensor arithmetic, traced only on CPU; this is build-only
    # evidence and does not claim a GPU capture or CUDA numerical validation.
    torch.jit.trace(Double(), (torch.arange(4, dtype=torch.float32),)).save(str(archive))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    contract = RegionArtifactContract(
        region_id=0, region_name=region.name,
        inputs=tuple(ValueContract.from_ir(value.name, value.type, device="cuda:0") for value in region.inputs),
        outputs=tuple(ValueContract.from_ir(f"output_{i}", value, device="cuda:0") for i, value in enumerate(region.outputs)),
        io_schema_digest=compile_module(invocation.module).certificate.io_schema_digest,
        identity=ArtifactIdentity("graph_policy_build_test", "test-source", "fixture:no-checkpoint", digest),
        artifact_kind=ArtifactKind.TORCHSCRIPT_ARCHIVE, artifact_path="artifacts/twice.pt",
        artifact_sha256=digest, artifact_size_bytes=archive.stat().st_size,
        workspace=WorkspaceContract(), effect_audit=EffectAudit(),
        capability=BackendCapability("torchscript", "sm_86", ("f32",), supports_execution_context=True,
                                     supports_device_resident_io=True, requires_synchronize=True),
        backend_variant="torchscript-aten-context/1",
    )
    calls = []
    original_run = subprocess.run

    def observe(command, *args, **kwargs):
        completed = original_run(command, *args, **kwargs)
        if command[:2] == ["cmake", "--build"]:
            build = Path(command[2])
            cache = (build / "CMakeCache.txt").read_text()
            flags = list(build.rglob("vlaforge_libtorch_graph_backend.dir/flags.make"))
            assert "VLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM:BOOL=ON" in cache
            assert len(flags) == 1 and "-DVLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=1" in flags[0].read_text()
            (tmp_path / "actual-CMakeCache.txt").write_text(cache)
            (tmp_path / "actual-provider-flags.make").write_text(flags[0].read_text())
        calls.append({"command": command, "returncode": completed.returncode,
                      "stdout": completed.stdout, "stderr": completed.stderr})
        return completed

    monkeypatch.setattr(subprocess, "run", observe)
    bundle = tmp_path / "bundle"
    manifest = build_artifact_compile_bundle(
        invocation.module, bundle, region_artifacts={region.name: contract}, artifact_sources={region.name: archive},
        validators=invocation.cpp_validators(),
        runner_source='#include "session_generated.h"\n#include "vlaforge/backends/libtorch_graph.h"\nint main() { return vlaforge_libtorch_graph_backend_api() == nullptr; }\n',
        runtime_root=runtime, cmake_prefix_path=torch.utils.cmake_prefix_path,
        backend_versions={"torchscript": str(torch.__version__)}, source_revision="build-only-test", source_dirty=True,
        environment={"CUDA_VISIBLE_DEVICES": "", "TORCH_CUDA_ARCH_LIST": "8.6", "CMAKE_BUILD_PARALLEL_LEVEL": "2"},
        default_device="cuda:0", state_device="cuda:0", libtorch_graph_memory_policy="scoped-reclaim",
    )
    assert not torch.cuda.is_initialized()
    manifest.verify_files(bundle)
    records = [item for item in manifest.generated_sources if item.role == "build_configuration"]
    assert len(records) == 1
    configuration = json.loads((bundle / records[0].path).read_text())
    assert configuration["libtorch_graph_memory_policy"] == "scoped-reclaim"
    assert "-DVLAFORGE_LIBTORCH_SCOPED_GRAPH_RECLAIM=ON" in manifest.reproducibility.build_commands[0]
    bad = tmp_path / "tampered-bundle"
    shutil.copytree(bundle, bad)
    (bad / records[0].path).write_text("{}\n")
    with pytest.raises(ValueError):
        manifest.verify_files(bad)
    (tmp_path / "build-commands.json").write_text(json.dumps(calls, indent=2) + "\n")
    (tmp_path / "evidence.json").write_text(json.dumps({
        "status": "actual_public_parameter_build_passed", "cuda_sdk_compiled": True,
        "gpu_execution": False, "model_validation": False, "metadata_tamper_rejected": True,
        "bundle_manifest_sha256": hashlib.sha256((bundle / "bundle.json").read_bytes()).hexdigest(),
        "implementation_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(build_implementation.__file__), runtime / "CMakeLists.txt",
                         runtime / "backends/libtorch_graph.cpp", runtime / "backends/libtorch_graph_cleanup.h")
        },
    }, indent=2) + "\n")
