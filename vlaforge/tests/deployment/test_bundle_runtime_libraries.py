"""Actual relocated CPU LibTorch bundles, not VLA or GPU numerical evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from vlaforge.codegen.numerical import generate_libtorch_worker_initializer
from vlaforge.compiler import compile_module
from vlaforge.deployment import (
    ArtifactIdentity,
    ArtifactKind,
    ArtifactResidency,
    BackendCapability,
    RegionArtifactContract,
    WorkspaceContract,
    build_artifact_compile_bundle,
)
from vlaforge.deployment.contract import ARTIFACT_SCHEMA, NUMERICAL_ARTIFACT_SCHEMA
from vlaforge.deployment.libtorch_numerical import policy_from_context
from vlaforge.deployment.numerical import (
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalRequirement,
    RegionNumericalBinding,
)
from vlaforge.deployment.torchscript_export import export_torchscript_region
from vlaforge.frontend import InvocationBuilder, capture_region, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType
from vlaforge.numerical_context import snapshot

_PROVIDER = "libvlaforge_libtorch_numerical_backend.so"
_RUNNER = r"""
#include "session_generated.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 4) return 1;
#ifdef VLAFORGE_TEST_NUMERICAL_BOOTSTRAP
  const auto initialized = vlaforge_initialize_numerical_worker();
  if (initialized.code != VLAFORGE_STATUS_OK) return 2;
#endif
  vlaforge_generated::ModelSession session(argv[1]);
  if (!session.initialization_status().ok()) {
    std::cerr << session.initialization_status().message << '\n'; return 3;
  }
  const std::int64_t shape[]{4}, flag_shape[]{1};
  std::uint8_t accepted = 1;
  std::ofstream raw(argv[2], std::ios::binary);
  for (unsigned run = 0; run < 3; ++run) {
    float values[]{1.0f + run, -2.0f + run, 3.0f + run, 0.0f + run};
    float expected[4];
    for (unsigned i = 0; i < 4; ++i) expected[i] = values[i] * 2.0f + 1.0f;
    const VLAForgeBoundTensor input{sizeof(input),
        {values, sizeof(values), shape, 1, VLAFORGE_DTYPE_F32,
         {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 4};
    const VLAForgeBoundTensor flag{sizeof(flag),
        {&accepted, 1, flag_shape, 1, VLAFORGE_DTYPE_BOOL,
         {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1};
    VLAForgeInputStamp stamp{};
    stamp.struct_size = sizeof(stamp);
    stamp.has_revision = 1;
    stamp.revision = run + 1;
    if (!session.BindTensor(0, input, &stamp).ok() ||
        !session.BindTensor(1, flag, &stamp).ok()) return 4;
    const auto ran = session.Run();
    if (!ran.ok()) { std::cerr << ran.message << '\n'; return 5; }
    VLAForgeBoundTensor output{};
    if (!session.ReadOutputTensor(0, &output).ok()) return 6;
    if (output.tensor.size_bytes != sizeof(expected) ||
        output.tensor.rank != 1 || output.tensor.dimensions[0] != 4 ||
        output.tensor.dtype != VLAFORGE_DTYPE_F32 ||
        output.tensor.device.kind != VLAFORGE_DEVICE_CPU ||
        output.layout != VLAFORGE_LAYOUT_CONTIGUOUS ||
        std::memcmp(output.tensor.data, expected, sizeof(expected)) != 0) return 7;
    raw.write(static_cast<const char*>(output.tensor.data), sizeof(expected));
  }
  if (!raw.good()) return 8;
  std::ifstream input("/proc/self/maps");
  const std::string maps((std::istreambuf_iterator<char>(input)), {});
  if (maps.empty() || maps.find("libpython") != std::string::npos) return 9;
  std::ofstream output(argv[3]);
  output << maps;
  if (!output.good()) return 10;
  std::cout << "{\"status\":\"passed\",\"complete_outputs\":3,"
               "\"evidence_level\":\"relocated_cpu_contract_fixture\","
               "\"vla_model_validation\":false}\n";
  return 0;
}
"""


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_nonempty_search_paths(dynamic):
    paths = re.findall(r"\((?:RUNPATH|RPATH)\).*\[([^\]]*)\]", dynamic)
    assert paths and all(all(path.split(":")) for path in paths)


@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_LIBTORCH_NUMERICAL_CPU") != "1",
    reason="opt-in actual CPU LibTorch public-bundle relocation build",
)
@pytest.mark.parametrize("policy_bound", (True, False), ids=("policy", "legacy"))
@pytest.mark.parametrize("backend", ("torchscript", "aoti"))
def test_public_libtorch_bundle_survives_build_cleanup_and_relocation(
    tmp_path, monkeypatch, policy_bound, backend
):
    torch = pytest.importorskip("torch")
    from vlaforge.deployment import build

    runtime_root = Path(__file__).resolve().parents[2]
    implementation_sources = {
        str(path): _sha256(path)
        for path in (Path(__file__), Path(build.__file__), runtime_root / "CMakeLists.txt")
    }
    assert torch.__version__.split("+", 1)[0] == "2.10.0"
    assert not torch.cuda.is_initialized()
    assert shutil.which("readelf") and shutil.which("ldd")

    class Affine(torch.nn.Module):
        def forward(self, value):
            return value * 2 + 1

    vector = TensorType((4,), "f32")

    @tensor_region("affine", inputs=(Value("x", vector),), outputs=(vector,))
    def affine(value):
        return value * 2 + 1

    builder = InvocationBuilder(
        "relocated_cpu_contract_fixture",
        inputs=(InputPort("x", vector), InputPort("accepted", TensorType((1,), "bool"))),
        outputs=(OutputPort("result", vector),),
    )
    (result,) = builder.call(affine, builder.input("x"))
    invocation = builder.finish({"result": result}, accepted=builder.input("accepted"))
    module = invocation.module
    examples = tuple(torch.tensor([1.0, -2.0, 3.0, 0.0]) + run for run in range(3))
    expected = b"".join(Affine()(value).numpy().tobytes() for value in examples)
    captured = capture_region(module.regions[0], Affine(), (examples[0],)).require_supported()
    evidence = captured.evidence
    (tmp_path / "capture.json").write_text(json.dumps(evidence.to_dict(), indent=2) + "\n")
    exported_path = tmp_path / "affine.pt2"
    torch.export.save(captured.exported_program, exported_path)
    observed = snapshot()
    if backend == "torchscript":
        artifact = tmp_path / "affine.pt"
        audit = export_torchscript_region(
            captured.exported_program, artifact,
            validation_cases=tuple((value,) for value in examples[1:]),
        )
        variant = "torchscript-aten/1"
        kind = ArtifactKind.TORCHSCRIPT_ARCHIVE
    else:
        from vlaforge.deployment.aoti_profile import aoti_configs
        artifact = tmp_path / "affine.aoti.pt2"
        torch._inductor.aoti_compile_and_package(captured.exported_program,
            package_path=str(artifact), inductor_configs=aoti_configs("aten-preserving"))
        audit = {"artifact_sha256": _sha256(artifact), "configs": aoti_configs("aten-preserving")}
        variant = "torch-" + str(torch.__version__)
        kind = ArtifactKind.AOTI_PACKAGE
    observed.require_current()
    (tmp_path / "backend-compile.json").write_text(json.dumps(audit, indent=2) + "\n")
    binding = None
    runner = _RUNNER
    if policy_bound:
        policy = policy_from_context(observed)
        record = NumericalCompileRecord(
            backend, "cpu", str(torch.__version__), _sha256(exported_path),
            evidence.graph_digest, _sha256(artifact), artifact.stat().st_size,
            policy_from_context(evidence.observed_numerical_context), policy,
            policy_from_context(snapshot()),
            json.dumps({"backend_variant": variant, "jit_optimization": False},
                       sort_keys=True, separators=(",", ":")),
        )
        binding = RegionNumericalBinding(
            module.regions[0].name,
            NumericalRequirement(policy, "same-precision", policy.digest(), record.digest()),
            record, PROVIDER_REQUIRED,
        )
        runner = (
            "#define VLAFORGE_TEST_NUMERICAL_BOOTSTRAP 1\n"
            + generate_libtorch_worker_initializer(
                (binding,), acknowledge_exclusive_process=True,
                acknowledge_calling_thread=True,
            )
            + runner
        )
    contract = RegionArtifactContract(
        region_id=0, region_name=module.regions[0].name,
        inputs=evidence.inputs, outputs=evidence.outputs,
        io_schema_digest=compile_module(module).certificate.io_schema_digest,
        identity=ArtifactIdentity(module.name, _sha256(__file__), "fixture:no-checkpoint",
                                  evidence.graph_digest),
        artifact_kind=kind,
        artifact_path="artifacts/" + artifact.name, artifact_sha256=_sha256(artifact),
        artifact_size_bytes=artifact.stat().st_size, residency=ArtifactResidency.SESSION,
        workspace=WorkspaceContract(),
        capability=BackendCapability(backend, "cpu", ("f32",), requires_synchronize=True),
        effect_audit=evidence.effect_audit, backend_variant=variant,
        numerical_binding=binding,
        schema=NUMERICAL_ARTIFACT_SCHEMA if policy_bound else ARTIFACT_SCHEMA,
    )
    if backend == "aoti":
        from vlaforge.deployment.aoti_materialized import (
            MaterializedAotiPackage,
            materialize_aoti_package,
            materialized_region_contract,
        )
        payload = materialize_aoti_package(artifact, tmp_path / "published",
            sha256=contract.artifact_sha256, size_bytes=contract.artifact_size_bytes)
        original_contract = contract
        contract = materialized_region_contract(contract, payload, artifact_path="artifacts/affine/model.vfaoti")
        assert contract.identity == original_contract.identity and contract.inputs == original_contract.inputs
        if policy_bound:
            assert contract.numerical_binding.requirement.policy == binding.requirement.policy
            assert contract.numerical_binding.requirement.compile_record_sha256 != binding.requirement.compile_record_sha256
            assert json.loads(contract.numerical_binding.compile_record.configuration_json)["byte_preserving_materialization"]["source_compile_record_sha256"] == binding.compile_record.digest()
            runner = "#define VLAFORGE_TEST_NUMERICAL_BOOTSTRAP 1\n" + generate_libtorch_worker_initializer(
                (contract.numerical_binding,), acknowledge_exclusive_process=True, acknowledge_calling_thread=True) + _RUNNER
        artifact = payload
    bundle = tmp_path / "before-move" / "bundle"
    temporary_directories = []
    real_temporary_directory = build.tempfile.TemporaryDirectory

    def record_real_temporary_directory(*args, **kwargs):
        directory = real_temporary_directory(*args, **kwargs)
        if kwargs.get("prefix") == "vlaforge-artifact-bundle-build-":
            temporary_directories.append(Path(directory.name))
        return directory

    environment = {
        "CUDA_VISIBLE_DEVICES": "", "CMAKE_BUILD_PARALLEL_LEVEL": "2",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
    }
    with monkeypatch.context() as context:
        context.setattr(build.tempfile, "TemporaryDirectory", record_real_temporary_directory)
        manifest = build_artifact_compile_bundle(
            module, bundle, region_artifacts={contract.region_name: contract},
            artifact_sources={contract.region_name: artifact},
            validators=invocation.cpp_validators(), runner_source=runner,
            runtime_root=runtime_root,
            cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={backend: torch.__version__},
            source_revision=_sha256(__file__), source_dirty=True, environment=environment,
            default_device="cpu", state_device="cpu", loop_execution="off",
        )
    assert temporary_directories and all(not path.exists() for path in temporary_directories)
    manifest.verify_files(bundle)
    records = [record for record in manifest.binaries if record.role == "runtime_shared_library"]
    assert records and any(Path(record.path).name == _PROVIDER for record in records)
    assert all(record.path.startswith("lib/libvlaforge_") for record in records)
    assert {path.name for path in (bundle / "lib").iterdir()} == {
        Path(record.path).name for record in records
    }

    relocated = tmp_path / "different-absolute-parent" / "relocated-bundle"
    relocated.parent.mkdir()
    shutil.move(str(bundle), str(relocated))
    assert not bundle.exists()
    manifest.verify_files(relocated)
    if backend == "aoti":
        materialized = MaterializedAotiPackage.parse(artifact.read_text())
        materialized.verify((relocated / contract.artifact_path).parent)
        # Removing the original publication proves the runtime uses only delivered files.
        shutil.rmtree(artifact.parent)
    clean_environment = {**os.environ, **environment}
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"):
        clean_environment.pop(name, None)
    command_log = []

    def execute(command):
        completed = subprocess.run(
            [str(item) for item in command], env=clean_environment, cwd=tmp_path,
            capture_output=True, text=True, check=False, timeout=60,
        )
        command_log.append({"command": [str(item) for item in command],
                            "returncode": completed.returncode,
                            "stdout": completed.stdout, "stderr": completed.stderr})
        (tmp_path / "commands.json").write_text(json.dumps(command_log, indent=2) + "\n")
        return completed

    binary = relocated / "bin/vlaforge_generated_runner"
    dynamic = execute(("readelf", "-d", binary))
    assert dynamic.returncode == 0
    _require_nonempty_search_paths(dynamic.stdout)
    assert "$ORIGIN/../lib" in dynamic.stdout
    assert all(str(path) not in dynamic.stdout for path in temporary_directories)
    for record in records:
        library_dynamic = execute(("readelf", "-d", relocated / record.path))
        assert library_dynamic.returncode == 0
        _require_nonempty_search_paths(library_dynamic.stdout)
        assert all(str(path) not in library_dynamic.stdout for path in temporary_directories)
    linked = execute(("ldd", binary))
    assert linked.returncode == 0 and "not found" not in linked.stdout
    assert "libpython" not in linked.stdout.lower()
    raw, maps = tmp_path / "outputs.f32", tmp_path / "process-maps.txt"
    native_command = (binary, relocated, raw, maps)
    native = execute(native_command)
    assert native.returncode == 0, native.stdout + native.stderr
    assert json.loads(native.stdout)["complete_outputs"] == 3
    assert raw.read_bytes() == expected and len(expected) == 48
    mapped = maps.read_text()
    provider = relocated / "lib" / _PROVIDER
    provider_mappings = {
        line.split()[-1] for line in mapped.splitlines() if _PROVIDER in line
    }
    assert provider_mappings == {str(provider.resolve())}
    assert str(bundle) not in mapped and "libpython" not in mapped.lower()
    assert all(str(path) not in mapped for path in temporary_directories)

    if backend == "aoti":
        member = (relocated / contract.artifact_path).parent / materialized.library
        original_member = member.read_bytes()
        try:
            member.write_bytes(b"BAD!" + original_member[4:])
            raw.unlink()
            maps.unlink()
            rejected = execute(native_command)
            assert rejected.returncode != 0 and not raw.exists() and not maps.exists()
        finally:
            member.write_bytes(original_member)
        manifest.verify_files(relocated)

    original_library = provider.read_bytes()
    for corruption in ("missing", "invalid-elf"):
        try:
            if corruption == "missing":
                provider.unlink()
            else:
                provider.write_bytes(b"BAD!" + original_library[4:])
            with pytest.raises((ValueError, FileNotFoundError)):
                manifest.verify_files(relocated)
            raw.unlink(missing_ok=True)
            maps.unlink(missing_ok=True)
            rejected = execute(native_command)
            assert rejected.returncode != 0
            assert _PROVIDER in rejected.stderr
            assert not raw.exists() and not maps.exists()
            assert '"status":"passed"' not in rejected.stdout
        finally:
            provider.write_bytes(original_library)
        manifest.verify_files(relocated)
    restored = execute(native_command)
    assert restored.returncode == 0 and raw.read_bytes() == expected
    assert all(_sha256(path) == digest for path, digest in implementation_sources.items())
    (tmp_path / "relocation-evidence.json").write_text(json.dumps({
        "status": "passed", "evidence_level": "actual_cpu_contract_fixture",
        "policy_bound": policy_bound, "vla_model_validation": False,
        "backend": backend,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "deleted_build_directories": [str(path) for path in temporary_directories],
        "moved_from": str(bundle), "moved_to": str(relocated),
        "bundle_manifest_sha256": _sha256(relocated / "bundle.json"),
        "runtime_libraries": [record.to_dict() for record in records],
        "implementation_sources_sha256": implementation_sources,
        "full_outputs_sha256": _sha256(raw), "process_maps_sha256": _sha256(maps),
        "native_environment_removed": ["PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"],
        "vendor_torch_and_cuda_are_external_sdk_dependencies": True,
    }, indent=2) + "\n")
    assert not torch.cuda.is_initialized()
