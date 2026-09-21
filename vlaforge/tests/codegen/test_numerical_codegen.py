from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from vlaforge.codegen import CppArtifactRegionDefinition, generate_compiled_cpp_session
from vlaforge.codegen.cpp import CodegenUnsupportedError
from vlaforge.compiler import NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA, compile_module
from vlaforge.deployment.numerical import (
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalContractError,
    NumericalPolicy,
    NumericalRequirement,
    RegionNumericalBinding,
)
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType


def _program():
    vector = TensorType((2,), "f32")

    @tensor_region("transform", inputs=(Value("x", vector),), outputs=(vector,))
    def transform(x):
        return x

    @tensor_region("z_project", inputs=(Value("x", vector),), outputs=(vector,))
    def project(x):
        return x

    builder = InvocationBuilder(
        "numerical_test",
        inputs=(
            InputPort("x", vector),
            InputPort("accepted", TensorType((1,), "bool")),
        ),
        outputs=(OutputPort("result", vector),),
    )
    (middle,) = builder.call(transform, builder.input("x"), cache=True)
    (output,) = builder.call(project, middle)
    return builder.finish({"result": output}, accepted=builder.input("accepted"))


def _binding(
    name,
    digest,
    size,
    *,
    precision=1,
    namespace="test.numerics/1",
    backend="shared_plugin",
):
    policy = NumericalPolicy(
        namespace,
        (
            ("epsilon", 0.125),
            ("mode", "strict"),
            ("precision", precision),
            ("strict", True),
        ),
    )
    record = NumericalCompileRecord(
        backend,
        "cpu",
        "test-compiler/1",
        "1" * 64,
        "2" * 64,
        digest,
        size,
        policy,
        policy,
        policy,
        '{"fixture":true}',
    )
    return RegionNumericalBinding(
        name,
        NumericalRequirement(
            policy,
            "same-precision",
            policy.digest(),
            record.digest(),
        ),
        record,
        runtime_enforcement=PROVIDER_REQUIRED,
    )


def _sources(digest="a" * 64, size=1, *, variant="normal", runner=None):
    program = _program()
    compilation = compile_module(program.module)
    definitions = {}
    for index, region in enumerate(compilation.module.regions):
        binding = _binding(
            region.name,
            digest,
            size,
            precision=2 if variant == "conflict" and index == 1 else 1,
            namespace="unknown.numerics/1"
            if variant == "namespace"
            else "test.numerics/1",
        )
        definitions[region.name] = CppArtifactRegionDefinition(
            region.name,
            "shared_plugin",
            "plugin.so",
            digest,
            size,
            compilation.plan.io_schema_digest,
            "cpu",
            "cpu",
            "shared-plugin/1",
            numerical_binding=None if variant == "legacy" else binding,
        )
    bindings = tuple(
        definitions[name].numerical_binding
        for name in sorted(definitions)
        if definitions[name].numerical_binding is not None
    )
    if bindings:
        compilation = replace(
            compilation,
            certificate=replace(
                compilation.certificate,
                schema=NUMERICAL_COMPILATION_CERTIFICATE_SCHEMA,
                numerical_bindings=bindings,
            ),
        )
    return (
        generate_compiled_cpp_session(
            compilation,
            artifact_regions=definitions,
            validators=program.cpp_validators(),
            runner_source=runner,
        ),
        compilation,
        definitions,
    )


def test_numerical_transport_is_explicit_and_two_phase():
    sources, compilation, definitions = _sources()
    source = sources.as_dict()["session_generated.cpp"]
    initialization = source.split("ModelSession::InitializeRegions(", 1)[1]
    assert initialization.index("kArtifactPath1") < initialization.index("plugin_open")
    assert initialization.index(
        "numerical_leases_.Add(\n        1u"
    ) < initialization.index("AcquireAll()")
    assert initialization.index("AcquireAll()") < initialization.index("LoadRegion(0u)")
    run = source.split("ModelSession::Run() noexcept", 1)[1]
    assert run.index("VLAFORGE_NUMERICAL_RUN_ENTRY") < run.index("PrepareInputs()")
    assert run.index("VLAFORGE_NUMERICAL_BEFORE_COMMIT") < run.index(
        "state_store_.Commit"
    )
    assert (
        "cache_"
        in source.split("ModelSession::FailNumerical", 1)[1].split(
            "ModelSession::CheckNumerical", 1
        )[0]
    )
    assert "VLAFORGE_NUMERICAL_F64" in source and "VLAFORGE_NUMERICAL_BOOL" in source
    for definition in definitions.values():
        assert definition.numerical_binding.requirement.digest() in source
    with pytest.raises(CodegenUnsupportedError, match="bindings differ"):
        generate_compiled_cpp_session(
            replace(
                compilation, certificate=compile_module(_program().module).certificate
            ),
            artifact_regions=definitions,
            validators={},
        )
    name = next(iter(definitions))
    with pytest.raises(NumericalContractError, match="unsupported.*LibTorch"):
        replace(
            definitions[name],
            backend="aoti",
            backend_variant="aoti-test",
            numerical_binding=_binding(name, "a" * 64, 1, backend="aoti"),
        )
    with pytest.raises(ValueError, match="Session-resident"):
        replace(definitions[name], residency="invocation")


@pytest.mark.parametrize(
    "backend,variant", [("torchscript", "torchscript-aten/1"), ("aoti", "aoti/1")]
)
def test_builtin_libtorch_provider_is_bound_before_region_load(backend, variant):
    from vlaforge.deployment.libtorch_numerical import NAMESPACE, REDUCTION_API
    from vlaforge.numerical_context import NumericalContext

    # This table checks code generation only, not observed model execution.
    booleans = {
        name: False
        for name in NumericalContext.__dataclass_fields__
        if name
        not in {
            "schema",
            "torch_version",
            "reduction_api",
            "float32_matmul_precision",
            "autocast_cpu_dtype",
            "autocast_cuda_dtype",
        }
    }
    policy = NumericalPolicy(
        NAMESPACE,
        tuple(
            sorted(
                {
                    **booleans,
                    "float32_matmul_precision": "highest",
                    "autocast_cpu_dtype": "bfloat16",
                    "autocast_cuda_dtype": "float16",
                    "torch_release": "2.10.0",
                    "reduction_api": REDUCTION_API,
                }.items()
            )
        ),
    )
    _, compilation, definitions = _sources()
    for name, definition in tuple(definitions.items()):
        record = replace(
            definition.numerical_binding.compile_record,
            backend=backend,
            reference_policy=policy,
            requested_compile_policy=policy,
            observed_compile_policy=policy,
        )
        binding = RegionNumericalBinding(
            name,
            NumericalRequirement(
                policy, "same-precision", policy.digest(), record.digest()
            ),
            record,
            PROVIDER_REQUIRED,
        )
        definitions[name] = replace(
            definition,
            backend=backend,
            backend_variant=variant,
            numerical_binding=binding,
        )
    compilation = replace(
        compilation,
        certificate=replace(
            compilation.certificate,
            numerical_bindings=tuple(
                definitions[name].numerical_binding for name in sorted(definitions)
            ),
        ),
    )
    sources = generate_compiled_cpp_session(
        compilation, artifact_regions=definitions, validators=_program().cpp_validators()
    ).as_dict()
    code = sources["session_generated.cpp"]
    assert '#include "vlaforge/backends/libtorch_numerical.h"' in code
    initialize = code.split("ModelSession::InitializeRegions(", 1)[1]
    assert initialize.count("vlaforge_libtorch_numerical_provider_api()") == 2
    assert "vlaforge_external_region_plugin_numerical_provider" not in initialize
    assert (
        initialize.index("numerical_leases_.Add")
        < initialize.index("AcquireAll()")
        < initialize.index("LoadRegion(0u)")
    )
    assert "setFloat32" not in code and "setAllow" not in code


_RUNNER = r"""
#include "session_generated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <thread>

using vlaforge_generated::ModelSession;
using Setter = void(*)(int);
unsigned commits = 0, aborts = 0;
void Trace(void*, const vlaforge::runtime::TraceEvent* event) {
  if (event->kind == vlaforge::runtime::TraceKind::kOutputGroupCommit) { ++commits; }
  if (event->kind == vlaforge::runtime::TraceKind::kTransactionAbort) { ++aborts; }
}
bool Bind(ModelSession& session, unsigned revision, float* data) {
  static const std::int64_t shape[]{2}, flag_shape[]{1};
  static std::uint8_t accepted = 1;
  const VLAForgeBoundTensor input{sizeof(input),
      {data, 8, shape, 1, VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 4};
  const VLAForgeBoundTensor flag{sizeof(flag),
      {&accepted, 1, flag_shape, 1, VLAFORGE_DTYPE_BOOL, {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1};
  VLAForgeInputStamp stamp{};
  stamp.struct_size = sizeof(stamp);
  stamp.has_revision = 1;
  stamp.revision = revision;
  return session.BindTensor(0, input, &stamp).ok() && session.BindTensor(1, flag, &stamp).ok();
}
bool Output(ModelSession& session, float expected) {
  VLAForgeBoundTensor output{};
  if (!session.ReadOutputTensor(0, &output).ok()) { return false; }
  auto* values = static_cast<const float*>(output.tensor.data);
  return values[0] == expected && values[1] == expected + 1;
}

int main(int argc, char** argv) {
  if (argc != 4) { return 1; }
  const char* mode = argv[2];
  setenv("VLAFORGE_NUMERICAL_TEST_MODE", mode, 1);
  ModelSession session(argv[1]);
  if (std::strcmp(argv[3], "initfail") == 0) {
    std::fprintf(stdout, "initialization_ok=%d\n", session.initialization_status().ok());
    return session.initialization_status().ok() ? 2 : 0;
  }
  if (!session.initialization_status().ok()) {
    std::fprintf(stderr, "initialization: %s\n", session.initialization_status().message);
    return 3;
  }
  session.SetTraceSink({nullptr, Trace});
  float data[]{1, 2};
  if (!Bind(session, 1, data) || !session.Run().ok() || !Output(session, 4) || commits != 1) { return 4; }
  const auto path = std::string(argv[1]) + "/plugin.so";
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) { return 5; }
  auto set_current = reinterpret_cast<Setter>(dlsym(handle, "numerical_test_current"));
  auto set_thread = reinterpret_cast<Setter>(dlsym(handle, "numerical_test_thread"));
  if (!set_current || !set_thread) { return 6; }
  if (std::strcmp(mode, "multisession") == 0) {
    {
      ModelSession other(argv[1]);
      if (!other.initialization_status().ok() || !Bind(other, 1, data) || !other.Run().ok()) { return 7; }
      set_current(2);
      { ModelSession conflicting(argv[1]); if (conflicting.initialization_status().ok()) { return 8; } }
      set_current(1);
    }
    if (!Bind(session, 2, data) || !session.Run().ok() || !Output(session, 4)) { return 9; }
  } else if (std::strcmp(argv[3], "runfail") == 0) {
    data[0] = 11; data[1] = 12;
    if (std::strcmp(mode, "entrydrift") == 0) { set_current(2); }
    if (!Bind(session, 2, data)) { return 10; }
    bool failed = false;
    if (std::strcmp(mode, "threaddrift") == 0) {
      std::thread worker([&] { set_thread(0); failed = !session.Run().ok(); });
      worker.join();
    } else { failed = !session.Run().ok(); }
    if (!failed || commits != 1 || !Output(session, 4)) { return 11; }
    const bool before_begin = std::strcmp(mode, "entrydrift") == 0 || std::strcmp(mode, "threaddrift") == 0;
    if (aborts != (before_begin ? 0u : 1u)) { return 14; }
    set_current(1);
    // A mismatch cannot be healed by external restoration or stale cache reuse.
    (void)Bind(session, 2, data);
    if (session.Run().ok() || !Output(session, 4)) { return 12; }
  } else {
    if (!Bind(session, 1, data) || !session.Run().ok() || !Output(session, 4) || commits != 2) { return 13; }
  }
  dlclose(handle);
  std::puts("complete_outputs_and_transaction_checks=passed");
  return 0;
}
"""


def _command(command, folder, **kwargs):
    result = subprocess.run(
        command, check=False, capture_output=True, text=True, **kwargs
    )
    with (folder / "commands.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "argv": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            + "\n"
        )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def compiled_cpu_sessions(tmp_path_factory):
    root = tmp_path_factory.mktemp("numerical-cpu-sessions")
    runtime = Path(__file__).resolve().parents[2]
    result = {}
    for variant in ("normal", "conflict", "namespace", "missing", "legacy"):
        folder = root / variant
        bundle = folder / "bundle"
        bundle.mkdir(parents=True)
        plugin = bundle / "plugin.so"
        command = [
            "c++",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{runtime / 'include'}",
            str(runtime / "tests/cpp/numerical_plugin_fixture.cpp"),
            "-o",
            str(plugin),
        ]
        if variant in ("missing", "legacy"):
            command.append("-DVLAFORGE_NUMERICAL_TEST_LEGACY=1")
        _command(command, folder)
        sources, _, _ = _sources(
            hashlib.sha256(plugin.read_bytes()).hexdigest(),
            plugin.stat().st_size,
            variant=variant,
            runner=_RUNNER,
        )
        generated, build = folder / "generated", folder / "build"
        sources.write(generated)
        _command(
            [
                "cmake",
                "-S",
                str(generated),
                "-B",
                str(build),
                f"-DVLAFORGE_RUNTIME_ROOT={runtime}",
                "-DBUILD_TESTING=OFF",
            ],
            folder,
        )
        _command(["cmake", "--build", str(build), "-j", "2"], folder)
        result[variant] = (folder, bundle, build / "vlaforge_generated_runner")
    return result


@pytest.mark.parametrize(
    "variant,mode,expect",
    (
        ("normal", "normal", "pass"),
        ("normal", "multisession", "pass"),
        *(
            ("normal", mode, "initfail")
            for mode in (
                "badabi",
                "nullprovider",
                "throwprovider",
                "queryfail",
                "acquirefail",
                "createfail",
                "bindfail",
                "loadfail",
                "driftload",
            )
        ),
        *(
            ("normal", mode, "runfail")
            for mode in (
                "entrydrift",
                "threaddrift",
                "exitdrift",
                "draindrift",
                "drainfail",
            )
        ),
        ("conflict", "normal", "initfail"),
        ("namespace", "normal", "initfail"),
        ("missing", "normal", "initfail"),
        ("legacy", "normal", "pass"),
    ),
)
def test_real_generated_cpu_session_lifecycle(
    compiled_cpu_sessions, variant, mode, expect
):
    folder, bundle, binary = compiled_cpu_sessions[variant]
    log = folder / f"{mode}.log"
    _command(
        [str(binary), str(bundle), mode, expect],
        folder,
        env={
            **os.environ,
            "VLAFORGE_NUMERICAL_TEST_LOG": str(log),
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        },
    )
    events = log.read_text().splitlines() if log.exists() else []
    assert not any(item.startswith("release_before_destroy") for item in events)
    if variant in ("conflict", "namespace", "missing") or mode in (
        "badabi",
        "nullprovider",
        "throwprovider",
        "queryfail",
        "acquirefail",
    ):
        assert not any(item.startswith(("create", "load", "run")) for item in events)
    if mode == "acquirefail":
        assert events[-1] == "release,0"
    if mode == "normal" and variant == "normal":
        assert events.index("acquire,1") < events.index("create,0")
        assert events.index("validate_preload,1") < events.index("load,0")
        assert (
            events.index("destroy,0")
            < events.index("release,1")
            < events.index("release,0")
        )
        assert (
            sum(event == "run,0" for event in events) == 1
        )  # Second call uses exact cache.
    if mode in ("entrydrift", "threaddrift"):
        assert sum(event == "run,1" for event in events) == 1
    if mode in ("exitdrift", "draindrift"):
        assert sum(event == "run,1" for event in events) == 2
    if mode == "drainfail":
        assert not any(event.startswith(("release", "destroy")) for event in events)
    if variant == "legacy":
        assert not any(
            event.startswith(("query", "acquire", "validate", "release"))
            for event in events
        )


def test_provider_contract_builds_and_runs_strict_no_python_bundle(
    tmp_path, compiled_cpu_sessions
):
    import sys

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
    from vlaforge.deployment.bundle import NUMERICAL_BUNDLE_SCHEMA, load_bundle_manifest
    from vlaforge.deployment.contract import NUMERICAL_ARTIFACT_SCHEMA

    _, original, _ = compiled_cpu_sessions["normal"]
    plugin = original / "plugin.so"
    digest, size = (
        hashlib.sha256(plugin.read_bytes()).hexdigest(),
        plugin.stat().st_size,
    )
    program = _program()
    compilation = compile_module(program.module)
    contracts = {}
    for index, region in enumerate(compilation.module.regions):
        contracts[region.name] = RegionArtifactContract(
            region_id=index,
            region_name=region.name,
            inputs=tuple(
                ValueContract.from_ir(value.name, value.type, device="cpu")
                for value in region.inputs
            ),
            outputs=tuple(
                ValueContract.from_ir(f"out_{i}", value, device="cpu")
                for i, value in enumerate(region.outputs)
            ),
            io_schema_digest=compilation.plan.io_schema_digest,
            identity=ArtifactIdentity(
                "numerical-cpu-fixture", "test", "fixture:no-checkpoint", "2" * 64
            ),
            artifact_kind=ArtifactKind.SHARED_LIBRARY,
            artifact_path=f"artifacts/{index}.so",
            artifact_sha256=digest,
            artifact_size_bytes=size,
            workspace=WorkspaceContract(),
            capability=BackendCapability("shared_plugin", "cpu", ("f32",)),
            effect_audit=EffectAudit(),
            backend_variant="shared-plugin/1",
            schema=NUMERICAL_ARTIFACT_SCHEMA,
            numerical_binding=_binding(region.name, digest, size),
        )
    root = tmp_path / "bundle"
    manifest = build_artifact_compile_bundle(
        compilation.module,
        root,
        region_artifacts=contracts,
        artifact_sources={name: plugin for name in contracts},
        validators=program.cpp_validators(),
        runner_source=_RUNNER.replace('"/plugin.so"', '"/artifacts/0.so"'),
        runtime_root=Path(__file__).resolve().parents[2],
        cmake_prefix_path=sys.prefix,
        backend_versions={"shared_plugin": "numerical-fixture/1"},
        source_revision="test-only",
        source_dirty=True,
    )
    assert manifest.schema == NUMERICAL_BUNDLE_SCHEMA
    restored = load_bundle_manifest(root / "bundle.json")
    assert restored.digest() == manifest.digest()
    restored.verify_files(root)
    restored.require_runtime_deployable()  # Eligibility only; provider accepts at actual load.
    binary = root / "bin/vlaforge_generated_runner"
    _command(
        [str(binary), str(root), "bundle", "pass"],
        tmp_path,
        env={
            **os.environ,
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
            "VLAFORGE_NUMERICAL_TEST_LOG": str(tmp_path / "bundle.log"),
        },
    )
    linked = _command(["ldd", str(binary)], tmp_path).stdout
    assert "libpython" not in linked and "libtorch" not in linked
