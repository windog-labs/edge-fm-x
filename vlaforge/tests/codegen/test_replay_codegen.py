from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from replay_cases import make_case
from vlaforge.codegen import (
    ZERO_STATE,
    CppArtifactRegionDefinition,
    CppRegionDefinition,
    generate_compiled_cpp_session,
)
from vlaforge.codegen.cpp import CodegenUnsupportedError
from vlaforge.compiler import compile_module
from vlaforge.deployment.capabilities import aoti_backend_capability
from vlaforge.deployment.contract import BackendCapability
from vlaforge.frontend import capture_region

ROOT = Path(__file__).resolve().parents[2]


def capture_cpu_case(kind, mode):
    program, examples, _, _, _ = make_case(kind, mode, "cpu")
    captures = {
        region.name: capture_region(
            region, program.regions[region.name], examples[region.name]
        ).require_supported()
        for region in program.module.regions
    }
    semantic = replace(
        program.module,
        inputs=tuple(
            replace(port, device="cuda:0") if port.name != "accepted" else port
            for port in program.module.inputs
        ),
        outputs=tuple(
            replace(port, device="cuda:0") for port in program.module.outputs
        ),
    )
    compilation = compile_module(
        semantic, default_device="cuda:0", state_device="cuda:0"
    )
    definitions = {
        region.name: CppArtifactRegionDefinition(
            region_name=region.name,
            backend="aoti",
            artifact_path=f"artifacts/{region.name}.pt2",
            artifact_sha256="0" * 64,
            artifact_size_bytes=1,
            io_schema_digest=compilation.certificate.io_schema_digest,
            target="sm_86",
            device="cuda:0",
            supports_external_cuda_graph=True,
            effect_audit=captures[region.name].evidence.effect_audit,
        )
        for region in semantic.regions
    }
    return program, compilation, definitions


@pytest.mark.parametrize("kind", ["continuous", "autoregressive"])
def test_generator_owns_replay_callbacks_and_uses_real_effect_audit(kind):
    program, compilation, definitions = capture_cpu_case(kind, "required")
    sources = generate_compiled_cpp_session(
        compilation,
        artifact_regions=definitions,
        validators=program.cpp_validators(),
        initial_state={state.name: ZERO_STATE for state in compilation.module.states},
    ).as_dict()
    report = json.loads(sources["replay_analysis.json"])
    assert report["loops"][0]["candidate"]
    source = sources["session_generated.cpp"]
    loop = next(task for task in compilation.plan.tasks if task.opcode == "vla.for")
    callback = source.split(f"VLAForgeStatus ModelSession::StepReplay{loop.id}(", 1)[
        1
    ].split("vlaforge::runtime::Status ModelSession::RunReplay", 1)[0]
    assert "->run(" in callback and "vlaforge_execution_context_copy(" in callback
    for forbidden in (
        "synchronize(",
        "bind_input",
        "bind_output",
        "CopyBytes",
        "EmitTrace",
        "Stage(",
        "Validate",
        "ReadBool",
    ):
        assert forbidden not in callback
    assert "vlaforge_bounded_replay_create" in source
    destructor = source.split("void ModelSession::DestroyRegions()", 1)[1]
    assert destructor.index("DestroyReplays();") < destructor.index(
        "DestroyRegion(index - 1u)"
    )
    assert "arena_.Abandon();" in destructor
    assert "DestroyReplays();" in source.split("ModelSession::ResetEpisode(", 1)[1]
    assert "vlaforge::runtime::TraceKind::kRegion" not in callback


@pytest.mark.parametrize(
    "change",
    [
        {"supports_external_cuda_graph": False},
        {"effect_audit": None},
        {"residency": "invocation"},
        {"device": "cuda:1"},
    ],
)
def test_required_is_fail_closed_without_a_usable_backend_contract(change):
    program, compilation, definitions = capture_cpu_case("continuous", "required")
    definitions["tensor_update"] = replace(definitions["tensor_update"], **change)
    with pytest.raises(CodegenUnsupportedError, match="requires replay"):
        generate_compiled_cpp_session(
            compilation,
            artifact_regions=definitions,
            validators=program.cpp_validators(),
        )


def test_capability_defaults_preserve_legacy_serialization():
    legacy = BackendCapability("aoti", "sm_86", ("f32",))
    assert "supports_external_cuda_graph" not in legacy.to_dict()
    assert not BackendCapability.from_dict(
        legacy.to_dict()
    ).supports_external_cuda_graph
    capability = aoti_backend_capability("sm_86", ("f32", "i64"))
    assert BackendCapability.from_dict(capability.to_dict()) == capability
    assert capability.supports_external_cuda_graph
    assert not aoti_backend_capability("cpu", ("f32",)).supports_external_cuda_graph


def test_cpu_prefer_generates_and_runs_explicit_ordinary_fallback(tmp_path):
    program, _, _, _, _ = make_case("continuous", "prefer", "cpu")
    compilation = compile_module(program.module)
    loop = next(task for task in compilation.plan.tasks if task.opcode == "vla.for")
    regions = {
        "tensor_transform": CppRegionDefinition(
            "tensor_transform",
            """
for (int i = 0; i < 6; ++i) {
  Output<float>(executable, 0)[i] = std::sin(Input<float>(executable, 0)[i]) +
      Input<float>(executable, 1)[i] * 0.1f + static_cast<float>(*Input<std::int64_t>(executable, 2)) * 0.01f;
}
return vlaforge_status_ok();""",
        ),
        "tensor_update": CppRegionDefinition(
            "tensor_update",
            """
for (int i = 0; i < 6; ++i) {
  Output<float>(executable, 0)[i] = Input<float>(executable, 0)[i] - Input<float>(executable, 1)[i] * 0.2f;
}
*Output<std::int64_t>(executable, 1) = *Input<std::int64_t>(executable, 2) + 1;
return vlaforge_status_ok();""",
        ),
    }
    runner = f"""#include "session_generated.h"
#include <cstring>
int main() {{
  vlaforge_generated::ModelSession session;
  VLAForgeBoundedReplayInfo info{{}};
  info.struct_size = sizeof(info);
  if (session.GetReplayInfo({loop.id}u, &info).code != VLAFORGE_STATUS_OK ||
      info.state != VLAFORGE_REPLAY_FALLBACK || std::strstr(info.reason, "requires_cuda") == nullptr) {{ return 1; }}
  float seed[6]{{}}, condition[6]{{}};
  std::int64_t position[1]{{0}}, shape[2]{{2, 3}}, one[1]{{1}};
  std::uint8_t accepted[1]{{1}};
  VLAForgeBoundTensor bindings[] = {{
      {{sizeof(VLAForgeBoundTensor), {{seed, sizeof(seed), shape, 2, VLAFORGE_DTYPE_F32, {{VLAFORGE_DEVICE_CPU, 0}}}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1}},
      {{sizeof(VLAForgeBoundTensor), {{position, sizeof(position), one, 1, VLAFORGE_DTYPE_I64, {{VLAFORGE_DEVICE_CPU, 0}}}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1}},
      {{sizeof(VLAForgeBoundTensor), {{condition, sizeof(condition), shape, 2, VLAFORGE_DTYPE_F32, {{VLAFORGE_DEVICE_CPU, 0}}}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1}},
      {{sizeof(VLAForgeBoundTensor), {{accepted, sizeof(accepted), one, 1, VLAFORGE_DTYPE_BOOL, {{VLAFORGE_DEVICE_CPU, 0}}}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1}}
  }};
  for (std::uint32_t index = 0; index < 4; ++index) {{
    if (!session.BindTensor(index, bindings[index], nullptr).ok()) {{ return 2; }}
  }}
  if (!session.Run().ok()) {{ return 3; }}
  return 0;
}}
"""
    sources = generate_compiled_cpp_session(
        compilation,
        regions=regions,
        validators=program.cpp_validators(),
        runner_source=runner,
    )
    assert "replay.requires_cuda" in sources.as_dict()["replay_analysis.json"]
    sources.write(tmp_path / "source")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(tmp_path / "source"),
            "-B",
            str(tmp_path / "build"),
            f"-DVLAFORGE_RUNTIME_ROOT={ROOT}",
            "-DBUILD_TESTING=OFF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(tmp_path / "build"), "-j", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(tmp_path / "build/vlaforge_generated_runner")], check=True)


def cuda_runner(program, steps, loop_id, *, replay):
    shapes = []
    bindings = []
    for port in program.module.inputs:
        index = port.input_id
        dimensions = ",".join(str(size) for size in port.payload.shape)
        shapes.append(f"const std::int64_t shape_{index}[] = {{{dimensions}}};")
        dtype = {"f32": "F32", "i64": "I64", "bool": "BOOL"}[port.payload.dtype]
        device = "CPU" if port.device == "cpu" else "CUDA"
        bindings.append(f"""
    held.emplace_back(std::make_unique<Buffer>(Load(pack + "/input_" + std::to_string(run) + "_{port.name}.bin"), {str(port.device != "cpu").lower()}));
    auto& input_{index} = *held.back();
    VLAForgeBoundTensor binding_{index}{{sizeof(VLAForgeBoundTensor),
        {{input_{index}.data(), input_{index}.host.size(), shape_{index}, {len(port.payload.shape)},
          VLAFORGE_DTYPE_{dtype}, {{VLAFORGE_DEVICE_{device}, 0}}}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1}};
    Check(session->BindTensor({index}u, binding_{index}, nullptr));
""")
    query = (
        f"""
    VLAForgeBoundedReplayInfo info{{}};
    info.struct_size = sizeof(info);
    CCheck(session->GetReplayInfo({loop_id}u, &info));
    Require(info.state == VLAFORGE_REPLAY_READY && info.captured_steps == {steps}u &&
            info.replay_count == static_cast<std::uint64_t>(run == 3 ? 1 : run + 1), "unexpected replay counters");
    std::printf("replay run=%d captured_steps=%u replay_count=%llu\\n", run, info.captured_steps,
                static_cast<unsigned long long>(info.replay_count));
"""
        if replay
        else ""
    )
    fatal = (
        f"""
    if (inject_failure) {{
      VLAForgeBoundedReplayInfo info{{}};
      info.struct_size = sizeof(info);
      CCheck(session->GetReplayInfo({loop_id}u, &info));
      VLAForgeBoundTensor unused{{}};
      Require(!status.ok() && info.state == VLAFORGE_REPLAY_POISONED && info.ordinary_count == 0 &&
              !session->Run().ok() && !session->ResetEpisode(1).ok() &&
              !session->ReadOutputTensor(0, &unused).ok(), "poisoned Session was reusable");
      session.reset();
      std::puts("GENERATED_REPLAY fatal Session quarantine passed");
      std::fflush(stdout);
      std::_Exit(78);
    }}
"""
        if replay
        else ""
    )
    return (
        r"""#include "session_generated.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
bool inject_failure = false;
void Require(bool value, const char* message) { if (!value) { throw std::runtime_error(message); } }
void Check(vlaforge::runtime::Status value) { Require(value.ok(), value.message); }
void CCheck(VLAForgeStatus value) { Require(value.code == VLAFORGE_STATUS_OK, value.message); }
void Cuda(cudaError_t value) { Require(value == cudaSuccess, cudaGetErrorString(value)); }
std::vector<char> Load(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  Require(file.good(), "missing tensor pack file");
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
struct Buffer {
  std::vector<char> host;
  void* device = nullptr;
  Buffer(std::vector<char> value, bool cuda) : host(std::move(value)) {
    if (cuda) { Cuda(cudaMalloc(&device, host.size())); Cuda(cudaMemcpy(device, host.data(), host.size(), cudaMemcpyHostToDevice)); Cuda(cudaStreamSynchronize(nullptr)); }
  }
  ~Buffer() { if (device) { (void)cudaFree(device); } }
  void* data() { return device ? device : host.data(); }
};
extern "C" VLAForgeStatus __real_vlaforge_execution_context_copy(
    VLAForgeExecutionContext*, const VLAForgeTensorView*, const VLAForgeTensorView*, std::uint64_t);
extern "C" VLAForgeStatus __wrap_vlaforge_execution_context_copy(
    VLAForgeExecutionContext* context, const VLAForgeTensorView* dst,
    const VLAForgeTensorView* src, std::uint64_t size) {
  if (inject_failure) {
    VLAForgeExecutionContextView view{}; view.struct_size = sizeof(view);
    CCheck(vlaforge_execution_context_get_view(context, &view));
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    Cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(view.native_stream), &capture));
    if (capture == cudaStreamCaptureStatusActive) {
      Require(cudaStreamSynchronize(static_cast<cudaStream_t>(view.native_stream)) != cudaSuccess, "expected capture invalidation");
      return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, "test capture invalidation");
    }
  }
  return __real_vlaforge_execution_context_copy(context, dst, src, size);
}
int main(int argc, char** argv) {
  try {
    Require(argc == 4, "usage: runner BUNDLE PACK success|fatal");
    inject_failure = std::string(argv[3]) == "fatal";
    const std::string pack(argv[2]);
    auto session = std::make_unique<vlaforge_generated::ModelSession>(argv[1]);
    Check(session->initialization_status());
    std::vector<std::unique_ptr<Buffer>> held;
"""
        + "\n".join(shapes)
        + r"""
    for (int run = 0; run < 4; ++run) {
      if (run == 3) { Check(session->ResetEpisode(1)); }
"""
        + "".join(bindings)
        + r"""
      const auto status = session->Run();
"""
        + fatal
        + r"""
      Require(status.ok() == (run != 1), "validator/transaction result mismatch");
      VLAForgeBoundTensor output{};
      Check(session->ReadOutputTensor(0, &output));
      const auto bytes = Load(pack + "/expected_" + std::to_string(run) + ".bin");
      Require(bytes.size() == output.tensor.size_bytes, "output byte size mismatch");
      std::vector<float> actual(bytes.size() / sizeof(float)), expected(actual.size());
      std::memcpy(expected.data(), bytes.data(), bytes.size());
      Cuda(cudaMemcpy(actual.data(), output.tensor.data, bytes.size(), cudaMemcpyDeviceToHost));
      float error = 0;
      for (std::size_t index = 0; index < actual.size(); ++index) {
        Require(std::isfinite(actual[index]) && std::isfinite(expected[index]), "nonfinite output");
        error = std::max(error, std::abs(actual[index] - expected[index]));
      }
      Require(error <= 2e-6f, "generated Session output differs from eager reference");
      std::ofstream raw(std::string(argv[1]) + "/observed_" + std::to_string(run) + ".bin", std::ios::binary);
      raw.write(reinterpret_cast<const char*>(actual.data()), static_cast<std::streamsize>(bytes.size()));
      Require(raw.good(), "failed to archive full output");
      std::printf("session run=%d accepted=%d max_abs=%.12g\n", run, status.ok(), error);
"""
        + query
        + r"""
    }
    session.reset();
    std::puts("GENERATED_REPLAY Session audit passed");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "generated replay audit: %s\n", error.what());
    return 1;
  }
}
"""
    )


@pytest.mark.cuda_aoti
@pytest.mark.skipif(
    os.environ.get("VLAFORGE_RUN_CUDA_AOTI") != "1",
    reason="opt-in real generated CUDA replay audit",
)
@pytest.mark.parametrize("kind", ["continuous", "autoregressive"])
def test_generated_session_real_aoti_whole_loop(kind, tmp_path, monkeypatch):
    import torch
    from vlaforge.deployment.build import build_artifact_compile_bundle
    from vlaforge.deployment.contract import (
        ArtifactIdentity,
        ArtifactKind,
        RegionArtifactContract,
        WorkspaceContract,
    )
    from vlaforge.ir.serializer import io_schema_digest

    program, examples, samples, expected, steps = make_case(kind, "required")
    major, minor = torch.cuda.get_device_capability(0)
    target = f"sm_{major}{minor}"
    packages, captures = {}, {}
    for region in program.module.regions:
        capture = capture_region(
            region, program.regions[region.name], examples[region.name]
        ).require_supported()
        captures[region.name] = capture
        path = tmp_path / f"{region.name}.pt2"
        torch._inductor.aoti_compile_and_package(
            capture.exported_program, package_path=str(path)
        )
        packages[region.name] = path
    pack = tmp_path / "inputs"
    pack.mkdir()
    for index, sample in enumerate(samples):
        for name, tensor in sample.items():
            (pack / f"input_{index}_{name}.bin").write_bytes(
                tensor.detach().cpu().contiguous().numpy().tobytes()
            )
        (pack / f"expected_{index}.bin").write_bytes(
            expected[index].detach().cpu().contiguous().numpy().tobytes()
        )
    env = {
        "TORCH_CUDA_ARCH_LIST": f"{major}.{minor}",
        "CMAKE_BUILD_PARALLEL_LEVEL": "2",
        "LDFLAGS": "-Wl,--wrap=vlaforge_execution_context_copy",
    }
    build_records = []
    original_run = subprocess.run

    def recorded_run(command, *args, **kwargs):
        result = original_run(command, *args, **kwargs)
        if command[0] == "cmake":
            build_records.append(
                {
                    "command": command,
                    "cwd": str(kwargs.get("cwd", Path.cwd())),
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "build_environment_overrides": env,
                }
            )
            (tmp_path / "cmake-executions.json").write_text(
                json.dumps(build_records, indent=2) + "\n"
            )
        return result

    monkeypatch.setattr(subprocess, "run", recorded_run)
    for mode in ("off", "required"):
        selected, _, _, _, _ = make_case(kind, mode)
        contracts = {}
        for index, region in enumerate(selected.module.regions):
            evidence = captures[region.name].evidence
            path = packages[region.name]
            contracts[region.name] = RegionArtifactContract(
                region_id=index,
                region_name=region.name,
                inputs=evidence.inputs,
                outputs=evidence.outputs,
                io_schema_digest=io_schema_digest(selected.module),
                identity=ArtifactIdentity(
                    "generic-tensor-replay-audit",
                    "local-test",
                    "synthetic-parameters-not-a-VLA-checkpoint",
                    evidence.graph_digest,
                ),
                artifact_kind=ArtifactKind.AOTI_PACKAGE,
                artifact_path=f"artifacts/{region.name}.pt2",
                artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                artifact_size_bytes=path.stat().st_size,
                workspace=WorkspaceContract(),
                capability=aoti_backend_capability(target, ("f32", "i64")),
                effect_audit=evidence.effect_audit,
            )
        compilation = compile_module(
            selected.module, default_device="cuda:0", state_device="cuda:0"
        )
        loop = next(task for task in compilation.plan.tasks if task.opcode == "vla.for")
        bundle = tmp_path / mode
        manifest = build_artifact_compile_bundle(
            selected.module,
            bundle,
            region_artifacts=contracts,
            artifact_sources=packages,
            validators=selected.cpp_validators(),
            runner_source=cuda_runner(selected, steps, loop.id, replay=mode != "off"),
            runtime_root=ROOT,
            cmake_prefix_path=torch.utils.cmake_prefix_path,
            backend_versions={"aoti": torch.__version__},
            source_revision="local-audit",
            source_dirty=True,
            environment=env,
            default_device="cuda:0",
            state_device="cuda:0",
            initial_state={state.name: ZERO_STATE for state in selected.module.states},
        )
        manifest.verify_files(bundle)
        binary = bundle / "bin/vlaforge_generated_runner"
        dependencies = subprocess.run(
            ["ldd", str(binary)], check=True, capture_output=True, text=True
        )
        (bundle / "ldd.log").write_text(dependencies.stdout)
        assert "libpython" not in dependencies.stdout.lower()
        for run_mode in ("success", "fatal") if mode == "required" else ("success",):
            command = [str(binary), str(bundle), str(pack), run_mode]
            result = subprocess.run(
                command,
                check=False,
                env={
                    **os.environ,
                    "PYTHONHOME": "/nonexistent",
                    "PYTHONPATH": "/nonexistent",
                    "PATH": "/usr/bin:/bin",
                },
                capture_output=True,
                text=True,
            )
            (bundle / f"{run_mode}.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "exit_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                    indent=2,
                )
            )
            assert result.returncode == (78 if run_mode == "fatal" else 0), (
                result.stdout + result.stderr
            )
            assert (
                "fatal Session quarantine passed"
                if run_mode == "fatal"
                else "Session audit passed"
            ) in result.stdout
    import numpy as np

    comparison = []
    for index in range(len(samples)):
        baseline = (tmp_path / "off" / f"observed_{index}.bin").read_bytes()
        replayed = (tmp_path / "required" / f"observed_{index}.bin").read_bytes()
        reference = np.frombuffer(
            (pack / f"expected_{index}.bin").read_bytes(), np.float32
        ).astype(np.float64)
        actual = np.frombuffer(replayed, np.float32).astype(np.float64)
        delta = actual - reference
        comparison.append(
            {
                "run": index,
                "ordinary_replay_bitwise_equal": baseline == replayed,
                "replay_sha256": hashlib.sha256(replayed).hexdigest(),
                "eager_max_abs": float(np.max(np.abs(delta))),
                "eager_mse": float(np.mean(delta * delta)),
                "eager_cosine": float(
                    np.dot(actual, reference)
                    / (np.linalg.norm(actual) * np.linalg.norm(reference))
                ),
            }
        )
    (tmp_path / "full-output-comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    assert all(row["ordinary_replay_bitwise_equal"] for row in comparison)


@pytest.mark.parametrize("policy", ["off", "batch-only", "prefer", "required"])
def test_offline_policy_preserves_original_semantics_and_archives_certificate(
    policy, tmp_path
):
    from vlaforge.deployment.build import _write_compilation_metadata
    from vlaforge.ir.serializer import module_digest, module_from_data

    program, original, definitions = capture_cpu_case("continuous", "off")
    module = original.module
    before = module_digest(module)
    selected = compile_module(module, loop_execution=policy, default_device="cuda:0")
    assert selected.input_module is module and module_digest(module) == before
    assert selected.certificate.input_semantic_digest == before
    assert selected.module.regions == module.regions
    assert (
        selected.module.inputs == module.inputs
        and selected.module.outputs == module.outputs
    )
    assert any(
        item.name == "bounded_loop_execution" and f"policy={policy}" in item.reason
        for item in selected.certificate.passes
    )
    assert all(
        task.attributes.get("replay", "off") == policy
        for task in selected.plan.tasks
        if task.opcode == "vla.for"
    )
    _write_compilation_metadata(selected, tmp_path)
    loaded = module_from_data(
        json.loads((tmp_path / "input_semantic_ir.json").read_text())
    )
    assert module_digest(loaded) == before
    assert (
        json.loads((tmp_path / "loop_execution.json").read_text())["requested"]
        == policy
    )
    generate_compiled_cpp_session(
        selected, artifact_regions=definitions, validators=program.cpp_validators()
    )


def test_batch_only_needs_context_but_not_graph_provider():
    program, original, definitions = capture_cpu_case("continuous", "off")
    selected = compile_module(
        original.module, loop_execution="batch-only", default_device="cuda:0"
    )
    definitions = {
        name: replace(
            value, supports_external_cuda_graph=False, supports_execution_context=True
        )
        for name, value in definitions.items()
    }
    source = generate_compiled_cpp_session(
        selected, artifact_regions=definitions, validators=program.cpp_validators()
    ).as_dict()["session_generated.cpp"]
    assert "VLAFORGE_REPLAY_ORDINARY" in source and "4u, 0u, nullptr" in source
    assert "vlaforge_bounded_replay_create(context, nullptr" in source
    assert "vlaforge_aoti_graph_backend_api" not in source
    assert "vlaforge_libtorch_graph_backend_api" not in source
    assert "vlaforge_model_session_get_replay_info" in source
    definitions["tensor_update"] = replace(
        definitions["tensor_update"], supports_execution_context=False
    )
    with pytest.raises(CodegenUnsupportedError, match="execution_context_unavailable"):
        generate_compiled_cpp_session(
            selected, artifact_regions=definitions, validators=program.cpp_validators()
        )


def test_source_default_preserves_plan_and_certificate_exactly():
    _, original, _ = capture_cpu_case("continuous", "off")
    implicit = compile_module(original.module, default_device="cuda:0")
    explicit = compile_module(
        original.module, loop_execution="source", default_device="cuda:0"
    )
    assert implicit == explicit and implicit.input_module is None


@pytest.mark.parametrize("policy", ["off", "batch-only", "required"])
def test_shared_libtorch_provider_is_backend_independent(policy):
    program, original, definitions = capture_cpu_case("continuous", "off")
    selected = compile_module(original.module, loop_execution=policy, default_device="cuda:0")
    definitions = {name: replace(value, backend="torchscript", backend_variant="torchscript-aten-context/1",
                                 supports_execution_context=True)
                   for name, value in definitions.items()}
    sources = generate_compiled_cpp_session(selected, artifact_regions=definitions,
                                            validators=program.cpp_validators()).as_dict()
    source = sources["session_generated.cpp"]
    assert "vlaforge_torchscript_region_execution_extension_api()" in source
    if policy == "required":
        assert "vlaforge_libtorch_graph_backend_api()" in source
        assert '#include "vlaforge/backends/libtorch_graph.h"' in source
    else:
        assert "vlaforge_libtorch_graph_backend_api()" not in source
    assert "VLAFORGE_TORCHSCRIPT_ENABLE_CUDA ON" in sources["CMakeLists.txt"]
    with pytest.raises(ValueError, match="loop execution policy"):
        compile_module(original.module, loop_execution="unknown")
