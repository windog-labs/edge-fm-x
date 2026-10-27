"""NONMODEL: real generated CPU shared-plugin calls with planned workspace.

These providers are software ABI fixtures, not VLA/model acceptance evidence.
"""

import hashlib
import os
from pathlib import Path
import re
import subprocess

import pytest

from vlaforge.codegen import CppArtifactRegionDefinition, generate_compiled_cpp_session
from vlaforge.compiler import compile_module
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType
from vlaforge.plan import ArtifactVariant


def _program():
    vector = TensorType((2,), "f32")

    @tensor_region("first_workspace", inputs=(Value("x", vector),), outputs=(vector,))
    def first(_x):
        raise AssertionError("NONMODEL declaration only")

    @tensor_region("second_workspace", inputs=(Value("x", vector),), outputs=(vector,))
    def second(_x):
        raise AssertionError("NONMODEL declaration only")

    builder = InvocationBuilder("NONMODEL_artifact_workspace", inputs=(
        InputPort("x", vector), InputPort("accepted", TensorType((1,), "bool"))),
        outputs=(OutputPort("result", vector),))
    (a,) = builder.call(first, builder.input("x"))
    (b,) = builder.call(second, a)
    (c,) = builder.call(first, b)
    return builder.finish({"result": c}, accepted=builder.input("accepted"))


def _compile(program):
    return compile_module(program.module, profile="off", default_device="cpu", state_device="cpu",
        artifact_variants={name: ArtifactVariant(backend="shared_plugin", variant="shared-plugin/1",
            artifact_path="nonmodel-workspace.so", workspace_size_bytes=size,
            workspace_alignment=alignment, workspace_device="cpu")
            for name, size, alignment in (("first_workspace", 384, 64), ("second_workspace", 448, 128))})


def _sources(program, compilation, *, digest, size, residency="session", runner=None):
    definitions = {region.name: CppArtifactRegionDefinition(region_name=region.name,
        backend="shared_plugin", artifact_path="nonmodel-workspace.so", artifact_sha256=digest,
        artifact_size_bytes=size, io_schema_digest=compilation.plan.io_schema_digest,
        target="cpu", device="cpu", backend_variant="shared-plugin/1", residency=residency)
        for region in compilation.module.regions}
    return generate_compiled_cpp_session(compilation, artifact_regions=definitions,
        validators=program.cpp_validators(), runner_source=runner)


def test_nonzero_artifact_workspace_codegen_uses_plan_contract_and_task_buffer():
    program = _program()
    compilation = _compile(program)
    # Only a source-generation unit fixture. The executable tests use real DSO bytes.
    source = _sources(program, compilation, digest="1" * 64, size=1).as_dict()["session_generated.cpp"]
    compact = re.sub(r"\s+", " ", source)
    for size, alignment in ((384, 64), (448, 128)):
        assert f"requirement.size_bytes != {size}u" in compact
        assert f"requirement.alignment != {alignment}u" in compact
    assert "requirement.device.kind != VLAFORGE_DEVICE_CPU" in compact
    assert "requirement.device.ordinal != 0" in compact
    tasks = [task for task in compilation.plan.tasks if task.kind == "region"]
    assert len(tasks) == 3
    for task in tasks:
        assert task.workspace_buffer is not None
        assert f"BufferData({task.workspace_buffer}u)" in source
        assert f"region_{task.id}_workspace" in source
        binding = compilation.plan.artifacts[task.artifact_id]
        assert (f"api->bind_workspace(executable, region_{task.id}_workspace, "
            f"{binding.workspace_size_bytes}u)") in compact
    assert "api->bind_workspace(executable, region_" in source
    assert "requirement.size_bytes != 0u" not in compact


_PLUGIN = r'''
#include "vlaforge/runtime/region_executable.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace {
VLAForgeStatus Ok() { return {VLAFORGE_STATUS_OK, nullptr, 0}; }
VLAForgeStatus Error() {
  static constexpr char message[] = "NONMODEL workspace fixture rejection";
  return {VLAFORGE_STATUS_FAILED_PRECONDITION, message, sizeof(message) - 1};
}
bool Mode(const char* name) {
  const auto* value = std::getenv("NONMODEL_WORKSPACE_MODE");
  return value != nullptr && std::strcmp(name, value) == 0;
}
std::uint64_t Bytes(unsigned id) { return id == 0 ? 384u : 448u; }
std::uint64_t Alignment(unsigned id) { return id == 0 ? 64u : 128u; }
void Log(const char* event, unsigned id, const void* pointer, std::uint64_t bytes) {
  const auto* name = std::getenv("NONMODEL_WORKSPACE_LOG");
  if (name == nullptr) return;
  if (auto* file = std::fopen(name, "a")) {
    std::fprintf(file, "%s,%u,%p,%llu\n", event, id, pointer,
        static_cast<unsigned long long>(bytes));
    std::fclose(file);
  }
}
}

struct VLAForgeRegionExecutable {
  unsigned id = 0;
  const VLAForgeValueView* input = nullptr;
  const VLAForgeValueView* output = nullptr;
  std::uint8_t* workspace = nullptr;
  unsigned binds = 0;
  unsigned invokes = 0;
};

namespace {
VLAForgeStatus Create(const VLAForgeRegionCreateOptions* options,
    VLAForgeRegionExecutable** result) {
  if (options == nullptr || result == nullptr || options->region_id > 1 ||
      options->device.kind != VLAFORGE_DEVICE_CPU || options->device.ordinal != 0) return Error();
  *result = new (std::nothrow) VLAForgeRegionExecutable;
  if (*result == nullptr) return Error();
  (*result)->id = options->region_id;
  Log("create", (*result)->id, nullptr, 0);
  return Ok();
}
VLAForgeStatus Load(VLAForgeRegionExecutable*, const VLAForgeArtifactDescriptor*) { return Ok(); }
VLAForgeStatus Query(const VLAForgeRegionExecutable* executable, VLAForgeWorkspaceRequirement* requirement) {
  if (executable == nullptr || requirement == nullptr) return Error();
  *requirement = {Bytes(executable->id), Alignment(executable->id), {VLAFORGE_DEVICE_CPU, 0}};
  if (executable->id == 1) {
    if (Mode("size")) ++requirement->size_bytes;
    if (Mode("alignment")) requirement->alignment *= 2;
    if (Mode("alignment-lower")) requirement->alignment /= 2;
    if (Mode("device-kind")) requirement->device.kind = VLAFORGE_DEVICE_CUDA;
    if (Mode("device-ordinal")) requirement->device.ordinal = 1;
  }
  Log("query", executable->id, nullptr, requirement->size_bytes);
  return Ok();
}
VLAForgeStatus Input(VLAForgeRegionExecutable* executable, std::uint32_t index, const VLAForgeValueView* value) {
  if (index != 0 || value == nullptr || value->kind != VLAFORGE_VALUE_TENSOR) return Error();
  executable->input = value;
  Log("input", executable->id, nullptr, 0);
  return Ok();
}
VLAForgeStatus Output(VLAForgeRegionExecutable* executable, std::uint32_t index, const VLAForgeValueView* value) {
  if (index != 0 || value == nullptr || value->kind != VLAFORGE_VALUE_TENSOR) return Error();
  executable->output = value;
  return Ok();
}
VLAForgeStatus Workspace(VLAForgeRegionExecutable* executable, void* pointer, std::uint64_t bytes) {
  Log("bind", executable->id, pointer, bytes);
  if (pointer == nullptr || bytes != Bytes(executable->id) ||
      reinterpret_cast<std::uintptr_t>(pointer) % Alignment(executable->id) != 0 ||
      (Mode("bind-fail") && executable->id == 1)) return Error();
  executable->workspace = static_cast<std::uint8_t*>(pointer);
  std::memset(pointer, 0x31, static_cast<std::size_t>(bytes));
  executable->workspace[0] = static_cast<std::uint8_t>(0xa0 + executable->id);
  executable->workspace[bytes - 1] = 0x5a;
  ++executable->binds;
  return Ok();
}
VLAForgeStatus Run(VLAForgeRegionExecutable* executable) {
  if (executable->workspace == nullptr || executable->binds != executable->invokes + 1 ||
      executable->input == nullptr || executable->output == nullptr ||
      executable->workspace[0] != 0xa0 + executable->id ||
      executable->workspace[Bytes(executable->id) - 1] != 0x5a) return Error();
  const auto* input = static_cast<const float*>(executable->input->value.tensor.tensor.data);
  auto* output = static_cast<float*>(executable->output->value.tensor.tensor.data);
  for (unsigned i = 0; i < 2; ++i) output[i] = input[i] + static_cast<float>(executable->id + 1);
  ++executable->invokes;
  Log("invoke", executable->id, executable->workspace, Bytes(executable->id));
  return Ok();
}
VLAForgeStatus Sync(VLAForgeRegionExecutable*) { return Ok(); }
void Destroy(VLAForgeRegionExecutable* executable) {
  Log("destroy", executable->id, nullptr, 0);
  delete executable;
}
const VLAForgeRegionExecutableValueApi api = {sizeof(api), VLAFORGE_REGION_EXECUTABLE_VALUE_ABI_VERSION,
    &Create, &Load, &Query, &Input, &Output, &Workspace, &Run, &Sync, &Destroy};
}
extern "C" const VLAForgeRegionExecutableValueApi* vlaforge_region_executable_value_api() { return &api; }
'''


_RUNNER = r'''
#include "session_generated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  const auto* mode = std::getenv("NONMODEL_WORKSPACE_MODE");
  const bool bind_failure = mode != nullptr && std::strcmp(mode, "bind-fail") == 0;
  const bool query_failure = mode != nullptr && std::strcmp(mode, "normal") != 0 && !bind_failure;
  vlaforge_generated::ModelSession session(argv[1]);
  if (!session.initialization_status().ok()) {
    std::printf("NONMODEL_INIT_REJECTION,%s\n", session.initialization_status().message);
    return query_failure ? 0 : 2;
  }
  float values[] = {1, 2};
  std::uint8_t accepted = 1;
  const std::int64_t dimensions[] = {2};
  const std::int64_t flag_dimensions[] = {1};
  const VLAForgeBoundTensor input{sizeof(input), {values, sizeof(values), dimensions, 1,
      VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 4};
  const VLAForgeBoundTensor flag{sizeof(flag), {&accepted, 1, flag_dimensions, 1,
      VLAFORGE_DTYPE_BOOL, {VLAFORGE_DEVICE_CPU, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 1};
  for (unsigned call = 0; call < 3; ++call) {
    values[0] = static_cast<float>(call * 7 + 1);
    values[1] = values[0] + 1;
    if (!session.BindTensor(0, input, nullptr).ok() || !session.BindTensor(1, flag, nullptr).ok()) return 3;
    const auto status = session.Run();
    if (!status.ok()) {
      std::printf("NONMODEL_RUN_REJECTION,%s\n", status.message);
      return query_failure || bind_failure ? 0 : 4;
    }
    if (query_failure || bind_failure) return 5;
    VLAForgeBoundTensor output{};
    if (!session.ReadOutputTensor(0, &output).ok()) return 6;
    const auto* actual = static_cast<const float*>(output.tensor.data);
    if (actual[0] != values[0] + 4 || actual[1] != values[1] + 4) return 7;
  }
  std::puts("NONMODEL_WORKSPACE_CPU_PASS,3,9");
  return 0;
}
'''


def _run(command, root, name, *, environment=None):
    result = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=180)
    (root / (name + ".stdout.log")).write_text(result.stdout)
    (root / (name + ".stderr.log")).write_text(result.stderr)
    assert result.returncode == 0, f"{command!r}: exit {result.returncode}\n{result.stdout}\n{result.stderr}"
    return result


@pytest.fixture(scope="module", params=("session", "invocation"))
def compiled_workspace(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("NONMODEL-workspace-" + request.param)
    runtime = Path(__file__).resolve().parents[2]
    bundle = root / "bundle"
    bundle.mkdir()
    plugin = bundle / "nonmodel-workspace.so"
    plugin_source = root / "nonmodel_workspace.cpp"
    plugin_source.write_text(_PLUGIN)
    _run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
        "-I" + str(runtime / "include"), str(plugin_source), "-o", str(plugin)], root, "provider-build")
    program = _program()
    compilation = _compile(program)
    generated = root / "generated"
    _sources(program, compilation, digest=hashlib.sha256(plugin.read_bytes()).hexdigest(),
        size=plugin.stat().st_size, residency=request.param, runner=_RUNNER).write(generated)
    _run(["cmake", "-S", str(generated), "-B", str(root / "build"),
        "-DVLAFORGE_RUNTIME_ROOT=" + str(runtime), "-DBUILD_TESTING=OFF", "-DCMAKE_BUILD_TYPE=Release",
        "-DVLAFORGE_BUILD_TENSORRT_BACKEND=OFF", "-DVLAFORGE_BUILD_AOTI_BACKEND=OFF",
        "-DVLAFORGE_BUILD_TORCHSCRIPT_BACKEND=OFF"], root, "configure")
    _run(["cmake", "--build", str(root / "build"), "--parallel", "2"], root, "session-build")
    return root, bundle, root / "build/vlaforge_generated_runner", request.param


@pytest.mark.parametrize("mode", ("normal", "size", "alignment", "alignment-lower", "device-kind", "device-ordinal", "bind-fail"))
def test_actual_cpu_nonmodel_workspace_calls_and_query_rejections(compiled_workspace, mode):
    root, bundle, binary, residency = compiled_workspace
    log = root / (mode + ".events.log")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "NONMODEL_WORKSPACE_MODE": mode,
        "NONMODEL_WORKSPACE_LOG": str(log), "PYTHONHOME": "/NONMODEL/no-python", "PYTHONPATH": "/NONMODEL/no-python"}
    result = _run([str(binary), str(bundle)], root, mode, environment=environment)
    events = [line.split(",") for line in log.read_text().splitlines()]
    invocations = [event for event in events if event[0] == "invoke"]
    binds = [event for event in events if event[0] == "bind"]
    if mode == "normal":
        assert "NONMODEL_WORKSPACE_CPU_PASS,3,9" in result.stdout
        assert len(invocations) == len(binds) == 9
        assert [int(event[1]) for event in invocations] == [0, 1, 0] * 3
        assert all(event[2] not in {"(nil)", "0x0"} for event in binds)
        assert all(int(event[3]) == (384 if event[1] == "0" else 448) for event in binds)
        for index, event in enumerate(events):
            if event[0] == "input":
                assert events[index - 1][0:2] == ["bind", event[1]]
    elif mode == "bind-fail":
        assert "NONMODEL_RUN_REJECTION" in result.stdout and "workspace" in result.stdout
        assert len(invocations) == 1 and [event[1] for event in binds] == ["0", "1"]
        assert not any(event[0:2] == ["input", "1"] for event in events)
    else:
        expected = "NONMODEL_INIT_REJECTION" if residency == "session" else "NONMODEL_RUN_REJECTION"
        assert expected in result.stdout and "workspace contract mismatch" in result.stdout
        assert not any(event[0] == "bind" and event[1] == "1" for event in events)
        assert len(invocations) == (0 if residency == "session" else 1)
