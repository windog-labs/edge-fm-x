from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from vlaforge.codegen import CppArtifactRegionDefinition, generate_cpp_session
from vlaforge.compiler import compile_module
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType


def _program():
    vector = TensorType((2,), "f32")

    @tensor_region("first", inputs=(Value("x", vector),), outputs=(vector,))
    def first(x):
        return tuple(value + 1 for value in x)

    @tensor_region("second", inputs=(Value("x", vector),), outputs=(vector,))
    def second(x):
        return tuple(value + 2 for value in x)

    builder = InvocationBuilder(
        "context_test",
        inputs=(
            InputPort("x", vector),
            InputPort("accepted", TensorType((1,), "bool")),
        ),
        outputs=(OutputPort("result", vector),),
    )
    (middle,) = builder.call(first, builder.input("x"))
    (output,) = builder.call(second, middle)
    return builder.finish({"result": output}, accepted=builder.input("accepted"))


def _definitions(
    compilation, *, digest="a" * 64, size=1, devices=("cpu", "cpu"), residency="session"
):
    return {
        region.name: CppArtifactRegionDefinition(
            region_name=region.name,
            backend="shared_plugin",
            artifact_path="plugin.so",
            artifact_sha256=digest,
            artifact_size_bytes=size,
            io_schema_digest=compilation.plan.io_schema_digest,
            target="cpu" if device == "cpu" else "sm_86",
            device=device,
            backend_variant="shared-plugin/1",
            residency=residency,
        )
        for region, device in zip(compilation.module.regions, devices, strict=True)
    }


def test_context_slots_are_shared_by_device_not_region():
    program = _program()
    compilation = compile_module(program.module)
    for devices, slots in (
        (("cuda:0", "cuda:0"), (0, 0)),
        (("cuda:0", "cuda:1"), (0, 1)),
    ):
        sources = generate_cpp_session(
            compilation.plan,
            compilation.module,
            artifact_regions=_definitions(compilation, devices=devices),
            validators=program.cpp_validators(),
        ).as_dict()
        source = sources["session_generated.cpp"]
        for region_id, slot in enumerate(slots):
            assert (
                f"region_executables_[{region_id}u],\n"
                f"          &execution_context_views_[{slot}u]"
            ) in source
        assert "api->synchronize(executable)" in source
        assert (
            source.index("DestroyRegion(index - 1u)")
            < source.index(
                "vlaforge_execution_context_destroy(execution_contexts_[slot])"
            )
            < source.index(
                "vlaforge_external_region_plugin_close(region_plugins_[slot])"
            )
        )


_RUNNER = r"""
#include "session_generated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

extern "C" void* __real_aligned_alloc(std::size_t, std::size_t);
extern "C" void* __wrap_aligned_alloc(std::size_t alignment, std::size_t size) {
  static unsigned count = 0;
  ++count;
  const auto* mode = std::getenv("VLAFORGE_CONTEXT_TEST_MODE");
  if (count == 3 && mode != nullptr && std::strcmp(mode, "outputallocfail") == 0) {
    return nullptr;
  }
  return __real_aligned_alloc(alignment, size);
}

extern "C" VLAForgeStatus __real_vlaforge_execution_context_create(
    const VLAForgeExecutionContextOptions*, VLAForgeExecutionContext**);
extern "C" VLAForgeStatus __wrap_vlaforge_execution_context_create(
    const VLAForgeExecutionContextOptions* options, VLAForgeExecutionContext** output) {
  const auto* mode = std::getenv("VLAFORGE_CONTEXT_TEST_MODE");
  if (mode != nullptr && std::strcmp(mode, "contextfail") == 0) {
    *output = nullptr;
    return vlaforge_status_error(VLAFORGE_STATUS_BACKEND_ERROR, "injected");
  }
  return __real_vlaforge_execution_context_create(options, output);
}
extern "C" void __real_vlaforge_execution_context_destroy(VLAForgeExecutionContext*);
extern "C" void __wrap_vlaforge_execution_context_destroy(VLAForgeExecutionContext* context) {
  if (context != nullptr) {
    if (auto* log = std::fopen(std::getenv("VLAFORGE_CONTEXT_TEST_LOG"), "a")) {
      std::fprintf(log, "context_destroy,0,%p\n", static_cast<void*>(context));
      std::fclose(log);
    }
  }
  __real_vlaforge_execution_context_destroy(context);
}

int Run(int argc, char** argv) {
  if (argc != 2) { return 1; }
  vlaforge_generated::ModelSession session(argv[1]);
  const bool expected_failure = std::getenv("VLAFORGE_CONTEXT_TEST_EXPECT_FAILURE") != nullptr;
  if (!session.initialization_status().ok()) { return expected_failure ? 0 : 2; }
  float data[2] = {1.0f, 2.0f};
  std::uint8_t accepted = 1;
  const std::int64_t shape[] = {2};
  const std::int64_t flag_shape[] = {1};
  const VLAForgeBoundTensor input{sizeof(input),
      {data, sizeof(data), shape, 1, VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CPU, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS, 4};
  const VLAForgeBoundTensor flag{sizeof(flag),
      {&accepted, 1, flag_shape, 1, VLAFORGE_DTYPE_BOOL, {VLAFORGE_DEVICE_CPU, 0}},
      VLAFORGE_LAYOUT_CONTIGUOUS, 1};
  for (int i = 0; i < 2; ++i) {
    if (!session.BindTensor(0, input, nullptr).ok() ||
        !session.BindTensor(1, flag, nullptr).ok()) { return 3; }
    const auto status = session.Run();
    if (!status.ok()) {
      std::fprintf(stderr, "Run: %s\n", status.message);
      return expected_failure ? 0 : 4;
    }
    VLAForgeBoundTensor output{};
    if (!session.ReadOutputTensor(0, &output).ok()) { return 5; }
    const auto* values = static_cast<const float*>(output.tensor.data);
    if (values[0] != 4.0f || values[1] != 5.0f) { return 6; }
  }
  return expected_failure ? 7 : 0;
}

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::bad_alloc&) {
    const auto* mode = std::getenv("VLAFORGE_CONTEXT_TEST_MODE");
    return mode != nullptr && std::strcmp(mode, "outputallocfail") == 0 ? 0 : 8;
  }
}
"""


@pytest.mark.parametrize("legacy", (False, True))
@pytest.mark.parametrize("residency", ("session", "invocation"))
def test_generated_context_lifecycle_and_legacy_fallback(tmp_path, legacy, residency):
    runtime = Path(__file__).resolve().parents[2]
    bundle = tmp_path / "bundle"
    bundle.mkdir()
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
        str(runtime / "tests/cpp/execution_context_plugin_fixture.cpp"),
        "-o",
        str(plugin),
    ]
    if legacy:
        command.append("-DVLAFORGE_CONTEXT_TEST_LEGACY=1")
    subprocess.run(command, check=True, capture_output=True, text=True)
    program = _program()
    compilation = compile_module(program.module)
    sources = generate_cpp_session(
        compilation.plan,
        compilation.module,
        artifact_regions=_definitions(
            compilation,
            digest=hashlib.sha256(plugin.read_bytes()).hexdigest(),
            size=plugin.stat().st_size,
            residency=residency,
        ),
        validators=program.cpp_validators(),
        runner_source=_RUNNER,
    )
    generated = tmp_path / "generated"
    sources.write(generated)
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(generated),
            "-B",
            str(build),
            f"-DVLAFORGE_RUNTIME_ROOT={runtime}",
            "-DBUILD_TESTING=OFF",
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--wrap=vlaforge_execution_context_create,--wrap=vlaforge_execution_context_destroy,--wrap=aligned_alloc",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "-j", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    modes = (
        ("normal",)
        if legacy
        else (
            "normal",
            "bindfail",
            "loadfail",
            "createfail",
        "contextfail",
        "outputallocfail",
            "badabi",
            "nullprovider",
            "throwprovider",
        )
    )
    for mode in modes:
        log = tmp_path / f"{mode}.log"
        environment = {
            **os.environ,
            "VLAFORGE_CONTEXT_TEST_LOG": str(log),
            "VLAFORGE_CONTEXT_TEST_MODE": mode,
            "PYTHONHOME": "/nonexistent",
            "PYTHONPATH": "/nonexistent",
        }
        if mode != "normal":
            environment["VLAFORGE_CONTEXT_TEST_EXPECT_FAILURE"] = "1"
        subprocess.run(
            [str(build / "vlaforge_generated_runner"), str(bundle)],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        events = (
            [line.split(",") for line in log.read_text().splitlines()]
            if log.exists()
            else []
        )
        binds = [event for event in events if event[0] == "bind"]
        destroyed = [i for i, event in enumerate(events) if event[0] == "destroy"]
        context_destroyed = [
            i for i, event in enumerate(events) if event[0] == "context_destroy"
        ]
        if legacy:
            assert not binds and not context_destroyed
        elif mode in {"badabi", "nullprovider", "throwprovider"}:
            assert not events
        elif mode == "contextfail":
            assert not context_destroyed and len(destroyed) == 1
        elif mode == "outputallocfail" and residency == "invocation":
            assert not events
        else:
            assert len(context_destroyed) == 1
            assert destroyed and max(destroyed) < context_destroyed[0]
            assert len({event[2] for event in binds}) <= 1
            if mode == "normal":
                assert len(binds) == (2 if residency == "session" else 4)
                assert len(destroyed) == len(binds)
