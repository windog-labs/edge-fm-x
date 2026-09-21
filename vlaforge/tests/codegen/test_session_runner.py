"""Typed transport fixtures; real model runs are archived separately."""

from dataclasses import replace
from pathlib import Path

import pytest
from vlaforge.codegen.session_runner import render_resident_tensor_runner
from vlaforge.deployment.libtorch_numerical import NAMESPACE, REDUCTION_API
from vlaforge.deployment.numerical import (
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalPolicy,
    NumericalRequirement,
    RegionNumericalBinding,
)
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, Value
from vlaforge.ir.types import TensorType
from vlaforge.numerical_context import NumericalContext


def example(kind="continuous"):
    first = TensorType((1, 3, 4), "f32")
    second = TensorType((1, 3, 2), "f64" if kind == "continuous" else "i64")
    builder = InvocationBuilder("typed_worker", inputs=(
        InputPort("x", first, device="cuda:0"), InputPort("accepted", TensorType((1,), "bool"), device="cuda:0")),
        outputs=(OutputPort("first", first, device="cuda:0", group="output"),
                 OutputPort("second", second, device="cuda:0", group="output", output_id=1)))

    @tensor_region("compute", inputs=(Value("x", first),), outputs=(first, second))
    def compute(value):
        raise AssertionError("declaration-only fixture")

    a, b = builder.call(compute, builder.input("x"))
    module = builder.finish({"first": a, "second": b}, accepted=builder.input("accepted")).module
    outputs = [{"name": "first", "role": "float" if kind == "continuous" else "primary-action"},
               {"name": "second", "role": "primary-action" if kind == "continuous" else "exact"}]
    template = (Path(__file__).resolve().parents[2] / "tools/session_benchmark_runner.cpp.in").read_text()
    return module, outputs, template


def binding(name="compute"):
    excluded = {"schema", "torch_version", "reduction_api", "float32_matmul_precision", "autocast_cpu_dtype", "autocast_cuda_dtype"}
    values = {key: False for key in NumericalContext.__dataclass_fields__ if key not in excluded}
    values.update(autocast_cpu_dtype="bfloat16", autocast_cuda_dtype="float16", float32_matmul_precision="highest",
                  torch_release="2.10.0", reduction_api=REDUCTION_API)
    policy = NumericalPolicy(NAMESPACE, tuple(sorted(values.items())))
    record = NumericalCompileRecord("torchscript", "sm_86", "test/1", "1" * 64, "2" * 64, "3" * 64, 1,
        policy, policy, policy, '{"fixture":true}')
    return RegionNumericalBinding(name, NumericalRequirement(policy, "same-precision", policy.digest(), record.digest()),
                                  record, PROVIDER_REQUIRED)


@pytest.mark.parametrize("kind", ["continuous", "tokens"])
def test_typed_outputs_all_preserved_without_model_dispatch(kind):
    module, outputs, template = example(kind)
    source, contract = render_resident_tensor_runner(module, template, outputs=outputs, samples=16)
    assert "@" not in source and "constexpr std::size_t kSamples = 16u;" in source
    assert "vlaforge_initialize_numerical_worker" not in source
    assert len(contract["outputs"]) == 2
    assert contract["index"] == (1 if kind == "continuous" else 0)
    assert contract["outputs"][1]["dtype"] == ("f64" if kind == "continuous" else "i64")
    assert ", false, \"output-1.i64\"" in source if kind == "tokens" else ", true, \"output-0.f32\"" in source


@pytest.mark.parametrize("count", [True, 0, -1, 100001, 1.0, "1"])
def test_sample_count_fails_closed(count):
    module, outputs, template = example()
    with pytest.raises(ValueError, match="sample count"):
        render_resident_tensor_runner(module, template, outputs=outputs, samples=count)


def test_worker_initializes_after_reset_and_before_input_allocation():
    module, outputs, template = example()
    source, _ = render_resident_tensor_runner(module, template, outputs=outputs, samples=1,
        numerical_bindings=(binding(),), acknowledge_exclusive_process=True, acknowledge_calling_thread=True)
    assert source.index("cudaDeviceReset()") < source.index("const auto numerical_status") < source.index("std::vector<Buffer> buffers")
    assert source.count("NUMERICAL_WORKER_BOOTSTRAP_OK,") == 1
    with pytest.raises(ValueError):
        render_resident_tensor_runner(module, template, outputs=outputs, samples=1, numerical_bindings=(binding(),))


@pytest.mark.parametrize("bindings", [(None,), (binding("unknown"),), (replace(binding(), runtime_enforcement="unimplemented"),)])
def test_unknown_or_partial_provider_coverage_rejected(bindings):
    module, outputs, template = example()
    with pytest.raises(ValueError, match="every Region"):
        render_resident_tensor_runner(module, template, outputs=outputs, samples=1, numerical_bindings=bindings,
            acknowledge_exclusive_process=True, acknowledge_calling_thread=True)


@pytest.mark.parametrize("change", ["unknown", "duplicate"])
def test_template_contract_fails_closed(change):
    module, outputs, template = example()
    marker = "// @NUMERICAL_WORKER_BOOTSTRAP@"
    template = template + "@UNKNOWN@" if change == "unknown" else template.replace(marker, marker + "\n" + marker)
    with pytest.raises(ValueError, match="template"):
        render_resident_tensor_runner(module, template, outputs=outputs, samples=1)


@pytest.mark.parametrize("policy", ["off", "batch-only", "required"])
def test_replay_telemetry_includes_application_warmup(policy):
    value_type = TensorType((1, 4), "f32")
    builder = InvocationBuilder("replay_runner", inputs=(
        InputPort("x", value_type, device="cuda:0"),
        InputPort("accepted", TensorType((1,), "bool"), device="cuda:0"),
    ), outputs=(OutputPort("result", value_type, device="cuda:0", group="output"),))

    @tensor_region("step", inputs=(Value("x", value_type),), outputs=(value_type,))
    def step(value):
        raise AssertionError("declaration-only fixture")

    initial = builder.input("x")
    (result,) = builder.iterate((initial,), lambda _index, state: builder.call(step, state), steps=3)
    module = builder.finish({"result": result}, accepted=builder.input("accepted")).module
    template = (Path(__file__).resolve().parents[2] / "tools/session_benchmark_runner.cpp.in").read_text()
    source, _ = render_resident_tensor_runner(module, template,
        outputs=[{"name": "result", "role": "primary-action"}], samples=1,
        replay_policy=policy)
    if policy == "off":
        assert "vlaforge_model_session_get_replay_info" not in source
        return
    assert "vlaforge_model_session_get_replay_info" in source
    assert "run < warmup ||" not in source
    assert "REPLAY,%zu" in source
    assert "REPLAY_FINAL," in source
    assert "std::_Exit(10)" in source
    if policy == "required":
        assert "info.replay_count != run + 1u" in source
        assert "info.captured_steps != 3u" in source
        assert "info.ordinary_count != 0u" in source
    else:
        assert "info.ordinary_count != run + 1u" in source
        assert "info.replay_count != 0u" in source


@pytest.mark.parametrize("policy", [None, "prefer", "unknown", True])
def test_unknown_replay_policy_rejected(policy):
    module, outputs, template = example()
    with pytest.raises(ValueError, match="replay policy"):
        render_resident_tensor_runner(module, template, outputs=outputs, samples=1, replay_policy=policy)


@pytest.mark.parametrize("host_io", [None, 1, "true"])
def test_host_io_requires_explicit_boolean(host_io):
    module, outputs, template = example()
    with pytest.raises(ValueError, match="host IO"):
        render_resident_tensor_runner(module, template, outputs=outputs, samples=1, host_io=host_io)


@pytest.mark.parametrize("host_io", [False, True])
def test_actual_cpp_runner_transfer_counts_and_timing_boundary(tmp_path, host_io):
    """Execute the entire runner with delayed host ABI stubs, never GPU evidence."""
    import csv
    import io
    import shutil
    import struct
    import subprocess
    from vlaforge.validation.session_benchmark import validate_host_timing

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    module, outputs, template = example()
    source, contract = render_resident_tensor_runner(module, template, outputs=outputs,
        samples=2, owner_handshake=False, host_io=host_io)
    (tmp_path / "runner.cpp").write_text(source)
    (tmp_path / "cuda_runtime_api.h").write_text(r'''
#pragma once
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
using cudaError_t = int;
constexpr int cudaSuccess = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
inline int h2d_count = 0, d2h_count = 0;
inline void pause() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
inline const char* cudaGetErrorString(int) { return "stub"; }
inline int cudaSetDevice(int) { return 0; }
inline int cudaDeviceReset() { return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline int cudaMalloc(void** p, std::size_t n) { *p = std::malloc(n); return *p ? 0 : 1; }
inline int cudaFree(void* p) { std::free(p); return 0; }
inline int cudaMemcpy(void* dst, const void* src, std::size_t n, int kind) {
  pause();
  if (kind == cudaMemcpyHostToDevice) ++h2d_count; else ++d2h_count;
  std::memcpy(dst, src, n); return 0;
}
'''.replace('pause()', 'fixture_pause()'))
    (tmp_path / "session_generated.h").write_text(r'''
#pragma once
#include "vlaforge/runtime/session_c.h"
#include "cuda_runtime_api.h"
#include <cstdio>
namespace vlaforge_generated { constexpr char kSchemaDigest[] = "fixture"; }
struct VLAForgeSession { const float* input = nullptr; float first[12]; double second[6]; };
inline VLAForgeStatus Bind(VLAForgeSession* s, std::uint32_t i,
    const VLAForgeBoundTensor* t, const VLAForgeInputStamp*) {
  fixture_pause(); if (i == 0) s->input = static_cast<const float*>(t->tensor.data);
  return {VLAFORGE_STATUS_OK};
}
inline VLAForgeStatus Run(VLAForgeSession* s) {
  fixture_pause(); fixture_pause();
  for (int i = 0; i < 12; ++i) s->first[i] = s->input[i];
  for (int i = 0; i < 6; ++i) s->second[i] = s->input[i];
  return {VLAFORGE_STATUS_OK};
}
inline VLAForgeStatus Output(const VLAForgeSession* s, std::uint32_t i, VLAForgeBoundTensor* t) {
  static const std::int64_t shape0[] = {1, 3, 4}, shape1[] = {1, 3, 2};
  *t = {sizeof(VLAForgeBoundTensor), {i ? static_cast<void*>(const_cast<double*>(s->second)) :
      static_cast<void*>(const_cast<float*>(s->first)), 48u, i ? shape1 : shape0, 3u,
      i ? VLAFORGE_DTYPE_F64 : VLAFORGE_DTYPE_F32, {VLAFORGE_DEVICE_CUDA, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 64u};
  return {VLAFORGE_STATUS_OK};
}
inline void Destroy(VLAForgeSession* s) { std::fprintf(stderr, "COPIES,%d,%d\n", h2d_count, d2h_count); delete s; }
inline const VLAForgeSessionApi* vlaforge_model_session_api() {
  static const VLAForgeSessionApi api = {sizeof(VLAForgeSessionApi), 2u, "fixture", 7u,
      &Bind, nullptr, &Run, &Output, nullptr, nullptr, &Destroy};
  return &api;
}
inline VLAForgeStatus vlaforge_model_session_create_from_bundle(const char*, std::size_t, VLAForgeSession** s) {
  *s = new VLAForgeSession(); return {VLAFORGE_STATUS_OK};
}
extern "C" inline VLAForgeStatus vlaforge_session_api_validate(const VLAForgeSessionApi*, const char*, std::size_t) {
  return {VLAFORGE_STATUS_OK};
}
''')
    include = Path(__file__).resolve().parents[2] / "include"
    subprocess.run([compiler, "-std=c++17", "-I", str(tmp_path), "-I", str(include),
                    str(tmp_path / "runner.cpp"), "-o", str(tmp_path / "runner")], check=True, capture_output=True)
    for sample in range(2):
        folder = tmp_path / "data" / str(sample)
        folder.mkdir(parents=True)
        values = [float(i + 1 + 100 * sample) for i in range(12)]
        (folder / "0.bin").write_bytes(struct.pack('<12f', *values))
        (folder / "1.bin").write_bytes(b'\1')
        for spec in contract['outputs']:
            raw = struct.pack('<12f', *values) if spec['index'] == 0 else struct.pack('<6d', *values[:6])
            for name in ('direct_file', 'eager_file'):
                (folder / spec[name]).write_bytes(raw)
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run([str(tmp_path / "runner"), "fixture", str(tmp_path / "data"),
                             str(out), "2", "2"], check=True, capture_output=True, text=True)
    assert f"COPIES,{12 if host_io else 4},8" in result.stderr
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    assert len(rows) == 4 and all(row['direct_exact'] == '1' for row in rows)
    for spec in contract['outputs']:
        expected = b''.join((tmp_path / 'data' / str(i % 2) / spec['direct_file']).read_bytes() for i in range(4))
        assert (out / spec['raw_file']).read_bytes() == expected
    if host_io:
        with (out / 'host-timing.csv').open() as stream:
            timings = list(csv.DictReader(stream))
        validate_host_timing(timings, rows)
        for timing in timings:
            for segment in ('h2d_ns', 'bind_ns', 'model_ns', 'd2h_ns'):
                assert int(timing[segment]) >= 2_000_000
    else:
        assert not (out / 'host-timing.csv').exists()
        assert all(int(row['latency_ns']) >= 2_000_000 for row in rows)
