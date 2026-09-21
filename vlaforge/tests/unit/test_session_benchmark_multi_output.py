import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vlaforge.validation.session_benchmark import (
    BOUNDARY,
    MULTI_SCHEMA,
    benchmark_output_contract,
    compare_output_bytes,
    encode_output_reference,
    input_specs,
    load_reference_array,
    validate_protocol,
)


def tool():
    spec = importlib.util.spec_from_file_location(
        "multi_benchmark_tool", Path(__file__).resolve().parents[2] / "tools/benchmark_session.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case():
    ports = [SimpleNamespace(name=name, device="cuda:0", payload=SimpleNamespace(dtype=dtype, shape=shape))
             for name, dtype, shape in (("raw", "f32", (1, 2, 3)), ("action", "f32", (1, 2, 3)),
                                       ("native", "f64", (2, 3)), ("receipt", "u8", (16,)),
                                       ("count", "i64", (1,)))]
    module = SimpleNamespace(inputs=[ports[-1]], outputs=ports)
    protocol = {"schema": MULTI_SCHEMA, "boundary": BOUNDARY, "warmup": 128,
                "measured": 1024, "processes": 5, "policies": ["off"], "bundles": {"off": "bundle"},
                "quality_gate": "passed", "eager_validation": "bitwise", "evidence": ["source.json"],
                "paper_gates": {"mse_max": 1e-5, "cosine_min": .9999},
                "outputs": [{"name": port.name, "role": "primary-action" if index == 1 else
                             "exact" if index > 2 else "float"} for index, port in enumerate(ports)]}
    protocol["outputs"][1]["active_dimensions"] = [0, 2]
    protocol["samples"] = [{"inputs": {"count": "count.bin"}, "outputs": {
        port.name: {"direct": port.name + "-direct.npy", "eager": port.name + "-eager.npy"}
        for port in ports}}] * 2
    return module, protocol


def test_complete_contract_preserves_abi_order_primary_index_and_integer_types():
    module, protocol = case()
    validate_protocol(protocol)
    contract = benchmark_output_contract(module, protocol)
    assert contract["primary_output"] == "action" and contract["index"] == 1
    assert len(contract["outputs"]) == 5 and contract["active_indices"] == [0, 2, 3, 5]
    assert contract["outputs"][2]["dtype"] == "f64" and contract["outputs"][3]["dtype"] == "u8"
    assert len({item["raw_file"] for item in contract["outputs"]}) == 5
    declarations, ordinal, count = input_specs(module, contract=contract)
    assert "VLAFORGE_DTYPE_I64" in declarations and ordinal == 0 and count == 6
    generated = tool().additional_output_declarations(contract)
    assert "1u, VLAFORGE" not in generated and "VLAFORGE_DTYPE_U8" in generated


def test_v2_explicit_extraction_root_is_a_supported_configuration():
    _, protocol = case()
    protocol["aoti_package_extraction_root"] = "/private/aoti-packages"
    validate_protocol(protocol)


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "missing_primary", "two_primary", "unknown_role",
                                    "extra_sample", "missing_ref", "extra_ref", "bad_ref", "active_exact", "old_schema"])
def test_strict_protocol_rejects_partial_or_ambiguous_output_contract(mutation):
    _, protocol = case()
    if mutation == "unknown": protocol["outputz"] = []
    if mutation == "duplicate": protocol["outputs"][0]["name"] = "action"
    if mutation == "missing_primary": protocol["outputs"][1] = {"name": "action", "role": "float"}
    if mutation == "two_primary": protocol["outputs"][0]["role"] = "primary-action"
    if mutation == "unknown_role": protocol["outputs"][0]["role"] = "skip"
    if mutation == "extra_sample": protocol["samples"][0]["eager"] = "legacy.npy"
    if mutation == "missing_ref": del protocol["samples"][0]["outputs"]["receipt"]
    if mutation == "extra_ref": protocol["samples"][0]["outputs"]["receipt"]["skip"] = True
    if mutation == "bad_ref": protocol["samples"][0]["outputs"]["receipt"]["eager"] = None
    if mutation == "active_exact": protocol["outputs"][3]["active_dimensions"] = [0]
    if mutation == "old_schema": protocol["schema"] = "vlaforge.session_latency_protocol/1"
    with pytest.raises(ValueError): validate_protocol(protocol)


@pytest.mark.parametrize("mutation", ["missing", "order", "device", "role_dtype", "shape", "unsupported"])
def test_module_cannot_drop_or_reinterpret_any_output(mutation):
    module, protocol = case()
    if mutation == "missing": module.outputs.pop()
    if mutation == "order": module.outputs.reverse()
    if mutation == "device": module.outputs[4].device = "cuda:1"
    if mutation == "role_dtype": module.outputs[3].payload.dtype = "f32"
    if mutation == "shape": module.outputs[4].payload.shape = (0,)
    if mutation == "unsupported": module.outputs[4].payload.dtype = "complex64"
    with pytest.raises(ValueError): benchmark_output_contract(module, protocol)


@pytest.mark.parametrize("dtype,value", [("u8", [0, 255]), ("i64", [2**60 + 1, -3]),
                                         ("u64", [2**64 - 1, 2**63 + 1]), ("bool", [True, False])])
def test_exact_references_never_round_through_float_and_check_both_sources(dtype, value):
    formats = {"u8": "u1", "i64": "<i8", "u64": "<u8", "bool": "?"}
    spec = {"dtype": dtype, "shape": [2], "role": "exact"}
    expected = np.array(value, dtype=formats[dtype])
    raw = encode_output_reference(expected, spec)
    assert raw == expected.tobytes()
    checked = compare_output_bytes(raw, raw, raw, spec)
    assert checked == {"same_artifact_bitwise_equal": True, "eager_bitwise_equal": True,
                       "comparison": "exact-bytes-only"}
    corrupt = bytes([raw[0] ^ 1]) + raw[1:]
    with pytest.raises(ValueError): compare_output_bytes(raw, raw, corrupt, spec)
    with pytest.raises(ValueError): compare_output_bytes(raw, corrupt, raw, spec)
    with pytest.raises(ValueError): encode_output_reference(expected.astype(np.float64), spec)


def test_npz_reference_is_resolved_by_complete_shape_and_dtype(tmp_path):
    spec = {"dtype": "f32", "shape": [1, 50, 32]}
    normalized = np.zeros(spec["shape"], dtype=np.float32)
    physical = np.zeros((50, 14), dtype=np.float64)
    path = tmp_path / "actions.npz"
    np.savez(path, normalized_reference=normalized, physical_reference=physical)
    actual = load_reference_array(path, "normalized_action_chunk", spec["dtype"], spec["shape"])
    assert actual.dtype == normalized.dtype and actual.shape == normalized.shape
    assert actual.tobytes() == normalized.tobytes()


def test_ambiguous_npz_reference_requires_a_unique_complete_array(tmp_path):
    spec = {"dtype": "f32", "shape": [2, 3]}
    path = tmp_path / "actions.npz"
    np.savez(path, first=np.ones(spec["shape"], dtype=np.float32),
             second=np.zeros(spec["shape"], dtype=np.float32))
    with pytest.raises(ValueError, match="exactly one complete"):
        load_reference_array(path, "normalized_action_chunk", spec["dtype"], spec["shape"])


def test_primary_metrics_remain_local_and_native_fp64_is_preserved():
    module, protocol = case()
    contract = benchmark_output_contract(module, protocol)
    value = np.ones((1, 2, 3), dtype=np.float32)
    changed = value.copy()
    changed[..., 0] += .01
    checked = compare_output_bytes(changed.tobytes(), changed.tobytes(), value.tobytes(), contract)
    assert checked["primary_metrics"]["count"] == 4
    assert checked["primary_metrics"]["mse"] > checked["complete_storage_metrics"]["mse"]
    native = np.array([[0., -0., 1. + 2**-40]] * 2, dtype=np.float64)
    raw = encode_output_reference(native, contract["outputs"][2])
    assert raw == native.tobytes() and len(raw) == 48
    with pytest.raises(ValueError): encode_output_reference(native.astype(np.float32), contract["outputs"][2])


def archive(tmp_path):
    module, protocol = case()
    contract = benchmark_output_contract(module, protocol)
    api = tool()
    folder = tmp_path / "run"
    folder.mkdir()
    api.write(tmp_path / "protocol.json", protocol)
    api.write(tmp_path / "tensor-contract.json", contract)
    api.write(folder / "execution.json", {"status": "executed", "exit_code": 0})
    (folder / "samples.csv").write_text("run,revision\n0,1\n1,2\n")
    values = [np.ones((1, 2, 3), dtype=np.float32), np.ones((1, 2, 3), dtype=np.float32),
              np.ones((2, 3), dtype=np.float64), np.arange(16, dtype=np.uint8),
              np.array([2**60 + 1], dtype=np.int64)]
    for spec, value in zip(contract["outputs"], values, strict=True):
        raw = encode_output_reference(value, spec)
        (folder / spec["raw_file"]).write_bytes(raw * 2)
        for sample in range(2):
            data = tmp_path / "data" / str(sample)
            data.mkdir(parents=True, exist_ok=True)
            (data / spec["direct_file"]).write_bytes(raw)
            (data / spec["eager_file"]).write_bytes(raw)
    rows = [{"revision": 1}, {"revision": 2}]
    return api, protocol, contract, folder, rows


def test_archive_revalidates_every_output_hash_and_complete_metrics(tmp_path):
    api, protocol, contract, folder, rows = archive(tmp_path)
    binding = api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, persist=True)
    assert len(binding["raw_sha256"]) == 5
    assert api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, expected=binding) == binding
    recorded = api.read(folder / "multi-output-fidelity.json")
    assert len(recorded["outputs"]) == 5
    assert "complete_storage_metrics" not in recorded["outputs"][4]["calls"][0]


def test_repeated_calls_read_each_reference_once_per_verification(tmp_path, monkeypatch):
    api, protocol, contract, folder, rows = archive(tmp_path)
    for spec in contract["outputs"]:
        path = folder / spec["raw_file"]
        path.write_bytes(path.read_bytes() * 6)
    rows *= 6
    original = Path.read_bytes
    reads = []

    def counted(path):
        if path.is_relative_to(tmp_path / "data"):
            reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    binding = api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, persist=True)
    assert len(reads) == 2 * len(protocol["samples"]) * len(contract["outputs"])
    assert len(set(reads)) == len(reads)
    assert all(len(item["calls"]) == 12 for item in api.read(folder / "multi-output-fidelity.json")["outputs"])
    reads.clear()
    api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, expected=binding)
    assert len(reads) == 20


@pytest.mark.parametrize("kind", ["direct", "eager"])
def test_reference_cache_never_survives_a_verification_call(tmp_path, kind):
    api, protocol, contract, folder, rows = archive(tmp_path)
    binding = api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, persist=True)
    path = tmp_path / "data/1" / contract["outputs"][-1][kind + "_file"]
    path.write_bytes(np.array([2**60], dtype=np.int64).tobytes())
    with pytest.raises(ValueError):
        api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, expected=binding)


def test_reference_reuse_still_checks_last_complete_actual_output(tmp_path):
    api, protocol, contract, folder, rows = archive(tmp_path)
    for spec in contract["outputs"]:
        path = folder / spec["raw_file"]
        path.write_bytes(path.read_bytes() * 6)
    rows *= 6
    spec = contract["outputs"][-1]
    path = folder / spec["raw_file"]
    raw = path.read_bytes()
    path.write_bytes(raw[:-8] + np.array([2**60], dtype=np.int64).tobytes())
    with pytest.raises(ValueError):
        api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, persist=True)


@pytest.mark.parametrize("mutation", ["receipt", "counter", "missing", "truncated", "nan", "report", "binding", "eager", "unknown_json"])
def test_corrupted_secondary_evidence_cannot_keep_passed_status(tmp_path, mutation):
    api, protocol, contract, folder, rows = archive(tmp_path)
    binding = api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, persist=True)
    if mutation in ("receipt", "counter", "truncated", "missing"):
        path = folder / contract["outputs"][4 if mutation == "counter" else 3]["raw_file"]
        raw = path.read_bytes()
        if mutation == "missing": path.unlink()
        elif mutation == "truncated": path.write_bytes(raw[:-1])
        else: path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    if mutation == "nan": (folder / contract["outputs"][2]["raw_file"]).write_bytes(np.full(12, np.nan, dtype=np.float64).tobytes())
    if mutation == "report": api.write(folder / "multi-output-fidelity.json", {"status": "passed"})
    if mutation == "binding": binding["raw_sha256"]["receipt"] = "0" * 64
    if mutation == "eager": (tmp_path / "data/1/eager-4.bin").write_bytes(np.array([2**60], dtype=np.int64).tobytes())
    if mutation == "unknown_json":
        (folder / "multi-output-fidelity.json").write_text('{"status":"passed","status":"passed"}')
    with pytest.raises((ValueError, FileNotFoundError)):
        api.verify_multi_output_evidence(tmp_path, folder, protocol, contract, rows, expected=binding)


def test_v2_unknown_nested_declaration_keys_fail_closed():
    _, protocol = case()
    protocol["outputs"][0]["optional"] = True
    with pytest.raises(ValueError, match="unknown"): validate_protocol(protocol)


def test_parser_rejects_duplicate_protocol_output_name_keys(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text('{"outputs":{"receipt":1,"receipt":2}}')
    with pytest.raises(ValueError, match="duplicate"): tool().read(path)


@pytest.mark.parametrize("mutation", [None, "raw", "summary", "count"])
def test_v2_aggregate_latency_is_recomputed_from_raw_csv(mutation):
    api = tool()
    rows = [{"latency_ns": "30"}, {"latency_ns": "10"}, {"latency_ns": "20"}]
    report = {"warmup": 1, "measured": 2, "validated_outputs": 3,
              "latency": api.latency_report([{"index": 0, "latency_ns": 10, "repeat_id": 4},
                                             {"index": 1, "latency_ns": 20, "repeat_id": 4}])}
    if mutation == "raw": report["latency"]["raw_samples"][0]["latency_ns"] = 1
    if mutation == "summary": report["latency"]["summary"]["mean_ns"] = 1
    if mutation == "count": report["validated_outputs"] = 1
    if mutation is None:
        api.verify_multi_output_latency(report, rows, warmup=1, measured=2, repeat=4)
    else:
        with pytest.raises(ValueError, match="raw samples"):
            api.verify_multi_output_latency(report, rows, warmup=1, measured=2, repeat=4)


def test_actual_cpp_secondary_gate_preserves_integer_bytes_and_rejects_bad_metadata(tmp_path):
    """Host ABI harness only; its explicit memcpy stub is not CUDA execution."""
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    api = tool()
    template = Path(api.__file__).with_name("session_benchmark_runner.cpp.in").read_text()
    includes = "\n".join(line for line in template.splitlines() if line.startswith("#include <") and "cuda" not in line)
    helpers = template[template.index("struct Metrics"):template.index("\n}\n\nint main")]
    module, protocol = case()
    contract = benchmark_output_contract(module, protocol)
    specs = [dict(contract["outputs"][3], index=0), dict(contract["outputs"][4], index=1)]
    helpers = helpers.replace("@ADDITIONAL_OUTPUT_SPECS@", api.additional_output_declarations({"outputs": specs}))
    io = template[template.index("bool Read("):template.index("bool OwnerStage(")]
    source = includes + r'''
#include "vlaforge/runtime/session_c.h"
#define VLAFORGE_BENCHMARK_MULTI_OUTPUT 1
const VLAForgeSessionApi* vlaforge_model_session_api();
constexpr auto kOutputDtype = VLAFORGE_DTYPE_F32;
constexpr std::size_t kOutputCount = 1u, kSamples = 1u;
constexpr int kOrdinal = 0;
const std::vector<std::size_t> active_indices = {0};
constexpr int cudaMemcpyDeviceToHost = 0;
int copies = 0;
int cudaMemcpy(void* dst, const void* src, std::size_t bytes, int) { ++copies; std::memcpy(dst, src, bytes); return 0; }
bool Cuda(int status) { return status == 0; }
int mode = 0;
std::uint8_t receipt[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,255};
std::int64_t count[1] = {(1LL << 60) + 1};
std::int64_t dims[2] = {16, 1};
VLAForgeStatus ReadOutput(const VLAForgeSession*, std::uint32_t index, VLAForgeBoundTensor* out) {
  if (mode == 6) return {VLAFORGE_STATUS_INVALID_ARGUMENT};
  *out = {sizeof(VLAForgeBoundTensor), {index ? static_cast<void*>(count) : static_cast<void*>(receipt),
          index ? 8u : 16u, &dims[index], 1u, index ? VLAFORGE_DTYPE_I64 : VLAFORGE_DTYPE_U8,
          {VLAFORGE_DEVICE_CUDA, 0}}, VLAFORGE_LAYOUT_CONTIGUOUS, 64u};
  if (mode == 1) receipt[0] = 1;
  if (mode == 2) out->tensor.dtype = VLAFORGE_DTYPE_F32;
  if (mode == 3) out->tensor.rank = 2u;
  if (mode == 4) out->tensor.device.ordinal = 1;
  if (mode == 5) out->tensor.size_bytes = 1;
  if (mode == 7) out->layout = VLAFORGE_LAYOUT_NHWC;
  if (mode == 8) out->tensor.data = nullptr;
  if (mode == 9) dims[0] = 15;
  if (mode == 10) out->tensor.dimensions = nullptr;
  return {VLAFORGE_STATUS_OK};
}
''' + io + helpers + r'''
int main(int argc, char** argv) {
  mode = std::atoi(argv[3]);
  AdditionalOutputs outputs;
  if (!outputs.Initialize(argv[1], argv[2])) return 21;
  VLAForgeSessionApi api{}; api.read_output_tensor = &ReadOutput;
  if (mode == 0) {
    if (!outputs.Capture(&api, nullptr) || copies != 2) return 23;
    for (auto& stream : outputs.raw) if (stream.tellp() != 0) return 24;
    if (!outputs.Validate(&api, nullptr, 0u)) return 22;
    if (copies != 2) return 25;
    return 0;
  }
  return outputs.Validate(&api, nullptr, 0u) ? 0 : 22;
}
'''
    (tmp_path / "gate.cpp").write_text(source)
    subprocess.run([compiler, "-std=c++17", "-I", str(Path(api.__file__).parents[1] / "include"),
                    str(tmp_path / "gate.cpp"), "-o", str(tmp_path / "gate")], check=True, capture_output=True)
    data = tmp_path / "data/0"
    data.mkdir(parents=True)
    arrays = [np.array([*range(15), 255], dtype=np.uint8), np.array([2**60 + 1], dtype=np.int64)]
    for spec, values in zip(specs, arrays, strict=True):
        for name in ("direct_file", "eager_file"):
            (data / spec[name]).write_bytes(values.tobytes())
    for mode in range(11):
        output = tmp_path / f"run-{mode}"
        output.mkdir()
        result = subprocess.run([str(tmp_path / "gate"), str(data.parent), str(output), str(mode)], check=False)
        assert result.returncode == (0 if mode == 0 else 22)
    for spec, values in zip(specs, arrays, strict=True):
        assert (tmp_path / "run-0" / spec["raw_file"]).read_bytes() == values.tobytes()
    (data / specs[1]["eager_file"]).write_bytes(np.array([2**60], dtype=np.int64).tobytes())
    out = tmp_path / "eager-mismatch"
    out.mkdir()
    assert subprocess.run([str(tmp_path / "gate"), str(data.parent), str(out), "0"], check=False).returncode == 22
