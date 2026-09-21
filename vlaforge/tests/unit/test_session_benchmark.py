import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vlaforge.validation.session_benchmark import (
    BOUNDARY,
    SCHEMA,
    active_indices,
    decode_tensor,
    encode_reference,
    input_specs,
    numeric_metrics,
    output_spec,
    process_order,
    protocol_policies,
    replay_checks,
    validate_fidelity,
    validate_loop_policy,
    validate_protocol,
    validate_rows,
)


def protocol():
    return {
        "schema": SCHEMA,
        "boundary": BOUNDARY,
        "warmup": 128,
        "measured": 1024,
        "processes": 5,
        "bundles": dict.fromkeys(("off", "batch-only", "required"), "bundle"),
        "quality_gate": "failed",
        "samples": [{}] * 16,
        "evidence": ["capture.json"],
    }


def test_frozen_protocol_and_counterbalanced_process_order():
    validate_protocol(protocol())
    order = process_order()
    assert len(order) == 15 and len(set(order)) == 15
    assert order[:6] == [
        (0, "off"),
        (0, "batch-only"),
        (0, "required"),
        (1, "batch-only"),
        (1, "required"),
        (1, "off"),
    ]


def test_explicit_single_policy_does_not_invent_unsupported_replay():
    value = protocol()
    value.update(policies=["off"], bundles={"off": "native-bundle"})
    validate_protocol(value)
    assert protocol_policies(value) == ("off",)
    assert process_order(5, ["off"]) == [(index, "off") for index in range(5)]
    value["bundles"]["required"] = "unrequested-bundle"
    with pytest.raises(ValueError, match="match"):
        validate_protocol(value)


@pytest.mark.parametrize("root", [None, "", "relative", "/a/../b", "/a//b", "/a/", "/a;$b", "/a\nb", 3])
def test_protocol_rejects_unsafe_extraction_literal(root):
    value = protocol()
    value["aoti_package_extraction_root"] = root
    with pytest.raises(ValueError, match="extraction root"):
        validate_protocol(value)


def test_benchmark_extraction_reuses_loader_contract_and_requires_private_root(tmp_path):
    tool = benchmark_tool()
    root = tmp_path / "packages"
    root.mkdir(mode=0o700)
    manifest = SimpleNamespace(region_artifacts=[SimpleNamespace(region_name="region", capability=SimpleNamespace(backend="aoti"))],
        backend_versions={"aoti": "2.10.0+cu128"})
    value = protocol()
    assert tool.aoti_extraction_configuration(value, manifest) is None
    value["aoti_package_extraction_root"] = str(root)
    validate_protocol(value)
    configuration = tool.aoti_extraction_configuration(value, manifest)
    assert configuration["root"] == str(root) and configuration["source_sha256"]
    root.chmod(0o755)
    with pytest.raises(ValueError, match="0700"):
        tool.aoti_extraction_configuration(value, manifest)
    root.chmod(0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    value["aoti_package_extraction_root"] = str(alias)
    with pytest.raises(ValueError, match="canonical"):
        tool.aoti_extraction_configuration(value, manifest)
    value["aoti_package_extraction_root"] = str(root)
    manifest.backend_versions["aoti"] = "2.9.0"
    with pytest.raises(ValueError, match="2.10"):
        tool.aoti_extraction_configuration(value, manifest)


@pytest.mark.parametrize("policies", [[], ["off", "off"], ["prefer"], "off"])
def test_bad_policy_subsets_fail_closed(policies):
    with pytest.raises(ValueError, match="policies"):
        protocol_policies({"policies": policies})


def test_quality_and_exactness_are_checked_not_copied_from_protocol():
    value = protocol()
    value.update(quality_gate="passed", eager_validation="bitwise")
    validate_fidelity(value, [{"paper_numeric_gate": True, "eager_bitwise_equal": True}])
    with pytest.raises(ValueError, match="paper numeric"):
        validate_fidelity(value, [{"paper_numeric_gate": False, "eager_bitwise_equal": True}])
    with pytest.raises(ValueError, match="bitwise"):
        validate_fidelity(value, [{"paper_numeric_gate": True, "eager_bitwise_equal": False}])
    with pytest.raises(ValueError, match="missing"):
        validate_fidelity(value, [])


@pytest.mark.parametrize(
    "key,value",
    [
        ("warmup", 127),
        ("warmup", True),
        ("measured", 1000),
        ("processes", 1),
        ("boundary", "sensor-to-action"),
        ("quality_gate", None),
        ("samples", [{}] * 3),
    ],
)
def test_protocol_rejects_unlocked_or_unbalanced_design(key, value):
    source = protocol()
    source[key] = value
    with pytest.raises(ValueError):
        validate_protocol(source)


def row(index):
    return {
        "run": str(index),
        "sample": str(index % 2),
        "measured": str(int(index >= 2)),
        "revision": str(index + 1),
        "latency_ns": "60000000",
        "finite": "1",
        "direct_exact": "1",
        "direct_mse": "0",
        "direct_max_abs": "0",
        "direct_cosine": "1",
        "direct_cosine_defined": "1",
        "eager_mse": "0.00002",
        "eager_max_abs": "0.02",
        "eager_cosine": "0.9998",
        "eager_cosine_defined": "1",
    }


def test_every_fresh_call_is_checked_even_when_eager_quality_fails():
    validate_rows([row(index) for index in range(6)], warmup=2, measured=4, samples=2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("revision", "1"),
        ("sample", "0"),
        ("measured", "0"),
        ("direct_exact", "0"),
        ("finite", "0"),
        ("direct_mse", "1e-10"),
        ("eager_cosine", "nan"),
        ("latency_ns", "0"),
    ],
)
def test_incomplete_or_bad_call_cannot_be_reported_passed(key, value):
    rows = [row(index) for index in range(6)]
    rows[3][key] = value
    with pytest.raises(ValueError):
        validate_rows(rows, warmup=2, measured=4, samples=2)


def test_replay_mode_checks_are_not_interchangeable():
    rows = [{"task_id": 11, "policy": "required", "steps": 10}]
    required, failure = replay_checks(rows)
    other = copy.deepcopy(rows)
    other[0]["policy"] = "batch-only"
    batch, _ = replay_checks(other)
    assert "VLAFORGE_REPLAY_READY" in required and "captured_steps != 10u" in required
    assert "VLAFORGE_REPLAY_UNPREPARED" in batch and "captured_steps != 0u" in batch
    assert "ordinary_count != run + 1u" in batch and "std::_Exit(10)" in failure
    assert replay_checks([]) == ("", "")


def test_resume_recognizes_closed_passing_process(tmp_path):
    tool = benchmark_tool()
    folder = tmp_path / "runs" / "00-off"
    folder.mkdir(parents=True)
    tool.write(folder / "execution.json", {"status": "executed", "exit_code": 0, "pilot": False})
    tool.write(folder / "report.json", {"status": "passed", "pilot": False, "repeat": 0, "policy": "off"})
    assert tool._completed_process(folder, 0, "off", pilot=False)


def test_resume_archives_incomplete_process_without_deleting_evidence(tmp_path):
    tool = benchmark_tool()
    folder = tmp_path / "runs" / "01-required"
    folder.mkdir(parents=True)
    (folder / "stderr.log").write_text("foreign owner\n")
    tool.write(folder / "execution.json", {"status": "incomplete", "pid": 999999999})
    archived = tool._archive_incomplete_process(tmp_path, folder, 1, "required")
    assert not folder.exists()
    assert archived.is_dir() and (archived / "stderr.log").read_text() == "foreign owner\n"


def benchmark_tool():
    spec = importlib.util.spec_from_file_location(
        "session_benchmark_tool",
        Path(__file__).resolve().parents[2] / "tools/benchmark_session.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preflight_never_starts_or_terminates_a_foreign_compute_owner(tmp_path, monkeypatch):
    tool = benchmark_tool()
    monkeypatch.setattr(tool, "verify_frozen", lambda root: None)
    monkeypatch.setattr(tool, "owners", lambda ordinal: [{"pid": 123, "process_name": "another-task"}])
    launched = []
    monkeypatch.setattr(tool.subprocess, "Popen", lambda *args, **kwargs: launched.append(args))
    with pytest.raises(RuntimeError, match="already active"):
        tool.one_process(tmp_path, {"gpu_ordinal": 0}, 0, "off", pilot=True)
    assert not launched and not list(tmp_path.iterdir())


def test_foreign_compute_owner_midrun_only_terminates_own_child(tmp_path, monkeypatch):
    tool = benchmark_tool()
    tool.write(tmp_path / "protocol.json", protocol())
    child = SimpleNamespace(pid=55, returncode=None, terminated=False)

    def terminate():
        child.terminated = True
        child.returncode = -15

    child.terminate = terminate
    child.poll = lambda: child.returncode
    child.wait = lambda **kwargs: child.returncode
    child.kill = lambda: pytest.fail("already terminated child must not be killed again")
    monkeypatch.setattr(tool, "verify_frozen", lambda root: None)
    monkeypatch.setattr(tool, "owners", lambda ordinal: [])
    monkeypatch.setattr(tool, "telemetry", lambda ordinal: {"compute_owners": [{"pid": 123}]})
    monkeypatch.setattr(tool.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(RuntimeError, match="another compute owner"):
        tool.one_process(tmp_path, {"gpu_ordinal": 0, "bundles": {"off": "bundle"}}, 0, "off", pilot=True)
    assert child.terminated
    result = tool.read(tmp_path / "pilot/00-off/execution.json")
    assert result["status"] == "incomplete" and result["foreign_compute_owners"] == [{"pid": 123}]


def test_frozen_provenance_rejects_data_and_binding_manifest_tampering(tmp_path):
    tool = benchmark_tool()
    data = tmp_path / "data.bin"
    data.write_bytes(b"original")
    tool.write(tmp_path / "frozen-files.json", {str(data): tool.sha(data)})
    tool.write(tmp_path / "prepared.json", {"frozen_files_sha256": tool.sha(tmp_path / "frozen-files.json")})
    tool.verify_frozen(tmp_path)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        tool.verify_frozen(tmp_path)
    tool.write(tmp_path / "frozen-files.json", {str(data): tool.sha(data)})
    with pytest.raises(ValueError, match="binding manifest changed"):
        tool.verify_frozen(tmp_path)


def test_zero_norm_cosine_remains_explicitly_undefined():
    rows = [row(index) for index in range(6)]
    rows[0]["direct_cosine_defined"] = "0"
    rows[0]["direct_cosine"] = "0"
    validate_rows(rows, warmup=2, measured=4, samples=2)


def test_aggregate_keeps_processes_separate_and_does_not_promote_quality(tmp_path):
    tool = benchmark_tool()
    locked = protocol()
    tool.write(tmp_path / "protocol.json", locked)
    tool.write(tmp_path / "frozen-files.json", {
        str(tmp_path / "protocol.json"): tool.sha(tmp_path / "protocol.json"),
    })
    tool.write(tmp_path / "prepared.json", {
        "frozen_files_sha256": tool.sha(tmp_path / "frozen-files.json"),
    })
    for repeat, policy in process_order():
        samples = [{"index": index, "repeat_id": repeat, "latency_ns": 60_000_000 + repeat}
                   for index in range(1024)]
        result = {"status": "passed", "pilot": False, "latency": tool.latency_report(samples),
                  "measured_wall_calls_per_second": 16.0, "boundary": locked["boundary"]}
        tool.write(tmp_path / "runs" / f"{repeat:02d}-{policy}" / "report.json", result)
    tool.aggregate(SimpleNamespace(output=tmp_path))
    result = tool.read(tmp_path / "report.json")
    assert result["quality_gate"] == "failed" and not result["full_paper_acceptance"]
    for value in result["policies"].values():
        assert len(value["processes"]) == 5
        assert value["combined"]["summary"]["count"] == 5120
        assert len(value["combined"]["per_repeat"]) == 5
        assert [item["index"] for item in value["combined"]["raw_samples"]] == list(range(5120))
        assert not value["eligible_for_lossless_paper_table"]


def test_bf16_roundtrip_is_explicit_two_byte_and_preserves_signed_zero():
    bits = np.array([0, 0x8000, 0x3f80, 0xbf80, 1, 0x7f7f], dtype="<u2")
    expected = (bits.astype("<u4") << 16).view("<f4").reshape(2, 3)
    raw = encode_reference(expected, "bf16", (2, 3))
    assert raw == bits.tobytes() and len(raw) == 12
    assert decode_tensor(raw, "bf16", (2, 3)).tobytes() == expected.tobytes()
    bad = expected.copy()
    bad.view(np.uint32)[0, 2] |= 1
    with pytest.raises(ValueError, match="losslessly"):
        encode_reference(bad, "bf16", bad.shape)
    with pytest.raises(ValueError, match="FP32"):
        encode_reference(expected.astype(np.float64), "bf16", expected.shape)


@pytest.mark.parametrize("dtype,numpy_dtype", [("f32", "float32"), ("f16", "float16")])
def test_native_floating_reference_no_implicit_cast(dtype, numpy_dtype):
    value = np.array([[0, -0.0, 1.5]], dtype=numpy_dtype)
    raw = encode_reference(value, dtype, value.shape)
    assert decode_tensor(raw, dtype, value.shape).tobytes() == value.tobytes()
    with pytest.raises(ValueError, match="dtype"):
        encode_reference(value.astype(np.float64), dtype, value.shape)


@pytest.mark.parametrize("raw,dtype,shape", [
    (b"\x00", "bf16", (1,)),
    (b"\x80\x7f", "bf16", (1,)),
    (b"\xc0\x7f", "bf16", (1,)),
    (b"\x00\x7c", "f16", (1,)),
    (b"\x02", "bool", (1,)),
    (b"", "i64", (0,)),
    (b"", "f64", (1,)),
])
def test_input_decode_rejects_nonfinite_bad_bool_length_or_shape(raw, dtype, shape):
    with pytest.raises(ValueError):
        decode_tensor(raw, dtype, shape)


def test_integer_and_bool_input_storage_is_not_reinterpreted_as_float():
    value = np.array([2**60 + 1, -3], dtype="<i8")
    assert np.array_equal(decode_tensor(value.tobytes(), "i64", (2,)), value)
    assert decode_tensor(b"\x00\x01", "bool", (2,)).tolist() == [0, 1]


@pytest.mark.parametrize("dimensions", [[], [0, 0], [-1], [3], [True], "0"])
def test_active_dimension_selection_fails_closed(dimensions):
    with pytest.raises(ValueError, match="active dimensions"):
        active_indices((1, 2, 3), dimensions)


def test_primary_active_metrics_cannot_be_diluted_by_inactive_padding():
    left = np.ones((1, 64, 128), dtype=np.float32)
    right = left.copy()
    right[..., 1] += 0.01
    indices = active_indices(left.shape, [1])
    full = numeric_metrics(left, right)
    active = numeric_metrics(left.reshape(-1)[indices], right.reshape(-1)[indices])
    assert full["mse"] < 1e-5 < active["mse"]
    assert active["count"] == 64 and full["count"] == 8192
    assert numeric_metrics([0, 0], [0, 0])["cosine"] is None
    with pytest.raises(ValueError, match="matching"):
        numeric_metrics([1, 2], [1])


def test_output_spec_uses_dtype_and_shape_not_reference_file_size():
    payload = SimpleNamespace(dtype="bf16", shape=(1, 64, 128))
    port = SimpleNamespace(payload=payload, device="cuda:0")
    module = SimpleNamespace(inputs=[port], outputs=[port])
    result = output_spec(module, [0, 4])
    assert result["size_bytes"] == 16384 and result["count"] == 8192
    assert result["raw_file"] == "outputs.bf16" and len(result["active_indices"]) == 128
    declarations, device, count = input_specs(module)
    assert "VLAFORGE_DTYPE_BF16" in declarations and device == 0 and count == 8192
    payload.dtype = "i64"
    with pytest.raises(ValueError, match="complete"):
        output_spec(module)


def test_ordinary_source_fallback_requires_explicit_mode_and_plan_off():
    plan = {"tasks": [{"opcode": "vla.for", "attributes": {}}]}
    validate_loop_policy(None, plan, "off", "ordinary-source")
    with pytest.raises(ValueError, match="label"):
        validate_loop_policy(None, plan, "off")
    with pytest.raises(ValueError, match="unselected"):
        validate_loop_policy(None, plan, "required", "ordinary-source")
    plan["tasks"][0]["attributes"]["replay"] = "required"
    with pytest.raises(ValueError, match="scheduled"):
        validate_loop_policy(None, plan, "off", "ordinary-source")


def test_gpu_monitor_identity_matches_isolated_cuda_mapping():
    locked = protocol()
    locked.update(gpu_ordinal=0, monitor_gpu="GPU-123", cuda_visible_devices="GPU-123")
    validate_protocol(locked)
    locked["cuda_visible_devices"] = "1"
    with pytest.raises(ValueError, match="UUID"):
        validate_protocol(locked)


@pytest.mark.parametrize("dtype", ["bf16", "f16", "f32", "f64"])
def test_actual_cpp_decoder_matches_declared_storage_on_cpu(tmp_path, dtype):
    import shutil
    import subprocess

    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    tool = benchmark_tool()
    template = Path(tool.__file__).with_name("session_benchmark_runner.cpp.in").read_text()
    helpers = template[template.index("struct Metrics"):template.index("\n}\n\nint main")]
    if dtype == "f64":
        raw = np.array([0, 0x8000000000000000, 1, 0x3ff0000000000000,
                        0x7ff0000000000000, 0x7ff8000000000000], dtype="<u8").tobytes()
        expected = np.frombuffer(raw, dtype="<f8")
    elif dtype == "f32":
        raw = np.array([0, 0x80000000, 1, 0x3f800000, 0x7f800000, 0x7fc00000], dtype="<u4").tobytes()
        expected = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    else:
        raw = np.arange(65536, dtype="<u2").tobytes()
        bits = np.frombuffer(raw, dtype="<u2")
        with np.errstate(invalid="ignore"):
            expected = ((bits.astype("<u4") << 16).view("<f4") if dtype == "bf16" else bits.view("<f2")).astype(np.float64)
    (tmp_path / "input.bin").write_bytes(raw)
    source = "\n".join(line for line in template.splitlines() if line.startswith("#include <") and "cuda" not in line)
    source += "\nenum VLAForgeDType { VLAFORGE_DTYPE_F32, VLAFORGE_DTYPE_F16, VLAFORGE_DTYPE_BF16, VLAFORGE_DTYPE_F64 };\n"
    source += f"constexpr auto kOutputDtype = VLAFORGE_DTYPE_{dtype.upper()}; constexpr std::size_t kOutputCount = {expected.size};\n"
    source += "const std::vector<std::size_t> active_indices = {0};\n" + helpers
    source += f'''\nint main(int argc, char** argv) {{
      std::vector<std::uint8_t> bytes({len(raw)}); std::ifstream in(argv[1], std::ios::binary);
      in.read(reinterpret_cast<char*>(bytes.data()), bytes.size()); auto values = Decode(bytes);
      std::ofstream out(argv[2], std::ios::binary);
      out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
    }}\n'''
    (tmp_path / "decode.cpp").write_text(source)
    subprocess.run([compiler, "-std=c++17", str(tmp_path / "decode.cpp"), "-o", str(tmp_path / "decode")], check=True, capture_output=True)
    subprocess.run([str(tmp_path / "decode"), str(tmp_path / "input.bin"), str(tmp_path / "decoded.bin")], check=True)
    actual = np.fromfile(tmp_path / "decoded.bin", dtype="<f8")
    np.testing.assert_equal(actual, expected)
    assert np.array_equal(np.signbit(actual[expected == 0]), np.signbit(expected[expected == 0]))


@pytest.mark.parametrize("active_error", [False, True])
def test_process_independent_bf16_raw_audit_and_primary_gate(tmp_path, monkeypatch, active_error):
    import csv

    tool = benchmark_tool()
    tool.write(tmp_path / "prepared.json", {})
    shape = (1, 2, 128)
    expected = np.ones(shape, dtype=np.float32)
    actual = expected.copy()
    if active_error:
        actual[..., 0] += 1 / 128
    eager = encode_reference(expected, "bf16", shape)
    direct = encode_reference(actual, "bf16", shape)
    port = SimpleNamespace(payload=SimpleNamespace(dtype="bf16", shape=shape), device="cuda:0")
    contract = output_spec(SimpleNamespace(outputs=[port]), [0])
    tool.write(tmp_path / "tensor-contract.json", contract)
    for index in range(16):
        path = tmp_path / "data" / str(index)
        path.mkdir(parents=True)
        (path / "direct.bin").write_bytes(direct)
        (path / "eager.bin").write_bytes(eager)
    locked = protocol()
    locked.update(gpu_ordinal=0, quality_gate="passed", paper_gates={"mse_max": 1e-5, "cosine_min": 0.9999})
    tool.write(tmp_path / "protocol.json", locked)
    complete = numeric_metrics(expected, actual)
    primary = numeric_metrics(expected[..., 0], actual[..., 0])

    def launch(args, stdout, **kwargs):
        folder = Path(args[3])
        output_rows = []
        for index in range(48):
            value = row(index)
            value.update(sample=str(index % 16), measured=str(int(index >= 16)))
            for prefix, result in (("eager", complete), ("active_eager", primary)):
                value.update({prefix + "_" + key: str(result[key]) for key in ("mse", "max_abs", "cosine")})
                value[prefix + "_cosine_defined"] = "1"
            output_rows.append(value)
        writer = csv.DictWriter(stdout, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
        (folder / "outputs.bf16").write_bytes(direct * 48)
        (folder / "process-maps.txt").write_text("libtorch_cpu.so\nlibtorch_cuda.so\n")
        kwargs["stderr"].write("WALL,2880000000,1920000000\n")
        return SimpleNamespace(pid=55, returncode=0, poll=lambda: 0, wait=lambda **kw: 0)

    monkeypatch.setattr(tool, "verify_frozen", lambda root: None)
    monkeypatch.setattr(tool, "owners", lambda ordinal: [])
    monkeypatch.setattr(tool.subprocess, "Popen", launch)
    if active_error:
        assert complete["mse"] < 1e-5 < primary["mse"]
        with pytest.raises(ValueError, match="paper numeric"):
            tool.one_process(tmp_path, locked, 0, "off", pilot=True)
        assert tool.read(tmp_path / "pilot/00-off/report.json")["status"] == "numerical_failed"
    else:
        tool.one_process(tmp_path, locked, 0, "off", pilot=True)
        report = tool.read(tmp_path / "pilot/00-off/report.json")
        assert report["validated_outputs"] == 48 and report["all_eager_bitwise_equal"]
    records = tool.read(tmp_path / "pilot/00-off/fidelity.json")
    assert records[0]["complete_storage_metrics"]["count"] == 256
    assert records[0]["primary_metrics"]["count"] == 2


def handshake_fixture(tmp_path):
    tool = benchmark_tool()
    for stage in ("registered-1", "reset", "registered-2"):
        tool.write(tmp_path / ("owner-" + stage + ".json"), {"pid": 55, "ordinal": 0})
    process = SimpleNamespace(pid=55, poll=lambda: None)
    return tool, process, {"gpu_ordinal": 0, "owner_identity_mode": "cuda-registration-handshake"}


def test_namespace_mapping_requires_register_reset_same_pid_reregistration(tmp_path, monkeypatch):
    tool, process, locked = handshake_fixture(tmp_path)
    observations = iter([[{"pid": 900}], [], [{"pid": 900}]])
    monkeypatch.setattr(tool, "owners", lambda ordinal: next(observations))
    assert tool.register_gpu_owner(process, tmp_path, locked) == 900
    report = tool.read(tmp_path / "owner-handshake.json")
    assert report["status"] == "passed" and report["nvml_pid"] == 900
    assert not report["identity_is_nspid_verified"]
    assert [row["stage"] for row in report["observations"]] == ["registered-1", "reset", "registered-2"]


@pytest.mark.parametrize("observations", [
    [[{"pid": 900}, {"pid": 901}]],
    [[{"pid": 900}], [{"pid": 901}]],
    [[{"pid": 900}], [], [{"pid": 901}]],
])
def test_namespace_mapping_never_accepts_foreign_or_changed_owner(tmp_path, monkeypatch, observations):
    tool, process, locked = handshake_fixture(tmp_path)
    values = iter(observations)
    monkeypatch.setattr(tool, "owners", lambda ordinal: next(values))
    with pytest.raises(ValueError, match="foreign or changed"):
        tool.register_gpu_owner(process, tmp_path, locked)
    assert tool.read(tmp_path / "owner-handshake.json")["status"] == "failed"
    assert not (tmp_path / "continue-registered-2").exists()


@pytest.mark.parametrize("stage", ["registered-1", "reset", "registered-2"])
def test_namespace_mapping_timeout_cannot_skip_a_phase(tmp_path, monkeypatch, stage):
    tool, process, locked = handshake_fixture(tmp_path)
    # Earlier phases succeed; the selected phase cannot reach its expected state.
    values = {"registered-1": [[]], "reset": [[{"pid": 900}], [{"pid": 900}]],
              "registered-2": [[{"pid": 900}], [], []]}[stage]
    sequence = iter(values)
    monkeypatch.setattr(tool, "owners", lambda ordinal: next(sequence))
    with pytest.raises(ValueError, match="timed out"):
        tool.register_gpu_owner(process, tmp_path, locked, timeout=0)
    assert tool.read(tmp_path / "owner-handshake.json")["status"] == "failed"


def test_namespace_marker_must_name_own_child_and_compiled_device(tmp_path, monkeypatch):
    tool, process, locked = handshake_fixture(tmp_path)
    tool.write(tmp_path / "owner-registered-1.json", {"pid": 56, "ordinal": 0})
    monkeypatch.setattr(tool, "owners", lambda ordinal: [{"pid": 900}])
    with pytest.raises(ValueError, match="expected child"):
        tool.register_gpu_owner(process, tmp_path, locked)


def test_registration_context_reset_is_before_any_input_or_session(tmp_path):
    tool = benchmark_tool()
    template = Path(tool.__file__).with_name("session_benchmark_runner.cpp.in").read_text()
    reset = template.index("!Cuda(cudaDeviceReset())")
    assert reset < template.index("std::vector<Buffer> buffers") < template.index("vlaforge_model_session_create_from_bundle")
    assert template.count("cudaDeviceReset()") == 1
