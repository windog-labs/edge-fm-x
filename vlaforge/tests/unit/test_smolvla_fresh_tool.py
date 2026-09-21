import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vlaforge.ir.program import InputPort, OutputPort
from vlaforge.ir.types import TensorType

spec = importlib.util.spec_from_file_location(
    "smolvla_fresh_tool",
    Path(__file__).resolve().parents[2] / "tools/build_real_smolvla_fresh.py",
)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.mark.parametrize("profile", ["eager-numerics", "aten-preserving"])
@pytest.mark.parametrize("change", [None, "config", "pass", "missing_configs", "translation", "pre_pass"])
def test_artifact_resume_binds_exact_backend_configuration(tmp_path, monkeypatch, profile, change):
    from vlaforge.deployment import aoti_package
    from vlaforge.deployment.aoti_export import backend_pass_records, backend_program_pass_records
    from vlaforge.deployment.aoti_package import package_pass_records
    from vlaforge.deployment.aoti_profile import aoti_configs

    folder = tmp_path / "artifacts"
    folder.mkdir()
    package = folder / "region.pt2"
    package.write_bytes(b"compiled-package")
    configs = aoti_configs(profile)
    package_audit = {"passes": package_pass_records(configs)}
    if profile == "aten-preserving":
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("model/code.so", b"isolated fixture code")
        monkeypatch.setattr(aoti_package, "native_enum_maps", lambda: ({}, {"torch": "2.10.0"}))
        package_audit = aoti_package.finalize_aoti_package(package, configs)
    record = {
        "status": "passed",
        "target": "sm_86",
        "inductor_profile": profile,
        "inductor_configs": configs,
        "backend_graph_passes": backend_pass_records(configs),
        "backend_program_audit": {"passes": backend_program_pass_records(configs), "rewrites": []},
        "backend_package_audit": package_audit,
        "exported_program": {"sha256": "source"},
        "artifact": {
            "sha256": tool.digest(package),
            "size_bytes": package.stat().st_size,
        },
    }
    if change == "config":
        record["inductor_configs"]["epilogue_fusion"] = True
    elif change == "pass":
        record["backend_graph_passes"] = [{"source_sha256": "old-implementation"}]
    elif change == "pre_pass":
        record["backend_program_audit"]["passes"] = [{"source_sha256": "old-implementation"}]
    elif change == "missing_configs":
        del record["inductor_configs"]
    elif change == "translation":
        record["backend_package_audit"]["translation"] = {"status": "failed", "artifact_sha256": "wrong"}
    tool.write_json(folder / "region.compile.json", record)
    region = {"name": "region", "export_sha256": "source"}
    if change is None:
        assert tool.verify_artifact(tmp_path, region, "sm_86", profile) == record
    else:
        with pytest.raises(ValueError, match="source/profile|package translation"):
            tool.verify_artifact(tmp_path, region, "sm_86", profile)


def pack(tmp_path, mode="published-compatibility"):
    values = {
        "noise": np.ones((1, 4, 3), dtype=np.float32),
        "observation.state": np.ones((1, 2), dtype=np.float32),
    }
    sample = tmp_path / "observation.npz"
    np.savez(sample, **values)
    record = {
        "path": sample.name,
        "sha256": tool.digest(sample),
        "noise_sha256": hashlib.sha256(values["noise"].tobytes()).hexdigest(),
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in values.items()
        },
        "state_normalization": {"exact_formula_match": True}
        if mode == "strict-statistics"
        else None,
    }
    manifest = {
        "schema": "vlaforge.smolvla_observations/2",
        "status": "materialized_tensor_boundary_only",
        "normalization": {
            "mode": mode,
            "normalization_statistics_verified": mode == "strict-statistics",
        },
        "noise_semantics": "saved_float32_tensor_is_authoritative",
        "records": [record],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


@pytest.mark.parametrize("mode", ["published-compatibility", "strict-statistics"])
def test_only_explicit_v2_processing_modes_are_accepted(tmp_path, mode):
    path, expected = pack(tmp_path, mode)
    assert tool.verify_input_pack(path, processing_mode=mode) == expected
    values = tool.load_observation(
        path.parent / expected["records"][0]["path"], expected["records"][0]
    )
    assert values["noise"].shape == (1, 4, 3)
    with pytest.raises(ValueError, match="differs"):
        tool.verify_input_pack(path, processing_mode="different")


@pytest.mark.parametrize(
    "change", ["v1", "wrong_status", "false_normalization", "wrong_noise", "tamper"]
)
def test_published_pack_cannot_silently_promote_its_semantics(tmp_path, change):
    path, value = pack(tmp_path)
    if change == "v1":
        value["schema"] = "vlaforge.smolvla_observations/1"
    elif change == "wrong_status":
        value["status"] = "verified_physical"
    elif change == "false_normalization":
        value["normalization"]["normalization_statistics_verified"] = True
    elif change == "wrong_noise":
        value["noise_semantics"] = "seed_only"
    else:
        (tmp_path / "observation.npz").write_bytes(b"tampered")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        tool.verify_input_pack(path, processing_mode="published-compatibility")


def test_strict_pack_requires_each_actual_state_transform_check(tmp_path):
    path, value = pack(tmp_path, "strict-statistics")
    value["records"][0]["state_normalization"] = None
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="state-transform"):
        tool.verify_input_pack(path, processing_mode="strict-statistics")


def test_identical_weights_do_not_allow_mismatched_processing_profiles(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"same-model")
    processor = tmp_path / "policy_preprocessor.json"
    processor.write_text('{"stats":"published"}')
    manifest = {
        "checkpoint_sha256": tool.digest(tmp_path / "model.safetensors"),
        "processor_files": {processor.name: tool.digest(processor)},
    }
    tool.verify_policy_provenance(tmp_path, manifest)
    processor.write_text('{"stats":"recovered"}')
    with pytest.raises(ValueError, match="processor files differ"):
        tool.verify_policy_provenance(tmp_path, manifest)


def test_recovery_provenance_must_match_input_pack_exactly(tmp_path, monkeypatch):
    from vlaforge.adapters.smolvla import smolvla_migration

    claimed = {"robot_type": "so100", "report_sha256": "input-profile"}
    actual = {**claimed, "report_sha256": "different-profile"}
    monkeypatch.setattr(
        smolvla_migration, "verify_recovered_profile", lambda *a, **k: actual
    )
    with pytest.raises(ValueError, match="recovered policy differs"):
        tool.verify_policy_provenance(tmp_path, {"statistics_recovery": claimed})


def test_record_path_cannot_escape_input_pack(tmp_path):
    root = tmp_path / "pack"
    root.mkdir()
    path, value = pack(root)
    outside = tmp_path / "outside.npz"
    outside.write_bytes((root / "observation.npz").read_bytes())
    value["records"][0]["path"] = "../outside.npz"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="path or digest"):
        tool.verify_input_pack(path, processing_mode="published-compatibility")


def test_technical_pass_does_not_mean_paper_numerical_pass():
    reference = np.ones((1, 50, 6), dtype=np.float32)
    actual = reference + 0.005
    result = tool.fidelity(
        reference,
        actual,
        sample_id="sample",
        space="tensor",
        max_abs=0.05,
        mean_abs=0.01,
    )
    assert result["gates"]["technical_tolerance"]["passed"]
    assert not result["gates"]["paper_numerics"]["passed"]
    assert not result["gates"]["full_paper_acceptance"]
    assert result["candidate"] and result["per_dimension"]


def test_bitwise_gate_detects_signed_zero_without_conflating_numerics():
    reference = np.zeros((1, 2, 2), dtype=np.float32)
    actual = -reference
    result = tool.fidelity(
        reference, actual, sample_id="zero", space="tensor", max_abs=0, mean_abs=0
    )
    assert result["metrics"]["exact_values"]
    assert not result["gates"]["bitwise_equal"]
    assert not result["gates"]["paper_numerics"]["passed"]


def test_rescaled_actions_do_not_inherit_tensor_output_tolerances():
    reference = np.ones((1, 2, 2), dtype=np.float32)
    result = tool.fidelity(
        reference,
        reference.copy(),
        sample_id="scale",
        space="uncalibrated_action_scale",
        max_abs=0.05,
        mean_abs=0.01,
        tensor_tolerances=False,
    )
    assert result["gates"]["bitwise_equal"]
    for gate in ("technical_tolerance", "paper_numerics"):
        assert result["gates"][gate]["passed"] is None
        assert not result["gates"][gate]["applicable"]
    assert not result["gates"]["full_paper_acceptance"]


def test_session_labels_keep_failed_runs_and_bundles_separate(tmp_path):
    assert tool.session_paths(tmp_path, None) == (
        tmp_path / "bundle",
        tmp_path / "session",
    )
    bundle, session = tool.session_paths(tmp_path, "copy-ordering-fixed")
    assert bundle == tmp_path / "session-attempts/copy-ordering-fixed/bundle"
    assert session == tmp_path / "session-attempts/copy-ordering-fixed/session"


def test_cpp_input_verification_is_read_only_and_checks_exact_bytes(tmp_path):
    folder = tmp_path / "inputs/sample-000000"
    folder.mkdir(parents=True)
    value = np.ones((1, 2), dtype=np.float32)
    np.savez(folder / "inputs.npz", state=value)
    (folder / "state.bin").write_bytes(value.tobytes())
    records = [{"path": "inputs/sample-000000"}]
    before = {p.name: tool.digest(p) for p in folder.iterdir()}
    tool.verify_cpp_input_bytes(tmp_path, records)
    assert {p.name: tool.digest(p) for p in folder.iterdir()} == before
    (folder / "state.bin").write_bytes((-value).tobytes())
    with pytest.raises(ValueError, match=r"C\+\+ input bytes"):
        tool.verify_cpp_input_bytes(tmp_path, records)


@pytest.mark.parametrize(
    "label", ["", ".", "..", "../other", "/tmp/x", "a/b", "x" * 81]
)
def test_session_labels_reject_escaping_or_ambiguous_names(tmp_path, label):
    with pytest.raises(ValueError, match="Session label"):
        tool.session_paths(tmp_path, label)


def test_explicit_session_policy_requires_bound_evidence(tmp_path):
    expected = {"loop_execution": "required", "artifact_sha256": "same-package"}
    with pytest.raises(ValueError, match="no compiler policy evidence"):
        tool.verify_session_selection(tmp_path, expected)
    path = tmp_path / "evidence/session-selection.json"
    tool.write_json(path, expected)
    tool.verify_session_selection(tmp_path, expected)
    for change in ({"loop_execution": "off"}, {"artifact_sha256": "other-package"}):
        with pytest.raises(ValueError, match="policy or captured artifacts changed"):
            tool.verify_session_selection(tmp_path, {**expected, **change})


def test_legacy_bundle_is_only_accepted_with_source_policy(tmp_path):
    tool.verify_session_selection(tmp_path, {"loop_execution": "source"})
    with pytest.raises(ValueError, match="new label"):
        tool.verify_session_selection(tmp_path, {"loop_execution": "off"})


@pytest.mark.parametrize("policy", ["batch-only", "required"])
def test_cpp_loop_audit_requires_correct_mode_and_isolates_fatal_process(policy):
    audit, failure = tool.replay_runner_fragments(
        [{"task_id": 11, "policy": policy, "steps": 10}]
    )
    assert "vlaforge_model_session_get_replay_info(session, 11u" in audit
    assert f"REPLAY,%zu,11,{policy}" in audit
    assert "VLAFORGE_REPLAY_POISONED" in failure and "std::_Exit(10)" in failure
    if policy == "required":
        assert (
            "info.captured_steps == 10u" in audit
            and "info.ordinary_count == 0u" in audit
        )
    else:
        assert (
            "info.captured_steps == 0u" in audit
            and "info.ordinary_count == run + 1u" in audit
        )


def test_absent_loop_policy_does_not_change_ordinary_harness_fragments():
    assert tool.replay_runner_fragments([]) == ("", "")


def test_cpp_harness_dimensions_and_dtype_come_from_the_module():
    module = SimpleNamespace(
        inputs=(
            InputPort("camera", TensorType((2, 3, 480, 640), "f32"), device="cuda:0"),
            InputPort("mask", TensorType((2,), "bool"), device="cuda:0"),
        ),
        outputs=(
            OutputPort("action_chunk", TensorType((2, 7, 5), "f32"), device="cuda:0"),
        ),
    )
    source = tool.runner_source(module, 3, "cuda:0")
    assert "@" not in source
    assert "kSamples = 3u" in source and "kOutputCount = 70u" in source
    assert '"camera", VLAFORGE_DTYPE_F32, {2,3,480,640}, 7372800u' in source
    assert "api->bind_tensor" in source and "api->read_output_tensor" in source
    assert "stamp.revision = run + 1u" in source
