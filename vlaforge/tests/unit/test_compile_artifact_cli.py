import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from vlaforge.cli import _compile_artifact, build_parser


def _compile_fixture(tmp_path, monkeypatch, *, target="sm_90", profile="default"):
    exported = tmp_path / "model.pt2"
    exported.write_bytes(b"export fixture")
    output = tmp_path / "artifact.pt2"
    manifest = tmp_path / "compile.json"
    captured = {}

    def compile_program(program, *, package_path, inductor_configs):
        captured.update(inductor_configs)
        Path(package_path).write_bytes(b"artifact fixture")
        return package_path

    torch = SimpleNamespace(
        __version__="test-only",
        version=SimpleNamespace(cuda="test-only"),
        cuda=SimpleNamespace(
            is_available=lambda: True, get_device_capability=lambda _: (9, 0)
        ),
        export=SimpleNamespace(
            load=lambda _: SimpleNamespace(
                graph_module=SimpleNamespace(graph=SimpleNamespace(nodes=(1, 2)))
            )
        ),
        _inductor=SimpleNamespace(aoti_compile_and_package=compile_program),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "vlaforge.deployment.aoti_export.prepare_backend_program",
        lambda program, configs: (program, {"passes": [], "rewrites": []}),
    )
    monkeypatch.setattr(
        "vlaforge.deployment.aoti_export.prepare_backend_options",
        lambda configs: (dict(configs), {"passes": [], "rewrites": []}),
    )
    monkeypatch.setattr(
        "vlaforge.deployment.aoti_package.finalize_aoti_package",
        lambda path, configs: {"passes": [], "translation": None},
    )
    args = build_parser().parse_args(
        [
            "compile-artifact",
            str(exported),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--target",
            target,
            "--inductor-profile",
            profile,
        ]
    )
    return args, captured, manifest


@pytest.mark.parametrize(
    "profile", ["default", "conservative", "eager-numerics", "aten-preserving"]
)
def test_compile_profile_is_explicit_and_recorded(tmp_path, monkeypatch, profile):
    args, captured, manifest = _compile_fixture(tmp_path, monkeypatch, profile=profile)
    assert _compile_artifact(args) == 0
    record = json.loads(manifest.read_text())
    assert record["inductor_profile"] == profile
    assert record["numeric_parity_verified"] is False
    assert record["backend_graph_rewrites"] == []
    assert record["backend_program_audit"] == {"passes": [], "rewrites": []}
    assert record["inductor_configs"] == captured
    assert captured.get("emulate_precision_casts", False) == (
        profile in ("eager-numerics", "aten-preserving")
    )
    assert captured.get("emulate_divison_rounding", False) == (
        profile in ("eager-numerics", "aten-preserving")
    )
    assert captured.get("epilogue_fusion", True) == (profile == "default")
    assert captured.get("fallback_by_default", False) == (profile == "aten-preserving")
    assert captured.get("selective_decompose", False) == (profile == "aten-preserving")


def test_profiles_are_fresh_and_unknown_names_fail_closed():
    from vlaforge.deployment.aoti_profile import aoti_configs

    first = aoti_configs("aten-preserving")
    assert first["aot_inductor.package"]
    assert not first["pattern_matcher"]
    assert not first["use_pre_grad_passes"]
    assert not first["use_joint_graph_passes"]
    assert first["use_post_grad_passes"]
    assert not first["reorder_for_locality"]
    first["selective_decompose"] = False
    assert aoti_configs("aten-preserving")["selective_decompose"]
    with pytest.raises(ValueError, match="unknown"):
        aoti_configs("missing")


def test_compile_consumes_prepared_program_and_records_its_separate_ledger(tmp_path, monkeypatch):
    args, _, manifest = _compile_fixture(tmp_path, monkeypatch, profile="aten-preserving")
    prepared = object()
    audit = {"passes": [{"stage": "pre_aot_exported_program"}], "rewrites": [{"node": "factory"}]}
    monkeypatch.setattr(
        "vlaforge.deployment.aoti_export.prepare_backend_program",
        lambda program, configs: (prepared, audit),
    )

    def compile_program(program, *, package_path, inductor_configs):
        assert program is prepared
        Path(package_path).write_bytes(b"prepared candidate")
        return package_path

    monkeypatch.setattr(sys.modules["torch"]._inductor, "aoti_compile_and_package", compile_program)
    assert _compile_artifact(args) == 0
    record = json.loads(manifest.read_text())
    assert record["backend_program_audit"] == audit
    assert record["backend_graph_rewrites"] == []


def test_compile_rejects_foreign_gpu_target_before_invoking_backend(
    tmp_path, monkeypatch
):
    args, captured, manifest = _compile_fixture(tmp_path, monkeypatch, target="sm_86")
    with pytest.raises(ValueError, match="current GPU is sm_90"):
        _compile_artifact(args)
    assert not captured
    assert not manifest.exists()


def test_compile_cuda_target_requires_real_cuda(tmp_path, monkeypatch):
    args, _, manifest = _compile_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys.modules["torch"].cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires the target GPU"):
        _compile_artifact(args)
    assert not manifest.exists()
