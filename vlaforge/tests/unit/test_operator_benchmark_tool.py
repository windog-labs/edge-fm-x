import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tool():
    spec = importlib.util.spec_from_file_location(
        "operator_benchmark_tool",
        Path(__file__).parents[2] / "tools/benchmark_operator_examples.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recipes_keep_precision_flags_fixed(tool):
    assert tool.recipe_configs("aten") is None
    base = tool.recipe_configs("inductor-aten")
    tuned = tool.recipe_configs("inductor-autotune")
    assert base["emulate_precision_casts"] and base["emulate_divison_rounding"]
    assert all(
        base[key] == tuned[key] for key in base if key != "max_autotune_gemm_backends"
    )
    assert tuned["max_autotune"] and tuned["max_autotune_gemm"]
    preserving = tool.recipe_configs("inductor-aten-preserving")
    assert preserving["fallback_by_default"]
    assert preserving["selective_decompose"]
    assert not preserving["pattern_matcher"]
    with pytest.raises(ValueError):
        tool.recipe_configs("unknown")


def test_registered_worker_handshake_precedes_backend_imports(tool, monkeypatch, tmp_path):
    events = []
    monkeypatch.setenv("COGACT_GPU_OWNER_FOLDER", str(tmp_path))
    monkeypatch.setitem(sys.modules, "cogact_gpu_monitor", SimpleNamespace(child_handshake=lambda: events.append("registered")))
    monkeypatch.setattr(sys, "argv", ["bench", "--examples", str(tmp_path / "examples.json"), "--node", "one",
        "--recipe", "aten", "--output", str(tmp_path / "out")])
    def check(*args):
        assert events == ["registered"]
        raise ValueError("stopped before backend imports")
    monkeypatch.setattr(tool, "verify_example", check)
    with pytest.raises(ValueError, match="stopped before backend"):
        tool.main()


@pytest.mark.parametrize("mutation", [None, "file", "status", "duplicate", "traversal"])
def test_workload_provenance_is_a_hard_gate(tool, tmp_path, mutation):
    folder = tmp_path / "linear"
    folder.mkdir()
    (folder / "inputs.pt").write_bytes(b"test-values")
    (folder / "operator.pt2").write_bytes(b"test-graph")
    record = {
        "node": "linear",
        "inputs_sha256": tool.digest(folder / "inputs.pt"),
        "export_sha256": tool.digest(folder / "operator.pt2"),
    }
    manifest = {
        "schema": "vlaforge.actual_operator_examples/1",
        "status": "extracted_and_eager_verified",
        "examples": [record],
    }
    if mutation == "file":
        (folder / "inputs.pt").write_bytes(b"different")
    if mutation == "status":
        manifest["status"] = "extracting"
    if mutation == "duplicate":
        manifest["examples"].append(record)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(manifest))
    if mutation:
        with pytest.raises(ValueError):
            tool.verify_example(
                path, "../linear" if mutation == "traversal" else "linear"
            )
    else:
        assert tool.verify_example(path, "linear") == record


def test_graph_benchmark_uses_capture_compatible_loader(tool, monkeypatch):
    calls = []
    backend = SimpleNamespace(
        aoti_load_package=lambda *args, **kw: calls.append((args, kw))
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(_inductor=backend))
    monkeypatch.setattr(importlib, "import_module", lambda name: None)
    tool.load_aoti_candidate(Path("candidate.pt2"))
    assert calls == [
        (("candidate.pt2",), {"run_single_threaded": True, "device_index": 0})
    ]


def test_standalone_package_load_initializes_codecache_before_loading(tool, monkeypatch):
    backend = SimpleNamespace()
    events = []

    def initialize(name):
        assert name == "torch._inductor.codecache"
        backend.codecache = object()
        events.append("codecache")

    def load(*args, **kwargs):
        assert hasattr(backend, "codecache"), "standalone package loader needs codecache"
        events.append("load")
        return "loaded"

    backend.aoti_load_package = load
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(_inductor=backend))
    monkeypatch.setattr(importlib, "import_module", initialize)
    assert tool.load_aoti_candidate(Path("candidate.pt2")) == "loaded"
    assert events == ["codecache", "load"]


def test_serialized_devices_are_never_blanket_remapped(tool, monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    tool.load_example_values(Path("inputs.pt"))
    assert calls == [((Path("inputs.pt"),), {"weights_only": True})]


@pytest.mark.parametrize("source", [None, "another GPU"])
def test_platform_transport_is_explicit_and_does_not_relabel_source(tool, source):
    manifest = {"gpu": source}
    with pytest.raises(ValueError, match="actual GPU"):
        tool.validate_reference_platform(manifest, "local GPU", "captured")
    tool.validate_reference_platform(manifest, "local GPU", "revalidate-workload")
    assert manifest == {"gpu": source}
    with pytest.raises(ValueError, match="reference mode"):
        tool.validate_reference_platform(manifest, "local GPU", "pretend")


def test_output_identity_covers_complete_bytes_dtype_shape_and_tree(tool):
    import torch

    left = (torch.tensor([0.0, 1.0]),)
    assert tool.output_identity(left) == tool.output_identity(
        tuple(x.clone() for x in left)
    )
    assert tool.output_identity(left) != tool.output_identity(
        (torch.tensor([-0.0, 1.0]),)
    )
    assert tool.output_identity(left) != tool.output_identity((left[0].double(),))
    assert tool.output_identity(left) != tool.output_identity((left[0].reshape(1, 2),))
    assert tool.output_identity(left) != tool.output_identity(list(left))
    assert all(item["bitwise_equal"] for item in tool.output_metrics(left, left))
    with pytest.raises(ValueError, match="structure"):
        tool.output_metrics(left, list(left))
    with pytest.raises(ValueError, match="tensor outputs"):
        tool.output_identity(())


def test_singleton_solver_index_output_preserves_complete_logical_bytes(tool):
    import torch

    value = torch.tensor([7], dtype=torch.int64).as_strided((1,), (1600,))
    canonical = torch.tensor([7], dtype=torch.int64)
    assert tool.output_identity((value,)) == tool.output_identity((canonical,))
    assert tool.output_metrics((value,), (canonical,))[0]["bitwise_equal"]
    assert value.stride() == (1600,)


@pytest.mark.parametrize(
    "field",
    [
        "reference_mode",
        "device_loading",
        "target_numerical_context",
        "target_reference_identity",
        "effect_audit_source_sha256",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "different"])
def test_v2_reuse_cannot_drop_target_reference_or_device_contract(
    tool, tmp_path, field, mutation
):
    current = {
        key: "fixture"
        for key in (
            "recipe",
            "inductor_configs",
            "source_manifest_sha256",
            "example",
            "torch",
            "cuda",
            "gpu",
            "compute_capability",
            "matmul_precision",
            "reference_mode",
            "device_loading",
            "target_numerical_context",
            "target_reference_identity",
            "effect_audit_source_sha256",
        )
    }
    current["schema"] = "vlaforge.operator_microbenchmark/2"
    previous = dict(current)
    if mutation == "missing":
        del previous[field]
    else:
        previous[field] = "another value"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match=field):
        tool.verified_reuse(path, current)


@pytest.mark.parametrize(
    "change",
    [
        None,
        "missing",
        "failed",
        "wrong_digest",
        "pre_missing",
        "pre_source",
        "pre_ledger",
    ],
)
def test_aten_preserving_reuse_requires_completed_bound_translation(
    tool, tmp_path, monkeypatch, change
):
    from vlaforge.deployment import aoti_package
    from vlaforge.deployment.aoti_export import (
        backend_pass_records,
        backend_program_pass_records,
    )

    configs = tool.recipe_configs("inductor-aten-preserving")
    artifact = tmp_path / "compiled.pt2"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("model/code.so", b"CPU fixture only")
    monkeypatch.setattr(
        aoti_package, "native_enum_maps", lambda: ({}, {"torch": "2.10.0"})
    )
    audit = aoti_package.finalize_aoti_package(artifact, configs)
    current = {
        "schema": "vlaforge.operator_microbenchmark/1",
        "recipe": "inductor-aten-preserving",
        "inductor_configs": configs,
        "source_manifest_sha256": "fixture",
        "example": {"node": "linear"},
        "torch": "2.10.0",
        "cuda": "fixture",
        "gpu": "fixture",
        "compute_capability": [8, 6],
        "matmul_precision": "highest",
    }
    previous = {
        **current,
        "artifact_sha256": tool.digest(artifact),
        "backend_package_audit": audit,
        "backend_graph_audit": {"passes": backend_pass_records(configs)},
        "backend_program_audit": {
            "passes": backend_program_pass_records(configs),
            "rewrites": [],
        },
    }
    if change == "missing":
        del audit["translation"]
    elif change == "failed":
        audit["translation"]["status"] = "failed"
    elif change == "wrong_digest":
        audit["translation"]["artifact_sha256"] = "f" * 64
    elif change == "pre_missing":
        del previous["backend_program_audit"]
    elif change == "pre_source":
        previous["backend_program_audit"]["passes"][0]["source_sha256"] = (
            "old-implementation"
        )
    elif change == "pre_ledger":
        del previous["backend_program_audit"]["rewrites"]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(previous))
    if change is None:
        assert tool.verified_reuse(report, current) == artifact
    else:
        with pytest.raises(ValueError, match="package translation|program preparation"):
            tool.verified_reuse(report, current)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "artifact",
        "inductor_configs",
        "example",
        "matmul_precision",
        "gpu",
        "backend_graph_audit",
    ],
)
def test_compiled_reuse_checks_identity_but_does_not_require_old_timing_pass(
    tool, tmp_path, mutation
):
    artifact = tmp_path / "compiled.pt2"
    artifact.write_bytes(b"test candidate")
    current = {
        "schema": "vlaforge.operator_microbenchmark/1",
        "recipe": "inductor-aten",
        "inductor_configs": {"force_same_precision": True},
        "source_manifest_sha256": "manifest",
        "example": {"node": "linear"},
        "torch": "test-only",
        "cuda": "test-only",
        "gpu": "test-gpu",
        "compute_capability": [9, 0],
        "matmul_precision": "highest",
    }
    previous = {**current, "artifact_sha256": tool.digest(artifact), "status": "failed"}
    if mutation == "artifact":
        artifact.write_bytes(b"changed candidate")
    elif mutation == "backend_graph_audit":
        previous[mutation] = {"passes": [{"source_sha256": "wrong"}]}
    elif mutation:
        previous[mutation] = "different"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(previous))
    if mutation:
        with pytest.raises(ValueError):
            tool.verified_reuse(path, current)
    else:
        assert tool.verified_reuse(path, current) == artifact
