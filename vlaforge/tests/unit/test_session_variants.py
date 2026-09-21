import copy
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from vlaforge.validation.session_benchmark import BOUNDARY, SCHEMA


def tool(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "tools"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("session_variants_tool", directory / "build_session_variants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_policy_variants_preserve_reference_and_measurement_contract(monkeypatch):
    api = tool(monkeypatch)
    original = {"schema": SCHEMA, "boundary": BOUNDARY, "warmup": 128, "measured": 1024,
                "processes": 5, "policies": ["off"], "bundles": {"off": "original"},
                "quality_gate": "failed", "gpu_ordinal": 0,
                "evidence": ["official.json"], "samples": [{}] * 16}
    before = copy.deepcopy(original)
    variants = {mode: "/new/" + mode for mode in ("off", "batch-only", "required")}
    result = api.variant_protocol(original, variants, "GPU-new", ["build.json"])
    assert original == before
    assert result["bundles"] == variants and result["policies"] == list(variants)
    assert result["evidence"] == ["official.json", "build.json"]
    assert result["monitor_gpu"] == result["cuda_visible_devices"] == "GPU-new"
    for key in ("samples", "quality_gate", "boundary", "warmup", "measured", "processes"):
        assert result[key] == original[key]


def test_rebuild_rejects_changed_source_before_creating_outputs(tmp_path, monkeypatch):
    api = tool(monkeypatch)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle.json").write_text("{}")
    args = SimpleNamespace(bundle=bundle, bundle_sha256="0" * 64, output=tmp_path / "output")
    with pytest.raises(ValueError, match="SHA changed"):
        api.build(args)
    assert not args.output.exists()


def test_source_fingerprint_excludes_generated_python_cache(tmp_path, monkeypatch):
    api = tool(monkeypatch)
    monkeypatch.setattr(api, "SOURCE", tmp_path)
    (tmp_path / "CMakeLists.txt").write_text("source")
    code = tmp_path / "python/model.py"
    code.parent.mkdir()
    code.write_text("source")
    cache = code.parent / "__pycache__/model.py"
    cache.parent.mkdir()
    cache.write_text("not source")
    assert set(api.source_files()) == {"CMakeLists.txt", "python/model.py"}


@dataclass(frozen=True)
class Artifact:
    region_name: str
    region_id: int
    io_schema_digest: str = "schema"
    artifact_sha256: str = "weights"


def test_canonical_region_order_rebinds_ids_without_changing_artifact_contract(monkeypatch):
    api = tool(monkeypatch)
    module = SimpleNamespace(regions=[SimpleNamespace(name="a"), SimpleNamespace(name="z")])
    monkeypatch.setattr(api, "io_schema_digest", lambda _: "schema")
    artifacts = [Artifact("z", 0), Artifact("a", 1)]
    result = api.bind_region_declarations(module, artifacts)
    assert result == {"a": replace(artifacts[1], region_id=0), "z": replace(artifacts[0], region_id=1)}
    with pytest.raises(ValueError, match="declarations"):
        api.bind_region_declarations(module, artifacts[:1])
    with pytest.raises(ValueError, match="schema identity"):
        api.bind_region_declarations(module, [replace(item, io_schema_digest="changed") for item in artifacts])


def cuda_torchscript_contract():
    from vlaforge.deployment import (
        ArtifactIdentity,
        ArtifactKind,
        EffectAudit,
        RegionArtifactContract,
        WorkspaceContract,
    )
    from vlaforge.deployment.capabilities import torchscript_backend_capability

    return RegionArtifactContract(
        region_id=0, region_name="generic_region", inputs=(), outputs=(),
        io_schema_digest="0" * 64,
        identity=ArtifactIdentity("test-only", "revision", "fixture:no-checkpoint", "1" * 64),
        artifact_kind=ArtifactKind.TORCHSCRIPT_ARCHIVE, artifact_path="artifacts/region.pt",
        artifact_sha256="2" * 64, artifact_size_bytes=17,
        workspace=WorkspaceContract(device="cuda:0"), effect_audit=EffectAudit(),
        capability=torchscript_backend_capability("sm_90", ("bf16", "f32")),
        backend_variant="torchscript-aten/1",
    )


def test_context_selection_is_explicit_and_preserves_computation_and_audit(monkeypatch):
    api = tool(monkeypatch)
    original = cuda_torchscript_contract()
    contracts = {original.region_name: original}
    assert api.bind_execution_profile(contracts) == contracts
    selected = api.bind_execution_profile(contracts, torchscript_shared_context=True)
    result = selected[original.region_name]
    assert result.backend_variant == "torchscript-aten-context/1"
    assert result.capability.supports_external_cuda_graph
    assert result.capability.supports_execution_context
    assert replace(result, capability=original.capability, backend_variant=original.backend_variant) == original
    assert contracts[original.region_name] is original
    assert api.bind_execution_profile(selected, torchscript_shared_context=True) == selected


@pytest.mark.parametrize("change", ["cpu", "workspace", "dynamic", "variant", "kind", "residency", "flags"])
def test_context_selection_rejects_unsupported_or_inconsistent_profiles(monkeypatch, change):
    from vlaforge.deployment.capabilities import torchscript_backend_capability
    from vlaforge.deployment.contract import (
        ArtifactKind,
        ArtifactResidency,
        WorkspaceContract,
    )

    api = tool(monkeypatch)
    original = cuda_torchscript_contract()
    if change == "cpu":
        original = replace(original, capability=torchscript_backend_capability("cpu", ("f32",)))
    elif change == "workspace":
        original = replace(original, workspace=WorkspaceContract(device="cpu"))
    elif change == "dynamic":
        original = replace(original, capability=replace(original.capability, supports_dynamic_shapes=True))
    elif change == "variant":
        original = replace(original, backend_variant="unknown/99")
    elif change == "kind":
        original = replace(original, artifact_kind=ArtifactKind.SHARED_LIBRARY)
    elif change == "residency":
        original = replace(original, residency=ArtifactResidency.INVOCATION)
    else:
        original = replace(original, capability=replace(original.capability, supports_execution_context=True))
    with pytest.raises(ValueError, match="static Session-resident CUDA"):
        api.bind_execution_profile({original.region_name: original}, torchscript_shared_context=True)


def test_context_selection_preserves_other_backends_and_rejects_empty_selection(monkeypatch):
    api = tool(monkeypatch)
    original = cuda_torchscript_contract()
    other = replace(original, region_name="other_backend",
                    capability=replace(original.capability, backend="aoti"))
    contracts = {original.region_name: original, other.region_name: other}
    result = api.bind_execution_profile(contracts, torchscript_shared_context=True)
    assert result[other.region_name] is other
    with pytest.raises(ValueError, match="matched no Regions"):
        api.bind_execution_profile({other.region_name: other}, torchscript_shared_context=True)


def test_materialization_reuses_public_contract_and_preserves_other_backends(tmp_path, monkeypatch):
    from vlaforge.deployment import aoti_materialized
    from vlaforge.deployment.contract import ArtifactKind

    api = tool(monkeypatch)
    source = tmp_path / "original"
    original = replace(cuda_torchscript_contract(), artifact_kind=ArtifactKind.AOTI_PACKAGE)
    other = replace(cuda_torchscript_contract(), region_name="other")
    contracts = {original.region_name: original, other.region_name: other}
    calls = []
    projected = replace(original, artifact_kind=ArtifactKind.AOTI_MATERIALIZED)

    def materialize(path, output, **options):
        assert path == source / original.artifact_path
        assert options == {"sha256": original.artifact_sha256, "size_bytes": original.artifact_size_bytes}
        output.mkdir(parents=True)
        manifest = output / "model.vfaoti"
        manifest.write_text("checked test payload")
        calls.append(manifest)
        return manifest

    def rebind(contract, path, *, artifact_path):
        assert contract is original and path == calls[0]
        assert artifact_path == "artifacts/materialized-0/model.vfaoti"
        return projected

    monkeypatch.setattr(aoti_materialized, "materialize_aoti_package", materialize)
    monkeypatch.setattr(aoti_materialized, "materialized_region_contract", rebind)
    selected, paths, records = api.materialize_contracts(contracts, source, tmp_path / "new")
    assert selected == {original.region_name: projected, other.region_name: other}
    assert contracts[original.region_name] is original
    assert paths[original.region_name] == calls[0]
    assert paths[other.region_name] == source / other.artifact_path
    assert records[0]["source_artifact_sha256"] == original.artifact_sha256
    assert records[0]["runtime_extraction"] is False


def test_materialization_rejects_an_empty_selection(tmp_path, monkeypatch):
    api = tool(monkeypatch)
    original = cuda_torchscript_contract()
    with pytest.raises(ValueError, match="matched no AOTI"):
        api.materialize_contracts({original.region_name: original}, tmp_path, tmp_path / "new")
