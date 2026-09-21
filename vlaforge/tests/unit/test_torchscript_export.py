import pytest
from vlaforge.codegen.model import CppArtifactRegionDefinition
from vlaforge.deployment.capabilities import torchscript_backend_capability


def _definition(**changes):
    values = {"region_name": "region", "backend": "torchscript", "artifact_path": "artifacts/region.pt",
              "artifact_sha256": "a" * 64, "artifact_size_bytes": 4, "io_schema_digest": "b" * 64,
              "target": "cpu", "device": "cpu", "backend_variant": "torchscript-aten/1"}
    return CppArtifactRegionDefinition(**(values | changes))


@pytest.mark.parametrize("changes", [
    {"backend_variant": "default"}, {"target": "sm_86"},
    {"device": "cuda:0"}, {"residency": "invocation"},
    {"supports_external_cuda_graph": True}, {"supports_execution_context": True},
])
def test_profile_rejects_unsupported_contract(changes):
    with pytest.raises(ValueError):
        _definition(**changes)


def test_profile_is_explicit_and_synchronous():
    _definition()
    _definition(target="sm_86", device="cuda:0")
    capability = torchscript_backend_capability("sm_86", ("f32", "bf16"))
    assert capability.requires_synchronize
    assert not capability.supports_external_cuda_graph
    assert not capability.supports_execution_context


def test_optional_context_variant_requires_cuda_and_explicit_contract():
    _definition(target="sm_86", device="cuda:0", backend_variant="torchscript-aten-context/1",
                supports_execution_context=True, supports_external_cuda_graph=True)
    with pytest.raises(ValueError, match="CUDA"):
        _definition(backend_variant="torchscript-aten-context/1", supports_execution_context=True)
    with pytest.raises(ValueError, match="shared context"):
        _definition(target="sm_86", device="cuda:0", backend_variant="torchscript-aten-context/1")
    with pytest.raises(ValueError, match="CUDA"):
        torchscript_backend_capability("cpu", ("f32",), shared_context=True)
    capability = torchscript_backend_capability("sm_86", ("f32",), shared_context=True)
    assert capability.supports_execution_context and capability.supports_external_cuda_graph


def test_static_tensor_region_serializes_and_validates(tmp_path):
    torch = pytest.importorskip("torch")
    from vlaforge.deployment.torchscript_export import export_torchscript_region

    class Module(torch.nn.Module):
        def forward(self, values):
            return values.sin(), values.square()

    inputs = (torch.arange(4, dtype=torch.float32),)
    program = torch.export.export(Module(), inputs)
    caller = torch._C._get_graph_executor_optimize()
    audit = export_torchscript_region(program, tmp_path / "model.pt",
                                     validation_cases=((inputs[0] + .1,),))
    assert audit["status"] == "region_cases_passed"
    assert len(audit["validation_cases"]) == 2
    assert not audit["full_model_verified"]
    assert torch._C._get_graph_executor_optimize() == caller
    with pytest.raises(FileExistsError):
        export_torchscript_region(program, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="shape/dtype/device"):
        export_torchscript_region(program, tmp_path / "wrong.pt",
                                  validation_cases=((torch.ones(2),),))
    assert not (tmp_path / "wrong.pt").exists()


@pytest.mark.parametrize("functionalized", [False, True])
def test_persistent_mutation_is_not_a_region(tmp_path, functionalized):
    torch = pytest.importorskip("torch")
    from vlaforge.deployment.torchscript_export import export_torchscript_region

    class Mutating(torch.nn.Module):
        def forward(self, values):
            values.add_(1)
            return values

    program = torch.export.export(Mutating(), (torch.zeros(4),))
    # Functional export makes writes explicit in the graph signature.
    if functionalized:
        program = program.run_decompositions({})
    before = program.example_inputs[0][0].clone()
    with pytest.raises(ValueError, match="mutate"):
        export_torchscript_region(program, tmp_path / "mutable.pt")
    assert torch.equal(program.example_inputs[0][0], before)
    assert not (tmp_path / "mutable.pt").exists()


def test_rng_must_be_an_explicit_input(tmp_path):
    torch = pytest.importorskip("torch")
    from vlaforge.deployment.torchscript_export import export_torchscript_region

    class Random(torch.nn.Module):
        def forward(self, values):
            return values + torch.randn_like(values)

    program = torch.export.export(Random(), (torch.zeros(4),))
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="random operator"):
        export_torchscript_region(program, tmp_path / "random.pt")
    assert torch.equal(torch.get_rng_state(), before)
    assert not (tmp_path / "random.pt").exists()


def test_cli_compiles_real_export_without_claiming_full_model_parity(tmp_path, capsys):
    import json

    torch = pytest.importorskip("torch")
    from vlaforge.cli import main

    class Module(torch.nn.Module):
        def forward(self, values):
            return values.square()

    exported, output, manifest = (tmp_path / name for name in ("export.pt2", "native.pt", "compile.json"))
    torch.export.save(torch.export.export(Module(), (torch.arange(4.),)), exported)
    assert main(["compile-torchscript", str(exported), "--output", str(output),
                 "--manifest", str(manifest)]) == 0
    record = json.loads(manifest.read_text())
    assert json.loads(capsys.readouterr().out) == record
    assert record["backend"] == "torchscript"
    assert not record["numeric_parity_verified"]
    assert record["region_validation"]["validation_cases"][0]["bitwise_equal"]
    assert record["region_validation"]["effect_audits"][0]["passed"]
    other = tmp_path / "other.pt"
    with pytest.raises(ValueError, match="new, separate"):
        main(["compile-torchscript", str(exported), "--output", str(other),
              "--manifest", str(manifest)])
    assert not other.exists()


def _numerical_export(tmp_path):
    torch = pytest.importorskip("torch")
    from vlaforge.numerical_context import snapshot

    class Region(torch.nn.Module):
        def forward(self, value):
            return value.sin(), value.to(torch.float64).square()

    source = tmp_path / "source.pt2"
    torch.export.save(torch.export.export(Region(), (torch.arange(4.),)), source)
    return torch, source, snapshot()


def test_numerical_compile_binds_actual_source_artifact_and_observed_policy(tmp_path):
    from vlaforge.deployment.torchscript_export import compile_torchscript_region
    from vlaforge.deployment.numerical import NumericalCompileRecord
    from vlaforge.deployment.libtorch_numerical import policy_from_context
    import hashlib

    torch, source, context = _numerical_export(tmp_path)
    output = tmp_path / "compiled.pt"
    result = compile_torchscript_region(source, output, reference_context=context,
        validation_cases=((torch.arange(4.) + .1,),))
    record = NumericalCompileRecord.from_dict(result["numerical_compile_record"])
    assert record.target == "cpu" and record.backend == "torchscript"
    assert record.exported_program_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.artifact_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert record.reference_policy == record.observed_compile_policy == policy_from_context(context)
    from vlaforge.frontend import exported_graph_digest
    assert record.graph_sha256 == exported_graph_digest(torch.export.load(source))
    assert result["observed_context_before"] == result["observed_context_after"] == context.to_dict()
    assert len(result["region_validation"]["validation_cases"]) == 2
    assert not result["full_model_verified"] and not result["numerical_provider_enforcement"]
    context.require_current()


def test_numerical_compile_does_not_invent_or_restore_reference_policy(tmp_path):
    from vlaforge.deployment.torchscript_export import compile_torchscript_region
    from vlaforge.numerical_context import NumericalContext

    _, source, context = _numerical_export(tmp_path)
    changed = context.to_dict()
    changed["cudnn_benchmark"] = not changed["cudnn_benchmark"]
    output = tmp_path / "refused.pt"
    with pytest.raises(ValueError):
        compile_torchscript_region(source, output, reference_context=NumericalContext.from_dict(changed))
    assert not output.exists()
    context.require_current()


def test_numerical_compile_rejects_relabeling_cpu_export_as_cuda(tmp_path):
    from vlaforge.deployment.torchscript_export import compile_torchscript_region

    _, source, context = _numerical_export(tmp_path)
    output = tmp_path / "refused.pt"
    with pytest.raises(ValueError, match="target"):
        compile_torchscript_region(source, output, reference_context=context, target="sm_90")
    assert not output.exists()


def test_numerical_compile_does_not_publish_after_compiler_policy_drift(tmp_path, monkeypatch):
    from vlaforge.deployment import torchscript_export as backend

    torch, source, context = _numerical_export(tmp_path)
    actual = backend.export_torchscript_region
    before = torch.get_float32_matmul_precision()

    def drift(*args, **kwargs):
        result = actual(*args, **kwargs)
        torch.set_float32_matmul_precision("high" if before == "highest" else "highest")
        return result

    monkeypatch.setattr(backend, "export_torchscript_region", drift)
    output = tmp_path / "refused.pt"
    try:
        with pytest.raises(ValueError):
            backend.compile_torchscript_region(source, output, reference_context=context)
        assert not output.exists()
    finally:
        torch.set_float32_matmul_precision(before)
    context.require_current()


def test_numerical_compile_preserves_existing_output(tmp_path):
    from vlaforge.deployment.torchscript_export import compile_torchscript_region

    _, source, context = _numerical_export(tmp_path)
    output = tmp_path / "existing.pt"
    output.write_bytes(b"existing caller output")
    with pytest.raises(FileExistsError):
        compile_torchscript_region(source, output, reference_context=context)
    assert output.read_bytes() == b"existing caller output"


def test_numerical_compile_rejects_source_changed_during_compilation(tmp_path, monkeypatch):
    from vlaforge.deployment import torchscript_export as backend

    _, source, context = _numerical_export(tmp_path)
    actual = backend.export_torchscript_region

    def change_source(*args, **kwargs):
        result = actual(*args, **kwargs)
        source.write_bytes(b"changed source")
        return result

    monkeypatch.setattr(backend, "export_torchscript_region", change_source)
    output = tmp_path / "refused.pt"
    with pytest.raises(ValueError, match="exported program changed"):
        backend.compile_torchscript_region(source, output, reference_context=context)
    assert not output.exists()


def test_numerical_compile_preserves_concurrent_publication(tmp_path, monkeypatch):
    from vlaforge.deployment import torchscript_export as backend

    _, source, context = _numerical_export(tmp_path)
    actual = backend.export_torchscript_region
    output = tmp_path / "concurrent.pt"

    def publish_other(*args, **kwargs):
        result = actual(*args, **kwargs)
        output.write_bytes(b"another writer")
        return result

    monkeypatch.setattr(backend, "export_torchscript_region", publish_other)
    with pytest.raises(FileExistsError):
        backend.compile_torchscript_region(source, output, reference_context=context)
    assert output.read_bytes() == b"another writer"
    assert not list(tmp_path.glob(".torchscript-*"))


def test_numerical_compile_cli_is_explicit_and_keeps_evidence_scope(tmp_path, capsys):
    import json
    from vlaforge.cli import main
    from vlaforge.deployment.numerical import NumericalCompileRecord

    _, source, context = _numerical_export(tmp_path)
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context.to_dict()))
    output = tmp_path / "compiled.pt"
    manifest = tmp_path / "compile.json"
    assert main(["compile-torchscript", str(source), "--output", str(output),
                 "--manifest", str(manifest), "--numerical-context", str(context_path),
                 "--target", "cpu"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(manifest.read_text())
    numerical = result["numerical_compilation"]
    record = NumericalCompileRecord.from_dict(numerical["numerical_compile_record"])
    assert record.target == "cpu"
    assert not result["numeric_parity_verified"]
    assert not numerical["full_model_verified"]
    assert not numerical["numerical_provider_enforcement"]


def test_compile_cli_refuses_target_without_numerical_context(tmp_path):
    from vlaforge.cli import main

    _, source, _ = _numerical_export(tmp_path)
    output = tmp_path / "refused.pt"
    with pytest.raises(ValueError, match="requires --numerical-context"):
        main(["compile-torchscript", str(source), "--output", str(output),
              "--target", "sm_90"])
    assert not output.exists()


def test_numerical_compile_does_not_upgrade_legacy_context(tmp_path):
    from vlaforge.deployment.torchscript_export import compile_torchscript_region
    from vlaforge.numerical_context import LEGACY_SCHEMA, NumericalContext

    _, source, context = _numerical_export(tmp_path)
    legacy = context.to_dict()
    legacy['schema'] = LEGACY_SCHEMA
    for name in ('torch_version', 'reduction_api',
                 'cuda_matmul_allow_fp16_reduced_precision_reduction_split_k',
                 'cuda_matmul_allow_bf16_reduced_precision_reduction_split_k'):
        legacy.pop(name)
    output = tmp_path / 'refused.pt'
    reference = NumericalContext.from_dict(legacy)
    with pytest.raises(ValueError, match='newly observed v2'):
        compile_torchscript_region(source, output, reference_context=reference)
    assert not output.exists()
    context.require_current()


def test_numerical_compile_uses_loaded_graph_identity_after_value_renaming(tmp_path):
    torch = pytest.importorskip('torch')
    from vlaforge.frontend import exported_graph_digest
    from vlaforge.numerical_context import snapshot
    from vlaforge.deployment.torchscript_export import compile_torchscript_region

    class Region(torch.nn.Module):
        def forward(self, value):
            left, right = value.unbind(0)
            return left.sin(), right.cos()

    program = torch.export.export(Region(), (torch.arange(8.).reshape(2, 4),))
    captured = exported_graph_digest(program)
    source = tmp_path / 'source.pt2'
    torch.export.save(program, source)
    loaded = exported_graph_digest(torch.export.load(source))
    assert captured != loaded
    result = compile_torchscript_region(source, tmp_path / 'compiled.pt', reference_context=snapshot())
    assert result['numerical_compile_record']['graph_sha256'] == loaded
    assert all(case['bitwise_equal'] for case in result['region_validation']['validation_cases'])
