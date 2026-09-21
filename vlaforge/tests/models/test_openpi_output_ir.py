"""Typed orchestration fixtures, not pretrained-model evidence."""

from dataclasses import dataclass, replace
from pathlib import Path
import pytest

from vlaforge.adapters.openpi.openpi_output_ir import attach_output_stage
from vlaforge.compiler import compile_module
from vlaforge.frontend import InvocationBuilder, tensor_region
from vlaforge.ir.program import InputPort, OutputPort, TensorRegion, Value
from vlaforge.ir.serializer import canonical_json, parse_canonical_json, io_schema_digest
from vlaforge.ir.types import TensorType


def example():
    state = TensorType((1, 4), "f64")
    action = TensorType((1, 3, 4), "f32")
    native = TensorType((1, 3, 2), "f64")
    boolean = TensorType((1,), "bool")
    builder = InvocationBuilder("output_interface_fixture",
        inputs=(InputPort("state", state), InputPort("noise", action)),
        outputs=(OutputPort("normalized_action_chunk", action, group="action"),))
    @tensor_region("finite", inputs=(Value("actions", action),), outputs=(boolean,))
    def finite(value):
        raise AssertionError("declaration-only fixture")
    builder.input("state")
    value = builder.input("noise")
    accepted, = builder.call(finite, value)
    module = builder.finish({"normalized_action_chunk": value}, accepted=accepted).module
    region = TensorRegion("finish", (Value("state", state), Value("actions", action), Value("incoming_accepted", boolean)), (native, boolean))
    return module, region


def test_output_stage_preserves_old_compute_and_changes_only_publication():
    original, region = example()
    before = canonical_json(original)
    result = attach_output_stage(original, region)
    assert canonical_json(original) == before
    assert result.inputs == original.inputs and result.states == original.states
    assert result.regions[:-1] == original.regions
    assert result.invocations[0].body.operations[:-7] == original.invocations[0].body.operations[:-5]
    assert result.outputs[0] == original.outputs[0]
    assert result.outputs[1].payload == TensorType((1, 3, 2), "f64")
    assert result.outputs[1].output_id == 1
    assert io_schema_digest(result) != io_schema_digest(original)
    assert parse_canonical_json(canonical_json(result)) == result
    compile_module(result, profile="verified")


@pytest.mark.parametrize("kind", ["missing_predicate", "wrong_action", "existing_name", "output_name", "transaction"])
def test_unknown_output_extension_contract_fails_closed(kind):
    module, region = example()
    kwargs = {}
    if kind == "missing_predicate":
        region = replace(region, outputs=region.outputs[:1])
    elif kind == "wrong_action":
        region = replace(region, inputs=(region.inputs[0], Value("actions", TensorType((1,), "f32"))))
    elif kind == "existing_name":
        region = replace(region, name="finite")
    elif kind == "output_name":
        kwargs["output_name"] = "normalized_action_chunk"
    else:
        module = replace(module, invocations=())
    with pytest.raises(ValueError):
        attach_output_stage(module, region, **kwargs)


def test_extended_ir_uses_existing_multi_output_runner_with_native_primary():
    from vlaforge.adapters.openpi.openpi_output_bundle import render_typed_runner
    original, region = example()
    original = replace(original,
        inputs=tuple(replace(item, device="cuda:0") for item in original.inputs),
        outputs=tuple(replace(item, device="cuda:0") for item in original.outputs))
    module = attach_output_stage(original, region)
    template = (Path(__file__).resolve().parents[2] / "tools/session_benchmark_runner.cpp.in").read_text()
    source, contract = render_typed_runner(module, template)
    assert contract["index"] == 1 and contract["dtype"] == "f64"
    assert [item["dtype"] for item in contract["outputs"]] == ["f32", "f64"]
    assert "@" not in source
    assert "VLAFORGE_BENCHMARK_MULTI_OUTPUT 1" in source
    assert "constexpr bool kOwnerHandshake = true;" in source
    with pytest.raises(ValueError, match="template"):
        render_typed_runner(module, template + "@UNKNOWN@")


@pytest.mark.parametrize("changed", [None, "sha256", "size_bytes", "md5_base64", "missing"])
def test_materialized_identity_uses_complete_existing_file_record(tmp_path, changed):
    from vlaforge.adapters.openpi.openpi_checkpoint import file_digest
    from vlaforge.adapters.openpi.openpi_output_bundle import materialized_package_path
    path = tmp_path / "model.vfaoti"
    path.write_bytes(b"record integrity fixture, not an executable model")
    record = {"path": str(path), **file_digest(path), "load_evidence": {}}
    if changed is None:
        assert materialized_package_path(record) == path
    else:
        if changed == "missing":
            record.pop("md5_base64")
        else:
            record[changed] = 1 if changed == "size_bytes" else "changed"
        with pytest.raises((ValueError, KeyError)):
            materialized_package_path(record)


@dataclass(frozen=True)
class ContractFixture:
    region_name: str
    region_id: int
    io_schema_digest: str
    artifact_sha256: str = "a" * 64


def test_new_interface_rebinds_ids_after_canonical_region_reordering():
    from vlaforge.adapters.openpi.openpi_output_bundle import rebind_existing_contracts
    original, output = example()
    finite = original.regions[0]
    original = replace(original, regions=(replace(finite, name="zzz"), finite, replace(finite, name="aaa")))
    contracts = tuple(ContractFixture(item.name, index, io_schema_digest(original)) for index, item in enumerate(original.regions))
    restored = parse_canonical_json(canonical_json(compile_module(original, profile="verified").module))
    assert [item.name for item in restored.regions] == ["aaa", "finite", "zzz"]
    updated = attach_output_stage(restored, output)
    bound = rebind_existing_contracts(updated, contracts)
    assert {key: value.region_id for key, value in bound.items()} == {"aaa": 0, "finite": 1, "zzz": 2}
    assert contracts[0].region_id == 0
    assert all(value.io_schema_digest == io_schema_digest(updated) for value in bound.values())
    assert all(value.artifact_sha256 == "a" * 64 for value in bound.values())


@pytest.mark.parametrize("kind", ["unknown", "duplicate_contract", "duplicate_region"])
def test_contract_rebind_rejects_ambiguous_identity(kind):
    from vlaforge.adapters.openpi.openpi_output_bundle import rebind_existing_contracts
    original, _ = example()
    contract = ContractFixture("finite", 0, io_schema_digest(original))
    contracts = (contract,)
    if kind == "unknown":
        contracts = (replace(contract, region_name="unknown"),)
    elif kind == "duplicate_contract":
        contracts = (contract, contract)
    else:
        with pytest.raises(ValueError, match="duplicate region"):
            replace(original, regions=(*original.regions, *original.regions))
        return
    with pytest.raises(ValueError):
        rebind_existing_contracts(original, contracts)
