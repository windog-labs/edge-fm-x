import json
from dataclasses import replace

import pytest
from vlaforge.deployment.numerical import (
    BINDING_SCHEMA,
    PROVIDER_BINDING_SCHEMA,
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalContractError,
    NumericalEnforcementUnavailable,
    NumericalPolicy,
    NumericalRequirement,
    RegionNumericalBinding,
)


def _policy(namespace="example.backend/1", value=True):
    return NumericalPolicy(namespace, (("precise_reduction", value),))


def _record():
    return NumericalCompileRecord(
        "example",
        "cpu",
        "compiler-1",
        "1" * 64,
        "2" * 64,
        "3" * 64,
        123,
        _policy("example.reference/1"),
        _policy(),
        _policy(),
        '{"options":{"fuse":false},"passes":[],"versions":{"runtime":"1"}}',
    )


def _binding():
    record = _record()
    requirement = NumericalRequirement(
        record.observed_compile_policy,
        "same-precision",
        record.reference_policy.digest(),
        record.digest(),
    )
    return RegionNumericalBinding("region", requirement, record)


@pytest.mark.parametrize(
    "factory", (_policy, _record, lambda: _binding().requirement, _binding)
)
def test_canonical_round_trip_and_digest(factory):
    record = factory()
    restored = type(record).from_json(record.canonical_json())
    assert restored == record
    assert restored.canonical_json() == record.canonical_json()
    assert restored.digest() == record.digest()
    assert len(restored.digest()) == 64


@pytest.mark.parametrize(
    "factory", (_policy, _record, lambda: _binding().requirement, _binding)
)
@pytest.mark.parametrize("change", ("unknown", "missing", "version", "fidelity"))
def test_strict_keys_and_versions(factory, change):
    record = factory()
    data = record.to_dict()
    if change == "unknown":
        data["typo"] = True
    elif change == "missing":
        data.pop(next(name for name in data if name != "schema"))
    elif change == "version":
        data["schema"] += "9"
    else:
        data["fidelity_verified"] = True
    with pytest.raises(NumericalContractError):
        type(record).from_dict(data)


@pytest.mark.parametrize(
    "payload",
    (
        '{"schema":1,"schema":2}',
        '{"configuration":{"options":{"same":1,"same":2}}}',
        '{"value":NaN}',
        '{"value":Infinity}',
    ),
)
def test_duplicate_nested_keys_and_nonfinite_json_rejected(payload):
    with pytest.raises(NumericalContractError):
        NumericalCompileRecord.from_json(payload)


@pytest.mark.parametrize(
    "values",
    (
        (("x", float("nan")),),
        (("x", float("inf")),),
        (("x", None),),
        (("x", []),),
        (("x", {}),),
        (("x", True), ("x", False)),
        (("z", True), ("a", False)),
        [["x", True]],
        (("", True),),
    ),
)
def test_invalid_policy_values(values):
    with pytest.raises(NumericalContractError):
        NumericalPolicy("example/1", values)


@pytest.mark.parametrize("namespace", ("", "example", "example/0", "example/1/2", 1))
def test_namespace_is_explicitly_versioned(namespace):
    with pytest.raises(NumericalContractError):
        _policy(namespace)


def test_policy_type_and_configuration_changes_change_digest():
    assert len({_policy(value=value).digest() for value in (True, 1, 1.0, "1")}) == 4
    record = _record()
    assert (
        replace(record, configuration_json='{"fuse":true}').digest() != record.digest()
    )
    with pytest.raises(NumericalContractError, match="requested and observed"):
        replace(record, observed_compile_policy=_policy(value=1))


@pytest.mark.parametrize(
    "field,value",
    (
        ("artifact_size_bytes", True),
        ("artifact_size_bytes", -1),
        ("exported_program_sha256", "g" * 64),
        ("artifact_sha256", "A" * 64),
        ("graph_sha256", "short"),
        ("compiler_version", ""),
        ("configuration_json", "{}"),
        ("configuration_json", '{"x": 1}'),
        ("configuration_json", '{"x":1e309}'),
        ("configuration_json", "[1]"),
        ("configuration_json", '{"x":1,"x":1}'),
    ),
)
def test_compile_record_rejects_invalid_fields(field, value):
    with pytest.raises(NumericalContractError):
        replace(_record(), **{field: value})


def test_configuration_uses_strict_json_string_keys():
    data = _record().to_dict()
    data["configuration"] = {1: True}
    with pytest.raises(NumericalContractError, match="keys"):
        NumericalCompileRecord.from_dict(data)


def test_caller_owned_configuration_and_policy_dicts_are_not_retained():
    data = _record().to_dict()
    restored = NumericalCompileRecord.from_dict(data)
    digest = restored.digest()
    data["configuration"]["options"]["fuse"] = True
    data["observed_compile_policy"]["values"]["precise_reduction"] = False
    assert restored.digest() == digest
    returned = restored.to_dict()
    returned["configuration"]["passes"].append("untrusted")
    assert restored.digest() == digest


@pytest.mark.parametrize(
    "field,value",
    (
        ("execution_lane", "lossless"),
        ("enforcement", "verified"),
        ("reference_context_sha256", "bad"),
        ("compile_record_sha256", False),
    ),
)
def test_requirement_rejects_unknown_claims(field, value):
    with pytest.raises(NumericalContractError):
        replace(_binding().requirement, **{field: value})


def test_policy_digest_cannot_be_forged():
    data = _binding().requirement.to_dict()
    data["policy_sha256"] = "f" * 64
    with pytest.raises(NumericalContractError, match="digest mismatch"):
        NumericalRequirement.from_dict(data)


@pytest.mark.parametrize("field", ("compile_record_sha256", "reference_context_sha256"))
def test_binding_rejects_wrong_reference_or_compile_identity(field):
    binding = _binding()
    requirement = replace(binding.requirement, **{field: "f" * 64})
    with pytest.raises(NumericalContractError, match="digest mismatch"):
        replace(binding, requirement=requirement)


def test_runtime_projection_is_not_silently_inferred():
    binding = _binding()
    requirement = replace(binding.requirement, policy=_policy(value=1))
    with pytest.raises(NumericalContractError, match="projection is unsupported"):
        replace(binding, requirement=requirement)


def test_recorded_does_not_imply_runtime_or_output_verified():
    binding = _binding()
    data = binding.to_dict()
    assert data["runtime_enforcement"] == "unimplemented"
    assert "fidelity_verified" not in json.dumps(data)
    assert "output_verified" not in json.dumps(data)
    with pytest.raises(NumericalEnforcementUnavailable, match="unimplemented"):
        binding.require_runtime_deployable()
    data["runtime_enforcement"] = "passed"
    with pytest.raises(NumericalContractError, match="unimplemented"):
        RegionNumericalBinding.from_dict(data)


def test_quantized_lane_is_a_different_requirement_not_a_pass():
    requirement = _binding().requirement
    quantized = replace(requirement, execution_lane="quantized")
    assert quantized.digest() != requirement.digest()
    assert "verified" not in quantized.to_dict()


def test_provider_binding_is_explicit_not_a_legacy_promotion():
    original = _binding()
    assert (
        RegionNumericalBinding.from_dict(original.to_dict()).runtime_enforcement
        == "unimplemented"
    )
    provider = replace(original, runtime_enforcement=PROVIDER_REQUIRED)
    data = provider.to_dict()
    assert data["schema"] == PROVIDER_BINDING_SCHEMA
    assert RegionNumericalBinding.from_dict(data) == provider
    assert provider.digest() != original.digest()
    # This test backend has no actual provider. Schema support is not execution support.
    with pytest.raises(NumericalEnforcementUnavailable, match="example.*unsupported"):
        provider.require_runtime_deployable()
    for schema, marker in (
        (BINDING_SCHEMA, PROVIDER_REQUIRED),
        (PROVIDER_BINDING_SCHEMA, "unimplemented"),
    ):
        with pytest.raises(NumericalContractError):
            RegionNumericalBinding.from_dict(
                {**data, "schema": schema, "runtime_enforcement": marker}
            )


@pytest.mark.parametrize("payload", (None, [], 1, "policy"))
def test_binding_requires_mapping(payload):
    with pytest.raises(NumericalContractError):
        RegionNumericalBinding.from_dict(payload)
