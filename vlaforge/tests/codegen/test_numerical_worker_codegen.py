"""Worker-source transport fixtures; actual LibTorch runs are separate tests."""

from dataclasses import replace

import pytest
from vlaforge.codegen.numerical import generate_libtorch_worker_initializer
from vlaforge.deployment.libtorch_numerical import NAMESPACE, REDUCTION_API
from vlaforge.deployment.numerical import (
    PROVIDER_REQUIRED,
    NumericalCompileRecord,
    NumericalContractError,
    NumericalEnforcementUnavailable,
    NumericalPolicy,
    NumericalRequirement,
    RegionNumericalBinding,
)
from vlaforge.numerical_context import NumericalContext


def _binding(*, precision="medium", backend="torchscript", name="region"):
    excluded = {
        "schema",
        "torch_version",
        "reduction_api",
        "float32_matmul_precision",
        "autocast_cpu_dtype",
        "autocast_cuda_dtype",
    }
    policy = NumericalPolicy(
        NAMESPACE,
        tuple(
            sorted(
                {
                    **{
                        key: False
                        for key in NumericalContext.__dataclass_fields__
                        if key not in excluded
                    },
                    "autocast_cpu_dtype": "bfloat16",
                    "autocast_cuda_dtype": "float16",
                    "float32_matmul_precision": precision,
                    "cuda_matmul_allow_tf32": precision != "highest",
                    "torch_release": "2.10.0",
                    "reduction_api": REDUCTION_API,
                }.items()
            )
        ),
    )
    record = NumericalCompileRecord(
        backend,
        "cpu",
        "worker-codegen-fixture/1",
        "1" * 64,
        "2" * 64,
        "3" * 64,
        1,
        policy,
        policy,
        policy,
        '{"fixture":true}',
    )
    return RegionNumericalBinding(
        name,
        NumericalRequirement(
            policy, "same-precision", policy.digest(), record.digest()
        ),
        record,
        PROVIDER_REQUIRED,
    )


def _generate(bindings, **kwargs):
    return generate_libtorch_worker_initializer(
        bindings,
        **{
            "acknowledge_exclusive_process": True,
            "acknowledge_calling_thread": True,
            **kwargs,
        },
    )


def test_explicit_bootstrap_is_separate_from_session_and_preserves_typed_policy():
    binding = _binding()
    source = _generate([binding])
    assert source.count("VLAFORGE_NUMERICAL_BOOL") == 19
    assert source.count("VLAFORGE_NUMERICAL_STRING") == 5
    assert binding.requirement.policy.digest() in source
    assert binding.requirement.digest() in source
    assert "ModelSession" not in source
    assert "setAllowTF32" not in source
    assert (
        "vlaforge_libtorch_numerical_initialize_worker(&kNumericalRequirementWorker, &options)"
        in source
    )
    assert (
        "VLAFORGE_LIBTORCH_NUMERICAL_EXCLUSIVE_PROCESS | VLAFORGE_LIBTORCH_NUMERICAL_CALLING_THREAD"
        in source
    )


@pytest.mark.parametrize(
    "field", ["acknowledge_exclusive_process", "acknowledge_calling_thread"]
)
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_ownership_requires_exact_explicit_booleans(field, value):
    with pytest.raises(NumericalContractError, match="ownership"):
        _generate([_binding()], **{field: value})


def test_both_ownership_arguments_are_mandatory():
    with pytest.raises(TypeError):
        generate_libtorch_worker_initializer([_binding()])


def test_mixed_libtorch_backends_with_same_policy_share_one_initializer():
    bindings = [_binding(), _binding(backend="aoti", name="action")]
    source = _generate(iter(bindings))
    assert source.count("constexpr VLAForgeNumericalRequirementView") == 1
    assert source.count(bindings[0].requirement.policy.digest()) == 1


def test_conflicting_policies_are_never_silently_selected():
    for bindings in (
        [_binding(), _binding(precision="high")],
        [_binding(precision="high"), _binding()],
    ):
        with pytest.raises(NumericalContractError, match="separate workers"):
            _generate(bindings)


@pytest.mark.parametrize("bindings", [[], [None], [{}], ["region"]])
def test_empty_or_untyped_bindings_cannot_initialize(bindings):
    with pytest.raises(NumericalContractError):
        _generate(bindings)


def test_other_providers_and_unenforced_records_cannot_initialize_libtorch():
    with pytest.raises(NumericalContractError, match="LibTorch bindings"):
        _generate([_binding(backend="shared_plugin")])
    binding = replace(_binding(), runtime_enforcement="unimplemented")
    with pytest.raises(NumericalEnforcementUnavailable, match="unimplemented"):
        _generate([binding])
